#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, datetime as dt
import re 
import numpy as np, pandas as pd
from datatree import open_datatree, DataTree
from dotenv import load_dotenv
from pathlib import Path

from harmony import BBox, Client, Collection, Request
from harmony.config import Environment

from config import ensure_dir, atomic_write_csv, get_config

# Collections (NRT V02)
COLLECTION_NO2_L3NRT = "C3685668637-LARC_CLOUD"  # L3 NRT V02
COLLECTION_NO2_L2NRT = "C3685668972-LARC_CLOUD"  # L2 NRT V02

def parse_bbox_ws_en(bbox_str: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("BBOX_WSEN must be 'W,S,E,N'")
    return tuple(parts)

def _to_minus180_180(lon_vals):
    if np.nanmax(lon_vals) > 180.0:
        return ((lon_vals + 180.0) % 360.0) - 180.0
    return lon_vals

def find_candidate_vars(tree: DataTree, patterns):
    pats = [p.lower() for p in patterns]
    hits = []
    def list_nodes(t: DataTree, prefix=""):
        yield (prefix, t)
        for name, child in t.children.items():
            child_prefix = f"{prefix}/{name}" if prefix else name
            yield from list_nodes(child, child_prefix)
    for path, node in list_nodes(tree):
        if hasattr(node, "data_vars"):
            for v in node.data_vars:
                full = f"{path}/{v}" if path else v
                if any(p in full.lower() for p in pats):
                    hits.append(full)
    return sorted(set(hits))

def pick_variable_path(cands):
    preferred = [
        "product/vertical_column_troposphere",
        "product/tropospheric_vertical_column",
        "support_data/vertical_column_total",
        "product/vertical_column_total",
    ]
    for p in preferred:
        if p in cands:
            return p
    for v in cands:
        low = v.lower()
        if "column" in low and "uncertainty" not in low and "stratosphere" not in low:
            return v
    return None

def _get_da(tree: DataTree, path: str):
    parts = path.split("/")
    node = tree
    for p in parts[:-1]:
        node = node[p]
    return node.to_dataset()[parts[-1]]

def harmony_fetch_files(auth: tuple[str,str], bbox: tuple[float,float,float,float],
                        t_stop_utc: dt.datetime, window_hours: int,
                        download_dir: Path) -> list[str]:
    W,S,E,N = bbox
    t_start_utc = t_stop_utc - dt.timedelta(hours=window_hours)
    ensure_dir(download_dir)
    h = Client(env=Environment.PROD, auth=auth)

    for coll in (COLLECTION_NO2_L2NRT, COLLECTION_NO2_L3NRT):
        req = Request(
            collection=Collection(id=coll),
            spatial=BBox(W,S,E,N),
            temporal={"start": t_start_utc, "stop": t_stop_utc},
        )
        if not req.is_valid():
            continue
        print(f"[Harmony] submit {coll} window {t_start_utc}..{t_stop_utc} bbox=({W},{S},{E},{N})")
        try:
            job_id = h.submit(req)
            h.wait_for_processing(job_id, show_progress=False)
            downloads = h.download_all(job_id, directory=download_dir.as_posix(), overwrite=False)
            files = [f.result() for f in downloads]
            files = [f for f in files if f and Path(f).exists() and Path(f).stat().st_size > 0]
            if files:
                return files
        except Exception as e:
            print(f"[Harmony] {coll} failed: {e}")
    return []

def subset_stack_to_df(files: list[str],
                       t0: dt.datetime, t1: dt.datetime,
                       bbox: tuple[float,float,float,float]) -> tuple[pd.DataFrame, str]:
    if not files:
        return pd.DataFrame(columns=["time","longitude","latitude","unit","no2_column"]), "molecules/cm^2"

    W,S,E,N = bbox
    dtree0 = open_datatree(files[0])
    patt = ("vertical","tropos","column","total","no2","nitrogen")
    cands0 = find_candidate_vars(dtree0, patt)
    var_path = pick_variable_path(cands0)
    if var_path is None:
        return pd.DataFrame(columns=["time","longitude","latitude","unit","no2_column"]), "molecules/cm^2"

    try:
        Da0 = _get_da(dtree0, var_path)
        units = Da0.attrs.get("units", "molecules/cm^2")
    except Exception:
        units = "molecules/cm^2"

    qa_paths = [("product/main_data_quality_flag","eq0"), ("product/qa_value","ge0.5"), (None,"none")]
    margins = [0.0, 0.1, 0.25, 0.5, 1.0]
    parts = []

    def subset_one(fp: str):
        tree = open_datatree(fp)

        # --- parse granule timestamp from filename (e.g., 20251001T201426Z) ---
        # default to day midpoint if not found (rare)
        tstamp = None
        try:
            m = re.search(r'(\d{8}T\d{6})Z', Path(fp).name)
            if m:
                tstamp = dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)
        except Exception:
            tstamp = None
        if tstamp is None:
            # very conservative fallback: middle of [t0,t1)
            tstamp = t0 + (t1 - t0) / 2

        # var per file
        try:
            Da = _get_da(tree, var_path)
        except Exception:
            cands = find_candidate_vars(tree, patt)
            alt = pick_variable_path(cands)
            if not alt:
                return None
            Da = _get_da(tree, alt)

        u = Da.attrs.get("units", units) or units

        lon_names = [c for c in Da.coords if 'lon' in c.lower() or 'longitude' in c.lower()] \
                    or [d for d in Da.dims if d.lower() in ("lon", "x", "longitude")]
        lat_names = [c for c in Da.coords if 'lat' in c.lower() or 'latitude' in c.lower()] \
                    or [d for d in Da.dims if d.lower() in ("lat", "y", "latitude")]
        lon_name = lon_names[0] if lon_names else None
        lat_name = lat_names[0] if lat_names else None

        def _maybe(path):
            try:
                return _get_da(tree, path)
            except Exception:
                return None

        GEO_LON_CANDS = ["geolocation/longitude", "GEOLOCATION/longitude", "lon", "longitude"]
        GEO_LAT_CANDS = ["geolocation/latitude", "GEOLOCATION/latitude", "lat", "latitude"]
        GeoLon = None
        GeoLat = None

        for qa_path, mode in qa_paths:
            Da_qa = Da
            if qa_path:
                try:
                    qa_da = _get_da(tree, qa_path)
                    if mode == "eq0":
                        Da_qa = Da.where(qa_da == 0)
                    elif mode == "ge0.5":
                        Da_qa = Da.where(qa_da >= 0.5)
                except Exception:
                    pass

            for m in margins:
                Wm, Sm, Em, Nm = (W - m, S - m, E + m, N + m)
                try:
                    # path A: coordinate subsetting
                    if lon_name and lat_name:
                        lons = Da_qa[lon_name].values
                        if np.nanmax(lons) > 180:
                            Da_qa = Da_qa.assign_coords({lon_name: _to_minus180_180(lons)})

                        lat_vals = Da_qa[lat_name].values
                        lon_vals = Da_qa[lon_name].values
                        lat_slice = slice(Sm, Nm) if lat_vals[0] < lat_vals[-1] else slice(Nm, Sm)
                        lon_slice = slice(Wm, Em) if lon_vals[0] < lon_vals[-1] else slice(Em, Wm)
                        try:
                            Da_sub = Da_qa.sel({lat_name: lat_slice, lon_name: lon_slice})
                        except Exception:
                            LonC = Da_qa[lon_name];
                            LatC = Da_qa[lat_name]
                            Da_sub = Da_qa.where(
                                (LonC >= Wm) & (LonC <= Em) & (LatC >= Sm) & (LatC <= Nm),
                                drop=True
                            )

                        dfp = Da_sub.to_dataframe(name="no2_column").reset_index()
                        dfp = dfp[np.isfinite(dfp["no2_column"])]
                        if dfp.empty:
                            continue

                        # ensure required columns
                        if "time" not in dfp.columns or not np.issubdtype(dfp["time"].dtype, np.datetime64):
                            dfp["time"] = pd.Timestamp(tstamp, tz="UTC")  # <-- REAL GRANULE TIME
                        if "longitude" not in dfp.columns:
                            cand = [c for c in dfp.columns if c.lower() in ("lon", "longitude", "x")]
                            if cand: dfp = dfp.rename(columns={cand[0]: "longitude"})
                        if "latitude" not in dfp.columns:
                            cand = [c for c in dfp.columns if c.lower() in ("lat", "latitude", "y")]
                            if cand: dfp = dfp.rename(columns={cand[0]: "latitude"})
                        dfp = dfp.dropna(subset=["time", "longitude", "latitude"])
                        dfp["unit"] = u if u else "molecules/cm^2"
                        keep = ["time", "longitude", "latitude", "unit", "no2_column"]
                        if m > 0:
                            print(f"    ✓ TEMPO pixels with margin {m}° (QA={mode})")
                        return dfp[[c for c in keep if c in dfp.columns]]

                    # path B: geolocation-array fallback
                    if GeoLon is None or GeoLat is None:
                        for p in GEO_LON_CANDS:
                            GeoLon = _maybe(p)
                            if GeoLon is not None: break
                        for p in GEO_LAT_CANDS:
                            GeoLat = _maybe(p)
                            if GeoLat is not None: break

                    if (GeoLon is not None) and (GeoLat is not None):
                        LonVals = _to_minus180_180(GeoLon.values)
                        mask = (LonVals >= Wm) & (LonVals <= Em) & \
                               (GeoLat.values >= Sm) & (GeoLat.values <= Nm)
                        V = Da_qa.values
                        if V is None:
                            continue
                        msk = np.isfinite(V) & mask
                        if not np.any(msk):
                            continue
                        dfp = pd.DataFrame({
                            "no2_column": V[msk].ravel(),
                            "longitude": LonVals[msk].ravel(),
                            "latitude": GeoLat.values[msk].ravel(),
                        })
                        dfp["time"] = pd.Timestamp(tstamp, tz="UTC")  # <-- REAL GRANULE TIME
                        dfp["unit"] = u if u else "molecules/cm^2"
                        dfp = dfp.dropna(subset=["time", "longitude", "latitude", "no2_column"])
                        if not dfp.empty:
                            if m > 0:
                                print(f"    ✓ TEMPO pixels via GEO arrays, margin {m}° (QA={mode})")
                            return dfp[["time", "longitude", "latitude", "unit", "no2_column"]]

                except Exception:
                    continue
        return None

    for i, fp in enumerate(files, 1):
        print(f"[Subset] ({i}/{len(files)}) {Path(fp).name}")
        part = subset_one(fp)
        if part is not None and len(part):
            parts.append(part)

    if not parts:
        print("[Subset] no valid TEMPO pixels after subsetting; return empty.")
        return pd.DataFrame(columns=["time","longitude","latitude","unit","no2_column"]), "molecules/cm^2"

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["time","longitude","latitude","unit"])
    return df, units

# Public runner variants
def run_once_with_temporal(start_iso: str, end_iso: str):
    cfg = get_config()
    load_dotenv()
    user = cfg["EARTHDATA_USER"]; pwd = cfg["EARTHDATA_PASS"]
    if not user or not pwd:
        raise RuntimeError("Set EARTHDATA_USER and EARTHDATA_PASS")

    W,S,E,N = parse_bbox_ws_en(cfg["BBOX_WSEN"])
    out_dir = Path(cfg["OUT_DIR"])
    downloads = Path(cfg["DOWNLOAD_DIR"])
    ensure_dir(out_dir); ensure_dir(downloads)

    t0 = dt.datetime.fromisoformat(start_iso.replace("Z","+00:00"))
    t1 = dt.datetime.fromisoformat(end_iso.replace("Z","+00:00"))

    files = harmony_fetch_files((user,pwd), (W,S,E,N), t1, window_hours=int((t1-t0).total_seconds()//3600), download_dir=downloads)
    df, unit_used = subset_stack_to_df(files, t0, t1, (W,S,E,N))
    stamp = f"{t0:%Y%m%dT%H%M}_{t1:%H%M}Z"
    csv_path = out_dir / f"tempo_no2_{stamp}.csv"
    atomic_write_csv(df, csv_path.as_posix())
    return csv_path.as_posix(), df

def run_once():
    start_iso = os.getenv("START_ISO"); end_iso = os.getenv("END_ISO")
    if start_iso and end_iso:
        return run_once_with_temporal(start_iso, end_iso)
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    t1 = now; t0 = now - dt.timedelta(hours=4)
    return run_once_with_temporal(t0.strftime("%Y-%m-%dT%H:%M:%SZ"), t1.strftime("%Y-%m-%dT%H:%M:%SZ"))
