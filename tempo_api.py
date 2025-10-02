#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TEMPO NO2 runner with bbox-aware daylight scanning and robust variable detection.

Deps:
  pip install harmony-py xarray netCDF4 numpy pandas matplotlib python-dotenv xarray-datatree requests
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datatree import open_datatree, DataTree
from dotenv import load_dotenv
from typing import Optional, Tuple, List

from harmony import BBox, Client, Collection, Request
from harmony.config import Environment

# Collections
COLLECTION_NO2_L3NRT = "C3685668637-LARC_CLOUD"  # L3-NRT V02
COLLECTION_NO2_L2NRT = "C3685668972-LARC_CLOUD"  # L2-NRT V02 (preferred if available)

PREFERRED_PATHS = [
    "product/vertical_column_troposphere",
    "product/tropospheric_vertical_column",
    "product/tropospheric_vertical_column_density_no2",
    "product/tropospheric_column_no2",
    "support_data/vertical_column_total",
    "product/vertical_column_total",
]

def _parse_bbox_ws_en(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("BBOX_WSEN must be 'W,S,E,N'")
    return tuple(parts)

def _lonlat_center(bbox: Tuple[float,float,float,float]) -> Tuple[float,float]:
    w,s,e,n = bbox
    return ((w+e)/2.0, (s+n)/2.0)

def is_daylight_over_bbox(mid_utc: dt.datetime, bbox: Tuple[float,float,float,float]) -> bool:
    """
    Simple, dependency-free daylight heuristic:
    Convert UTC to local solar time by longitude: LST ≈ UTC + lon/15.
    Treat daylight as 08:00–18:00 local solar time.
    """
    lon, _ = _lonlat_center(bbox)
    lst = mid_utc + dt.timedelta(hours=lon/15.0)
    return 8 <= lst.hour < 18

def round_down_to_hours(t: dt.datetime, hours: int) -> dt.datetime:
    q = (t.hour // hours) * hours
    return t.replace(minute=0, second=0, microsecond=0, hour=q)

def _to_minus180_180(lon_vals):
    if np.nanmax(lon_vals) > 180.0:
        return ((lon_vals + 180.0) % 360.0) - 180.0
    return lon_vals

def list_nodes(tree: DataTree, prefix=""):
    yield (prefix, tree)
    for name, child in tree.children.items():
        child_prefix = f"{prefix}/{name}" if prefix else name
        yield from list_nodes(child, child_prefix)

def find_candidate_vars(tree: DataTree, patterns):
    pats = [p.lower() for p in patterns]
    hits = []
    for path, node in list_nodes(tree):
        if hasattr(node, "data_vars"):
            for v in node.data_vars:
                full = f"{path}/{v}" if path else v
                low = full.lower()
                if any(p in low for p in pats) and "uncert" not in low and "qa" not in low:
                    hits.append(full)
    return sorted(set(hits))

def pick_variable_path(tree: DataTree) -> Optional[str]:
    cands = find_candidate_vars(
        tree,
        ("no2","nitrogen","vertical","column","tropos")
    )
    # exact preferred ordering first
    for p in PREFERRED_PATHS:
        if p in cands:
            return p
    # otherwise first sensible candidate
    return cands[0] if cands else None

def _get_dataarray_by_path(tree: DataTree, path: str):
    parts = path.split("/")
    node = tree
    for p in parts[:-1]:
        node = node[p]
    return node.to_dataset()[parts[-1]]

def _maybe_get(tree: DataTree, path: str):
    try:
        return _get_dataarray_by_path(tree, path)
    except Exception:
        return None

def _guess_lon_lat_names(da):
    lon_names = [c for c in da.coords if 'lon' in c.lower() or 'longitude' in c.lower()]
    lat_names = [c for c in da.coords if 'lat' in c.lower() or 'latitude' in c.lower()]
    if lon_names and lat_names:
        return lon_names[0], lat_names[0]
    # fallback based on dims
    if not lon_names:
        ln = [d for d in da.dims if d.lower() in ("lon","x","longitude")]
        if ln: lon_names = [ln[0]]
    if not lat_names:
        lt = [d for d in da.dims if d.lower() in ("lat","y","latitude")]
        if lt: lat_names = [lt[0]]
    return (lon_names[0] if lon_names else None,
            lat_names[0] if lat_names else None)

def harmony_fetch_files(
    auth: Tuple[str,str],
    bbox: Tuple[float,float,float,float],
    t_stop_utc: dt.datetime,
    window_hours: int
) -> Tuple[List[str], dt.datetime, dt.datetime]:
    t_start_utc = t_stop_utc - dt.timedelta(hours=window_hours)
    W,S,E,N = bbox
    # Prefer L2, fallback L3
    for coll in (COLLECTION_NO2_L2NRT, COLLECTION_NO2_L3NRT):
        req = Request(
            collection=Collection(id=coll),
            spatial=BBox(W,S,E,N),
            temporal={"start": t_start_utc, "stop": t_stop_utc},
        )
        if not req.is_valid():
            continue
        print(f"[Harmony] Using {coll}")
        print(f"[Harmony] Window: {t_start_utc.isoformat()} .. {t_stop_utc.isoformat()} UTC")
        print(f"[Harmony] BBOX: ({W}, {S}, {E}, {N})")

        h = Client(env=Environment.PROD, auth=auth)
        job_id = h.submit(req)
        print("[Harmony] Job:", job_id)
        h.wait_for_processing(job_id, show_progress=True)
        downloads = h.download_all(job_id, directory=None, overwrite=True)
        files = [f.result() for f in downloads]
        if files:
            return files, t_start_utc, t_stop_utc
        print(f"[Harmony] No files for {coll}, trying fallback …")
    raise RuntimeError("No matching granules found for L2/L3 NRT.")

def subset_stack_to_df(files: List[str],
                       t0: dt.datetime, t1: dt.datetime,
                       bbox: Tuple[float,float,float,float]) -> Tuple[pd.DataFrame, str]:
    W,S,E,N = bbox
    parts = []
    unit_fallback = "molecules/cm^2"

    def subset_one(fp: str):
        tree = open_datatree(fp)
        var_path = pick_variable_path(tree)
        if not var_path:
            print("  [Subset] skip file (no suitable NO2 variable found)")
            return None
        Da = _get_dataarray_by_path(tree, var_path)
        units = Da.attrs.get("units", unit_fallback) or unit_fallback

        lon_name, lat_name = _guess_lon_lat_names(Da)
        if lon_name is None or lat_name is None:
            # Try geolocation groups
            for cand in ["geolocation/longitude", "GEOLOCATION/longitude", "lon", "longitude"]:
                LON = _maybe_get(tree, cand)
                if LON is not None:
                    lon_name = cand
                    break
            for cand in ["geolocation/latitude", "GEOLOCATION/latitude", "lat", "latitude"]:
                LAT = _maybe_get(tree, cand)
                if LAT is not None:
                    lat_name = cand
                    break

        margins = [0.0, 0.2, 0.5]
        for m in margins:
            Wm,Sm,Em,Nm = (W-m, S-m, E+m, N+m)
            try:
                Da2 = Da
                # normalize lon range if needed
                if lon_name and np.nanmax(Da2[lon_name].values) > 180:
                    Da2 = Da2.assign_coords({lon_name: _to_minus180_180(Da2[lon_name].values)})

                if lon_name and lat_name and (lon_name in Da2.coords) and (lat_name in Da2.coords):
                    lon_vals = Da2[lon_name].values
                    lat_vals = Da2[lat_name].values
                    lat_slice = slice(Sm, Nm) if lat_vals[0] < lat_vals[-1] else slice(Nm, Sm)
                    lon_slice = slice(Wm, Em) if lon_vals[0] < lon_vals[-1] else slice(Em, Wm)
                    try:
                        sub = Da2.sel({lat_name: lat_slice, lon_name: lon_slice})
                    except Exception:
                        LonC = Da2[lon_name]; LatC = Da2[lat_name]
                        sub = Da2.where((LonC >= Wm) & (LonC <= Em) & (LatC >= Sm) & (LatC <= Nm), drop=True)
                else:
                    # final fallback: no coords usable
                    return None

                dfp = sub.to_dataframe(name="no2_column").reset_index()
                dfp = dfp[np.isfinite(dfp["no2_column"])]
                if dfp.empty:
                    continue

                # Add/normalize essentials
                mid = t0 + (t1 - t0)/2
                if "time" not in dfp.columns:
                    dfp["time"] = np.datetime64(mid)
                if "longitude" not in dfp.columns:
                    cand = [c for c in dfp.columns if c.lower() in ("lon","longitude","x")]
                    if cand: dfp = dfp.rename(columns={cand[0]:"longitude"})
                if "latitude" not in dfp.columns:
                    cand = [c for c in dfp.columns if c.lower() in ("lat","latitude","y")]
                    if cand: dfp = dfp.rename(columns={cand[0]:"latitude"})

                dfp["unit"] = units
                keep = [c for c in ("time","longitude","latitude","no2_column","unit") if c in dfp.columns]
                return dfp[keep]
            except Exception:
                continue
        return None

    for i, fp in enumerate(files, 1):
        print(f"[Subset] ({i}/{len(files)}) {os.path.basename(fp)}")
        part = subset_one(fp)
        if part is not None and len(part):
            parts.append(part)

    if not parts:
        print("[Subset] No valid pixels after QA/coord subsetting; returning empty frame.")
        empty = pd.DataFrame(columns=["time", "longitude", "latitude", "unit", "no2_column"])
        return empty, "molecules/cm^2"
    df = pd.concat(parts, ignore_index=True).dropna(subset=["time","longitude","latitude","unit"])
    return df, df["unit"].iloc[0]

def find_daylight_window(stop_utc: dt.datetime, window_hours: int,
                         bbox: Tuple[float,float,float,float],
                         max_windows_back: int = 16) -> Tuple[Optional[dt.datetime], Optional[dt.datetime]]:
    for k in range(max_windows_back):
        cand_stop = stop_utc - dt.timedelta(hours=window_hours * k)
        cand_start = cand_stop - dt.timedelta(hours=window_hours)
        mid = cand_start + (cand_stop - cand_start) / 2
        if is_daylight_over_bbox(mid, bbox):
            return cand_start, cand_stop
    return None, None

def run_once() -> Tuple[str, pd.DataFrame]:
    load_dotenv()
    user = os.getenv("EARTHDATA_USER")
    pwd  = os.getenv("EARTHDATA_PASS")
    if not user or not pwd:
        raise RuntimeError("Set EARTHDATA_USER and EARTHDATA_PASS")

    bbox_str = os.getenv("BBOX_WSEN", "-119.2,33.3,-117.3,34.7").strip()
    bbox = _parse_bbox_ws_en(bbox_str)
    out_dir = os.getenv("OUT_DIR", "./out").strip()
    os.makedirs(out_dir, exist_ok=True)

    window_hours = int(os.getenv("WINDOW_HOURS", "4"))
    max_windows_back = int(os.getenv("MAX_WINDOWS_BACK", "16"))

    now_utc = dt.datetime.utcnow()
    rounded_stop = round_down_to_hours(now_utc, window_hours)

    t0, t1 = find_daylight_window(rounded_stop, window_hours, bbox, max_windows_back)
    if t0 is None:
        raise RuntimeError(f"No daylight {window_hours}h window found in last {window_hours*max_windows_back}h.")

    files, _, _ = harmony_fetch_files((user, pwd), bbox, t1, window_hours)
    df, unit_used = subset_stack_to_df(files, t0, t1, bbox)

    stamp = f"{t0:%Y%m%dT%H%M}_{t1:%H%M}Z"
    csv_path = os.path.join(out_dir, f"tempo_no2_nrt_{stamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Save] CSV: {csv_path}  rows: {len(df):,}")

    # quicklook
    try:
        import matplotlib
        matplotlib.use("Agg")
        plt.figure(figsize=(9,6.5))
        dplot = df if len(df) <= 250_000 else df.sample(250_000, random_state=0)
        sc = plt.scatter(dplot["longitude"], dplot["latitude"], s=0.6, c=dplot["no2_column"])
        plt.colorbar(sc, label=f"NO₂ column ({unit_used})")
        plt.title(f"TEMPO NRT NO₂  {t0.isoformat()}..{t1.isoformat()}  files:{len(files)}")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        W,S,E,N = bbox
        plt.xlim([W,E]); plt.ylim([S,N])
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"tempo_no2_nrt_{stamp}.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[Save] PNG: {png_path}")
    except Exception as e:
        print(f"[Plot] skipped: {e}")

    return csv_path, df

# Advanced: pass explicit window (used by Main.run_tempo_exact)
def run_once_with_temporal(start_iso: str, end_iso: str) -> Tuple[str, pd.DataFrame]:
    load_dotenv()
    user = os.getenv("EARTHDATA_USER")
    pwd  = os.getenv("EARTHDATA_PASS")
    bbox_str = os.getenv("BBOX_WSEN", "-119.2,33.3,-117.3,34.7").strip()
    bbox = _parse_bbox_ws_en(bbox_str)
    out_dir = os.getenv("OUT_DIR", "./out").strip()
    os.makedirs(out_dir, exist_ok=True)

    t0 = dt.datetime.fromisoformat(start_iso.replace("Z","+00:00"))
    t1 = dt.datetime.fromisoformat(end_iso.replace("Z","+00:00"))

    files, _, _ = harmony_fetch_files((user,pwd), bbox, t1, int((t1-t0).total_seconds()//3600))
    df, _unit = subset_stack_to_df(files, t0, t1, bbox)
    stamp = f"{t0:%Y%m%dT%H%M}_{t1:%H%M}Z"
    csv_path = os.path.join(out_dir, f"tempo_no2_nrt_{stamp}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, df
