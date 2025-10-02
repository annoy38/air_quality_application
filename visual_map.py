# visual_api.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse

OUT_DIR = os.getenv("OUT_DIR", "./out")
GRID_STEP = float(os.getenv("WEATHER_GRID_STEP", "0.25"))

def _snap(val: float, step: float) -> float:
    return float(round(val/step)*step)

def _latest_fused_path() -> Optional[str]:
    paths = sorted(Path(OUT_DIR).glob("fused_*.csv"))
    return paths[-1].as_posix() if paths else None

def _latest_day_str() -> str:
    p = _latest_fused_path()
    if not p:
        raise FileNotFoundError("No fused_*.csv in OUT_DIR")
    fname = Path(p).name  # fused_YYYYMMDD.csv
    day = fname.replace("fused_","").replace(".csv","")
    return f"{day[0:4]}-{day[4:6]}-{day[6:8]}"

def _load_fused_day(day: str) -> pd.DataFrame:
    p = Path(OUT_DIR) / f"fused_{day.replace('-','')}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Fused CSV not found: {p}")
    df = pd.read_csv(p)
    df.columns = [c.replace(" ","_").replace(".","_").lower() for c in df.columns]
    if "time_hour" not in df.columns:
        raise ValueError("Missing time_hour column in fused CSV")
    df["time_hour"] = pd.to_datetime(df["time_hour"], utc=True, errors="coerce")
    if "no2_ground_ug_m3" not in df.columns: df["no2_ground_ug_m3"] = np.nan
    if "no2_trop_col_corrected_ug_m3" not in df.columns: df["no2_trop_col_corrected_ug_m3"] = np.nan
    df["no2_fused_ugm3"] = df["no2_ground_ug_m3"].where(
        df["no2_ground_ug_m3"].notna(), df["no2_trop_col_corrected_ug_m3"]
    )
    return df

# =========================
# Service functions (call these from api_server.py)
# =========================

def svc_latest_day():
    """Return latest fused day as {'day': 'YYYY-MM-DD'}"""
    try:
        return {"day": _latest_day_str()}
    except Exception as e:
        raise HTTPException(500, str(e))

def svc_grid_hour(day: Optional[str], hour: Optional[int], limit: int = 2000):
    """Return grid points for a given day/hour for the map."""
    try:
        if not day:
            day = _latest_day_str()
        df = _load_fused_day(day)
        if hour is None:
            target = df["time_hour"].max()
        else:
            target = datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00")
        sub = df[df["time_hour"] == target].copy()
        if sub.empty:
            return {"time_hour": target.isoformat(), "points": []}
        pts = sub[["lat_bin","lon_bin","no2_fused_ugm3","no2_ground_ug_m3","no2_trop_col_corrected_ug_m3"]]
        pts = pts.rename(columns={"lat_bin":"lat","lon_bin":"lon","no2_fused_ugm3":"no2"})
        if len(pts) > limit:
            pts = pts.sample(limit, random_state=0)
        return {"time_hour": target.isoformat(), "points": pts.round(4).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"grid/hour error: {e}")

def svc_series(lat: float, lon: float, days: int = 7):
    """Return a time series for charts at the nearest grid cell for the last N days + forecast if available."""
    try:
        # find last N fused_* files
        paths = sorted(Path(OUT_DIR).glob("fused_*.csv"))[-days:]
        if not paths:
            raise FileNotFoundError("No fused CSVs found")
        lat_bin_req = _snap(lat, GRID_STEP)
        lon_bin_req = _snap(lon, GRID_STEP)
        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            df.columns = [c.replace(" ","_").replace(".","_").lower() for c in df.columns]
            if "time_hour" not in df.columns:
                continue
            df["time_hour"] = pd.to_datetime(df["time_hour"], utc=True, errors="coerce")
            if "no2_ground_ug_m3" not in df.columns: df["no2_ground_ug_m3"] = np.nan
            if "no2_trop_col_corrected_ug_m3" not in df.columns: df["no2_trop_col_corrected_ug_m3"] = np.nan
            df["no2_fused_ugm3"] = df["no2_ground_ug_m3"].where(
                df["no2_ground_ug_m3"].notna(), df["no2_trop_col_corrected_ug_m3"]
            )
            # nearest present bin
            uniq = df[["lat_bin","lon_bin"]].drop_duplicates()
            if uniq.empty:
                continue
            d2 = (uniq["lat_bin"]-lat_bin_req)**2 + (uniq["lon_bin"]-lon_bin_req)**2
            i = int(np.argmin(d2.values))
            nb = float(uniq.iloc[i]["lat_bin"]); mb = float(uniq.iloc[i]["lon_bin"])
            dfs.append(df[(df["lat_bin"]==nb)&(df["lon_bin"]==mb)][
                ["time_hour","no2_fused_ugm3","no2_ground_ug_m3","no2_trop_col_corrected_ug_m3"]
            ])
        out = pd.concat(dfs, ignore_index=True).sort_values("time_hour")

        # attach forecast if exists
        fpath = Path(OUT_DIR)/"no2_forecast.csv"
        if fpath.exists():
            fc = pd.read_csv(fpath)
            fc.columns = [c.replace(" ","_").replace(".","_").lower() for c in fc.columns]
            if {"valid_time_utc","lat_bin","lon_bin","no2_ugm3"}.issubset(set(fc.columns)):
                sel = fc[(fc["lat_bin"].round(6)==round(lat_bin_req,6)) &
                         (fc["lon_bin"].round(6)==round(lon_bin_req,6))].copy()
                sel = sel.rename(columns={"valid_time_utc":"time_hour","no2_ugm3":"forecast_ugm3"})
                sel["time_hour"] = pd.to_datetime(sel["time_hour"], utc=True, errors="coerce")
                out = out.merge(sel[["time_hour","forecast_ugm3"]], on="time_hour", how="outer")

        out = out.dropna(subset=["time_hour"]).sort_values("time_hour").reset_index(drop=True)
        return JSONResponse({
            "lat_bin": float(lat_bin_req),
            "lon_bin": float(lon_bin_req),
            "series": [
                {
                    "time": t.isoformat(),
                    "no2_fused_ugm3": (0.0 if pd.isna(v) else float(v)),
                    "no2_ground_ug_m3": (None if pd.isna(g) else float(g)),
                    "no2_trop_col_corr_ug_m3": (None if pd.isna(s) else float(s)),
                    "forecast_ugm3": (None if pd.isna(f) else float(f)),
                }
                for t, v, g, s, f in zip(
                    out["time_hour"], out["no2_fused_ugm3"],
                    out["no2_ground_ug_m3"], out["no2_trop_col_corrected_ug_m3"],
                    out.get("forecast_ugm3", [np.nan]*len(out)),
                )
            ]
        })
    except Exception as e:
        raise HTTPException(500, f"series error: {e}")

def svc_map_latest():
    """Return FileResponse to the latest single-hour map HTML."""
    day = _latest_day_str().replace("-","")
    p = Path(OUT_DIR)/f"fused_map_{day}.html"
    if not p.exists():
        raise HTTPException(404, f"map not found: {p}")
    return FileResponse(p)

def svc_map_multi():
    """Return FileResponse to the latest multi-hour (time slider) map HTML."""
    day = _latest_day_str().replace("-","")
    p = Path(OUT_DIR)/f"fused_multihour_map_{day}.html"
    if not p.exists():
        raise HTTPException(404, f"multihour map not found: {p}")
    return FileResponse(p)

# (Optional) If you still want to mount a router here, you can wrap these
# with FastAPI decorators in this module. But since you asked to define routes
# in api_server.py, we'll call svc_* from there instead.
