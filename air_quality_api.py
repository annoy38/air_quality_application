#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI server exposing NO₂ (California) endpoints for your TEMPO + OpenAQ + Weather pipeline.

Endpoints (all JSON):
- GET  /healthz
- GET  /api/meta/datasources
- GET  /api/aqi/no2/breakpoints
- GET  /api/no2/latest?grid=0.25&bbox=-124.48,32.53,-114.13,42.01
- GET  /api/no2/timeseries?lat=34.05&lon=-118.25&hours=48
- GET  /api/no2/forecast?h=6&grid=0.25
- GET  /api/validate/no2/hourly?start=2025-10-01T00:00:00Z&end=2025-10-03T00:00:00Z
- GET  /api/alerts/no2?threshold_aqi=150&h=6
- GET  /api/cells?grid=0.25&bbox=...  (grid metadata)
- GET  /api/stations?bbox=...         (OpenAQ station overlay)
- GET  /api/stations/{id}/timeseries?hours=48  (OpenAQ station series)

Implementation notes:
- Reads your generated daily CSVs: fused_YYYYMMDD.csv and fused_metrics_YYYYMMDD.csv in OUT_DIR.
- Fused NO₂ per cell = prefer ground (µg/m³) else satellite-bias-corrected (µg/m³).
- Converts to ppb using cell temperature/pressure (fallback: 293.15 K, 101325 Pa).
- AQI computed via configurable breakpoints (with safe defaults here; override later if needed).
- Forecast model = persistence baseline (copy latest forward for h=1..6).

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8080 --reload

Dependencies:
    pip install fastapi uvicorn pydantic pandas numpy python-dotenv

This file depends on your existing modules in the same project:
    config.py, fuse_validate_merge.py, openaq_api.py, net_utils.py
"""
from __future__ import annotations

import os
import glob
import math
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import alert_user
from alert_user import No2AlertResponse, No2AlertRequest, no2_alert
from fastapi.responses import FileResponse, JSONResponse

# import the service functions
from visual_map import (
    svc_latest_day,
    svc_grid_hour,
    svc_series,
    svc_map_latest,
    svc_map_multi,
)
from pydantic import BaseModel

from config import get_config
from net_utils import SESSION

# ---------- Constants & helpers ----------
R = 8.31446261815324  # J/(mol*K) = Pa*m^3/(mol*K)
MW_NO2 = 46.0055      # g/mol

cfg = get_config()
OUT_DIR = Path(cfg.get("OUT_DIR", "./out"))
BBOX_WSEN = tuple(float(x) for x in cfg.get("BBOX_WSEN", "-124.48,32.53,-114.13,42.01").split(","))
GRID_STEP_DEFAULT = float(cfg.get("GRID_STEP", cfg.get("WEATHER_GRID_STEP", 0.25)))
OPENAQ_API_KEY = cfg.get("OPENAQ_API_KEY", "")

# Safe default NO₂ 1-hour AQI breakpoints (ppb). Adjust via /api/aqi/no2/breakpoints later.
# Each row: (bp_low_ppb, bp_high_ppb, aqi_low, aqi_high, category)
AQI_NO2_BREAKPOINTS_DEFAULT = [
    (0,   53,   0,   50,  "Good"),
    (54,  100,  51, 100,  "Moderate"),
    (101, 360,  101, 150, "Unhealthy for Sensitive Groups"),
    (361, 649,  151, 200, "Unhealthy"),
    (650, 1249, 201, 300, "Very Unhealthy"),
    (1250,2049, 301, 500, "Hazardous"),
]

# Optionally load alternate breakpoints from env JSON: NO2_AQI_BREAKPOINTS_JSON
try:
    _bp_env = os.getenv("NO2_AQI_BREAKPOINTS_JSON")
    AQI_NO2_BREAKPOINTS = json.loads(_bp_env) if _bp_env else AQI_NO2_BREAKPOINTS_DEFAULT
except Exception:
    AQI_NO2_BREAKPOINTS = AQI_NO2_BREAKPOINTS_DEFAULT


def _round_grid(x: float, step: float) -> float:
    return float(round(x / step) * step)


def _cell_id(lat_bin: float, lon_bin: float, step: float) -> str:
    return f"CA_{step:.2f}_{lat_bin:+.2f}_{lon_bin:+.2f}"


def _ugm3_to_ppb(ugm3: float, temp_k: float, pressure_pa: float) -> Optional[float]:
    if ugm3 is None or not math.isfinite(ugm3):
        return None
    T = temp_k if (temp_k and math.isfinite(temp_k) and temp_k > 0) else 293.15
    P = pressure_pa if (pressure_pa and math.isfinite(pressure_pa) and pressure_pa > 0) else 101_325.0
    # ppb = ug/m3 * R * T / (MW * P) * 1e3
    return float(ugm3) * R * T / (MW_NO2 * P) * 1e3


def _compute_aqi_no2_ppb(x_ppb: Optional[float]) -> Tuple[Optional[int], str]:
    if x_ppb is None or not math.isfinite(x_ppb):
        return None, "Unknown"
    for lo, hi, aqi_lo, aqi_hi, cat in AQI_NO2_BREAKPOINTS:
        if lo <= x_ppb <= hi:
            aqi = (aqi_hi - aqi_lo) / (hi - lo) * (x_ppb - lo) + aqi_lo
            return int(round(aqi)), cat
    # If above last breakpoint
    if x_ppb > AQI_NO2_BREAKPOINTS[-1][1]:
        return 500, AQI_NO2_BREAKPOINTS[-1][4]
    return None, "Out of range"


def _list_daily_files(pattern: str, since_days: int = 3) -> List[Path]:
    paths = sorted(Path(OUT_DIR).glob(pattern))
    if not paths:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    filtered = []
    for p in reversed(paths):  # newest first by filename convention
        try:
            # pattern has date; parse YYYYMMDD from name if possible
            s = p.name
            i = s.find("_")
            j = s.rfind(".")
            if i >= 0 and j > i:
                date_part = s[i+1:j]
                if len(date_part) >= 8:
                    d = datetime.strptime(date_part[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
                    if d >= cutoff:
                        filtered.append(p)
        except Exception:
            filtered.append(p)
    return list(reversed(filtered))  # oldest→newest within window


def _load_fused_frames(days: int = 2) -> pd.DataFrame:
    files = _list_daily_files("fused_*.csv", since_days=days)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # Normalize cols
    cols = [c.replace(" ", "_").replace(".", "_").lower() for c in df.columns]
    df.columns = cols
    # time
    tcol = "time_hour" if "time_hour" in df.columns else ("time" if "time" in df.columns else None)
    if not tcol:
        return pd.DataFrame()
    df["time_utc"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    # bins
    latc = "lat_bin" if "lat_bin" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lonc = "lon_bin" if "lon_bin" in df.columns else ("longitude" if "longitude" in df.columns else None)
    if (latc is None) or (lonc is None):
        return pd.DataFrame()
    df["lat_bin"] = pd.to_numeric(df[latc], errors="coerce")
    df["lon_bin"] = pd.to_numeric(df[lonc], errors="coerce")

    # Provide temperature & pressure if present
    if "temp_k" not in df.columns:
        if "temperature_2m" in df.columns:
            df["temp_k"] = pd.to_numeric(df["temperature_2m"], errors="coerce") + 273.15
        else:
            df["temp_k"] = np.nan
    if "pressure_pa" not in df.columns:
        if "pressure_msl" in df.columns:
            df["pressure_pa"] = pd.to_numeric(df["pressure_msl"], errors="coerce") * 100.0
        elif "surface_pressure" in df.columns:
            df["pressure_pa"] = pd.to_numeric(df["surface_pressure"], errors="coerce") * 100.0
        else:
            df["pressure_pa"] = np.nan

    # Fused choices (µg/m³) — prefer ground, else corrected satellite
    gcol = "no2_ground_ug_m3"
    scol = "no2_trop_col_corrected_ug_m3"
    if gcol not in df.columns:
        df[gcol] = np.nan
    if scol not in df.columns:
        df[scol] = np.nan
    df["no2_fused_ugm3"] = df[gcol].where(pd.notna(df[gcol]), df[scol])

    # Conversions
    df["no2_ppb"] = df.apply(lambda r: _ugm3_to_ppb(r["no2_fused_ugm3"], r["temp_k"], r["pressure_pa"]), axis=1)
    aqi_and_cat = df["no2_ppb"].apply(_compute_aqi_no2_ppb)
    df["aqi"] = [v[0] for v in aqi_and_cat]
    df["category"] = [v[1] for v in aqi_and_cat]

    # Clip to configured bbox
    W, S, E, N = BBOX_WSEN
    df = df[(df["lat_bin"] >= S) & (df["lat_bin"] <= N) & (df["lon_bin"] >= W) & (df["lon_bin"] <= E)]

    return df


# ---------- FastAPI app ----------
app = FastAPI(title="CA NO₂ API", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/meta/datasources")
def meta_datasources():
    def _last(path_glob: str) -> Optional[str]:
        files = sorted(OUT_DIR.glob(path_glob))
        if not files:
            return None
        ts = datetime.fromtimestamp(files[-1].stat().st_mtime, tz=timezone.utc)
        return ts.isoformat()

    return {
        "out_dir": str(OUT_DIR),
        "bbox_wsen": BBOX_WSEN,
        "grid_step_default": GRID_STEP_DEFAULT,
        "last_fused_csv_mtime": _last("fused_*.csv"),
        "last_metrics_csv_mtime": _last("fused_metrics_*.csv"),
        "sources": {
            "satellite": "NASA TEMPO (Harmony; L2/L3 NRT)",
            "ground": "OpenAQ v3",
            "weather": "Open-Meteo",
        },
    }


@app.get("/api/aqi/no2/breakpoints")
def aqi_no2_breakpoints():
    return {
        "units": "ppb (1-hour)",
        "breakpoints": [
            {
                "bp_low_ppb": lo,
                "bp_high_ppb": hi,
                "aqi_low": aqi_lo,
                "aqi_high": aqi_hi,
                "category": cat,
            }
            for (lo, hi, aqi_lo, aqi_hi, cat) in AQI_NO2_BREAKPOINTS
        ],
    }


@app.get("/api/no2/latest")
def no2_latest(
    grid: float = Query(GRID_STEP_DEFAULT, ge=0.05, le=2.0),
    bbox: Optional[str] = Query(None, description="W,S,E,N (override default CA bbox)"),
):
    df = _load_fused_frames(days=2)
    if df.empty:
        raise HTTPException(status_code=404, detail="No fused data available")

    if bbox:
        try:
            W, S, E, N = [float(x) for x in bbox.split(",")]
            df = df[(df["lat_bin"] >= S) & (df["lat_bin"] <= N) & (df["lon_bin"] >= W) & (df["lon_bin"] <= E)]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox")

    t_latest = df["time_utc"].max()
    cur = df[df["time_utc"] == t_latest].copy()

    # Re-bin if different grid requested
    if abs(grid - GRID_STEP_DEFAULT) > 1e-9:
        cur["lat_bin"] = cur["lat_bin"].apply(lambda v: _round_grid(v, grid))
        cur["lon_bin"] = cur["lon_bin"].apply(lambda v: _round_grid(v, grid))
        cur = (
            cur.groupby(["lat_bin", "lon_bin"], as_index=False)
               .agg({"no2_fused_ugm3":"mean","no2_ppb":"mean","aqi":"mean"})
        )
        cur["category"] = cur["aqi"].apply(lambda a: _compute_aqi_no2_ppb(a if a is not None else None)[1])
    else:
        cur = cur[["lat_bin","lon_bin","no2_fused_ugm3","no2_ppb","aqi","category"]].copy()

    cells = [
        {
            "cell_id": _cell_id(float(r.lat_bin), float(r.lon_bin), grid),
            "lat": float(r.lat_bin),
            "lon": float(r.lon_bin),
            "no2_ugm3": None if pd.isna(r.no2_fused_ugm3) else float(r.no2_fused_ugm3),
            "no2_ppb": None if pd.isna(r.no2_ppb) else float(r.no2_ppb),
            "aqi": None if pd.isna(r.aqi) else int(round(float(r.aqi))),
            "category": r.category if isinstance(r.category, str) else _compute_aqi_no2_ppb(r.no2_ppb)[1],
        }
        for r in cur.itertuples(index=False)
    ]

    return {
        "time_utc": t_latest.isoformat(),
        "grid": grid,
        "cells": cells,
        "provenance": {"satellite": "TEMPO", "ground": "OpenAQ", "weather": "Open-Meteo"},
    }


@app.get("/api/no2/timeseries")
def no2_timeseries(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    cell_id: Optional[str] = None,
    hours: int = Query(48, ge=1, le=168),
):
    if (lat is None or lon is None) and not cell_id:
        raise HTTPException(status_code=400, detail="Provide (lat & lon) or cell_id")

    step = GRID_STEP_DEFAULT
    if cell_id:
        try:
            # format CA_{step}_{lat}_{lon}
            parts = cell_id.split("_")
            step = float(parts[1])
            lat_bin = float(parts[2])
            lon_bin = float(parts[3])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cell_id format")
    else:
        lat_bin = _round_grid(float(lat), step)
        lon_bin = _round_grid(float(lon), step)

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = _load_fused_frames(days=7)
    if df.empty:
        raise HTTPException(status_code=404, detail="No fused data available")
    sub = df[(df["time_utc"] >= since) & (df["lat_bin"] == lat_bin) & (df["lon_bin"] == lon_bin)].copy()

    if sub.empty:
        return {"cell_id": _cell_id(lat_bin, lon_bin, step), "series": []}

    series = [
        {
            "time_utc": t.isoformat(),
            "no2_ugm3": None if pd.isna(u) else float(u),
            "no2_ppb": None if pd.isna(p) else float(p),
            "aqi": None if pd.isna(a) else int(round(float(a))),
            "category": c if isinstance(c, str) else _compute_aqi_no2_ppb(p)[1],
        }
        for t, u, p, a, c in zip(sub["time_utc"], sub["no2_fused_ugm3"], sub["no2_ppb"], sub["aqi"], sub["category"])
    ]

    return {
        "cell_id": _cell_id(lat_bin, lon_bin, step),
        "lat": lat_bin,
        "lon": lon_bin,
        "series": series,
    }


@app.get("/api/no2/forecast")
def no2_forecast(h: int = Query(6, ge=1, le=24), grid: float = GRID_STEP_DEFAULT):
    """Persistence nowcast: copy the latest hour forward for h=1..N per cell."""
    df = _load_fused_frames(days=2)
    if df.empty:
        raise HTTPException(status_code=404, detail="No fused data available")
    t_latest = df["time_utc"].max()
    base = df[df["time_utc"] == t_latest].copy()

    # Re-bin if different grid
    if abs(grid - GRID_STEP_DEFAULT) > 1e-9:
        base["lat_bin"] = base["lat_bin"].apply(lambda v: _round_grid(v, grid))
        base["lon_bin"] = base["lon_bin"].apply(lambda v: _round_grid(v, grid))
        base = (
            base.groupby(["lat_bin","lon_bin"], as_index=False)
                .agg({"no2_fused_ugm3":"mean","no2_ppb":"mean","aqi":"mean"})
        )
        base["category"] = base["aqi"].apply(lambda a: _compute_aqi_no2_ppb(a if a is not None else None)[1])

    horizons = list(range(1, h+1))
    forecasts = []
    for hh in horizons:
        vt = t_latest + timedelta(hours=hh)
        for r in base.itertuples(index=False):
            aqi_int = None if pd.isna(r.aqi) else int(round(float(r.aqi)))
            cat = r.category if hasattr(r, "category") and isinstance(r.category, str) else _compute_aqi_no2_ppb(getattr(r, "no2_ppb", None))[1]
            forecasts.append({
                "valid_time_utc": vt.isoformat(),
                "cell_id": _cell_id(float(r.lat_bin), float(r.lon_bin), grid),
                "lat": float(r.lat_bin),
                "lon": float(r.lon_bin),
                "no2_ugm3": None if pd.isna(r.no2_fused_ugm3) else float(r.no2_fused_ugm3),
                "no2_ppb": None if hasattr(r, "no2_ppb") and pd.isna(r.no2_ppb) else (None if not hasattr(r, "no2_ppb") else float(r.no2_ppb)),
                "aqi": aqi_int,
                "category": cat,
                "model": "persistence",
            })

    return {
        "time_run_utc": datetime.now(timezone.utc).isoformat(),
        "based_on_time": t_latest.isoformat(),
        "horizon_hours": horizons,
        "grid": grid,
        "forecasts": forecasts,
    }


@app.get("/api/validate/no2/hourly")
def validate_no2_hourly(start: Optional[str] = None, end: Optional[str] = None):
    def _parse(iso: Optional[str], default: datetime) -> datetime:
        if not iso:
            return default
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid datetime format")

    now = datetime.now(timezone.utc)
    t0 = _parse(start, now - timedelta(days=2))
    t1 = _parse(end, now)

    files = _list_daily_files("fused_metrics_*.csv", since_days=7)
    if not files:
        return {"metrics": []}

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.replace(" ", "_").lower() for c in df.columns]
            if "time_hour" in df.columns:
                df["time_utc"] = pd.to_datetime(df["time_hour"], utc=True, errors="coerce")
            else:
                continue
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return {"metrics": []}

    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["time_utc"] >= t0) & (df["time_utc"] <= t1)].copy()

    metrics = [
        {
            "time_utc": r.time_utc.isoformat(),
            "n_cells": int(r.n_cells) if hasattr(r, "n_cells") and not pd.isna(r.n_cells) else None,
            "r_sat_grd": None if pd.isna(getattr(r, "corr_weighted", np.nan)) else float(r.corr_weighted),
            "mae_sat_grd": None,  # not in current metrics CSV; RMSE provided instead
            "rmse_sat_grd": None if pd.isna(getattr(r, "rmse", np.nan)) else float(r.rmse),
            "r_fused_grd": None,  # (optionally compute in the future)
            "mae_fused_grd": None,
        }
        for r in df.itertuples(index=False)
    ]

    return {"metrics": metrics}


@app.get("/api/alerts/no2")
def alerts_no2(threshold_aqi: int = Query(150, ge=0, le=500), h: int = Query(6, ge=1, le=24)):
    fc = no2_forecast(h=h)
    hits: Dict[str, Dict[str, Any]] = {}
    for item in fc["forecasts"]:
        aqi = item.get("aqi")
        if aqi is None:
            continue
        if aqi >= threshold_aqi:
            cid = item["cell_id"]
            rec = hits.get(cid)
            if not rec:
                hits[cid] = {
                    "cell_id": cid,
                    "lat": item["lat"],
                    "lon": item["lon"],
                    "first_exceed_time": item["valid_time_utc"],
                    "peak_aqi": aqi,
                    "peak_category": item.get("category"),
                }
            else:
                if aqi > rec["peak_aqi"]:
                    rec["peak_aqi"] = aqi
                    rec["peak_category"] = item.get("category")
    return {
        "threshold_aqi": threshold_aqi,
        "until_utc": (datetime.now(timezone.utc) + timedelta(hours=h)).isoformat(),
        "hits": list(hits.values()),
    }


@app.get("/api/cells")
def cells(grid: float = Query(GRID_STEP_DEFAULT, ge=0.05, le=2.0), bbox: Optional[str] = None):
    if bbox:
        try:
            W, S, E, N = [float(x) for x in bbox.split(",")]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox")
    else:
        W, S, E, N = BBOX_WSEN

    lat_vals = np.arange(S, N + 1e-9, grid)
    lon_vals = np.arange(W, E + 1e-9, grid)
    cells = []
    for la in lat_vals:
        for lo in lon_vals:
            cells.append({
                "cell_id": _cell_id(_round_grid(la, grid), _round_grid(lo, grid), grid),
                "lat": float(_round_grid(la, grid)),
                "lon": float(_round_grid(lo, grid)),
                "bounds": [float(lo), float(la), float(lo + grid), float(la + grid)],
            })
    return {"grid": grid, "bbox": [W, S, E, N], "cells": cells}


# ---------- OpenAQ station helpers ----------
OPENAQ_BASE = "https://api.openaq.org/v3"

def _openaq_paged_get(path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    headers = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
    results: List[Dict[str, Any]] = []
    page = 1
    while True:
        qp = dict(params)
        qp.update({"limit": 1000, "page": page})
        r = SESSION.get(f"{OPENAQ_BASE}/{path}", headers=headers, params=qp, timeout=30)
        r.raise_for_status()
        js = r.json()
        chunk = js.get("results", [])
        results.extend(chunk)
        meta = js.get("meta", {})
        found = meta.get("found")
        if isinstance(found, str):
            try:
                found = int("".join([c for c in found if c.isdigit()]))
            except Exception:
                found = None
        if not chunk or (found is not None and len(results) >= int(found)):
            break
        page += 1
    return results


@app.get("/api/stations")
def stations(bbox: Optional[str] = None):
    if bbox:
        try:
            W, S, E, N = [float(x) for x in bbox.split(",")]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox")
    else:
        W, S, E, N = BBOX_WSEN

    locs = _openaq_paged_get("locations", {"bbox": f"{W},{S},{E},{N}"})
    out = []
    for L in locs:
        coords = L.get("coordinates") or {}
        lid = L.get("locationsId") or L.get("id") or L.get("locationId")
        if lid is None:
            continue
        out.append({
            "id": lid,
            "name": L.get("name"),
            "lat": coords.get("latitude"),
            "lon": coords.get("longitude"),
            "provider": (L.get("provider") or {}).get("name"),
            "owner": (L.get("owner") or {}).get("name"),
        })
    return {"stations": out}


@app.get("/api/stations/{station_id}/timeseries")
def station_timeseries(station_id: int, hours: int = Query(48, ge=1, le=168)):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    # Get sensors for this location
    sensors = _openaq_paged_get(f"locations/{station_id}/sensors", {})
    no2_sids = [s.get("sensorsId") or s.get("id") for s in sensors if str((s.get("parameter") or {}).get("name", "")).lower() == "no2"]
    if not no2_sids:
        return {"station_id": station_id, "series": []}

    rows: List[Dict[str, Any]] = []
    for sid in no2_sids:
        res = _openaq_paged_get(f"sensors/{sid}/hours", {
            "date_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        for m in res:
            utc = (((m.get("period") or {}).get("datetimeTo") or {}).get("utc")) or ((m.get("datetime") or {}).get("utc"))
            val = m.get("value")
            unit = ((m.get("parameter") or {}).get("units"))
            if utc is None or val is None:
                continue
            rows.append({"time_utc": utc, "value": val, "unit": unit})

    if not rows:
        return {"station_id": station_id, "series": []}

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.sort_values("time_utc").drop_duplicates("time_utc", keep="last")

    # NOTE: We do not convert station units to ppb here due to lack of collocated T/P per station.
    # Frontend can show raw units; or extend by sampling weather at station position.

    series = [
        {"time_utc": t.isoformat(), "value": None if pd.isna(v) else float(v), "unit": u}
        for t, v, u in zip(df["time_utc"], df["value"], df.get("unit", [None]*len(df)))
    ]
    return {"station_id": station_id, "series": series}


@app.post("/alerts/no2", response_model=No2AlertResponse)
def no2_alert(req: No2AlertRequest):
    return no2_alert(req)

@app.get("/api/latest_day")
def api_latest_day():
    return svc_latest_day()

@app.get("/api/grid/hour")
def api_grid_hour(
    day: str = Query(None, description="YYYY-MM-DD; default = latest"),
    hour: int = Query(None, ge=0, le=23, description="UTC hour 0..23; default = latest in file"),
    limit: int = Query(2000, ge=1, le=10000, description="Max points returned"),
):
    return svc_grid_hour(day=day, hour=hour, limit=limit)

@app.get("/api/series")
def api_series(
    lat: float = Query(...),
    lon: float = Query(...),
    days: int = Query(7, ge=1, le=30),
):
    # svc_series already returns a JSONResponse
    return svc_series(lat=lat, lon=lon, days=days)

@app.get("/api/map/latest")
def api_map_latest():
    # returns FileResponse
    return svc_map_latest()

@app.get("/api/map/multi")
def api_map_multi():
    # returns FileResponse
    return svc_map_multi()

# --- add near bottom of api_server.py ---

def _bootstrap_pipeline():
    """Run TEMPO → OpenAQ → Weather → Fuse once before serving the API."""
    from main import Main, run_range
    from datetime import datetime, timezone

    start = os.getenv("START_DAY", "").strip()
    end   = os.getenv("END_DAY", "").strip()
    day   = os.getenv("DAY", "").strip()

    print("[BOOTSTRAP] Starting data build...")
    if start and end:
        print(f"[BOOTSTRAP] Range: {start} → {end}")
        run_range(start, end)
    else:
        if not day:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print(f"[BOOTSTRAP] Single day: {day}")
        Main(day).run()

    # Optional: forecast (if you implemented No2Forecaster)
    try:
        from forecaster import No2Forecaster
        fc = No2Forecaster(
            fused_glob=os.getenv("FUSED_GLOB", "./out/fused_*.csv"),
            out_csv=os.getenv("FORECAST_OUT", "./out/no2_forecast.csv"),
            horizon_hours=int(os.getenv("FORECAST_H", "10")),
        )
        fc.train_and_forecast()
        print("[BOOTSTRAP] Forecast generated.")
    except Exception as e:
        print(f"[BOOTSTRAP] Forecast step skipped: {e}")


if __name__ == "__main__":
    # Only run the bootstrap when launching this file directly
    if os.getenv("BOOTSTRAP", "1") == "1":
        _bootstrap_pipeline()

    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=True)

