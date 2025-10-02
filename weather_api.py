#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Open-Meteo fetcher with forecast/archive splice + dedupe + optional regridding.
"""

import argparse, time
from datetime import datetime, timezone, timedelta
from math import ceil
from typing import List, Tuple, Optional

import pandas as pd
import requests
from tqdm import tqdm

FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_BASE  = "https://archive-api.open-meteo.com/v1/era5"

DEFAULT_HOURLY = [
    "temperature_2m","relative_humidity_2m","dewpoint_2m",
    "surface_pressure","pressure_msl",
    "wind_speed_10m","wind_direction_10m","wind_gusts_10m",
    "precipitation","rain","snowfall",
    "cloud_cover","shortwave_radiation","visibility"
]

def _frange(a: float, b: float, step: float) -> List[float]:
    out, x = [], a
    while x <= b + 1e-9:
        out.append(round(x, 6)); x += step
    return out

def _grid_from_bbox(w: float, s: float, e: float, n: float, step: float) -> List[Tuple[float, float]]:
    return [(lat, lon) for lat in _frange(s, n, step) for lon in _frange(w, e, step)]

def _forecast_point(lat: float, lon: float, hourly_vars: List[str], past_days: int, forecast_hours: int, tz: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly_vars),
              "timeformat": "iso8601", "timezone": tz}
    if past_days > 0: params["past_days"] = past_days
    if forecast_hours > 0: params["forecast_days"] = max(1, (forecast_hours + 23)//24)
    r = requests.get(FORECAST_BASE, params=params, timeout=60); r.raise_for_status()
    js = r.json()
    if "hourly" not in js or "time" not in js["hourly"]:
        return pd.DataFrame()
    hourly = js["hourly"]
    df = pd.DataFrame({"time": hourly["time"]})
    for k, v in hourly.items():
        if k != "time": df[k] = v
    df["time"] = pd.to_datetime(df["time"])
    df.insert(1, "lat", lat); df.insert(2, "lon", lon)
    df["source"] = "forecast"
    return df

def _archive_point(lat: float, lon: float, hourly_vars: List[str], start_date: str, end_date: str, tz: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly_vars),
              "start_date": start_date, "end_date": end_date,
              "timeformat": "iso8601", "timezone": tz}
    r = requests.get(ARCHIVE_BASE, params=params, timeout=60); r.raise_for_status()
    js = r.json()
    if "hourly" not in js or "time" not in js["hourly"]:
        return pd.DataFrame()
    hourly = js["hourly"]
    df = pd.DataFrame({"time": hourly["time"]})
    for k, v in hourly.items():
        if k != "time": df[k] = v
    df["time"] = pd.to_datetime(df["time"])
    df.insert(1, "lat", lat); df.insert(2, "lon", lon)
    df["source"] = "archive"
    return df

def _snap(val: pd.Series, step: float) -> pd.Series:
    return (val / step).round(0) * step

def fetch_weather_to_csv(
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,  # (W,S,E,N)
    grid_step: float = 0.25,
    past_hours: Optional[int] = None,
    past_days: int = 1,
    forecast_hours: int = 24,
    timezone_str: str = "UTC",
    hourly_vars: Optional[List[str]] = None,
    out_csv: str = "",
    sleep: float = 0.25,
    regrid_step: Optional[float] = None,   # NEW: if set, snap to this grid and aggregate mean
) -> tuple[str, pd.DataFrame]:
    if (lat is None or lon is None) and bbox is None:
        raise ValueError("Provide either (lat & lon) or bbox=(W,S,E,N).")
    vars_ = hourly_vars or DEFAULT_HOURLY

    effective_past_days = past_days
    if past_hours is not None:
        effective_past_days = max(1, ceil(past_hours / 24))

    points: List[Tuple[float, float]]
    if bbox is not None:
        w, s, e, n = bbox
        if w > e or s > n: raise ValueError("Invalid bbox.")
        points = _grid_from_bbox(w, s, e, n, grid_step)
    else:
        points = [(float(lat), float(lon))]

    now_utc = datetime.now(timezone.utc)
    splice_threshold = now_utc - timedelta(hours=48)   # forecast API covers ~past 48h reliably

    dfs = []
    for (la, lo) in tqdm(points, desc="Open-Meteo"):
        try:
            # Forecast chunk (past + future)
            df_f = _forecast_point(la, lo, vars_, past_days=effective_past_days,
                                   forecast_hours=forecast_hours, tz=timezone_str)
            # Archive only if request goes earlier than splice threshold
            need_archive = False
            if past_hours is not None:
                need_archive = (now_utc - timedelta(hours=past_hours)) < df_f["time"].min(tz=None)
            # Safer: if earliest desired < now-48h, pull archive
            need_archive = need_archive or ((now_utc - timedelta(hours=past_hours or 0)) < splice_threshold)

            if need_archive:
                start_date = (now_utc - timedelta(hours=past_hours or 0)).date().isoformat()
                end_date   = now_utc.date().isoformat()
                df_a = _archive_point(la, lo, vars_, start_date, end_date, tz=timezone_str)
            else:
                df_a = pd.DataFrame()

            df = pd.concat([df_a, df_f], ignore_index=True)
            if df.empty:
                continue

            # Normalize & dedupe
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values(["time","lat","lon","source"]).drop_duplicates(subset=["time","lat","lon"], keep="last")

            dfs.append(df)
        except Exception as ex:
            print(f"[WARN] {la},{lo}: {ex}")
        time.sleep(sleep)

    if not dfs:
        raise RuntimeError("No weather rows returned — try changing grid/vars/time window.")

    out_df = pd.concat(dfs, ignore_index=True)
    if past_hours is not None:
        start_utc = now_utc - timedelta(hours=past_hours)
        out_df = out_df[(out_df["time"] >= start_utc) & (out_df["time"] <= now_utc)]

    out_df = out_df.sort_values(["time", "lat", "lon"]).reset_index(drop=True)

    # Optional regrid at source (align to fusion grid early)
    if regrid_step:
        out_df["lat_bin"] = _snap(out_df["lat"], regrid_step)
        out_df["lon_bin"] = _snap(out_df["lon"], regrid_step)
        num_cols = [c for c in out_df.columns if c not in ("time","lat","lon","source","lat_bin","lon_bin")]
        out_df = (out_df.groupby(["time","lat_bin","lon_bin"], as_index=False)
                        .agg({c: "mean" for c in num_cols}))
        # rename back for downstream compatibility
        out_df = out_df.rename(columns={"lat_bin":"lat","lon_bin":"lon"})

    out_df.to_csv(out_csv, index=False)
    return out_csv, out_df

# ------------- CLI -------------
def _parse():
    ap = argparse.ArgumentParser(description="Fetch Open-Meteo hourly weather to CSV.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lat", type=float, help="Latitude")
    ap.add_argument("--lon", type=float, help="Longitude (required if --lat used)")
    mode.add_argument("--bbox", nargs=4, type=float, metavar=("WEST","SOUTH","EAST","NORTH"))
    ap.add_argument("--grid-step", type=float, default=0.25)
    ap.add_argument("--past-hours", type=int, help="Only last N hours (trim).")
    ap.add_argument("--forecast-hours", type=int, default=24)
    ap.add_argument("--timezone", type=str, default="UTC")
    ap.add_argument("--vars", type=str, help="Comma-separated hourly vars")
    ap.add_argument("--regrid-step", type=float, help="Snap to grid and aggregate mean.")
    ap.add_argument("-o","--out", type=str, required=True)
    return ap.parse_args()

def main():
    args = _parse()
    hv = DEFAULT_HOURLY if not args.vars else [v.strip() for v in args.vars.split(",") if v.strip()]
    bbox = tuple(args.bbox) if args.bbox else None
    csv_path, _ = fetch_weather_to_csv(
        lat=args.lat, lon=args.lon, bbox=bbox, grid_step=args.grid_step,
        past_hours=args.past_hours, past_days=1,
        forecast_hours=args.forecast_hours,
        timezone_str=args.timezone, hourly_vars=hv, out_csv=args.out,
        regrid_step=args.regrid_step
    )
    print(f"✅ Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
