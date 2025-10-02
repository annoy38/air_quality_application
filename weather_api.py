#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timezone, timedelta
from math import ceil
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from net_utils import SESSION, HTTP_TIMEOUT
from config import atomic_write_csv

FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"

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
        out.append(round(x, 6))
        x += step
    return out

def _grid_from_bbox(w: float, s: float, e: float, n: float, step: float) -> List[Tuple[float, float]]:
    return [(lat, lon) for lat in _frange(s, n, step) for lon in _frange(w, e, step)]

def _forecast_point_session(lat: float, lon: float, hourly_vars: List[str], past_days: int, forecast_hours: int, tz: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly_vars), "timeformat": "iso8601", "timezone": tz}
    if past_days > 0: params["past_days"] = past_days
    if forecast_hours > 0: params["forecast_days"] = max(1, (forecast_hours + 23)//24)
    r = SESSION.get(FORECAST_BASE, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js or "time" not in js["hourly"]:
        return pd.DataFrame()
    hourly = js["hourly"]
    df = pd.DataFrame({"time": hourly["time"]})
    for k, v in hourly.items():
        if k != "time": df[k] = v
    df["time"] = pd.to_datetime(df["time"])
    df.insert(1, "lat", lat)
    df.insert(2, "lon", lon)
    return df

def fetch_weather_to_csv(
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,  # (W, S, E, N)
    grid_step: float = 0.5,
    past_hours: Optional[int] = None,
    past_days: int = 1,
    forecast_hours: int = 0,
    timezone_str: str = "UTC",
    hourly_vars: Optional[List[str]] = None,
    out_csv: str = "",
) -> tuple[str, pd.DataFrame]:
    if (lat is None or lon is None) and bbox is None:
        raise ValueError("Provide either (lat & lon) or bbox=(W,S,E,N).")
    vars_ = hourly_vars or DEFAULT_HOURLY
    effective_past_days = past_days
    if past_hours is not None:
        effective_past_days = max(1, ceil(past_hours / 24))
    if bbox is not None:
        w, s, e, n = bbox
        points = _grid_from_bbox(w, s, e, n, grid_step)
    else:
        points = [(float(lat), float(lon))]

    dfs = []
    max_workers = 12
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_forecast_point_session, la, lo, vars_, effective_past_days, forecast_hours, timezone_str)
                for (la, lo) in points]
        for fut in as_completed(futs):
            try:
                df = fut.result()
                if not df.empty: dfs.append(df)
            except Exception:
                pass

    if not dfs:
        raise RuntimeError("No weather rows returned — try changing grid/vars/time window.")

    out_df = pd.concat(dfs, ignore_index=True)
    out_df["time"] = pd.to_datetime(out_df["time"], utc=True)
    if past_hours is not None:
        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(hours=past_hours)
        out_df = out_df[(out_df["time"] >= start_utc) & (out_df["time"] <= now_utc)]

    out_df = out_df.sort_values(["time", "lat", "lon"]).reset_index(drop=True)
    atomic_write_csv(out_df, out_csv)
    return out_csv, out_df

# Optional CLI
def _parse():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch Open-Meteo hourly weather to CSV.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lat", type=float, help="Latitude")
    ap.add_argument("--lon", type=float, help="Longitude (required if --lat used)")
    mode.add_argument("--bbox", nargs=4, type=float, metavar=("WEST","SOUTH","EAST","NORTH"))
    ap.add_argument("--grid-step", type=float, default=0.5)
    ap.add_argument("--past-hours", type=int, help="Only keep the last N hours (trim). Overrides past_days.")
    ap.add_argument("--forecast-hours", type=int, default=0)
    ap.add_argument("--timezone", type=str, default="UTC")
    ap.add_argument("--vars", type=str, help="Comma-separated hourly vars")
    ap.add_argument("-o","--out", type=str, required=True)
    return ap.parse_args()

def main():
    args = _parse()
    hv = DEFAULT_HOURLY if not args.vars else [v.strip() for v in args.vars.split(",") if v.strip()]
    bbox = tuple(args.bbox) if args.bbox else None
    csv_path, _ = fetch_weather_to_csv(
        lat=args.lat, lon=args.lon, bbox=bbox, grid_step=args.grid_step,
        past_hours=args.past_hours, past_days=1,
        forecast_hours=args.forecast_hours, timezone_str=args.timezone,
        hourly_vars=hv, out_csv=args.out
    )
    print(f"✅ Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
