#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NO₂ Forecaster (CSV-only)
-------------------------
Train simple per-cell models from your fused_*.csv files and forecast the next N hours
without any database.

• Uses past fused NO₂ (µg/m³) per grid cell (lat_bin, lon_bin).
• Target y: no2_fused_ugm3 = ground NO₂ if available else bias-corrected TEMPO.
• Features: lag-1, lag-2, lag-24, hour-of-day (sin/cos). Optional T/P if present.
• Model: Ridge linear regression per cell. Fallback to AR(1) or persistence when data is sparse.
• Output CSV schema:
    time_run_utc, valid_time_utc, horizon_h, lat_bin, lon_bin,
    no2_ugm3, no2_ppb, aqi, category, model_tag, n_history, r2_train

Run:
    python no2_forecaster.py --out ./out/no2_forecast.csv --h 10 --days 7

Requirements:
    pip install numpy pandas scikit-learn python-dotenv

Notes:
- For AQI conversion (needs ppb), we convert µg/m³→ppb using last known T (K) and P (Pa)
  per cell; if missing, use T=293.15 K, P=101325 Pa.
- You can schedule this after your fuse step completes for the latest day.
"""
from __future__ import annotations

import os
import glob
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from dotenv import load_dotenv

R = 8.31446261815324  # J/(mol*K)
MW_NO2 = 46.0055      # g/mol

# Default 1h NO₂ AQI breakpoints in ppb (EPA-like, configurable).
AQI_NO2_BREAKPOINTS = [
    (0,   53,   0,   50,  "Good"),
    (54,  100,  51, 100,  "Moderate"),
    (101, 360,  101, 150, "Unhealthy for Sensitive Groups"),
    (361, 649,  151, 200, "Unhealthy"),
    (650, 1249, 201, 300, "Very Unhealthy"),
    (1250,2049, 301, 500, "Hazardous"),
]


def _ugm3_to_ppb(ugm3: float, temp_k: float | None, pressure_pa: float | None) -> float | None:
    if ugm3 is None or not (isinstance(ugm3, (int, float)) and math.isfinite(ugm3)):
        return None
    T = temp_k if (temp_k and math.isfinite(temp_k) and temp_k > 0) else 293.15
    P = pressure_pa if (pressure_pa and math.isfinite(pressure_pa) and pressure_pa > 0) else 101_325.0
    return float(ugm3) * R * T / (MW_NO2 * P) * 1e3


def _aqi_from_no2_ppb(x_ppb: float | None) -> tuple[Optional[int], str]:
    if x_ppb is None or not math.isfinite(x_ppb):
        return None, "Unknown"
    for lo, hi, aqi_lo, aqi_hi, cat in AQI_NO2_BREAKPOINTS:
        if lo <= x_ppb <= hi:
            aqi = (aqi_hi - aqi_lo) / (hi - lo) * (x_ppb - lo) + aqi_lo
            return int(round(aqi)), cat
    if x_ppb > AQI_NO2_BREAKPOINTS[-1][1]:
        return 500, AQI_NO2_BREAKPOINTS[-1][4]
    return None, "Out of range"


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace(" ", "_").replace(".", "_").lower() for c in df.columns]
    return df


def _load_fused_frames(out_dir: str | Path, days: int = 7) -> pd.DataFrame:
    paths = sorted(Path(out_dir).glob("fused_*.csv"))
    if not paths:
        return pd.DataFrame()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    keep: List[Path] = []
    for p in paths:
        name = p.name
        # Expect fused_YYYYMMDD.csv; parse date if possible
        try:
            # get last 8 digits before extension
            base = name.split(".")[0]
            date_str = base.split("_")[-1][:8]
            d = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            if d >= cutoff:
                keep.append(p)
        except Exception:
            keep.append(p)
    if not keep:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for p in keep:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = _norm_cols(df)

    # time column
    tcol = "time_hour" if "time_hour" in df.columns else ("time" if "time" in df.columns else None)
    if not tcol:
        return pd.DataFrame()
    df["time_utc"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")

    # bins
    if "lat_bin" not in df.columns:
        if "latitude" in df.columns:
            df["lat_bin"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "lon_bin" not in df.columns:
        if "longitude" in df.columns:
            df["lon_bin"] = pd.to_numeric(df["longitude"], errors="coerce")

    # fused target (µg/m³): prefer ground, else corrected TEMPO
    if "no2_ground_ug_m3" not in df.columns:
        df["no2_ground_ug_m3"] = np.nan
    if "no2_trop_col_corrected_ug_m3" not in df.columns:
        df["no2_trop_col_corrected_ug_m3"] = np.nan
    df["no2_fused_ugm3"] = df["no2_ground_ug_m3"].where(df["no2_ground_ug_m3"].notna(), df["no2_trop_col_corrected_ug_m3"])

    # weather support
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

    # Keep only relevant columns
    keep_cols = [
        "time_utc", "lat_bin", "lon_bin", "no2_fused_ugm3", "temp_k", "pressure_pa",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["time_utc", "lat_bin", "lon_bin"])
    return df


@dataclass
class NO2Forecaster:
    out_dir: str = "./out"
    aqi_breakpoints: Optional[List[Tuple[float, float, int, int, str]]] = None

    def __post_init__(self):
        load_dotenv()
        self.aqi_breakpoints = self.aqi_breakpoints or AQI_NO2_BREAKPOINTS

    # --------- public API ---------
    def forecast(self, *, horizon_h: int = 10, days_history: int = 7,
                 min_points_lr: int = 48) -> pd.DataFrame:
        """Return forecast dataframe for next horizon_h hours for all cells found in fused CSVs."""
        hist = _load_fused_frames(self.out_dir, days=days_history)
        if hist.empty:
            return pd.DataFrame(columns=[
                "time_run_utc","valid_time_utc","horizon_h","lat_bin","lon_bin",
                "no2_ugm3","no2_ppb","aqi","category","model_tag","n_history","r2_train"
            ])

        # Group by cell and build forecasts
        results: List[pd.DataFrame] = []
        for (la, lo), g in hist.groupby(["lat_bin","lon_bin"], dropna=True):
            fc_cell = self._forecast_one_cell(g, (la, lo), horizon_h=horizon_h, min_points_lr=min_points_lr)
            if fc_cell is not None and not fc_cell.empty:
                results.append(fc_cell)

        if not results:
            return pd.DataFrame()
        out = pd.concat(results, ignore_index=True)
        return out

    def save(self, df: pd.DataFrame, out_csv: str) -> str:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return out_csv

    # --------- internals ---------
    def _forecast_one_cell(self, g: pd.DataFrame, cell: Tuple[float,float], *, horizon_h: int, min_points_lr: int) -> Optional[pd.DataFrame]:
        la, lo = float(cell[0]), float(cell[1])
        g = g.sort_values("time_utc").copy()

        # hourly reindex to avoid gaps
        t0, t1 = g["time_utc"].min(), g["time_utc"].max()
        full_idx = pd.date_range(t0.floor("H"), t1.ceil("H"), freq="1H", tz=timezone.utc)
        g = g.set_index("time_utc").reindex(full_idx)
        g.index.name = "time_utc"
        # forward fill T/P; keep NO₂ as-is (do not fill)
        for c in ("temp_k","pressure_pa"):
            if c in g.columns:
                g[c] = g[c].ffill()

        # target
        y = g["no2_fused_ugm3"].astype(float)

        # build features
        df = pd.DataFrame(index=g.index)
        df["y"] = y
        df["lag1"] = y.shift(1)
        df["lag2"] = y.shift(2)
        df["lag24"] = y.shift(24)
        hours = df.index.tz_convert(None).hour
        df["sin_h"] = np.sin(2*np.pi*hours/24.0)
        df["cos_h"] = np.cos(2*np.pi*hours/24.0)
        if "temp_k" in g.columns:
            df["temp_k"] = g["temp_k"].astype(float)
        if "pressure_pa" in g.columns:
            df["pressure_pa"] = g["pressure_pa"].astype(float)

        # train split: use all available historic points with non-null features
        feat_cols = [c for c in ["lag1","lag2","lag24","sin_h","cos_h","temp_k","pressure_pa"] if c in df.columns]
        train = df.dropna(subset=["y"] + feat_cols)
        n_hist = len(train)

        time_run = datetime.now(timezone.utc)
        rows: List[Dict] = []

        if n_hist >= min_points_lr:
            X = train[feat_cols].values
            yv = train["y"].values
            model = Ridge(alpha=0.1)
            try:
                model.fit(X, yv)
                r2 = float(model.score(X, yv))
            except Exception:
                model = None
                r2 = None
        else:
            model = None
            r2 = None

        # Set starting state (last observed values)
        last_row = df.iloc[-1]
        last_y = last_row["y"] if math.isfinite(last_row.get("y", np.nan)) else None
        last_lag1 = last_row.get("lag1")
        last_lag2 = last_row.get("lag2")
        last_lag24 = last_row.get("lag24")
        last_T = last_row.get("temp_k", np.nan)
        last_P = last_row.get("pressure_pa", np.nan)

        # We will roll forward hour by hour; for lags, we update recursively
        cur_time = df.index[-1]
        lag_buffer = {
            "lag1": last_y if last_y is not None and math.isfinite(last_y) else last_lag1,
            "lag2": last_lag1,
            "lag24": last_lag24,
        }

        for h in range(1, horizon_h+1):
            vt = cur_time + timedelta(hours=h)
            sin_h = math.sin(2*math.pi*vt.hour/24.0)
            cos_h = math.cos(2*math.pi*vt.hour/24.0)
            T = last_T  # use last known; can be replaced by weather forecast if available
            P = last_P

            # Choose prediction strategy
            if model is not None and all(math.isfinite(lag_buffer.get(k, np.nan)) for k in ["lag1","lag2","lag24"]):
                feat_vec = [lag_buffer["lag1"], lag_buffer["lag2"], lag_buffer["lag24"], sin_h, cos_h]
                if "temp_k" in feat_cols:
                    feat_vec.append(T)
                if "pressure_pa" in feat_cols:
                    feat_vec.append(P)
                yhat = float(model.predict(np.array(feat_vec, dtype=float).reshape(1, -1))[0])
                tag = "ridge_lags"
            elif math.isfinite(lag_buffer.get("lag1", np.nan)) and math.isfinite(lag_buffer.get("lag2", np.nan)):
                # AR(1) approximate via Y_t ≈ Y_{t-1} + (Y_{t-1} - Y_{t-2})
                yhat = float(lag_buffer["lag1"] + (lag_buffer["lag1"] - lag_buffer["lag2"]))
                tag = "ar1_trend"
            elif math.isfinite(lag_buffer.get("lag1", np.nan)):
                yhat = float(lag_buffer["lag1"])  # persistence
                tag = "persistence"
            else:
                # No history → skip this cell
                break

            # Clamp negatives to zero (NO₂ cannot be negative)
            if not math.isfinite(yhat):
                break
            if yhat < 0:
                yhat = 0.0

            ppb = _ugm3_to_ppb(yhat, T, P)
            aqi, cat = _aqi_from_no2_ppb(ppb)

            rows.append({
                "time_run_utc": time_run.isoformat(),
                "valid_time_utc": vt.isoformat(),
                "horizon_h": h,
                "lat_bin": la,
                "lon_bin": lo,
                "no2_ugm3": yhat,
                "no2_ppb": ppb,
                "aqi": aqi,
                "category": cat,
                "model_tag": tag,
                "n_history": n_hist,
                "r2_train": r2,
            })

            # Update lags for next step
            lag_buffer["lag24"] = lag_buffer["lag24"]  # we don't have 25h history; keeping last known is acceptable for short horizons
            lag_buffer["lag2"] = lag_buffer["lag1"]
            lag_buffer["lag1"] = yhat

        if not rows:
            return None
        return pd.DataFrame(rows)


# -------------- CLI --------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NO₂ forecaster from fused CSV history (per-cell).")
    ap.add_argument("--out", required=True, help="Output CSV path for forecasts")
    ap.add_argument("--h", "--horizon", dest="h", type=int, default=10, help="Forecast horizon in hours (default 10)")
    ap.add_argument("--days", type=int, default=7, help="Days of fused history to use (default 7)")
    ap.add_argument("--min-points", type=int, default=48, help="Minimum training rows for Ridge; else fallback (default 48)")
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "./out"), help="Directory where fused_*.csv are stored")
    args = ap.parse_args()

    fore = NO2Forecaster(out_dir=args.out_dir)
    fc = fore.forecast(horizon_h=args.h, days_history=args.days, min_points_lr=args.min_points)
    if fc.empty:
        print("⚠️ No forecasts generated (insufficient history?)")
    else:
        path = fore.save(fc, args.out)
        print(f"✅ Saved forecast CSV: {path}  rows={len(fc):,}")
