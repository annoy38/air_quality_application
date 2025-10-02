#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, smtplib
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta
from email.policy import SMTP
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

# ---------- config helpers ----------
OUT_DIR = os.getenv("OUT_DIR", "./out")
GRID_STEP = float(os.getenv("WEATHER_GRID_STEP", "0.25"))

# Default “ideal” NO2 (µg/m³). WHO 24-h guideline is 25 µg/m³; override via env if you prefer.
DEFAULT_THRESHOLD_UGM3 = float(os.getenv("IDEAL_NO2_UGM3", "25"))

# SMTP settings (set these in env for real emails)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@example.com")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") == "1"

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# ---------- models ----------
class No2AlertRequest(BaseModel):
    email: EmailStr
    lat: float
    lon: float
    threshold_ugm3: Optional[float] = None
    horizon_h: int = 10  # look-ahead hours (default 10)


class No2AlertResponse(BaseModel):
    email: EmailStr
    lat_bin: float
    lon_bin: float
    threshold_ugm3: float
    now_value_ugm3: Optional[float]
    exceeded_now: bool
    max_future_ugm3: Optional[float]
    exceeded_future: bool
    horizon_h: int
    email_sent: bool
    detail: str


# ---------- utils ----------
def _snap_to_grid(val: float, step: float) -> float:
    return float(round(val / step) * step)

def _latlon_to_bins(lat: float, lon: float, step: float) -> Tuple[float, float]:
    return _snap_to_grid(lat, step), _snap_to_grid(lon, step)

def _latest_fused_path(out_dir: str) -> Optional[str]:
    paths = sorted(Path(out_dir).glob("fused_*.csv"))
    return paths[-1].as_posix() if paths else None

def _load_fused_df(out_dir: str) -> pd.DataFrame:
    path = _latest_fused_path(out_dir)
    if not path or not Path(path).exists():
        raise FileNotFoundError("No fused_*.csv found in OUT_DIR")
    df = pd.read_csv(path)
    # normalize
    df.columns = [c.replace(" ", "_").replace(".", "_").lower() for c in df.columns]
    if "time_hour" not in df.columns:
        # accept 'time' as fallback
        tcol = "time" if "time" in df.columns else None
        if not tcol:
            raise ValueError("Fused CSV missing 'time_hour'/'time' column")
        df["time_hour"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    else:
        df["time_hour"] = pd.to_datetime(df["time_hour"], utc=True, errors="coerce")

    # pick fused value: ground first, else corrected TEMPO
    if "no2_ground_ug_m3" not in df.columns:
        df["no2_ground_ug_m3"] = np.nan
    if "no2_trop_col_corrected_ug_m3" not in df.columns:
        df["no2_trop_col_corrected_ug_m3"] = np.nan
    df["no2_fused_ugm3"] = df["no2_ground_ug_m3"].where(
        df["no2_ground_ug_m3"].notna(), df["no2_trop_col_corrected_ug_m3"]
    )

    # ensure bins
    for col in ("lat_bin", "lon_bin"):
        if col not in df.columns:
            raise ValueError(f"Fused CSV missing '{col}'")
    return df

def _nearest_cell(df: pd.DataFrame, lat_bin: float, lon_bin: float) -> Tuple[float, float]:
    # If exact bin exists, use it; else find nearest bin present in df
    mask = (np.isclose(df["lat_bin"], lat_bin)) & (np.isclose(df["lon_bin"], lon_bin))
    if mask.any():
        return lat_bin, lon_bin
    uniq = df[["lat_bin","lon_bin"]].drop_duplicates()
    d2 = (uniq["lat_bin"] - lat_bin)**2 + (uniq["lon_bin"] - lon_bin)**2
    i = int(np.argmin(d2.values))
    nb = float(uniq.iloc[i]["lat_bin"])
    mb = float(uniq.iloc[i]["lon_bin"])
    return nb, mb

def _now_utc_floor_hour() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

def _load_forecast_df(out_dir: str) -> Optional[pd.DataFrame]:
    path = Path(out_dir) / "no2_forecast.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.replace(" ", "_").replace(".", "_").lower() for c in df.columns]
    if not {"valid_time_utc","lat_bin","lon_bin","no2_ugm3"}.issubset(set(df.columns)):
        return None
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True, errors="coerce")
    return df

def _send_email(to_addr: str, subject: str, html_body: str):
    msg = EmailMessage()
    msg["From"] = SMTP["from"]
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content("HTML-capable email client required.")
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(SMTP["host"], SMTP["port"], timeout=30) as s:
        if SMTP["starttls"]:
            s.starttls()
        if SMTP["user"]:
            s.login(SMTP["user"], SMTP["pwd"])
        s.send_message(msg)



def no2_alert(req: No2AlertRequest):
    threshold = req.threshold_ugm3 if (req.threshold_ugm3 and req.threshold_ugm3 > 0) else DEFAULT_THRESHOLD_UGM3
    horizon = int(max(1, min(req.horizon_h, 48)))  # safety clamp 1..48
    lat_bin = _snap_to_grid(req.lat, GRID_STEP)
    lon_bin = _snap_to_grid(req.lon, GRID_STEP)

    # Load fused/current
    try:
        fused = _load_fused_df(OUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to read fused CSVs: {e}")

    # ensure valid cell (maybe nearest if empty)
    lat_bin, lon_bin = _nearest_cell(fused, lat_bin, lon_bin)

    # current (latest hour <= now)
    now_floor = _now_utc_floor_hour()
    latest_hour = fused["time_hour"].max()
    # if fused newest is in the future (unlikely), clamp down; else use newest we have
    cur_hour = min(now_floor, latest_hour) if pd.notna(latest_hour) else now_floor

    cur_row = fused[(fused["lat_bin"]==lat_bin)&(fused["lon_bin"]==lon_bin)&(fused["time_hour"]==cur_hour)]
    now_val = float(cur_row["no2_fused_ugm3"].iloc[0]) if not cur_row.empty and pd.notna(cur_row["no2_fused_ugm3"].iloc[0]) else None
    exceeded_now = (now_val is not None) and (now_val > threshold)

    # Forecast window
    fc = _load_forecast_df(OUT_DIR)
    if fc is not None:
        t0 = cur_hour + timedelta(hours=1)  # looking ahead
        t1 = cur_hour + timedelta(hours=horizon)
        fc_sub = fc[(fc["lat_bin"].round(6)==round(lat_bin,6)) &
                    (fc["lon_bin"].round(6)==round(lon_bin,6)) &
                    (fc["valid_time_utc"] >= t0) & (fc["valid_time_utc"] <= t1)]
        max_future = float(fc_sub["no2_ugm3"].max()) if not fc_sub.empty else None
    else:
        # fallback: persistence (use current for next N hours)
        max_future = now_val

    exceeded_future = (max_future is not None) and (max_future > threshold)

    # Build email if needed
    should_email = exceeded_now or exceeded_future
    email_sent = False
    detail = "OK: threshold not exceeded."
    if should_email:
        subj = "NO₂ Alert — Air quality threshold exceeded"
        tstamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        body_txt = (
            f"NO2 alert for ({req.lat:.4f}, {req.lon:.4f}) at {tstamp}\n"
            f"Grid cell: ({lat_bin:.2f}, {lon_bin:.2f})\n"
            f"Threshold: {threshold:.1f} µg/m³\n"
            f"Current:   {('n/a' if now_val is None else f'{now_val:.1f}')} µg/m³\n"
            f"Max next {horizon}h: {('n/a' if max_future is None else f'{max_future:.1f}')} µg/m³\n"
            f"Exceeded now: {exceeded_now} | Exceeded future: {exceeded_future}\n"
        )
        body_html = f"""
        <h3>NO₂ alert</h3>
        <p><b>Location:</b> ({req.lat:.4f}, {req.lon:.4f}) → grid ({lat_bin:.2f}, {lon_bin:.2f})</p>
        <p><b>Threshold:</b> {threshold:.1f} µg/m³<br/>
           <b>Current:</b> {('n/a' if now_val is None else f'{now_val:.1f}')} µg/m³<br/>
           <b>Max next {horizon}h:</b> {('n/a' if max_future is None else f'{max_future:.1f}')} µg/m³</p>
        <p><b>Exceeded now:</b> {exceeded_now} &nbsp; | &nbsp; <b>Exceeded future:</b> {exceeded_future}</p>
        <p><small>Sent at {tstamp}</small></p>
        """
        email_sent = _send_email(req.email, subj, body_html, body_txt)
        detail = "Email alert sent." if email_sent else "Alert condition met, but email not sent (SMTP not configured)."

    return No2AlertResponse(
        email=req.email,
        lat_bin=lat_bin,
        lon_bin=lon_bin,
        threshold_ugm3=threshold,
        now_value_ugm3=now_val,
        exceeded_now=exceeded_now,
        max_future_ugm3=max_future,
        exceeded_future=exceeded_future,
        horizon_h=horizon,
        email_sent=email_sent,
        detail=detail,
    )
# Make both names available to importers
alerts_router = router
__all__ = ["alerts_router", "router"]
