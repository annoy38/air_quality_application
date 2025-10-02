#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, tempfile, shutil
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import pandas as pd

__CFG_CACHE: Dict[str, Any] | None = None

def load_env_once():
    # Load .env only once to avoid noisy logs
    global __CFG_CACHE
    if __CFG_CACHE is None:
        load_dotenv()
        __CFG_CACHE = {}
    return True

def get_config() -> Dict[str, Any]:
    load_env_once()
    # Read env with sensible defaults
    cfg = {
        "OPENAQ_API_KEY": os.getenv("OPENAQ_API_KEY", "").strip(),
        "BBOX_WSEN": os.getenv("BBOX_WSEN", "-119.2,33.3,-117.3,34.7").strip(),
        "LOOKBACK_H": os.getenv("LOOKBACK_H", "24").strip(),
        "OUT_DIR": os.getenv("OUT_DIR", "./out").strip(),
        "DOWNLOAD_DIR": os.getenv("DOWNLOAD_DIR", "./out/downloads").strip(),
        "EARTHDATA_USER": os.getenv("EARTHDATA_USER", "").strip(),
        "EARTHDATA_PASS": os.getenv("EARTHDATA_PASS", "").strip(),
        "TEMPO_POLLUTANT": os.getenv("TEMPO_POLLUTANT", "NO2").strip(),
        "WEATHER_GRID_STEP": os.getenv("WEATHER_GRID_STEP", "0.5").strip(),
        "WEATHER_TIMEZONE": os.getenv("WEATHER_TIMEZONE", "UTC").strip(),
        "WEATHER_HOURLY_VARS": os.getenv(
            "WEATHER_HOURLY_VARS",
            "temperature_2m,pressure_msl,wind_speed_10m,cloud_cover,relative_humidity_2m"
        ).strip(),
        "OPENAQ_CONCURRENCY": os.getenv("OPENAQ_CONCURRENCY", "24").strip(),
        "WEATHER_CONCURRENCY": os.getenv("WEATHER_CONCURRENCY", "12").strip(),
        "OPENAQ_MAX_SENSORS": os.getenv("OPENAQ_MAX_SENSORS", "80").strip(),
        "HTTP_TIMEOUT": os.getenv("HTTP_TIMEOUT", "30").strip(),
        "SMTP_HOST": os.getenv("SMTP_HOST", "").strip(),
        "SMTP_PORT": os.getenv("SMTP_PORT", "587").strip(),
        "SMTP_USER": os.getenv("SMTP_USER", "").strip(),
        "SMTP_PASS": os.getenv("SMTP_PASS", "").strip(),
        "SMTP_FROM": os.getenv("SMTP_FROM", "").strip(),
        "SMTP_STARTTLS": os.getenv("SMTP_STARTTLS", "1").strip(),
    }
    return cfg

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def atomic_write_csv(df: pd.DataFrame, out_path: str):
    """Write CSV atomically to prevent partial files on crashes."""
    ensure_dir(Path(out_path).parent)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".tmp_csv_", dir=str(Path(out_path).parent))
    os.close(tmp_fd)
    try:
        df.to_csv(tmp_name, index=False)
        shutil.move(tmp_name, out_path)
    finally:
        try:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink()
        except Exception:
            pass
