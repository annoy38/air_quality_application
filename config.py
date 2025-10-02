# config.py
import os, sys, json, logging
from pathlib import Path
from typing import Optional
try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    print("Install python-dotenv:  pip install python-dotenv", file=sys.stderr)
    raise

def _mask(val: Optional[str], show: int = 4) -> str:
    if not val:
        return "<empty>"
    v = str(val)
    if len(v) <= show:
        return "*" * len(v)
    return v[:show] + "…" + "*" * (len(v) - show)

def load_env():
    here = Path(__file__).resolve()
    candidates = [
        here.parent / ".env",
        Path(find_dotenv()) if find_dotenv() else None,
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p and p.exists():
            load_dotenv(p.as_posix(), override=False)
            return p
    return None

def require_env(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        print(f"ERROR: Missing required env var: {key}", file=sys.stderr)
        sys.exit(1)
    return val

def get_config():
    dotenv_path = load_env()
    if dotenv_path:
        print(f"[config] Loaded .env from: {dotenv_path}", file=sys.stderr)
    else:
        print("[config] No .env found; relying on process environment.", file=sys.stderr)

    OPENAQ_API_KEY = require_env("OPENAQ_API_KEY").strip()
    BBOX_WSEN = os.getenv("BBOX_WSEN", "-168.0,14.0,-52.0,72.0").strip()
    LOOKBACK_H = int(os.getenv("OPENAQ_LOOKBACK_H", "24"))
    OUT_DIR = os.getenv("OUT_DIR", "./out").strip()

    EARTHDATA_USER = os.getenv("EARTHDATA_USER", "").strip()
    EARTHDATA_PASS = os.getenv("EARTHDATA_PASS", "").strip()
    TEMPO_POLLUTANT = os.getenv("TEMPO_POLLUTANT", "NO2").strip()

    # Weather & grid
    WEATHER_GRID_STEP = float(os.getenv("WEATHER_GRID_STEP", "0.25"))
    WEATHER_TIMEZONE = os.getenv("WEATHER_TIMEZONE", "UTC")
    WEATHER_HOURLY_VARS = os.getenv("WEATHER_HOURLY_VARS", "")
    GRID_STEP = float(os.getenv("GRID_STEP", str(WEATHER_GRID_STEP)))  # fusion grid

    # Logging / ops
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    METRICS_PORT = int(os.getenv("METRICS_PORT", "9308"))
    HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8088"))
    SCHED_ENABLE = os.getenv("SCHED_ENABLE", "0") == "1"
    SCHED_CRON = os.getenv("SCHED_CRON", "0 * * * *")  # top of hour
    CIRCUIT_FAIL_THRESHOLD = int(os.getenv("CIRCUIT_FAIL_THRESHOLD", "3"))
    CIRCUIT_RESET_MINUTES = int(os.getenv("CIRCUIT_RESET_MINUTES", "30"))

    # Masked diagnostics
    print("[config] OPENAQ_API_KEY:", _mask(OPENAQ_API_KEY))
    print("[config] BBOX_WSEN:", BBOX_WSEN)
    print("[config] LOOKBACK_H:", LOOKBACK_H)
    print("[config] OUT_DIR:", OUT_DIR)
    if EARTHDATA_USER:
        print("[config] EARTHDATA_USER:", EARTHDATA_USER)
    if EARTHDATA_PASS:
        print("[config] EARTHDATA_PASS:", _mask(EARTHDATA_PASS))
    print("[config] TEMPO_POLLUTANT:", TEMPO_POLLUTANT)
    print("[config] WEATHER_GRID_STEP:", WEATHER_GRID_STEP)
    print("[config] GRID_STEP:", GRID_STEP)
    print("[config] LOG_LEVEL:", LOG_LEVEL)
    print("[config] METRICS_PORT:", METRICS_PORT, "HEALTH_PORT:", HEALTH_PORT)
    print("[config] SCHED_ENABLE:", SCHED_ENABLE, "CRON:", SCHED_CRON)

    return {
        "OPENAQ_API_KEY": OPENAQ_API_KEY,
        "BBOX_WSEN": BBOX_WSEN,
        "LOOKBACK_H": LOOKBACK_H,
        "OUT_DIR": OUT_DIR,
        "EARTHDATA_USER": EARTHDATA_USER,
        "EARTHDATA_PASS": EARTHDATA_PASS,
        "TEMPO_POLLUTANT": TEMPO_POLLUTANT,
        "WEATHER_GRID_STEP": WEATHER_GRID_STEP,
        "WEATHER_TIMEZONE": WEATHER_TIMEZONE,
        "WEATHER_HOURLY_VARS": WEATHER_HOURLY_VARS,
        "GRID_STEP": GRID_STEP,
        "LOG_LEVEL": LOG_LEVEL,
        "METRICS_PORT": METRICS_PORT,
        "HEALTH_PORT": HEALTH_PORT,
        "SCHED_ENABLE": SCHED_ENABLE,
        "SCHED_CRON": SCHED_CRON,
        "CIRCUIT_FAIL_THRESHOLD": CIRCUIT_FAIL_THRESHOLD,
        "CIRCUIT_RESET_MINUTES": CIRCUIT_RESET_MINUTES,
    }

def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
