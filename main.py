#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import traceback

from config import get_config, ensure_dir, atomic_write_csv
from openaq_api import fetch_openaq_no2_day_bbox_fast
try:
    from tempo_api import run_once_with_temporal as tempo_run_temporal
    HAS_TEMPO_TEMPORAL = True
except Exception:
    tempo_run_temporal = None
    HAS_TEMPO_TEMPORAL = False

from weather_api import fetch_weather_to_csv
from fuse_validate_merge import FuseValidateMerge

def _log(msg: str):
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}", flush=True)

def _day_bounds_utc(day_str: str):
    d0 = datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc)
    start = d0.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)
    return start, end

class Main:
    def __init__(self, day_str: str):
        cfg = get_config()

        self.openaq_key = cfg["OPENAQ_API_KEY"]
        self.bbox_str   = cfg["BBOX_WSEN"]
        self.out_dir    = Path(cfg["OUT_DIR"])
        self.download_dir = Path(cfg["DOWNLOAD_DIR"])
        ensure_dir(self.out_dir); ensure_dir(self.download_dir)

        self.weather_grid_step = float(cfg.get("WEATHER_GRID_STEP", 0.5))
        self.weather_vars = cfg.get(
            "WEATHER_HOURLY_VARS",
            "temperature_2m,pressure_msl,wind_speed_10m,cloud_cover,relative_humidity_2m"
        )

        self.OPENAQ_CONCURRENCY = int(cfg.get("OPENAQ_CONCURRENCY", 24))
        self.OPENAQ_MAX_SENSORS = int(cfg.get("OPENAQ_MAX_SENSORS", 80))

        self.start_utc, self.end_utc = _day_bounds_utc(day_str)
        self.day_str = day_str

    def run_tempo_day(self):
        _log(f"[TEMPO] window {self.start_utc.isoformat()} → {self.end_utc.isoformat()} bbox={self.bbox_str}")
        if not HAS_TEMPO_TEMPORAL:
            _log("[TEMPO] Skipped (temporal runner not available).")
            return None, pd.DataFrame()

        os.environ["BBOX_WSEN"] = self.bbox_str
        os.environ["OUT_DIR"]   = self.out_dir.as_posix()
        os.environ["START_ISO"] = self.start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["END_ISO"]   = self.end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            csv_path, df = tempo_run_temporal(os.environ["START_ISO"], os.environ["END_ISO"])
            if isinstance(df, pd.DataFrame) and "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df[(df["time"] >= self.start_utc) & (df["time"] < self.end_utc)].copy()
                if csv_path:
                    atomic_write_csv(df, csv_path)
            rows = 0 if df is None or df.empty else len(df)
            _log(f"[TEMPO] Done. rows={rows} csv={csv_path or '(skipped)'}")
            return csv_path, df
        except Exception as e:
            _log(f"[TEMPO] Fetch failed (skipping): {e.__class__.__name__}: {e}")
            return None, pd.DataFrame()

    def run_openaq_day(self):
        _log(f"[OpenAQ] Fetching NO₂ for {self.day_str} bbox={self.bbox_str} …")
        try:
            out_csv = self.out_dir / f"openaq_no2_{self.start_utc:%Y-%m-%d}.csv"
            df = fetch_openaq_no2_day_bbox_fast(
                api_key=self.openaq_key,
                bbox_ws_en=self.bbox_str,
                day_yyyy_mm_dd=self.start_utc.strftime("%Y-%m-%d"),
                per_page=500,
                max_workers=self.OPENAQ_CONCURRENCY,
                max_sensors=self.OPENAQ_MAX_SENSORS,
            )
            atomic_write_csv(df, out_csv.as_posix())
            _log(f"[OpenAQ] Done. rows={0 if df is None or df.empty else len(df)} csv={out_csv}")
            return out_csv.as_posix(), df
        except Exception as e:
            _log(f"[OpenAQ] ERROR: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            # Keep going; return empty
            out_csv = self.out_dir / f"openaq_no2_{self.start_utc:%Y-%m-%d}.csv"
            atomic_write_csv(pd.DataFrame(), out_csv.as_posix())
            return out_csv.as_posix(), pd.DataFrame()

    def run_weather_day(self):
        _log(f"[Weather] Fetching Open-Meteo (grid={self.weather_grid_step}°) for {self.day_str} …")
        w, s, e, n = [float(x.strip()) for x in self.bbox_str.split(",")]
        out_csv = self.out_dir / f"weather_{self.start_utc:%Y%m%d}.csv"
        try:
            _, df = fetch_weather_to_csv(
                bbox=(w, s, e, n),
                grid_step=self.weather_grid_step,
                past_hours=48,
                past_days=2,
                forecast_hours=0,
                timezone_str="UTC",
                hourly_vars=[v.strip() for v in self.weather_vars.split(",") if v.strip()],
                out_csv=out_csv.as_posix(),
            )
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df[(df["time"] >= self.start_utc) & (df["time"] < self.end_utc)].copy()
                atomic_write_csv(df, out_csv.as_posix())
            _log(f"[Weather] Done. rows={0 if df is None or df.empty else len(df)} csv={out_csv}")
            return out_csv.as_posix(), df
        except Exception as e:
            _log(f"[Weather] ERROR: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            atomic_write_csv(pd.DataFrame(), out_csv.as_posix())
            return out_csv.as_posix(), pd.DataFrame()

    def run(self):
        _log(f"=== RUN DAY {self.day_str} ===")
        tempo_csv, _   = self.run_tempo_day()
        openaq_csv, _  = self.run_openaq_day()
        weather_csv, _ = self.run_weather_day()

        fused_out   = self.out_dir / f"fused_{self.start_utc:%Y%m%d}.csv"
        metrics_out = self.out_dir / f"fused_metrics_{self.start_utc:%Y%m%d}.csv"
        map_out     = self.out_dir / f"fused_map_{self.start_utc:%Y%m%d}.html"
        multimap    = self.out_dir / f"fused_multimap_{self.start_utc:%Y%m%d}.html"

        _log("[Fuse] Starting fusion …")
        try:
            runner = FuseValidateMerge(
                tempo_csv=tempo_csv or "",
                openaq_csv=openaq_csv,
                weather_csv=weather_csv,
                grid_step=self.weather_grid_step,
                out_fused_csv=fused_out.as_posix(),
                out_metrics_csv=metrics_out.as_posix(),
                out_map_html=map_out.as_posix(),
                out_multihour_map_html=multimap.as_posix(),
                multihour_panels=8,
            )
            runner.run()
            _log(f"[Fuse] Done. fused={fused_out} metrics={metrics_out} map={map_out}")
        except Exception as e:
            _log(f"[Fuse] ERROR: {e.__class__.__name__}: {e}")
            traceback.print_exc()
        _log("=== RUN COMPLETE ===")

if __name__ == "__main__":
    # Hard-code your UTC date here:
    DAY = "2025-10-01"
    Main(DAY).run()
