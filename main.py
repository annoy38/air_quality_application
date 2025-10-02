#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, logging
from datetime import datetime, timezone
import pandas as pd
from typing import Tuple

from config import get_config, configure_logging
from openaq_api import (
    fetch_openaq_no2_bbox_via_sensors,
    fetch_openaq_no2_latest_bbox,
    select_active_no2_sensor_ids,
    coverage_3h_report,
)
from tempo_api import run_once_with_temporal as tempo_run_temporal, run_once as tempo_run_legacy
from weather_api import fetch_weather_to_csv
from fuse_validate_merge import FuseValidateMerge

# Ops
from prometheus_client import Counter, Gauge, start_http_server
from apscheduler.schedulers.background import BackgroundScheduler

log = logging.getLogger("orchestrator")

# Prometheus metrics
RUNS = Counter("aq_backend_runs_total", "Total pipeline runs")
FAILS = Counter("aq_backend_failures_total", "Pipeline failures")
LAST_SUCCESS_TS = Gauge("aq_backend_last_success_unixtime", "Last success time")
LAST_RUN_TS = Gauge("aq_backend_last_run_unixtime", "Last run time")
CIRCUIT_OPEN = Gauge("aq_backend_circuit_open", "Circuit breaker state (1=open)")

class CircuitBreaker:
    def __init__(self, fail_threshold: int, reset_minutes: int):
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_minutes*60
        self.fail_count = 0
        self.opened_at = None

    def record_success(self):
        self.fail_count = 0
        self.opened_at = None
        CIRCUIT_OPEN.set(0)

    def record_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.fail_threshold and self.opened_at is None:
            self.opened_at = time.time()
            CIRCUIT_OPEN.set(1)

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if (time.time() - self.opened_at) >= self.reset_seconds:
            # half-open
            self.fail_count = 0
            self.opened_at = None
            CIRCUIT_OPEN.set(0)
            return True
        return False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

class Main:
    def __init__(self):
        cfg = get_config()
        configure_logging(cfg["LOG_LEVEL"])
        self.cfg = cfg
        self.openaq_key = cfg["OPENAQ_API_KEY"]
        self.bbox_str = cfg["BBOX_WSEN"]
        self.out_dir = cfg["OUT_DIR"]
        self.grid_step = cfg["GRID_STEP"]

        self.weather_grid_step = cfg["WEATHER_GRID_STEP"]
        self.weather_timezone = cfg["WEATHER_TIMEZONE"]
        self.weather_vars = [v.strip() for v in cfg["WEATHER_HOURLY_VARS"].split(",") if v.strip()] if cfg["WEATHER_HOURLY_VARS"] else None

        ensure_dir(self.out_dir)
        self.circuit = CircuitBreaker(cfg["CIRCUIT_FAIL_THRESHOLD"], cfg["CIRCUIT_RESET_MINUTES"])

        # Start metrics/health server
        start_http_server(cfg["METRICS_PORT"])
        log.info(f"Prometheus metrics server started on :{cfg['METRICS_PORT']}")

    # ---------- TEMPO (explicit hour window) ----------
    def run_tempo_exact(self, start_utc: datetime, end_utc: datetime) -> Tuple[str, pd.DataFrame]:
        os.environ["BBOX_WSEN"] = self.bbox_str
        os.environ["OUT_DIR"] = self.out_dir
        os.environ["START_ISO"] = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["END_ISO"]   = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        log.info(f"[TEMPO] {os.environ['START_ISO']} → {os.environ['END_ISO']} UTC, bbox {self.bbox_str}")
        try:
            csv_path, df = tempo_run_temporal(os.environ["START_ISO"], os.environ["END_ISO"])
        except Exception:
            # fallback legacy (env-driven)
            csv_path, df = tempo_run_legacy()

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df[(df["time"] >= start_utc) & (df["time"] < end_utc)].copy()

        out_csv = os.path.join(self.out_dir, f"tempo_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.csv")
        df.to_csv(out_csv, index=False)
        log.info(f"[TEMPO] CSV {out_csv} rows={len(df):,}")
        return out_csv, df

    # ---------- OpenAQ ----------
    def run_openaq_exact(self, start_utc: datetime, end_utc: datetime) -> Tuple[str, pd.DataFrame]:
        wsen = self.bbox_str
        date_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_to   = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        log.info(f"[OpenAQ] selecting sensors bbox={wsen}")
        try:
            sensor_ids = select_active_no2_sensor_ids(self.openaq_key, wsen, max_n=60, per_page=1000)
        except Exception as e:
            log.warning(f"[OpenAQ] sensor select failed: {e}")
            sensor_ids = []

        log.info(f"[OpenAQ] got {len(sensor_ids)} sensors; pulling {date_from}→{date_to}")
        df = pd.DataFrame()
        try:
            df = fetch_openaq_no2_bbox_via_sensors(
                self.openaq_key, wsen, date_from, date_to,
                per_page=200, hourly=True, restrict_sensor_ids=sensor_ids
            )
        except Exception as e:
            log.warning(f"[OpenAQ] per-sensor flow failed: {e}")

        if df.empty:
            log.info("[OpenAQ] falling back to parameter 'latest' and post-filter")
            try:
                df = fetch_openaq_no2_latest_bbox(self.openaq_key, wsen, per_page=1000)
                if "utc" in df.columns:
                    df["utc"] = pd.to_datetime(df["utc"], utc=True, errors="coerce")
                    df = df[(df["utc"] >= start_utc) & (df["utc"] < end_utc)].copy()
            except Exception as e:
                log.error(f"[OpenAQ] latest fallback failed: {e}")
                df = pd.DataFrame()

        out_csv = os.path.join(self.out_dir, f"openaq_no2_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.csv")
        df.to_csv(out_csv, index=False)
        log.info(f"[OpenAQ] CSV {out_csv} rows={len(df):,}")

        try:
            cov = coverage_3h_report(df)
            cov_csv = os.path.join(self.out_dir, f"openaq_coverage_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.csv")
            cov.to_csv(cov_csv, index=False)
            log.info(f"[OpenAQ] Coverage CSV: {cov_csv}")
        except Exception:
            pass

        return out_csv, df

    # ---------- Weather ----------
    def run_weather_exact(self, start_utc: datetime, end_utc: datetime) -> Tuple[str, pd.DataFrame]:
        w, s, e, n = [float(x.strip()) for x in self.bbox_str.split(",")]
        log.info(f"[Weather] Fetching bbox={self.bbox_str} grid_step={self.weather_grid_step}")
        broad_csv_tag = f"{datetime.now(timezone.utc):%Y%m%dT%H%MZ}"
        broad_csv = os.path.join(self.out_dir, f"weather_{broad_csv_tag}.csv")

        # pull 48h back to be safe; no future hours needed here
        _, df_all = fetch_weather_to_csv(
            bbox=(w, s, e, n),
            grid_step=self.weather_grid_step,
            past_hours=48,
            past_days=2,
            forecast_hours=0,
            timezone_str="UTC",
            hourly_vars=self.weather_vars,
            out_csv=broad_csv,
            regrid_step=self.grid_step,  # align early to fusion grid
        )

        time_col = "time" if "time" in df_all.columns else ("datetime" if "datetime" in df_all.columns else None)
        if time_col:
            df_all[time_col] = pd.to_datetime(df_all[time_col], utc=True, errors="coerce")
            mask = (df_all[time_col] >= start_utc) & (df_all[time_col] < end_utc)
            df = df_all.loc[mask].copy()
        else:
            df = pd.DataFrame()

        out_csv = os.path.join(self.out_dir, f"weather_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.csv")
        df.to_csv(out_csv, index=False)
        log.info(f"[Weather] CSV {out_csv} rows={len(df):,}")
        return out_csv, df

    # ---------- Orchestrate one run ----------
    def run_once(self, start_utc: datetime, end_utc: datetime):
        RUNS.inc(); LAST_RUN_TS.set(time.time())
        if not self.circuit.allow():
            log.warning("Circuit breaker OPEN — skipping run.")
            return

        try:
            tempo_csv, _ = self.run_tempo_exact(start_utc, end_utc)
            openaq_csv, _ = self.run_openaq_exact(start_utc, end_utc)
            weather_csv, _ = self.run_weather_exact(start_utc, end_utc)

            fused_out = os.path.join(self.out_dir, f"fused_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.csv")
            metrics_out = os.path.join(self.out_dir, f"metrics_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.csv")
            map_out = os.path.join(self.out_dir, f"map_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.html")
            multimap_out = os.path.join(self.out_dir, f"map_multi_{start_utc:%Y%m%dT%H%MZ}_{end_utc:%H%MZ}.html")

            runner = FuseValidateMerge(
                tempo_csv=tempo_csv, openaq_csv=openaq_csv, weather_csv=weather_csv,
                grid_step=self.grid_step, out_fused_csv=fused_out,
                out_metrics_csv=metrics_out, out_map_html=map_out,
                out_multihour_map_html=multimap_out, multihour_panels=6
            )
            runner.run()
            self.circuit.record_success()
            LAST_SUCCESS_TS.set(time.time())
            log.info("Run SUCCESS")
        except Exception as e:
            FAILS.inc()
            self.circuit.record_failure()
            log.exception(f"Run FAILED: {e}")

def _parse_hour_window(now_utc: datetime) -> Tuple[datetime, datetime]:
    start_utc = now_utc.replace(minute=0, second=0, microsecond=0)
    end_utc = start_utc + timedelta(hours=4)
    return start_utc, end_utc

# ---------- Health server (very simple) ----------
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from datetime import timedelta

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/healthz":
            self.send_response(404); self.end_headers(); return
        try:
            last = LAST_SUCCESS_TS._value.get()
            age = time.time() - last if last else 1e9
            status = 200 if age < 3600*3 else 503  # healthy if success in last 3h
            body = f'{{"ok":{str(status==200).lower()},"last_success_age_s":{int(age)}}}'
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        except Exception:
            self.send_response(500); self.end_headers()

def start_health_server(port: int):
    def _serve():
        httpd = HTTPServer(("0.0.0.0", port), HealthHandler)
        log.info(f"Health server on :{port}")
        httpd.serve_forever()
    Thread(target=_serve, daemon=True).start()

# ---------- Entry ----------
if __name__ == "__main__":
    from datetime import timedelta
    cfg = get_config()
    configure_logging(cfg["LOG_LEVEL"])
    start_health_server(cfg["HEALTH_PORT"])
    m = Main()

    # Immediate one run for the last full hour
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_utc = now - timedelta(hours=4)
    end_utc = now
    m.run_once(start_utc, end_utc)

    # Scheduler
    if cfg["SCHED_ENABLE"]:
        sched = BackgroundScheduler(timezone="UTC")
        # run at top of every hour for the PREVIOUS hour window
        def job():
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            start_utc = now - timedelta(hours=4); end_utc = now
            m.run_once(start_utc, end_utc)
        sched.add_job(job, "cron", **{"minute": 0})
        sched.start()
        log.info("Scheduler started (cron: top of hour).")
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt:
            sched.shutdown()
