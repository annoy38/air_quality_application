#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from net_utils import SESSION, HTTP_TIMEOUT

BASE = "https://api.openaq.org"

# ---------- helpers ----------
def _coerce_found(found_raw) -> int:
    if found_raw is None: return 0
    if isinstance(found_raw, (int, float)): return int(found_raw)
    s = str(found_raw); digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0

def _safe(v, *keys, default=None):
    cur = v
    for k in keys:
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur

def _paged_get(url: str, api_key: str, params: Dict[str, Any], per_page: int = 1000) -> List[Dict[str, Any]]:
    results, page = [], 1
    headers = {"X-API-Key": api_key} if api_key else {}
    while True:
        qp = dict(params); qp.update({"limit": per_page, "page": page})
        r = SESSION.get(url, headers=headers, params=qp, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        chunk = data.get("results", [])
        results.extend(chunk)
        found = _coerce_found((data.get("meta") or {}).get("found", len(chunk)))
        pages = max(1, math.ceil(found / per_page)) if per_page else 1
        if page >= pages or not chunk: break
        page += 1
    return results

def _get_parameter_id(api_key: str, name: str) -> Optional[int]:
    items = _paged_get(f"{BASE}/v3/parameters", api_key, {"name": name}, per_page=1000)
    if not items:
        items = _paged_get(f"{BASE}/v3/parameters", api_key, {}, per_page=1000)
    for p in items:
        if str(p.get("name", "")).lower().strip() == name.lower():
            return p.get("id")
    return None

# ---------- new: fast one-day fetch ----------
def fetch_openaq_no2_day_bbox_fast(
    api_key: str,
    bbox_ws_en: str,
    day_yyyy_mm_dd: str,
    per_page: int = 1000,
    max_workers: int = 24,
    max_sensors: int = 80,
) -> pd.DataFrame:
    """
    Fast path for a single UTC day. Tries parameter-measurements first;
    falls back to per-sensor hours if needed.
    """
    pid = _get_parameter_id(api_key, "no2")
    day0 = datetime.fromisoformat(day_yyyy_mm_dd).replace(tzinfo=timezone.utc)
    day1 = day0 + timedelta(days=1)
    df_all = pd.DataFrame()

    if pid:
        try:
            url = f"{BASE}/v3/parameters/{pid}/measurements"
            results = _paged_get(url, api_key, {
                "bbox": bbox_ws_en,
                "date_from": day0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date_to":   day1.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, per_page=per_page)
            rows = []
            for m in (results or []):
                coords = m.get("coordinates") or {}
                rows.append({
                    "utc": _safe(m, "datetime", "utc"),
                    "local": _safe(m, "datetime", "local"),
                    "value": m.get("value"),
                    "unit": _safe(m, "parameter", "units"),
                    "parameter": "no2",
                    "sensorsId": m.get("sensorsId"),
                    "locationsId": m.get("locationsId"),
                    "lat": coords.get("latitude"),
                    "lon": coords.get("longitude"),
                    "provider": _safe(m, "provider", "name"),
                    "owner": _safe(m, "owner", "name"),
                    "flag": (m.get("flags") or [None])[0] if m.get("flags") else None,
                })
            df_all = pd.DataFrame(rows)
        except Exception:
            df_all = pd.DataFrame()

    if df_all.empty or len(df_all) < 200:
        sens = select_active_no2_sensor_ids(api_key, bbox_ws_en, max_n=max_sensors, per_page=1000)
        if not sens:
            return df_all
        date_from = day0.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_to   = day1.strftime("%Y-%m-%dT%H:%M:%SZ")

        def pull_sid(sid):
            try:
                url = f"{BASE}/v3/sensors/{sid}/hours"
                res = _paged_get(url, api_key, {"date_from": date_from, "date_to": date_to}, per_page=per_page)
            except Exception:
                res = []
            rows = []
            for m in (res or []):
                coords = _safe(m, "coordinates") or {}
                rows.append({
                    "utc": _safe(m, "period", "datetimeTo", "utc") or _safe(m, "datetime", "utc"),
                    "local": _safe(m, "period", "datetimeTo", "local") or _safe(m, "datetime", "local"),
                    "value": m.get("value"),
                    "unit": _safe(m, "parameter", "units"),
                    "parameter": "no2",
                    "sensorsId": m.get("sensorsId") or sid,
                    "locationsId": m.get("locationsId"),
                    "lat": coords.get("latitude"),
                    "lon": coords.get("longitude"),
                    "provider": _safe(m, "provider", "name"),
                    "owner": _safe(m, "owner", "name"),
                    "flag": (m.get("flags") or [None])[0] if m.get("flags") else None,
                })
            return rows

        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(pull_sid, sid) for sid in sens]
            for fut in as_completed(futs):
                try:
                    rows.extend(fut.result())
                except Exception:
                    pass
        df_all = pd.DataFrame(rows)

    if not df_all.empty and "utc" in df_all.columns:
        df_all = df_all[df_all["utc"].notna()]
        df_all = df_all.sort_values(["sensorsId","utc"], kind="stable")\
                       .drop_duplicates(subset=["sensorsId","utc"], keep="last")
    return df_all

# ---------- existing utilities kept (cleaned) ----------
def select_active_no2_sensor_ids(api_key: str, bbox_ws_en: str, max_n: int = 60, per_page: int = 1000) -> List[int]:
    chosen: List[int] = []
    pid = None
    try:
        pid = _get_parameter_id(api_key, "no2")
    except Exception:
        pass

    if pid:
        try:
            url = f"{BASE}/v3/parameters/{pid}/latest"
            latest = _paged_get(url, api_key, {"bbox": bbox_ws_en}, per_page=per_page)
            best_by_sensor: Dict[int, Dict[str, Any]] = {}
            for r in latest or []:
                sid = r.get("sensorsId")
                if sid is None: continue
                ts = _safe(r, "datetime", "utc")
                if ts is None: continue
                sid_int = int(sid)
                prev_ts = _safe(best_by_sensor.get(sid_int, {}), "datetime", "utc")
                if (prev_ts is None) or (str(ts) > str(prev_ts)):
                    best_by_sensor[sid_int] = r
            chosen = [int(sid) for sid, _ in sorted(
                best_by_sensor.items(),
                key=lambda kv: str(_safe(kv[1], "datetime", "utc") or ""),
                reverse=True
            )[:max_n]]
        except Exception:
            pass

    if len(chosen) < max_n:
        try:
            locs = _paged_get(f"{BASE}/v3/locations", api_key, {"bbox": bbox_ws_en}, per_page=per_page)
        except Exception:
            locs = []
        for loc in (locs or []):
            lid = loc.get("locationsId") or loc.get("id") or loc.get("locationId")
            if lid is None: continue
            try:
                sensors = _paged_get(f"{BASE}/v3/locations/{lid}/sensors", api_key, {}, per_page=per_page)
            except Exception:
                sensors = []
            for s in (sensors or []):
                p = s.get("parameter")
                pname = (p.get("name") if isinstance(p, dict) else p) if p is not None else None
                if isinstance(pname, str) and pname.lower().strip() == "no2":
                    sid = s.get("sensorsId") or s.get("id")
                    try: sid_int = int(sid) if sid is not None else None
                    except Exception: sid_int = None
                    if sid_int is not None and sid_int not in chosen:
                        chosen.append(sid_int)
                        if len(chosen) >= max_n: break
            if len(chosen) >= max_n: break
    return chosen
