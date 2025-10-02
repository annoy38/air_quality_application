#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, time, requests, pandas as pd
from typing import List, Dict, Any, Tuple, Optional

BASE = "https://api.openaq.org"
TIMEOUT = 60

# ---------- HTTP with exponential backoff ----------
def _get(url: str, api_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"X-API-Key": api_key}
    backoff = 1.0
    for attempt in range(7):
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
            return r.json()
        ra = r.headers.get("Retry-After")
        wait_s = float(ra) if (ra and ra.isdigit()) else backoff
        time.sleep(wait_s)
        backoff = min(backoff * 2.0, 60.0)
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _coerce_found(found_raw) -> int:
    if found_raw is None: return 0
    if isinstance(found_raw, (int, float)): return int(found_raw)
    s = str(found_raw); digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0

def _paged_get(url: str, api_key: str, params: Dict[str, Any], per_page: int = 1000) -> List[Dict[str, Any]]:
    results, page = [], 1
    while True:
        qp = dict(params); qp.update({"limit": per_page, "page": page})
        data = _get(url, api_key, qp)
        chunk = data.get("results", [])
        results.extend(chunk)
        found = _coerce_found((data.get("meta") or {}).get("found", len(chunk)))
        pages = max(1, math.ceil(found / per_page)) if per_page else 1
        if page >= pages or not chunk: break
        page += 1
    return results

def _safe(v, *keys, default=None):
    cur = v
    for k in keys:
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur

def _get_parameter_id(api_key: str, name: str) -> Optional[int]:
    items = _paged_get(f"{BASE}/v3/parameters", api_key, {"name": name}, per_page=1000)
    if not items:
        items = _paged_get(f"{BASE}/v3/parameters", api_key, {}, per_page=1000)
    for p in items:
        if str(p.get("name", "")).lower().strip() == name.lower():
            return p.get("id")
    return None

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

def fetch_openaq_no2_bbox_via_sensors(
    api_key: str,
    bbox_ws_en: str,
    date_from: str,
    date_to: str,
    per_page: int = 1000,
    hourly: bool = True,
    sleep_between: float = 0.0,
    restrict_sensor_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    try:
        locs = _paged_get(f"{BASE}/v3/locations", api_key, {"bbox": bbox_ws_en}, per_page=per_page)
    except Exception:
        locs = []
    if not locs:
        return pd.DataFrame()

    loc_meta: Dict[int, Dict[str, Any]] = {}
    for loc in locs:
        lid_raw = loc.get("locationsId") or loc.get("id") or loc.get("locationId")
        try: lid = int(lid_raw) if lid_raw is not None else None
        except Exception: lid = None
        if lid is None: continue
        coords = loc.get("coordinates") or {}
        loc_meta[lid] = {
            "lat": coords.get("latitude"),
            "lon": coords.get("longitude"),
            "provider": _safe(loc, "provider", "name"),
            "owner": _safe(loc, "owner", "name"),
        }

    targets: List[Tuple[int, int]] = []
    want: set[int] = set(int(x) for x in (restrict_sensor_ids or []))

    for loc in locs:
        lid_raw = loc.get("locationsId") or loc.get("id") or loc.get("locationId")
        try: lid = int(lid_raw) if lid_raw is not None else None
        except Exception: lid = None
        if lid is None: continue

        try:
            sensors = _paged_get(f"{BASE}/v3/locations/{lid}/sensors", api_key, {}, per_page=per_page)
        except Exception:
            sensors = []
        for s in (sensors or []):
            p = s.get("parameter")
            pname = (p.get("name") if isinstance(p, dict) else p) if p is not None else None
            if not (isinstance(pname, str) and pname.lower().strip() == "no2"):
                continue
            sid_raw = s.get("sensorsId") or s.get("id")
            try: sid = int(sid_raw) if sid_raw is not None else None
            except Exception: sid = None
            if sid is None: continue
            if restrict_sensor_ids is not None and sid not in want:
                continue
            targets.append((lid, sid))

    if not targets:
        return pd.DataFrame()

    endpoint = "hours" if hourly else "measurements"
    all_rows: List[Dict[str, Any]] = []

    for lid, sid in targets:
        url = f"{BASE}/v3/sensors/{sid}/{endpoint}"
        tries = 0
        while True:
            tries += 1
            try:
                meas = _paged_get(url, api_key, {"date_from": date_from, "date_to": date_to}, per_page=per_page)
                break
            except Exception as ex:
                if tries >= 3: meas = []; break
                time.sleep(1.0 * tries)

        meta = loc_meta.get(lid, {})
        for m in (meas or []):
            utc = _safe(m, "period", "datetimeTo", "utc") or _safe(m, "datetime", "utc")
            local = _safe(m, "period", "datetimeTo", "local") or _safe(m, "datetime", "local")
            unit = _safe(m, "parameter", "units")
            param_name = _safe(m, "parameter", "name") or "no2"

            lat = meta.get("lat"); lon = meta.get("lon")
            provider = meta.get("provider"); owner = meta.get("owner")
            row_sid = m.get("sensorsId") or sid
            row_lid = m.get("locationsId") or lid
            flags = m.get("flags"); flag_first = flags[0] if isinstance(flags, list) and flags else None

            all_rows.append({
                "utc": utc,
                "local": local,
                "value": m.get("value"),
                "unit": unit,
                "parameter": param_name,
                "sensorsId": row_sid,
                "locationsId": row_lid,
                "lat": lat, "lon": lon,
                "provider": provider, "owner": owner,
                "flag": flag_first,
                "period_label": _safe(m, "period", "label") or ("1hour" if hourly else "raw"),
            })

        if sleep_between and sleep_between > 0:
            time.sleep(float(sleep_between))

    df = pd.DataFrame(all_rows)
    if not df.empty:
        if "utc" in df.columns:
            df = df[df["utc"].notna()]
        if not df.empty:
            df = df.sort_values(["sensorsId", "utc"], kind="stable").drop_duplicates(subset=["sensorsId","utc"], keep="last")
    return df


# -----------------------------
# Coverage report helper
# -----------------------------
def coverage_3h_report(
    df: pd.DataFrame,
    *,
    time_cols: tuple[str, ...] = ("utc", "datetime", "time", "date_utc"),
    sensor_col: str = "sensorsId",
    window_hours: int = 3,
) -> pd.DataFrame:
    """
    Compute a tiny per-sensor coverage summary over the rows present in `df`.

    - Finds the first existing time column among `time_cols` and parses to UTC.
    - Counts DISTINCT hour buckets per sensor (so duplicate rows in the same hour aren't overcounted).
    - Returns columns:
        sensorsId, n_rows, n_hours, min_time, max_time, expected_hours, completeness

    Notes:
    - `expected_hours` is min(window_hours, hours between min_time and max_time inclusive).
      If your `df` only contains a sub-window of 3 hours, completeness reflects that.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            sensor_col, "n_rows", "n_hours", "min_time", "max_time", "expected_hours", "completeness"
        ])

    # pick a usable time column
    tcol = next((c for c in time_cols if c in df.columns), None)
    if tcol is None:
        # nothing we can do without a timestamp
        return pd.DataFrame(columns=[
            sensor_col, "n_rows", "n_hours", "min_time", "max_time", "expected_hours", "completeness"
        ])

    # basic hygiene
    tmp = df[[sensor_col, tcol]].copy() if sensor_col in df.columns else df[[tcol]].copy()
    tmp[tcol] = pd.to_datetime(tmp[tcol], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=[tcol])

    if tmp.empty:
        return pd.DataFrame(columns=[
            sensor_col, "n_rows", "n_hours", "min_time", "max_time", "expected_hours", "completeness"
        ])

    # If sensor id column is missing, synthesize a single bucket
    if sensor_col not in tmp.columns:
        tmp[sensor_col] = -1  # single pseudo-sensor

    # distinct hours per sensor
    tmp["_hour"] = tmp[tcol].dt.floor("1h")

    g = tmp.groupby(sensor_col, dropna=False)
    out = g.agg(
        n_rows=(tcol, "size"),
        n_hours=("_hour", pd.Series.nunique),
        min_time=(tcol, "min"),
        max_time=(tcol, "max"),
    ).reset_index()

    # expected hours based on observed span, capped at window_hours
    span_hours = ((out["max_time"] - out["min_time"]).dt.total_seconds() / 3600.0).fillna(0)
    # +1 because a span of 0h with one hour bucket is still 1 expected hour
    expected = (span_hours.round(6) + 1).clip(lower=1).astype(int)
    out["expected_hours"] = expected.clip(upper=window_hours)
    out["completeness"] = (out["n_hours"] / out["expected_hours"]).clip(upper=1.0)

    # nice ordering
    out = out.sort_values(["n_hours", "n_rows", sensor_col], ascending=[True, True, True]).reset_index(drop=True)
    return out



def fetch_openaq_no2_latest_bbox(api_key: str, bbox_ws_en: str, per_page: int = 1000) -> pd.DataFrame:
    pid = _get_parameter_id(api_key, "no2")
    if pid:
        try:
            url = f"{BASE}/v3/parameters/{pid}/latest"
            results = _paged_get(url, api_key, {"bbox": bbox_ws_en}, per_page=per_page)
            rows = []
            for r in results or []:
                coords = r.get("coordinates") or {}
                rows.append({
                    "utc": _safe(r, "datetime", "utc"),
                    "local": _safe(r, "datetime", "local"),
                    "value": r.get("value"),
                    "unit": r.get("unit"),
                    "parameter": "no2",
                    "sensorsId": r.get("sensorsId"),
                    "locationsId": r.get("locationsId"),
                    "lat": coords.get("latitude"),
                    "lon": coords.get("longitude"),
                    "provider": _safe(r, "provider", "name"),
                    "owner": _safe(r, "owner", "name"),
                    "flag": (r.get("flags") or [None])[0] if r.get("flags") else None,
                })
            return pd.DataFrame(rows)
        except Exception:
            pass
    try:
        locs = _paged_get(f"{BASE}/v3/locations", api_key, {"bbox": bbox_ws_en}, per_page=per_page)
    except Exception:
        locs = []
    rows = []
    for loc in (locs or []):
        lid = loc.get("locationsId") or loc.get("id") or loc.get("locationId")
        if lid is None: continue
        try:
            latest = _paged_get(f"{BASE}/v3/locations/{lid}/latest", api_key, {}, per_page=per_page)
        except Exception:
            latest = []
        for r in (latest or []):
            pname = r.get("parameter")
            pname = pname.lower() if isinstance(pname, str) else str(pname or "").lower()
            if pname and pname != "no2": continue
            coords = r.get("coordinates") or {}
            rows.append({
                "utc": _safe(r, "datetime", "utc"),
                "local": _safe(r, "datetime", "local"),
                "value": r.get("value"),
                "unit": r.get("unit"),
                "parameter": "no2" if not pname else pname,
                "sensorsId": r.get("sensorsId"),
                "locationsId": r.get("locationsId"),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "provider": _safe(r, "provider", "name"),
                "owner": _safe(r, "owner", "name"),
                "flag": (r.get("flags") or [None])[0] if r.get("flags") else None,
            })
    return pd.DataFrame(rows)
