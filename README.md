# 🌍 TEMPO Air Quality API

This project integrates **NASA TEMPO satellite data**, **OpenAQ ground measurements**, and **Open-Meteo weather** into a unified API.
It processes data into **CSV files**, provides **forecasts**, and exposes a **FastAPI service** with health & metrics endpoints.

---

## ✨ Features

* 🚀 Pulls **TEMPO NO₂** data via Harmony API
* 📡 Collects **ground sensor data** from OpenAQ
* ☁️ Fetches **weather data** from Open-Meteo
* 🔄 Fuses satellite, ground, and weather into **CSV outputs**
* ⚡ Provides an **API (FastAPI + Uvicorn)** for forecasts & monitoring
* ⚙️ Configurable via `.env` file (**no database required — CSV-based**)

---

## 📂 Project Structure

```text
tempo_api-call/
├── air_quality_api.py     # API server (main entrypoint)
├── alert_user.py          # Email/SMS alert system
├── config.py              # Loads environment variables
├── forecaster.py          # Forecasting logic
├── fuse_validate_merge.py # Data fusion & validation
├── logging_utils.py       # Logging helpers
├── main.py                # Alternate entrypoint
├── openaq_api.py          # OpenAQ fetcher
├── tempo_api.py           # TEMPO fetcher (Harmony)
├── weather_api.py         # Weather fetcher
├── visual_map.py          # Folium map visualizer
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
└── out/                   # Output CSV files
```

---

## ⚙️ Requirements

* 🐍 **Python 3.11**
* Or 🐳 **Docker (recommended)**

If running locally, install system dependencies:

* `libhdf5`, `libnetcdf` (for `netCDF4/xarray`)
* `curl` (for healthchecks)

---

## 🚀 Quickstart

### 1️⃣ Clone

```bash
git clone https://github.com/<your-username>/tempo_api-call.git
cd tempo_api-call
```

### 2️⃣ Configure `.env`

Create a `.env` file in the root folder. Example:

```env
EARTHDATA_USER=annoy
EARTHDATA_PASS=Shahriar@2938
OPENAQ_API_KEY=4533f7b7...
BBOX_WSEN=-124.48,32.53,-114.13,42.01
OUT_DIR=./out
GRID_STEP=0.25
WEATHER_GRID_STEP=0.25
WEATHER_TIMEZONE=UTC
WEATHER_HOURLY_VARS=temperature_2m,pressure_msl,wind_speed_10m,cloud_cover
LOG_LEVEL=INFO
METRICS_PORT=9308
HEALTH_PORT=8088
SCHED_ENABLE=1
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=bookf826@gmail.com
SMTP_PASS=lwrg...vaas
SMTP_FROM=Your App air quality
SMTP_STARTTLS=1
```

⚠️ **Never commit secrets** → add `.env.example` to your repo and keep `.env` private.

---

### 3️⃣ Run with Docker (Recommended)

**Build:**

```bash
docker build -t tempo-aq .
```

**Run:**

```bash
docker run --rm -p 8080:8080 --env-file .env tempo-aq
```

👉 Your API will be live at:
[http://localhost:8080](http://localhost:8080)

---

### 4️⃣ Run Locally (Without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python air_quality_api.py
```

---

## 📡 API Reference

Base URL (local): `http://localhost:8080`

> Tip: Every endpoint returns JSON unless noted. Times are **UTC ISO-8601**.

### 🩺 Health

| Method | Path       | Description           |
| -----: | ---------- | --------------------- |
|    GET | `/healthz` | Simple liveness probe |

```bash
curl http://localhost:8080/healthz
# → {"ok": true}
```

---

### 🧭 Metadata

| Method | Path                    | Description                                    |
| -----: | ----------------------- | ---------------------------------------------- |
|    GET | `/api/meta/datasources` | Info about bbox, grid, and last CSV timestamps |

**Response (example)**

```json
{
  "out_dir": "./out",
  "bbox_wsen": [-124.48, 32.53, -114.13, 42.01],
  "grid_step_default": 0.25,
  "last_fused_csv_mtime": "2025-10-03T03:20:11+00:00",
  "last_metrics_csv_mtime": "2025-10-03T03:21:06+00:00",
  "sources": {
    "satellite": "NASA TEMPO (Harmony; L2/L3 NRT)",
    "ground": "OpenAQ v3",
    "weather": "Open-Meteo"
  }
}
```

---

### 🧪 NO₂ AQI Breakpoints

| Method | Path                       | Description                       |
| -----: | -------------------------- | --------------------------------- |
|    GET | `/api/aqi/no2/breakpoints` | US-style NO₂ 1-hr AQI breakpoints |

```bash
curl http://localhost:8080/api/aqi/no2/breakpoints
```

---

### 🟩 Latest NO₂ Grid

| Method | Path              | Query Params                           | Description                             |
| -----: | ----------------- | -------------------------------------- | --------------------------------------- |
|    GET | `/api/no2/latest` | `grid` (0.05–2.0), `bbox` as `W,S,E,N` | Latest hour, rebinned if `grid` differs |

```bash
curl "http://localhost:8080/api/no2/latest?grid=0.25&bbox=-119.2,33.3,-117.3,34.7"
```

**Response (trimmed)**

```json
{
  "time_utc": "2025-10-03T02:00:00+00:00",
  "grid": 0.25,
  "cells": [
    {
      "cell_id": "CA_0.25_34.00_-118.25",
      "lat": 34.0,
      "lon": -118.25,
      "no2_ugm3": 28.7,
      "no2_ppb": 15.2,
      "aqi": 53,
      "category": "Moderate"
    }
  ],
  "provenance": { "satellite": "TEMPO", "ground": "OpenAQ", "weather": "Open-Meteo" }
}
```

Errors: `400 Invalid bbox`, `404 No fused data available`.

---

### 📈 NO₂ Time-series (cell or lat/lon)

| Method | Path                  | Query Params                                   | Description                     |
| -----: | --------------------- | ---------------------------------------------- | ------------------------------- |
|    GET | `/api/no2/timeseries` | `lat`, `lon` **or** `cell_id`; `hours` (1–168) | Hourly series for one grid cell |

```bash
# by coordinates
curl "http://localhost:8080/api/no2/timeseries?lat=34.05&lon=-118.25&hours=48"

# by cell id (format CA_{grid}_{lat}_{lon})
curl "http://localhost:8080/api/no2/timeseries?cell_id=CA_0.25_34.00_-118.25&hours=72"
```

Errors: `400 Provide (lat & lon) or cell_id`, `400 Invalid cell_id format`, `404 No fused data available`.

---

### 🔮 NO₂ Forecast (persistence nowcast)

| Method | Path                | Query Params       | Description                                   |
| -----: | ------------------- | ------------------ | --------------------------------------------- |
|    GET | `/api/no2/forecast` | `h` (1–24), `grid` | Copies latest hour forward `h` hours per cell |

```bash
curl "http://localhost:8080/api/no2/forecast?h=6&grid=0.25"
```

**Response fields:** `time_run_utc`, `based_on_time`, `horizon_hours`, `grid`, `forecasts[]`.

---

### ✅ Validation Metrics (hourly)

| Method | Path                       | Query Params                   | Description                               |
| -----: | -------------------------- | ------------------------------ | ----------------------------------------- |
|    GET | `/api/validate/no2/hourly` | `start`, `end` (ISO or `...Z`) | Reads `fused_metrics_*.csv` (last 7 days) |

```bash
curl "http://localhost:8080/api/validate/no2/hourly?start=2025-10-02T00:00:00Z&end=2025-10-03T00:00:00Z"
```

Returns per-hour: `n_cells`, `corr_weighted` (as `r_sat_grd`), `rmse`.

---

### 🚨 Alerts (NO₂ AQI threshold)

**Forecast-based alerts (GET)**

| Method | Path              | Query Params                        | Description                                  |
| -----: | ----------------- | ----------------------------------- | -------------------------------------------- |
|    GET | `/api/alerts/no2` | `threshold_aqi` (0–500), `h` (1–24) | Cells predicted ≥ threshold within `h` hours |

```bash
curl "http://localhost:8080/api/alerts/no2?threshold_aqi=150&h=6"
```

**Create an alert job (POST)**

> Your code exposes `POST /alerts/no2` with a `No2AlertRequest` body (e.g., email/phone). Document your exact schema there.

```bash
curl -X POST http://localhost:8080/alerts/no2 \
  -H "Content-Type: application/json" \
  -d '{ "threshold_aqi": 150, "hours": 6, "channels": ["email"], "target": "user@example.com" }'
```

---

### 🧱 Grid Cells Helper

| Method | Path         | Query Params   | Description                            |
| -----: | ------------ | -------------- | -------------------------------------- |
|    GET | `/api/cells` | `grid`, `bbox` | Returns grid cells + bounds for a bbox |

```bash
curl "http://localhost:8080/api/cells?grid=0.25&bbox=-119.2,33.3,-117.3,34.7"
```

---

### 📍 OpenAQ Stations

| Method | Path                                    | Query Params    | Description                     |
| -----: | --------------------------------------- | --------------- | ------------------------------- |
|    GET | `/api/stations`                         | `bbox`          | Station list in bbox            |
|    GET | `/api/stations/{station_id}/timeseries` | `hours` (1–168) | Hourly NO₂ series for a station |

```bash
curl "http://localhost:8080/api/stations?bbox=-119.2,33.3,-117.3,34.7"

curl "http://localhost:8080/api/stations/12345/timeseries?hours=72"
```

Notes: Station series are returned in the station’s **native unit** (no ppb conversion).

---

### 📅 Convenience Services

| Method | Path              | Description                                         |
| -----: | ----------------- | --------------------------------------------------- |
|    GET | `/api/latest_day` | Returns latest day key available in outputs         |
|    GET | `/api/grid/hour`  | Grid sample for a given `day` & `hour` (limit rows) |
|    GET | `/api/series`     | Time-series at a lat/lon over `days` (1–30)         |
|    GET | `/api/map/latest` | Returns latest preview map (file download)          |
|    GET | `/api/map/multi`  | Returns multi-layer preview map (file download)     |

**Examples**

```bash
curl "http://localhost:8080/api/grid/hour?day=2025-10-02&hour=18&limit=1000"
curl "http://localhost:8080/api/series?lat=34.05&lon=-118.25&days=7" 
curl -OJ "http://localhost:8080/api/map/latest"
```

---

### 🔁 Common Errors

| Code | Meaning                                      |
| ---: | -------------------------------------------- |
|  400 | Bad request (e.g., bbox/cell_id/time format) |
|  404 | No fused data available                      |
|  500 | Unexpected server error                      |

---

### 🔐 Auth

These endpoints are open for demo/hackathon use. If you need auth later, add a proxy or FastAPI dependency for tokens.
---
## 📦 Outputs

Processed CSV files are stored in the `out/` directory:

* `tempo_*.csv` → TEMPO satellite data
* `openaq_*.csv` → Ground sensor data
* `weather_*.csv` → Weather snapshots
* `fused_truth_dataset.csv` → Combined dataset
* `fused_metrics.csv` → Validation metrics

---

## 🛠 Development

Format & lint:

```bash
black .
ruff check .
```

---

## 🧪 Example Usage

**Fetch forecast for Los Angeles:**

```bash
curl "http://localhost:8080/forecast?lat=34.05&lon=-118.25&hours=24"
```

**Get validation metrics:**

```bash
curl "http://localhost:8080/metrics"
```

---

## 📜 License

MIT © 2025 **Team Saviour**

---

## 🙏 Acknowledgements

* 🌍 [NASA TEMPO](https://tempo.si.edu/) (via Harmony API)
* 📡 [OpenAQ API](https://openaq.org/)
* ☁️ [Open-Meteo](https://open-meteo.com/)

---
