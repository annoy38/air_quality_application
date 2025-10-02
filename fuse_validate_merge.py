#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
from sklearn.linear_model import LinearRegression

from config import get_config

R = 8.31446261815324  # J/(mol*K)
MW_NO2 = 46.0055      # g/mol

_cfg = get_config()
_W, _S, _E, _N = [float(x) for x in _cfg["BBOX_WSEN"].split(",")]

def _clip_bbox(df, lat_col="lat_bin", lon_col="lon_bin"):
    if df is None or df.empty:
        return df
    return df[(df[lat_col] >= _S) & (df[lat_col] <= _N) &
              (df[lon_col] >= _W) & (df[lon_col] <= _E)].copy()

def _to_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _floor_hour(s: pd.Series) -> pd.Series:
    return s.dt.floor("1h")

def _snap_to_grid(vals: pd.Series, step: float) -> pd.Series:
    return (vals / step).round(0) * step

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.replace(" ", "_").replace(".", "_").lower() for c in df.columns]
    return df

def _require(df: pd.DataFrame, cols, name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: missing required columns: {miss}")

def _first_nonnull(*series: pd.Series | None) -> pd.Series | None:
    out = None
    for s in series:
        if s is None: continue
        if out is None: out = s.copy()
        else:
            m = out.isna()
            out.loc[m] = s[m]
    return out

def unit_to_normalized_string(u: str) -> str:
    if u is None: return ""
    s = str(u).strip().lower()
    # Unicode and spacing
    s = (s.replace("µ","u")
           .replace("μ","u")
           .replace("³","3")
           .replace("m³","m3")
           .replace("per","/")
           .replace(" ", ""))
    # common variants
    s = s.replace("ug/m^3", "ug/m3").replace("ug/m³","ug/m3").replace("u g/m3","ug/m3")
    s = s.replace("microgram/m3","ug/m3").replace("micrograms/m3","ug/m3")
    s = s.replace("ppm(v)","ppm").replace("ppbv","ppb").replace("ppb(v)","ppb")
    s = s.replace("partspermillion","ppm").replace("partsperbillion","ppb")
    return s


def no2_to_ugm3(value: float, unit: str, temp_k: float, pressure_pa: float) -> float | None:
    if value is None or not isfinite(value): return None
    u = unit_to_normalized_string(unit)
    if u in ("ug/m3", "µg/m3", "ugm3"): return float(value)
    if u in ("ppb","ppm"):
        if not (isfinite(temp_k) and isfinite(pressure_pa) and temp_k > 0 and pressure_pa > 0):
            return None
        ppb = float(value) if u == "ppb" else float(value) * 1000.0
        ugm3 = ppb * MW_NO2 * pressure_pa / (R * temp_k) * 1e-3
        return float(ugm3)
    if u == "mg/m3": return float(value) * 1000.0
    if u == "ng/m3": return float(value) * 1e-3
    return None

@dataclass
class FuseValidateMerge:
    tempo_csv: str
    openaq_csv: str
    weather_csv: str
    grid_step: float = 0.5
    out_fused_csv: str = "./out/fused_truth_dataset.csv"
    out_metrics_csv: str = "./out/fused_metrics.csv"
    out_map_html: str = "./out/fused_map.html"
    out_multihour_map_html: str = "./out/fused_multihour_map.html"
    multihour_panels: int = 6

    # ---------- TEMPO ----------
    def _load_tempo(self) -> pd.DataFrame:
        # Robust: handle missing/blank path
        if not self.tempo_csv or not os.path.isfile(self.tempo_csv):
            return pd.DataFrame(columns=[
                "time_hour","lat_bin","lon_bin",
                "no2_trop_col_molec_cm2","tempo_n_pix","tempo_pix_std","tempo_qc_note"
            ])

        df = pd.read_csv(self.tempo_csv)
        df = _normalize_cols(df)

        # time
        if "time" not in df.columns:
            tcol = [c for c in df.columns if c.startswith("time") or c.startswith("date")]
            if not tcol:
                return pd.DataFrame(columns=[
                    "time_hour","lat_bin","lon_bin",
                    "no2_trop_col_molec_cm2","tempo_n_pix","tempo_pix_std","tempo_qc_note"
                ])
            df["time"] = df[tcol[0]]
        df["time"] = _to_datetime_utc(df["time"])
        df["time_hour"] = _floor_hour(df["time"])

        # lat/lon
        latc = next((c for c in ("latitude", "lat") if c in df.columns), None)
        lonc = next((c for c in ("longitude", "lon", "lng") if c in df.columns), None)
        if not latc or not lonc:
            return pd.DataFrame(columns=[
                "time_hour","lat_bin","lon_bin",
                "no2_trop_col_molec_cm2","tempo_n_pix","tempo_pix_std","tempo_qc_note"
            ])
        df["lat"] = pd.to_numeric(df[latc], errors="coerce")
        df["lon"] = pd.to_numeric(df[lonc], errors="coerce")

        # NO2 column
        n2cands = [c for c in df.columns if ("no2" in c and "column" in c)]
        if not n2cands:
            n2cands = [c for c in df.columns if ("vertical" in c and "tropos" in c)]
        if not n2cands:
            return pd.DataFrame(columns=[
                "time_hour","lat_bin","lon_bin",
                "no2_trop_col_molec_cm2","tempo_n_pix","tempo_pix_std","tempo_qc_note"
            ])
        vcol = n2cands[0]
        df["no2_trop_col_molec_cm2"] = pd.to_numeric(df[vcol], errors="coerce")
        ucol = next((c for c in df.columns if ("uncert" in c and "no2" in c) or ("sigma" in c and "no2" in c)), None)
        df["tempo_pixel_sigma"] = pd.to_numeric(df[ucol], errors="coerce") if ucol else np.nan

        df = df[np.isfinite(df["no2_trop_col_molec_cm2"])]
        df = df[df["no2_trop_col_molec_cm2"] >= 0]
        df = df[df["no2_trop_col_molec_cm2"] <= 5e16]

        df["lat_bin"] = _snap_to_grid(df["lat"], self.grid_step)
        df["lon_bin"] = _snap_to_grid(df["lon"], self.grid_step)

        tempo = (
            df.groupby(["time_hour","lat_bin","lon_bin"])
              .agg(no2_trop_col_molec_cm2=("no2_trop_col_molec_cm2","mean"),
                   tempo_n_pix=("no2_trop_col_molec_cm2","size"),
                   tempo_pix_std=("no2_trop_col_molec_cm2","std"))
              .reset_index()
        )
        tempo["tempo_pix_std"] = tempo["tempo_pix_std"].fillna(0.0)
        tempo["tempo_qc_note"] = np.where(tempo["tempo_n_pix"] >= 3, "", "sparse_pixel_coverage")
        return tempo

    # ---------- Weather ----------
    def _load_weather(self) -> pd.DataFrame:
        df = pd.read_csv(self.weather_csv)
        df = _normalize_cols(df)
        _require(df, ["time","lat","lon"], "Weather")
        df["time"] = _to_datetime_utc(df["time"])
        df["time_hour"] = _floor_hour(df["time"])
        df["lat_bin"] = _snap_to_grid(df["lat"], self.grid_step)
        df["lon_bin"] = _snap_to_grid(df["lon"], self.grid_step)

        tcol = "temperature_2m" if "temperature_2m" in df.columns else None
        df["temp_c"] = df[tcol] if tcol else np.nan
        df["temp_k"] = df["temp_c"] + 273.15

        pcol = "pressure_msl" if "pressure_msl" in df.columns else ("surface_pressure" if "surface_pressure" in df.columns else None)
        df["pressure_pa"] = df[pcol] * 100.0 if pcol else np.nan

        num_cols = [c for c in df.columns if c not in ("time","time_hour","lat","lon","lat_bin","lon_bin","source")]
        weather = (df.groupby(["time_hour","lat_bin","lon_bin"]).agg({c:"mean" for c in num_cols}).reset_index())
        return weather

    # ---------- OpenAQ ----------
    def _load_openaq(self) -> pd.DataFrame:
        df = pd.read_csv(self.openaq_csv)
        df = _normalize_cols(df)

        time_col = None
        for c in ("utc","date_utc","dateutc","datetime","time","date__utc","date"):
            if c in df.columns: time_col = c; break
        if time_col is None:
            cands = [c for c in df.columns if "date" in c and "utc" in c]
            if cands: time_col = cands[0]
        if time_col is None:
            raise ValueError("OpenAQ: cannot find UTC time column")
        df["time"] = _to_datetime_utc(df[time_col]); df["time_hour"] = _floor_hour(df["time"])

        lat = _first_nonnull(df.get("coordinates_latitude"), df.get("latitude"), df.get("lat"))
        lon = _first_nonnull(df.get("coordinates_longitude"), df.get("longitude"), df.get("lon"))
        if lat is None or lon is None:
            raise ValueError("OpenAQ: cannot find coordinates")
        df["lat"] = pd.to_numeric(lat, errors="coerce")
        df["lon"] = pd.to_numeric(lon, errors="coerce")
        df = df.dropna(subset=["lat","lon","time_hour"])

        valcol = "value" if "value" in df.columns else next((c for c in df.columns if c.endswith("value")), None)
        if valcol is None:
            raise ValueError("OpenAQ: cannot find measurement 'value' column")
        unitcol = "unit" if "unit" in df.columns else next((c for c in df.columns if c.endswith("unit")), None)
        df["no2_ground_value_raw"] = pd.to_numeric(df[valcol], errors="coerce")
        df["no2_ground_unit_raw"] = df[unitcol] if unitcol else "unknown"

        if "parameter" in df.columns:
            df = df[df["parameter"].str.lower().str.contains("no2", na=False)]

        df = df[np.isfinite(df["no2_ground_value_raw"])]
        df = df[df["no2_ground_value_raw"] >= 0]
        df["lat_bin"] = _snap_to_grid(df["lat"], self.grid_step)
        df["lon_bin"] = _snap_to_grid(df["lon"], self.grid_step)

        by_cell = (
            df.groupby(["time_hour","lat_bin","lon_bin"])
              .agg(
                  no2_ground_value_raw_mean=("no2_ground_value_raw","mean"),
                  no2_ground_unit_raw_mode=("no2_ground_unit_raw",
                     lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
                  openaq_n_obs=("no2_ground_value_raw","size"),
                  openaq_std=("no2_ground_value_raw","std"),
              ).reset_index()
        )
        by_cell["openaq_std"] = by_cell["openaq_std"].fillna(0.0)
        return by_cell

    def _enrich_openaq_with_weather(self, openaq: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        w = weather.copy()
        if "temp_k" not in w.columns and "temperature_2m" in w.columns:
            w["temp_k"] = w["temperature_2m"] + 273.15
        if "pressure_pa" not in w.columns:
            pcol = "pressure_msl" if "pressure_msl" in w.columns else ("surface_pressure" if "surface_pressure" in w.columns else None)
            if pcol: w["pressure_pa"] = w[pcol] * 100.0
        for c in ("temp_k","pressure_pa"):
            if c not in w.columns: w[c] = np.nan

        keep = w[["time_hour","lat_bin","lon_bin","temp_k","pressure_pa"]]
        m = openaq.merge(keep, on=["time_hour","lat_bin","lon_bin"], how="left")
        m["no2_ground_ug_m3"] = m.apply(
            lambda r: no2_to_ugm3(r["no2_ground_value_raw_mean"], r["no2_ground_unit_raw_mode"], r["temp_k"], r["pressure_pa"]),
            axis=1,
        )
        m["openaq_qc_note"] = np.where(m["no2_ground_ug_m3"].isna(), "unit_or_env_missing","")
        m["openaq_se"] = m.apply(
            lambda r: (r["openaq_std"]/max(1, np.sqrt(r["openaq_n_obs"]))) if isfinite(r["openaq_std"]) else np.nan, axis=1
        )
        return m

    # ---------- Fusion ----------
    def fuse(self) -> pd.DataFrame:
        tempo = self._load_tempo()
        weather = self._load_weather()
        openaq = self._load_openaq()

        tempo   = _clip_bbox(tempo,   "lat_bin","lon_bin")
        weather = _clip_bbox(weather, "lat_bin","lon_bin")
        openaq  = _clip_bbox(openaq,  "lat_bin","lon_bin")

        openaq_conv = self._enrich_openaq_with_weather(openaq, weather)

        fused = (weather
            .merge(tempo, on=["time_hour","lat_bin","lon_bin"], how="outer")
            .merge(
                openaq_conv[[
                    "time_hour","lat_bin","lon_bin",
                    "no2_ground_ug_m3","no2_ground_value_raw_mean",
                    "no2_ground_unit_raw_mode","openaq_n_obs","openaq_std","openaq_se","openaq_qc_note"
                ]],
                on=["time_hour","lat_bin","lon_bin"], how="outer"
            )
        )

        mask = fused["no2_trop_col_molec_cm2"].notna() & fused["no2_ground_ug_m3"].notna()
        if mask.sum() >= 5:
            X = fused.loc[mask, ["no2_trop_col_molec_cm2"]].values
            y = fused.loc[mask, "no2_ground_ug_m3"].values
            model = LinearRegression().fit(X, y)
            fused["no2_trop_col_corrected_ug_m3"] = model.predict(
                fused[["no2_trop_col_molec_cm2"]].fillna(0).values
            )
            fused["bias_model_slope"] = model.coef_[0]
            fused["bias_model_intercept"] = model.intercept_
            print(f"✅ Bias correction: slope={model.coef_[0]:.3e}, intercept={model.intercept_:.2f}")
        else:
            fused["no2_trop_col_corrected_ug_m3"] = np.nan
            fused["bias_model_slope"] = np.nan
            fused["bias_model_intercept"] = np.nan
            print("⚠️ Not enough overlap for bias correction")

        fused["fused_qc_note"] = np.where(
            fused["no2_trop_col_molec_cm2"].isna() & fused["no2_ground_ug_m3"].isna(),
            "no_no2_observed_in_cell_hour",""
        )

        front = [
            "time_hour","lat_bin","lon_bin",
            "no2_trop_col_molec_cm2","tempo_pix_std","no2_trop_col_corrected_ug_m3",
            "tempo_n_pix","tempo_qc_note",
            "no2_ground_ug_m3","no2_ground_value_raw_mean",
            "no2_ground_unit_raw_mode","openaq_n_obs","openaq_std","openaq_se","openaq_qc_note"
        ]
        weather_cols = [c for c in weather.columns if c not in ("time_hour","lat_bin","lon_bin")]
        ordered = front + [c for c in weather_cols if c not in front]
        fused = fused.reindex(columns=[c for c in ordered if c in fused.columns]).sort_values(["time_hour","lat_bin","lon_bin"])

        Path(self.out_fused_csv).parent.mkdir(parents=True, exist_ok=True)
        fused.to_csv(self.out_fused_csv, index=False)
        print(f"✅ Fused dataset saved: {self.out_fused_csv}  rows={len(fused):,}")

        fused["has_tempo"]   = fused["no2_trop_col_molec_cm2"].notna()
        fused["has_ground"]  = fused["no2_ground_ug_m3"].notna()
        fused["has_weather"] = fused["temp_k"].notna() & fused["pressure_pa"].notna()

        fused_qc = fused[(fused["has_weather"]) & (fused["has_tempo"] | fused["has_ground"])].copy()
        qc_path = self.out_fused_csv.replace(".csv", "_qc.csv")
        fused_qc.to_csv(qc_path, index=False)
        print(f"✅ Fused (QC) saved: {qc_path}  rows={len(fused_qc):,}")

        return fused

    def compute_metrics(self, fused: pd.DataFrame) -> pd.DataFrame:
        df = fused.copy()
        mask = df["no2_trop_col_molec_cm2"].notna() & df["no2_ground_ug_m3"].notna()
        df = df[mask]
        if df.empty:
            print("⚠️ No overlap; metrics skipped.")
            Path(self.out_metrics_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(self.out_metrics_csv, index=False)
            return pd.DataFrame()

        def wpearson(x, y, w):
            x, y, w = np.array(x, float), np.array(y, float), np.array(w, float)
            w = np.where(~np.isfinite(w) | (w<=0), 1.0, w)
            mx = np.average(x, weights=w); my = np.average(y, weights=w)
            cov = np.average((x-mx)*(y-my), weights=w)
            sx  = np.sqrt(np.average((x-mx)**2, weights=w) + 1e-12)
            sy  = np.sqrt(np.average((y-my)**2, weights=w) + 1e-12)
            return float(cov/(sx*sy))

        rows = []
        for t, g in df.groupby("time_hour"):
            if len(g) < 2: continue
            w = g["openaq_n_obs"].fillna(1.0)
            corr_w = wpearson(g["no2_trop_col_molec_cm2"], g["no2_ground_ug_m3"], w)
            rmse = np.sqrt(np.mean((g["no2_trop_col_molec_cm2"] - g["no2_ground_ug_m3"])**2))
            rows.append({"time_hour": t, "corr_weighted": corr_w, "rmse": rmse, "n_cells": len(g)})
        metrics = pd.DataFrame(rows).sort_values("time_hour").reset_index(drop=True)
        Path(self.out_metrics_csv).parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(self.out_metrics_csv, index=False)
        print(f"✅ Metrics saved: {self.out_metrics_csv}  rows={len(metrics):,}")
        return metrics

    def export_map(self, fused: pd.DataFrame):
        if fused.empty:
            print("⚠️ Empty fused dataset; map skipped."); return
        last_time = fused["time_hour"].max()
        sub = fused[fused["time_hour"] == last_time].copy()
        if sub.empty:
            print("⚠️ No rows for latest hour; map skipped."); return

        sub = sub[(sub["lat_bin"] >= _S) & (sub["lat_bin"] <= _N) &
                  (sub["lon_bin"] >= _W) & (sub["lon_bin"] <= _E)].copy()
        center_lat = (_S + _N)/2.0
        center_lon = (_W + _E)/2.0
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True)
        fmap.fit_bounds([[_S, _W], [_N, _E]])

        tempo_layer = folium.FeatureGroup(name="TEMPO NO₂ (column)")
        for _, r in sub.iterrows():
            v = r.get("no2_trop_col_molec_cm2")
            if pd.notna(v):
                folium.CircleMarker(
                    location=[r["lat_bin"], r["lon_bin"]], radius=6,
                    popup=(f"<b>Cell:</b> ({r['lat_bin']}, {r['lon_bin']})<br>"
                           f"<b>TEMPO NO₂:</b> {v:.2e} molecules/cm²<br>"
                           f"<b>Ground NO₂:</b> {r.get('no2_ground_ug_m3', np.nan):.1f} µg/m³"),
                    color="blue", fill=True, fill_opacity=0.55,
                ).add_to(tempo_layer)
        tempo_layer.add_to(fmap)

        ground_layer = folium.FeatureGroup(name="Ground NO₂ (µg/m³)")
        for _, r in sub.iterrows():
            gval = r.get("no2_ground_ug_m3")
            if pd.notna(gval):
                folium.CircleMarker(
                    location=[r["lat_bin"], r["lon_bin"]], radius=4,
                    popup=(f"<b>Cell:</b> ({r['lat_bin']}, {r['lon_bin']})<br>"
                           f"<b>Ground NO₂:</b> {gval:.1f} µg/m³<br>"
                           f"<b>TEMPO NO₂:</b> {r.get('no2_trop_col_molec_cm2', np.nan):.2e} molecules/cm²"),
                    color="red", fill=True, fill_opacity=0.55,
                ).add_to(ground_layer)
        ground_layer.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        Path(self.out_map_html).parent.mkdir(parents=True, exist_ok=True)
        fmap.save(self.out_map_html)
        print(f"✅ Interactive map saved: {self.out_map_html}")

    def export_multihour_map(self, fused: pd.DataFrame):
        if fused.empty:
            print("⚠️ Empty fused dataset; multi-hour map skipped.");
            return
        times = sorted(fused["time_hour"].dropna().unique())[-self.multihour_panels:]
        if not times:
            print("⚠️ No time slices for multi-hour map.");
            return
        sub = fused[fused["time_hour"].isin(times)].copy()

        sub = sub[(sub["lat_bin"] >= _S) & (sub["lat_bin"] <= _N) &
                  (sub["lon_bin"] >= _W) & (sub["lon_bin"] <= _E)].copy()
        center_lat = (_S + _N)/2.0
        center_lon = (_W + _E)/2.0
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True)
        fmap.fit_bounds([[_S, _W], [_N, _E]])

        features = []
        for t in times:
            g = sub[(sub["time_hour"] == t) & sub["no2_ground_ug_m3"].notna()]
            for _, r in g.iterrows():
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(r["lon_bin"]), float(r["lat_bin"])]},
                    "properties": {
                        "time": pd.to_datetime(t).tz_convert(None).isoformat()+"Z",
                        "style": {"color": "red"},
                        "popup": f"{t} — Ground NO₂: {r['no2_ground_ug_m3']:.1f} µg/m³"
                    }
                })
        if features:
            TimestampedGeoJson(
                {"type":"FeatureCollection","features":features},
                period="PT1H", add_last_point=True, auto_play=False, loop=False
            ).add_to(fmap)

        Path(self.out_multihour_map_html).parent.mkdir(parents=True, exist_ok=True)
        fmap.save(self.out_multihour_map_html)
        print(f"✅ Multi-hour map saved: {self.out_multihour_map_html}")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fused = self.fuse()
        metrics = self.compute_metrics(fused)
        self.export_map(fused)
        self.export_multihour_map(fused)
        return fused, metrics
