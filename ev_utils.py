"""
ev_utils.py
────────────
EV Brake Telemetry prediction pipeline.

Uses:
  1. ML Path  → if ev_health_model.pkl + ev_failure_model.pkl exist in models/ev/
  2. Heuristic Fallback → physics-based brake threshold rules
  3. LLM Analysis → always called for Warning / Critical states
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from ev_models import ev_model_manager
from ev_feature_engineering import generate_ev_features
from llm_diagnostics import generate_ev_feedback


# ─────────────────────────────────────────────────────────────────
# EV Risk & Recommendation Helpers
# ─────────────────────────────────────────────────────────────────
def calculate_ev_risk(health_status: str) -> str:
    mapping = {"normal": "Low", "warning": "Medium", "critical": "High"}
    return mapping.get(health_status.lower(), "Unknown")


def generate_ev_recommendation(risk_level: str, failure_mode: str) -> str:
    if risk_level == "Low":
        return "EV brake system operating normally. No action required."

    rec_map = {
        "Brake_Fluid_Overheating":        "Inspect brake fluid condition and cooling channels",
        "ABS_Sensor_Fault":               "Run ABS self-diagnostic and inspect wheel-speed sensors",
        "Hydraulic_Pressure_Loss":        "Check master cylinder, calipers, and brake lines for leaks",
        "Brake_Pad_Wear":                 "Measure pad thickness and inspect rotors for scoring",
        "Fluid_Contamination":            "Flush and replace brake fluid; check for moisture ingress",
        "Multiple_Simultaneous_Failures": "Perform full brake system overhaul — multiple systems affected",
        "Normal_Operation":               "Continue routine monitoring",
    }
    rec = rec_map.get(failure_mode, "Perform full EV brake system diagnostic scan")

    if risk_level == "Medium":
        return f"{rec} within 7 days."
    elif risk_level == "High":
        return f"CRITICAL: {rec} immediately. Do not operate until inspected."
    return "Monitor EV brake parameters."


# ─────────────────────────────────────────────────────────────────
# EV Brake Heuristic Health Detection (ML fallback)
# Physics-based threshold rules for EV brake systems
# ─────────────────────────────────────────────────────────────────
def heuristic_ev_health(telemetry: dict) -> tuple:
    """
    Returns (health_status, failure_mode, health_confidence, failure_confidence)
    using EV brake engineering threshold rules.
    """
    fluid_temp   = telemetry.get("Brake_Fluid_Temperature_C", 0)
    fluid_level  = telemetry.get("Brake_Fluid_Level_pct", 100)
    hydraulic_p  = telemetry.get("Brake_Hydraulic_Pressure_bar", 0)
    line_p       = telemetry.get("Brake_Line_Pressure_bar", 0)
    abs_freq     = telemetry.get("ABS_Activation_Frequency", 0)
    vibration    = telemetry.get("Vibration_g", 0)
    pedal_pos    = telemetry.get("Brake_Pedal_Position_pct", 0)
    batt_soc     = telemetry.get("Battery_SOC", 100)
    batt_temp    = telemetry.get("Battery_Temperature", 25)

    # ── Critical thresholds ─────────────────────────────────────
    if fluid_temp > 100:
        return "Critical", "Brake_Fluid_Overheating", 0.91, 0.87
    if fluid_level < 55:
        return "Critical", "Hydraulic_Pressure_Loss", 0.90, 0.86
    if abs(hydraulic_p - line_p) > 25:
        return "Critical", "Hydraulic_Pressure_Loss", 0.88, 0.83
    if abs_freq > 8 and vibration > 0.5:
        return "Critical", "ABS_Sensor_Fault", 0.87, 0.82
    if vibration > 0.6:
        return "Critical", "Multiple_Simultaneous_Failures", 0.86, 0.81

    # ── Warning thresholds ──────────────────────────────────────
    if fluid_temp > 65:
        return "Warning", "Brake_Fluid_Overheating", 0.79, 0.74
    if fluid_level < 75:
        return "Warning", "Fluid_Contamination", 0.76, 0.71
    if abs(hydraulic_p - line_p) > 10:
        return "Warning", "Hydraulic_Pressure_Loss", 0.74, 0.69
    if abs_freq > 5:
        return "Warning", "ABS_Sensor_Fault", 0.73, 0.68
    if vibration > 0.3:
        return "Warning", "Brake_Pad_Wear", 0.71, 0.66
    if batt_temp > 40 or batt_soc < 15:
        return "Warning", "Fluid_Contamination", 0.70, 0.65

    return "Normal", "Normal_Operation", 0.93, 0.90


# ─────────────────────────────────────────────────────────────────
# Build a 300-row history DataFrame from a single snapshot
# (Required for rolling-window features)
# ─────────────────────────────────────────────────────────────────
def _build_history_df(telemetry: dict, vehicle_type: str) -> pd.DataFrame:
    """
    Generates a 300-row DataFrame with small Gaussian noise around the
    snapshot values, so that rolling features can be computed correctly.
    """
    noise = 0.005
    base_time = datetime.now(timezone.utc) - timedelta(seconds=300)
    rows = []

    for i in range(299):
        t = base_time + timedelta(seconds=i)
        row = {"Timestamp": t.isoformat()}
        for key, val in telemetry.items():
            if key == "Vehicle_Type":
                row[key] = val
                continue
            if isinstance(val, (int, float)):
                row[key] = round(val * (1 + random.uniform(-noise, noise)), 4)
            else:
                row[key] = val
        rows.append(row)

    # Append the actual live snapshot as the last row
    last = {"Timestamp": (base_time + timedelta(seconds=299)).isoformat()}
    last.update(telemetry)
    rows.append(last)

    df = pd.DataFrame(rows)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# Main EV Prediction Pipeline
# ML → Heuristic fallback → LLM analysis
# ─────────────────────────────────────────────────────────────────
def predict_ev_pipeline(vehicle_id: str, telemetry: dict) -> Dict[str, Any]:
    """
    Main EV Brake prediction pipeline.

    1. If EV ML models are trained and loaded → use ML prediction.
    2. Otherwise → use physics-based heuristic brake threshold rules.
    3. Always call LLM for deep analysis on Warning/Critical.
    """
    vehicle_type  = telemetry.get("Vehicle_Type", "EV Vehicle")
    health_status = None
    failure_mode  = None
    health_conf   = 0.0
    failure_conf  = 0.0
    analysis_type = "ML + AI Analysis (EV Brake Module)"

    # ── ML Path ──────────────────────────────────────────────────
    if ev_model_manager.is_loaded:
        try:
            random.seed(42)
            df = _build_history_df(telemetry, vehicle_type)
            df_feat = generate_ev_features(df)
            latest = df_feat.iloc[[-1]].copy()

            # Use saved feature column order if available, else fall back to model's own
            if ev_model_manager.feature_columns:
                feat_cols = [c for c in ev_model_manager.feature_columns if c in latest.columns]
            else:
                feat_cols = [c for c in ev_model_manager.health_model.feature_names_in_
                             if c in latest.columns]

            X = latest[feat_cols].fillna(0)

            h_pred   = ev_model_manager.health_model.predict(X)
            h_probs  = ev_model_manager.health_model.predict_proba(X)
            health_conf = float(h_probs.max(axis=1)[0])

            f_pred   = ev_model_manager.failure_model.predict(X)
            f_probs  = ev_model_manager.failure_model.predict_proba(X)
            failure_conf = float(f_probs.max(axis=1)[0])

            health_status = ev_model_manager.health_encoder.inverse_transform(h_pred)[0]
            failure_mode  = ev_model_manager.failure_encoder.inverse_transform(f_pred)[0]

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"EV ML prediction failed, falling back to heuristic: {e}")
            health_status = None

    # ── Heuristic Fallback ───────────────────────────────────────
    if health_status is None:
        health_status, failure_mode, health_conf, failure_conf = heuristic_ev_health(telemetry)
        analysis_type = "Heuristic + AI Analysis (EV Brake Module — ML models not trained yet)"

    # ── Risk & Recommendation ────────────────────────────────────
    risk_level         = calculate_ev_risk(health_status)
    recommended_action = generate_ev_recommendation(risk_level, failure_mode)

    # ── LLM Deep Analysis (Warning / Critical only) ───────────────
    ai_analysis = None
    if risk_level != "Low":
        ai_analysis = generate_ev_feedback(
            vehicle_type  = vehicle_type,
            telemetry_row = telemetry,
            health_status = health_status,
            failure_mode  = failure_mode,
            risk_level    = risk_level
        )

    return {
        "vehicle_id":         vehicle_id,
        "vehicle_type":       vehicle_type,
        "health_status":      health_status,
        "risk_level":         risk_level,
        "failure_mode":       failure_mode,
        "health_confidence":  round(health_conf,   2),
        "failure_confidence": round(failure_conf,  2),
        "recommended_action": recommended_action,
        "analysis_type":      analysis_type,
        "ai_analysis":        ai_analysis,
    }
