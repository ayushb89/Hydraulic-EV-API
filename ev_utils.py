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
    if health_status.lower() == "normal":
        return "Low"
    elif health_status.lower() == "warning":
        return "Medium"
    elif health_status.lower() == "critical":
        return "High"
    return "Unknown"


def generate_ev_recommendation(risk_level: str, failure_mode: str) -> str:
    if risk_level == "Low":
        return "Vehicle operating normally. No action required."

    rec = ""
    if "Inverter" in failure_mode:
        rec = "Inspect inverter cooling system and thermal paste"
    elif "Motor_Bearing" in failure_mode:
        rec = "Schedule motor bearing inspection"
    elif "Phase_Current" in failure_mode:
        rec = "Run motor winding resistance test and check inverter outputs"
    elif "Battery" in failure_mode:
        rec = "Perform battery cell balancing and BMS diagnostics"
    elif "Coolant" in failure_mode:
        rec = "Check coolant lines, pump, and reservoir level"
    else:
        rec = "Perform full EV powertrain diagnostic scan"

    if risk_level == "Medium":
        return f"{rec} within 7 days."
    elif risk_level == "High":
        return f"CRITICAL: {rec} immediately. Do not operate until inspected."
    return "Monitor EV powertrain parameters."


# ─────────────────────────────────────────────────────────────────
# EV Heuristic Health Detection (when ML models not available)
# Uses physics-based threshold rules derived from EV engineering
# ─────────────────────────────────────────────────────────────────
def heuristic_ev_health(telemetry: dict) -> tuple:
    """
    Returns (health_status, failure_mode, health_confidence, failure_confidence)
    using EV engineering threshold rules.
    """
    motor_temp    = telemetry.get("Motor_Temp", 0)
    inverter_temp = telemetry.get("Inverter_Temp", 0)
    battery_temp  = telemetry.get("Battery_Temperature", 0)
    battery_soc   = telemetry.get("Battery_SOC", 100)
    phase_current = telemetry.get("Phase_Current", 0)
    motor_rpm     = telemetry.get("Motor_RPM", 0)

    # Critical thresholds
    if motor_temp > 150 or inverter_temp > 90:
        return "Critical", "Inverter_Thermal_Throttling", 0.91, 0.87
    if battery_temp > 50 or battery_soc < 5:
        return "Critical", "Battery_Cell_Imbalance", 0.88, 0.84
    if phase_current > 700 and motor_rpm < 1000:
        return "Critical", "Phase_Current_Imbalance", 0.85, 0.80

    # Warning thresholds
    if motor_temp > 120 or inverter_temp > 75:
        return "Warning", "Inverter_Thermal_Throttling", 0.78, 0.72
    if battery_temp > 42 or battery_soc < 15:
        return "Warning", "Battery_Cell_Imbalance", 0.75, 0.70
    if phase_current > 550:
        return "Warning", "Phase_Current_Imbalance", 0.72, 0.68
    if motor_rpm > 15000 and motor_temp > 100:
        return "Warning", "Motor_Bearing_Wear", 0.70, 0.65

    return "Normal", "Normal_Operation", 0.92, 0.88


# ─────────────────────────────────────────────────────────────────
# EV Prediction Pipeline
# Uses ML if models available, else heuristic rules + LLM
# ─────────────────────────────────────────────────────────────────
def predict_ev_pipeline(vehicle_id: str, telemetry: dict) -> Dict[str, Any]:
    """
    Main EV prediction pipeline.
    1. If EV ML models are trained and loaded → use ML prediction.
    2. Otherwise → use physics-based heuristic rules.
    3. Always call LLM for deep analysis on Warning/Critical.
    """
    vehicle_type   = telemetry.get("Vehicle_Type", "EV Vehicle")
    health_status  = None
    failure_mode   = None
    health_conf    = 0.0
    failure_conf   = 0.0
    analysis_type  = "ML + AI Analysis (EV Module)"

    # ── ML Path ─────────────────────────────────────────────────
    if ev_model_manager.is_loaded:
        try:
            # Build 300-row history for rolling features
            random.seed(42)
            noise = 0.005
            base_time = datetime.now(timezone.utc) - timedelta(seconds=300)
            rows = []
            for i in range(299):
                t = base_time + timedelta(seconds=i)
                rows.append({
                    "timestamp":           t.isoformat(),
                    "Vehicle_Type":        vehicle_type,
                    "Motor_RPM":           round(telemetry["Motor_RPM"]           * (1 + random.uniform(-noise, noise)), 1),
                    "Motor_Temp":          round(telemetry["Motor_Temp"]           * (1 + random.uniform(-noise, noise)), 2),
                    "Inverter_Temp":       round(telemetry["Inverter_Temp"]        * (1 + random.uniform(-noise, noise)), 2),
                    "Motor_Torque":        round(telemetry["Motor_Torque"]         * (1 + random.uniform(-noise, noise)), 2),
                    "Phase_Current":       round(telemetry["Phase_Current"]        * (1 + random.uniform(-noise, noise)), 2),
                    "Battery_SOC":         round(min(100.0, telemetry["Battery_SOC"] + random.uniform(-0.1, 0.1)), 2),
                    "Battery_Temperature": round(telemetry["Battery_Temperature"]  * (1 + random.uniform(-noise, noise)), 2),
                    "Vehicle_Speed":       round(telemetry["Vehicle_Speed"]        * (1 + random.uniform(-noise, noise)), 2),
                    "Operating_Hours":     round(telemetry["Operating_Hours"] - (299 - i) / 3600.0, 4),
                })
            # Append actual snapshot
            rows.append({
                "timestamp":           (base_time + timedelta(seconds=299)).isoformat(),
                **{k: telemetry[k] for k in [
                    "Motor_RPM", "Motor_Temp", "Inverter_Temp", "Motor_Torque",
                    "Phase_Current", "Battery_SOC", "Battery_Temperature",
                    "Vehicle_Speed", "Operating_Hours"
                ]},
                "Vehicle_Type": vehicle_type,
            })

            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df_feat = generate_ev_features(df)
            latest = df_feat.iloc[[-1]].copy()

            expected_cols = ev_model_manager.health_model.feature_names_in_
            X = latest[[c for c in expected_cols if c in latest.columns]]

            h_pred  = ev_model_manager.health_model.predict(X)
            h_probs = ev_model_manager.health_model.predict_proba(X)
            health_conf = float(h_probs.max(axis=1)[0])

            f_pred  = ev_model_manager.failure_model.predict(X)
            f_probs = ev_model_manager.failure_model.predict_proba(X)
            failure_conf = float(f_probs.max(axis=1)[0])

            health_status = ev_model_manager.health_encoder.inverse_transform(h_pred)[0]
            failure_mode  = ev_model_manager.failure_encoder.inverse_transform(f_pred)[0]

        except Exception as e:
            # ML failed — fall through to heuristic
            health_status = None

    # ── Heuristic Fallback ───────────────────────────────────────
    if health_status is None:
        health_status, failure_mode, health_conf, failure_conf = heuristic_ev_health(telemetry)
        analysis_type = "Heuristic + AI Analysis (EV Module — ML models not trained yet)"

    # ── Risk & Recommendation ────────────────────────────────────
    risk_level         = calculate_ev_risk(health_status)
    recommended_action = generate_ev_recommendation(risk_level, failure_mode)

    # ── LLM Deep Analysis (Warning / Critical only) ──────────────
    ai_analysis = None
    if risk_level != "Low":
        ai_analysis = generate_ev_feedback(
            vehicle_type   = vehicle_type,
            telemetry_row  = telemetry,
            health_status  = health_status,
            failure_mode   = failure_mode,
            risk_level     = risk_level
        )

    return {
        "vehicle_id":         vehicle_id,
        "vehicle_type":       vehicle_type,
        "health_status":      health_status,
        "risk_level":         risk_level,
        "failure_mode":       failure_mode,
        "health_confidence":  round(health_conf, 2),
        "failure_confidence": round(failure_conf, 2),
        "recommended_action": recommended_action,
        "analysis_type":      analysis_type,
        "ai_analysis":        ai_analysis,
    }
