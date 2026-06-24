import pandas as pd
import numpy as np
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from models import model_manager
from feature_engineering import generate_features
from llm_diagnostics import generate_llm_feedback, generate_unseen_vehicle_feedback

# ─────────────────────────────────────────────────────────────────
# Closest-Match Vehicle Mapping
# When an unseen vehicle type arrives, map it to the nearest
# trained vehicle before running the ML model.
# ─────────────────────────────────────────────────────────────────
VEHICLE_CLOSEST_MATCH = {
    "excavator":          "Backhoe",
    "mini excavator":     "Backhoe",
    "pipelayer":          "Backhoe",
    "trencher":           "Backhoe",
    "crane":              "WheelLoader",
    "rough terrain crane":"WheelLoader",
    "skid steer":         "WheelLoader",
    "motor grader":       "WheelLoader",
    "bulldozer":          "DumpTruck",
    "articulated truck":  "DumpTruck",
    "haul truck":         "DumpTruck",
    "reach stacker":      "Forklift",
    "counterbalance":     "Forklift",
    "pallet truck":       "Forklift",
    "telehandler":        "Telehandler",
    "telescopic handler": "Telehandler",
    "boom lift":          "Telehandler",
    "scissor lift":       "Telehandler",
}

def _find_closest_match(vehicle_type_raw: str) -> Optional[str]:
    """
    Looks up the closest trained vehicle type for an unknown vehicle.
    Returns None if no match found.
    """
    normalized = vehicle_type_raw.strip().lower()
    # Direct lookup
    if normalized in VEHICLE_CLOSEST_MATCH:
        return VEHICLE_CLOSEST_MATCH[normalized]
    # Partial substring match
    for key, matched in VEHICLE_CLOSEST_MATCH.items():
        if key in normalized or normalized in key:
            return matched
    return None


# ─────────────────────────────────────────────────────────────────
# Risk & Recommendation Helpers
# ─────────────────────────────────────────────────────────────────
def calculate_risk_level(health_status: str) -> str:
    """Calculates risk level based on health status."""
    if health_status.lower() == "normal":
        return "Low"
    elif health_status.lower() == "warning":
        return "Medium"
    elif health_status.lower() == "critical":
        return "High"
    return "Unknown"

def generate_recommendation(risk_level: str, failure_mode: str) -> str:
    """Generates an actionable recommendation based on risk level and failure mode."""
    if risk_level == "Low":
        return "Continue normal operation. No immediate action required."

    recommendation = ""
    if "Hydraulic" in failure_mode:
        recommendation = "Schedule hydraulic inspection"
    elif "Thermal" in failure_mode or "Temperature" in failure_mode:
        recommendation = "Check cooling systems and fluid levels"
    elif "Actuator" in failure_mode:
        recommendation = "Inspect actuator and load sensors"
    elif "Battery" in failure_mode:
        recommendation = "Perform battery diagnostics"
    elif "Vibration" in failure_mode:
        recommendation = "Inspect structural integrity and mounts"
    else:
        recommendation = "Perform general system inspection"

    if risk_level == "Medium":
        return f"{recommendation} within 7 days."
    elif risk_level == "High":
        return f"CRITICAL: {recommendation} immediately. Suspend operation if necessary."

    return "Monitor system parameters."


# ─────────────────────────────────────────────────────────────────
# Input Preprocessing (existing — from JSON request)
# ─────────────────────────────────────────────────────────────────
def preprocess_input(request) -> pd.DataFrame:
    """Converts request readings to DataFrame and sorts chronologically."""
    readings = [r.dict() for r in request.readings]
    df = pd.DataFrame(readings)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# Core Prediction Pipeline (existing — from DataFrame)
# ─────────────────────────────────────────────────────────────────
def predict_pipeline_from_df(vehicle_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Runs the prediction pipeline directly from a pandas DataFrame."""
    # 1. Generate Features
    df_features = generate_features(df)

    # 2. Select latest engineered row
    latest_row = df_features.iloc[[-1]].copy()

    # 3. Encode Vehicle_Type
    vehicle_type_raw = latest_row['Vehicle_Type'].iloc[0]
    matched_vehicle_note = None

    try:
        latest_row['Vehicle_Type'] = model_manager.vehicle_encoder.transform(latest_row['Vehicle_Type'])

    except ValueError:
        # ── Closest-Match Logic ──────────────────────────────────
        matched = _find_closest_match(vehicle_type_raw)

        if matched:
            # Use the matched vehicle type for ML prediction
            matched_vehicle_note = (
                f"Unknown vehicle type '{vehicle_type_raw}'. "
                f"Closest match used for ML prediction: '{matched}'. "
                f"Results may have reduced accuracy."
            )
            latest_row['Vehicle_Type'] = model_manager.vehicle_encoder.transform([matched])
        else:
            # No match found — fall back to LLM-only
            raw_telemetry = df.iloc[-1].to_dict()
            ai_feedback = generate_unseen_vehicle_feedback(vehicle_type_raw, raw_telemetry)
            return {
                "vehicle_id": vehicle_id,
                "health_status": "Unknown",
                "health_confidence": 0.0,
                "failure_mode": "Unknown",
                "failure_confidence": 0.0,
                "risk_level": "Unknown",
                "recommended_action": "ML model could not match vehicle type. See AI analysis.",
                "ai_analysis": ai_feedback,
                "matched_vehicle_note": f"No closest match found for '{vehicle_type_raw}'. LLM fallback used."
            }

    # 4. Run ML models
    expected_cols = model_manager.health_model.feature_names_in_
    X = latest_row[expected_cols]

    health_pred_encoded = model_manager.health_model.predict(X)
    health_probs        = model_manager.health_model.predict_proba(X)
    health_conf         = float(health_probs.max(axis=1)[0])

    failure_pred_encoded = model_manager.failure_model.predict(X)
    failure_probs        = model_manager.failure_model.predict_proba(X)
    failure_conf         = float(failure_probs.max(axis=1)[0])

    # 5. Decode outputs
    health_status = model_manager.health_encoder.inverse_transform(health_pred_encoded)[0]
    failure_mode  = model_manager.failure_encoder.inverse_transform(failure_pred_encoded)[0]

    # 6. Risk & recommendation
    risk_level         = calculate_risk_level(health_status)
    recommended_action = generate_recommendation(risk_level, failure_mode)

    # 7. LLM deep analysis (only on Medium/High risk)
    ai_analysis = None
    if risk_level != "Low":
        raw_telemetry = df.iloc[-1].to_dict()
        prediction_result = {
            "health_status": health_status,
            "failure_mode":  failure_mode,
            "risk_level":    risk_level
        }
        ai_analysis = generate_llm_feedback(vehicle_type_raw, prediction_result, raw_telemetry)

    return {
        "vehicle_id":          vehicle_id,
        "health_status":       health_status,
        "health_confidence":   round(health_conf, 2),
        "failure_mode":        failure_mode,
        "failure_confidence":  round(failure_conf, 2),
        "risk_level":          risk_level,
        "recommended_action":  recommended_action,
        "ai_analysis":         ai_analysis,
        "matched_vehicle_note": matched_vehicle_note
    }


# ─────────────────────────────────────────────────────────────────
# Existing pipeline entry point (JSON request)
# ─────────────────────────────────────────────────────────────────
def predict_pipeline(request) -> Dict[str, Any]:
    """Runs the complete prediction pipeline for a given request."""
    df = preprocess_input(request)
    return predict_pipeline_from_df(request.vehicle_id, df)


# ─────────────────────────────────────────────────────────────────
# NEW: Live Snapshot Simulation Pipeline
# User sends ONE reading — we auto-generate 300-row history.
# ─────────────────────────────────────────────────────────────────
def simulate_and_predict(vehicle_id: str, snapshot: dict) -> Dict[str, Any]:
    """
    Accepts a single sensor snapshot and auto-generates 300 rows of
    stable historical context around it. Runs the full prediction pipeline.

    This enables real-time IoT-style predictions without any CSV upload.
    """
    random.seed(42)
    base_time = datetime.now(timezone.utc) - timedelta(seconds=300)
    rows = []

    # Sensor noise bands (±% of value, or fixed minimum)
    noise_pct = 0.005  # 0.5% noise for stable history

    for i in range(299):
        t = base_time + timedelta(seconds=i)
        row = {
            "timestamp":           t.isoformat(),
            "Vehicle_Type":        snapshot["Vehicle_Type"],
            "Hydraulic_Pressure":  round(snapshot["Hydraulic_Pressure"]  * (1 + random.uniform(-noise_pct, noise_pct)), 2),
            "Oil_Temperature":     round(snapshot["Oil_Temperature"]      * (1 + random.uniform(-noise_pct, noise_pct)), 2),
            "Actuator_Angle":      round(snapshot["Actuator_Angle"]       * (1 + random.uniform(-noise_pct, noise_pct)), 2),
            "Actuator_Position":   round(snapshot["Actuator_Position"]    * (1 + random.uniform(-noise_pct, noise_pct)), 2),
            "Load_Weight":         round(snapshot["Load_Weight"]          * (1 + random.uniform(-noise_pct, noise_pct)), 2),
            "Hydraulic_Flow_Rate": round(snapshot["Hydraulic_Flow_Rate"]  * (1 + random.uniform(-noise_pct, noise_pct)), 2),
            "Vibration":           round(snapshot["Vibration"]            * (1 + random.uniform(-noise_pct, noise_pct)), 2),
            "Operating_Hours":     round(snapshot["Operating_Hours"] - (299 - i) / 3600.0, 4),
            "Battery_SOC":         round(min(100.0, snapshot["Battery_SOC"]          + random.uniform(-0.1, 0.1)), 2),
            "Battery_Temperature": round(snapshot["Battery_Temperature"]  * (1 + random.uniform(-noise_pct, noise_pct)), 2),
        }
        rows.append(row)

    # Append the actual user snapshot as the final (current) reading
    rows.append({
        "timestamp":           (base_time + timedelta(seconds=299)).isoformat(),
        "Vehicle_Type":        snapshot["Vehicle_Type"],
        "Hydraulic_Pressure":  snapshot["Hydraulic_Pressure"],
        "Oil_Temperature":     snapshot["Oil_Temperature"],
        "Actuator_Angle":      snapshot["Actuator_Angle"],
        "Actuator_Position":   snapshot["Actuator_Position"],
        "Load_Weight":         snapshot["Load_Weight"],
        "Hydraulic_Flow_Rate": snapshot["Hydraulic_Flow_Rate"],
        "Vibration":           snapshot["Vibration"],
        "Operating_Hours":     snapshot["Operating_Hours"],
        "Battery_SOC":         snapshot["Battery_SOC"],
        "Battery_Temperature": snapshot["Battery_Temperature"],
    })

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return predict_pipeline_from_df(vehicle_id, df)
