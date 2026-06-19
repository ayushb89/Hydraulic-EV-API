import pandas as pd
from typing import Dict, Any

from models import model_manager
from feature_engineering import generate_features
from llm_diagnostics import generate_llm_feedback, generate_unseen_vehicle_feedback

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

def preprocess_input(request) -> pd.DataFrame:
    """Converts request readings to DataFrame and sorts chronologically."""
    # Convert readings to list of dicts
    readings = [r.dict() for r in request.readings]
    df = pd.DataFrame(readings)
    
    # Ensure timestamp is datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def predict_pipeline_from_df(vehicle_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Runs the prediction pipeline directly from a pandas DataFrame."""
    # 2. Generate Features
    df_features = generate_features(df)
    
    # 3. Select latest engineered row
    latest_row = df_features.iloc[[-1]].copy()
    
    # 4. Encode Vehicle_Type using vehicle_encoder.pkl
    vehicle_type_raw = latest_row['Vehicle_Type'].iloc[0]
    try:
        latest_row['Vehicle_Type'] = model_manager.vehicle_encoder.transform(latest_row['Vehicle_Type'])
    except ValueError as e:
        # Unseen vehicle type fallback using LLM
        raw_telemetry = df.iloc[-1].to_dict()
        ai_feedback = generate_unseen_vehicle_feedback(vehicle_type_raw, raw_telemetry)
        return {
            "vehicle_id": vehicle_id,
            "health_status": "Unknown",
            "health_confidence": 0.0,
            "failure_mode": "Unknown",
            "failure_confidence": 0.0,
            "risk_level": "Unknown",
            "recommended_action": "ML model rejected unseen vehicle type.",
            "ai_analysis": ai_feedback
        }
    
    # Ensure columns match exactly what the model expects
    expected_cols = model_manager.health_model.feature_names_in_
    X = latest_row[expected_cols]
    
    # 5. Run health_model prediction
    health_pred_encoded = model_manager.health_model.predict(X)
    health_probs = model_manager.health_model.predict_proba(X)
    health_conf = float(health_probs.max(axis=1)[0])
    
    # 6. Run failure_model prediction
    failure_pred_encoded = model_manager.failure_model.predict(X)
    failure_probs = model_manager.failure_model.predict_proba(X)
    failure_conf = float(failure_probs.max(axis=1)[0])
    
    # 7. Decode outputs using encoders
    health_status = model_manager.health_encoder.inverse_transform(health_pred_encoded)[0]
    failure_mode = model_manager.failure_encoder.inverse_transform(failure_pred_encoded)[0]
    
    # 8. Calculate risk level and recommendations
    risk_level = calculate_risk_level(health_status)
    recommended_action = generate_recommendation(risk_level, failure_mode)
    
    # 9. Ask LLM for deep analysis if there is an issue
    ai_analysis = None
    if risk_level != "Low":
        raw_telemetry = df.iloc[-1].to_dict()
        prediction_result = {
            "health_status": health_status,
            "failure_mode": failure_mode,
            "risk_level": risk_level
        }
        ai_analysis = generate_llm_feedback(vehicle_type_raw, prediction_result, raw_telemetry)
    
    return {
        "vehicle_id": vehicle_id,
        "health_status": health_status,
        "health_confidence": round(health_conf, 2),
        "failure_mode": failure_mode,
        "failure_confidence": round(failure_conf, 2),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "ai_analysis": ai_analysis
    }

def predict_pipeline(request) -> Dict[str, Any]:
    """Runs the complete prediction pipeline for a given request."""
    # 1. Preprocess Input
    df = preprocess_input(request)
    return predict_pipeline_from_df(request.vehicle_id, df)
