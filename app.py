from fastapi import FastAPI, HTTPException, status, Security, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
import os
import io
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "default-dev-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )

from schemas import (
    PredictionRequest, PredictionResponse, HealthResponse,
    LiveSnapshotRequest, EVSnapshotRequest, EVPredictionResponse
)
from models import model_manager
from utils import predict_pipeline, predict_pipeline_from_df, simulate_and_predict
from llm_diagnostics import generate_ev_feedback
from ev_models import ev_model_manager
from ev_utils import predict_ev_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models and encoders...")
    try:
        model_manager.load_models()
        logger.info("Hydraulic ML models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load hydraulic models: {str(e)}")
        raise e

    logger.info("Loading EV ML models...")
    try:
        ev_model_manager.load_models()
        logger.info("EV ML models loaded successfully.")
    except Exception as e:
        logger.warning(f"EV models not found or failed to load: {str(e)}. EV endpoint will use LLM-only mode.")

    yield
    logger.info("Shutting down model service.")

app = FastAPI(
    title="Hydraulic & EV Predictive Maintenance API",
    description=(
        "API for predicting health status and failure modes of Hydraulic EV equipment "
        "and Electric Passenger Vehicles. Supports real-time single-snapshot predictions, "
        "CSV batch uploads, and EV brake-specific telemetry analysis with full ML models."
    ),
    version="2.1.0",
    lifespan=lifespan
)

# CORS — allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the health status of the API."""
    return HealthResponse(status="healthy")


# ─────────────────────────────────────────────────────────────────
# Hydraulic: 300-Row JSON Prediction (existing)
# ─────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
    """
    Predicts health status and failure mode based on 300 historical telemetry readings.
    Requires chronological readings to compute rolling and delta features correctly.
    """
    logger.info(f"Received prediction request for vehicle: {request.vehicle_id} with {len(request.readings)} readings")

    try:
        if len(request.readings) < 300:
            raise HTTPException(
                status_code=400,
                detail="At least 300 historical readings are required."
            )
        result = predict_pipeline(request)
        return PredictionResponse(**result)

    except HTTPException as e:
        raise e
    except ValueError as e:
        logger.warning(f"Validation Error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred during prediction processing.")


# ─────────────────────────────────────────────────────────────────
# Hydraulic: CSV Upload Prediction (existing)
# ─────────────────────────────────────────────────────────────────
@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(
    vehicle_id: str = Form(...),
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """
    Predicts health status from an uploaded CSV file containing telemetry readings.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        if len(df) < 300:
            raise HTTPException(status_code=400,
                                detail="At least 300 historical readings are required in the CSV.")

        result = predict_pipeline_from_df(vehicle_id, df)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV format or processing error: {str(e)}")


# ─────────────────────────────────────────────────────────────────
# Hydraulic: Live Single-Snapshot Prediction (NEW)
# No CSV upload needed — backend auto-generates 300-row history
# ─────────────────────────────────────────────────────────────────
@app.post("/predict/live", response_model=PredictionResponse)
def predict_live(request: LiveSnapshotRequest, api_key: str = Depends(get_api_key)):
    """
    Accepts a SINGLE real-time sensor snapshot and auto-generates 300 rows
    of stable historical context internally.

    Perfect for IoT/real-time use cases where no CSV is available.
    The backend simulates realistic historical readings around the snapshot
    and runs the full ML prediction pipeline.
    """
    logger.info(f"Live snapshot prediction request for vehicle: {request.vehicle_id} ({request.Vehicle_Type})")

    try:
        snapshot = {
            "Vehicle_Type":        request.Vehicle_Type,
            "Hydraulic_Pressure":  request.Hydraulic_Pressure,
            "Oil_Temperature":     request.Oil_Temperature,
            "Actuator_Angle":      request.Actuator_Angle,
            "Actuator_Position":   request.Actuator_Position,
            "Load_Weight":         request.Load_Weight,
            "Hydraulic_Flow_Rate": request.Hydraulic_Flow_Rate,
            "Vibration":           request.Vibration,
            "Operating_Hours":     request.Operating_Hours,
            "Battery_SOC":         request.Battery_SOC,
            "Battery_Temperature": request.Battery_Temperature,
        }
        result = simulate_and_predict(request.vehicle_id, snapshot)
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Live prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Live prediction failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────
# EV Car: Single-Snapshot Prediction (NEW)
# Uses dedicated EV ML models + EV-specific LLM analysis
# ─────────────────────────────────────────────────────────────────
@app.post("/predict/ev", response_model=EVPredictionResponse)
def predict_ev(request: EVSnapshotRequest, api_key: str = Depends(get_api_key)):
    """
    Predicts health status for an Electric Vehicle using EV-specific
    sensor telemetry (Motor RPM, Inverter Temp, Motor Torque, etc.).

    Uses dedicated EV ML models if available, falls back to AI-only analysis.
    Returns EV-specific failure modes and maintenance recommendations.
    """
    logger.info(f"EV prediction request for vehicle: {request.vehicle_id} ({request.Vehicle_Type})")

    try:
        telemetry = {
            "Vehicle_Type":                  request.Vehicle_Type,
            "Brake_Hydraulic_Pressure_bar":  request.Brake_Hydraulic_Pressure_bar,
            "Brake_Fluid_Temperature_C":     request.Brake_Fluid_Temperature_C,
            "Brake_Pedal_Position_pct":      request.Brake_Pedal_Position_pct,
            "Brake_Line_Pressure_bar":       request.Brake_Line_Pressure_bar,
            "Brake_Fluid_Level_pct":         request.Brake_Fluid_Level_pct,
            "ABS_Activation_Frequency":      request.ABS_Activation_Frequency,
            "Vibration_g":                   request.Vibration_g,
            "Vehicle_Speed_kmh":             request.Vehicle_Speed_kmh,
            "Acceleration_ms2":              request.Acceleration_ms2,
            "Operating_Hours":               request.Operating_Hours,
            "Battery_SOC":                   request.Battery_SOC,
            "Battery_Temperature":           request.Battery_Temperature,
        }

        result = predict_ev_pipeline(request.vehicle_id, telemetry)
        return EVPredictionResponse(**result)

    except Exception as e:
        logger.error(f"EV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EV prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
