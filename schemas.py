from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


# ─────────────────────────────────────────────
# Hydraulic Vehicle Schemas (existing)
# ─────────────────────────────────────────────

class ReadingItem(BaseModel):
    timestamp: datetime
    Vehicle_Type: str
    Hydraulic_Pressure: float
    Oil_Temperature: float
    Actuator_Angle: float
    Actuator_Position: float
    Load_Weight: float
    Hydraulic_Flow_Rate: float
    Vibration: float
    Operating_Hours: float
    Battery_SOC: float
    Battery_Temperature: float

class PredictionRequest(BaseModel):
    vehicle_id: str
    readings: List[ReadingItem]

    @validator("readings")
    def validate_readings_not_empty(cls, v):
        if not v:
            raise ValueError("Readings list cannot be empty")
        return v

class PredictionResponse(BaseModel):
    vehicle_id: str
    health_status: str
    health_confidence: float
    failure_mode: str
    failure_confidence: float
    risk_level: str
    recommended_action: str
    ai_analysis: Optional[str] = None
    matched_vehicle_note: Optional[str] = None  # Set when closest-match was used

class HealthResponse(BaseModel):
    status: str


# ─────────────────────────────────────────────
# Live Simulation Schema (NEW)
# Single-snapshot endpoint — no CSV or 300 rows needed
# ─────────────────────────────────────────────

class LiveSnapshotRequest(BaseModel):
    """
    A single real-time sensor snapshot.
    The backend auto-generates 300-row historical context internally.
    """
    vehicle_id: str
    Vehicle_Type: str
    Hydraulic_Pressure: float
    Oil_Temperature: float
    Actuator_Angle: float
    Actuator_Position: float
    Load_Weight: float
    Hydraulic_Flow_Rate: float
    Vibration: float
    Operating_Hours: float
    Battery_SOC: float
    Battery_Temperature: float


# ─────────────────────────────────────────────
# EV Car Schemas (NEW)
# ─────────────────────────────────────────────

class EVSnapshotRequest(BaseModel):
    """
    A single real-time EV sensor snapshot.
    Used by the /predict/ev endpoint.
    """
    vehicle_id: str
    Vehicle_Type: str          # e.g. "Tesla Model 3", "Rivian R1T"
    Motor_RPM: float           # Traction motor speed (0–18000 RPM)
    Motor_Temp: float          # Stator temperature (°C)
    Inverter_Temp: float       # Inverter/controller temperature (°C)
    Motor_Torque: float        # Nm (negative = regen braking)
    Phase_Current: float       # RMS phase current (Amps)
    Battery_SOC: float         # State of Charge (%)
    Battery_Temperature: float # Pack temperature (°C)
    Vehicle_Speed: float       # km/h
    Operating_Hours: float     # Total powered hours

class EVPredictionResponse(BaseModel):
    vehicle_id: str
    vehicle_type: str
    health_status: str
    risk_level: str
    failure_mode: str
    health_confidence: float
    failure_confidence: float
    recommended_action: str
    analysis_type: str = "ML + AI Analysis (EV Module)"
    ai_analysis: Optional[str] = None
