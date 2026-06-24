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
# EV Car Schemas — Brake Telemetry
# ─────────────────────────────────────────────

class EVSnapshotRequest(BaseModel):
    """
    A single real-time EV brake sensor snapshot.
    Used by the /predict/ev endpoint.
    Sensor columns match EV_Car_Brake_Telemetry_v3_FINAL.csv.
    """
    vehicle_id: str
    Vehicle_Type: str                    # e.g. "Tesla Model 3", "Rivian R1T"
    Brake_Hydraulic_Pressure_bar: float  # Master cylinder pressure (bar)
    Brake_Fluid_Temperature_C: float     # Fluid temperature at caliper (°C)
    Brake_Pedal_Position_pct: float      # Pedal travel 0–100 (%)
    Brake_Line_Pressure_bar: float       # Downstream line pressure (bar)
    Brake_Fluid_Level_pct: float         # Reservoir level 0–100 (%)
    ABS_Activation_Frequency: float      # ABS events per second (Hz)
    Vibration_g: float                   # Caliper/rotor vibration (g)
    Vehicle_Speed_kmh: float             # Vehicle speed (km/h)
    Acceleration_ms2: float              # Longitudinal acceleration (m/s²)
    Operating_Hours: float               # Total powered hours
    Battery_SOC: float                   # State of Charge (%)
    Battery_Temperature: float           # Battery pack temperature (°C)

class EVPredictionResponse(BaseModel):
    vehicle_id: str
    vehicle_type: str
    health_status: str
    risk_level: str
    failure_mode: str
    health_confidence: float
    failure_confidence: float
    recommended_action: str
    analysis_type: str = "ML + AI Analysis (EV Brake Module)"
    ai_analysis: Optional[str] = None
