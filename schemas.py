from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

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
    ai_analysis: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
