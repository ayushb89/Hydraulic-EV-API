from fastapi import FastAPI, HTTPException, status, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import logging
from contextlib import asynccontextmanager
import os
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

from schemas import PredictionRequest, PredictionResponse, HealthResponse
from models import model_manager
from utils import predict_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    logger.info("Loading ML models and encoders...")
    try:
        model_manager.load_models()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise e
    yield
    # Cleanup on shutdown
    logger.info("Shutting down model service.")

app = FastAPI(
    title="Hydraulic EV Predictive Maintenance API",
    description="API for predicting health status and failure modes of Hydraulic EV equipment.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the health status of the API."""
    return HealthResponse(status="healthy")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
    """
    Predicts health status and failure mode based on historical telemetry readings.
    Requires chronological readings to compute rolling and delta features correctly.
    """
    logger.info(f"Received prediction request for vehicle: {request.vehicle_id} with {len(request.readings)} readings")
    
    try:
        # Check if we have enough readings for 300s rolling features
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction processing."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
