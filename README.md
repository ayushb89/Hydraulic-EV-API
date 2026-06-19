# Hydraulic EV Predictive Maintenance System

A production-ready FastAPI backend for predicting health status and failure modes of Hydraulic EV equipment.

## Features

* **Real-time Inference**: Accepts historical telemetry data to predict equipment health and failure modes.
* **Feature Engineering**: Accurately recreates complex rolling, delta, thermal, and hydraulic features required by the V2 ML models.
* **Risk Assessment**: Automatically calculates risk levels (Low/Medium/High) and generates actionable maintenance recommendations.
* **Robust Validation**: Ensures data integrity with comprehensive validation via Pydantic.

## Project Structure

```text
Hydraulic_EV_API/
├── app.py                  # Main FastAPI application
├── feature_engineering.py  # V2 feature engineering logic
├── models.py               # Model loading and management
├── schemas.py              # Pydantic data models
├── utils.py                # Prediction pipeline and helper functions
├── requirements.txt        # Python dependencies
└── models/                 # Pre-trained ML models and encoders
    ├── health_model.pkl
    ├── failure_model.pkl
    ├── vehicle_encoder.pkl
    ├── health_encoder.pkl
    └── failure_encoder.pkl
```

## Getting Started

### Prerequisites

* Python 3.9+
* Pre-trained ML models placed in the `models/` directory

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd Hydraulic_EV_API
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

Start the FastAPI server locally using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, you can access:

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Health Check
* **GET** `/health`
* **Description**: Verifies that the service is running.
* **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

### 2. Predict
* **POST** `/predict`
* **Description**: Predicts the health status and failure mode for a given vehicle based on historical telemetry readings.
* **Request Body**:
  ```json
  {
    "vehicle_id": "EV_001",
    "readings": [
      {
        "timestamp": "2026-06-18T10:00:00Z",
        "Vehicle_Type": "Forklift",
        "Hydraulic_Pressure": 280,
        ...
      }
    ]
  }
  ```
* **Response**:
  ```json
  {
    "vehicle_id": "EV_001",
    "health_status": "Warning",
    "health_confidence": 0.94,
    "failure_mode": "Hydraulic_System_Failure",
    "failure_confidence": 0.88,
    "risk_level": "Medium",
    "recommended_action": "Schedule hydraulic inspection within 7 days."
  }
  ```
