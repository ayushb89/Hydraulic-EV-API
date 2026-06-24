import requests
import random
import json
from datetime import datetime, timezone, timedelta

def generate_hydraulic_failure_payload():
    base_time = datetime.now(timezone.utc)
    readings = []
    
    for i in range(300):
        current_time = base_time + timedelta(seconds=i)
        
        # We start normal, but in the last 50 readings, things go very wrong
        if i < 250:
            pressure = random.uniform(250.0, 320.0)      # Normal pressure
            oil_temp = random.uniform(70.0, 95.0)        # Normal temperature
            flow_rate = random.uniform(20.0, 60.0)       # Normal flow
        else:
            # HYDRAULIC FAILURE SIMULATION (Extreme Pressure & Temperature)
            pressure = random.uniform(400.0, 450.0)      # DANGEROUSLY HIGH PRESSURE
            oil_temp = random.uniform(115.0, 130.0)      # OVERHEATING OIL
            flow_rate = random.uniform(5.0, 10.0)        # SEVERE BLOCKAGE / LOW FLOW
            
        reading = {
            "timestamp": current_time.isoformat(),
            "Vehicle_Type": "WheelLoader",                   # Valid Vehicle Type
            "Hydraulic_Pressure": round(pressure, 2),
            "Oil_Temperature": round(oil_temp, 2),
            "Actuator_Angle": round(random.uniform(10.0, 50.0), 2),
            "Actuator_Position": round(random.uniform(500.0, 1500.0), 2),
            "Load_Weight": round(random.uniform(2000.0, 4500.0), 2),
            "Hydraulic_Flow_Rate": round(flow_rate, 2),
            "Vibration": round(random.uniform(0.5, 5.0), 2),
            "Operating_Hours": round(5000.0 + (i / 3600.0), 4),
            "Battery_SOC": round(random.uniform(40.0, 90.0), 2),
            "Battery_Temperature": round(random.uniform(25.0, 45.0), 2)
        }
        readings.append(reading)
        
    return {
        "vehicle_id": "EV_002", # Different Vehicle ID
        "readings": readings
    }

def main():
    url = "https://hydraulic-ev-api.onrender.com/predict"
    print("Generating 300 readings for EV_002 (Simulating Hydraulic Failure)...")
    payload = generate_hydraulic_failure_payload()
    
    headers = {
        "X-API-Key": "5611ef24790093d63ba87a8ba22fd6e12e6e0a4da34d2290b402bb8062efd261",
        "Content-Type": "application/json"
    }
    print(f"Sending POST request to {url}...")
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        print("\n--- API Response ---")
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
            
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    main()
