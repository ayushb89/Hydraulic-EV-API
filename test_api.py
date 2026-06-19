import requests
import random
import json
from datetime import datetime, timezone, timedelta

def generate_test_payload():
    base_time = datetime.now(timezone.utc)
    readings = []
    
    for i in range(300):
        # Time increases by 1 second for each reading to ensure chronological order
        current_time = base_time + timedelta(seconds=i)
        
        reading = {
            "timestamp": current_time.isoformat(),
            "Vehicle_Type": "Forklift",
            "Hydraulic_Pressure": round(random.uniform(250.0, 320.0), 2),
            "Oil_Temperature": round(random.uniform(70.0, 95.0), 2),
            "Actuator_Angle": round(random.uniform(10.0, 50.0), 2),
            "Actuator_Position": round(random.uniform(500.0, 1500.0), 2),
            "Load_Weight": round(random.uniform(500.0, 2500.0), 2),
            "Hydraulic_Flow_Rate": round(random.uniform(20.0, 60.0), 2),
            "Vibration": round(random.uniform(0.5, 5.0), 2),
            "Operating_Hours": round(5000.0 + (i / 3600.0), 4),
            "Battery_SOC": round(random.uniform(40.0, 90.0), 2),
            "Battery_Temperature": round(random.uniform(25.0, 45.0), 2)
        }
        readings.append(reading)
        
    return {
        "vehicle_id": "EV_001",
        "readings": readings
    }

def main():
    url = "http://127.0.0.1:8000/predict"
    print("Generating 300 realistic historical readings...")
    payload = generate_test_payload()
    
    print(f"Sending POST request to {url}...")
    try:
        response = requests.post(url, json=payload)
        
        print("\n--- API Response ---")
        print(f"Status Code: {response.status_code}")
        
        try:
            print("Response Body:")
            print(json.dumps(response.json(), indent=2))
        except ValueError:
            print(f"Raw Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    main()
