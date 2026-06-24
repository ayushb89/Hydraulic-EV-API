import pandas as pd
import random
from datetime import datetime, timezone, timedelta

def generate_csv(filename, vehicle_type, profile):
    base_time = datetime.now(timezone.utc)
    readings = []
    
    for i in range(300):
        current_time = base_time + timedelta(seconds=i)
        
        # Base normal values
        pressure = random.uniform(100.0, 150.0)
        oil_temp = random.uniform(70.0, 95.0)
        flow_rate = random.uniform(40.0, 60.0)
        vibration = random.uniform(0.5, 2.0)
        
        # Apply profile overrides for the last 50 readings
        if i >= 250:
            if profile == "multiple_failures":
                pressure = random.uniform(400.0, 450.0)      # High
                oil_temp = random.uniform(115.0, 130.0)      # High
                flow_rate = random.uniform(5.0, 10.0)        # Low
                vibration = random.uniform(4.0, 6.0)         # High
            elif profile == "only_oil_overheating":
                # Keep pressure and flow completely normal, ONLY spike oil temp
                pressure = random.uniform(100.0, 150.0)
                flow_rate = random.uniform(40.0, 60.0)
                oil_temp = random.uniform(125.0, 140.0)      # High
            elif profile == "only_hydraulic_leak":
                # Massive drop in pressure and flow, but temperature remains normal
                pressure = random.uniform(10.0, 30.0)        # Dangerously Low
                flow_rate = random.uniform(5.0, 15.0)        # Dangerously Low
                oil_temp = random.uniform(70.0, 95.0)        # Normal
        
        reading = {
            "timestamp": current_time.isoformat(),
            "Vehicle_Type": vehicle_type,
            "Hydraulic_Pressure": round(pressure, 2),
            "Oil_Temperature": round(oil_temp, 2),
            "Actuator_Angle": round(random.uniform(10.0, 50.0), 2),
            "Actuator_Position": round(random.uniform(500.0, 1500.0), 2),
            "Load_Weight": round(random.uniform(2000.0, 4500.0), 2),
            "Hydraulic_Flow_Rate": round(flow_rate, 2),
            "Vibration": round(vibration, 2),
            "Operating_Hours": round(5000.0 + (i / 3600.0), 4),
            "Battery_SOC": round(random.uniform(40.0, 90.0), 2),
            "Battery_Temperature": round(random.uniform(25.0, 45.0), 2)
        }
        readings.append(reading)
        
    df = pd.DataFrame(readings)
    df.to_csv(filename, index=False)
    print(f"Created: {filename} ({profile} | {vehicle_type})")

def main():
    # 1. Normal Operation (Valid Vehicle)
    generate_csv("dataset_1_normal_operation.csv", "DumpTruck", "normal")
    
    # 2. Multiple Simultaneous Failures (Valid Vehicle)
    generate_csv("dataset_2_multiple_failures.csv", "WheelLoader", "multiple_failures")
    
    # 3. ONLY Oil Overheating - Single Fault (Valid Vehicle)
    generate_csv("dataset_3_single_fault_oil_overheating.csv", "Forklift", "only_oil_overheating")
    
    # 4. ONLY Hydraulic Leak - Single Fault (Valid Vehicle)
    generate_csv("dataset_4_single_fault_hydraulic_leak.csv", "Backhoe", "only_hydraulic_leak")
    
    # 5. Unregistered Vehicle (Fallback to LLM)
    generate_csv("dataset_5_unregistered_vehicle.csv", "Crane", "multiple_failures")

if __name__ == "__main__":
    main()
