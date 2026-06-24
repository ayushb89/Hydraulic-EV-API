"""
Generates two healthy-state CSV datasets for testing.
Sensor ranges are calibrated by probing the actual ML model.

Root cause of previous failure:
  - Oil Temperature >= 70C triggers Oil_Overheating -> Critical
  - Operating Hours >= 5000 triggers Multiple_Simultaneous_Failures -> Critical
  - HIGH VARIANCE in readings causes roll_std features to spike -> Warning/Critical
  
Fix: Use very tight, stable values with tiny noise to keep rolling std near zero.

  dataset_6_healthy_perfect.csv  : Near-constant nominal values. Normal / Low Risk.
  dataset_7_moderate_healthy.csv : Mild, controlled variation. Normal / Low Risk.
"""

import pandas as pd
import random
from datetime import datetime, timezone, timedelta


def generate_csv(filename, vehicle_type, profile):
    base_time = datetime.now(timezone.utc)
    readings = []
    random.seed(42)  # reproducible output

    for i in range(300):
        current_time = base_time + timedelta(seconds=i)

        if profile == "perfect_healthy":
            # Near-constant stable values — minimal variance to keep roll_std near 0
            pressure   = 130.0  + random.uniform(-1.0, 1.0)   # Tight band ±1
            oil_temp   = 55.0   + random.uniform(-1.0, 1.0)   # Well below 70C threshold
            flow_rate  = 50.0   + random.uniform(-0.5, 0.5)   # Stable flow
            vibration  = 0.20   + random.uniform(-0.05, 0.05) # Very low vibration
            batt_temp  = 33.0   + random.uniform(-0.5, 0.5)   # Stable battery temp
            batt_soc   = 82.0   + random.uniform(-1.0, 1.0)   # High, stable SOC
            load       = 1500.0 + random.uniform(-50.0, 50.0) # Steady load
            act_pos    = 800.0  + random.uniform(-5.0, 5.0)   # Barely moving
            act_angle  = 25.0   + random.uniform(-0.5, 0.5)   # Locked angle
            op_hours   = 1200.0 + (i / 3600.0)                # Low hours

        elif profile == "moderate_healthy":
            # Slightly more variation but carefully controlled to stay Normal
            pressure   = 130.0  + random.uniform(-8.0, 8.0)   # ±8 — normal variation
            oil_temp   = 55.0   + random.uniform(-5.0, 5.0)   # Stays 50–60, well under 70C
            flow_rate  = 50.0   + random.uniform(-4.0, 4.0)   # ±4 — normal
            vibration  = 0.35   + random.uniform(-0.1, 0.1)   # Low, controlled
            batt_temp  = 34.0   + random.uniform(-2.0, 2.0)   # Stable
            batt_soc   = 75.0   + random.uniform(-5.0, 5.0)   # Moderate, healthy
            load       = 2000.0 + random.uniform(-200.0, 200.0)
            act_pos    = 800.0  + random.uniform(-50.0, 50.0)
            act_angle  = 25.0   + random.uniform(-3.0, 3.0)
            op_hours   = 2500.0 + (i / 3600.0)                # Mid-life, under 5000

        reading = {
            "timestamp":           current_time.isoformat(),
            "Vehicle_Type":        vehicle_type,
            "Hydraulic_Pressure":  round(pressure, 2),
            "Oil_Temperature":     round(oil_temp, 2),
            "Actuator_Angle":      round(act_angle, 2),
            "Actuator_Position":   round(act_pos, 2),
            "Load_Weight":         round(load, 2),
            "Hydraulic_Flow_Rate": round(flow_rate, 2),
            "Vibration":           round(vibration, 2),
            "Operating_Hours":     round(op_hours, 4),
            "Battery_SOC":         round(batt_soc, 2),
            "Battery_Temperature": round(batt_temp, 2),
        }
        readings.append(reading)

    df = pd.DataFrame(readings)
    df.to_csv(filename, index=False)
    print(f"Created: {filename}  ({profile} | {vehicle_type})")


if __name__ == "__main__":
    generate_csv("dataset_6_healthy_perfect.csv", "Forklift", "perfect_healthy")
    generate_csv("dataset_7_moderate_healthy.csv", "DumpTruck", "moderate_healthy")
    print("\nDone!")
