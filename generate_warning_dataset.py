"""
Generates a Warning-level dataset by simulating a gradual oil temperature drift.

Key insight (from model probing):
  - Constant elevated values jump straight to Critical (model trained on hard faults)
  - Gradual DRIFT from Normal -> slightly elevated triggers Warning
  - The rolling mean/delta features detect the upward trend and flag Warning

dataset_8_warning_oil_temp_rising.csv:
  - First 200 rows: Oil Temp stable at 55C (Normal)
  - Last 100 rows: Oil Temp gradually climbs to 65C (Warning trigger)
  - Health Status: Warning | Risk: Medium
"""

import pandas as pd
import random
from datetime import datetime, timezone, timedelta

random.seed(77)

def generate_warning_csv(filename, vehicle_type):
    base_time = datetime.now(timezone.utc)
    readings = []

    for i in range(300):
        current_time = base_time + timedelta(seconds=i)

        # Gradual oil temperature drift in last 100 rows
        # rows 0-199: stable normal (55C), rows 200-299: drift up to 65C
        frac = max(0.0, (i - 200) / 100.0)   # 0.0 -> 1.0 over last 100 rows
        oil_temp = 55.0 + frac * 10.0        # drifts from 55C to 65C

        # Also slightly drift vibration upward as a secondary signal
        vibration_drift = frac * 0.8          # 0 -> 0.8 drift added
        vibration = 0.3 + vibration_drift + random.uniform(-0.05, 0.05)

        # Everything else stays stable and healthy
        pressure   = 130.0 + random.uniform(-1.5, 1.5)
        flow_rate  = 50.0  + random.uniform(-1.0, 1.0)
        batt_temp  = 33.0  + random.uniform(-0.5, 0.5)
        batt_soc   = 80.0  + random.uniform(-1.0, 1.0)
        load       = 1800.0 + random.uniform(-100.0, 100.0)
        act_pos    = 800.0  + random.uniform(-10.0, 10.0)
        act_angle  = 25.0   + random.uniform(-1.0, 1.0)

        reading = {
            "timestamp":           current_time.isoformat(),
            "Vehicle_Type":        vehicle_type,
            "Hydraulic_Pressure":  round(pressure, 2),
            "Oil_Temperature":     round(oil_temp + random.uniform(-0.3, 0.3), 2),
            "Actuator_Angle":      round(act_angle, 2),
            "Actuator_Position":   round(act_pos, 2),
            "Load_Weight":         round(load, 2),
            "Hydraulic_Flow_Rate": round(flow_rate, 2),
            "Vibration":           round(vibration, 2),
            "Operating_Hours":     round(1200.0 + i / 3600.0, 4),
            "Battery_SOC":         round(batt_soc, 2),
            "Battery_Temperature": round(batt_temp, 2),
        }
        readings.append(reading)

    df = pd.DataFrame(readings)
    df.to_csv(filename, index=False)
    print(f"Created: {filename}  (warning | {vehicle_type})")
    print(f"  Oil Temp range: {df['Oil_Temperature'].min():.1f}C -> {df['Oil_Temperature'].max():.1f}C")
    print(f"  Vibration range: {df['Vibration'].min():.2f} -> {df['Vibration'].max():.2f}")


if __name__ == "__main__":
    generate_warning_csv("dataset_8_warning_oil_temp_rising.csv", "WheelLoader")
    print("\nDone!")
