"""
test_ev_realistic.py
Tests the /predict/ev endpoint using sensor values directly from
real Warning/Critical rows in EV_Car_Brake_Telemetry_v3_FINAL.csv
"""
import requests

API = "http://127.0.0.1:8000"
HEADERS = {
    "X-API-Key": "5611ef24790093d63ba87a8ba22fd6e12e6e0a4da34d2290b402bb8062efd261",
    "Content-Type": "application/json"
}

# Row 85002 from actual dataset (Warning, Air_In_Hydraulic_Line)
warning_real = {
    "vehicle_id": "REAL_WARNING_001",
    "Vehicle_Type": "EV Car",
    "Brake_Hydraulic_Pressure_bar": 9.9,
    "Brake_Fluid_Temperature_C": 80.92,
    "Brake_Pedal_Position_pct": 0.0,
    "Brake_Line_Pressure_bar": 6.92,
    "Brake_Fluid_Level_pct": 78.16,
    "ABS_Activation_Frequency": 3.9,
    "Vibration_g": 0.305,
    "Vehicle_Speed_kmh": 118.52,
    "Acceleration_ms2": -1.03,
    "Operating_Hours": 273.611,
    "Battery_SOC": 91.5,
    "Battery_Temperature": 23.11
}

# Row 99985 from actual dataset (Critical, Multiple_Simultaneous_Failures)
critical_real = {
    "vehicle_id": "REAL_CRITICAL_001",
    "Vehicle_Type": "EV Car",
    "Brake_Hydraulic_Pressure_bar": 0.61,
    "Brake_Fluid_Temperature_C": 107.48,
    "Brake_Pedal_Position_pct": 5.89,
    "Brake_Line_Pressure_bar": 0.47,
    "Brake_Fluid_Level_pct": 63.99,
    "ABS_Activation_Frequency": 5.0,
    "Vibration_g": 0.661,
    "Vehicle_Speed_kmh": 36.43,
    "Acceleration_ms2": -0.44,
    "Operating_Hours": 277.773,
    "Battery_SOC": 88.29,
    "Battery_Temperature": 24.48
}

# Row 2 from actual dataset (Normal)
normal_real = {
    "vehicle_id": "REAL_NORMAL_001",
    "Vehicle_Type": "EV Car",
    "Brake_Hydraulic_Pressure_bar": 10.21,
    "Brake_Fluid_Temperature_C": 22.8,
    "Brake_Pedal_Position_pct": 16.46,
    "Brake_Line_Pressure_bar": 10.2,
    "Brake_Fluid_Level_pct": 99.89,
    "ABS_Activation_Frequency": 0.2,
    "Vibration_g": 0.066,
    "Vehicle_Speed_kmh": 21.02,
    "Acceleration_ms2": -1.2,
    "Operating_Hours": 250.0,
    "Battery_SOC": 92.0,
    "Battery_Temperature": 21.0
}

scenarios = [
    ("NORMAL  (from CSV row 2)",              normal_real,   "Normal"),
    ("WARNING (from CSV row 85002)",          warning_real,  "Warning"),
    ("CRITICAL (from CSV row 99985)",         critical_real, "Critical"),
]

print("=" * 65)
print("  EV BRAKE MODULE -- Real-Data Validation Test")
print("=" * 65)

for label, payload, expected in scenarios:
    r = requests.post(f"{API}/predict/ev", headers=HEADERS, json=payload, timeout=30)
    d = r.json()
    ok = "PASS" if d["health_status"] == expected else f"FAIL (expected {expected})"
    print(f"\n{label}")
    print(f"  Expected:     {expected}")
    print(f"  Predicted:    {d['health_status']}  [{ok}]")
    print(f"  Risk:         {d['risk_level']}")
    print(f"  Failure:      {d['failure_mode']}")
    print(f"  Health Conf:  {d['health_confidence']}")
    print(f"  Fail Conf:    {d['failure_confidence']}")
    print(f"  Analysis:     {d['analysis_type']}")
    print(f"  Action:       {d['recommended_action'][:80]}")
    if d.get("ai_analysis"):
        print(f"  --- LLM Report ---")
        print(d["ai_analysis"][:500])

print("\n" + "=" * 65)
