import requests, json, time

time.sleep(3)  # give server a moment

API = "http://127.0.0.1:8000"
HEADERS = {
    "X-API-Key": "5611ef24790093d63ba87a8ba22fd6e12e6e0a4da34d2290b402bb8062efd261",
    "Content-Type": "application/json"
}

normal = {
    "vehicle_id": "TESLA_001",
    "Vehicle_Type": "Tesla Model 3",
    "Brake_Hydraulic_Pressure_bar": 12.5,
    "Brake_Fluid_Temperature_C": 28.0,
    "Brake_Pedal_Position_pct": 15.0,
    "Brake_Line_Pressure_bar": 12.3,
    "Brake_Fluid_Level_pct": 99.5,
    "ABS_Activation_Frequency": 0.3,
    "Vibration_g": 0.07,
    "Vehicle_Speed_kmh": 35.0,
    "Acceleration_ms2": -1.2,
    "Operating_Hours": 250.0,
    "Battery_SOC": 92.0,
    "Battery_Temperature": 22.0
}

critical = {
    "vehicle_id": "RIVIAN_002",
    "Vehicle_Type": "Rivian R1T",
    "Brake_Hydraulic_Pressure_bar": 3.0,
    "Brake_Fluid_Temperature_C": 112.0,
    "Brake_Pedal_Position_pct": 5.0,
    "Brake_Line_Pressure_bar": 2.8,
    "Brake_Fluid_Level_pct": 58.0,
    "ABS_Activation_Frequency": 7.5,
    "Vibration_g": 0.55,
    "Vehicle_Speed_kmh": 45.0,
    "Acceleration_ms2": -0.5,
    "Operating_Hours": 277.0,
    "Battery_SOC": 88.0,
    "Battery_Temperature": 24.5
}

warning = {
    "vehicle_id": "BYD_003",
    "Vehicle_Type": "BYD Atto 3",
    "Brake_Hydraulic_Pressure_bar": 18.0,
    "Brake_Fluid_Temperature_C": 70.0,
    "Brake_Pedal_Position_pct": 35.0,
    "Brake_Line_Pressure_bar": 17.5,
    "Brake_Fluid_Level_pct": 72.0,
    "ABS_Activation_Frequency": 4.2,
    "Vibration_g": 0.28,
    "Vehicle_Speed_kmh": 60.0,
    "Acceleration_ms2": -2.1,
    "Operating_Hours": 265.0,
    "Battery_SOC": 85.0,
    "Battery_Temperature": 28.0
}

scenarios = [
    ("NORMAL  - Tesla Model 3", normal),
    ("WARNING - BYD Atto 3",    warning),
    ("CRITICAL - Rivian R1T",   critical),
]

for label, payload in scenarios:
    r = requests.post(f"{API}/predict/ev", headers=HEADERS, json=payload, timeout=30)
    d = r.json()
    print(f"\n=== {label} ===")
    print(f"  Health:       {d['health_status']}")
    print(f"  Risk:         {d['risk_level']}")
    print(f"  Failure:      {d['failure_mode']}")
    print(f"  Health Conf:  {d['health_confidence']}")
    print(f"  Fail Conf:    {d['failure_confidence']}")
    print(f"  Analysis:     {d['analysis_type']}")
    print(f"  Action:       {d['recommended_action'][:80]}")
    if d.get("ai_analysis"):
        print(f"  LLM (first 120 chars): {d['ai_analysis'][:120]}...")
