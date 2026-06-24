import requests, json

API_KEY = "5611ef24790093d63ba87a8ba22fd6e12e6e0a4da34d2290b402bb8062efd261"
headers = {"X-API-Key": API_KEY}
BASE    = "http://127.0.0.1:8000"

# ── Test 1: /predict/live (healthy forklift snapshot) ──────────────────
print("=== Test 1: /predict/live (Forklift - healthy) ===")
r = requests.post(f"{BASE}/predict/live", headers=headers, json={
    "vehicle_id":          "LIVE_TEST_001",
    "Vehicle_Type":        "Forklift",
    "Hydraulic_Pressure":  130.0,
    "Oil_Temperature":     55.0,
    "Actuator_Angle":      25.0,
    "Actuator_Position":   800.0,
    "Load_Weight":         1500.0,
    "Hydraulic_Flow_Rate": 50.0,
    "Vibration":           0.3,
    "Operating_Hours":     1200.0,
    "Battery_SOC":         82.0,
    "Battery_Temperature": 33.0,
})
d = r.json()
print(f"  Status  : {d.get('health_status')} | Risk: {d.get('risk_level')} | Conf: {d.get('health_confidence')}")
print(f"  Failure : {d.get('failure_mode')}")
print(f"  Note    : {d.get('matched_vehicle_note')}")

print()

# ── Test 2: /predict/live (unknown vehicle - closest match) ────────────
print("=== Test 2: /predict/live (Excavator -> closest match: Backhoe) ===")
r2 = requests.post(f"{BASE}/predict/live", headers=headers, json={
    "vehicle_id":          "EXCAV_001",
    "Vehicle_Type":        "Excavator",
    "Hydraulic_Pressure":  130.0,
    "Oil_Temperature":     55.0,
    "Actuator_Angle":      25.0,
    "Actuator_Position":   800.0,
    "Load_Weight":         1500.0,
    "Hydraulic_Flow_Rate": 50.0,
    "Vibration":           0.3,
    "Operating_Hours":     1200.0,
    "Battery_SOC":         82.0,
    "Battery_Temperature": 33.0,
})
d2 = r2.json()
print(f"  Status  : {d2.get('health_status')} | Risk: {d2.get('risk_level')} | Conf: {d2.get('health_confidence')}")
print(f"  Note    : {d2.get('matched_vehicle_note')}")

print()

# ── Test 3: /predict/ev (Tesla - warning scenario) ─────────────────────
print("=== Test 3: /predict/ev (Tesla Model 3 - warning) ===")
r3 = requests.post(f"{BASE}/predict/ev", headers=headers, json={
    "vehicle_id":          "TESLA_001",
    "Vehicle_Type":        "Tesla Model 3",
    "Motor_RPM":           8000.0,
    "Motor_Temp":          128.0,
    "Inverter_Temp":       78.0,
    "Motor_Torque":        200.0,
    "Phase_Current":       420.0,
    "Battery_SOC":         65.0,
    "Battery_Temperature": 38.0,
    "Vehicle_Speed":       95.0,
    "Operating_Hours":     3500.0,
})
d3 = r3.json()
print(f"  Status  : {d3.get('health_status')} | Risk: {d3.get('risk_level')} | Conf: {d3.get('health_confidence')}")
print(f"  Failure : {d3.get('failure_mode')}")
print(f"  Type    : {d3.get('analysis_type')}")
