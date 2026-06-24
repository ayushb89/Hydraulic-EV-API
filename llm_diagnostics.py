import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
is_gemini_configured = False
is_openrouter = False

if GEMINI_API_KEY and GEMINI_API_KEY != "your_api_key_here":
    if GEMINI_API_KEY.startswith("sk-or-"):
        is_openrouter = True
        is_gemini_configured = True
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            llm_model = genai.GenerativeModel('gemini-1.5-flash')
            is_gemini_configured = True
        except Exception as e:
            print(f"Warning: Failed to configure Gemini API: {e}")


# ─────────────────────────────────────────────────────────────────
# Core LLM Caller
# ─────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    if not is_gemini_configured:
        return None

    if is_openrouter:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return None
    else:
        try:
            response = llm_model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None
    return None


# ─────────────────────────────────────────────────────────────────
# Hydraulic Vehicle — ML-backed Analysis
# Called when health is Warning or Critical
# ─────────────────────────────────────────────────────────────────
def generate_llm_feedback(vehicle_type: str, prediction_result: dict, telemetry_row: dict) -> str:
    """
    Calls the LLM API to generate a professional maintenance report
    from ML predictions + raw sensor telemetry.
    """
    if not is_gemini_configured:
        return None

    prompt = f"""
    You are an Industrial Predictive Maintenance AI Engineer working for a Fortune 500 heavy-equipment company.
    Your task is to generate a professional maintenance report from sensor telemetry and ML predictions.

    STRICT FORMAT RULES:
    1. NEVER write long paragraphs.
    2. NEVER write essays.
    3. NEVER exceed 250 words.
    4. Use clear section headings.
    5. Use bullet points only.
    6. Use concise engineering language.
    7. Write like a maintenance engineer, not a chatbot.
    8. Highlight only the most critical findings.

    A {vehicle_type} has just triggered an automated diagnostic alert.
    The Machine Learning model predicted the following:
    - Health Status: {prediction_result.get('health_status')}
    - Failure Mode: {prediction_result.get('failure_mode')}
    - Risk Level: {prediction_result.get('risk_level')}

    Here is the most recent sensor telemetry reading:
    {telemetry_row}

    OUTPUT FORMAT:

    # Asset Health Summary
    Vehicle Type: {vehicle_type}
    Health Status: {prediction_result.get('health_status')}
    Failure Mode: {prediction_result.get('failure_mode')}
    Risk Level: {prediction_result.get('risk_level')}

    # Critical Findings
    • Finding 1
    • Finding 2
    • Finding 3

    # Root Cause Analysis
    • Cause 1
    • Cause 2
    • Cause 3

    # Sensor Deviations
    Parameter | Current Value | Expected Range | Severity
    Display maximum 5 rows.

    # Remaining Useful Life (Estimated)
    Estimated Time to Failure:
    Confidence Level:
    Explain estimation in 2 bullet points only.

    # Recommended Actions
    Immediate (0–24 hrs)
    • Action 1
    • Action 2

    Short Term (1–7 days)
    • Action 1
    • Action 2

    Long Term (>7 days)
    • Action 1
    • Action 2

    # Business Impact
    • Potential downtime:
    • Safety impact:
    • Maintenance priority:

    RULES FOR RUL ESTIMATION:
    If pressure, temperature, vibration and flow are all severely abnormal: RUL = 0–3 days
    If 3 parameters are abnormal: RUL = 3–7 days
    If 2 parameters are abnormal: RUL = 7–30 days
    If only 1 parameter is abnormal: RUL = 30–90 days
    If health status is Normal: RUL = >90 days
    Always provide a numerical estimate.

    The final output must look exactly like the professional maintenance dashboard report requested.
    """
    return call_llm(prompt)


# ─────────────────────────────────────────────────────────────────
# Closest-Match Vehicle — Professional Fallback Report
# Called when vehicle type was unknown but closest match was found
# ─────────────────────────────────────────────────────────────────
def generate_unseen_vehicle_feedback(vehicle_type: str, telemetry_row: dict, matched_type: str = None) -> str:
    """
    Generates a professional heuristic report for unknown vehicle types.
    Uses the same Fortune-500 structured format as the main ML report.
    """
    if not is_gemini_configured:
        return "Unseen vehicle type detected. No ML or AI prediction available."

    match_note = (
        f"The system matched this to the nearest known vehicle type: '{matched_type}' for ML analysis."
        if matched_type
        else "No close ML match was found. This is a pure heuristic LLM assessment."
    )

    prompt = f"""
    You are an Industrial Predictive Maintenance AI Engineer.

    An unregistered vehicle type '{vehicle_type}' sent telemetry data.
    {match_note}

    Analyze the raw sensor data below and generate a professional heuristic health assessment
    using hydraulic equipment engineering standards.

    Sensor Data:
    {telemetry_row}

    STRICT FORMAT RULES:
    1. NEVER write long paragraphs.
    2. Use bullet points only.
    3. Use concise engineering language.
    4. Max 200 words.

    OUTPUT FORMAT:

    # Asset Health Summary
    Vehicle Type: {vehicle_type}
    Assessment Type: Heuristic AI Analysis (Unregistered Vehicle)
    Closest ML Match: {matched_type if matched_type else 'None found'}

    # Key Observations
    • Observation 1
    • Observation 2
    • Observation 3

    # Risk Assessment
    • Overall Risk: (Low / Medium / High)
    • Primary Concern: (what is most abnormal)

    # Recommended Actions
    • Action 1
    • Action 2

    RULES:
    - Flag any temperature above 100°C as High Risk.
    - Flag pressure below 50 or above 400 as abnormal.
    - Flag vibration above 3.0 as concerning.
    - Flag flow rate below 15 as very low.
    """
    res = call_llm(prompt)
    if res:
        return res
    return "Error generating AI feedback for unseen vehicle."


# ─────────────────────────────────────────────────────────────────
# EV Passenger Car — AI Analysis
# Dedicated LLM prompt for EV-specific failure modes
# ─────────────────────────────────────────────────────────────────
def generate_ev_feedback(vehicle_type: str, telemetry_row: dict,
                         health_status: str, failure_mode: str,
                         risk_level: str) -> str:
    """
    Generates a professional EV maintenance report using EV-specific
    engineering context and failure modes.
    """
    if not is_gemini_configured:
        return "LLM not configured. Cannot generate EV analysis."

    prompt = f"""
    You are an Expert EV Powertrain Diagnostics Engineer at a leading electric vehicle company.
    Generate a professional EV health report from the sensor telemetry and ML predictions below.

    STRICT FORMAT RULES:
    1. NEVER write long paragraphs or essays.
    2. Use bullet points only.
    3. Use concise EV engineering language.
    4. Max 250 words.
    5. Focus on Motor, Inverter, and Battery systems.

    Vehicle: {vehicle_type}
    ML Predicted Health Status: {health_status}
    ML Predicted Failure Mode:  {failure_mode}
    Risk Level:                 {risk_level}

    Live Sensor Telemetry:
    {telemetry_row}

    OUTPUT FORMAT:

    # EV Asset Health Summary
    Vehicle Type: {vehicle_type}
    Health Status: {health_status}
    Failure Mode: {failure_mode}
    Risk Level: {risk_level}

    # Critical Findings
    • Finding 1
    • Finding 2
    • Finding 3

    # Root Cause Analysis
    • Cause 1
    • Cause 2

    # Sensor Deviations
    Parameter | Current Value | Normal Range | Severity
    (Motor_RPM | Inverter_Temp | Motor_Temp | Phase_Current | Battery_SOC — max 5 rows)

    # Remaining Useful Life (Estimated)
    Estimated Time to Failure:
    Confidence Level:
    • Bullet point 1
    • Bullet point 2

    # Recommended Actions
    Immediate (0–24 hrs): • Action
    Short Term (1–7 days): • Action
    Long Term (>7 days): • Action

    # Business Impact
    • Range impact:
    • Safety impact:
    • Priority:

    EV FAILURE MODE REFERENCE (use these exactly):
    - Inverter_Thermal_Throttling: Inverter_Temp > 85°C
    - Motor_Bearing_Wear: High vibration at high RPM
    - Phase_Current_Imbalance: Phase_Current fluctuating erratically
    - Battery_Cell_Imbalance: SOC < 15% or Battery_Temp > 45°C
    - Coolant_Leak: Motor_Temp rising while Inverter_Temp drops
    - Normal_Operation: All parameters within range

    RUL RULES FOR EV:
    If Motor_Temp > 150°C or Inverter_Temp > 90°C: RUL = 0–1 days
    If 3+ parameters abnormal: RUL = 1–7 days
    If 2 parameters abnormal: RUL = 7–30 days
    If 1 parameter abnormal: RUL = 30–90 days
    If all normal: RUL = >90 days
    """
    res = call_llm(prompt)
    if res:
        return res
    return "Error generating EV AI analysis."
