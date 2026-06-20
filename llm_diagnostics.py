import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini if API key is present
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
is_gemini_configured = False
is_openrouter = False

if GEMINI_API_KEY and GEMINI_API_KEY != "your_api_key_here":
    if GEMINI_API_KEY.startswith("sk-or-"):
        # It's an OpenRouter API key
        is_openrouter = True
        is_gemini_configured = True
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Using the standard gemini model for text generation
            llm_model = genai.GenerativeModel('gemini-1.5-flash')
            is_gemini_configured = True
        except Exception as e:
            print(f"Warning: Failed to configure Gemini API: {e}")

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

def generate_llm_feedback(vehicle_type: str, prediction_result: dict, telemetry_row: dict) -> str:
    """
    Calls the LLM API to generate contextual engineering feedback based on telemetry.
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
    If pressure, temperature, vibration and flow are all severely abnormal:
    RUL = 0–3 days
    If 3 parameters are abnormal:
    RUL = 3–7 days
    If 2 parameters are abnormal:
    RUL = 7–30 days
    If only 1 parameter is abnormal:
    RUL = 30–90 days
    If health status is Normal:
    RUL = >90 days
    Always provide a numerical estimate.

    The final output must look exactly like the professional maintenance dashboard report requested.
    """
    
    return call_llm(prompt)

def generate_unseen_vehicle_feedback(vehicle_type: str, telemetry_row: dict) -> str:
    """
    Fallback call for when the ML model rejects an unseen vehicle category.
    """
    if not is_gemini_configured:
        return "Unseen vehicle type detected. No ML or AI prediction available."
        
    prompt = f"""
    You are an expert Hydraulic EV Maintenance Engineer.
    
    A new/unseen vehicle type ('{vehicle_type}') sent telemetry data. Our Machine Learning models were not trained on this vehicle, so they rejected it.
    However, we need you to look at the raw sensor data and give a heuristic estimation of its health.
    
    Here is the telemetry data:
    {telemetry_row}
    
    Provide a concise assessment. Does anything look dangerously high or low for a generic heavy-duty hydraulic EV? Give a quick recommendation.
    Format your response as a clear list of bullet points using standard dashes (-). Do NOT write a paragraph.
    """
    
    res = call_llm(prompt)
    if res:
        return res
    return "Error generating AI feedback for unseen vehicle."
