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
                    "max_tokens": 300
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
    You are an expert Hydraulic EV Maintenance Engineer.
    
    A {vehicle_type} has just triggered an automated diagnostic alert.
    The Machine Learning model predicted the following:
    - Health Status: {prediction_result.get('health_status')}
    - Failure Mode: {prediction_result.get('failure_mode')}
    - Risk Level: {prediction_result.get('risk_level')}
    
    Here is the most recent sensor telemetry reading at the time of the alert:
    {telemetry_row}
    
    Please provide a concise, highly technical diagnostic report. 
    Look closely at the specific sensor values (like pressure, temperature, flow rate, vibration, etc.).
    Point out which specific sensor values are likely causing the '{prediction_result.get('failure_mode')}' and explain the mechanical reasoning.
    Provide a specific recommendation for the maintenance crew.
    Format your response as a clear list of bullet points using standard dashes (-). Do NOT write a paragraph.
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
