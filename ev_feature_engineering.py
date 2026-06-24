"""
ev_feature_engineering.py
─────────────────────────
Feature engineering for EV Brake Telemetry.
Matches the exact features produced during training in train_ev_models.py.

Input columns (raw sensor data):
  Brake_Hydraulic_Pressure_bar, Brake_Fluid_Temperature_C,
  Brake_Pedal_Position_pct, Brake_Line_Pressure_bar,
  Brake_Fluid_Level_pct, ABS_Activation_Frequency, Vibration_g,
  Vehicle_Speed_kmh, Acceleration_ms2, Operating_Hours,
  Battery_SOC, Battery_Temperature
"""

import pandas as pd
import numpy as np


def generate_ev_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for EV Brake Telemetry.
    Computes rolling statistics, delta features, and EV-specific stress indices.
    Matches EXACTLY the features used during model training.
    """
    df = df.copy()

    # ── Rolling & Delta Features (5 key brake sensors) ────────────
    roll_cols = [
        "Brake_Hydraulic_Pressure_bar",
        "Brake_Fluid_Temperature_C",
        "Brake_Line_Pressure_bar",
        "ABS_Activation_Frequency",
        "Vibration_g",
    ]
    for col in roll_cols:
        df[f"{col}_roll_mean_30"]  = df[col].rolling(30,  min_periods=1).mean()
        df[f"{col}_roll_std_30"]   = df[col].rolling(30,  min_periods=1).std().fillna(0)
        df[f"{col}_delta_30"]      = df[col] - df[col].shift(30).bfill()
        df[f"{col}_roll_mean_300"] = df[col].rolling(300, min_periods=1).mean()
        df[f"{col}_roll_std_300"]  = df[col].rolling(300, min_periods=1).std().fillna(0)
        df[f"{col}_delta_300"]     = df[col] - df[col].shift(300).bfill()

    # ── Derived / Composite Features ──────────────────────────────
    df["Pressure_Differential"]  = df["Brake_Hydraulic_Pressure_bar"] - df["Brake_Line_Pressure_bar"]
    df["Thermal_Load"]           = df["Brake_Fluid_Temperature_C"] * df["ABS_Activation_Frequency"]
    df["Brake_Effort_Index"]     = df["Brake_Pedal_Position_pct"] * df["Brake_Hydraulic_Pressure_bar"]
    df["Fluid_Health_Index"]     = df["Brake_Fluid_Level_pct"] / (df["Brake_Fluid_Temperature_C"] + 1)
    df["Deceleration_Stress"]    = df["Acceleration_ms2"].abs() * df["Vehicle_Speed_kmh"]
    df["ABS_Vibration_Coupling"] = df["ABS_Activation_Frequency"] * df["Vibration_g"]
    df["Battery_Thermal_Stress"] = (100 - df["Battery_SOC"]) * df["Battery_Temperature"]
    df["Speed_Brake_Ratio"]      = df["Vehicle_Speed_kmh"] / (df["Brake_Hydraulic_Pressure_bar"] + 0.01)

    # ── Binary Flag Features ───────────────────────────────────────
    df["Hard_Braking_Flag"]  = (
        (df["Brake_Pedal_Position_pct"] > 60) & (df["Acceleration_ms2"] < -2.5)
    ).astype(int)
    df["ABS_Active_Flag"]    = (df["ABS_Activation_Frequency"] > 3.0).astype(int)
    df["Overheating_Risk"]   = (df["Brake_Fluid_Temperature_C"] > 60).astype(int)
    df["Low_Fluid_Flag"]     = (df["Brake_Fluid_Level_pct"] < 75).astype(int)

    # ── RUL Normalized (degradation trajectory indicator) ─────────
    # At inference time we don't have ground-truth RUL, so we default to 1.0 (healthy)
    # The model is trained on this feature — keeping it avoids feature mismatch.
    if "RUL_seconds" in df.columns:
        max_rul = df["RUL_seconds"].max()
        df["RUL_normalized"] = df["RUL_seconds"] / (max_rul + 1)
    else:
        df["RUL_normalized"] = 1.0  # unknown → assume healthy baseline

    return df
