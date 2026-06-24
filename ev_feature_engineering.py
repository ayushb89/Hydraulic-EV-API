import pandas as pd
import numpy as np


def generate_ev_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for EV passenger vehicle telemetry.
    Computes rolling statistics, delta features, and EV-specific stress indices.
    """
    df = df.copy()

    # ── Rolling & Delta Features ──────────────────────────────────
    cols_to_roll = ['Motor_RPM', 'Motor_Temp', 'Inverter_Temp', 'Phase_Current', 'Battery_Temperature']

    for col in cols_to_roll:
        df[f'{col}_roll_mean_30s']  = df[col].rolling(window=30, min_periods=1).mean()
        df[f'{col}_roll_std_30s']   = df[col].rolling(window=30, min_periods=1).std().fillna(0)
        df[f'{col}_delta_30s']      = df[col] - df[col].shift(30).bfill().fillna(0)

        df[f'{col}_roll_mean_300s'] = df[col].rolling(window=300, min_periods=1).mean()
        df[f'{col}_roll_std_300s']  = df[col].rolling(window=300, min_periods=1).std().fillna(0)
        df[f'{col}_delta_300s']     = df[col] - df[col].shift(300).bfill().fillna(0)

    # ── Thermal Features ─────────────────────────────────────────
    df['Thermal_Spread']        = df['Motor_Temp'] - df['Inverter_Temp']
    df['Motor_Thermal_Stress']  = df['Motor_Temp'] * df['Motor_RPM'] / (df['Motor_RPM'].max() + 1e-5)
    df['Inverter_Stress_Index'] = df['Inverter_Temp'] * df['Phase_Current']

    # ── Motor & Power Features ────────────────────────────────────
    df['Motor_Power_kW']        = (df['Motor_Torque'] * df['Motor_RPM'] * 2 * np.pi / 60) / 1000.0
    df['Torque_RPM_Ratio']      = df['Motor_Torque'] / (df['Motor_RPM'] + 1e-5)
    df['Motor_Load_Factor']     = df['Motor_RPM'] / 18000.0  # Normalized 0–1

    # ── Battery Features ──────────────────────────────────────────
    df['Battery_Stress_Index']  = (100 - df['Battery_SOC']) * df['Battery_Temperature']
    df['SOC_Depletion_Rate']    = df['Battery_SOC'].diff().fillna(0).abs()

    # ── Speed & Efficiency ────────────────────────────────────────
    df['Speed_RPM_Ratio']       = df['Vehicle_Speed'] / (df['Motor_RPM'] + 1e-5)
    df['Regen_Flag']            = (df['Motor_Torque'] < 0).astype(int)  # 1 = regen braking

    # ── Actuator-style delta features ─────────────────────────────
    df['RPM_delta']             = df['Motor_RPM'].diff().fillna(0)
    df['RPM_roll_std_30s']      = df['Motor_RPM'].rolling(window=30, min_periods=1).std().fillna(0)

    return df
