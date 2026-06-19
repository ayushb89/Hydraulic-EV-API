import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreates the feature engineering used during V2 training.
    Assumes df is sorted chronologically.
    """
    df = df.copy()
    
    # Required phase columns - initialize to 0 since they are not in the raw readings
    phase_cols = [
        'Phase_approach', 'Phase_charging', 'Phase_digging', 'Phase_dump',
        'Phase_dumping', 'Phase_hauling', 'Phase_lifting', 'Phase_loading',
        'Phase_pick', 'Phase_place', 'Phase_placing', 'Phase_return',
        'Phase_returning', 'Phase_standby', 'Phase_travel', 'Phase_traveling'
    ]
    for col in phase_cols:
        if col not in df.columns:
            df[col] = 0
            
    # List of columns to calculate rolling/delta features for
    cols_to_roll = [
        'Hydraulic_Pressure', 'Oil_Temperature', 'Hydraulic_Flow_Rate',
        'Vibration', 'Battery_Temperature'
    ]
    
    # Calculate rolling and delta features
    # Since we need 30s and 300s, we will use pandas rolling with min_periods=1
    # assuming 1 reading per second, so window size 30 and 300
    for col in cols_to_roll:
        df[f'{col}_roll_mean_30s'] = df[col].rolling(window=30, min_periods=1).mean()
        df[f'{col}_roll_std_30s'] = df[col].rolling(window=30, min_periods=1).std().fillna(0)
        df[f'{col}_delta_30s'] = df[col] - df[col].shift(30).bfill().fillna(0)
        
        df[f'{col}_roll_mean_300s'] = df[col].rolling(window=300, min_periods=1).mean()
        df[f'{col}_roll_std_300s'] = df[col].rolling(window=300, min_periods=1).std().fillna(0)
        df[f'{col}_delta_300s'] = df[col] - df[col].shift(300).bfill().fillna(0)

    # Thermal Features
    df['Thermal_Spread'] = df['Oil_Temperature'] - df['Battery_Temperature']
    df['Thermal_Stress_Index'] = df['Oil_Temperature'] * df['Battery_Temperature']
    
    # Hydraulic Features
    df['Hydraulic_Stress_Index'] = df['Hydraulic_Pressure'] * df['Hydraulic_Flow_Rate']
    df['Pressure_Load_Ratio'] = df['Hydraulic_Pressure'] / (df['Load_Weight'] + 1e-5)
    df['Pressure_Temp_Ratio'] = df['Hydraulic_Pressure'] / (df['Oil_Temperature'] + 1e-5)
    
    # Battery Features
    df['Battery_Stress_Index'] = (100 - df['Battery_SOC']) * df['Battery_Temperature']
    
    # Vibration Features
    df['Vibration_Severity'] = df['Vibration'] * df['Hydraulic_Pressure']
    
    # V2 Actuator Features
    df['Actuator_Position_delta'] = df['Actuator_Position'].diff().fillna(0)
    df['Actuator_Angle_delta'] = df['Actuator_Angle'].diff().fillna(0)
    df['Actuator_Position_roll_std_30s'] = df['Actuator_Position'].rolling(window=30, min_periods=1).std().fillna(0)
    df['Actuator_Angle_roll_std_30s'] = df['Actuator_Angle'].rolling(window=30, min_periods=1).std().fillna(0)
    df['Actuator_Wear_Index'] = df['Actuator_Position'].abs() * df['Operating_Hours']
    df['Load_Position_Ratio'] = df['Load_Weight'] / (df['Actuator_Position'].abs() + 1)
    df['Load_Angle_Ratio'] = df['Load_Weight'] / (df['Actuator_Angle'].abs() + 1)
    df['Actuator_Stress_Index'] = df['Actuator_Position'].abs() * df['Actuator_Angle'].abs() * df['Load_Weight']
    df['Actuator_Load_Product'] = df['Actuator_Position'] * df['Load_Weight']
    df['Actuator_Angle_Load_Product'] = df['Actuator_Angle'] * df['Load_Weight']
    
    df['Actuator_Usage_Rate'] = df['Actuator_Position_delta'].abs().rolling(window=30, min_periods=1).mean()
    
    return df
