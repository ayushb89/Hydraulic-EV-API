"""
train_ev_models.py
──────────────────
Trains Random Forest ML models for EV Brake Telemetry prediction.
Uses: EV_Car_Brake_Telemetry_v3_FINAL.csv (100,000 real rows)

Output (saved to models/ev/):
  ev_health_model.pkl
  ev_failure_model.pkl
  ev_health_encoder.pkl
  ev_failure_encoder.pkl
  ev_feature_columns.pkl  ← list of feature column names used at training
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(__file__), "EV_Car_Brake_Telemetry_v3_FINAL.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models", "ev")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Load Dataset ──────────────────────────────────────────────
print("[INFO] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"   Rows: {len(df):,}  |  Columns: {list(df.columns)}")

# ─── 2. Clean ──────────────────────────────────────────────────────
# Drop timestamp (not a feature)
df = df.drop(columns=["Timestamp"], errors="ignore")

# Fill empty Failure_Mode (Normal rows have blank) with 'Normal_Operation'
df["Failure_Mode"] = df["Failure_Mode"].fillna("Normal_Operation").replace("", "Normal_Operation")

# Drop rows with any NaN in feature columns
feature_cols_raw = [
    "Brake_Hydraulic_Pressure_bar",
    "Brake_Fluid_Temperature_C",
    "Brake_Pedal_Position_pct",
    "Brake_Line_Pressure_bar",
    "Brake_Fluid_Level_pct",
    "ABS_Activation_Frequency",
    "Vibration_g",
    "Vehicle_Speed_kmh",
    "Acceleration_ms2",
    "Operating_Hours",
    "Battery_SOC",
    "Battery_Temperature",
]
df = df.dropna(subset=feature_cols_raw + ["Health_Status", "Failure_Mode"])
print(f"   After cleaning: {len(df):,} rows")

# ─── 3. Feature Engineering ───────────────────────────────────────
print("[INFO] Engineering features...")

def add_ev_brake_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rolling statistics (30-row and 300-row windows)
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

    # Derived / composite features
    df["Pressure_Differential"]   = df["Brake_Hydraulic_Pressure_bar"] - df["Brake_Line_Pressure_bar"]
    df["Thermal_Load"]            = df["Brake_Fluid_Temperature_C"] * df["ABS_Activation_Frequency"]
    df["Brake_Effort_Index"]      = df["Brake_Pedal_Position_pct"] * df["Brake_Hydraulic_Pressure_bar"]
    df["Fluid_Health_Index"]      = df["Brake_Fluid_Level_pct"] / (df["Brake_Fluid_Temperature_C"] + 1)
    df["Deceleration_Stress"]     = df["Acceleration_ms2"].abs() * df["Vehicle_Speed_kmh"]
    df["ABS_Vibration_Coupling"]  = df["ABS_Activation_Frequency"] * df["Vibration_g"]
    df["Battery_Thermal_Stress"]  = (100 - df["Battery_SOC"]) * df["Battery_Temperature"]
    df["Speed_Brake_Ratio"]       = df["Vehicle_Speed_kmh"] / (df["Brake_Hydraulic_Pressure_bar"] + 0.01)
    df["Hard_Braking_Flag"]       = (
        (df["Brake_Pedal_Position_pct"] > 60) & (df["Acceleration_ms2"] < -2.5)
    ).astype(int)
    df["ABS_Active_Flag"]         = (df["ABS_Activation_Frequency"] > 3.0).astype(int)
    df["Overheating_Risk"]        = (df["Brake_Fluid_Temperature_C"] > 60).astype(int)
    df["Low_Fluid_Flag"]          = (df["Brake_Fluid_Level_pct"] < 75).astype(int)

    # RUL as a feature (helps ML understand degradation trajectory)
    df["RUL_normalized"]          = df["RUL_seconds"] / (df["RUL_seconds"].max() + 1)

    return df

df = add_ev_brake_features(df)

# ─── 4. Define Feature Columns ────────────────────────────────────
feature_columns = [c for c in df.columns if c not in [
    "Health_Status", "Failure_Mode", "Operational_Phase",
    "RUL_seconds",  # raw RUL is a target/leakage risk — use normalized only
]]
print(f"   Feature count: {len(feature_columns)}")

X = df[feature_columns].fillna(0)

# ─── 5. Encode Targets ────────────────────────────────────────────
health_encoder  = LabelEncoder()
failure_encoder = LabelEncoder()

y_health  = health_encoder.fit_transform(df["Health_Status"])
y_failure = failure_encoder.fit_transform(df["Failure_Mode"])

print("\n[INFO] Class distribution:")
print(f"   Health classes:  {dict(zip(health_encoder.classes_, np.bincount(y_health)))}")
print(f"   Failure classes: {dict(zip(failure_encoder.classes_, np.bincount(y_failure)))}")

# ─── 6. Train/Test Split ──────────────────────────────────────────
X_train, X_test, yh_train, yh_test, yf_train, yf_test = train_test_split(
    X, y_health, y_failure, test_size=0.2, random_state=42, stratify=y_health
)
print(f"\n[INFO] Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# ─── 7. Train Health Model ────────────────────────────────────────
print("\n[TRAIN] Training Health Model (Normal / Warning / Critical)...")
health_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
health_model.fit(X_train, yh_train)
yh_pred = health_model.predict(X_test)
print(f"   Health Accuracy: {accuracy_score(yh_test, yh_pred):.4f}")
print(classification_report(yh_test, yh_pred, target_names=health_encoder.classes_))

# ─── 8. Train Failure Model ───────────────────────────────────────
print("\n[TRAIN] Training Failure Mode Model...")
failure_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
failure_model.fit(X_train, yf_train)
yf_pred = failure_model.predict(X_test)
print(f"   Failure Accuracy: {accuracy_score(yf_test, yf_pred):.4f}")
print(classification_report(yf_test, yf_pred, target_names=failure_encoder.classes_))

# ─── 9. Save Everything ───────────────────────────────────────────
print("\n[SAVE] Saving models...")

with open(os.path.join(MODEL_DIR, "ev_health_model.pkl"),   "wb") as f: pickle.dump(health_model,  f)
with open(os.path.join(MODEL_DIR, "ev_failure_model.pkl"),  "wb") as f: pickle.dump(failure_model, f)
with open(os.path.join(MODEL_DIR, "ev_health_encoder.pkl"), "wb") as f: pickle.dump(health_encoder, f)
with open(os.path.join(MODEL_DIR, "ev_failure_encoder.pkl"),"wb") as f: pickle.dump(failure_encoder, f)
with open(os.path.join(MODEL_DIR, "ev_feature_columns.pkl"),"wb") as f: pickle.dump(feature_columns, f)

print(f"\n[DONE] Models saved to: {MODEL_DIR}")
print("   Files created:")
for fname in os.listdir(MODEL_DIR):
    size = os.path.getsize(os.path.join(MODEL_DIR, fname))
    print(f"   - {fname}  ({size/1024:.1f} KB)")

print("\n[READY] Restart the API server -- /predict/ev will now use full ML mode.")
