import pandas as pd
import numpy as np
import os
import csv

# ALWAYS LOAD THE CORRECT FULL FILE
base = os.path.dirname(__file__)
file_path = os.path.join(base, "data", "raw", "prediction_data.csv")

print("Reading file:", file_path)

clean_rows = []

# ---- SAFEST WAY: READ LINE BY LINE AND FIX BROKEN LINES ----
with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        # Skip empty or corrupted rows
        if len(row) < 25:  
            continue
        
        clean_rows.append(row)

print("Rows read (raw):", len(clean_rows))

# Convert to DataFrame
header = clean_rows[0]
clean_data = clean_rows[1:]

df = pd.DataFrame(clean_data, columns=header)
print("DataFrame loaded:", df.shape)

# CONVERT ALL COLUMNS TO NUMERIC WHERE POSSIBLE
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# -------------------------------------------------------------
# SAME RISK SCORE FIXING LOGIC
# -------------------------------------------------------------

dose_risk = (df["daily_dosage_mg"] / (df["max_daily_safe_dose"] + 1)).clip(0, 3) * 25
escalation_risk = (df["dosage_escalation_rate"] / df["dosage_escalation_rate"].max()) * 15
refill_risk = (df["refill_count"] / df["refill_count"].max()) * 10
early_refill_risk = df["early_refill_flag"] * 10
combo_risk = df["opioid_benzo_combo_flag"] * 25

organ_risk = (
    (1 - df["liver_function_score"] / 100) * 10 +
    (1 - df["kidney_function_score"] / 100) * 10
)

psych_risk = (
    df["anxiety_level_score"] / 100 * 10 +
    df["depression_level_score"] / 100 * 10
)

behavior_risk = (
    df["missed_doses_last_month"] * 2 +
    df["overuse_incidents_last_month"] * 5
)

poly_risk = df["polypharmacy_count"] * 2.5
bp_risk = ((df["blood_pressure_level"] - 110) / 70).clip(0, 1) * 5

pharm_risk = (
    df["drug_addiction_potential"] / 100 * 10 +
    (df["drug_half_life_hours"] / df["drug_half_life_hours"].max()) * 5
)

df["corrected_risk_score"] = (
    dose_risk +
    escalation_risk +
    refill_risk +
    early_refill_risk +
    combo_risk +
    organ_risk +
    psych_risk +
    behavior_risk +
    poly_risk +
    bp_risk +
    pharm_risk
)

# NORMALIZE 0–100
df["corrected_risk_score"] = (
    (df["corrected_risk_score"] - df["corrected_risk_score"].min()) /
    (df["corrected_risk_score"].max() - df["corrected_risk_score"].min())
) * 100

save_path = os.path.join(base, "data", "raw", "prediction_data_fixed.csv")
df.to_csv(save_path, index=False)

print("SAVED FIXED DATASET:", save_path)
print("FINAL SHAPE:", df.shape)
