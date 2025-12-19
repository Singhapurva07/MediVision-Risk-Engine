import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression

import joblib


print("="*80)
print("🔥 TRAINING ADVANCED STACKING ENSEMBLE ADDICTION MODEL")
print("="*80)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
print("\n📂 Loading complete dataset...")

df = pd.read_csv("data/raw/prediction_data.csv")

print(f"✔ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------------------------------------------
# FEATURE/TARGET SPLIT
# ------------------------------------------------------------
TARGET = "addiction_risk_score"

y = df[TARGET]

X = df.drop(columns=[TARGET])

# ------------------------------------------------------------
# ENCODE CATEGORICALS
# ------------------------------------------------------------
categorical_cols = ["gender", "income_class", "drug_class"]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# ------------------------------------------------------------
# SCALING NUMERICAL FEATURES
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)

print(f"\n✔ Train: {len(X_train)}  Test: {len(X_test)}")

# ------------------------------------------------------------
# MODELS FOR STACKING
# ------------------------------------------------------------
models = {
    "LightGBM": LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        num_leaves=45, random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300, depth=6, learning_rate=0.05,
        random_state=42, verbose=False
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=220, max_depth=15,
        random_state=42, n_jobs=-1
    )
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

stack_train = np.zeros((len(X_train), len(models)))
stack_test = np.zeros((len(X_test), len(models)))

print("\n🤖 Training base models...")
print("-"*80)

# ------------------------------------------------------------
# K-FOLD STACKING
# ------------------------------------------------------------
for idx, (name, model) in enumerate(models.items()):
    fold = 1
    print(f"➡ {name}")

    test_fold_preds = []

    for tr_idx, val_idx in kf.split(X_train):

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)

        pred_val = model.predict(X_val)
        stack_train[val_idx, idx] = pred_val

        pred_test = model.predict(X_test)
        test_fold_preds.append(pred_test)

        print(f"Fold {fold} MAE = {mean_absolute_error(y_val, pred_val):.4f}")
        fold += 1

    stack_test[:, idx] = np.mean(test_fold_preds, axis=0)

# ------------------------------------------------------------
# META MODEL (FINAL LEVEL)
# ------------------------------------------------------------
print("\n📌 Training final stacker model (meta learner)...")

meta = LinearRegression()
meta.fit(stack_train, y_train)

final_pred = meta.predict(stack_test)

mae = mean_absolute_error(y_test, final_pred)
r2 = r2_score(y_test, final_pred)

print("\n🎯 FINAL PERFORMANCE:")
print(f"📘 MAE = {mae:.4f}")
print(f"📗 R²  = {r2:.4f}")

# ------------------------------------------------------------
# SAVE EVERYTHING
# ------------------------------------------------------------
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

joblib.dump(models, "data/models/base_models.pkl")
joblib.dump(meta, "data/models/meta_model.pkl")
joblib.dump(scaler, "data/processed/scaler.pkl")
joblib.dump(encoders, "data/processed/encoders.pkl")
joblib.dump(list(X.columns), "data/processed/features.pkl")

print("\n💾 SAVED MODELS + ENCODERS + SCALER")
print("="*80)
