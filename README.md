# MediVision-Risk-Engine
Machine learning system for predicting prescription medication addiction risk using clinical, behavioral, and prescription-pattern variables.
Overview

The MediVision Risk Engine is a supervised machine learning model that estimates the probability of addiction-related risk based on patient demographics, dosage patterns, physiological indicators, and behavioral factors.

The system is designed for responsible research and clinical decision support. It does not replace medical judgment. It assists practitioners by highlighting concerning patterns and identifying risk factors that require monitoring.

This project includes:

a full dataset preprocessing pipeline

feature engineering and normalization

model stacking architecture

API endpoints for prediction

interactive React frontend for profile analysis

risk factor explanation module

Key Features
1. Advanced ML Stack

The risk model uses stacked ensemble learning:

LightGBM

XGBoost

CatBoost

Random Forest base models

Linear/ElasticNet meta learner

2. Comprehensive Feature Inputs

The model accepts 20+ structured inputs, including:

medication parameters

refill and escalation patterns

psychosocial variables

physiological function indicators

3. Automated Encoding and Scaling

numeric standardization

categorical label encoding with fallback mapping for unseen values

robust preprocessing persistence

4. Real-time API

Flask backend

JSON-based prediction requests

error handling and validation

5. Frontend User Interface

React + Tailwind CSS

optional advanced inputs

real-time visualization of results

detailed risk analysis and recommendations

Repository Structure

backend/

train.py

app.py

src/

preprocessing.py

feature_engineering.py

model_utils.py

predictor.py

data/

raw/

processed/

models/

frontend/

src/

app.jsx

components/

public/

package.json

API Usage

POST /predict

Request body example:

{
  "age": 42,
  "gender": "F",
  "income_class": "middle",
  "urban_flag": 1,
  "drug_class": "Opioid",
  "drug_potency": 80,
  "daily_dose": 45,
  "max_safe_dose": 60,
  "refill_count": 3,
  "early_refill": 0,
  "dose_escalation_rate": 0.15,
  "number_of_doctors": 2,
  "number_of_pharmacies": 1,
  "anxiety": 65,
  "depression": 50,
  "stress": 70,
  "compulsive_use": 3,
  "missed_doses": 2,
  "overuse": 1,
  "liver_score": 80,
  "kidney_score": 85,
  "blood_pressure": 135,
  "heart_rate": 88
}


Response example:

{
  "success": true,
  "risk_score": 34.35
}

Model Output

The score is scaled 0–100

0–29: Low risk
30–59: Moderate risk
60+: High risk

Intended Use

This model is intended for research and clinical decision augmentation. It should not be used as a standalone diagnostic tool.

Local Setup

Backend:

cd backend
pip install -r requirements.txt
python train.py      (first time only)
python app.py


Frontend:

cd frontend
npm install
npm run dev


The frontend and backend communicate via HTTP.

Future Improvements

uncertainty estimation (Bayesian stacking)

longitudinal temporal modeling of escalation

clinical dataset integration

adversarial and robustness evaluation

explainability enhancement using SHAP
