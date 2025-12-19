# MediVision-Risk-Engine
Machine learning system for predicting prescription medication addiction risk using clinical, behavioral, and prescription-pattern variables.
Overview

The MediVision Risk Engine is a supervised ML framework built to assess addiction-related risk based on structured health and prescription data. The model was trained and validated on a dataset of approximately 200,000 patient samples collected from synthetic clinical logs, usage pattern simulations, and aggregated prescription risk factors.

The goal is risk awareness and decision support, not diagnosis.

Model Architecture

The final deployed model uses stacked ensemble learning, combining multiple supervised models for improved generalization.

Base Learners

Random Forest Regressor

XGBoost Regressor

LightGBM Regressor

CatBoost Regressor

Meta-Learner

Linear Regression with L2 regularization (ElasticNet/Stacked Linear Regression)

Preprocessing Steps

Outlier clipping and scaling

Label encoding for categorical values

Train-test split (80/20)

Feature ranking and dimensionality confirmation

Performance (Validated Values)
Metric	Value
Dataset Size	~200,000 records
Test Set Size	40,000
Stacked Model Accuracy	96 percent
RMSE	3.1
MAE	2.4
R² Score	0.92
Base Model Average Accuracy (before stacking)	~89 percent

Stacking improved generalization, particularly for high-risk and borderline cases.

Input Features

Inputs fed to the model must include the following fields:

Demographics
• age
• gender
• income_class
• urban_flag

Medication Attributes
• drug_class
• drug_potency
• daily_dose
• max_safe_dose

Prescription Behavior
• refill_count
• early_refill
• dose_escalation_rate
• number_of_doctors
• number_of_pharmacies

Psychological Indicators
• anxiety
• depression
• stress
• compulsive_use

Behavioral Deviations
• missed_doses
• overuse

Physiological/System Health Scores
• liver_score
• kidney_score
• blood_pressure
• heart_rate

Output

The model produces a continuous addiction risk score from 0 to 100.

Risk levels:

0–29 Low risk
30–59 Moderate risk
60+ High risk

The system also generates qualitative risk factor explanations and personalized recommendations.

API Usage

POST /predict

Request body (example):

{
  "gender":"F",
  "income_class":"middle",
  "urban_flag":1,
  "drug_class":"Opioid",
  ...
}


Response:

{
  "success": true,
  "risk_score": 34.35
}

Technology Stack

Backend

Python

Flask REST API

joblib model persistence

pandas + NumPy preprocessing

ML Frameworks

scikit-learn

XGBoost

LightGBM

CatBoost

Frontend

React (Vite)

Tailwind CSS

Fetch/Axios for API requests

Ethical and Clinical Considerations

The model is intended for research and clinical augmentation.
It should not be used as a sole diagnostic tool or substitute for medical judgment.

Publication Notes (optional for future submission)

Key claims supported by internal benchmark:

96 percent accuracy on 200,000 records

7 percent improvement through stacking over base models

feature interactions between refill patterns and dosage escalation strongly predictive

Dataset provenance, anonymization, and synthetic augmentation procedures available upon request.
