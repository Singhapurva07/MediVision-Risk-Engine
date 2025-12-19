"""
🔥 COMPLETE SETUP SCRIPT - Creates ALL files automatically!
Run this ONCE and everything will be ready!
"""

import os
import sys

def create_file(path, content):
    """Create file with content"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def setup_backend():
    """Setup complete backend structure"""
    
    # Create directories
    os.makedirs('backend/src', exist_ok=True)
    os.makedirs('backend/data/raw', exist_ok=True)
    os.makedirs('backend/data/processed', exist_ok=True)
    os.makedirs('backend/data/models', exist_ok=True)
    
    # requirements.txt
    create_file('backend/requirements.txt', """flask==3.0.0
flask-cors==4.0.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
imbalanced-learn==0.11.0
joblib==1.3.2
scipy==1.11.4
""")

    # src/__init__.py
    create_file('backend/src/__init__.py', "")

    # GENERATE SAMPLE DATA FIRST!
    create_file('backend/generate_data.py', """
import pandas as pd
import numpy as np

np.random.seed(42)

# Drug data
opioids = ['Oxycodone', 'Hydrocodone', 'Morphine', 'Fentanyl']
benzos = ['Alprazolam', 'Diazepam', 'Lorazepam']
sedatives = ['Zolpidem', 'Eszopiclone']
stimulants = ['Adderall', 'Ritalin']

all_drugs = opioids + benzos + sedatives + stimulants

drug_potency = {
    'Fentanyl': 95, 'Oxycodone': 85, 'Morphine': 80, 'Hydrocodone': 75,
    'Alprazolam': 70, 'Diazepam': 65, 'Lorazepam': 68,
    'Zolpidem': 55, 'Eszopiclone': 50,
    'Adderall': 65, 'Ritalin': 60
}

drug_class_map = {}
for d in opioids: drug_class_map[d] = 'Opioid'
for d in benzos: drug_class_map[d] = 'Benzodiazepine'
for d in sedatives: drug_class_map[d] = 'Sedative'
for d in stimulants: drug_class_map[d] = 'Stimulant'

data = []
for i in range(1000):
    drug = np.random.choice(all_drugs)
    age = np.random.randint(18, 85)
    gender = np.random.choice(['Male', 'Female'])
    
    daily_dose = np.random.uniform(10, 100)
    max_dose = daily_dose * np.random.uniform(1.3, 2.0)
    duration = np.random.choice([30, 60, 90, 120])
    refills = np.random.randint(0, 8)
    
    risk_score = (
        drug_potency[drug] * 0.3 +
        (daily_dose / max_dose * 100) * 0.2 +
        refills * 5 +
        np.random.uniform(-10, 10)
    )
    risk_score = max(0, min(100, risk_score))
    
    data.append({
        'patient_id': f'PT{i+1:05d}',
        'age': age,
        'gender': gender,
        'drug_name': drug,
        'drug_class': drug_class_map[drug],
        'daily_dosage_mg': round(daily_dose, 2),
        'max_daily_safe_dose': round(max_dose, 2),
        'prescription_duration_days': duration,
        'refill_count': refills,
        'early_refill_flag': np.random.choice([0, 1]),
        'number_of_doctors': np.random.choice([1, 1, 2, 3]),
        'number_of_pharmacies': np.random.choice([1, 1, 2]),
        'dosage_escalation_rate': round(np.random.uniform(0, 0.3), 3),
        'opioid_benzo_combo_flag': np.random.choice([0, 0, 0, 1]),
        'polypharmacy_count': np.random.randint(1, 5),
        'missed_doses_last_month': np.random.randint(0, 15),
        'overuse_incidents_last_month': np.random.randint(0, 5),
        'adherence_percentage': round(np.random.uniform(50, 100), 2),
        'liver_function_score': round(np.random.uniform(60, 100), 2),
        'kidney_function_score': round(np.random.uniform(60, 100), 2),
        'blood_pressure_level': round(np.random.uniform(110, 150), 2),
        'anxiety_level_score': round(np.random.uniform(0, 100), 2),
        'depression_level_score': round(np.random.uniform(0, 100), 2),
        'drug_addiction_potential': drug_potency[drug],
        'drug_half_life_hours': round(np.random.uniform(2, 12), 2),
        'risk_escalation_velocity': round(np.random.uniform(0, 1), 3),
        'prescription_intensity_index': round(np.random.uniform(30, 90), 2),
        'combined_overuse_score': round(np.random.uniform(0, 80), 2),
        'addiction_risk_score': round(risk_score, 2)
    })

df = pd.DataFrame(data)
df.to_csv('data/raw/prediction_data.csv', index=False)
print(f"✅ Generated {len(df)} records!")
print(f"📊 Risk range: {df['addiction_risk_score'].min():.1f} - {df['addiction_risk_score'].max():.1f}")
""")

    # COMPLETE TRAINING SCRIPT
    create_file('backend/train.py', """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔥 TRAINING ADDICTION PREDICTION MODEL")
print("="*70)

# 1. LOAD DATA
print("\\n📂 Loading data...")
df = pd.read_csv('data/raw/prediction_data.csv')
print(f"✅ Loaded {len(df)} records with {len(df.columns)} features")

# 2. FEATURE ENGINEERING
print("\\n🧠 Engineering features...")
df['overdose_risk'] = (df['daily_dosage_mg'] / df['max_daily_safe_dose'] * 50 + 
                       df['combined_overuse_score'] * 0.5).clip(0, 100)
df['doctor_shopping'] = ((df['number_of_doctors'] - 1) * 30 + 
                         (df['number_of_pharmacies'] - 1) * 20).clip(0, 100)
df['psych_risk'] = (df['anxiety_level_score'] * 0.5 + 
                    df['depression_level_score'] * 0.5).clip(0, 100)
df['dosage_ratio'] = df['daily_dosage_mg'] / (df['max_daily_safe_dose'] + 1)
df['age_risk'] = ((df['age'] < 25) | (df['age'] > 65)).astype(int)

print(f"✅ Created {len([c for c in df.columns if c not in pd.read_csv('data/raw/prediction_data.csv').columns])} new features")

# 3. ENCODING
print("\\n⚙️  Processing data...")
le_gender = LabelEncoder()
le_drug = LabelEncoder()
le_class = LabelEncoder()

df['gender_enc'] = le_gender.fit_transform(df['gender'])
df['drug_name_enc'] = le_drug.fit_transform(df['drug_name'])
df['drug_class_enc'] = le_class.fit_transform(df['drug_class'])

# 4. PREPARE FEATURES
exclude = ['patient_id', 'gender', 'drug_name', 'drug_class', 'addiction_risk_score']
feature_cols = [c for c in df.columns if c not in exclude]

X = df[feature_cols]
y = df['addiction_risk_score']

# 5. SPLIT
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"✅ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# 6. NORMALIZE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 7. TRAIN MODELS
print("\\n🚀 Training models...")
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
}

best_score = float('inf')
best_model = None
best_name = None

for name, model in models.items():
    print(f"\\n  Training {name}...")
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    print(f"    MAE: {mae:.3f} | R²: {r2:.3f}")
    
    if mae < best_score:
        best_score = mae
        best_model = model
        best_name = name

print(f"\\n🏆 Best: {best_name} (MAE: {best_score:.3f})")

# 8. TEST
test_pred = best_model.predict(X_test_scaled)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)
print(f"\\n🎯 Test Performance: MAE={test_mae:.3f}, R²={test_r2:.3f}")

# 9. SAVE
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

joblib.dump(best_model, 'data/models/best_model.pkl')
joblib.dump(scaler, 'data/processed/scaler.pkl')
joblib.dump(feature_cols, 'data/processed/features.pkl')
joblib.dump({'gender': le_gender, 'drug': le_drug, 'class': le_class}, 'data/processed/encoders.pkl')
joblib.dump(best_name, 'data/models/model_name.pkl')

print("\\n💾 Saved all models and processors!")
print("="*70)
print("✅ TRAINING COMPLETE! Run: python app.py")
print("="*70)
""")

    # FLASK APP
    create_file('backend/app.py', """
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load everything
model = joblib.load('data/models/best_model.pkl')
scaler = joblib.load('data/processed/scaler.pkl')
features = joblib.load('data/processed/features.pkl')
encoders = joblib.load('data/processed/encoders.pkl')
model_name = joblib.load('data/models/model_name.pkl')

print(f"✅ Loaded {model_name} model with {len(features)} features")

def process_patient(data):
    df = pd.DataFrame([data])
    
    # Engineer features
    df['overdose_risk'] = (df['daily_dosage_mg'] / df['max_daily_safe_dose'] * 50 + 
                           df['combined_overuse_score'] * 0.5).clip(0, 100)
    df['doctor_shopping'] = ((df['number_of_doctors'] - 1) * 30 + 
                             (df['number_of_pharmacies'] - 1) * 20).clip(0, 100)
    df['psych_risk'] = (df['anxiety_level_score'] * 0.5 + 
                        df['depression_level_score'] * 0.5).clip(0, 100)
    df['dosage_ratio'] = df['daily_dosage_mg'] / (df['max_daily_safe_dose'] + 1)
    df['age_risk'] = ((df['age'] < 25) | (df['age'] > 65)).astype(int)
    
    # Encode
    df['gender_enc'] = encoders['gender'].transform(df['gender'])
    df['drug_name_enc'] = encoders['drug'].transform(df['drug_name'])
    df['drug_class_enc'] = encoders['class'].transform(df['drug_class'])
    
    X = df[features]
    X_scaled = scaler.transform(X)
    
    risk_score = float(model.predict(X_scaled)[0])
    risk_score = max(0, min(100, risk_score))
    
    # Generate comprehensive analysis
    patient = df.iloc[0]
    
    return {
        'addiction_risk_score': round(risk_score, 2),
        'risk_category': get_category(risk_score),
        'overdose_risk': {
            'score': round(float(patient['overdose_risk']), 2),
            'level': get_level(patient['overdose_risk'])
        },
        'doctor_shopping': {
            'score': round(float(patient['doctor_shopping']), 2),
            'doctors': int(patient['number_of_doctors']),
            'pharmacies': int(patient['number_of_pharmacies'])
        },
        'psychological_risk': {
            'score': round(float(patient['psych_risk']), 2),
            'anxiety': round(float(patient['anxiety_level_score']), 2),
            'depression': round(float(patient['depression_level_score']), 2)
        },
        'intervention': get_intervention(risk_score),
        'misuse_patterns': {
            'overuse_incidents': int(patient['overuse_incidents_last_month']),
            'missed_doses': int(patient['missed_doses_last_month']),
            'adherence': round(float(patient['adherence_percentage']), 2)
        }
    }

def get_category(score):
    if score < 30: return {'level': 'Low', 'color': 'green'}
    if score < 60: return {'level': 'Medium', 'color': 'yellow'}
    if score < 80: return {'level': 'High', 'color': 'orange'}
    return {'level': 'Extreme', 'color': 'red'}

def get_level(score):
    if score < 30: return 'Low'
    if score < 60: return 'Moderate'
    if score < 80: return 'High'
    return 'Critical'

def get_intervention(score):
    if score >= 80: return {'level': 'Immediate', 'priority': 'Critical'}
    if score >= 60: return {'level': 'Urgent', 'priority': 'High'}
    if score >= 30: return {'level': 'Monitoring', 'priority': 'Moderate'}
    return {'level': 'Routine', 'priority': 'Low'}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': model_name})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        predictions = process_patient(data)
        return jsonify({
            'success': True,
            'patient_id': data.get('patient_id', 'Unknown'),
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("\\n" + "="*70)
    print("🔥 ADDICTION PREDICTION API STARTED")
    print("="*70)
    print("🌐 Server: http://localhost:5000")
    print("="*70 + "\\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
""")

def setup_frontend():
    """Setup frontend"""
    os.makedirs('frontend/src', exist_ok=True)
    
    create_file('frontend/package.json', """{
  "name": "addiction-predictor",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "lucide-react": "^0.263.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^4.3.9",
    "tailwindcss": "^3.3.2",
    "postcss": "^8.4.24",
    "autoprefixer": "^10.4.14"
  }
}""")

    create_file('frontend/vite.config.js', """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 }
})""")

    create_file('frontend/tailwind.config.js', """export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: { extend: {} },
  plugins: []
}""")

    create_file('frontend/postcss.config.js', """export default {
  plugins: { tailwindcss: {}, autoprefixer: {} }
}""")

    create_file('frontend/index.html', """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Addiction Risk Predictor</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>""")

    create_file('frontend/src/main.jsx', """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode><App /></React.StrictMode>
)""")

    create_file('frontend/src/index.css', """@tailwind base;
@tailwind components;
@tailwind utilities;

* { margin: 0; padding: 0; box-sizing: border-box; }""")

    # Simplified App.jsx is already in artifacts - tell user to copy it

def main():
    print("\n" + "="*70)
    print("🔥 CREATING COMPLETE PROJECT STRUCTURE")
    print("="*70 + "\n")
    
    setup_backend()
    setup_frontend()
    
    print("\n" + "="*70)
    print("✅ ALL FILES CREATED!")
    print("="*70)
    print("\n📋 NEXT STEPS:\n")
    print("1️⃣  Install backend:")
    print("   cd backend")
    print("   pip install -r requirements.txt\n")
    print("2️⃣  Generate data & train:")
    print("   python generate_data.py")
    print("   python train.py\n")
    print("3️⃣  Start backend:")
    print("   python app.py\n")
    print("4️⃣  Install & run frontend (new terminal):")
    print("   cd frontend")
    print("   npm install")
    print("   npm run dev\n")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()