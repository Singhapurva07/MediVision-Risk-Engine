from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app)

# ------------------ LOAD MODELS + ARTIFACTS ------------------

stacker = joblib.load("./data/models/meta_model.pkl")
base_models = joblib.load("./data/models/base_models.pkl")

scaler = joblib.load("./data/processed/scaler.pkl")
feature_cols = joblib.load("./data/processed/features.pkl")
encoders = joblib.load("./data/processed/encoders.pkl")

print("\n⚡ MODELS + SCALER + ENCODERS LOADED\n")


# ------------------ CATEGORY NORMALIZATION ------------------

value_maps = {
    "gender": {
        "male": "M", "m": "M",
        "female": "F", "f": "F"
    },

    "income_class": {
        "mid": "middle", "middle": "middle", "medium": "middle",
        "high": "high", "upper": "high",
        "low": "low", "lower": "low"
    },

    "drug_class": {
        "opioid": "Opioid",
        "morphine": "Opioid",
        "fentanyl": "Opioid",

        "stimulant": "Stimulant",
        "adderall": "Stimulant",

        "sedative": "Sedative",
        "zolpidem": "Sedative",

        "benzodiazepine": "Benzodiazepine",
        "benzo": "Benzodiazepine"
    }
}


def normalize_values(df):
    for col, mapping in value_maps.items():
        if col in df:

            raw = str(df[col].iloc[0]).strip().lower()

            if raw in mapping:
                df[col] = mapping[raw]
            else:
                # default fallback to first allowed category
                df[col] = list(mapping.values())[0]

    return df


# ------------------ PREDICT ROUTE ------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("\nREQUEST RECEIVED:", data)
        
        df = pd.DataFrame([data])

        # normalize case for allowed categorical cols
        normalize_cols = ["gender", "income_class", "drug_class"]
        for col in normalize_cols:
            if col in df:
                df[col] = df[col].astype(str).str.lower().str.strip()

        # fix unseen categorical labels automatically
        for col, enc in encoders.items():
            if col in df:
                allowed = list(enc.classes_)
                if df[col].iloc[0] not in allowed:
                    print(f"⚠ unseen {col}={df[col].iloc[0]} → switching to {allowed[0]}")
                    df[col] = allowed[0]

                df[col] = enc.transform(df[col])

        # scale complete input
        X = df[feature_cols]
        X_scaled = scaler.transform(X)

        # predict using stacked model - FIX IS HERE
        base_outputs = []
        for model_name, model in base_models.items():  # Changed from 'for model in base_models'
            base_outputs.append(model.predict(X_scaled)[0])

        meta_input = [base_outputs]

        pred = float(stacker.predict(meta_input)[0])

        return jsonify({"success": True, "risk_score": pred})

    except Exception as e:
        print("❌ BACKEND ERROR:", e)
        traceback.print_exc()  # Added for better debugging
        return jsonify({"success": False, "error": str(e)}), 400

# ------------------ HEALTH CHECK ------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


# ------------------ RUN APP ------------------

if __name__ == "__main__":
    print("\n==============================================")
    print("🔥 ADDICTION PREDICTION API RUNNING ON PORT 5000")
    print("==============================================\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
