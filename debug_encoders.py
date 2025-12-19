import joblib

encoders = joblib.load("./data/processed/encoders.pkl")

for col, enc in encoders.items():
    print(col, list(enc.classes_))
