from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
FEATURES_PATH = BASE_DIR / "ml" / "features.json"


class PredictionServiceError(Exception):
    pass


def _load_model():
    if not MODEL_PATH.exists():
        raise PredictionServiceError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)


def _load_features() -> list[str]:
    if not FEATURES_PATH.exists():
        raise PredictionServiceError("Feature file not found. Train the model first.")
    with FEATURES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_from_record(record: dict) -> float:
    model = _load_model()
    features = _load_features()

    feature_values = {}
    for feature in features:
        if feature not in record:
            raise PredictionServiceError(f"Missing feature '{feature}' in input record.")
        feature_values[feature] = float(record[feature])

    input_df = pd.DataFrame([feature_values])
    prediction = model.predict(input_df)[0]
    return float(prediction)
