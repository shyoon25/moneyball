from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "player_season_training.csv"
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
FEATURES_PATH = BASE_DIR / "ml" / "features.json"

FEATURES = [
    "age",
    "pa",
    "hr",
    "runs",
    "rbi",
    "sb",
    "walk_rate",
    "strikeout_rate",
    "batting_avg",
    "obp",
    "slg",
    "ops",
    "iso",
    "babip",
    "woba",
    "wrc_plus",
    "war",
]
TARGET = "next_season_war"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. Run data ingestion first."
        )

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Training dataset is empty.")

    train_df = df[df["season"] <= 2022].copy()
    test_df = df[df["season"] >= 2023].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split by season produced an empty split.")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with FEATURES_PATH.open("w", encoding="utf-8") as f:
        json.dump(FEATURES, f, indent=2)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved features: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
