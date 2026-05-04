# MLB Performance Predictor API

## Project overview
This project trains a machine learning model on real MLB batting data and serves predictions through a FastAPI backend.

Main target: predict a hitter's **next-season WAR** using current-season batting metrics.

## Why this is useful
- Turns real baseball performance history into forward-looking projections.
- Demonstrates an end-to-end ML system: ingestion, feature/target engineering, training, and API inference.
- Uses a season-based split for realistic evaluation.

## Tech stack
- Python 3.10+
- FastAPI + Uvicorn
- pandas
- scikit-learn
- pybaseball
- joblib
- pytest + httpx

## Data source
Data is fetched from `pybaseball.batting_stats(start_season, end_season, qual=100)`, which retrieves FanGraphs season-level batting data.

No fake sample data is used.

## How next-season prediction works
1. Pull seasons 2018-2024.
2. Keep selected batting/age/WAR features.
3. For each player-season row, assign `next_season_war` by shifting WAR one season forward within each player.
4. Drop rows that do not have a next season WAR target.
5. Train model to predict `next_season_war`.

## Setup commands
```bash
pip install -r requirements.txt
```

## Data refresh command
```bash
python app/services/data_ingestion.py
```

Or via API:
```bash
curl -X POST http://127.0.0.1:8000/data/refresh/2018/2024
```

## Model training command
```bash
python ml/train_model.py
```

## API run command
```bash
uvicorn app.main:app --reload
```

## Test command
```bash
pytest
```

## Example endpoint calls
```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/players
curl http://127.0.0.1:8000/players/18401
curl http://127.0.0.1:8000/predict/player/18401
curl http://127.0.0.1:8000/rankings/projected-war
curl -X POST http://127.0.0.1:8000/predict/custom-player \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "pa": 600,
    "hr": 30,
    "runs": 90,
    "rbi": 95,
    "sb": 10,
    "walk_rate": 0.10,
    "strikeout_rate": 0.22,
    "batting_avg": 0.275,
    "obp": 0.360,
    "slg": 0.500,
    "ops": 0.860,
    "iso": 0.225,
    "babip": 0.310,
    "woba": 0.370,
    "wrc_plus": 135,
    "war": 4.5
  }'
```

## Future improvements
- Add model comparison (XGBoost, ElasticNet).
- Add confidence intervals for predictions.
- Add pitcher model endpoints.
- Add scheduled retraining.
