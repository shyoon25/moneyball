from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "player_season_training.csv"


def load_player_data() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_PATH)


def get_all_players() -> list[dict]:
    df = load_player_data()
    if df.empty:
        return []
    return df.to_dict(orient="records")


def get_player_history(player_id: int) -> list[dict]:
    df = load_player_data()
    if df.empty:
        return []
    player_df = df[df["player_id"] == player_id].sort_values("season")
    return player_df.to_dict(orient="records")


def get_latest_player_row(player_id: int) -> dict | None:
    df = load_player_data()
    if df.empty:
        return None
    player_df = df[df["player_id"] == player_id].sort_values("season")
    if player_df.empty:
        return None
    return player_df.iloc[-1].to_dict()


def get_latest_rows_for_all_players() -> pd.DataFrame:
    df = load_player_data()
    if df.empty:
        return pd.DataFrame()
    latest = (
        df.sort_values(["player_id", "season"]).groupby("player_id", as_index=False).tail(1)
    )
    return latest.reset_index(drop=True)
