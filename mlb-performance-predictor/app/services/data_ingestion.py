from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from pybaseball import batting_stats

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_PATH_TEMPLATE = "batting_stats_{start}_{end}.csv"
PROCESSED_PATH = PROCESSED_DIR / "player_season_training.csv"

COLUMN_MAP: Dict[str, str] = {
    "IDfg": "player_id",
    "Name": "name",
    "Team": "team",
    "Season": "season",
    "Age": "age",
    "PA": "pa",
    "HR": "hr",
    "R": "runs",
    "RBI": "rbi",
    "SB": "sb",
    "BB%": "walk_rate",
    "K%": "strikeout_rate",
    "AVG": "batting_avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "ISO": "iso",
    "BABIP": "babip",
    "wOBA": "woba",
    "wRC+": "wrc_plus",
    "WAR": "war",
}


class DataIngestionError(Exception):
    pass


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _patch_requests_user_agent() -> None:
    original_get = requests.get

    def wrapped_get(url, *args, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault(
            "User-Agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        headers.setdefault("Referer", "https://www.fangraphs.com/")
        return original_get(url, *args, headers=headers, **kwargs)

    requests.get = wrapped_get


def _fetch_batting_data(start_year: int, end_year: int) -> pd.DataFrame:
    _patch_requests_user_agent()
    try:
        return batting_stats(start_year, end_year, qual=100)
    except Exception:
        yearly_frames: List[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            yearly_frames.append(batting_stats(year, year, qual=100))
        if not yearly_frames:
            raise
        return pd.concat(yearly_frames, ignore_index=True)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in COLUMN_MAP if col not in df.columns]
    if missing:
        raise DataIngestionError(
            f"Missing expected columns from pybaseball response: {missing}"
        )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()

    numeric_columns = [
        "player_id",
        "season",
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
    for col in numeric_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned["name"] = cleaned["name"].fillna("Unknown")
    cleaned["team"] = cleaned["team"].fillna("Unknown")

    for col in numeric_columns:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    cleaned = cleaned.sort_values(["player_id", "season"]).reset_index(drop=True)
    cleaned["next_season_war"] = cleaned.groupby("player_id")["war"].shift(-1)

    training = cleaned.dropna(subset=["next_season_war"]).copy()
    training["next_season_war"] = pd.to_numeric(training["next_season_war"], errors="coerce")
    training = training.dropna(subset=["next_season_war"]).reset_index(drop=True)

    return training


def refresh_data(start_year: int, end_year: int) -> Tuple[int, int]:
    _ensure_dirs()

    try:
        raw_df = _fetch_batting_data(start_year, end_year)
    except Exception as exc:
        raise DataIngestionError(
            "Failed to fetch batting stats from pybaseball/FanGraphs. "
            "Please retry shortly in case of temporary rate-limits."
        ) from exc

    if raw_df.empty:
        raise DataIngestionError("pybaseball returned no batting data.")

    _validate_columns(raw_df)

    raw_path = RAW_DIR / RAW_PATH_TEMPLATE.format(start=start_year, end=end_year)
    raw_df.to_csv(raw_path, index=False)

    processed_df = _clean_dataframe(raw_df)
    processed_df.to_csv(PROCESSED_PATH, index=False)

    return len(raw_df), len(processed_df)


def main() -> None:
    start_year = 2018
    end_year = 2024
    raw_rows, processed_rows = refresh_data(start_year, end_year)
    print(f"Saved raw rows: {raw_rows}")
    print(f"Saved processed rows: {processed_rows}")
    print(f"Processed dataset: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
