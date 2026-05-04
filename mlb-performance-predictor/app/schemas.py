from pydantic import BaseModel


class RootResponse(BaseModel):
    project: str
    status: str
    model_target: str


class RefreshDataResponse(BaseModel):
    start_year: int
    end_year: int
    raw_rows: int
    processed_rows: int


class PlayerPredictionResponse(BaseModel):
    player_id: int
    name: str
    latest_season: int
    predicted_next_season_war: float


class CustomPlayerInput(BaseModel):
    age: float
    pa: float
    hr: float
    runs: float
    rbi: float
    sb: float
    walk_rate: float
    strikeout_rate: float
    batting_avg: float
    obp: float
    slg: float
    ops: float
    iso: float
    babip: float
    woba: float
    wrc_plus: float
    war: float


class CustomPredictionResponse(BaseModel):
    predicted_next_season_war: float
