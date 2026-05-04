from fastapi import APIRouter, HTTPException

from app.schemas import (
    CustomPlayerInput,
    CustomPredictionResponse,
    PlayerPredictionResponse,
)
from app.services.data_service import (
    get_latest_player_row,
    get_latest_rows_for_all_players,
    load_player_data,
)
from app.services.prediction_service import PredictionServiceError, predict_from_record

router = APIRouter(prefix="/predict", tags=["predictions"])
rankings_router = APIRouter(tags=["predictions"])


@router.get("/player/{player_id}", response_model=PlayerPredictionResponse)
def predict_player(player_id: int) -> PlayerPredictionResponse:
    if load_player_data().empty:
        raise HTTPException(status_code=404, detail="No processed player data found")

    player_row = get_latest_player_row(player_id)
    if not player_row:
        raise HTTPException(status_code=404, detail="Player not found")

    try:
        prediction = predict_from_record(player_row)
    except PredictionServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return PlayerPredictionResponse(
        player_id=int(player_row["player_id"]),
        name=str(player_row["name"]),
        latest_season=int(player_row["season"]),
        predicted_next_season_war=round(prediction, 3),
    )


@router.post("/custom-player", response_model=CustomPredictionResponse)
def predict_custom_player(payload: CustomPlayerInput) -> CustomPredictionResponse:
    try:
        prediction = predict_from_record(payload.model_dump())
    except PredictionServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return CustomPredictionResponse(predicted_next_season_war=round(prediction, 3))


@rankings_router.get("/rankings/projected-war")
def projected_war_rankings() -> list[dict]:
    latest_rows = get_latest_rows_for_all_players()
    if latest_rows.empty:
        raise HTTPException(status_code=404, detail="No processed player data found")

    results = []
    for _, row in latest_rows.iterrows():
        row_dict = row.to_dict()
        try:
            prediction = predict_from_record(row_dict)
        except PredictionServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        results.append(
            {
                "player_id": int(row_dict["player_id"]),
                "name": str(row_dict["name"]),
                "latest_season": int(row_dict["season"]),
                "predicted_next_season_war": round(float(prediction), 3),
            }
        )

    results.sort(key=lambda item: item["predicted_next_season_war"], reverse=True)
    return results[:25]
