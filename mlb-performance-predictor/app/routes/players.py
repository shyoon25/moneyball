from fastapi import APIRouter, HTTPException

from app.services.data_service import get_all_players, get_player_history

router = APIRouter(tags=["players"])


@router.get("/players")
def list_players() -> list[dict]:
    return get_all_players()


@router.get("/players/{player_id}")
def player_history(player_id: int) -> list[dict]:
    history = get_player_history(player_id)
    if not history:
        raise HTTPException(status_code=404, detail="Player not found")
    return history
