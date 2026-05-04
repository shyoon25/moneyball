from fastapi import FastAPI

from app.routes.data import router as data_router
from app.routes.players import router as players_router
from app.routes.predictions import rankings_router, router as predictions_router
from app.schemas import RootResponse

app = FastAPI(title="MLB Performance Predictor API")


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        project="MLB Performance Predictor API",
        status="running",
        model_target="next_season_war",
    )


app.include_router(data_router)
app.include_router(players_router)
app.include_router(predictions_router)
app.include_router(rankings_router)
