from fastapi import APIRouter, HTTPException

from app.schemas import RefreshDataResponse
from app.services.data_ingestion import DataIngestionError, refresh_data

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/refresh/{start_year}/{end_year}", response_model=RefreshDataResponse)
def refresh_dataset(start_year: int, end_year: int) -> RefreshDataResponse:
    if start_year > end_year:
        raise HTTPException(status_code=400, detail="start_year must be <= end_year")
    try:
        raw_rows, processed_rows = refresh_data(start_year, end_year)
    except DataIngestionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RefreshDataResponse(
        start_year=start_year,
        end_year=end_year,
        raw_rows=raw_rows,
        processed_rows=processed_rows,
    )
