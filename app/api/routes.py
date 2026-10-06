from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.database.database import get_db
from app.market.constants import MAX_CANDLE_LIMIT, SUPPORTED_INTERVALS, SUPPORTED_SYMBOLS
from app.market.models import Candle
from app.market.repository import get_candles

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Crypto AI Trader",
        "environment": settings.app_env,
    }


@router.get("/market/candles", response_model=list[Candle])
def read_market_candles(
    symbol: str = Query(...),
    interval: str = Query(...),
    limit: int = Query(10, ge=1, le=MAX_CANDLE_LIMIT),
    session: Session = Depends(get_db),
) -> list[Candle]:
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported symbol; choose from {', '.join(sorted(SUPPORTED_SYMBOLS))}",
        )
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported interval; choose from {', '.join(sorted(SUPPORTED_INTERVALS))}",
        )
    return get_candles(session, symbol=symbol, interval=interval, limit=limit)
