"""Prepare historical candles and indicators for offline backtests."""

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.indicators.technical import calculate_indicators
from app.market.constants import EXCHANGE
from app.market.repository import get_candles_in_range


def prepare_backtest_data(
    session: Session,
    symbol: str,
    interval: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    exchange: str = EXCHANGE,
) -> pd.DataFrame:
    """Read one bounded candle series and calculate its indicators in memory."""
    candles = get_candles_in_range(
        session,
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        exchange=exchange,
    )
    return calculate_indicators(candles)
