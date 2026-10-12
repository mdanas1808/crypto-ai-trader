from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.main import app
from app.market.models import Candle
from app.market.repository import persist_candles


def test_market_api_reads_database_chronologically(db_session: Session, monkeypatch) -> None:
    def fail_if_exchange_is_called(*args, **kwargs):
        raise AssertionError("market API must not call the exchange client")

    monkeypatch.setattr("app.market.exchange.BinanceSpotClient", fail_if_exchange_is_called)
    candles = [
        Candle(
            exchange="binance_spot",
            symbol="BTCUSDT",
            interval="1h",
            open_time=datetime(2024, 1, day, tzinfo=timezone.utc),
            open="100",
            high="110",
            low="90",
            close=str(100 + day),
            volume="2.5",
        )
        for day in (1, 2, 3)
    ]
    persist_candles(db_session, candles)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as client:
            response = client.get("/market/candles?symbol=BTCUSDT&interval=1h&limit=2")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    result = response.json()
    assert len(result) == 2
    assert [row["open_time"] for row in result] == [
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
    ]
    assert result[0]["exchange"] == "binance_spot"
    assert result[0]["open"] == "100.000000000000000000"


def test_market_api_rejects_invalid_symbol_and_interval() -> None:
    with TestClient(app) as client:
        invalid_symbol = client.get("/market/candles?symbol=NOTREAL&interval=1h")
        invalid_interval = client.get("/market/candles?symbol=BTCUSDT&interval=15m")
        low_limit = client.get("/market/candles?symbol=BTCUSDT&interval=1h&limit=0")
        high_limit = client.get("/market/candles?symbol=BTCUSDT&interval=1h&limit=1001")

    assert invalid_symbol.status_code == 422
    assert invalid_interval.status_code == 422
    assert low_limit.status_code == 422
    assert high_limit.status_code == 422


def test_market_api_returns_empty_list_without_calling_exchange(
    db_session: Session, monkeypatch
) -> None:
    def fail_if_exchange_is_called(*args, **kwargs):
        raise AssertionError("market API must not call the exchange client")

    monkeypatch.setattr("app.market.exchange.BinanceSpotClient", fail_if_exchange_is_called)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as client:
            response = client.get("/market/candles?symbol=BTCUSDT&interval=1h")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json() == []
