import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.backtest.models import BacktestConfig
from app.main import app, build_app, main
from app.market.models import Candle, CandleOpenObservation
from app.trading.paper import PaperExecutor
from app.trading.session import PaperSession

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
D = Decimal


class MockStream:
    """Mock stream that sleeps until cancelled."""

    async def events(self) -> AsyncIterator[Candle | CandleOpenObservation]:
        while True:
            await asyncio.sleep(0.01)
            if False:
                yield None  # type: ignore


class QuickExitServer:
    """Mock server that immediately exits."""

    def __init__(self) -> None:
        self.should_exit = False
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True
        await asyncio.sleep(0.02)
        self.should_exit = True


class InfiniteServer:
    """Mock server that runs until should_exit is set or cancelled."""

    def __init__(self) -> None:
        self.should_exit = False
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True
        while not self.should_exit:
            await asyncio.sleep(0.01)


def test_app_global_export_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_build_app_contains_all_routes():
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    test_app = build_app(session)
    client = TestClient(test_app)

    # Market routes
    assert client.get("/health").status_code == 200
    # Dashboard routes
    assert client.get("/").status_code == 200
    assert client.get("/ui").status_code == 200
    assert client.get("/portfolio").status_code == 200
    assert client.get("/positions").status_code == 200
    assert client.get("/trades").status_code == 200
    assert client.get("/signals").status_code == 200


@pytest.mark.asyncio
async def test_main_runs_and_shuts_down_when_server_exits():
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    server = QuickExitServer()

    await main(
        session=session,
        server=server,  # type: ignore
    )

    assert server.serve_called
    assert not session.is_running


@pytest.mark.asyncio
async def test_main_shuts_down_on_task_cancellation():
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    server = InfiniteServer()

    main_task = asyncio.create_task(
        main(session=session, server=server)  # type: ignore
    )
    await asyncio.sleep(0.03)

    assert session.is_running
    assert server.serve_called

    # Cancel the main task (simulating SIGINT or container stop)
    main_task.cancel()
    await main_task

    assert not session.is_running
    assert server.should_exit is True


@pytest.mark.asyncio
async def test_main_initializes_components_when_session_none():
    server = QuickExitServer()
    mock_stream = MockStream()

    await main(
        stream=mock_stream,
        server=server,  # type: ignore
    )

    assert server.serve_called


class CrashingServer:
    """Mock server that crashes during serve."""

    def __init__(self) -> None:
        self.should_exit = False
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True
        await asyncio.sleep(0.01)
        raise RuntimeError("Uvicorn failed to bind address")


@pytest.mark.asyncio
async def test_main_logs_crashed_task_exception(caplog):
    executor = PaperExecutor(BacktestConfig(minimum_quantity=D("0.000001")))
    session = PaperSession(stream=MockStream(), executor=executor)
    server = CrashingServer()

    with caplog.at_level(logging.ERROR):
        await main(
            session=session,
            server=server,  # type: ignore
        )

    assert "crashed with exception" in caplog.text
    assert "Uvicorn failed to bind address" in caplog.text
    assert not session.is_running


@pytest.mark.asyncio
async def test_fetch_dynamic_trending_symbols_parses_and_filters_fixed(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from app.main import FIXED_SYMBOLS, fetch_dynamic_trending_symbols

    assert FIXED_SYMBOLS == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ZECUSDT"]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "coins": [
            {"item": {"symbol": "btc"}},  # in FIXED_SYMBOLS -> filtered out
            {"item": {"symbol": "tao"}},
            {"item": {"symbol": "fet"}},
            {"item": {"symbol": "render"}},
        ]
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    symbols = await fetch_dynamic_trending_symbols(limit=2)
    assert symbols == ["TAOUSDT", "FETUSDT"]


@pytest.mark.asyncio
async def test_fetch_dynamic_trending_symbols_fallback_on_network_error(monkeypatch):
    from unittest.mock import AsyncMock

    import httpx

    from app.main import FALLBACK_DYNAMIC_SYMBOLS, fetch_dynamic_trending_symbols

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.RequestError("network down"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    symbols = await fetch_dynamic_trending_symbols(limit=5)
    assert symbols == FALLBACK_DYNAMIC_SYMBOLS
