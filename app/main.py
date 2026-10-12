"""Application entry point for live paper trading and dashboard API.

Initializes database models, Binance market WebSocket ingestion, PaperExecutor,
AI trade explainer, alert handlers, and runs the FastAPI dashboard alongside
the PaperSession orchestrator in an asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI

from app.analysis.explainer import TradeExplainer
from app.api.dashboard import create_dashboard_router, set_dashboard_session
from app.api.routes import router as market_router
from app.backtest.models import BacktestConfig
from app.database import models as _market_models  # noqa: F401
from app.database import paper_models as _paper_models  # noqa: F401
from app.database.database import Base, engine
from app.market.constants import SUPPORTED_SYMBOLS
from app.market.websocket import BinanceSpotKlineStream
from app.trading.alerts import LoggingAlertHandler
from app.trading.paper import PaperExecutor
from app.trading.persistence import DatabaseAlertHandler
from app.trading.session import PaperSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
FIXED_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ZECUSDT"]
FALLBACK_DYNAMIC_SYMBOLS: list[str] = [
    "TAOUSDT",
    "FETUSDT",
    "RENDERUSDT",
    "NEARUSDT",
    "INJUSDT",
]
ALL_SYMBOLS: list[str] = list(dict.fromkeys(FIXED_SYMBOLS + FALLBACK_DYNAMIC_SYMBOLS))


async def fetch_dynamic_trending_symbols(
    limit: int = 5,
    *,
    async_client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Fetch trending coin symbols from CoinGecko and format as USDT pairs."""
    if limit <= 0:
        return []

    fallback = [s for s in FALLBACK_DYNAMIC_SYMBOLS if s not in FIXED_SYMBOLS][:limit]

    try:
        if async_client is not None:
            response = await async_client.get(COINGECKO_TRENDING_URL, timeout=4.0)
        else:
            async with httpx.AsyncClient(timeout=4.0) as client:
                response = await client.get(COINGECKO_TRENDING_URL)

        if response.status_code != 200:
            logger.warning(
                "CoinGecko trending API returned HTTP %s; using fallback symbols",
                response.status_code,
            )
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        payload = response.json()
        if not isinstance(payload, dict):
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        coins = payload.get("coins")
        if not isinstance(coins, list) or not coins:
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        discovered: list[str] = []
        for entry in coins:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item", entry)
            if not isinstance(item, dict):
                continue
            raw_symbol = item.get("symbol")
            if not isinstance(raw_symbol, str) or not raw_symbol.strip():
                continue
            clean_ticker = "".join(ch for ch in raw_symbol.strip().upper() if ch.isalnum())
            if not clean_ticker or clean_ticker in ("USDT", "USDC", "DAI", "FDUSD"):
                continue
            pair = clean_ticker if clean_ticker.endswith("USDT") else f"{clean_ticker}USDT"
            if pair not in FIXED_SYMBOLS and pair not in discovered:
                discovered.append(pair)
                if len(discovered) >= limit:
                    break

        if not discovered:
            SUPPORTED_SYMBOLS.update(fallback)
            return fallback

        SUPPORTED_SYMBOLS.update(discovered)
        return discovered

    except Exception as exc:
        logger.warning(
            "Failed to fetch dynamic trending symbols from CoinGecko (%s): %s; using fallback",
            type(exc).__name__,
            exc,
        )
        SUPPORTED_SYMBOLS.update(fallback)
        return fallback


def build_app(session: Any | None = None) -> FastAPI:
    """Build the unified FastAPI application with market and dashboard routes."""
    application = FastAPI(title="Crypto AI Trader")
    application.include_router(market_router)
    dashboard_router = create_dashboard_router(session)
    application.include_router(dashboard_router)
    return application


# Global FastAPI application instance for ASGI servers
app = build_app()


async def main(
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    symbols: tuple[str, ...] | list[str] | None = None,
    stream: Any | None = None,
    session: PaperSession | None = None,
    server: uvicorn.Server | None = None,
) -> None:
    """Initialize all subsystems and run paper session and API concurrently."""
    global ALL_SYMBOLS
    logger.info("Initializing Crypto AI Trader system...")

    # 1. Ensure database tables exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schemas verified.")
    except Exception as exc:
        logger.warning("Database schema verification warning: %s", exc)

    # 2. Initialize Paper Trading components if not injected
    if session is None:
        if symbols is not None:
            resolved_symbols = list(dict.fromkeys(symbols))
            SUPPORTED_SYMBOLS.update(resolved_symbols)
        else:
            dynamic_symbols = await fetch_dynamic_trending_symbols()
            ALL_SYMBOLS = list(dict.fromkeys(FIXED_SYMBOLS + dynamic_symbols))
            SUPPORTED_SYMBOLS.update(ALL_SYMBOLS)
            resolved_symbols = ALL_SYMBOLS

        kline_stream = stream or BinanceSpotKlineStream(symbols=resolved_symbols)
        config = BacktestConfig(
            minimum_quantity=Decimal("0.000001"),
            quantity_step=Decimal("0.000001"),
        )
        executor = PaperExecutor(config)
        explainer = TradeExplainer()
        logging_handler = LoggingAlertHandler()
        db_handler = DatabaseAlertHandler()

        session = PaperSession(
            stream=kline_stream,
            executor=executor,
            alert_handler=[logging_handler, db_handler],
            explainer=explainer,
        )

    set_dashboard_session(session)

    # 3. Build FastAPI application wired to the session
    api_app = build_app(session)

    # 4. Prepare Uvicorn server if not injected
    if server is None:
        server_config = uvicorn.Config(
            app=api_app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(server_config)

    # 5. Run PaperSession and FastAPI concurrently with graceful shutdown
    session_task = asyncio.create_task(session.run(), name="paper_session")
    server_task = asyncio.create_task(server.serve(), name="uvicorn_server")

    logger.info("Active trading symbols: %s", ALL_SYMBOLS)
    logger.info("Starting PaperSession and Dashboard API on http://%s:%d", host, port)

    try:
        done, pending = await asyncio.wait(
            [session_task, server_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for completed_task in done:
            task_name = completed_task.get_name()
            if completed_task.cancelled():
                logger.info("Task %s was cancelled.", task_name)
            elif completed_task.exception() is not None:
                exc = completed_task.exception()
                logger.error(
                    "Task %s crashed with exception: %s",
                    task_name,
                    exc,
                    exc_info=exc,
                )
            else:
                logger.info("Task %s exited normally.", task_name)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Termination signal received.")
    finally:
        logger.info("Shutting down paper session and API server...")
        session.stop()
        server.should_exit = True

        for task in (session_task, server_task):
            if not task.done():
                task.cancel()

        await asyncio.gather(session_task, server_task, return_exceptions=True)
        logger.info("All services successfully stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
