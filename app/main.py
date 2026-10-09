"""Application entry point for live paper trading and dashboard API.

Initializes database models, Binance market WebSocket ingestion, PaperExecutor,
AI trade explainer, alert handlers, and runs the FastAPI dashboard alongside
the PaperSession orchestrator in an asyncio event loop.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import logging
import signal
from typing import Any

from fastapi import FastAPI
import uvicorn

from app.analysis.explainer import TradeExplainer
from app.api.dashboard import create_dashboard_router, set_dashboard_session
from app.api.routes import router as market_router
from app.backtest.models import BacktestConfig
from app.database import models as _market_models  # noqa: F401
from app.database import paper_models as _paper_models  # noqa: F401
from app.database.database import Base, engine
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
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    stream: Any | None = None,
    session: PaperSession | None = None,
    server: uvicorn.Server | None = None,
) -> None:
    """Initialize all subsystems and run paper session and API concurrently."""
    logger.info("Initializing Crypto AI Trader system...")

    # 1. Ensure database tables exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schemas verified.")
    except Exception as exc:
        logger.warning("Database schema verification warning: %s", exc)

    # 2. Initialize Paper Trading components if not injected
    if session is None:
        kline_stream = stream or BinanceSpotKlineStream(symbols=symbols)
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

    logger.info("Starting PaperSession and Dashboard API on http://%s:%d", host, port)

    try:
        done, pending = await asyncio.wait(
            [session_task, server_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
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
