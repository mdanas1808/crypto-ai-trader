"""Read-only dashboard REST API for real-time paper trading sessions.

Exposes portfolio equity, active positions, closed trades, and pending signals
via standard FastAPI JSON endpoints without mutating state.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from app.backtest.models import Fill, Position, Trade
from app.strategy.breakout import BreakoutSignal


class FillResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    exchange: str
    side: str
    timestamp: datetime
    reference_price: Decimal
    price: Decimal
    quantity: Decimal
    fee: Decimal
    reason: str
    notional: Decimal
    observed_at: datetime | None = None

    @classmethod
    def from_fill(cls, fill: Fill) -> FillResponse:
        return cls(
            symbol=fill.symbol,
            exchange=fill.exchange,
            side=fill.side,
            timestamp=fill.timestamp,
            reference_price=fill.reference_price,
            price=fill.price,
            quantity=fill.quantity,
            fee=fill.fee,
            reason=fill.reason,
            notional=fill.notional,
            observed_at=fill.observed_at,
        )


class SignalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    exchange: str
    symbol: str
    side: str
    decision_time: datetime
    signal_open_time: datetime
    atr_14: float
    explanation: str | None = None
    status: str | None = None

    @classmethod
    def from_signal(cls, signal: BreakoutSignal) -> SignalResponse:
        return cls(
            exchange=signal.exchange,
            symbol=signal.symbol,
            side=signal.side,
            decision_time=signal.decision_time,
            signal_open_time=signal.signal_open_time,
            atr_14=signal.atr_14,
            explanation=getattr(signal, "explanation", None),
            status=getattr(signal, "status", "PENDING"),
        )


class PositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    exchange: str
    quantity: Decimal
    entry_fill: FillResponse
    stop_price: Decimal
    target_price: Decimal
    initial_risk: Decimal
    signal: SignalResponse

    @classmethod
    def from_position(cls, position: Position) -> PositionResponse:
        return cls(
            symbol=position.symbol,
            exchange=position.exchange,
            quantity=position.quantity,
            entry_fill=FillResponse.from_fill(position.entry_fill),
            stop_price=position.stop_price,
            target_price=position.target_price,
            initial_risk=position.initial_risk,
            signal=SignalResponse.from_signal(position.signal),
        )


class TradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    exchange: str
    quantity: Decimal
    entry_fill: FillResponse
    exit_fill: FillResponse
    stop_price: Decimal
    target_price: Decimal
    exit_reason: str
    gross_pnl: Decimal
    net_pnl: Decimal
    signal: SignalResponse
    explanation: str | None = None

    @classmethod
    def from_trade(cls, trade: Trade) -> TradeResponse:
        return cls(
            symbol=trade.signal.symbol,
            exchange=trade.signal.exchange,
            quantity=trade.quantity,
            entry_fill=FillResponse.from_fill(trade.entry_fill),
            exit_fill=FillResponse.from_fill(trade.exit_fill),
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            exit_reason=trade.exit_reason,
            gross_pnl=trade.gross_pnl,
            net_pnl=trade.net_pnl,
            signal=SignalResponse.from_signal(trade.signal),
            explanation=getattr(trade, "explanation", None),
        )


class PortfolioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cash: Decimal
    equity: Decimal
    open_risk: Decimal
    daily_loss_breached: bool


class DashboardState:
    """Thread-safe holder for the global active session/executor reference."""

    def __init__(self) -> None:
        self._session: Any = None

    def set_session(self, session: Any) -> None:
        self._session = session

    def get_session(self) -> Any:
        if self._session is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Paper session is not initialized or running",
            )
        return self._session


dashboard_state = DashboardState()


def get_active_session() -> Any:
    """Dependency returning the currently active dashboard paper session."""
    return dashboard_state.get_session()


def set_dashboard_session(session: Any) -> None:
    """Set the active paper session or executor for dashboard endpoints."""
    dashboard_state.set_session(session)


def _resolve_executor(session_or_executor: Any) -> Any:
    """Extract the PaperExecutor from a session or return target directly."""
    if hasattr(session_or_executor, "executor"):
        return session_or_executor.executor
    return session_or_executor


def create_dashboard_router(
    session_source: Any | Callable[[], Any] | None = None,
) -> APIRouter:
    """Create a FastAPI APIRouter configured for a specific paper session source."""
    router = APIRouter()

    def _get_target_session() -> Any:
        if callable(session_source):
            target = session_source()
        elif session_source is not None:
            target = session_source
        else:
            target = get_active_session()

        if target is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Paper session is not available",
            )
        return target

    @router.get("/", response_class=FileResponse)
    @router.get("/ui", response_class=FileResponse)
    def serve_ui() -> FileResponse:
        static_index = Path(__file__).resolve().parent.parent / "static" / "index.html"
        if not static_index.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dashboard static index.html not found",
            )
        return FileResponse(static_index, media_type="text/html")

    @router.get("/portfolio", response_model=PortfolioResponse)
    def get_portfolio(
        session: Any = Depends(_get_target_session),
    ) -> PortfolioResponse:
        executor = _resolve_executor(session)
        portfolio = getattr(executor, "portfolio", None)
        marks = getattr(executor, "_marks", {})

        cash = getattr(executor, "cash", Decimal("0"))
        if portfolio is not None:
            equity = portfolio.equity(marks)
            open_risk = portfolio.open_risk
            daily_loss_breached = portfolio.daily_loss_breached
        else:
            equity = getattr(executor, "equity", cash)
            open_risk = getattr(executor, "open_risk", Decimal("0"))
            daily_loss_breached = getattr(executor, "daily_loss_breached", False)

        return PortfolioResponse(
            cash=cash,
            equity=equity,
            open_risk=open_risk,
            daily_loss_breached=daily_loss_breached,
        )

    @router.get("/positions", response_model=list[PositionResponse])
    def get_positions(
        session: Any = Depends(_get_target_session),
    ) -> list[PositionResponse]:
        executor = _resolve_executor(session)
        positions = getattr(executor, "positions", {})
        pos_list = list(positions.values()) if isinstance(positions, dict) else list(positions)
        return [PositionResponse.from_position(p) for p in pos_list]

    @router.get("/trades", response_model=list[TradeResponse])
    def get_trades(
        session: Any = Depends(_get_target_session),
    ) -> list[TradeResponse]:
        executor = _resolve_executor(session)
        trades = getattr(executor, "trades", ())
        return [TradeResponse.from_trade(t) for t in trades]

    @router.get("/signals", response_model=list[SignalResponse])
    def get_signals(
        session: Any = Depends(_get_target_session),
    ) -> list[SignalResponse]:
        signals = getattr(session, "pending_signals", None)
        if signals is None:
            executor = _resolve_executor(session)
            signals = getattr(executor, "pending_signals", ())
        vetoed = getattr(session, "vetoed_signals", ())
        combined = list(signals) + list(vetoed)
        return [SignalResponse.from_signal(s) for s in combined]

    return router


def create_dashboard_app(
    session_source: Any | Callable[[], Any] | None = None,
) -> FastAPI:
    """Create a standalone read-only dashboard FastAPI application."""
    app = FastAPI(title="Crypto AI Trader Dashboard")
    router = create_dashboard_router(session_source)
    app.include_router(router)
    return app


# Default router using the global active session dependency
router = create_dashboard_router(None)
