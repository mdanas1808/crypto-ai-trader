"""Asynchronous paper trading session orchestrator.

Coordinates public Binance Kline event streams, deterministic breakout
strategy evaluation, the real-time PaperExecutor, and optional alerting/explanations.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import signal
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from app.analysis.eval_logger import FundamentalEvalLogger
from app.analysis.explainer import TradeExplainer
from app.analysis.fundamentals import (
    NewsFetchError,
    fetch_recent_news,
    format_news_xml,
)
from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import Position, Trade
from app.market.constants import FIXED_SYMBOLS, SUPPORTED_SYMBOLS
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal, generate_breakout_signals
from app.trading.alerts import AlertHandler, CompositeAlertHandler
from app.trading.momentum import BtcRegimeFilter
from app.trading.paper import PaperExecutor

logger = logging.getLogger(__name__)

DEFAULT_REDUCE_RATIO = Decimal("0.50")

VARIANT_A = "variant_a"  # AI Gate + Regime Filter (Current)
VARIANT_B = "variant_b"  # Regime Filter only (No AI Gate)
VARIANT_C = "variant_c"  # AI Gate only (No Regime Filter)
VARIANT_D = "variant_d"  # Baseline (No Gate, No Regime Filter)
VARIANT_E = "variant_e"  # AI Gate + Regime Filter applied to Core Alts too


def _compute_evidence_hash(news: Any) -> str:
    """Compute SHA256 hash of concatenated headlines evaluated."""
    if isinstance(news, list):
        headlines = "".join(str(item.get("headline", "")) for item in news)
    elif isinstance(news, str):
        matches = re.findall(r"\]\s*(.*?)(?:\s*::|\s*</|\n|$)", news)
        if matches:
            headlines = "".join(m.strip() for m in matches if m.strip())
        else:
            headlines = news.strip()
    else:
        headlines = str(news).strip()
    return hashlib.sha256(headlines.encode("utf-8")).hexdigest()


class PaperSession:
    """Orchestrate live paper trading events and deterministic multi-portfolio signal execution."""

    def __init__(
        self,
        stream: Any,
        executor: PaperExecutor | None = None,
        strategy: Callable[..., Sequence[BreakoutSignal]] | None = None,
        *,
        data_provider: Callable[[Candle], MultiTimeframeBacktestData | None] | None = None,
        alert_handler: AlertHandler | Sequence[AlertHandler] | None = None,
        explainer: TradeExplainer | None = None,
        review_interval: float = 1800.0,
        history_file: str | None = None,
        eval_logger: FundamentalEvalLogger | None = None,
        btc_regime_filter: BtcRegimeFilter | Callable[[], bool | Awaitable[bool]] | None = None,
        wait_if_live: bool = False,
        executors: dict[str, PaperExecutor] | None = None,
    ) -> None:
        if executors is not None:
            self.executors = dict(executors)
        elif isinstance(executor, PaperExecutor):
            self.executors = {
                "variant_a": executor,
                "variant_b": PaperExecutor(executor.config),
                "variant_c": PaperExecutor(executor.config),
                "variant_d": PaperExecutor(executor.config),
                "variant_e": PaperExecutor(executor.config),
            }
        else:
            raise TypeError("executor must be PaperExecutor, or executors dictionary provided")

        if not hasattr(stream, "events") or not callable(stream.events):
            raise TypeError("stream must provide a callable events() method")

        self.stream = stream
        self.strategy = strategy if strategy is not None else generate_breakout_signals
        self.data_provider = data_provider
        if isinstance(alert_handler, (list, tuple, set)):
            self.alert_handler: AlertHandler | None = CompositeAlertHandler(alert_handler)
        else:
            self.alert_handler = alert_handler
        self.explainer = explainer
        self._review_interval = review_interval
        self._history_file = str(history_file) if history_file is not None else None
        self.eval_logger = eval_logger
        self.btc_regime_filter = btc_regime_filter
        self._wait_if_live = wait_if_live

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._review_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._events_processed = 0
        self._signals_submitted = 0
        self._daily_loss_notified = False
        self._vetoed_signals: list[BreakoutSignal] = []
        self._portfolio_lock = asyncio.Lock()
        self._entry_eval_cache: dict[tuple[str, Any], tuple[bool, str, dict[str, Any], Any]] = {}
        self._position_review_cache: dict[
            tuple[str, Any], tuple[bool, str, dict[str, Any], Any]
        ] = {}
        self._position_review_history: dict[str, dict[str, Any]] = self._load_review_history()

    @property
    def executor(self) -> PaperExecutor:
        """Primary executor representing Variant A (AI Gate + Regime Filter)."""
        return self.executors["variant_a"]

    @executor.setter
    def executor(self, value: PaperExecutor) -> None:
        self.executors["variant_a"] = value

    def _should_submit_to_variant(
        self,
        variant_name: str,
        symbol: str,
        is_btc_bullish: bool,
        ai_approved: bool,
    ) -> bool:
        """Evaluate submission eligibility for each isolated experiment variant."""
        is_core = symbol in FIXED_SYMBOLS
        is_btc_or_eth = symbol in ("BTCUSDT", "ETHUSDT")
        is_core_alt = is_core and not is_btc_or_eth

        if variant_name == "variant_a":
            # Variant A: AI Gate + Regime Filter (Current)
            if is_core:
                return True
            return is_btc_bullish and ai_approved

        elif variant_name == "variant_b":
            # Variant B: Regime Filter only (No AI Gate)
            if is_core:
                return True
            return is_btc_bullish

        elif variant_name == "variant_c":
            # Variant C: AI Gate only (No Regime Filter)
            if is_core:
                return True
            return ai_approved

        elif variant_name == "variant_d":
            # Variant D: Baseline (No Gate, No Regime Filter)
            return True

        elif variant_name == "variant_e":
            # Variant E: AI Gate + Regime Filter applied to Core Alts too
            # (SOL, BNB, ZEC blocked during BTC downtrend; only BTC/ETH untouched)
            if is_btc_or_eth:
                return True
            if is_core_alt:
                return is_btc_bullish
            return is_btc_bullish and ai_approved

        return False

    def _load_review_history(self) -> dict[str, dict[str, Any]]:
        """Load persisted position review history from disk."""
        if not self._history_file:
            return {}
        try:
            if os.path.exists(self._history_file):
                with open(self._history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception as exc:
            logger.warning(
                "Failed to load position review history from %s: %s",
                self._history_file,
                exc,
            )
        return {}

    def _save_review_history(self) -> None:
        """Persist position review history to disk."""
        if not self._history_file:
            return
        try:
            with open(self._history_file, "w", encoding="utf-8") as f:
                json.dump(self._position_review_history, f, indent=2)
        except Exception as exc:
            logger.warning(
                "Failed to save position review history to %s: %s",
                self._history_file,
                exc,
            )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def signals_submitted(self) -> int:
        return self._signals_submitted

    @property
    def positions(self) -> dict[str, Position]:
        return self.executor.positions

    @property
    def trades(self) -> tuple[Trade, ...]:
        return self.executor.trades

    @property
    def pending_signals(self) -> tuple[BreakoutSignal, ...]:
        return self.executor.pending_signals

    @property
    def vetoed_signals(self) -> tuple[BreakoutSignal, ...]:
        return tuple(self._vetoed_signals)

    @property
    def review_task(self) -> asyncio.Task[None] | None:
        return self._review_task

    async def _notify(self, method_name: str, *args: Any) -> None:
        """Safely invoke an alert handler callback without interrupting execution."""
        if self.alert_handler is None:
            return
        handler_method = getattr(self.alert_handler, method_name, None)
        if handler_method is None or not callable(handler_method):
            return
        try:
            res = handler_method(*args)
            if inspect.iscoroutine(res):
                await res
        except Exception as exc:
            logger.warning(
                "Error in alert handler callback %s: %s",
                method_name,
                exc,
                exc_info=True,
            )

    async def _check_daily_loss(self) -> None:
        """Trigger callback when daily loss transitions to breached."""
        is_breached = self.executor.daily_loss_breached
        if is_breached and not self._daily_loss_notified:
            self._daily_loss_notified = True
            await self._notify("on_daily_loss_breached", self.executor)
        elif not is_breached and self._daily_loss_notified:
            self._daily_loss_notified = False

    async def _check_btc_regime(self) -> bool:
        """Check whether BTC is in an uptrend (>= 50-day SMA). Returns True if uptrend/bullish."""
        if self.btc_regime_filter is None:
            return True
        try:
            if hasattr(self.btc_regime_filter, "is_uptrend"):
                res = self.btc_regime_filter.is_uptrend()
                if inspect.isawaitable(res):
                    return bool(await res)
                return bool(res)
            elif callable(self.btc_regime_filter):
                res = self.btc_regime_filter()
                if inspect.isawaitable(res):
                    return bool(await res)
                return bool(res)
        except Exception as exc:
            logger.warning("Error evaluating BTC regime filter: %s", exc)
        return True

    async def _build_signal_explanation(self, sig: BreakoutSignal, context_data: Any) -> str:
        if self.explainer is None:
            return ""
        if isinstance(context_data, MultiTimeframeBacktestData):
            try:
                res = self.explainer.explain_signal(sig, context_data)
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception as exc:
                logger.warning("Failed to generate signal explanation: %s", exc)
                return ""
        return (
            f"Breakout Long Signal for {sig.symbol} "
            f"({sig.exchange}) at {sig.decision_time.isoformat()}."
        )

    async def process_event(self, event: Candle | CandleOpenObservation) -> Position | Trade | None:
        """Route a single market event to executor, strategy, and alert handlers."""
        if isinstance(event, CandleOpenObservation):
            positions: dict[str, Position] = {}
            async with self._portfolio_lock:
                for var_key, exec_inst in self.executors.items():
                    pos = exec_inst.on_open_observation(event)
                    if pos is not None:
                        positions[var_key] = pos
            position = positions.get("variant_a")
            self._events_processed += 1
            if position is not None and self.alert_handler is not None:
                await self._notify("on_entry_filled", position)
            await self._check_daily_loss()
            return position

        if isinstance(event, Candle):
            trade = None
            if event.interval == "1h":
                trades: dict[str, Trade] = {}
                async with self._portfolio_lock:
                    for var_key, exec_inst in self.executors.items():
                        tr = exec_inst.on_candle_closed(event)
                        if tr is not None:
                            trades[var_key] = tr
                trade = trades.get("variant_a")
                if trade is not None:
                    self._position_review_history.pop(trade.signal.symbol, None)
                    self._save_review_history()
                    explanation = ""
                    if self.explainer is not None:
                        try:
                            res = self.explainer.explain_trade_exit(trade)
                            if inspect.isawaitable(res):
                                explanation = await res
                            else:
                                explanation = res
                        except Exception as exc:
                            logger.warning("Failed to generate trade exit explanation: %s", exc)
                    if explanation:
                        try:
                            object.__setattr__(trade, "explanation", explanation)
                        except Exception:
                            pass
                    if self.alert_handler is not None:
                        await self._notify("on_exit_filled", trade, explanation)

                signals, context_data = await self._generate_signals_with_context(event)
                for sig in signals:
                    SUPPORTED_SYMBOLS.add(sig.symbol)

                    # 1. Determine BTC Regime Filter status
                    is_btc_bullish = True
                    if self.btc_regime_filter is not None:
                        is_btc_bullish = await self._check_btc_regime()
                    regime_state = "UPTREND" if is_btc_bullish else "DOWNTREND"
                    try:
                        object.__setattr__(sig, "regime_state", regime_state)
                    except Exception:
                        pass

                    is_core = sig.symbol in FIXED_SYMBOLS
                    ai_approved = True
                    reason = ""
                    metadata: dict[str, Any] = {}
                    news_context: Any = ""

                    # 2. Evaluate AI Fundamental Gate for Satellite symbols (checked once, cached)
                    cache_key = (
                        sig.symbol,
                        getattr(sig, "decision_time", getattr(event, "open_time", None)),
                    )
                    cached_eval = self._entry_eval_cache.get(cache_key)

                    if not is_core:
                        if cached_eval is not None:
                            ai_approved, reason, metadata, news_context = cached_eval
                        else:
                            gatekeeper = (
                                self.explainer if self.explainer is not None else TradeExplainer()
                            )
                            if hasattr(
                                gatekeeper, "is_entries_halted"
                            ) and gatekeeper.is_entries_halted(sig.symbol):
                                reason = (
                                    f"REJECT: Entries halted for {sig.symbol} due to consecutive "
                                    "fundamental check fallbacks."
                                )
                                metadata = {
                                    "verdict": "REJECT",
                                    "risk_level": "HIGH",
                                    "risk_type": "other",
                                    "confidence": 0.0,
                                    "injection_suspected": False,
                                    "driving_news_id": "",
                                    "reason": reason,
                                }
                                ai_approved = False
                            else:
                                try:
                                    news = await fetch_recent_news(sig.symbol)
                                    news_context = news
                                    try:
                                        eval_res = gatekeeper.evaluate_fundamentals(
                                            symbol=sig.symbol, news_context=news, mode="ENTRY"
                                        )
                                    except TypeError:
                                        eval_res = gatekeeper.evaluate_fundamentals(
                                            sig.symbol, news
                                        )

                                    res = (
                                        await eval_res
                                        if inspect.isawaitable(eval_res)
                                        else eval_res
                                    )
                                    if isinstance(res, (tuple, list)) and len(res) == 3:
                                        ai_approved, reason, metadata = res
                                    elif isinstance(res, (tuple, list)) and len(res) == 2:
                                        ai_approved, reason = res
                                        metadata = {}
                                    else:
                                        ai_approved, reason, metadata = bool(res), str(res), {}
                                except NewsFetchError as exc:
                                    logger.warning(
                                        "News fetch failed for %s in ENTRY mode: %s",
                                        sig.symbol,
                                        exc,
                                    )
                                    if hasattr(gatekeeper, "handle_news_fetch_error"):
                                        ai_approved, reason, metadata = (
                                            gatekeeper.handle_news_fetch_error(
                                                sig.symbol, mode="ENTRY"
                                            )
                                        )
                                    else:
                                        reason = "REJECT: News fetch failed"
                                        ai_approved = False
                                        metadata = {
                                            "verdict": "REJECT",
                                            "risk_level": "HIGH",
                                            "risk_type": "other",
                                            "confidence": 0.0,
                                            "injection_suspected": False,
                                            "driving_news_id": "",
                                            "reason": reason,
                                        }

                            verdict = (
                                str(
                                    metadata.get("verdict")
                                    or ("APPROVE" if ai_approved else "REJECT")
                                )
                                .strip()
                                .upper()
                            )

                            self._entry_eval_cache[cache_key] = (
                                ai_approved,
                                reason,
                                metadata,
                                news_context,
                            )

                            if self.eval_logger is not None:
                                try:
                                    entry_price = getattr(sig, "entry_price", None)
                                    if entry_price is None or entry_price <= Decimal("0"):
                                        entry_price = getattr(event, "close", Decimal("0"))
                                    raw_news_str = (
                                        news_context
                                        if isinstance(news_context, str)
                                        else (
                                            format_news_xml(news_context)
                                            if isinstance(news_context, list)
                                            else "<news_data></news_data>"
                                        )
                                    )
                                    eval_ts = getattr(
                                        sig,
                                        "decision_time",
                                        getattr(event, "open_time", datetime.now(timezone.utc)),
                                    )
                                    is_fallback_decision = bool(
                                        metadata.get("is_fallback", False)
                                        or verdict == "FALLBACK_REJECT"
                                        or "fallback" in reason.lower()
                                    )
                                    is_blocked_by_regime = 1 if (not is_btc_bullish) else 0
                                    self.eval_logger.log_decision(
                                        symbol=sig.symbol,
                                        current_price=entry_price,
                                        raw_inputs=raw_news_str,
                                        verdict=verdict,
                                        risk_level=str(
                                            metadata.get(
                                                "risk_level",
                                                "LOW" if ai_approved else "HIGH",
                                            )
                                        ),
                                        confidence=float(metadata.get("confidence", 0.0)),
                                        reason=reason,
                                        mode="ENTRY",
                                        metadata=metadata,
                                        timestamp=eval_ts,
                                        is_fallback=is_fallback_decision,
                                        regime_state=regime_state,
                                        regime_blocked=is_blocked_by_regime,
                                    )
                                except Exception as exc:
                                    logger.warning("Failed to log evaluation: %s", exc)

                    verdict = (
                        str(metadata.get("verdict") or ("APPROVE" if ai_approved else "REJECT"))
                        .strip()
                        .upper()
                    )

                    # Always attach cached gate outcome to sig
                    try:
                        object.__setattr__(sig, "verdict", verdict)
                        object.__setattr__(sig, "ai_approved", ai_approved)
                        if reason:
                            object.__setattr__(sig, "fundamental_reason", reason)
                        if metadata:
                            object.__setattr__(sig, "fundamental_metadata", metadata)
                    except Exception:
                        pass

                    # Prepare approval metadata on sig if allowed in Variant A
                    var_a_allowed = self._should_submit_to_variant(
                        variant_name="variant_a",
                        symbol=sig.symbol,
                        is_btc_bullish=is_btc_bullish,
                        ai_approved=ai_approved,
                    )
                    explanation = ""
                    if var_a_allowed:
                        explanation = await self._build_signal_explanation(sig, context_data)
                        if reason:
                            explanation = (
                                f"{explanation}\nAI Fundamental Approval: {reason}"
                                if explanation
                                else f"AI Fundamental Approval: {reason}"
                            )
                        try:
                            object.__setattr__(sig, "status", "APPROVED")
                            object.__setattr__(sig, "verdict", "APPROVE")
                            if reason:
                                object.__setattr__(sig, "fundamental_reason", reason)
                            if explanation:
                                object.__setattr__(sig, "explanation", explanation)
                            if metadata:
                                object.__setattr__(sig, "fundamental_metadata", metadata)
                        except Exception:
                            pass

                    # 3. Fan out to all 5 isolated experiment variants
                    for var_name, exec_inst in self.executors.items():
                        should_sub = self._should_submit_to_variant(
                            variant_name=var_name,
                            symbol=sig.symbol,
                            is_btc_bullish=is_btc_bullish,
                            ai_approved=ai_approved,
                        )
                        if should_sub:
                            var_sig = copy.copy(sig)
                            try:
                                object.__setattr__(var_sig, "regime_state", regime_state)
                                object.__setattr__(var_sig, "verdict", verdict)
                                object.__setattr__(var_sig, "gate_verdict", verdict)
                                object.__setattr__(var_sig, "ai_approved", ai_approved)
                                if reason:
                                    object.__setattr__(var_sig, "fundamental_reason", reason)
                                if metadata:
                                    object.__setattr__(var_sig, "fundamental_metadata", metadata)
                            except Exception:
                                pass
                            async with self._portfolio_lock:
                                submitted = exec_inst.submit_signal(var_sig)
                            if submitted and var_name == "variant_a":
                                self._signals_submitted += 1
                                if self.alert_handler is not None:
                                    await self._notify("on_signal_generated", sig, explanation)

                    # 4. Handle rejection / veto bookkeeping for Variant A and Variant E
                    var_a_allowed = self._should_submit_to_variant(
                        variant_name="variant_a",
                        symbol=sig.symbol,
                        is_btc_bullish=is_btc_bullish,
                        ai_approved=ai_approved,
                    )
                    is_core_alt = is_core and sig.symbol not in ("BTCUSDT", "ETHUSDT")
                    var_e_allowed = self._should_submit_to_variant(
                        variant_name="variant_e",
                        symbol=sig.symbol,
                        is_btc_bullish=is_btc_bullish,
                        ai_approved=ai_approved,
                    )

                    if not is_btc_bullish and (not is_core or (is_core_alt and not var_e_allowed)):
                        target_variant = "Variant A" if not is_core else "Variant E"
                        regime_reason = (
                            f"BLOCKED: Entry for {sig.symbol} blocked by "
                            f"BTC downtrend regime (BTC below 50-day SMA, {target_variant})."
                        )
                        logger.info(regime_reason)
                        if not is_core:
                            try:
                                object.__setattr__(sig, "status", "BLOCKED_BY_BTC_REGIME")
                                object.__setattr__(sig, "veto_reason", regime_reason)
                                object.__setattr__(sig, "explanation", regime_reason)
                            except Exception:
                                pass
                            self._vetoed_signals.append(sig)

                        if is_core and self.eval_logger is not None:
                            try:
                                entry_price = getattr(sig, "entry_price", None)
                                if entry_price is None or entry_price <= Decimal("0"):
                                    entry_price = getattr(event, "close", Decimal("0"))
                                eval_ts = getattr(
                                    sig,
                                    "decision_time",
                                    getattr(event, "open_time", datetime.now(timezone.utc)),
                                )
                                self.eval_logger.log_regime_blocked_signal(
                                    symbol=sig.symbol,
                                    current_price=entry_price,
                                    reason=regime_reason,
                                    raw_inputs="BTC downtrend regime (BTC below 50-day SMA)",
                                    metadata={
                                        "gate_verdict": "APPROVE",
                                        "regime": "DOWNTREND",
                                        "reason": regime_reason,
                                        "target_variant": target_variant,
                                    },
                                    timestamp=eval_ts,
                                    verdict="APPROVE",
                                    regime_state="DOWNTREND",
                                )
                            except Exception as exc:
                                logger.warning("Failed to log regime-blocked evaluation: %s", exc)
                    elif not var_a_allowed:
                        if not ai_approved and not is_core:
                            logger.warning(
                                "Trade vetoed by AI Risk Gatekeeper for %s: %s (metadata=%s)",
                                sig.symbol,
                                reason,
                                metadata,
                            )
                            veto_explanation = f"VETOED: {reason}"
                            try:
                                object.__setattr__(sig, "status", "VETOED")
                                object.__setattr__(sig, "verdict", "REJECT")
                                object.__setattr__(sig, "veto_reason", reason)
                                object.__setattr__(sig, "explanation", veto_explanation)
                                if metadata:
                                    object.__setattr__(sig, "fundamental_metadata", metadata)
                            except Exception:
                                pass
                            self._vetoed_signals.append(sig)

                await self._check_daily_loss()

            self._events_processed += 1
            return trade

        raise TypeError(f"unsupported event type: {type(event).__name__}")

    async def _generate_signals(self, candle: Candle) -> Sequence[BreakoutSignal]:
        """Generate strategy signals for a completed candle."""
        signals, _ = await self._generate_signals_with_context(candle)
        return signals

    async def _generate_signals_with_context(
        self, candle: Candle
    ) -> tuple[Sequence[BreakoutSignal], Any]:
        """Generate strategy signals and return context data if available."""
        if self.data_provider is not None:
            data = self.data_provider(candle)
            if inspect.iscoroutine(data):
                data = await data
            if data is None:
                return (), None
            signals = self.strategy(data)
            if inspect.iscoroutine(signals):
                signals = await signals
            return signals or (), data

        if self.strategy is not generate_breakout_signals:
            try:
                res = self.strategy(candle)
            except TypeError:
                res = self.strategy()
            if inspect.iscoroutine(res):
                res = await res
            return res or (), None

        return (), None

    def stop(self) -> None:
        """Signal the orchestrator to shut down gracefully."""
        self._running = False
        if (
            self._review_task is not None
            and not self._review_task.done()
            and self._review_task is not asyncio.current_task()
        ):
            self._review_task.cancel()
        if (
            self._task is not None
            and not self._task.done()
            and self._task is not asyncio.current_task()
        ):
            self._task.cancel()
        for bg_task in list(self._background_tasks):
            if not bg_task.done() and bg_task is not asyncio.current_task():
                bg_task.cancel()
        self._background_tasks.clear()

    async def _periodic_position_review(self) -> None:
        """Background loop executing every 30 minutes to review open Satellite positions."""
        logger.info("Started periodic fundamental position review loop.")
        try:
            while self._running:
                await asyncio.sleep(self._review_interval)
                if not self._running:
                    break
                await self._review_open_positions()
        except asyncio.CancelledError:
            logger.info("Periodic position review task cancelled.")
            raise
        except Exception as exc:
            logger.exception("Unexpected error in periodic position review loop: %s", exc)

    async def _review_open_positions(self, cycle_id: str | None = None) -> list[Trade]:
        """Iterate over open Satellite positions and evaluate fundamental risks across all variants."""
        executed_trades: list[Trade] = []
        # Clear the review cache at the start of each review pass so each 30-minute pass checks fresh news
        self._position_review_cache.clear()
        if cycle_id is None:
            cycle_id = datetime.now(timezone.utc).isoformat()

        async with self._portfolio_lock:
            active_symbols: set[str] = set()
            satellite_symbols: set[str] = set()
            snapshot_quantities: dict[str, dict[str, Decimal]] = {}
            for var_key, exec_inst in self.executors.items():
                for sym, pos in exec_inst.positions.items():
                    active_symbols.add(sym)
                    if sym not in snapshot_quantities:
                        snapshot_quantities[sym] = {}
                    snapshot_quantities[sym][var_key] = pos.quantity
                    is_sat = (
                        callable(getattr(exec_inst, "get_asset_category", None))
                        and exec_inst.get_asset_category(sym) == "satellite"
                    ) or sym not in FIXED_SYMBOLS
                    if is_sat:
                        satellite_symbols.add(sym)

        for recorded_sym in list(self._position_review_history.keys()):
            if recorded_sym not in active_symbols:
                self._position_review_history.pop(recorded_sym, None)
        self._save_review_history()

        if not satellite_symbols:
            return executed_trades

        gatekeeper = self.explainer if self.explainer is not None else TradeExplainer()

        for symbol in sorted(satellite_symbols):
            try:
                # 1. Single evaluation & caching per symbol across all variants for this pass
                cycle_key = (symbol, cycle_id)
                cached_review = self._position_review_cache.get(cycle_key)

                if cached_review is not None:
                    approved, reason, metadata, news = cached_review
                else:
                    try:
                        news = await fetch_recent_news(symbol)
                        try:
                            eval_res = gatekeeper.evaluate_fundamentals(
                                symbol=symbol, news_context=news, mode="POSITION_REVIEW"
                            )
                        except TypeError:
                            eval_res = gatekeeper.evaluate_fundamentals(symbol, news)

                        res = await eval_res if inspect.isawaitable(eval_res) else eval_res
                        if isinstance(res, (tuple, list)) and len(res) == 3:
                            approved, reason, metadata = res
                        elif isinstance(res, (tuple, list)) and len(res) == 2:
                            approved, reason = res
                            metadata = {}
                        else:
                            approved, reason, metadata = bool(res), str(res), {}
                    except NewsFetchError as exc:
                        logger.warning(
                            "News fetch error for %s in POSITION_REVIEW: %s",
                            symbol,
                            exc,
                        )
                        if hasattr(gatekeeper, "handle_news_fetch_error"):
                            approved, reason, metadata = gatekeeper.handle_news_fetch_error(
                                symbol, mode="POSITION_REVIEW"
                            )
                        else:
                            reason = (
                                "HOLD: Fundamental review fallback; retaining position "
                                "under protective stop."
                            )
                            approved = True
                            metadata = {
                                "verdict": "HOLD",
                                "risk_level": "LOW",
                                "risk_type": "none",
                                "confidence": 0.0,
                                "injection_suspected": False,
                                "driving_news_id": "",
                                "reason": reason,
                                "is_fallback": True,
                            }
                        news = "<news_data></news_data>"

                    self._position_review_cache[cycle_key] = (
                        approved,
                        reason,
                        metadata,
                        news,
                    )

                driving_news_id = str(metadata.get("driving_news_id") or "").strip()
                if not driving_news_id:
                    driving_news_id = hashlib.sha256(reason.encode("utf-8")).hexdigest()[:16]

                verdict = str(metadata.get("verdict") or "").strip().upper()
                if not verdict:
                    verdict = "HOLD" if approved else "EXIT"

                injection_suspected = bool(metadata.get("injection_suspected", False))
                if injection_suspected:
                    logger.critical(
                        "CRITICAL SECURITY ALERT: Suspected prompt injection detected during position review for %s! "
                        "Headline vector is untrusted. Preventing full liquidation; executing at most 50%% reduction and flagging for human review. (metadata=%s)",
                        symbol,
                        metadata,
                    )
                    if verdict == "EXIT":
                        verdict = "REDUCE"

                history = self._position_review_history.get(symbol, {})
                prev_action = history.get("last_reviewed_action")
                prev_driving_id = history.get("driving_news_id")
                consecutive_reduce_cycles = history.get("consecutive_reduce_cycles", 0)

                if self.eval_logger is not None:
                    try:
                        raw_news_str = (
                            news
                            if isinstance(news, str)
                            else (
                                format_news_xml(news)
                                if isinstance(news, list)
                                else "<news_data></news_data>"
                            )
                        )
                        ref_price = Decimal("0")
                        for exec_inst in self.executors.values():
                            if symbol in exec_inst.positions:
                                ref_price = exec_inst.positions[symbol].entry_fill.price
                                break
                        is_fallback_review = bool(
                            metadata.get("is_fallback", False) or "fallback" in reason.lower()
                        )
                        self.eval_logger.log_decision(
                            symbol=symbol,
                            current_price=ref_price,
                            raw_inputs=raw_news_str,
                            verdict=verdict,
                            risk_level=str(metadata.get("risk_level", "LOW")),
                            confidence=float(metadata.get("confidence", 0.0)),
                            reason=reason,
                            mode="POSITION_REVIEW",
                            metadata=metadata,
                            is_fallback=is_fallback_review,
                        )
                    except Exception as exc:
                        logger.warning("Failed to log position review evaluation: %s", exc)

                if verdict == "HOLD":
                    self._position_review_history[symbol] = {
                        "last_reviewed_action": "HOLD",
                        "driving_news_id": driving_news_id,
                        "consecutive_reduce_cycles": 0,
                        "reduced_for_catalyst": history.get("reduced_for_catalyst", False),
                    }
                    self._save_review_history()
                    for exec_inst in self.executors.values():
                        pos = exec_inst.positions.get(symbol)
                        if pos is not None:
                            try:
                                object.__setattr__(pos, "last_reviewed_action", "HOLD")
                                object.__setattr__(pos, "driving_news_id", driving_news_id)
                            except Exception:
                                pass
                    logger.info("AI Position Review: HOLD position for %s: %s", symbol, reason)

                elif verdict == "REDUCE":
                    if prev_action == "REDUCE" and prev_driving_id == driving_news_id:
                        consecutive_reduce_cycles += 1
                        history["consecutive_reduce_cycles"] = consecutive_reduce_cycles
                        self._position_review_history[symbol] = history
                        self._save_review_history()

                        if consecutive_reduce_cycles > 4:
                            logger.warning(
                                "Position for %s has remained in REDUCE for driving_news_id '%s' for %d consecutive cycles (>4); escalating to EXIT.",
                                symbol,
                                driving_news_id,
                                consecutive_reduce_cycles,
                            )
                            verdict = "EXIT"
                        else:
                            logger.info(
                                "Duplicate REDUCE verdict for %s on driving_news_id '%s' (cycle %d); retaining current exposure.",
                                symbol,
                                driving_news_id,
                                consecutive_reduce_cycles,
                            )
                            continue
                    else:
                        consecutive_reduce_cycles = 1

                    if verdict == "REDUCE":
                        logger.warning(
                            "AI Position Review triggered REDUCE for %s: %s (metadata=%s)",
                            symbol,
                            reason,
                            metadata,
                        )
                        trade = None
                        async with self._portfolio_lock:
                            # Apply REDUCE to variants that use the gate (A, C, E)
                            for var_key in ("variant_a", "variant_c", "variant_e"):
                                exec_inst = self.executors.get(var_key)
                                if exec_inst is None or symbol not in exec_inst.positions:
                                    if var_key == "variant_a":
                                        logger.info(
                                            "Position for %s was closed before reduction; skipping.",
                                            symbol,
                                        )
                                    continue
                                current_pos = exec_inst.positions[symbol]
                                expected_qty = snapshot_quantities.get(symbol, {}).get(var_key)
                                if (
                                    expected_qty is not None
                                    and current_pos.quantity != expected_qty
                                ):
                                    logger.warning(
                                        "Position quantity for %s changed (%s != %s) before reduction; skipping.",
                                        symbol,
                                        current_pos.quantity,
                                        expected_qty,
                                    )
                                    continue

                                if hasattr(exec_inst, "reduce_position") and callable(
                                    exec_inst.reduce_position
                                ):
                                    tr = exec_inst.reduce_position(
                                        symbol,
                                        fraction=DEFAULT_REDUCE_RATIO,
                                        reduce_ratio=DEFAULT_REDUCE_RATIO,
                                        reason=f"fundamental_review_{verdict.lower()}",
                                    )
                                    if var_key == "variant_a":
                                        trade = tr
                                elif hasattr(exec_inst, "close_position") and callable(
                                    exec_inst.close_position
                                ):
                                    tr = exec_inst.close_position(
                                        symbol,
                                        reason=f"fundamental_review_{verdict.lower()}",
                                    )
                                    if var_key == "variant_a":
                                        trade = tr

                            # Annotate variants B and D with the gate outcome for offline eval
                            for var_key in ("variant_b", "variant_d"):
                                exec_inst = self.executors.get(var_key)
                                if exec_inst is not None and symbol in exec_inst.positions:
                                    try:
                                        pos = exec_inst.positions[symbol]
                                        object.__setattr__(pos, "last_reviewed_action", "REDUCE")
                                        object.__setattr__(pos, "driving_news_id", driving_news_id)
                                    except Exception:
                                        pass

                        if trade is not None:
                            executed_trades.append(trade)
                            self._position_review_history[symbol] = {
                                "last_reviewed_action": "REDUCE",
                                "driving_news_id": driving_news_id,
                                "consecutive_reduce_cycles": consecutive_reduce_cycles,
                                "reduced_for_catalyst": True,
                            }
                            self._save_review_history()
                            pos = self.executor.positions.get(symbol)
                            if pos is not None:
                                try:
                                    object.__setattr__(pos, "last_reviewed_action", "REDUCE")
                                    object.__setattr__(pos, "driving_news_id", driving_news_id)
                                except Exception:
                                    pass
                            if self.alert_handler is not None:
                                await self._notify(
                                    "on_exit_filled",
                                    trade,
                                    f"AI Position Review ({verdict}): {reason}",
                                )

                if verdict == "EXIT":
                    logger.warning(
                        "AI Position Review triggered EXIT for %s: %s (metadata=%s)",
                        symbol,
                        reason,
                        metadata,
                    )
                    trade = None
                    async with self._portfolio_lock:
                        # Apply EXIT to variants that use the gate (A, C, E)
                        for var_key in ("variant_a", "variant_c", "variant_e"):
                            exec_inst = self.executors.get(var_key)
                            if exec_inst is None or symbol not in exec_inst.positions:
                                if var_key == "variant_a":
                                    logger.info(
                                        "Position for %s was closed before exit; skipping.",
                                        symbol,
                                    )
                                continue
                            current_pos = exec_inst.positions[symbol]
                            expected_qty = snapshot_quantities.get(symbol, {}).get(var_key)
                            if expected_qty is not None and current_pos.quantity != expected_qty:
                                logger.warning(
                                    "Position quantity for %s changed (%s != %s) before exit; skipping.",
                                    symbol,
                                    current_pos.quantity,
                                    expected_qty,
                                )
                                continue

                            if hasattr(exec_inst, "close_position") and callable(
                                exec_inst.close_position
                            ):
                                tr = exec_inst.close_position(
                                    symbol,
                                    reason=f"fundamental_review_{verdict.lower()}",
                                )
                                if var_key == "variant_a":
                                    trade = tr
                            elif hasattr(exec_inst, "exit_position") and callable(
                                exec_inst.exit_position
                            ):
                                tr = exec_inst.exit_position(
                                    symbol,
                                    reason=f"fundamental_review_{verdict.lower()}",
                                )
                                if var_key == "variant_a":
                                    trade = tr

                        # Annotate variants B and D with the gate outcome for offline eval
                        for var_key in ("variant_b", "variant_d"):
                            exec_inst = self.executors.get(var_key)
                            if exec_inst is not None and symbol in exec_inst.positions:
                                try:
                                    pos = exec_inst.positions[symbol]
                                    object.__setattr__(pos, "last_reviewed_action", "EXIT")
                                    object.__setattr__(pos, "driving_news_id", driving_news_id)
                                except Exception:
                                    pass

                    if trade is not None:
                        executed_trades.append(trade)
                        self._position_review_history.pop(symbol, None)
                        self._save_review_history()
                        if self.alert_handler is not None:
                            await self._notify(
                                "on_exit_filled",
                                trade,
                                f"AI Position Review ({verdict}): {reason}",
                            )

            except Exception as exc:
                logger.warning(
                    "Error evaluating position review for %s: %s",
                    symbol,
                    exc,
                    exc_info=True,
                )

        return executed_trades

    async def run(self) -> None:
        """Run the asynchronous event consumption loop."""
        if self._running:
            raise RuntimeError("session is already running")

        self._running = True
        self._task = asyncio.current_task()
        self._review_task = asyncio.create_task(self._periodic_position_review())

        loop = asyncio.get_running_loop()
        signals_to_handle = (signal.SIGINT, signal.SIGTERM)
        registered_signals: list[signal.Signals] = []

        def _handle_signal() -> None:
            logger.info("Signal received, stopping paper session...")
            self.stop()

        for sig in signals_to_handle:
            try:
                loop.add_signal_handler(sig, _handle_signal)
                registered_signals.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

        try:
            async for event in self.stream.events():
                if not self._running:
                    break
                await self.process_event(event)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Paper session cancelled or interrupted, exiting gracefully.")
        except Exception:
            logger.exception("Stream error in PaperSession")
        else:
            if self._running:
                logger.warning("Market stream ended unexpectedly.")
        finally:
            for sig in registered_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass
            if self._review_task is not None:
                if not self._review_task.done():
                    self._review_task.cancel()
                    try:
                        await self._review_task
                    except (asyncio.CancelledError, Exception):
                        pass
                self._review_task = None
            self._running = False
            self._task = None
