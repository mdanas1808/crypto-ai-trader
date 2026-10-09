"""Read-only AI analysis layer for deterministic strategy actions.

Provides human-readable reasoning and summaries for Breakout signals
and Trade exit outcomes without modifying execution state. Supports
live LLM commentary via Groq with a guaranteed deterministic template fallback.
"""

from __future__ import annotations

from decimal import Decimal
import json
import logging
import os
from typing import Any

import httpx
import pandas as pd

from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import Trade
from app.strategy.breakout import BreakoutSignal

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 5.0


class TradeExplainer:
    """Generate structured explanations for strategy signals and executed trades."""

    def __init__(
        self,
        *,
        price_decimals: int = 4,
        currency: str = "USDT",
        api_key: str | None = None,
        model: str = DEFAULT_GROQ_MODEL,
        base_url: str = GROQ_COMPLETIONS_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        async_client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ) -> None:
        self.price_decimals = price_decimals
        self.currency = currency
        env_key = os.getenv("GROQ_API_KEY")
        resolved_key = api_key if api_key is not None else env_key
        self.api_key = (
            resolved_key.strip()
            if resolved_key and resolved_key.strip()
            else None
        )
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._async_client = async_client
        self._sync_client = sync_client

    def _extract_signal_row(
        self, signal: BreakoutSignal, trading: pd.DataFrame
    ) -> pd.Series:
        if trading.empty:
            raise ValueError("context_data trading_frame is empty")

        row = None
        if signal.decision_time in trading.index:
            matched = trading.loc[signal.decision_time]
            row = matched.iloc[-1] if isinstance(matched, pd.DataFrame) else matched
        elif "open_time" in trading.columns:
            matched = trading.loc[trading["open_time"] == signal.signal_open_time]
            if not matched.empty:
                row = matched.iloc[0]

        if row is None:
            raise ValueError(
                f"Signal decision time {signal.decision_time} not found in trading_frame"
            )
        return row

    def _extract_reference_high(
        self, signal: BreakoutSignal, context_data: MultiTimeframeBacktestData
    ) -> float | None:
        context = context_data.context_frame
        trading = context_data.trading_frame
        history_highs: list[float] = []
        if not context.empty and "high" in context.columns:
            history_highs.extend(float(h) for h in context["high"])

        if "open_time" in trading.columns:
            prior_trading = trading.loc[trading["open_time"] < signal.signal_open_time]
            if not prior_trading.empty and "high" in prior_trading.columns:
                history_highs.extend(float(h) for h in prior_trading["high"])

        if len(history_highs) >= 20:
            return max(history_highs[-20:])
        return None

    def _describe_exit_outcome(self, trade: Trade) -> tuple[str, str]:
        exit_price = trade.exit_fill.price
        ref_price = trade.exit_fill.reference_price
        stop_price = trade.stop_price
        target_price = trade.target_price

        if trade.exit_reason == "take_profit":
            if ref_price > target_price:
                outcome = "Take Profit Hit (Gap Open Slippage)"
                details = (
                    f"Market opened above profit target at {ref_price:.{self.price_decimals}f} "
                    f"(target was {target_price:.{self.price_decimals}f}); filled at {exit_price:.{self.price_decimals}f} "
                    f"with adverse slippage."
                )
            else:
                outcome = "Take Profit Hit"
                details = (
                    f"Profit target ({target_price:.{self.price_decimals}f}) was reached; "
                    f"filled at {exit_price:.{self.price_decimals}f} with adverse slippage."
                )
        elif trade.exit_reason == "stop_loss":
            if ref_price < stop_price:
                outcome = "Stopped Out (Gap Down Slippage)"
                details = (
                    f"Market gapped down below stop loss to {ref_price:.{self.price_decimals}f} "
                    f"(stop was {stop_price:.{self.price_decimals}f}); filled at {exit_price:.{self.price_decimals}f} "
                    f"with adverse slippage."
                )
            else:
                outcome = "Stopped Out"
                details = (
                    f"Stop loss ({stop_price:.{self.price_decimals}f}) was triggered; "
                    f"filled at {exit_price:.{self.price_decimals}f} with adverse slippage."
                )
        elif trade.exit_reason == "end_of_backtest":
            outcome = "Closed at End of Session"
            details = (
                f"Position was liquidated at simulation close ({ref_price:.{self.price_decimals}f}); "
                f"filled at {exit_price:.{self.price_decimals}f}."
            )
        else:
            outcome = f"Exited ({trade.exit_reason})"
            details = (
                f"Position exited due to {trade.exit_reason} at {exit_price:.{self.price_decimals}f}."
            )
        return outcome, details

    def explain_signal_template(
        self, signal: BreakoutSignal, context_data: MultiTimeframeBacktestData
    ) -> str:
        """Format a clear deterministic template explanation of why a breakout signal fired."""
        if not isinstance(signal, BreakoutSignal):
            raise TypeError("signal must be BreakoutSignal")
        if not isinstance(context_data, MultiTimeframeBacktestData):
            raise TypeError("context_data must be MultiTimeframeBacktestData")

        trading = context_data.trading_frame
        row = self._extract_signal_row(signal, trading)
        reference_high = self._extract_reference_high(signal, context_data)

        close = float(row["close"])
        volume = float(row["volume"])
        volume_sma = float(row.get("volume_sma_20", 0.0))
        rsi = float(row.get("rsi_14", 0.0))
        ema_200 = float(row.get("ema_200", 0.0))
        atr = float(row.get("atr_14", signal.atr_14))

        four_hour_close = float(row.get("4h_close", 0.0))
        four_hour_ema_20 = float(row.get("4h_ema_20", 0.0))
        four_hour_ema_50 = float(row.get("4h_ema_50", 0.0))
        four_hour_ema_200 = float(row.get("4h_ema_200", 0.0))

        volume_ratio = (volume / volume_sma) if volume_sma > 0 else 0.0

        ref_high_text = (
            f"broke above the 20-period resistance ({reference_high:.{self.price_decimals}f})"
            if reference_high is not None
            else "broke above the 20-period resistance"
        )

        lines = [
            f"Breakout Long Signal for {signal.symbol} ({signal.exchange}) at {signal.decision_time.isoformat()}:",
            f"1. Regime (4H): Bullish alignment. 4H Close ({four_hour_close:.{self.price_decimals}f}) is above 4H EMA 200 ({four_hour_ema_200:.{self.price_decimals}f}), and 4H EMA 20 ({four_hour_ema_20:.{self.price_decimals}f}) is above 4H EMA 50 ({four_hour_ema_50:.{self.price_decimals}f}).",
            f"2. Trend & Breakout (1H): 1H Close ({close:.{self.price_decimals}f}) is above 1H EMA 200 ({ema_200:.{self.price_decimals}f}) and {ref_high_text}.",
            f"3. Momentum (1H): 1H RSI(14) is {rsi:.2f}, indicating strong momentum within the 55.0-70.0 qualifying zone.",
            f"4. Volume (1H): 1H Volume ({volume:.2f}) surged to {volume_ratio:.2f}x its 20-period average ({volume_sma:.2f}), exceeding the 1.5x surge threshold.",
            f"5. Volatility (ATR): 14-period ATR is {atr:.{self.price_decimals}f}, establishing volatility-based stop (1.5x ATR) and target (2R).",
        ]
        return "\n".join(lines)

    def explain_trade_exit_template(self, trade: Trade) -> str:
        """Format a clear deterministic template summary of a completed trade exit outcome."""
        if not isinstance(trade, Trade):
            raise TypeError("trade must be Trade")

        symbol = trade.signal.symbol
        entry_price = trade.entry_fill.price
        exit_price = trade.exit_fill.price
        quantity = trade.quantity
        entry_notional = trade.entry_fill.notional
        net_pnl = trade.net_pnl
        gross_pnl = trade.gross_pnl
        total_fee = trade.entry_fill.fee + trade.exit_fill.fee
        entry_time = trade.entry_fill.timestamp.isoformat()
        exit_time = trade.exit_fill.timestamp.isoformat()

        pnl_pct = (
            (net_pnl / entry_notional * Decimal("100"))
            if entry_notional > 0
            else Decimal("0")
        )

        outcome, details = self._describe_exit_outcome(trade)
        sign = "+" if net_pnl >= 0 else ""
        gross_sign = "+" if gross_pnl >= 0 else ""

        lines = [
            f"Trade Exit for {symbol} (LONG):",
            f"- Outcome: {outcome}",
            f"- Reason & Pricing: {details}",
            f"- Execution: Entry {entry_price:.{self.price_decimals}f} ({entry_time}) -> Exit {exit_price:.{self.price_decimals}f} ({exit_time})",
            f"- Quantity: {quantity} (Notional: {entry_notional:.2f} {self.currency})",
            f"- Financial Result: Net PnL {sign}{net_pnl:.4f} {self.currency} ({sign}{pnl_pct:.2f}%) [Gross: {gross_sign}{gross_pnl:.4f} {self.currency}]",
            f"- Fees: Total {total_fee:.4f} {self.currency} (Entry: {trade.entry_fill.fee:.4f}, Exit: {trade.exit_fill.fee:.4f})",
        ]
        return "\n".join(lines)

    def _build_signal_prompts(
        self, signal: BreakoutSignal, context_data: MultiTimeframeBacktestData
    ) -> tuple[str, str]:
        trading = context_data.trading_frame
        row = self._extract_signal_row(signal, trading)
        reference_high = self._extract_reference_high(signal, context_data)

        close = float(row["close"])
        volume = float(row["volume"])
        volume_sma = float(row.get("volume_sma_20", 0.0))
        rsi = float(row.get("rsi_14", 0.0))
        ema_200 = float(row.get("ema_200", 0.0))
        atr = float(row.get("atr_14", signal.atr_14))

        four_hour_close = float(row.get("4h_close", 0.0))
        four_hour_ema_20 = float(row.get("4h_ema_20", 0.0))
        four_hour_ema_50 = float(row.get("4h_ema_50", 0.0))
        four_hour_ema_200 = float(row.get("4h_ema_200", 0.0))

        volume_ratio = (volume / volume_sma) if volume_sma > 0 else 0.0

        ref_high_text = (
            f"20-period resistance ({reference_high:.{self.price_decimals}f})"
            if reference_high is not None
            else "20-period resistance"
        )

        system_prompt = (
            "You are a professional quantitative crypto trading analyst. "
            "Output a tight, professional 2-3 sentence market commentary for traders explaining "
            "why this breakout long signal triggered. Focus strictly on the technical indicators, regime alignment, "
            "and volatility parameters provided. Do not provide financial advice."
        )

        user_prompt = (
            f"Breakout Long Signal for {signal.symbol} ({signal.exchange}) at {signal.decision_time.isoformat()}:\n"
            f"- 4H Higher-Timeframe Regime: Bullish alignment. 4H Close ({four_hour_close:.{self.price_decimals}f}) > 4H EMA 200 ({four_hour_ema_200:.{self.price_decimals}f}), "
            f"4H EMA 20 ({four_hour_ema_20:.{self.price_decimals}f}) > 4H EMA 50 ({four_hour_ema_50:.{self.price_decimals}f}).\n"
            f"- 1H Trend & Breakout: 1H Close ({close:.{self.price_decimals}f}) > 1H EMA 200 ({ema_200:.{self.price_decimals}f}), broke above {ref_high_text}.\n"
            f"- 1H Momentum: RSI(14) is {rsi:.2f} (qualifying zone: 55.0-70.0).\n"
            f"- 1H Volume: {volume:.2f} ({volume_ratio:.2f}x of 20-period SMA {volume_sma:.2f}, > 1.5x surge threshold).\n"
            f"- Volatility & Risk (ATR): 14-period ATR is {atr:.{self.price_decimals}f} (Stop: 1.5x ATR, Target: 2R).\n\n"
            "Provide a concise 2-3 sentence market commentary."
        )
        return system_prompt, user_prompt

    def _build_trade_exit_prompts(self, trade: Trade) -> tuple[str, str]:
        symbol = trade.signal.symbol
        entry_price = trade.entry_fill.price
        exit_price = trade.exit_fill.price
        ref_price = trade.exit_fill.reference_price
        stop_price = trade.stop_price
        target_price = trade.target_price
        quantity = trade.quantity
        entry_notional = trade.entry_fill.notional
        net_pnl = trade.net_pnl
        gross_pnl = trade.gross_pnl
        total_fee = trade.entry_fill.fee + trade.exit_fill.fee
        entry_time = trade.entry_fill.timestamp.isoformat()
        exit_time = trade.exit_fill.timestamp.isoformat()

        pnl_pct = (
            (net_pnl / entry_notional * Decimal("100"))
            if entry_notional > 0
            else Decimal("0")
        )

        outcome, details = self._describe_exit_outcome(trade)
        sign = "+" if net_pnl >= 0 else ""
        gross_sign = "+" if gross_pnl >= 0 else ""

        system_prompt = (
            "You are a quantitative trading post-trade analyst. "
            "Output a tight, analytical 2-3 sentence trade retrospective summarizing the trade exit outcome, "
            "execution quality, slippage, and net financial results. "
            "Do not provide financial advice."
        )

        user_prompt = (
            f"Trade Exit Retrospective for {symbol} (LONG):\n"
            f"- Outcome: {outcome}\n"
            f"- Reason & Pricing Mechanics: {details}\n"
            f"- Execution: Entry {entry_price:.{self.price_decimals}f} ({entry_time}) -> Exit {exit_price:.{self.price_decimals}f} ({exit_time}) [Reference Price: {ref_price:.{self.price_decimals}f}]\n"
            f"- Parameters: Stop Loss {stop_price:.{self.price_decimals}f}, Target {target_price:.{self.price_decimals}f}\n"
            f"- Position Size: {quantity} (Notional: {entry_notional:.2f} {self.currency})\n"
            f"- Performance: Net PnL {sign}{net_pnl:.4f} {self.currency} ({sign}{pnl_pct:.2f}%), Gross PnL {gross_sign}{gross_pnl:.4f} {self.currency}\n"
            f"- Fees: Total {total_fee:.4f} {self.currency} (Entry: {trade.entry_fill.fee:.4f}, Exit: {trade.exit_fill.fee:.4f})\n\n"
            "Provide a concise 2-3 sentence analytical trade retrospective."
        )
        return system_prompt, user_prompt

    async def _call_llm_async(
        self, system_prompt: str, user_prompt: str, fallback: str
    ) -> str:
        if not self.api_key:
            return fallback

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 250,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            if self._async_client is not None:
                response = await self._async_client.post(
                    self.base_url, json=payload, headers=headers, timeout=self.timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.base_url, json=payload, headers=headers
                    )

            if response.status_code != 200:
                logger.warning(
                    "Groq LLM API returned HTTP %s: %s; falling back to template",
                    response.status_code,
                    response.text,
                )
                return fallback

            data = response.json()
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                logger.warning(
                    "Groq LLM API response missing choices; falling back to template"
                )
                return fallback

            content = choices[0].get("message", {}).get("content")
            if not content or not isinstance(content, str):
                logger.warning(
                    "Groq LLM API response missing content; falling back to template"
                )
                return fallback

            return content.strip()

        except httpx.TimeoutException as exc:
            logger.warning(
                "Groq LLM call timed out after %ss: %s; falling back to template",
                self.timeout,
                exc,
            )
            return fallback
        except Exception as exc:
            logger.warning(
                "Groq LLM call failed (%s): %s; falling back to template",
                type(exc).__name__,
                exc,
            )
            return fallback

    def _call_llm_sync(
        self, system_prompt: str, user_prompt: str, fallback: str
    ) -> str:
        if not self.api_key:
            return fallback

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 250,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            if self._sync_client is not None:
                response = self._sync_client.post(
                    self.base_url, json=payload, headers=headers, timeout=self.timeout
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.base_url, json=payload, headers=headers
                    )

            if response.status_code != 200:
                logger.warning(
                    "Groq LLM API returned HTTP %s: %s; falling back to template",
                    response.status_code,
                    response.text,
                )
                return fallback

            data = response.json()
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                logger.warning(
                    "Groq LLM API response missing choices; falling back to template"
                )
                return fallback

            content = choices[0].get("message", {}).get("content")
            if not content or not isinstance(content, str):
                logger.warning(
                    "Groq LLM API response missing content; falling back to template"
                )
                return fallback

            return content.strip()

        except httpx.TimeoutException as exc:
            logger.warning(
                "Groq LLM call timed out after %ss: %s; falling back to template",
                self.timeout,
                exc,
            )
            return fallback
        except Exception as exc:
            logger.warning(
                "Groq LLM call failed (%s): %s; falling back to template",
                type(exc).__name__,
                exc,
            )
            return fallback

    # Asynchronous LLM methods (primary for live async PaperSession)
    async def explain_signal(
        self, signal: BreakoutSignal, context_data: MultiTimeframeBacktestData
    ) -> str:
        """Format explanation of why a breakout signal fired, using LLM if available."""
        template = self.explain_signal_template(signal, context_data)
        if not self.api_key:
            return template
        system_prompt, user_prompt = self._build_signal_prompts(signal, context_data)
        return await self._call_llm_async(system_prompt, user_prompt, fallback=template)

    async def explain_trade_exit(self, trade: Trade) -> str:
        """Format summary of completed trade exit outcome, using LLM if available."""
        template = self.explain_trade_exit_template(trade)
        if not self.api_key:
            return template
        system_prompt, user_prompt = self._build_trade_exit_prompts(trade)
        return await self._call_llm_async(system_prompt, user_prompt, fallback=template)

    # Synchronous LLM methods (for sync scripts/tools)
    def explain_signal_sync(
        self, signal: BreakoutSignal, context_data: MultiTimeframeBacktestData
    ) -> str:
        """Synchronously format explanation of breakout signal using LLM or template."""
        template = self.explain_signal_template(signal, context_data)
        if not self.api_key:
            return template
        system_prompt, user_prompt = self._build_signal_prompts(signal, context_data)
        return self._call_llm_sync(system_prompt, user_prompt, fallback=template)

    def explain_trade_exit_sync(self, trade: Trade) -> str:
        """Synchronously format summary of trade exit using LLM or template."""
        template = self.explain_trade_exit_template(trade)
        if not self.api_key:
            return template
        system_prompt, user_prompt = self._build_trade_exit_prompts(trade)
        return self._call_llm_sync(system_prompt, user_prompt, fallback=template)

    # Aliases
    aexplain_signal = explain_signal
    aexplain_trade_exit = explain_trade_exit
