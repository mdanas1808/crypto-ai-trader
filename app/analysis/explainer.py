"""Read-only AI analysis layer for deterministic strategy actions.

Provides human-readable reasoning and summaries for Breakout signals
and Trade exit outcomes without modifying execution state.
"""

from __future__ import annotations

from decimal import Decimal
import pandas as pd

from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import Trade
from app.strategy.breakout import BreakoutSignal


class TradeExplainer:
    """Generate structured explanations for strategy signals and executed trades."""

    def __init__(self, *, price_decimals: int = 4, currency: str = "USDT") -> None:
        self.price_decimals = price_decimals
        self.currency = currency

    def explain_signal(
        self, signal: BreakoutSignal, context_data: MultiTimeframeBacktestData
    ) -> str:
        """Format a clear explanation of why a breakout signal fired."""
        if not isinstance(signal, BreakoutSignal):
            raise TypeError("signal must be BreakoutSignal")
        if not isinstance(context_data, MultiTimeframeBacktestData):
            raise TypeError("context_data must be MultiTimeframeBacktestData")

        trading = context_data.trading_frame
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

        # Reconstruct 20-candle resistance reference high if available
        context = context_data.context_frame
        history_highs: list[float] = []
        if not context.empty and "high" in context.columns:
            history_highs.extend(float(h) for h in context["high"])

        if "open_time" in trading.columns:
            prior_trading = trading.loc[trading["open_time"] < signal.signal_open_time]
            if not prior_trading.empty and "high" in prior_trading.columns:
                history_highs.extend(float(h) for h in prior_trading["high"])

        reference_high = None
        if len(history_highs) >= 20:
            reference_high = max(history_highs[-20:])

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

    def explain_trade_exit(self, trade: Trade) -> str:
        """Format a clear summary of a completed trade exit outcome."""
        if not isinstance(trade, Trade):
            raise TypeError("trade must be Trade")

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

