from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.backtest.models import BacktestConfig
from app.market.constants import EXCHANGE
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import (
    CORE_SLIPPAGE,
    DEFAULT_STARTING_CASH,
    FUTURES_FEE_RATE,
    PaperExecutor,
)

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)
D = Decimal


def config(*, minimum_quantity=D("0.000001"), quantity_step=D("0.000001")):
    return BacktestConfig(minimum_quantity=minimum_quantity, quantity_step=quantity_step)


def candle(
    symbol="BTCUSDT", opened=START, *, open=D("10"), high=D("10.2"), low=D("9.9"), close=D("10")
):
    return Candle(
        exchange=EXCHANGE,
        symbol=symbol,
        interval="1h",
        open_time=opened,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=D("1"),
    )


def observation(
    symbol="BTCUSDT", opened=START + timedelta(hours=1), *, price=D("10"), observed=None
):
    return CandleOpenObservation(
        exchange=EXCHANGE,
        symbol=symbol,
        interval="1h",
        open_time=opened,
        observed_at=observed or opened + timedelta(seconds=1),
        open=price,
    )


def signal(symbol="BTCUSDT", opened=START, *, side="long", atr=1.0, score=2.5, leverage=5):
    return BreakoutSignal(
        exchange=EXCHANGE,
        symbol=symbol,
        side=side,
        decision_time=opened + timedelta(hours=1),
        signal_open_time=opened,
        atr_14=atr,
        score=score,
        leverage=leverage,
    )


def prepare_signal(executor, symbol="BTCUSDT", opened=START, *, side="long", atr=1.0, leverage=5):
    executor.on_candle_closed(candle(symbol, opened))
    intent = signal(symbol, opened, side=side, atr=atr, leverage=leverage)
    assert executor.submit_signal(intent)
    return intent


def enter_default(
    executor=None,
    symbol="BTCUSDT",
    opened=START,
    *,
    side="long",
    price=D("10"),
    atr=1.0,
    leverage=5,
):
    executor = executor or PaperExecutor(config())
    prepare_signal(executor, symbol, opened, side=side, atr=atr, leverage=leverage)
    position = executor.on_open_observation(
        observation(symbol, opened + timedelta(hours=1), price=price)
    )
    return executor, position


# --- Long Entry, SL, TP, Fees, Sizing Tests ---


def test_long_entry_notional_margin_fees_and_slippage():
    executor, position = enter_default(leverage=5)
    assert position is not None
    assert position.direction == "LONG"
    assert position.leverage == 5

    fill = position.entry_fill
    assert fill.timestamp == START + timedelta(hours=1)
    assert fill.reference_price == D("10")
    # Adverse slippage for long: 10 * (1 + 0.0005) = 10.0050
    assert fill.price == D("10.0050")

    # 95% margin allocated on $100 starting cash: $95.00
    # Notional = 95 * 5 = $475.00
    # Quantity = 475 / 10.0050 = 47.476261
    assert fill.quantity == D("47.476261")
    # Futures fee is 0.04%
    expected_fee = fill.price * fill.quantity * FUTURES_FEE_RATE
    assert fill.fee == expected_fee

    # Stop Loss is 1.5x ATR below entry: 10.0050 - 1.5 * 1.0 = 8.5050
    assert position.stop_price == D("8.5050")
    assert position.stop_loss == D("8.5050")

    # Take Profit is 3.0x ATR above entry: 10.0050 + 3.0 * 1.0 = 13.0050
    assert position.target_price == D("13.0050")
    assert position.take_profit == D("13.0050")

    # Liquidation price for 5x long: 10.0050 * (1 - 1/5) = 8.0040
    assert position.liquidation_price == D("10.0050") * (D("1") - D("1") / D("5"))


def test_long_take_profit_exit_and_pnl_math():
    executor, position = enter_default(leverage=5)
    assert position is not None

    # Candle triggers take-profit at target_price
    trade = executor.on_candle_closed(
        candle(opened=START + timedelta(hours=1), high=D("14"), low=D("9"), close=D("13.5"))
    )
    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.direction == "LONG"

    # Exit fill price with 0.05% adverse slippage: target_price * (1 - 0.0005)
    expected_exit_price = position.target_price * (D("1") - CORE_SLIPPAGE)
    assert trade.exit_fill.price == expected_exit_price

    # Gross PnL for LONG: (exit_price - entry_price) * quantity
    expected_gross = (trade.exit_fill.price - trade.entry_fill.price) * trade.quantity
    assert trade.gross_pnl == expected_gross

    # Net PnL = gross_pnl - total_fees
    total_fees = trade.entry_fill.fee + trade.exit_fill.fee
    assert trade.net_pnl == expected_gross - total_fees
    assert trade.realized_pnl == trade.net_pnl

    # Cash should now be higher than starting cash $100
    assert executor.cash > DEFAULT_STARTING_CASH
    assert len(executor.positions) == 0


def test_long_stop_loss_exit_and_pnl_math():
    executor, position = enter_default(leverage=5)
    assert position is not None

    # Candle triggers stop-loss at stop_price
    trade = executor.on_candle_closed(
        candle(opened=START + timedelta(hours=1), high=D("11"), low=D("8.2"), close=D("8.4"))
    )
    assert trade is not None
    assert trade.exit_reason == "stop_loss"

    # Exit price with adverse slippage: stop_price * (1 - 0.0005)
    expected_exit_price = position.stop_price * (D("1") - CORE_SLIPPAGE)
    assert trade.exit_fill.price == expected_exit_price

    # Gross PnL is negative
    expected_gross = (trade.exit_fill.price - trade.entry_fill.price) * trade.quantity
    assert trade.gross_pnl == expected_gross
    assert trade.net_pnl < Decimal("0")

    # Cash updated
    assert executor.cash < DEFAULT_STARTING_CASH
    assert len(executor.positions) == 0


# --- Short Entry, SL, TP, Fees, Sizing Tests ---


def test_short_entry_notional_margin_fees_and_slippage():
    executor, position = enter_default(side="short", leverage=10)
    assert position is not None
    assert position.direction == "SHORT"
    assert position.leverage == 10

    fill = position.entry_fill
    # Adverse slippage for short: 10 * (1 - 0.0005) = 9.9950
    assert fill.price == D("9.9950")

    # 95% margin allocated on $100: $95.00
    # Notional = 95 * 10 = $950.00
    # Quantity = 950 / 9.9950 = 95.047523
    assert fill.quantity == D("95.047523")
    expected_fee = fill.price * fill.quantity * FUTURES_FEE_RATE
    assert fill.fee == expected_fee

    # Stop Loss is 1.5x ATR above entry: 9.9950 + 1.5 * 1.0 = 11.4950
    assert position.stop_price == D("11.4950")
    assert position.stop_loss == D("11.4950")

    # Take Profit is 3.0x ATR below entry: 9.9950 - 3.0 * 1.0 = 6.9950
    assert position.target_price == D("6.9950")
    assert position.take_profit == D("6.9950")

    # Liquidation price for 10x short: 9.9950 * (1 + 1/10) = 10.9945
    assert position.liquidation_price == D("9.9950") * (D("1") + D("1") / D("10"))


def test_short_take_profit_exit_and_pnl_math():
    executor, position = enter_default(side="short", leverage=10)
    assert position is not None

    # Candle price drops to 6.5, triggering take-profit on short
    trade = executor.on_candle_closed(
        candle(
            opened=START + timedelta(hours=1),
            open=D("9.5"),
            high=D("9.8"),
            low=D("6.5"),
            close=D("6.8"),
        )
    )
    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.direction == "SHORT"

    # Exit price with adverse slippage for short exit: target_price * (1 + 0.0005)
    expected_exit_price = position.target_price * (D("1") + CORE_SLIPPAGE)
    assert trade.exit_fill.price == expected_exit_price

    # Gross PnL for SHORT: (entry_price - exit_price) * quantity
    expected_gross = (trade.entry_fill.price - trade.exit_fill.price) * trade.quantity
    assert trade.gross_pnl == expected_gross
    assert trade.gross_pnl > Decimal("0")

    # Net PnL = gross - fees
    total_fees = trade.entry_fill.fee + trade.exit_fill.fee
    assert trade.net_pnl == expected_gross - total_fees
    assert trade.realized_pnl == trade.net_pnl
    assert executor.cash > DEFAULT_STARTING_CASH


def test_short_stop_loss_exit_and_pnl_math():
    executor, position = enter_default(side="short", leverage=5)
    assert position is not None

    # Stop price is 11.4950, liquidation price is 11.9940.
    # Candle high rises to 11.6, triggering stop-loss on short before liquidation
    trade = executor.on_candle_closed(
        candle(
            opened=START + timedelta(hours=1),
            open=D("10"),
            high=D("11.6"),
            low=D("9.8"),
            close=D("11.5"),
        )
    )
    assert trade is not None
    assert trade.exit_reason == "stop_loss"

    # Exit price with adverse slippage for short: stop_price * (1 + 0.0005)
    expected_exit_price = position.stop_price * (D("1") + CORE_SLIPPAGE)
    assert trade.exit_fill.price == expected_exit_price

    # Gross PnL for SHORT when price rises is negative: (entry - exit) * qty
    expected_gross = (trade.entry_fill.price - trade.exit_fill.price) * trade.quantity
    assert trade.gross_pnl == expected_gross
    assert trade.gross_pnl < Decimal("0")
    assert trade.net_pnl < Decimal("0")


# --- Liquidation Protection Tests ---


def test_long_liquidation_loss_capped_at_margin():
    executor, position = enter_default(leverage=5)
    assert position is not None
    margin_allocated = position.margin_used

    # Severe flash crash drops price to 1.0 (far below liquidation price 8.004)
    trade = executor.on_candle_closed(
        candle(
            opened=START + timedelta(hours=1), open=D("5"), high=D("5"), low=D("1"), close=D("2")
        )
    )
    assert trade is not None
    assert trade.exit_reason == "liquidation"
    # Net loss must not exceed allocated margin
    assert trade.net_pnl >= -margin_allocated
    assert executor.cash >= Decimal("0")


def test_short_liquidation_loss_capped_at_margin():
    executor, position = enter_default(side="short", leverage=5)
    assert position is not None
    margin_allocated = position.margin_used

    # Massive surge drops short price (rises to 25.0, far above short liquidation price 12.0)
    trade = executor.on_candle_closed(
        candle(
            opened=START + timedelta(hours=1),
            open=D("15"),
            high=D("25"),
            low=D("10"),
            close=D("22"),
        )
    )
    assert trade is not None
    assert trade.exit_reason == "liquidation"
    # Net loss must not exceed allocated margin
    assert trade.net_pnl >= -margin_allocated
    assert executor.cash >= Decimal("0")


# --- Single-Trade State Machine: Lock & Unlock/Repeat Tests ---


def test_single_trade_lock_blocks_new_signals():
    executor, position = enter_default()
    assert position is not None
    assert len(executor.positions) == 1

    # While position is active, submitting any new signal is strictly blocked
    reentry = signal(symbol="ETHUSDT", opened=START + timedelta(hours=1), side="short")
    executor.on_candle_closed(candle("ETHUSDT", opened=START + timedelta(hours=1)))
    assert not executor.submit_signal(reentry)
    assert len(executor.positions) == 1


def test_single_trade_unlock_and_repeat_after_exit():
    # 1. Enter First Trade (Long)
    executor, position1 = enter_default(symbol="BTCUSDT")
    assert position1 is not None
    assert len(executor.positions) == 1

    # 2. First Trade Hits Take-Profit and Closes
    trade1 = executor.on_candle_closed(
        candle(
            "BTCUSDT", opened=START + timedelta(hours=1), high=D("15"), low=D("9"), close=D("14")
        )
    )
    assert trade1 is not None
    assert len(executor.positions) == 0  # Scanner unlocked!

    # 3. Enter Second Trade (Short on ETHUSDT) Immediately in Next Cycle
    prepare_signal(
        executor, symbol="ETHUSDT", opened=START + timedelta(hours=1), side="short", leverage=7
    )
    position2 = executor.on_open_observation(
        observation("ETHUSDT", opened=START + timedelta(hours=2), price=D("20"))
    )
    assert position2 is not None
    assert position2.direction == "SHORT"
    assert position2.leverage == 7
    assert len(executor.positions) == 1


# --- Tie-break & Boundary Condition Tests ---


def test_stop_wins_when_stop_and_target_both_breached():
    executor, position = enter_default()
    assert position is not None
    # Stop price is 8.5050, target is 13.0050, liquidation is 8.0040.
    # Candle breaches both low (8.3 < stop 8.505, but > liq 8.004) and high (14.0 > target 13.005)
    trade = executor.on_candle_closed(
        candle(
            opened=START + timedelta(hours=1),
            open=D("10"),
            high=D("14"),
            low=D("8.3"),
            close=D("10"),
        )
    )
    assert trade is not None
    assert trade.exit_reason == "stop_loss"


def test_invalid_atr_signal_fails_closed():
    executor = PaperExecutor(config())
    executor.on_candle_closed(candle())
    assert not executor.submit_signal(signal(atr=float("nan")))
    assert executor.pending_signals == ()


def test_minimum_quantity_rejection():
    # If notional / price is below minimum_quantity
    executor = PaperExecutor(config(minimum_quantity=D("1000")))
    prepare_signal(executor)
    assert executor.on_open_observation(observation()) is None
    assert executor.positions == {}
