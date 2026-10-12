from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.backtest.models import BacktestConfig
from app.market.constants import EXCHANGE
from app.market.models import Candle, CandleOpenObservation
from app.strategy.breakout import BreakoutSignal
from app.trading.paper import PaperExecutor

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


def signal(symbol="BTCUSDT", opened=START, *, atr=1.0):
    return BreakoutSignal(
        exchange=EXCHANGE,
        symbol=symbol,
        side="long",
        decision_time=opened + timedelta(hours=1),
        signal_open_time=opened,
        atr_14=atr,
    )


def prepare_signal(executor, symbol="BTCUSDT", opened=START, *, atr=1.0):
    executor.on_candle_closed(candle(symbol, opened))
    intent = signal(symbol, opened, atr=atr)
    assert executor.submit_signal(intent)
    return intent


def enter_default(executor=None, symbol="BTCUSDT", opened=START, *, price=D("10"), atr=1.0):
    executor = executor or PaperExecutor(config())
    prepare_signal(executor, symbol, opened, atr=atr)
    position = executor.on_open_observation(
        observation(symbol, opened + timedelta(hours=1), price=price)
    )
    return executor, position


def test_signal_and_next_open_create_decimal_entry_fill_with_both_timestamps():
    executor, position = enter_default()
    assert position is not None
    fill = position.entry_fill
    assert fill.timestamp == START + timedelta(hours=1)
    assert fill.observed_at == START + timedelta(hours=1, seconds=1)
    assert fill.reference_price == D("10")
    assert fill.price == D("10.0050")
    assert fill.fee == fill.price * fill.quantity * D("0.001")
    assert fill.quantity == D("3.333333")


def test_signal_after_observed_open_is_not_filled_retroactively():
    executor = PaperExecutor(config())
    executor.on_candle_closed(candle())
    assert executor.on_open_observation(observation()) is None
    assert not executor.submit_signal(signal())
    assert executor.positions == {}


def test_entry_uses_only_first_open_observation_for_price():
    executor, position = enter_default()
    assert position is not None
    later_update = observation(price=D("99"), observed=START + timedelta(hours=1, seconds=5))
    assert executor.on_open_observation(later_update) is None
    assert executor.positions["BTCUSDT"].entry_fill.reference_price == D("10")


def test_entry_stop_target_and_risk_use_decimal_atr_conversion():
    _, position = enter_default()
    assert position is not None
    assert position.stop_price == D("8.5050")
    assert position.target_price == D("13.0050")
    assert position.initial_risk == D("4.9999995")


def test_minimum_quantity_rejects_entry():
    executor = PaperExecutor(config(minimum_quantity=D("10")))
    prepare_signal(executor)
    assert executor.on_open_observation(observation()) is None
    assert executor.positions == {}


def test_insufficient_cash_including_entry_fee_rejects_entry():
    executor = PaperExecutor(config())
    prepare_signal(executor, atr=0.03335)
    assert executor.on_open_observation(observation()) is None
    assert executor.cash == D("100")


def test_quantity_step_is_floored_for_paper_entry():
    executor = PaperExecutor(config(quantity_step=D("0.1"), minimum_quantity=D("0.1")))
    _, position = enter_default(executor)
    assert position is not None
    assert position.quantity == D("3.3")


def test_same_symbol_open_position_rejects_new_signal_without_pyramiding():
    executor, position = enter_default()
    assert position is not None
    next_candle = candle(opened=START + timedelta(hours=1))
    assert executor.on_candle_closed(next_candle) is None
    assert not executor.submit_signal(signal(opened=START + timedelta(hours=1)))
    assert len(executor.positions) == 1


def test_pending_signal_is_expired_when_next_candle_closes_without_open_observation():
    executor = PaperExecutor(config())
    prepare_signal(executor)
    assert executor.on_candle_closed(candle(opened=START + timedelta(hours=1))) is None
    assert executor.pending_signals == ()
    assert executor.positions == {}


def test_completed_candle_stop_loss_uses_stop_and_adverse_slippage():
    executor, position = enter_default()
    assert position is not None
    trade = executor.on_candle_closed(
        candle(opened=START + timedelta(hours=1), high=D("12"), low=D("8"), close=D("9"))
    )
    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_fill.reference_price == position.stop_price
    assert trade.exit_fill.price == position.stop_price * D("0.9995")


def test_completed_candle_take_profit_uses_target_and_adverse_slippage():
    executor, position = enter_default()
    assert position is not None
    trade = executor.on_candle_closed(
        candle(opened=START + timedelta(hours=1), high=D("14"), low=D("9"), close=D("13.5"))
    )
    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.exit_fill.reference_price == position.target_price
    assert trade.exit_fill.price == position.target_price * D("0.9995")


def test_completed_candle_stop_wins_when_stop_and_target_are_both_touched():
    executor, position = enter_default()
    assert position is not None
    trade = executor.on_candle_closed(
        candle(opened=START + timedelta(hours=1), high=D("14"), low=D("8"), close=D("10"))
    )
    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_fill.reference_price == position.stop_price


def test_completed_gap_stop_uses_bar_open_not_stop_level():
    executor, _ = enter_default()
    trade = executor.on_candle_closed(
        candle(
            opened=START + timedelta(hours=1),
            open=D("8"),
            high=D("9"),
            low=D("7.5"),
            close=D("8.5"),
        )
    )
    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_fill.reference_price == D("8")
    assert trade.exit_fill.price == D("8") * D("0.9995")


def test_completed_gap_target_uses_bar_open_when_stop_is_not_touched():
    executor, _ = enter_default()
    trade = executor.on_candle_closed(
        candle(
            opened=START + timedelta(hours=1),
            open=D("14"),
            high=D("15"),
            low=D("13.5"),
            close=D("14.5"),
        )
    )
    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.exit_fill.reference_price == D("14")


def test_signal_at_same_time_as_exit_can_be_accepted_after_exit_is_processed():
    executor, _ = enter_default()
    exit_candle = candle(
        opened=START + timedelta(hours=1),
        open=D("12"),
        high=D("14"),
        low=D("11.5"),
        close=D("13.5"),
    )
    assert executor.on_candle_closed(exit_candle) is not None
    reentry = signal(opened=START + timedelta(hours=1))
    assert executor.submit_signal(reentry)
    filled = executor.on_open_observation(
        observation(opened=START + timedelta(hours=2), price=D("13.5"))
    )
    assert filled is not None


def test_daily_loss_blocks_new_entry_but_does_not_prevent_existing_exit():
    executor = PaperExecutor(config(quantity_step=D("0.1"), minimum_quantity=D("0.1")))
    prepare_signal(executor, "BTCUSDT")
    prepare_signal(executor, "ETHUSDT")
    assert executor.on_open_observation(observation("BTCUSDT")) is not None
    assert executor.on_open_observation(observation("ETHUSDT")) is not None

    btc_trade = executor.on_candle_closed(
        candle(
            "BTCUSDT",
            START + timedelta(hours=1),
            open=D("1"),
            high=D("2"),
            low=D("0.5"),
            close=D("1"),
        )
    )
    assert btc_trade is not None
    assert executor.daily_loss_breached
    assert "ETHUSDT" in executor.positions

    executor.on_candle_closed(
        candle(
            "ETHUSDT",
            START + timedelta(hours=1),
            open=D("10"),
            high=D("11"),
            low=D("9"),
            close=D("10"),
        )
    )
    prepare_signal(executor, "SOLUSDT", START + timedelta(hours=1))
    assert (
        executor.on_open_observation(
            observation("SOLUSDT", START + timedelta(hours=2), price=D("10"))
        )
        is None
    )
    eth_trade = executor.on_candle_closed(
        candle(
            "ETHUSDT",
            START + timedelta(hours=2),
            open=D("8"),
            high=D("9"),
            low=D("7"),
            close=D("8"),
        )
    )
    assert eth_trade is not None
    assert eth_trade.exit_reason == "stop_loss"


def test_multiple_symbols_share_open_risk_and_position_limits():
    executor = PaperExecutor(config())
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        prepare_signal(executor, symbol)
    positions = [
        executor.on_open_observation(observation(symbol))
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    ]
    # The shared 15% risk cap (max_concurrent_positions * core_risk_per_trade) rejects the third position after costs.
    assert sum(position is not None for position in positions) == 2
    assert executor.portfolio.open_risk <= executor.portfolio.equity(
        executor.portfolio.positions
        and {symbol: D("10") for symbol in executor.portfolio.positions}
        or {}
    ) * D("0.15")
    assert len(executor.positions) <= 3


def test_three_positions_can_be_opened_when_quantity_rounding_leaves_risk_headroom():
    executor = PaperExecutor(config(quantity_step=D("0.1"), minimum_quantity=D("0.1")))
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        prepare_signal(executor, symbol)
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert executor.on_open_observation(observation(symbol)) is not None
    assert len(executor.positions) == 3


def test_non_one_hour_open_observation_cannot_fill_strategy_signal():
    executor = PaperExecutor(config())
    prepare_signal(executor)
    four_hour = CandleOpenObservation(
        exchange=EXCHANGE,
        symbol="BTCUSDT",
        interval="4h",
        open_time=START + timedelta(hours=1),
        observed_at=START + timedelta(hours=1, seconds=1),
        open=D("10"),
    )
    assert executor.on_open_observation(four_hour) is None
    assert executor.positions == {}


def test_invalid_atr_signal_fails_closed():
    executor = PaperExecutor(config())
    executor.on_candle_closed(candle())
    assert not executor.submit_signal(signal(atr=float("nan")))
    assert executor.pending_signals == ()


def test_out_of_order_or_duplicate_closed_candle_is_rejected():
    executor = PaperExecutor(config())
    executor.on_candle_closed(candle())
    with pytest.raises(ValueError, match="strictly chronological"):
        executor.on_candle_closed(candle(opened=START))


def test_missing_completed_candle_interval_is_rejected():
    executor = PaperExecutor(config())
    executor.on_candle_closed(candle())
    with pytest.raises(ValueError, match="contiguous"):
        executor.on_candle_closed(candle(opened=START + timedelta(hours=2)))


def test_repeated_event_sequence_is_deterministic():
    def run_once():
        executor, _ = enter_default()
        trade = executor.on_candle_closed(
            candle(opened=START + timedelta(hours=1), high=D("14"), low=D("8"), close=D("10"))
        )
        return trade, executor.cash, executor.trades

    assert run_once() == run_once()


def test_core_vs_satellite_category_and_dollar_risk_sizing():
    executor = PaperExecutor(config())
    assert executor.get_asset_category("BTCUSDT") == "core"
    assert executor.get_asset_category("ZECUSDT") == "core"
    assert executor.get_asset_category("TAOUSDT") == "satellite"
    assert executor.get_asset_category("FETUSDT") == "satellite"

    assert executor.calculate_dollar_risk("BTCUSDT", D("1000")) == D("50.00")
    assert executor.calculate_dollar_risk("TAOUSDT", D("1000")) == D("40.00")


def test_satellite_daily_loss_blocks_satellite_but_allows_core():
    executor = PaperExecutor(config(quantity_step=D("0.1"), minimum_quantity=D("0.1")))
    prepare_signal(executor, "TAOUSDT")
    assert executor.on_open_observation(observation("TAOUSDT")) is not None

    # Trigger >4% loss on TAOUSDT (satellite basket)
    tao_trade = executor.on_candle_closed(
        candle(
            "TAOUSDT",
            START + timedelta(hours=1),
            open=D("1"),
            high=D("2"),
            low=D("0.5"),
            close=D("1"),
        )
    )
    assert tao_trade is not None
    assert executor.satellite_daily_loss_breached is True
    assert executor.core_daily_loss_breached is False

    # New satellite entry is blocked
    prepare_signal(executor, "FETUSDT", START + timedelta(hours=1))
    assert (
        executor.on_open_observation(
            observation("FETUSDT", START + timedelta(hours=2), price=D("10"))
        )
        is None
    )

    # Core entry is still allowed
    prepare_signal(executor, "BTCUSDT", START + timedelta(hours=1))
    assert (
        executor.on_open_observation(
            observation("BTCUSDT", START + timedelta(hours=2), price=D("10"))
        )
        is not None
    )


def test_tiered_slippage_core_vs_satellite():
    # 1. Verify get_slippage returns 0.05% for core coins and 0.20% for satellites
    for core_sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ZECUSDT"):
        assert PaperExecutor.get_slippage(core_sym) == D("0.0005")

    for sat_sym in ("TAOUSDT", "FETUSDT", "RENDERUSDT", "NEARUSDT", "INJUSDT"):
        assert PaperExecutor.get_slippage(sat_sym) == D("0.0020")

    # 2. Verify Core Coin fill price (BTCUSDT: 0.05% slippage on entry and exit, 0.1% fee)
    exec_core = PaperExecutor(config())
    prepare_signal(exec_core, "BTCUSDT")
    pos_core = exec_core.on_open_observation(observation("BTCUSDT", price=D("10")))
    assert pos_core is not None
    # Entry price: 10 * (1 + 0.0005) = 10.005
    assert pos_core.entry_fill.price == D("10.0050")
    # Fee: 10.005 * quantity * 0.001
    assert pos_core.entry_fill.fee == pos_core.entry_fill.price * pos_core.quantity * D("0.001")

    # Exit core via take-profit
    trade_core = exec_core.on_candle_closed(
        candle("BTCUSDT", START + timedelta(hours=1), high=D("15"), low=D("9"), close=D("14"))
    )
    assert trade_core is not None
    assert trade_core.exit_reason == "take_profit"
    # Exit price: target_price * (1 - 0.0005)
    assert trade_core.exit_fill.price == pos_core.target_price * D("0.9995")
    assert trade_core.exit_fill.fee == trade_core.exit_fill.price * trade_core.quantity * D("0.001")

    # 3. Verify Dynamic Satellite Coin fill price (TAOUSDT: 0.20% slippage on entry and exit, 0.1% fee)
    exec_sat = PaperExecutor(config())
    prepare_signal(exec_sat, "TAOUSDT")
    pos_sat = exec_sat.on_open_observation(observation("TAOUSDT", price=D("10")))
    assert pos_sat is not None
    # Entry price: 10 * (1 + 0.0020) = 10.02
    assert pos_sat.entry_fill.price == D("10.0200")
    # Fee remains flat 0.1%: 10.02 * quantity * 0.001
    assert pos_sat.entry_fill.fee == pos_sat.entry_fill.price * pos_sat.quantity * D("0.001")

    # Exit satellite via take-profit
    trade_sat = exec_sat.on_candle_closed(
        candle("TAOUSDT", START + timedelta(hours=1), high=D("15"), low=D("9"), close=D("14"))
    )
    assert trade_sat is not None
    assert trade_sat.exit_reason == "take_profit"
    # Exit price: target_price * (1 - 0.0020) = target_price * 0.9980
    assert trade_sat.exit_fill.price == pos_sat.target_price * D("0.9980")
    assert trade_sat.exit_fill.fee == trade_sat.exit_fill.price * trade_sat.quantity * D("0.001")
