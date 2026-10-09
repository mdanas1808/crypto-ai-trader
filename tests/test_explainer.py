from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pandas as pd
import pytest

from app.analysis.explainer import TradeExplainer
from app.backtest.data import MultiTimeframeBacktestData
from app.backtest.models import Fill, Trade
from app.strategy.breakout import BreakoutSignal

UTC = timezone.utc
START = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
D = Decimal


def make_test_data() -> MultiTimeframeBacktestData:
    """Generate minimal valid MultiTimeframeBacktestData for breakout signals."""
    context_rows = []
    context_times = []
    for offset in range(-20, 0):
        t = START + timedelta(hours=offset)
        context_times.append(t + timedelta(hours=1))
        context_rows.append(
            {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "open_time": t,
                "high": D("100.00"),
            }
        )
    context = pd.DataFrame(
        context_rows,
        index=pd.DatetimeIndex(context_times, name="decision_time", tz="UTC"),
    )

    t0 = START
    trading_rows = [
        {
            "exchange": "binance_spot",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open_time": t0,
            "open": D("100.00"),
            "high": D("102.50"),
            "low": D("99.50"),
            "close": D("101.50"),
            "volume": D("30.00"),
            "volume_sma_20": 15.00,
            "ema_200": 95.00,
            "rsi_14": 62.50,
            "atr_14": 2.50,
            "4h_close": 110.00,
            "4h_ema_20": 105.00,
            "4h_ema_50": 100.00,
            "4h_ema_200": 98.00,
        }
    ]
    trading = pd.DataFrame(
        trading_rows,
        index=pd.DatetimeIndex([t0 + timedelta(hours=1)], name="decision_time", tz="UTC"),
    )
    return MultiTimeframeBacktestData(context_frame=context, trading_frame=trading)


def make_test_signal() -> BreakoutSignal:
    return BreakoutSignal(
        exchange="binance_spot",
        symbol="BTCUSDT",
        side="long",
        decision_time=START + timedelta(hours=1),
        signal_open_time=START,
        atr_14=2.50,
    )


def make_test_trade(
    *,
    exit_reason: str = "take_profit",
    entry_price: Decimal = D("10.0050"),
    exit_price: Decimal = D("13.0000"),
    ref_exit_price: Decimal = D("13.0065"),
    stop_price: Decimal = D("8.5050"),
    target_price: Decimal = D("13.0050"),
    quantity: Decimal = D("0.333333"),
    net_pnl: Decimal = D("0.9900"),
    gross_pnl: Decimal = D("0.9983"),
) -> Trade:
    sig = make_test_signal()
    entry_fill = Fill(
        symbol="BTCUSDT",
        exchange="binance_spot",
        side="buy",
        timestamp=START + timedelta(hours=1),
        reference_price=D("10.0000"),
        price=entry_price,
        quantity=quantity,
        fee=D("0.0033"),
        reason="entry",
    )
    exit_fill = Fill(
        symbol="BTCUSDT",
        exchange="binance_spot",
        side="sell",
        timestamp=START + timedelta(hours=2),
        reference_price=ref_exit_price,
        price=exit_price,
        quantity=quantity,
        fee=D("0.0043"),
        reason=exit_reason,
    )
    return Trade(
        signal=sig,
        quantity=quantity,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        stop_price=stop_price,
        target_price=target_price,
        exit_reason=exit_reason,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
    )


def test_explainer_initialization():
    explainer = TradeExplainer()
    assert explainer.price_decimals == 4
    assert explainer.currency == "USDT"

    custom = TradeExplainer(price_decimals=2, currency="USD")
    assert custom.price_decimals == 2
    assert custom.currency == "USD"


def test_explain_signal_generates_detailed_reasoning():
    explainer = TradeExplainer()
    data = make_test_data()
    sig = make_test_signal()

    explanation = explainer.explain_signal(sig, data)

    # Core elements must be present
    assert "BTCUSDT" in explanation
    assert "binance_spot" in explanation
    # 4H regime
    assert "4H" in explanation
    assert "EMA 20" in explanation
    assert "105.00" in explanation
    assert "EMA 50" in explanation
    assert "100.00" in explanation
    assert "EMA 200" in explanation
    # 1H trend & resistance breakout
    assert "101.50" in explanation  # 1H close
    assert "95.00" in explanation   # 1H EMA 200
    assert "100.00" in explanation  # 20-candle resistance high
    # 1H momentum
    assert "RSI(14) is 62.50" in explanation
    # 1H volume surge
    assert "30.00" in explanation
    assert "15.00" in explanation
    assert "2.00x" in explanation
    # ATR volatility
    assert "2.50" in explanation


def test_explain_signal_validation():
    explainer = TradeExplainer()
    data = make_test_data()
    sig = make_test_signal()

    with pytest.raises(TypeError, match="signal must be BreakoutSignal"):
        explainer.explain_signal("invalid_signal", data)  # type: ignore

    with pytest.raises(TypeError, match="context_data must be MultiTimeframeBacktestData"):
        explainer.explain_signal(sig, "invalid_data")  # type: ignore

    # Missing decision time
    unknown_sig = BreakoutSignal(
        exchange="binance_spot",
        symbol="BTCUSDT",
        side="long",
        decision_time=START + timedelta(days=10),
        signal_open_time=START + timedelta(days=10, hours=-1),
        atr_14=1.0,
    )
    with pytest.raises(ValueError, match="not found in trading_frame"):
        explainer.explain_signal(unknown_sig, data)


def test_explain_trade_exit_take_profit_normal():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="take_profit",
        ref_exit_price=D("13.0050"),
        target_price=D("13.0050"),
    )

    explanation = explainer.explain_trade_exit(trade)

    assert "Take Profit Hit" in explanation
    assert "BTCUSDT" in explanation
    assert "Net PnL +0.9900 USDT" in explanation
    assert "Gross: +0.9983 USDT" in explanation
    assert "Entry 10.0050" in explanation
    assert "Exit 13.0000" in explanation
    assert "0.333333" in explanation


def test_explain_trade_exit_take_profit_with_gap():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="take_profit",
        ref_exit_price=D("14.0000"),  # Gapped up above target
        target_price=D("13.0050"),
        exit_price=D("13.9930"),
    )

    explanation = explainer.explain_trade_exit(trade)

    assert "Take Profit Hit (Gap Open Slippage)" in explanation
    assert "opened above profit target" in explanation
    assert "14.0000" in explanation


def test_explain_trade_exit_stop_loss_normal():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="stop_loss",
        exit_price=D("8.5007"),
        ref_exit_price=D("8.5050"),
        stop_price=D("8.5050"),
        net_pnl=D("-0.5090"),
        gross_pnl=D("-0.5014"),
    )

    explanation = explainer.explain_trade_exit(trade)

    assert "Stopped Out" in explanation
    assert "Stop loss (8.5050) was triggered" in explanation
    assert "Net PnL -0.5090 USDT" in explanation


def test_explain_trade_exit_stop_loss_with_gap_down():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="stop_loss",
        exit_price=D("7.9960"),
        ref_exit_price=D("8.0000"),  # Gapped down below stop
        stop_price=D("8.5050"),
        net_pnl=D("-0.6760"),
        gross_pnl=D("-0.6683"),
    )

    explanation = explainer.explain_trade_exit(trade)

    assert "Stopped Out (Gap Down Slippage)" in explanation
    assert "gapped down below stop loss" in explanation
    assert "8.0000" in explanation


def test_explain_trade_exit_end_of_backtest():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="end_of_backtest",
        exit_price=D("11.0000"),
        ref_exit_price=D("11.0000"),
    )

    explanation = explainer.explain_trade_exit(trade)

    assert "Closed at End of Session" in explanation
    assert "simulation close" in explanation


def test_explain_trade_exit_type_error():
    explainer = TradeExplainer()
    with pytest.raises(TypeError, match="trade must be Trade"):
        explainer.explain_trade_exit("invalid_trade")  # type: ignore


def test_explainer_does_not_mutate_inputs():
    explainer = TradeExplainer()
    data = make_test_data()
    sig = make_test_signal()
    trade = make_test_trade()

    orig_signal_dict = sig.__dict__.copy()
    orig_trade_pnl = trade.net_pnl
    orig_trading_shape = data.trading_frame.shape

    explainer.explain_signal(sig, data)
    explainer.explain_trade_exit(trade)

    assert sig.__dict__ == orig_signal_dict
    assert trade.net_pnl == orig_trade_pnl
    assert data.trading_frame.shape == orig_trading_shape

