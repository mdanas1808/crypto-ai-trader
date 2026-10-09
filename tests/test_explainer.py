from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
from unittest.mock import AsyncMock, MagicMock
import httpx
import pandas as pd
import pytest

from app.analysis.explainer import (
    DEFAULT_GROQ_MODEL,
    GROQ_COMPLETIONS_URL,
    TradeExplainer,
)
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


def test_explainer_initialization(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    explainer = TradeExplainer()
    assert explainer.price_decimals == 4
    assert explainer.currency == "USDT"
    assert explainer.model == DEFAULT_GROQ_MODEL
    assert explainer.base_url == GROQ_COMPLETIONS_URL
    assert explainer.timeout == 5.0
    assert explainer.api_key is None

    custom = TradeExplainer(price_decimals=2, currency="USD", api_key="test_key", model="test-model")
    assert custom.price_decimals == 2
    assert custom.currency == "USD"
    assert custom.api_key == "test_key"
    assert custom.model == "test-model"

    # From env
    monkeypatch.setenv("GROQ_API_KEY", "  env_key_123  ")
    from_env = TradeExplainer()
    assert from_env.api_key == "env_key_123"

    # Empty env
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    from_empty = TradeExplainer()
    assert from_empty.api_key is None


@pytest.mark.asyncio
async def test_explain_signal_generates_detailed_reasoning():
    explainer = TradeExplainer()
    data = make_test_data()
    sig = make_test_signal()

    explanation = await explainer.explain_signal(sig, data)

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
    assert "2.5000" in explanation


def test_explain_signal_sync_and_template():
    explainer = TradeExplainer()
    data = make_test_data()
    sig = make_test_signal()

    sync_exp = explainer.explain_signal_sync(sig, data)
    tmpl_exp = explainer.explain_signal_template(sig, data)

    assert sync_exp == tmpl_exp
    assert "BTCUSDT" in sync_exp
    assert "EMA 200" in sync_exp


@pytest.mark.asyncio
async def test_explain_signal_validation():
    explainer = TradeExplainer()
    data = make_test_data()
    sig = make_test_signal()

    with pytest.raises(TypeError, match="signal must be BreakoutSignal"):
        await explainer.explain_signal("invalid_signal", data)  # type: ignore

    with pytest.raises(TypeError, match="context_data must be MultiTimeframeBacktestData"):
        await explainer.explain_signal(sig, "invalid_data")  # type: ignore

    # Empty trading frame
    empty_data = MultiTimeframeBacktestData(
        context_frame=pd.DataFrame(), trading_frame=pd.DataFrame()
    )
    with pytest.raises(ValueError, match="trading_frame is empty"):
        await explainer.explain_signal(sig, empty_data)

    # Unknown decision time
    unknown_sig = BreakoutSignal(
        exchange="binance_spot",
        symbol="BTCUSDT",
        side="long",
        decision_time=START + timedelta(days=99),
        signal_open_time=START + timedelta(days=99),
        atr_14=2.50,
    )
    with pytest.raises(ValueError, match="not found in trading_frame"):
        await explainer.explain_signal(unknown_sig, data)


@pytest.mark.asyncio
async def test_explain_trade_exit_take_profit_normal():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="take_profit",
        ref_exit_price=D("13.0050"),
        target_price=D("13.0050"),
    )

    explanation = await explainer.explain_trade_exit(trade)

    assert "Take Profit Hit" in explanation
    assert "Profit target (13.0050) was reached" in explanation
    assert "Net PnL +0.9900 USDT" in explanation
    assert "Fees: Total 0.0076 USDT" in explanation


@pytest.mark.asyncio
async def test_explain_trade_exit_take_profit_with_gap():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="take_profit",
        ref_exit_price=D("14.0000"),  # Gapped up above target
        target_price=D("13.0050"),
        exit_price=D("13.9930"),
    )

    explanation = await explainer.explain_trade_exit(trade)

    assert "Take Profit Hit (Gap Open Slippage)" in explanation
    assert "Market opened above profit target at 14.0000" in explanation


@pytest.mark.asyncio
async def test_explain_trade_exit_stop_loss_normal():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="stop_loss",
        exit_price=D("8.5007"),
        ref_exit_price=D("8.5050"),
        stop_price=D("8.5050"),
        net_pnl=D("-0.5090"),
        gross_pnl=D("-0.5014"),
    )

    explanation = await explainer.explain_trade_exit(trade)

    assert "Stopped Out" in explanation
    assert "Stop loss (8.5050) was triggered" in explanation
    assert "Net PnL -0.5090 USDT" in explanation


@pytest.mark.asyncio
async def test_explain_trade_exit_stop_loss_with_gap_down():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="stop_loss",
        exit_price=D("7.9960"),
        ref_exit_price=D("8.0000"),  # Gapped down below stop
        stop_price=D("8.5050"),
        net_pnl=D("-0.6760"),
        gross_pnl=D("-0.6683"),
    )

    explanation = await explainer.explain_trade_exit(trade)

    assert "Stopped Out (Gap Down Slippage)" in explanation
    assert "gapped down below stop loss" in explanation
    assert "8.0000" in explanation


@pytest.mark.asyncio
async def test_explain_trade_exit_end_of_backtest():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="end_of_backtest",
        exit_price=D("11.0000"),
        ref_exit_price=D("11.0000"),
    )

    explanation = await explainer.explain_trade_exit(trade)

    assert "Closed at End of Session" in explanation
    assert "simulation close" in explanation


def test_explain_trade_exit_sync_and_template():
    explainer = TradeExplainer()
    trade = make_test_trade(
        exit_reason="take_profit",
        ref_exit_price=D("13.0050"),
        target_price=D("13.0050"),
    )

    sync_exp = explainer.explain_trade_exit_sync(trade)
    tmpl_exp = explainer.explain_trade_exit_template(trade)

    assert sync_exp == tmpl_exp
    assert "Take Profit Hit" in sync_exp


@pytest.mark.asyncio
async def test_explain_trade_exit_type_error():
    explainer = TradeExplainer()
    with pytest.raises(TypeError, match="trade must be Trade"):
        await explainer.explain_trade_exit("invalid_trade")  # type: ignore


@pytest.mark.asyncio
async def test_explainer_does_not_mutate_inputs():
    explainer = TradeExplainer()
    data = make_test_data()
    sig = make_test_signal()
    trade = make_test_trade()

    orig_signal_dict = sig.__dict__.copy()
    orig_trade_pnl = trade.net_pnl
    orig_trading_shape = data.trading_frame.shape

    await explainer.explain_signal(sig, data)
    await explainer.explain_trade_exit(trade)

    assert sig.__dict__ == orig_signal_dict
    assert trade.net_pnl == orig_trade_pnl
    assert data.trading_frame.shape == orig_trading_shape


# =========================================================================
# LLM Integration Tests (Groq API mock)
# =========================================================================


@pytest.mark.asyncio
async def test_llm_explain_signal_success():
    expected_commentary = (
        "BTCUSDT breakout confirmed above 100.00 resistance with 2.0x volume surge "
        "and bullish 4H regime alignment. RSI at 62.5 confirms strong momentum."
    )
    mock_post = AsyncMock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"  {expected_commentary}  \n",
                        }
                    }
                ]
            },
        )
    )
    mock_client = AsyncMock()
    mock_client.post = mock_post

    explainer = TradeExplainer(api_key="gsk_test_key", async_client=mock_client)
    data = make_test_data()
    sig = make_test_signal()

    result = await explainer.explain_signal(sig, data)

    assert result == expected_commentary
    assert mock_post.await_count == 1
    call_args, call_kwargs = mock_post.call_args
    assert call_kwargs["json"]["model"] == DEFAULT_GROQ_MODEL
    assert call_kwargs["headers"]["Authorization"] == "Bearer gsk_test_key"
    messages = call_kwargs["json"]["messages"]
    assert "quantitative crypto trading analyst" in messages[0]["content"]
    assert "Breakout Long Signal for BTCUSDT" in messages[1]["content"]
    assert "4H Higher-Timeframe Regime" in messages[1]["content"]
    assert "RSI(14) is 62.50" in messages[1]["content"]


@pytest.mark.asyncio
async def test_llm_explain_trade_exit_success():
    expected_retrospective = (
        "BTCUSDT long closed successfully at profit target 13.0000 yielding +0.9900 USDT (+29.7%). "
        "Execution achieved with minimal slippage and favorable risk-reward realization."
    )
    mock_post = AsyncMock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": expected_retrospective,
                        }
                    }
                ]
            },
        )
    )
    mock_client = AsyncMock()
    mock_client.post = mock_post

    explainer = TradeExplainer(api_key="gsk_test_key", async_client=mock_client)
    trade = make_test_trade()

    result = await explainer.explain_trade_exit(trade)

    assert result == expected_retrospective
    assert mock_post.await_count == 1
    call_args, call_kwargs = mock_post.call_args
    messages = call_kwargs["json"]["messages"]
    assert "quantitative trading post-trade analyst" in messages[0]["content"]
    assert "Trade Exit Retrospective for BTCUSDT" in messages[1]["content"]
    assert "Take Profit Hit" in messages[1]["content"]


def test_llm_explain_signal_and_exit_sync_success():
    expected_signal = "Sync LLM signal commentary."
    expected_exit = "Sync LLM exit retrospective."

    mock_client = MagicMock()
    mock_client.post.side_effect = [
        httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": expected_signal}}]},
        ),
        httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": expected_exit}}]},
        ),
    ]

    explainer = TradeExplainer(api_key="gsk_test_key", sync_client=mock_client)
    data = make_test_data()
    sig = make_test_signal()
    trade = make_test_trade()

    assert explainer.explain_signal_sync(sig, data) == expected_signal
    assert explainer.explain_trade_exit_sync(trade) == expected_exit
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_llm_fallback_when_api_key_missing_or_empty():
    mock_client = AsyncMock()
    explainer_none = TradeExplainer(api_key=None, async_client=mock_client)
    explainer_empty = TradeExplainer(api_key="", async_client=mock_client)
    explainer_spaces = TradeExplainer(api_key="   ", async_client=mock_client)

    data = make_test_data()
    sig = make_test_signal()
    trade = make_test_trade()

    for exp in (explainer_none, explainer_empty, explainer_spaces):
        res_sig = await exp.explain_signal(sig, data)
        res_exit = await exp.explain_trade_exit(trade)
        assert "Breakout Long Signal for BTCUSDT" in res_sig
        assert "Trade Exit for BTCUSDT" in res_exit

    # Zero network calls made
    assert mock_client.post.await_count == 0


@pytest.mark.asyncio
async def test_llm_fallback_on_timeout(caplog):
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.TimeoutException("Connection timed out after 5.0s")

    explainer = TradeExplainer(api_key="gsk_key", timeout=5.0, async_client=mock_client)
    data = make_test_data()
    sig = make_test_signal()

    with caplog.at_level(logging.WARNING):
        result = await explainer.explain_signal(sig, data)

    # Returns template fallback
    assert "Breakout Long Signal for BTCUSDT" in result
    assert "1. Regime (4H): Bullish alignment" in result
    # Logs warning
    assert "Groq LLM call timed out" in caplog.text


@pytest.mark.asyncio
async def test_llm_fallback_on_rate_limit_429(caplog):
    mock_client = AsyncMock()
    mock_client.post.return_value = httpx.Response(
        status_code=429, text='{"error": {"message": "Rate limit exceeded"}}'
    )

    explainer = TradeExplainer(api_key="gsk_key", async_client=mock_client)
    trade = make_test_trade()

    with caplog.at_level(logging.WARNING):
        result = await explainer.explain_trade_exit(trade)

    assert "Trade Exit for BTCUSDT" in result
    assert "Groq LLM API returned HTTP 429" in caplog.text


@pytest.mark.asyncio
async def test_llm_fallback_on_server_error_500(caplog):
    mock_client = AsyncMock()
    mock_client.post.return_value = httpx.Response(
        status_code=500, text="Internal Server Error"
    )

    explainer = TradeExplainer(api_key="gsk_key", async_client=mock_client)
    data = make_test_data()
    sig = make_test_signal()

    with caplog.at_level(logging.WARNING):
        result = await explainer.explain_signal(sig, data)

    assert "Breakout Long Signal for BTCUSDT" in result
    assert "Groq LLM API returned HTTP 500" in caplog.text


@pytest.mark.asyncio
async def test_llm_fallback_on_network_error(caplog):
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("Failed to resolve host api.groq.com")

    explainer = TradeExplainer(api_key="gsk_key", async_client=mock_client)
    data = make_test_data()
    sig = make_test_signal()

    with caplog.at_level(logging.WARNING):
        result = await explainer.explain_signal(sig, data)

    assert "Breakout Long Signal for BTCUSDT" in result
    assert "Groq LLM call failed (ConnectError)" in caplog.text


@pytest.mark.asyncio
async def test_llm_fallback_on_malformed_json_or_empty_choices(caplog):
    mock_client = AsyncMock()
    mock_client.post.return_value = httpx.Response(
        status_code=200, json={"choices": []}  # Empty choices
    )

    explainer = TradeExplainer(api_key="gsk_key", async_client=mock_client)
    trade = make_test_trade()

    with caplog.at_level(logging.WARNING):
        result = await explainer.explain_trade_exit(trade)

    assert "Trade Exit for BTCUSDT" in result
    assert "missing choices" in caplog.text
