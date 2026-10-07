from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtest.data import (
    align_timeframes,
    candle_completion_times,
    prepare_backtest_data,
    prepare_multi_timeframe_backtest_data,
)
from app.backtest.engine import iter_walk_forward
from app.database.models import CandleRecord
from app.indicators.technical import INDICATOR_COLUMNS
from app.market.models import Candle
from app.market.repository import persist_candles
from app.market.validation import CandleSequenceValidationError


def make_candles(count: int, *, gap_at: int | None = None) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for index in range(count):
        hour = index + (1 if gap_at is not None and index >= gap_at else 0)
        close = Decimal(100 + index)
        candles.append(
            Candle(
                exchange="binance_spot",
                symbol="BTCUSDT",
                interval="1h",
                open_time=start + timedelta(hours=hour),
                open=close,
                high=close + Decimal("2"),
                low=close - Decimal("2"),
                close=close,
                volume=Decimal("1.25"),
            )
        )
    return candles


def test_prepare_backtest_data_retrieves_and_calculates_without_exchange(
    db_session: Session, monkeypatch
) -> None:
    candles = make_candles(25)
    persist_candles(db_session, candles)

    def fail_if_exchange_is_used(*args, **kwargs):
        raise AssertionError("backtest data preparation must not contact Binance")

    monkeypatch.setattr("app.market.exchange.BinanceSpotClient", fail_if_exchange_is_used)

    result = prepare_backtest_data(db_session, "BTCUSDT", "1h")

    assert len(result) == 25
    assert tuple(column for column in INDICATOR_COLUMNS if column in result.columns) == INDICATOR_COLUMNS
    assert result.loc[19, "sma_20"] == pytest.approx(sum(range(100, 120)) / 20)
    assert all(isinstance(value, Decimal) for value in result["open"])
    assert all(isinstance(value, Decimal) for value in result["high"])
    assert all(isinstance(value, Decimal) for value in result["low"])
    assert all(isinstance(value, Decimal) for value in result["close"])
    assert all(isinstance(value, Decimal) for value in result["volume"])
    assert result.loc[0, "ema_20"] != result.loc[0, "ema_20"]  # warm-up remains NaN
    assert len(db_session.scalars(select(CandleRecord)).all()) == len(candles)


def test_prepare_backtest_data_propagates_gap_validation_error(db_session: Session) -> None:
    persist_candles(db_session, make_candles(3, gap_at=2))

    with pytest.raises(CandleSequenceValidationError) as captured:
        prepare_backtest_data(db_session, "BTCUSDT", "1h")

    assert captured.value.reason == "gap"


def make_indicator_frame(count: int = 4) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "open_time": [start + timedelta(hours=index) for index in range(count)],
            "close": list(range(100, 100 + count)),
            "ema_20": [None] * count,
        }
    )


def test_walk_forward_yields_chronological_prefixes_without_future_rows() -> None:
    frame = make_indicator_frame()
    steps = list(iter_walk_forward(frame))

    assert [step["open_time"].iloc[-1] for step in steps] == list(frame["open_time"])
    assert len(steps[0]) == 1
    assert steps[0]["close"].tolist() == [100]
    for index, step in enumerate(steps):
        assert step["close"].tolist() == list(range(100, 101 + index))
        assert len(step) == index + 1
    assert len(steps[-1]) == len(frame)
    pd.testing.assert_frame_equal(frame, make_indicator_frame())


def test_walk_forward_prefixes_are_copies_and_do_not_mutate_input() -> None:
    frame = make_indicator_frame()
    original = frame.copy(deep=True)

    first_step = next(iter_walk_forward(frame))
    first_step.loc[0, "close"] = -1

    pd.testing.assert_frame_equal(frame, original)


def test_walk_forward_empty_dataframe_yields_no_steps() -> None:
    frame = pd.DataFrame(columns=["open_time", "close", "ema_20"])

    assert list(iter_walk_forward(frame)) == []


def test_walk_forward_single_row_yields_one_single_row_frame() -> None:
    frame = make_indicator_frame(1)

    steps = list(iter_walk_forward(frame))

    assert len(steps) == 1
    assert len(steps[0]) == 1
    pd.testing.assert_frame_equal(steps[0].reset_index(drop=True), frame)


def test_walk_forward_prefixes_are_independent_from_each_other() -> None:
    frame = make_indicator_frame(3)
    original = frame.copy(deep=True)
    steps = list(iter_walk_forward(frame))

    steps[1].loc[0, "close"] = -1

    pd.testing.assert_frame_equal(steps[2].reset_index(drop=True), original)
    pd.testing.assert_frame_equal(frame, original)


def test_walk_forward_rejects_unordered_or_duplicate_times() -> None:
    frame = make_indicator_frame()
    frame.loc[1, "open_time"], frame.loc[2, "open_time"] = (
        frame.loc[2, "open_time"],
        frame.loc[1, "open_time"],
    )

    with pytest.raises(ValueError, match="strictly chronological"):
        list(iter_walk_forward(frame))


def make_aligned_input(interval: str, opens: list[datetime]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": ["binance_spot"] * len(opens),
            "symbol": ["BTCUSDT"] * len(opens),
            "open_time": opens,
            "interval": [interval] * len(opens),
            "close": [float(100 + index) for index in range(len(opens))],
            **{
                column: [float(1000 + index) for index in range(len(opens))]
                for column in INDICATOR_COLUMNS
            },
        }
    )


def test_four_hour_candle_ending_at_decision_boundary_is_eligible() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hourly = make_aligned_input("1h", [start + timedelta(hours=3)])
    four_hour = make_aligned_input("4h", [start])

    aligned = align_timeframes(hourly, four_hour)

    assert aligned.index[0] == start + timedelta(hours=4)
    assert aligned.iloc[0]["4h_open_time"] == start
    assert aligned.iloc[0]["4h_ema_20"] == 1000.0
    assert aligned.iloc[0]["4h_completion_time"] == start + timedelta(hours=4)


def test_four_hour_candle_ending_after_decision_boundary_is_not_eligible() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hourly = make_aligned_input("1h", [start])
    four_hour = make_aligned_input("4h", [start])

    aligned = align_timeframes(hourly, four_hour)

    assert aligned.index[0] == start + timedelta(hours=1)
    assert pd.isna(aligned.iloc[0]["4h_ema_20"])
    assert pd.isna(aligned.iloc[0]["4h_open_time"])


def test_alignment_selects_latest_eligible_four_hour_candle() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hourly = make_aligned_input("1h", [start + timedelta(hours=7)])
    four_hour = make_aligned_input(
        "4h", [start, start + timedelta(hours=4), start + timedelta(hours=8)]
    )

    aligned = align_timeframes(hourly, four_hour)

    assert aligned.iloc[0]["4h_open_time"] == start + timedelta(hours=4)
    assert aligned.iloc[0]["4h_ema_20"] == 1001.0


def test_alignment_has_no_four_hour_value_before_first_completion() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hourly = make_aligned_input("1h", [start])
    four_hour = make_aligned_input("4h", [start])

    aligned = align_timeframes(hourly, four_hour)

    assert pd.isna(aligned.iloc[0]["4h_ema_20"])
    assert pd.isna(aligned.iloc[0]["4h_completion_time"])


def test_future_four_hour_indicator_never_leaks_into_earlier_decision() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hourly = make_aligned_input(
        "1h", [start, start + timedelta(hours=1), start + timedelta(hours=2)]
    )
    four_hour = make_aligned_input("4h", [start])
    four_hour.loc[0, "ema_20"] = 987654.0

    aligned = align_timeframes(hourly, four_hour)

    assert aligned["4h_ema_20"].isna().all()
    assert aligned["4h_open_time"].isna().all()


def test_completion_time_helpers_use_interval_duration_and_utc() -> None:
    utc = datetime(2024, 1, 1, 3, tzinfo=timezone.utc)
    offset = datetime(2024, 1, 1, 5, tzinfo=timezone(timedelta(hours=2)))

    assert candle_completion_times([utc], "1h")[0] == datetime(
        2024, 1, 1, 4, tzinfo=timezone.utc
    )
    assert candle_completion_times([offset], "1h")[0] == datetime(
        2024, 1, 1, 4, tzinfo=timezone.utc
    )
    assert candle_completion_times(
        [datetime(2024, 1, 1, tzinfo=timezone.utc)], "4h"
    )[0] == datetime(2024, 1, 1, 4, tzinfo=timezone.utc)


def test_alignment_preserves_utc_decision_timeline_for_offset_inputs() -> None:
    hourly_open = datetime(2024, 1, 1, 5, tzinfo=timezone(timedelta(hours=2)))
    four_hour_open = datetime(2024, 1, 1, 2, tzinfo=timezone(timedelta(hours=2)))
    aligned = align_timeframes(
        make_aligned_input("1h", [hourly_open]),
        make_aligned_input("4h", [four_hour_open]),
    )

    assert aligned.index[0] == datetime(2024, 1, 1, 4, tzinfo=timezone.utc)
    assert str(aligned.index.tz) == "UTC"
    assert aligned.iloc[0]["4h_open_time"] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_prepare_multitimeframe_data_uses_validated_repository_series(
    db_session: Session,
) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = []
    for interval, count, step in (("1h", 5, 1), ("4h", 2, 4)):
        for index in range(count):
            close = Decimal(100 + index)
            records.append(
                Candle(
                    exchange="binance_spot",
                    symbol="BTCUSDT",
                    interval=interval,
                    open_time=start + timedelta(hours=step * index),
                    open=close,
                    high=close + Decimal("2"),
                    low=close - Decimal("2"),
                    close=close,
                    volume=Decimal("1"),
                )
            )
    persist_candles(db_session, records)

    result = prepare_multi_timeframe_backtest_data(
        db_session,
        "BTCUSDT",
        start_time=start,
        end_time=start + timedelta(hours=4),
    )

    assert len(result) == 5
    assert result.index[0] == start + timedelta(hours=1)
    assert result.index[-1] == start + timedelta(hours=5)
    assert pd.isna(result.iloc[0]["4h_ema_20"])
    assert result.iloc[3]["4h_open_time"] == start
    assert all(isinstance(value, Decimal) for value in result["close"])
