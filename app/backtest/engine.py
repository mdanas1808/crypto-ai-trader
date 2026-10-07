"""Causal walk-forward preparation for indicator-enriched candle data."""

from collections.abc import Iterator

import pandas as pd


def iter_walk_forward(indicator_frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield independent chronological prefixes, one current candle per step.

    At step ``i`` the returned frame contains rows ``0`` through ``i`` only.
    The input must already be in strictly increasing ``open_time`` order; it is
    validated rather than sorted. Each yielded prefix is a copy, so consumers
    cannot mutate the original frame through a yielded value.
    """
    if not isinstance(indicator_frame, pd.DataFrame):
        raise TypeError("indicator_frame must be a pandas DataFrame")
    if "open_time" not in indicator_frame.columns:
        raise ValueError("indicator_frame must contain an open_time column")

    open_times = indicator_frame["open_time"]
    if open_times.isna().any() or not open_times.is_monotonic_increasing or not open_times.is_unique:
        raise ValueError("indicator_frame must be strictly chronological by open_time")

    for stop in range(1, len(indicator_frame) + 1):
        yield indicator_frame.iloc[:stop].copy(deep=True)
