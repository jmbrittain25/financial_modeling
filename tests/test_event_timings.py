"""Tests for all Timing strategies (OneTime, Interval, Random, Seasonal).

Focus: correctness, edge cases, reproducibility where applicable, and interaction
with the simulation time window.
"""

import datetime as dt

import numpy as np
import pytest
from pydantic import ValidationError

from financial_simulator.core.event import (
    IntervalTiming,
    OneTimeTiming,
    RandomTiming,
    SeasonalTiming,
    create_timing,
)

# --- OneTimeTiming ---


def test_onetime_fires_exactly_once_inside_window():
    t = OneTimeTiming(time=dt.datetime(2026, 6, 15))
    t.reset()
    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 12, 31)

    nt = t.next_time(start, end, {})
    assert nt == dt.datetime(2026, 6, 15)
    t.advance()
    assert t.next_time(nt, end, {}) is None


def test_onetime_before_start_returns_none():
    t = OneTimeTiming(time=dt.datetime(2025, 12, 31))
    t.reset()
    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 12, 31)
    assert t.next_time(start, end, {}) is None


def test_onetime_after_end_returns_none():
    t = OneTimeTiming(time=dt.datetime(2027, 1, 1))
    t.reset()
    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 12, 31)
    assert t.next_time(start, end, {}) is None


def test_onetime_exactly_at_start_and_end():
    t0 = OneTimeTiming(time=dt.datetime(2026, 1, 1))
    t0.reset()
    assert t0.next_time(dt.datetime(2026, 1, 1), dt.datetime(2026, 12, 31), {}) == dt.datetime(
        2026, 1, 1
    )

    t1 = OneTimeTiming(time=dt.datetime(2026, 12, 31))
    t1.reset()
    assert t1.next_time(dt.datetime(2026, 1, 1), dt.datetime(2026, 12, 31), {}) == dt.datetime(
        2026, 12, 31
    )


# --- IntervalTiming ---


def test_interval_with_start_time_generates_correct_sequence():
    t = IntervalTiming(interval=dt.timedelta(days=30), start_time=dt.datetime(2026, 2, 1))
    t.reset()
    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 5, 1)

    times = []
    current = start
    while True:
        nt = t.next_time(current, end, {})
        if nt is None:
            break
        times.append(nt)
        t.advance()
        current = nt + dt.timedelta(seconds=1)

    assert times[0] == dt.datetime(2026, 2, 1)
    assert len(times) >= 3


def test_interval_start_time_none_uses_current_as_first():
    t = IntervalTiming(interval=dt.timedelta(days=7))
    t.reset()
    start = dt.datetime(2026, 3, 5)
    end = dt.datetime(2026, 3, 20)

    nt = t.next_time(start, end, {})
    assert nt == dt.datetime(2026, 3, 12)  # current + 7 days


def test_interval_with_start_in_past_catches_up():
    t = IntervalTiming(interval=dt.timedelta(days=30), start_time=dt.datetime(2020, 1, 1))
    t.reset()
    start = dt.datetime(2026, 3, 1)
    end = dt.datetime(2026, 5, 1)

    nt = t.next_time(start, end, {})
    # Should have fast-forwarded to the first date >= start
    assert nt >= start
    assert nt.month in (3, 4, 5)


def test_interval_zero_duration_rejected_at_validation():
    with pytest.raises(ValidationError, match="strictly positive"):
        IntervalTiming(interval=dt.timedelta(0))


def test_interval_exhausted_returns_none():
    t = IntervalTiming(interval=dt.timedelta(days=1), start_time=dt.datetime(2026, 1, 1))
    t.reset()
    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 1, 1)  # only one day

    nt = t.next_time(start, end, {})
    assert nt == dt.datetime(2026, 1, 1)
    t.advance()
    assert t.next_time(nt, end, {}) is None


# --- RandomTiming ---


def test_random_timing_reproducible_given_rng():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    t1 = RandomTiming(start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 12, 31), n=5)
    t1.reset(rng1)
    times1 = []
    cur = dt.datetime(2026, 1, 1)
    while (nt := t1.next_time(cur, dt.datetime(2026, 12, 31), {})) is not None:
        times1.append(nt)
        t1.advance()
        cur = nt

    t2 = RandomTiming(start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 12, 31), n=5)
    t2.reset(rng2)
    times2 = []
    cur = dt.datetime(2026, 1, 1)
    while (nt := t2.next_time(cur, dt.datetime(2026, 12, 31), {})) is not None:
        times2.append(nt)
        t2.advance()
        cur = nt

    assert times1 == times2
    assert len(times1) == 5


def test_random_timing_unsupported_distribution_rejected_at_construction():
    # The Literal["uniform"] is validated by Pydantic at model construction time
    with pytest.raises(Exception, match="uniform"):
        RandomTiming(
            start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 12, 31), n=3, distribution="weird"
        )


def test_random_timing_n_equals_one():
    t = RandomTiming(start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 1, 2), n=1)
    rng = np.random.default_rng(99)
    t.reset(rng)
    nt = t.next_time(dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2), {})
    assert nt is not None
    t.advance()
    assert t.next_time(nt, dt.datetime(2026, 1, 2), {}) is None


# --- SeasonalTiming ---


def test_seasonal_filters_to_allowed_months():
    inner = IntervalTiming(interval=dt.timedelta(days=15), start_time=dt.datetime(2026, 1, 1))
    st = SeasonalTiming(inner=inner, months=[6, 7, 8])
    st.reset()

    times = []
    cur = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 12, 31)
    while (nt := st.next_time(cur, end, {})) is not None:
        times.append(nt)
        st.advance()
        cur = nt + dt.timedelta(days=1)

    assert all(t.month in (6, 7, 8) for t in times)
    assert len(times) >= 5  # roughly every 15 days in 3 months


def test_seasonal_with_no_overlap_produces_no_events():
    inner = IntervalTiming(interval=dt.timedelta(days=30), start_time=dt.datetime(2026, 1, 1))
    st = SeasonalTiming(inner=inner, months=[12])  # December only, but interval starts Jan
    st.reset()
    times = []
    cur = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 11, 30)  # end before December
    while (nt := st.next_time(cur, end, {})) is not None:
        times.append(nt)
        st.advance()
        cur = nt
    assert times == []


def test_seasonal_wraps_another_seasonal():
    inner = IntervalTiming(interval=dt.timedelta(days=10), start_time=dt.datetime(2026, 1, 1))
    summer = SeasonalTiming(inner=inner, months=[6, 7, 8])
    q3 = SeasonalTiming(inner=summer, months=[7, 8, 9])
    q3.reset()

    times = []
    cur = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 12, 31)
    while (nt := q3.next_time(cur, end, {})) is not None:
        times.append(nt)
        q3.advance()
        cur = nt + dt.timedelta(days=1)

    # Only July and August should survive both filters
    assert all(t.month in (7, 8) for t in times)


# --- create_timing factory (legacy + modern shapes) ---


def test_create_timing_modern_interval():
    data = {
        "type": "Interval",
        "interval": dt.timedelta(days=30),
        "start_time": dt.datetime(2026, 1, 1),
    }
    timing = create_timing(data)
    assert isinstance(timing, IntervalTiming)


def test_create_timing_legacy_interval_days():
    data = {"type": "Interval", "interval_days": 15}
    timing = create_timing(data)
    assert isinstance(timing, IntervalTiming)
    assert timing.interval == dt.timedelta(days=15)


def test_create_timing_nested_seasonal():
    data = {
        "type": "Seasonal",
        "months": [1, 2, 3],
        "inner": {"type": "Interval", "interval_days": 7},
    }
    timing = create_timing(data)
    assert isinstance(timing, SeasonalTiming)
    assert isinstance(timing.inner, IntervalTiming)
