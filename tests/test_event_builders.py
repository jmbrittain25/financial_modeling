"""Tests for ComposedEventBuilder and the create_* factory functions."""

import datetime as dt

import numpy as np
import pytest

from financial_simulator.core.event import (
    ComposedEventBuilder,
    IntervalTiming,
    OneTimeTiming,
    FixedValue,
    DistributionValue,
    create_event_builder,
    create_timing,
    create_value_generator,
)
from financial_simulator.core.distributions import NormalDistribution


def test_composed_builder_produces_events_in_order():
    builder = ComposedEventBuilder(
        timing=IntervalTiming(interval=dt.timedelta(days=30), start_time=dt.datetime(2026, 1, 1)),
        value_gen=FixedValue(value=100.0),
        metadata={"category": "rent"},
    )
    builder.reset()

    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 4, 1)

    events = []
    current = start
    while True:
        nt = builder.next_event_time(current, end, {})
        if nt is None:
            break
        ev = builder.generate_event(nt, {}, None)
        if ev:
            events.append(ev)
        current = nt + dt.timedelta(seconds=1)

    assert len(events) >= 3
    assert all(e.metadata.get("category") == "rent" for e in events)
    assert all(e.value == 100.0 for e in events)


def test_composed_builder_skips_stale_times():
    """If current time jumps past several scheduled times, builder should catch up."""
    builder = ComposedEventBuilder(
        timing=IntervalTiming(interval=dt.timedelta(days=1), start_time=dt.datetime(2026, 1, 1)),
        value_gen=FixedValue(value=1.0),
    )
    builder.reset()

    # Jump straight to March
    nt = builder.next_event_time(dt.datetime(2026, 3, 1), dt.datetime(2026, 3, 10), {})
    assert nt is not None
    assert nt.month == 3


def test_composed_builder_with_distribution_is_reproducible():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    b1 = ComposedEventBuilder(
        timing=OneTimeTiming(time=dt.datetime(2026, 6, 1)),
        value_gen=DistributionValue(dist=NormalDistribution(mean=0, std=100)),
    )
    b1.reset(rng1)
    ev1 = b1.generate_event(dt.datetime(2026, 6, 1), {}, rng1)

    b2 = ComposedEventBuilder(
        timing=OneTimeTiming(time=dt.datetime(2026, 6, 1)),
        value_gen=DistributionValue(dist=NormalDistribution(mean=0, std=100)),
    )
    b2.reset(rng2)
    ev2 = b2.generate_event(dt.datetime(2026, 6, 1), {}, rng2)

    assert ev1 is not None and ev2 is not None
    assert ev1.value == ev2.value


# --- create_value_generator factory ---

def test_create_value_generator_fixed():
    vg = create_value_generator({"type": "Fixed", "value": 999})
    assert isinstance(vg, FixedValue)
    assert vg.value == 999


def test_create_value_generator_distribution_nested():
    data = {"type": "Distribution", "dist": {"type": "normal", "mean": 10, "std": 1}}
    vg = create_value_generator(data)
    assert isinstance(vg, DistributionValue)
    assert isinstance(vg.dist, NormalDistribution)


# --- create_event_builder factory ---

def test_create_event_builder_full_shape():
    data = {
        "name": "test_rent",
        "timing": {"type": "Interval", "interval_days": 30, "start_time": "2026-02-01T00:00:00"},
        "value_gen": {"type": "Fixed", "value": 2200.0},
        "metadata": {"category": "income"},
    }
    builder = create_event_builder(data)
    assert isinstance(builder, ComposedEventBuilder)
    assert builder.name == "test_rent"
    assert builder.metadata["category"] == "income"


def test_create_event_builder_minimal():
    data = {
        "timing": {"type": "OneTime", "time": "2026-07-15T00:00:00"},
        "value_gen": {"type": "Fixed", "value": -500},
    }
    builder = create_event_builder(data)
    assert isinstance(builder, ComposedEventBuilder)