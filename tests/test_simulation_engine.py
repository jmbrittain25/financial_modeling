"""Tests for SimulationEngine and MonteCarloRunner."""

import pytest
from datetime import datetime, timedelta
import numpy as np

from financial_simulator.core import SimulationEngine, ComposedEventBuilder
from financial_simulator.core.event import IntervalTiming, FixedValue
from financial_simulator.monte_carlo.runner import MonteCarloRunner


def test_basic_engine_run():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 4, 1)

    engine = SimulationEngine(
        name="Test",
        start=start,
        end=end,
        initial_state={"cumulative_cash": 0.0},
    )

    engine.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=timedelta(days=30), start_time=start),
            value_gen=FixedValue(value=1000.0),
        )
    )

    engine.run()
    result = engine.get_result()

    assert len(result.events) >= 2
    assert result.final_state["cumulative_cash"] > 0


def test_engine_reproducibility():
    def make_engine(seed):
        eng = SimulationEngine(
            name="Repro",
            start=datetime(2026, 1, 1),
            end=datetime(2026, 7, 1),
            initial_state={"cumulative_cash": 0.0},
            seed=seed,
        )
        eng.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=FixedValue(value=500.0),
            )
        )
        return eng

    e1 = make_engine(123)
    e1.run()
    r1 = e1.get_result()

    e2 = make_engine(123)
    e2.run()
    r2 = e2.get_result()

    assert r1.final_state["cumulative_cash"] == r2.final_state["cumulative_cash"]


def test_monte_carlo_runner_variation():
    def factory(i):
        eng = SimulationEngine(
            name=f"MC-{i}",
            start=datetime(2026, 1, 1),
            end=datetime(2026, 6, 1),
            initial_state={"cumulative_cash": 0.0},
            seed=100 + i,  # give each sim its own seed
        )
        # Add some randomness so different seeds produce different results
        from financial_simulator.core import DistributionValue, NormalDistribution
        eng.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=DistributionValue(dist=NormalDistribution(mean=1000, std=200)),
            )
        )
        return eng

    runner = MonteCarloRunner(n_jobs=2)
    results = runner.run(6, factory, base_seed=99)

    finals = [r.final_state["cumulative_cash"] for r in results]
    assert len(set(finals)) > 1  # randomness should create variation
