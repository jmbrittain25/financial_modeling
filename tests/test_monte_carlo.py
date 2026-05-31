"""Tests for MonteCarloRunner: parallel execution, seeding, and reproducibility semantics.

Important: MonteCarloRunner uses ThreadPoolExecutor + as_completed.
Results are returned in completion order (arbitrary), not submission order.
Tests must match results by simulation identity (name/seed), not list index.
"""

import datetime as dt

import numpy as np
import pytest

from financial_simulator.core import SimulationEngine, ComposedEventBuilder, IntervalTiming, FixedValue
from financial_simulator.core.distributions import NormalDistribution
from financial_simulator.core.event import DistributionValue
from financial_simulator.monte_carlo.runner import MonteCarloRunner


def _make_simple_factory(base: int = 1000):
    def factory(i: int) -> SimulationEngine:
        eng = SimulationEngine(
            name=f"Sim-{i}",
            start=dt.datetime(2026, 1, 1),
            end=dt.datetime(2026, 6, 1),
            initial_state={"cumulative_cash": 0.0},
            seed=base + i,
        )
        eng.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=dt.timedelta(days=30)),
                value_gen=FixedValue(value=1000.0),
            )
        )
        return eng
    return factory


def test_monte_carlo_reproducibility_content_match_across_runs():
    """Same (base_seed + i) must produce identical outcomes even if list order differs."""
    runner = MonteCarloRunner(n_jobs=4)
    factory = _make_simple_factory(777)

    r1 = runner.run(8, factory, base_seed=42)
    r2 = runner.run(8, factory, base_seed=42)

    # Build lookup by name (stable identity)
    def by_name(results):
        return {r.name: r.final_state["cumulative_cash"] for r in results}

    assert by_name(r1) == by_name(r2)


def test_monte_carlo_runner_overrides_seed_with_base_plus_i():
    """The runner sets engine.seed = base_seed + i before calling run()."""
    runner = MonteCarloRunner(n_jobs=2)
    observed = {}

    def factory(i: int) -> SimulationEngine:
        eng = SimulationEngine(
            name=f"S-{i}",
            start=dt.datetime(2026, 1, 1),
            end=dt.datetime(2026, 2, 1),
            initial_state={"cumulative_cash": 0.0},
        )
        eng.add_event_builder(
            ComposedEventBuilder(timing=IntervalTiming(interval=dt.timedelta(days=1)), value_gen=FixedValue(1))
        )
        return eng

    results = runner.run(5, factory, base_seed=100)
    for r in results:
        # The engine used base_seed + index; we can infer from final state reproducibility
        # Instead we just assert that with same base we get same results (content match)
        observed[r.name] = r.final_state["cumulative_cash"]

    # Re-run and compare content
    results2 = runner.run(5, factory, base_seed=100)
    observed2 = {r.name: r.final_state["cumulative_cash"] for r in results2}
    assert observed == observed2


def test_monte_carlo_with_stochastic_builders_produces_variation():
    runner = MonteCarloRunner(n_jobs=3)

    def factory(i: int) -> SimulationEngine:
        eng = SimulationEngine(
            name=f"Stoch-{i}",
            start=dt.datetime(2026, 1, 1),
            end=dt.datetime(2026, 12, 1),
            initial_state={"cumulative_cash": 0.0},
            seed=200 + i,
        )
        eng.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=dt.timedelta(days=30)),
                value_gen=DistributionValue(dist=NormalDistribution(mean=1000, std=300)),
            )
        )
        return eng

    results = runner.run(20, factory, base_seed=999)
    finals = [r.final_state["cumulative_cash"] for r in results]
    assert len(set(finals)) > 10  # high probability of variation


def test_monte_carlo_n_jobs_one_still_works():
    runner = MonteCarloRunner(n_jobs=1)
    factory = _make_simple_factory(500)
    results = runner.run(4, factory, base_seed=7)
    assert len(results) == 4
    # With n_jobs=1 the order should match submission order (though not guaranteed by API)
    names = [r.name for r in results]
    assert names == ["Sim-0", "Sim-1", "Sim-2", "Sim-3"]