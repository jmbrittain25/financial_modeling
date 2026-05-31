"""Tests for ContinuousProcess reproducibility (the RNG wiring fix)."""

import numpy as np
from datetime import datetime, timedelta

from financial_simulator.core import (
    SimulationEngine,
    GBMContinuousProcess,
    MeanRevertingContinuousProcess,
    GeometricBrownianMotion,
    MeanRevertingProcess,
)
from financial_simulator.core.simulation import AppreciationProcess


def test_gbm_continuous_reproducible_with_seed():
    """Two engines with same seed + GBMContinuousProcess must produce identical paths."""
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 10)

    def make_engine(seed: int):
        eng = SimulationEngine(
            name="gbm-test",
            start=start,
            end=end,
            initial_state={"portfolio": 100.0},
            seed=seed,
        )
        eng.add_continuous_process(
            GBMContinuousProcess(
                process=GeometricBrownianMotion(drift=0.08, volatility=0.20),
                var="portfolio",
            )
        )
        return eng

    e1 = make_engine(12345)
    e1.run()
    r1 = e1.get_result()

    e2 = make_engine(12345)
    e2.run()
    r2 = e2.get_result()

    # Final values must match exactly (reproducibility)
    assert r1.final_state["portfolio"] == r2.final_state["portfolio"]

    # History must be identical at every snapshot
    for t in r1.state_history:
        assert r1.state_history[t]["portfolio"] == r2.state_history[t]["portfolio"]


def test_mean_reverting_continuous_reproducible():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 4, 1)

    def make_engine(seed: int):
        eng = SimulationEngine(
            name="mr-test",
            start=start,
            end=end,
            initial_state={"rate": 0.05},
            seed=seed,
        )
        eng.add_continuous_process(
            MeanRevertingContinuousProcess(
                process=MeanRevertingProcess(long_term_mean=0.04, speed=1.5, volatility=0.01),
                var="rate",
            )
        )
        return eng

    e1 = make_engine(42)
    e1.run()
    r1 = e1.get_result()

    e2 = make_engine(42)
    e2.run()
    r2 = e2.get_result()

    assert abs(r1.final_state["rate"] - r2.final_state["rate"]) < 1e-12


def test_appreciation_unaffected_by_rng_wiring():
    """AppreciationProcess (deterministic) must produce identical results regardless of the RNG change.

    NOTE: Continuous processes only advance when there are discrete events driving the simulation clock.
    We add a dummy monthly event so the full year elapses.
    """
    from financial_simulator.core.event import ComposedEventBuilder, IntervalTiming, FixedValue

    start = datetime(2026, 1, 1)
    end = datetime(2027, 1, 1)

    eng = SimulationEngine(
        name="app-test",
        start=start,
        end=end,
        initial_state={"house": 400_000.0, "cumulative_cash": 0.0},
        seed=999,
    )
    eng.add_continuous_process(AppreciationProcess(rate=0.03, var="house"))
    # Dummy monthly events to advance the global clock across the full year
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=timedelta(days=30), start_time=start),
            value_gen=FixedValue(value=0.0),
        )
    )
    eng.run()
    result = eng.get_result()

    # Pure geometric growth formula check (engine ran ~1 year with monthly ticks)
    expected = 400_000.0 * (1.03 ** 1.0)
    # Monthly stepping means the final delta is not exactly 1.0 year; allow small relative error
    assert abs(result.final_state["house"] - expected) < 200.0
