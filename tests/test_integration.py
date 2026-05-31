"""Integration tests: full end-to-end simulations and MC + risk pipelines.

These are marked 'integration' and may be deselected with:
    pytest -m "not integration"
"""

import datetime as dt
import pytest

from financial_simulator.core import (
    SimulationEngine,
    ComposedEventBuilder,
    IntervalTiming,
    FixedValue,
    AppreciationProcess,
)
from financial_simulator.monte_carlo.runner import MonteCarloRunner
from financial_simulator.analytics.risk import MonteCarloAnalyzer


pytestmark = pytest.mark.integration


def test_full_simulation_using_retirement_style_pattern():
    """Replicate the spirit of examples/retirement.py with a 5-year slice."""
    start = dt.datetime(2026, 1, 1)
    end = dt.datetime(2031, 1, 1)

    eng = SimulationEngine(
        name="retirement-slice",
        start=start,
        end=end,
        initial_state={"portfolio": 100_000.0, "cumulative_cash": 0.0},
        seed=42,
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=dt.timedelta(days=30)),
            value_gen=FixedValue(value=-800.0),  # contribution
            metadata={"type": "contribution"},
        )
    )
    eng.add_continuous_process(AppreciationProcess(rate=0.065, var="portfolio"))

    eng.run()
    result = eng.get_result()

    assert len(result.events) > 50
    assert result.final_state["portfolio"] > 100_000  # growth should dominate contributions


def test_end_to_end_monte_carlo_plus_risk_analysis():
    """Run a small MC batch and feed results into MonteCarloAnalyzer."""
    def factory(i: int) -> SimulationEngine:
        eng = SimulationEngine(
            name=f"MC-Int-{i}",
            start=dt.datetime(2026, 1, 1),
            end=dt.datetime(2028, 1, 1),
            initial_state={"cumulative_cash": 0.0},
            seed=1000 + i,
        )
        eng.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=dt.timedelta(days=30)),
                value_gen=FixedValue(value=2000.0 if i % 2 == 0 else 1800.0),
            )
        )
        return eng

    runner = MonteCarloRunner(n_jobs=2)
    results = runner.run(12, factory, base_seed=999)

    mca = MonteCarloAnalyzer(risk_free_rate=0.02)
    report = mca.analyze_results(results)

    assert report.n_simulations == 12
    assert "mean" in report.metrics
    assert "sharpe" in report.metrics  # because we have >1 result
    assert report.metrics["prob_ruin"] == 0.0  # all positive outcomes