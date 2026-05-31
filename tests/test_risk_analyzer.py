"""Tests for RiskAnalyzer and MonteCarloAnalyzer, including edge cases."""

import datetime as dt

import numpy as np
import pytest

from financial_simulator.analytics.risk import MonteCarloAnalyzer, RiskAnalyzer, RiskReport
from financial_simulator.core import (
    ComposedEventBuilder,
    FixedValue,
    IntervalTiming,
    SimulationEngine,
)


def test_var_cvar_simple():
    rng = np.random.default_rng(42)
    outcomes = rng.normal(100_000, 15_000, size=5000)

    analyzer = RiskAnalyzer()
    var = analyzer.compute_var(outcomes, 0.95)
    cvar = analyzer.compute_cvar(outcomes, 0.95)

    assert var < 80_000
    assert cvar < var


def test_sharpe_ratio():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.08, 0.15, 1000)

    analyzer = RiskAnalyzer(risk_free_rate=0.02)
    sharpe = analyzer.compute_sharpe(returns)

    assert 0.2 < sharpe < 0.7


def test_sortino_ratio_downside_only():
    rng = np.random.default_rng(7)
    # Mix of positive and negative excess returns
    excess = rng.normal(0.01, 0.08, 2000)
    analyzer = RiskAnalyzer()
    sortino = analyzer.compute_sortino(excess, target=0.0)
    assert sortino > 0


# --- Edge cases ---


def test_var_cvar_single_element():
    analyzer = RiskAnalyzer()
    out = np.array([42.0])
    assert analyzer.compute_var(out) == 42.0
    assert analyzer.compute_cvar(out) == 42.0


def test_var_cvar_empty_returns_nan():
    """Per project decision: make robust rather than crash."""
    analyzer = RiskAnalyzer()
    out = np.array([], dtype=float)
    var = analyzer.compute_var(out)
    cvar = analyzer.compute_cvar(out)
    assert np.isnan(var)
    assert np.isnan(cvar)


def test_sharpe_zero_std_returns_zero():
    analyzer = RiskAnalyzer(risk_free_rate=0.0)
    returns = np.array([0.05, 0.05, 0.05])
    assert analyzer.compute_sharpe(returns) == 0.0


def test_sortino_zero_downside_returns_zero():
    analyzer = RiskAnalyzer()
    returns = np.array([0.1, 0.2, 0.3])
    assert analyzer.compute_sortino(returns) == 0.0


def test_max_drawdown_monotonic_path_is_zero():
    analyzer = RiskAnalyzer()
    path = np.array([100, 110, 120, 130])
    assert analyzer.max_drawdown(path) == 0.0


def test_max_drawdown_typical_path():
    analyzer = RiskAnalyzer()
    path = np.array([100, 90, 95, 80, 85, 70])
    mdd = analyzer.max_drawdown(path)
    assert 0.25 < mdd <= 0.3


def test_probability_of_ruin():
    analyzer = RiskAnalyzer()
    outcomes = np.array([-10, 5, 20, -3, 0, 100])
    assert analyzer.probability_of_ruin(outcomes, 0.0) == 0.5


# --- MonteCarloAnalyzer ---


def test_monte_carlo_analyzer_basic_flow():
    # Build a tiny set of deterministic results
    def make_eng(i):
        eng = SimulationEngine(
            name=f"R{i}",
            start=dt.datetime(2026, 1, 1),
            end=dt.datetime(2026, 4, 1),
            initial_state={"cumulative_cash": 0.0},
            seed=10 + i,
        )
        eng.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=dt.timedelta(days=30)),
                value_gen=FixedValue(value=1000.0),
            )
        )
        return eng

    from financial_simulator.monte_carlo.runner import MonteCarloRunner

    runner = MonteCarloRunner(n_jobs=2)
    results = runner.run(5, make_eng, base_seed=1)

    mca = MonteCarloAnalyzer(risk_free_rate=0.0)
    report = mca.analyze_results(results)
    assert isinstance(report, RiskReport)
    assert report.n_simulations == 5
    assert "var_95" in report.metrics
    assert "prob_ruin" in report.metrics


def test_monte_carlo_analyzer_empty_results_raises():
    mca = MonteCarloAnalyzer()
    with pytest.raises(ValueError, match="No simulation results"):
        mca.analyze_results([])
