"""Tests for RiskAnalyzer."""

import numpy as np
from financial_simulator.analytics.risk import RiskAnalyzer


def test_var_cvar_simple():
    rng = np.random.default_rng(42)
    # Simple normal distribution of outcomes
    outcomes = rng.normal(100_000, 15_000, size=5000)

    analyzer = RiskAnalyzer()
    var = analyzer.compute_var(outcomes, 0.95)
    cvar = analyzer.compute_cvar(outcomes, 0.95)

    assert var < 80_000
    assert cvar < var  # CVaR should be worse than VaR


def test_sharpe_ratio():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.08, 0.15, 1000)  # 8% mean, 15% vol

    analyzer = RiskAnalyzer(risk_free_rate=0.02)
    sharpe = analyzer.compute_sharpe(returns)

    # Rough expected Sharpe around (0.08-0.02)/0.15 ≈ 0.4
    assert 0.2 < sharpe < 0.7
