"""
Tests for the pure (non-Streamlit) parts of the External Driver UI component.

We deliberately avoid importing the full render function (which pulls in Streamlit)
so these tests remain runnable in minimal CI environments.
"""

from datetime import datetime

import pytest

try:
    import plotly  # noqa: F401
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from app.components.driver_viz import get_driver_stats, plot_driver_preview
from financial_simulator.scenarios.drivers import (
    make_inflation_driver,
    make_interest_rate_driver,
    make_stock_market_driver,
)


def test_get_driver_stats_returns_expected_keys_and_reasonable_values():
    d = make_stock_market_driver(initial_value=100_000.0)
    stats = get_driver_stats(d, n_paths=50, seed=42)

    for key in ("mean", "std", "min", "max", "p5", "p95"):
        assert key in stats
        assert isinstance(stats[key], float)

    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["p5"] <= stats["mean"] <= stats["p95"]
    # With 50 paths of a volatile GBM we should have visible spread
    assert stats["std"] > 1000


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed in this environment")
def test_plot_driver_preview_does_not_crash_and_returns_figure():
    d = make_interest_rate_driver()
    fig = plot_driver_preview(d, n_paths=4, seed=123, height=300)
    # plotly Figure should have data traces
    assert hasattr(fig, "data")
    assert len(fig.data) >= 1  # at least the mean path
    assert fig.layout.height == 300


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed in this environment")
def test_plot_driver_preview_mean_path_is_somewhere_in_the_fan():
    """The bold mean line should be between the min and max of the sampled paths at every point."""
    d = make_inflation_driver(initial_value=1.0)
    fig = plot_driver_preview(d, start=datetime(2026, 1, 1), end=datetime(2026, 6, 1), n_paths=6, seed=7)

    # Find the mean trace (we name it "Mean path")
    mean_y = None
    all_ys = []
    for trace in fig.data:
        if trace.name == "Mean path":
            mean_y = list(trace.y)
        else:
            all_ys.append(list(trace.y))

    assert mean_y is not None
    assert len(all_ys) > 0

    for i in range(len(mean_y)):
        column = [ys[i] for ys in all_ys]
        lo, hi = min(column), max(column)
        # mean should be inside the envelope (allowing tiny floating point tolerance)
        assert lo - 1e-9 <= mean_y[i] <= hi + 1e-9


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed in this environment")
def test_driver_viz_helpers_work_with_all_driver_types():
    """Smoke test that the viz helpers don't blow up on any driver kind."""
    drivers = [
        make_interest_rate_driver(),
        make_inflation_driver(),
        make_stock_market_driver(initial_value=50.0),
    ]
    for d in drivers:
        stats = get_driver_stats(d, n_paths=8, seed=1)
        assert "mean" in stats
        fig = plot_driver_preview(d, n_paths=3, seed=1)
        assert len(fig.data) >= 1
