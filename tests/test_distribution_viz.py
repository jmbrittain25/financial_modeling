"""Lightweight tests for the distribution visualization helpers (non-Streamlit parts)."""

import numpy as np

from app.components.distribution_viz import (
    get_analytical_pdf,
    get_distribution_stats,
    plot_distribution_preview,
)
from financial_simulator.core.distributions import (
    BetaDistribution,
    ConstantDistribution,
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    TriangularDistribution,
    UniformDistribution,
)


def test_analytical_pdfs_produce_reasonable_output():
    x = np.linspace(-5, 15, 200)

    d = NormalDistribution(mean=5, std=2)
    pdf = get_analytical_pdf(d, x)
    assert pdf is not None
    assert np.all(pdf >= 0)
    assert np.max(pdf) > 0.1

    d = UniformDistribution(low=0, high=10)
    pdf = get_analytical_pdf(d, x)
    assert pdf is not None
    assert np.allclose(np.sum(pdf) * (x[1] - x[0]), 1.0, atol=0.02)  # roughly integrates to 1

    d = TriangularDistribution(low=0, mode=5, high=10)
    pdf = get_analytical_pdf(d, x)
    assert pdf is not None
    assert np.max(pdf) > 0

    d = ExponentialDistribution(rate=0.5)
    pdf = get_analytical_pdf(d, np.linspace(0, 20, 200))
    assert pdf is not None

    d = LogNormalDistribution(mean=0, sigma=0.8)
    pdf = get_analytical_pdf(d, np.linspace(0.01, 10, 200))
    assert pdf is not None

    d = BetaDistribution(alpha=2, beta=5)
    pdf = get_analytical_pdf(d, np.linspace(0, 1, 200))
    assert pdf is not None


def test_constant_returns_none_for_pdf():
    d = ConstantDistribution(value=42)
    x = np.linspace(0, 100, 50)
    assert get_analytical_pdf(d, x) is None


def test_plot_and_stats_run_without_error():
    d = NormalDistribution(mean=100, std=15)

    # plot function requires plotly (only available in the Streamlit runtime)
    try:
        fig = plot_distribution_preview(d, n_samples=2000)
        assert len(fig.data) >= 1
    except ModuleNotFoundError:
        pass  # acceptable in test env without plotly

    stats = get_distribution_stats(d, n_samples=3000)
    assert "mean" in stats
    assert stats["std"] > 0
    assert stats["p5"] < stats["median"] < stats["p95"]


def test_streamlit_app_imports_without_legacy_crash():
    """The main Streamlit app module must be importable.
    We tolerate Streamlit singleton errors (common when multiple test files
    import the app during collection) and missing optional deps.
    """
    try:
        import app.streamlit_app  # noqa: F401
    except (NameError, RuntimeError) as e:
        if "DeltaGeneratorSingleton" in str(e) or "singleton" in str(e).lower():
            # Expected when other tests have already initialized Streamlit in the same process
            pass
        else:
            raise AssertionError(f"Streamlit app import error: {e}") from e
    except ModuleNotFoundError:
        # Acceptable in minimal test envs
        pass
