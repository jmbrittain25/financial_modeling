"""Pytest configuration and shared fixtures."""

import datetime as dt
from collections.abc import Callable

import numpy as np
import pytest

from financial_simulator.analytics.risk import RiskReport
from financial_simulator.core import (
    AppreciationProcess,
    ComposedEventBuilder,
    DistributionValue,
    FixedValue,
    IntervalTiming,
    NormalDistribution,
    OneTimeTiming,
    SimulationEngine,
)


@pytest.fixture
def seeded_rng():
    """Provide a reproducible random number generator (seed=42)."""
    return np.random.default_rng(42)


@pytest.fixture
def rng_factory() -> Callable[[int], np.random.Generator]:
    """Factory for creating independent RNGs from integer seeds."""

    def _make(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    return _make


# --- Date fixtures ---
@pytest.fixture
def start_2026() -> dt.datetime:
    return dt.datetime(2026, 1, 1)


@pytest.fixture
def end_2026() -> dt.datetime:
    return dt.datetime(2026, 12, 31)


@pytest.fixture
def end_2026_mid() -> dt.datetime:
    return dt.datetime(2026, 6, 30)


# --- Simple deterministic builders ---
@pytest.fixture
def fixed_income_builder(start_2026: dt.datetime) -> ComposedEventBuilder:
    """Monthly +1000 income starting Feb 2026."""
    return ComposedEventBuilder(
        timing=IntervalTiming(interval=dt.timedelta(days=30), start_time=dt.datetime(2026, 2, 1)),
        value_gen=FixedValue(value=1000.0),
        metadata={"type": "income"},
    )


@pytest.fixture
def fixed_expense_builder(start_2026: dt.datetime) -> ComposedEventBuilder:
    """Monthly -400 expense starting Jan 15 2026."""
    return ComposedEventBuilder(
        timing=IntervalTiming(interval=dt.timedelta(days=30), start_time=dt.datetime(2026, 1, 15)),
        value_gen=FixedValue(value=-400.0),
        metadata={"type": "expense"},
    )


@pytest.fixture
def one_time_builder() -> ComposedEventBuilder:
    """One-time +5000 on 2026-07-01."""
    return ComposedEventBuilder(
        timing=OneTimeTiming(time=dt.datetime(2026, 7, 1)),
        value_gen=FixedValue(value=5000.0),
        metadata={"type": "bonus"},
    )


# --- Engine fixtures ---
@pytest.fixture
def simple_engine(
    start_2026: dt.datetime,
    end_2026_mid: dt.datetime,
    fixed_income_builder: ComposedEventBuilder,
) -> SimulationEngine:
    """Basic engine with one income stream and cumulative_cash tracking."""
    eng = SimulationEngine(
        name="simple-fixture",
        start=start_2026,
        end=end_2026_mid,
        initial_state={"cumulative_cash": 0.0},
        seed=123,
    )
    eng.add_event_builder(fixed_income_builder)
    return eng


@pytest.fixture
def engine_with_continuous(
    start_2026: dt.datetime,
    end_2026: dt.datetime,
) -> SimulationEngine:
    """Engine with monthly contribution + continuous appreciation."""
    eng = SimulationEngine(
        name="appreciation-fixture",
        start=start_2026,
        end=end_2026,
        initial_state={"portfolio": 100_000.0, "cumulative_cash": 0.0},
        seed=456,
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=dt.timedelta(days=30)),
            value_gen=FixedValue(value=-2000.0),  # contribution (outflow)
            metadata={"type": "contribution"},
        )
    )
    eng.add_continuous_process(AppreciationProcess(rate=0.07, var="portfolio"))
    return eng


# --- Stochastic builder factory ---
@pytest.fixture
def stochastic_builder_factory() -> Callable[[float, float], ComposedEventBuilder]:
    """Returns a factory that creates a monthly DistributionValue builder."""

    def _make(mean: float, std: float) -> ComposedEventBuilder:
        return ComposedEventBuilder(
            timing=IntervalTiming(interval=dt.timedelta(days=30)),
            value_gen=DistributionValue(dist=NormalDistribution(mean=mean, std=std)),
            metadata={"type": "stochastic"},
        )

    return _make


# --- Sample RiskReport for analyzer tests ---
@pytest.fixture
def sample_risk_report() -> RiskReport:
    return RiskReport(
        n_simulations=1000,
        metrics={
            "var_95": -12345.0,
            "cvar_95": -15678.0,
            "mean": 23456.0,
            "std": 34567.0,
            "median": 21000.0,
            "p5": -8000.0,
            "p95": 78000.0,
            "sharpe": 0.42,
            "sortino": 0.61,
            "prob_ruin": 0.03,
        },
    )


# --- Mark integration tests as slow ---
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as full end-to-end simulations (select with '-m integration')",
    )
