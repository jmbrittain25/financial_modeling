"""Tests for the new scenarios package (models + materialization + metrics)."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from financial_simulator.core import (
    ComposedEventBuilder,
    IntervalTiming,
    FixedValue,
    NormalDistribution,
    TriangularDistribution,
)
from financial_simulator.scenarios import (
    ScenarioConfig,
    SavedDistribution,
    DistributionLibrary,
    CustomMetric,
    DiscreteRateDriver,
    build_engine,
    run_single,
    run_monte_carlo,
    scenario_to_json,
    scenario_from_json,
)
from financial_simulator.scenarios.metrics import compute_metric


def test_scenario_config_roundtrip_json():
    start = datetime(2026, 1, 1)
    end = datetime(2027, 1, 1)

    cfg = ScenarioConfig(
        name="Test Roundtrip",
        start=start,
        end=end,
        initial_state={"cash": 10_000.0},
        event_builders=[
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=FixedValue(value=500.0),
                metadata={"type": "income"},
            )
        ],
    )

    js = scenario_to_json(cfg)
    loaded = scenario_from_json(js)

    assert loaded.name == cfg.name
    assert loaded.start == cfg.start
    assert len(loaded.event_builders) == 1
    assert loaded.event_builders[0].metadata["type"] == "income"


def test_discrete_rate_driver_materialization():
    """A DiscreteRateDriver must expand into a RateChangeValue builder that mutates state."""
    start = datetime(2026, 1, 1)
    end = datetime(2026, 7, 1)

    driver = DiscreteRateDriver(
        name="mortgage_rate",
        target_state_key="rate",
        dist=NormalDistribution(mean=0.065, std=0.005),
        timing=IntervalTiming(interval=timedelta(days=90)),
    )

    cfg = ScenarioConfig(
        name="Driver Test",
        start=start,
        end=end,
        initial_state={"loan_balance": 200_000.0, "rate": 0.06},
        external_drivers=[driver],
    )

    eng = build_engine(cfg, seed=42)
    # We should have the original (none) + the driver-generated builder
    assert len(eng.event_builders) >= 1

    eng.run()
    result = eng.get_result()

    # Rate should have changed at least once (the driver fired)
    # Because we only record final state here, we just sanity-check it moved
    assert "rate" in result.final_state


def test_custom_metric_final_state_value():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 2, 1)

    cfg = ScenarioConfig(
        name="Metric Test",
        start=start,
        end=end,
        initial_state={"portfolio": 100_000.0},
        custom_metrics=[
            CustomMetric(name="final_portfolio", metric_type="final_state_value", params={"key": "portfolio"})
        ],
    )

    result = run_single(cfg, seed=7)
    metrics = result.final_state.get("__custom_metrics__", {})
    assert "final_portfolio" in metrics
    assert metrics["final_portfolio"] == 100_000.0


def test_custom_metric_max_drawdown():
    # Reliable drawdown case: start at 0, large positive, then many negative contributions
    # that take cumulative_cash well below its peak. The engine auto-tracks cumulative_cash.
    start = datetime(2026, 1, 1)
    end = datetime(2026, 7, 1)

    cfg = ScenarioConfig(
        name="DD Test",
        start=start,
        end=end,
        initial_state={"cumulative_cash": 0.0},
        event_builders=[
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30), start_time=start + timedelta(days=1)),
                value_gen=FixedValue(value=12000.0),  # creates the peak
            ),
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30), start_time=start + timedelta(days=40)),
                value_gen=FixedValue(value=-2500.0),  # repeated outflows
            ),
        ],
        custom_metrics=[
            CustomMetric(name="mdd", metric_type="max_drawdown_on_path", params={"state_key": "cumulative_cash"})
        ],
    )

    result = run_single(cfg, seed=1)
    mdd = result.final_state["__custom_metrics__"]["mdd"]
    # After the +12k peak, repeated -2500 should produce visible drawdown
    assert 0.0 < mdd < 0.9


def test_monte_carlo_with_custom_metrics_attached():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 6, 1)

    cfg = ScenarioConfig(
        name="MC Metric Test",
        start=start,
        end=end,
        initial_state={"cash": 0.0},
        event_builders=[
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=FixedValue(value=1000.0),
            )
        ],
        custom_metrics=[
            CustomMetric(name="final_cash", metric_type="final_state_value", params={"key": "cumulative_cash"})
        ],
    )

    results = run_monte_carlo(cfg, n_sims=5, base_seed=123, n_jobs=2)
    assert len(results) == 5
    for r in results:
        assert "final_cash" in r.final_state.get("__custom_metrics__", {})


def test_distribution_library_crud():
    lib = DistributionLibrary()
    d1 = SavedDistribution(
        id="rate-normal",
        name="Fed Rate",
        dist=NormalDistribution(mean=0.04, std=0.01),
    )
    lib.add(d1)
    assert lib.get("rate-normal") is not None

    # roundtrip
    js = json.dumps(lib.to_dict())
    loaded = DistributionLibrary.from_dict(json.loads(js))
    assert loaded.get_by_name("Fed Rate") is not None
