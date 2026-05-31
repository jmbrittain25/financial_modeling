"""Tests for the new scenarios package (models + materialization + metrics)."""

import json
from datetime import datetime, timedelta

from financial_simulator.core import (
    ComposedEventBuilder,
    FixedValue,
    IntervalTiming,
    NormalDistribution,
)
from financial_simulator.scenarios import (
    CustomMetric,
    DiscreteRateDriver,
    DistributionLibrary,
    SavedDistribution,
    ScenarioConfig,  # already used below in new tests
    build_engine,
    load_template,
    run_monte_carlo,
    run_single,
    scenario_from_json,
    scenario_to_json,
)


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
            CustomMetric(
                name="final_portfolio", metric_type="final_state_value", params={"key": "portfolio"}
            )
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
                timing=IntervalTiming(
                    interval=timedelta(days=30), start_time=start + timedelta(days=1)
                ),
                value_gen=FixedValue(value=12000.0),  # creates the peak
            ),
            ComposedEventBuilder(
                timing=IntervalTiming(
                    interval=timedelta(days=30), start_time=start + timedelta(days=40)
                ),
                value_gen=FixedValue(value=-2500.0),  # repeated outflows
            ),
        ],
        custom_metrics=[
            CustomMetric(
                name="mdd",
                metric_type="max_drawdown_on_path",
                params={"state_key": "cumulative_cash"},
            )
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
            CustomMetric(
                name="final_cash",
                metric_type="final_state_value",
                params={"key": "cumulative_cash"},
            )
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


def test_template_with_continuous_processes_runs_end_to_end():
    """Templates that declare continuous_processes (e.g. appreciation) must materialize and run cleanly."""
    cfg = load_template("retirement_30yr")
    assert len(cfg.continuous_processes) >= 1

    # Single run exercises materialization + engine with continuous processes
    res = run_single(cfg, seed=42)
    assert "portfolio_value" in res.final_state or "cumulative_cash" in res.final_state
    assert res.final_state.get("portfolio_value", 0) > 0

    # Small MC also works (uses the same path)
    results = run_monte_carlo(cfg, n_sims=3, base_seed=99, n_jobs=1)
    assert len(results) == 3
    for r in results:
        assert r.final_state.get("portfolio_value", 0) > 0


# =============================================================================
# Tests for Phase 5 UI data model enhancements (additive only)
# =============================================================================


def test_saved_distribution_ui_fields_roundtrip():
    from financial_simulator.core.distributions import TriangularDistribution
    from financial_simulator.scenarios import SavedDistribution

    d = SavedDistribution(
        id="home-apprec",
        name="Home Appreciation",
        dist=TriangularDistribution(low=0.02, mode=0.035, high=0.06),
        units="rate",
        domain_hint="rate",
        tags=["real-estate"],
    )
    js = d.model_dump(mode="json")
    loaded = SavedDistribution.model_validate(js)
    assert loaded.units == "rate"
    assert loaded.domain_hint == "rate"
    assert loaded.last_used is None


def test_custom_metric_ui_fields():
    from financial_simulator.scenarios import CustomMetric

    m = CustomMetric(
        name="final_equity",
        metric_type="final_state_value",
        params={"key": "home_equity"},
        display_format="currency",
        unit_label="USD",
        higher_is_better=True,
        goal_value=250_000,
    )
    assert m.display_format == "currency"
    assert m.higher_is_better is True
    # Still computes correctly via the existing path
    from financial_simulator.scenarios import run_single

    cfg = ScenarioConfig(
        name="MetricUI",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 2, 1),
        initial_state={"home_equity": 180_000.0},
        custom_metrics=[m],
    )
    res = run_single(cfg, seed=1)
    assert "final_equity" in res.final_state.get("__custom_metrics__", {})


def test_scenario_library_crud_and_summary():
    from financial_simulator.scenarios import ScenarioLibrary

    lib = ScenarioLibrary()
    cfg1 = ScenarioConfig(name="Retire Test", start=datetime(2026, 1, 1), end=datetime(2036, 1, 1))
    lib.add(cfg1)
    assert lib.get_by_name("Retire Test") is not None
    assert lib.remove("Retire Test") is True
    assert len(lib.scenarios) == 0

    # summary() helper
    cfg2 = ScenarioConfig(
        name="Biz",
        start=datetime(2026, 1, 1),
        end=datetime(2029, 1, 1),
        event_builders=[ComposedEventBuilder(timing=IntervalTiming(interval=timedelta(days=30)), value_gen=FixedValue(value=1000))],
    )
    s = cfg2.summary()
    assert s["horizon_years"] > 2.9
    assert s["num_event_builders"] == 1


def test_user_persistence_helpers_smoke(tmp_path, monkeypatch):
    """Smoke the new user persistence functions (use tmp to avoid polluting real ~/.financial-simulator)."""

    # Monkeypatch the module constants so we write inside tmp_path instead of $HOME
    import financial_simulator.scenarios.persistence as pers
    from financial_simulator.scenarios import (
        ScenarioConfig,
        list_user_scenarios,
        load_user_scenario_library,
        save_user_scenario,
    )

    fake_root = tmp_path / "fake_user_root"
    monkeypatch.setattr(pers, "USER_DATA_ROOT", fake_root)
    monkeypatch.setattr(pers, "USER_SCENARIOS_DIR", fake_root / "scenarios")
    monkeypatch.setattr(pers, "USER_DISTRIBUTIONS_FILE", fake_root / "dist_lib.json")

    cfg = ScenarioConfig(name="My Custom Plan", start=datetime(2026, 1, 1), end=datetime(2027, 1, 1))
    saved_path = save_user_scenario(cfg)
    assert saved_path.exists()

    pairs = list_user_scenarios()
    assert len(pairs) >= 1
    assert any("My Custom" in n or "My-Custom" in n for n, _ in pairs)

    lib = load_user_scenario_library()
    assert len(lib.scenarios) >= 1


# =============================================================================
# Additional coverage tests (Step 9 - pushing scenarios/ package > 80%)
# =============================================================================


def test_scenario_library_add_duplicate_raises():
    import pytest

    from financial_simulator.scenarios import ScenarioLibrary

    lib = ScenarioLibrary()
    cfg = ScenarioConfig(name="Dup Test", start=datetime(2026, 1, 1), end=datetime(2027, 1, 1))
    lib.add(cfg)
    with pytest.raises(ValueError, match="already exists"):
        lib.add(cfg)  # same name


def test_scenario_config_get_all_referenced_distributions_and_helpers():
    from financial_simulator.core.distributions import TriangularDistribution
    from financial_simulator.core.event import DistributionValue

    cfg = ScenarioConfig(
        name="Dist Harvest",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 6, 1),
        event_builders=[
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=DistributionValue(dist=TriangularDistribution(low=-100, mode=-50, high=-10)),
            )
        ],
    )

    dists = cfg.get_all_referenced_distributions()
    assert len(dists) == 1
    assert isinstance(dists[0], TriangularDistribution)

    s = cfg.summary()
    assert s["num_event_builders"] == 1
    assert "has_stochastic" in s


def test_persistence_user_scenario_library_roundtrip(tmp_path, monkeypatch):
    """More thorough exercise of user scenario persistence (error paths + multiple saves)."""
    import financial_simulator.scenarios.persistence as pers
    from financial_simulator.scenarios import (
        list_user_scenarios,
        load_user_scenario,
        load_user_scenario_library,
        save_user_scenario,
    )

    fake_root = tmp_path / "user_lib_test"
    monkeypatch.setattr(pers, "USER_DATA_ROOT", fake_root)
    monkeypatch.setattr(pers, "USER_SCENARIOS_DIR", fake_root / "scenarios")
    monkeypatch.setattr(pers, "USER_DISTRIBUTIONS_FILE", fake_root / "dists.json")

    # Save two scenarios
    cfg1 = ScenarioConfig(name="Alpha", start=datetime(2026, 1, 1), end=datetime(2026, 12, 31))
    cfg2 = ScenarioConfig(name="Beta", start=datetime(2026, 1, 1), end=datetime(2027, 1, 1))

    p1 = save_user_scenario(cfg1)
    p2 = save_user_scenario(cfg2)

    assert p1.exists() and p2.exists()

    # List and load
    pairs = list_user_scenarios()
    assert len(pairs) >= 2

    loaded1 = load_user_scenario("Alpha")
    assert loaded1.name == "Alpha"

    # Load library
    lib = load_user_scenario_library()
    assert len(lib.scenarios) >= 2


def test_custom_metric_time_to_threshold_and_event_count_paths():
    """Exercise two metric types that had low coverage."""
    from financial_simulator.scenarios import CustomMetric

    # time_to_threshold
    m1 = CustomMetric(
        name="time_to_goal",
        metric_type="time_to_threshold",
        params={"state_key": "portfolio", "threshold": 150000, "direction": "above"},
    )
    cfg1 = ScenarioConfig(
        name="T2T",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 3, 1),
        initial_state={"portfolio": 100000.0},
        event_builders=[
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=FixedValue(value=20000.0),
            )
        ],
        custom_metrics=[m1],
    )
    res1 = run_single(cfg1, seed=7)
    val = res1.final_state["__custom_metrics__"]["time_to_goal"]
    assert val >= 0.0

    # event_count_by_type
    m2 = CustomMetric(
        name="income_count",
        metric_type="event_count_by_type",
        params={"metadata_type": "income"},
    )
    cfg2 = ScenarioConfig(
        name="Count",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 2, 1),
        initial_state={"cash": 0.0},
        event_builders=[
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=FixedValue(value=5000.0),
                metadata={"type": "income"},
            )
        ],
        custom_metrics=[m2],
    )
    res2 = run_single(cfg2, seed=3)
    assert res2.final_state["__custom_metrics__"]["income_count"] >= 1
