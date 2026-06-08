"""Tests for required macro environment materialization and linking."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.components.scenario_links import format_macro_links
from financial_simulator.core import FixedValue, IntervalTiming
from financial_simulator.core.event import ComposedEventBuilder, VariableRateLoanValue
from financial_simulator.scenarios import (
    ScenarioConfig,
    build_engine,
    default_macro_environment,
    ensure_macro_environment,
    run_single,
)
from financial_simulator.scenarios.drivers import make_interest_rate_driver
from financial_simulator.scenarios.macro_environment import (
    bind_macro_to_initial_state,
    migrate_macro_from_external_drivers,
    sample_macro_paths,
)


def _minimal_cfg(**kwargs) -> ScenarioConfig:
    return ScenarioConfig(
        name="Macro Test",
        start=datetime(2026, 1, 1),
        end=datetime(2028, 1, 1),
        initial_state={"cash": 0.0},
        **kwargs,
    )


def test_default_macro_has_three_slots():
    macro = default_macro_environment()
    assert len(macro.slots()) == 3
    assert macro.state_keys() == ["market_rate", "home_value", "stocks"]


def test_migrate_legacy_interest_driver():
    legacy = [make_interest_rate_driver(target_state_key="market_rate")]
    migrated = migrate_macro_from_external_drivers(legacy)
    assert migrated is not None
    assert migrated.interest_rates.mode == "stochastic"
    assert migrated.interest_rates.state_key == "market_rate"


def test_build_engine_applies_constant_macro_keys():
    macro = default_macro_environment()
    macro = macro.model_copy(
        update={
            "housing": macro.housing.model_copy(update={"mode": "constant", "value": 750_000.0})
        }
    )
    cfg = _minimal_cfg(macro_environment=macro)
    eng = build_engine(cfg, seed=1)
    assert eng.initial_state["market_rate"] == macro.interest_rates.value
    assert eng.initial_state["home_value"] == 750_000.0


def test_growth_macro_adds_appreciation_process():
    macro = default_macro_environment()
    macro = macro.model_copy(
        update={
            "stock_market": macro.stock_market.model_copy(
                update={"mode": "growth", "value": 50_000.0, "annual_rate": 0.06}
            )
        }
    )
    cfg = _minimal_cfg(macro_environment=macro)
    eng = build_engine(cfg, seed=1)
    proc_names = [getattr(p, "name", "") for p in eng.continuous_processes]
    assert "macro:stock_market" in proc_names


def test_sample_growth_path_is_deterministic():
    macro = default_macro_environment()
    var = macro.housing.model_copy(
        update={"mode": "growth", "value": 400_000.0, "annual_rate": 0.05}
    )
    data = sample_macro_paths(
        var,
        datetime(2026, 1, 1),
        datetime(2027, 1, 1),
        n_paths=3,
    )
    assert len(data["paths"]) == 3
    assert data["paths"][0] == data["paths"][1]
    assert data["paths"][0][-1] > 400_000.0


def test_format_macro_links_for_variable_loan():
    macro = ensure_macro_environment(_minimal_cfg(macro_environment=default_macro_environment()))
    loan = ComposedEventBuilder(
        name="mortgage",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=VariableRateLoanValue(
            principal=300_000.0,
            initial_rate=0.06,
            term_months=360,
            rate_key="market_rate",
        ),
    )
    assert "Interest rates" in format_macro_links(loan, macro)


def test_ensure_macro_falls_back_to_defaults():
    cfg = _minimal_cfg()
    macro = ensure_macro_environment(cfg)
    assert macro.interest_rates.mode == "constant"


def test_bind_macro_maps_stocks_not_portfolio_value():
    macro = default_macro_environment()
    bound = bind_macro_to_initial_state(
        macro,
        {"cash": 1.0, "stocks": 265_000.0, "house equity": 100_000.0},
    )
    assert bound.stock_market.state_key == "stocks"
    assert bound.stock_market.value == 265_000.0
    assert bound.housing.state_key == "house equity"
    assert bound.housing.value == 100_000.0


def test_budget_scenario_grows_stocks_with_stock_market_macro():
    macro = default_macro_environment()
    macro = macro.model_copy(
        update={
            "stock_market": macro.stock_market.model_copy(
                update={
                    "state_key": "portfolio_value",
                    "mode": "growth",
                    "value": 100_000.0,
                    "annual_rate": 0.07,
                }
            )
        }
    )
    cfg = ScenarioConfig(
        name="budget-like",
        start=datetime(2026, 1, 1),
        end=datetime(2030, 12, 31),
        initial_state={"cash": 27_000.0, "stocks": 265_000.0},
        event_builders=[
            ComposedEventBuilder(
                name="heartbeat",
                timing=IntervalTiming(interval=timedelta(days=30)),
                value_gen=FixedValue(value=0.0),
            )
        ],
        macro_environment=macro,
    )

    macro = ensure_macro_environment(cfg)
    assert macro.stock_market.state_key == "stocks"
    assert macro.stock_market.value == 265_000.0

    result = run_single(cfg, seed=42)
    assert result.final_state["stocks"] > 265_000.0
    # ~5 years of 7% growth between monthly ticks (close to annual compounding)
    assert result.final_state["stocks"] == pytest.approx(265_000.0 * (1.07**5), rel=0.02)
