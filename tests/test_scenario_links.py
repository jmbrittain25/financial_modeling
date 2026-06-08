"""Tests for generator / environment / continuous-process linking."""

from __future__ import annotations

from datetime import timedelta

from app.components.event_builder_editor import GENERATOR_ID_KEY
from app.components.scenario_links import (
    CONTINUOUS_PROCESS_META_KEY,
    build_process_from_config,
    format_driver_links,
    sync_continuous_processes,
)
from financial_simulator.core import FixedValue, IntervalTiming
from financial_simulator.core.event import ComposedEventBuilder, VariableRateLoanValue
from financial_simulator.scenarios.drivers import make_interest_rate_driver
from financial_simulator.scenarios.models import ContinuousGBMDriver


def test_sync_continuous_processes_from_generator_metadata():
    gen_id = "abc-123"
    builder = ComposedEventBuilder(
        name="portfolio",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=FixedValue(100.0),
        metadata={
            GENERATOR_ID_KEY: gen_id,
            CONTINUOUS_PROCESS_META_KEY: {
                "enabled": True,
                "type": "appreciation",
                "var": "stocks",
                "rate": 0.05,
            },
        },
    )
    standalone = build_process_from_config(
        {"enabled": True, "type": "gbm", "var": "cash", "drift": 0.1, "volatility": 0.2},
        "orphan",
        "orphan",
    )
    standalone.name = "manual_process"

    synced = sync_continuous_processes([builder], [standalone])
    assert len(synced) == 2
    linked = [p for p in synced if getattr(p, "name", "").startswith("@gen:")]
    assert len(linked) == 1
    assert linked[0].var == "stocks"
    assert linked[0].rate == 0.05


def test_format_driver_links_matches_state_keys():
    rate_driver = make_interest_rate_driver(target_state_key="market_rate")
    equity_driver = ContinuousGBMDriver(
        name="equity",
        target_state_key="portfolio_value",
        drift=0.08,
        volatility=0.16,
        initial_value=1.0,
    )
    loan_gen = ComposedEventBuilder(
        name="mortgage",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=VariableRateLoanValue(
            principal=300_000.0,
            initial_rate=0.06,
            term_months=360,
            rate_key="market_rate",
        ),
    )
    assert format_driver_links(loan_gen, [rate_driver, equity_driver]) == "interest_rate_path"
    assert format_driver_links(loan_gen, [equity_driver]) == ""