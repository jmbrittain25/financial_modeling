"""
Environmental / macro drivers — interest rates, housing, equity markets.

These write into simulation state keys that generators and background processes
can read (e.g. variable mortgage rate, portfolio value for dividends).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.components.external_drivers_editor import render_external_drivers_editor
from financial_simulator.scenarios.drivers import (
    make_interest_rate_driver,
    make_stock_market_driver,
)
from financial_simulator.scenarios.models import ContinuousGBMDriver


def make_housing_market_driver(
    name: str = "housing_market",
    target_state_key: str = "home_value",
    drift: float = 0.04,
    volatility: float = 0.08,
    initial_value: float = 500_000.0,
) -> ContinuousGBMDriver:
    """Stochastic home-value path (GBM with moderate drift / volatility)."""
    return ContinuousGBMDriver(
        name=name,
        target_state_key=target_state_key,
        drift=drift,
        volatility=volatility,
        initial_value=initial_value,
        metadata={"example": "housing"},
    )


def render_environment_editor(
    key_prefix: str = "environment",
    drivers: list[Any] | None = None,
    scenario_start: datetime | None = None,
    scenario_end: datetime | None = None,
    generators: list[Any] | None = None,
) -> list[Any]:
    """
    Top-level environmental drivers section with domain presets.
    """
    import streamlit as st

    from app.components.scenario_links import format_driver_links, get_driver_target_keys

    if drivers is None:
        drivers = []

    st.subheader("Market & macro environment")
    st.caption(
        "Define how interest rates, housing, and markets evolve over time. "
        "Generators link to these automatically when they read the same **state key** "
        "(e.g. a variable mortgage uses `market_rate`)."
    )

    preset_cols = st.columns(3)
    if preset_cols[0].button(
        "Interest rates",
        key=f"{key_prefix}_preset_rates",
        use_container_width=True,
        help="Mean-reverting short rate → writes to `market_rate`",
    ):
        drivers.append(make_interest_rate_driver())
        st.toast("Added interest rate environment", icon="✅")
        st.rerun()
    if preset_cols[1].button(
        "Housing market",
        key=f"{key_prefix}_preset_housing",
        use_container_width=True,
        help="Stochastic home value → writes to `home_value`",
    ):
        drivers.append(make_housing_market_driver())
        st.toast("Added housing market environment", icon="✅")
        st.rerun()
    if preset_cols[2].button(
        "Stock market",
        key=f"{key_prefix}_preset_equity",
        use_container_width=True,
        help="Equity GBM → writes to `portfolio_value`",
    ):
        drivers.append(make_stock_market_driver())
        st.toast("Added stock market environment", icon="✅")
        st.rerun()

    if drivers and generators:
        env_keys = get_driver_target_keys(drivers)
        linked_gens = []
        for gen in generators:
            label = format_driver_links(gen, drivers)
            if label:
                gname = getattr(gen, "name", None) or "Unnamed generator"
                linked_gens.append(f"**{gname}** → {label}")
        if linked_gens:
            with st.expander("Generator ↔ environment links", expanded=False):
                st.markdown("\n".join(f"- {line}" for line in linked_gens))
                st.caption(f"Active environment state keys: `{', '.join(env_keys)}`")

    drivers = render_external_drivers_editor(
        key_prefix=key_prefix,
        drivers=drivers,
        scenario_start=scenario_start,
        scenario_end=scenario_end,
        embedded=True,
    )
    return drivers


__all__ = [
    "make_housing_market_driver",
    "render_environment_editor",
]