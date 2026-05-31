"""
External Drivers Editor.

Currently focused on DiscreteRateDriver (the most useful and commonly used pattern)
with support for ConstantDriver. Other driver types are stubbed for future extension.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.components.distribution_viz import render_distribution_picker
from financial_simulator.core.distributions import NormalDistribution
from financial_simulator.core.event import IntervalTiming
from financial_simulator.scenarios import ConstantDriver, DiscreteRateDriver


def get_default_discrete_rate_driver() -> DiscreteRateDriver:
    return DiscreteRateDriver(
        name="interest_rate_driver",
        target_state_key="market_rate",
        dist=NormalDistribution(mean=0.05, std=0.012),
        timing=IntervalTiming(interval=timedelta(days=90)),
        metadata={"source": "external_driver"},
    )


def get_default_constant_driver() -> ConstantDriver:
    return ConstantDriver(
        name="initial_inflation",
        target_state_key="inflation_rate",
        value=0.025,
    )


def render_external_drivers_editor(
    key_prefix: str = "drivers",
    drivers: list[Any] | None = None,
) -> list[Any]:
    """
    Editor for external drivers list.
    """
    import streamlit as st

    if drivers is None:
        drivers = []

    st.markdown("### 🔗 External Drivers")
    st.caption(
        "External drivers inject stochastic or constant values into the simulation state on their own schedule. "
        "The most powerful use case is variable interest rates that feed into loans (VariableRateLoanValue)."
    )

    # Quick add
    cols = st.columns(2)
    if cols[0].button("➕ Add Discrete Rate Driver (recommended)", key=f"{key_prefix}_add_rate", use_container_width=True):
        drivers = drivers + [get_default_discrete_rate_driver()]
        st.rerun()
    if cols[1].button("➕ Add Constant Driver", key=f"{key_prefix}_add_const", use_container_width=True):
        drivers = drivers + [get_default_constant_driver()]
        st.rerun()

    if not drivers:
        st.info("No external drivers yet. Rate drivers are excellent for modeling variable mortgages, credit lines, or inflation.")
    else:
        to_delete = []
        for idx, drv in enumerate(drivers):
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                drv_name = getattr(drv, "name", f"Driver {idx}")
                c1.markdown(f"**{drv_name}** — `{getattr(drv, 'type', 'driver')}` → `{getattr(drv, 'target_state_key', '?')}`")

                if c2.button("🗑️", key=f"{key_prefix}_del_{idx}", use_container_width=True):
                    to_delete.append(idx)

                if isinstance(drv, DiscreteRateDriver):
                    new_name = st.text_input("Driver Name", value=drv.name, key=f"{key_prefix}_rate_name_{idx}")
                    new_target = st.text_input("Target State Key", value=drv.target_state_key, key=f"{key_prefix}_rate_target_{idx}")

                    st.markdown("**Rate Distribution** (this value will be written into state on the schedule below)")
                    new_dist = render_distribution_picker(
                        key_prefix=f"{key_prefix}_rate_dist_{idx}",
                        initial=drv.dist,
                        show_save_section=False,
                    )

                    # Timing (simplified)
                    interval_days = drv.timing.interval.days if hasattr(drv.timing, "interval") else 90
                    new_interval = st.number_input(
                        "Update every N days",
                        min_value=1, max_value=3650, value=interval_days,
                        key=f"{key_prefix}_rate_interval_{idx}",
                    )

                    new_timing = IntervalTiming(interval=timedelta(days=int(new_interval)))

                    drivers[idx] = DiscreteRateDriver(
                        name=new_name,
                        target_state_key=new_target,
                        dist=new_dist,
                        timing=new_timing,
                        metadata=drv.metadata,
                    )

                elif isinstance(drv, ConstantDriver):
                    new_name = st.text_input("Name", value=drv.name, key=f"{key_prefix}_const_name_{idx}")
                    new_target = st.text_input("Target State Key", value=drv.target_state_key, key=f"{key_prefix}_const_target_{idx}")
                    new_val = st.number_input("Constant Value", value=drv.value, key=f"{key_prefix}_const_val_{idx}")

                    drivers[idx] = ConstantDriver(
                        name=new_name,
                        target_state_key=new_target,
                        value=float(new_val),
                    )
                else:
                    st.warning(f"Unsupported driver type: {type(drv)} (advanced)")

        if to_delete:
            for i in sorted(to_delete, reverse=True):
                del drivers[i]
            st.rerun()

    return drivers


__all__ = ["render_external_drivers_editor", "get_default_discrete_rate_driver"]
