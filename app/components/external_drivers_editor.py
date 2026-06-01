"""
External Drivers Editor (Phase 5+ integrated).

Supports all four driver types:
- DiscreteRateDriver (variable rates feeding VariableRateLoanValue)
- ConstantDriver
- ContinuousGBMDriver (equity markets / growth with volatility)
- ContinuousMeanRevertDriver (inflation, interest rates with realistic reversion)

Uses sample_driver_path for live previews on continuous drivers.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.components.distribution_viz import render_distribution_picker
from financial_simulator.core.distributions import NormalDistribution
from financial_simulator.core.event import IntervalTiming
from financial_simulator.scenarios import (
    ConstantDriver,
    ContinuousGBMDriver,
    ContinuousMeanRevertDriver,
    DiscreteRateDriver,
)
from financial_simulator.scenarios.drivers import (
    make_inflation_driver,  # noqa: F401 - available for future quick-adds / examples
    make_interest_rate_driver,  # noqa: F401
    make_stock_market_driver,  # noqa: F401
    sample_driver_path,
)


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


def get_default_gbm_driver() -> ContinuousGBMDriver:
    return ContinuousGBMDriver(
        name="equity_market_driver",
        target_state_key="portfolio_value",
        drift=0.08,
        volatility=0.16,
        initial_value=500_000.0,
    )


def get_default_mean_revert_driver() -> ContinuousMeanRevertDriver:
    return ContinuousMeanRevertDriver(
        name="inflation_driver",
        target_state_key="inflation_index",
        long_term_mean=0.025,
        speed=0.6,
        volatility=0.005,
        initial_value=1.0,
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

    # Quick add - support for all 4 driver types
    cols = st.columns(4)
    if cols[0].button(
        "➕ Discrete Rate",
        key=f"{key_prefix}_add_rate",
        use_container_width=True,
        help="For variable interest rates feeding loans",
    ):
        drivers = drivers + [get_default_discrete_rate_driver()]
        st.rerun()
    if cols[1].button(
        "➕ Constant",
        key=f"{key_prefix}_add_const",
        use_container_width=True,
    ):
        drivers = drivers + [get_default_constant_driver()]
        st.rerun()
    if cols[2].button(
        "➕ GBM (Markets)",
        key=f"{key_prefix}_add_gbm",
        use_container_width=True,
        help="Equity-style growth with volatility",
    ):
        drivers = drivers + [get_default_gbm_driver()]
        st.rerun()
    if cols[3].button(
        "➕ Mean-Revert",
        key=f"{key_prefix}_add_mr",
        use_container_width=True,
        help="Inflation or rates with reversion to long-term mean",
    ):
        drivers = drivers + [get_default_mean_revert_driver()]
        st.rerun()

    if not drivers:
        st.info(
            "No external drivers yet. Rate drivers are excellent for modeling variable mortgages, credit lines, or inflation."
        )
    else:
        to_delete = []
        for idx, drv in enumerate(drivers):
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                drv_name = getattr(drv, "name", f"Driver {idx}")
                c1.markdown(
                    f"**{drv_name}** — `{getattr(drv, 'type', 'driver')}` → `{getattr(drv, 'target_state_key', '?')}`"
                )

                if c2.button("🗑️", key=f"{key_prefix}_del_{idx}", use_container_width=True):
                    to_delete.append(idx)

                if isinstance(drv, DiscreteRateDriver):
                    new_name = st.text_input(
                        "Driver Name", value=drv.name, key=f"{key_prefix}_rate_name_{idx}"
                    )
                    new_target = st.text_input(
                        "Target State Key",
                        value=drv.target_state_key,
                        key=f"{key_prefix}_rate_target_{idx}",
                    )

                    st.markdown(
                        "**Rate Distribution** (this value will be written into state on the schedule below)"
                    )
                    new_dist = render_distribution_picker(
                        key_prefix=f"{key_prefix}_rate_dist_{idx}",
                        initial=drv.dist,
                        show_save_section=False,
                    )

                    # Timing (simplified)
                    interval_days = (
                        drv.timing.interval.days if hasattr(drv.timing, "interval") else 90
                    )
                    new_interval = st.number_input(
                        "Update every N days",
                        min_value=1,
                        max_value=3650,
                        value=interval_days,
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
                    new_name = st.text_input(
                        "Name", value=drv.name, key=f"{key_prefix}_const_name_{idx}"
                    )
                    new_target = st.text_input(
                        "Target State Key",
                        value=drv.target_state_key,
                        key=f"{key_prefix}_const_target_{idx}",
                    )
                    new_val = st.number_input(
                        "Constant Value", value=drv.value, key=f"{key_prefix}_const_val_{idx}"
                    )

                    drivers[idx] = ConstantDriver(
                        name=new_name,
                        target_state_key=new_target,
                        value=float(new_val),
                    )

                elif isinstance(drv, ContinuousGBMDriver):
                    new_name = st.text_input(
                        "Name", value=drv.name, key=f"{key_prefix}_gbm_name_{idx}"
                    )
                    new_target = st.text_input(
                        "Target State Key",
                        value=drv.target_state_key,
                        key=f"{key_prefix}_gbm_target_{idx}",
                    )

                    c1, c2, c3 = st.columns(3)
                    new_drift = c1.number_input(
                        "Drift (annual)",
                        value=drv.drift,
                        step=0.01,
                        format="%.3f",
                        key=f"{key_prefix}_gbm_drift_{idx}",
                    )
                    new_vol = c2.number_input(
                        "Volatility",
                        value=drv.volatility,
                        min_value=0.001,
                        step=0.01,
                        format="%.3f",
                        key=f"{key_prefix}_gbm_vol_{idx}",
                    )
                    new_init = c3.number_input(
                        "Initial Value",
                        value=drv.initial_value,
                        step=1000.0,
                        key=f"{key_prefix}_gbm_init_{idx}",
                    )

                    try:
                        preview = sample_driver_path(drv, n_paths=4, seed=42)
                        st.caption("Live sample paths (preview)")
                        st.text(
                            f"Terminal mean: {preview['summary']['mean_terminal']:.2f} | std: {preview['summary']['std_terminal']:.2f}"
                        )
                    except Exception:
                        pass

                    drivers[idx] = ContinuousGBMDriver(
                        name=new_name,
                        target_state_key=new_target,
                        drift=float(new_drift),
                        volatility=float(new_vol),
                        initial_value=float(new_init),
                        metadata=getattr(drv, "metadata", {}),
                    )

                elif isinstance(drv, ContinuousMeanRevertDriver):
                    new_name = st.text_input(
                        "Name", value=drv.name, key=f"{key_prefix}_mr_name_{idx}"
                    )
                    new_target = st.text_input(
                        "Target State Key",
                        value=drv.target_state_key,
                        key=f"{key_prefix}_mr_target_{idx}",
                    )

                    c1, c2 = st.columns(2)
                    new_mean = c1.number_input(
                        "Long-term Mean",
                        value=drv.long_term_mean,
                        step=0.005,
                        format="%.4f",
                        key=f"{key_prefix}_mr_mean_{idx}",
                    )
                    new_speed = c1.number_input(
                        "Reversion Speed",
                        value=drv.speed,
                        min_value=0.01,
                        step=0.1,
                        key=f"{key_prefix}_mr_speed_{idx}",
                    )
                    new_vol = c2.number_input(
                        "Volatility",
                        value=drv.volatility,
                        min_value=0.0001,
                        step=0.001,
                        format="%.4f",
                        key=f"{key_prefix}_mr_vol_{idx}",
                    )
                    new_init = c2.number_input(
                        "Initial Value",
                        value=drv.initial_value,
                        step=0.01,
                        format="%.4f",
                        key=f"{key_prefix}_mr_init_{idx}",
                    )

                    try:
                        preview = sample_driver_path(drv, n_paths=4, seed=42)
                        st.caption("Live sample paths (preview)")
                        st.text(f"Terminal mean: {preview['summary']['mean_terminal']:.4f}")
                    except Exception:
                        pass

                    drivers[idx] = ContinuousMeanRevertDriver(
                        name=new_name,
                        target_state_key=new_target,
                        long_term_mean=float(new_mean),
                        speed=float(new_speed),
                        volatility=float(new_vol),
                        initial_value=float(new_init),
                        metadata=getattr(drv, "metadata", {}),
                    )

                else:
                    st.warning(f"Unsupported driver type: {type(drv)} (advanced)")

        if to_delete:
            for i in sorted(to_delete, reverse=True):
                del drivers[i]
            st.rerun()

    return drivers


__all__ = [
    "render_external_drivers_editor",
    "get_default_discrete_rate_driver",
    "get_default_constant_driver",
    "get_default_gbm_driver",
    "get_default_mean_revert_driver",
]
