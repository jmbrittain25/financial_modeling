"""
Continuous Processes Editor.

Supports the three built-in process types:
- appreciation (simple geometric growth)
- gbm (Geometric Brownian Motion)
- mean_reverting (Ornstein-Uhlenbeck style)
"""

from __future__ import annotations

from typing import Any

from financial_simulator.core.simulation import (
    AppreciationProcess,
    GBMContinuousProcess,
    GeometricBrownianMotion,
    MeanRevertingContinuousProcess,
    MeanRevertingProcess,
)

PROCESS_TYPE_INFO = {
    "appreciation": {
        "label": "Appreciation (simple growth)",
        "description": "Deterministic geometric growth — great for home values, salary inflation, etc.",
    },
    "gbm": {
        "label": "GBM (Geometric Brownian Motion)",
        "description": "Stochastic log-normal growth with drift and volatility. Classic equity model.",
    },
    "mean_reverting": {
        "label": "Mean Reverting",
        "description": "Process that tends to revert to a long-term mean (interest rates, inflation, etc.).",
    },
}


def get_default_appreciation() -> AppreciationProcess:
    return AppreciationProcess(rate=0.04, var="property_value", name="Home Appreciation")


def get_default_gbm() -> GBMContinuousProcess:
    return GBMContinuousProcess(
        var="stocks",
        process=GeometricBrownianMotion(drift=0.08, volatility=0.16),
        name="Equity GBM",
    )


def get_default_mean_reverting() -> MeanRevertingContinuousProcess:
    return MeanRevertingContinuousProcess(
        var="interest_rate",
        process=MeanRevertingProcess(long_term_mean=0.045, speed=1.2, volatility=0.008),
        name="Rate Mean Reversion",
    )


def render_continuous_processes_editor(
    key_prefix: str = "processes",
    processes: list[Any] | None = None,
) -> list[Any]:
    """
    Editor for continuous processes.
    """
    import streamlit as st

    if processes is None:
        processes = []

    st.markdown("### 📈 Continuous Processes")
    st.caption(
        "These evolve state variables continuously between discrete events using time deltas. "
        "Use them for investment growth, inflation, home appreciation, etc."
    )

    cols = st.columns(3)
    if cols[0].button("➕ Simple Appreciation", key=f"{key_prefix}_add_app", use_container_width=True):
        processes = processes + [get_default_appreciation()]
        st.rerun()
    if cols[1].button("➕ GBM (Stocks/Equity)", key=f"{key_prefix}_add_gbm", use_container_width=True):
        processes = processes + [get_default_gbm()]
        st.rerun()
    if cols[2].button("➕ Mean Reverting", key=f"{key_prefix}_add_mr", use_container_width=True):
        processes = processes + [get_default_mean_reverting()]
        st.rerun()

    if not processes:
        st.info("No continuous processes defined. These are excellent for modeling growth of investments or assets over time.")
    else:
        to_delete = []
        for idx, proc in enumerate(processes):
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                pname = getattr(proc, "name", f"Process {idx}")
                c1.markdown(f"**{pname}** — `{proc.type}` on `{getattr(proc, 'var', '?')}`")

                if c2.button("🗑️", key=f"{key_prefix}_del_{idx}", use_container_width=True):
                    to_delete.append(idx)

                new_name = st.text_input("Name (optional)", value=getattr(proc, "name", ""), key=f"{key_prefix}_pname_{idx}")
                new_var = st.text_input("State Variable to Affect", value=getattr(proc, "var", ""), key=f"{key_prefix}_pvar_{idx}")

                if isinstance(proc, AppreciationProcess):
                    rate = st.number_input("Annual Growth Rate", value=proc.rate, step=0.005, format="%.3f", key=f"{key_prefix}_app_rate_{idx}")
                    processes[idx] = AppreciationProcess(rate=float(rate), var=new_var, name=new_name or None)

                elif isinstance(proc, GBMContinuousProcess):
                    drift = st.number_input("Drift (expected return)", value=proc.process.drift, step=0.005, format="%.3f", key=f"{key_prefix}_gbm_drift_{idx}")
                    vol = st.number_input("Volatility (std dev)", value=proc.process.volatility, min_value=0.001, step=0.01, format="%.3f", key=f"{key_prefix}_gbm_vol_{idx}")
                    processes[idx] = GBMContinuousProcess(
                        var=new_var,
                        process=GeometricBrownianMotion(drift=float(drift), volatility=float(vol)),
                        name=new_name or None,
                    )

                elif isinstance(proc, MeanRevertingContinuousProcess):
                    ltm = st.number_input("Long-term Mean", value=proc.process.long_term_mean, step=0.005, format="%.3f", key=f"{key_prefix}_mr_mean_{idx}")
                    speed = st.number_input("Speed of Reversion", value=proc.process.speed, min_value=0.01, step=0.1, key=f"{key_prefix}_mr_speed_{idx}")
                    vol = st.number_input("Volatility", value=proc.process.volatility, min_value=0.001, step=0.005, format="%.3f", key=f"{key_prefix}_mr_vol_{idx}")
                    processes[idx] = MeanRevertingContinuousProcess(
                        var=new_var,
                        process=MeanRevertingProcess(long_term_mean=float(ltm), speed=float(speed), volatility=float(vol)),
                        name=new_name or None,
                    )
                else:
                    st.warning("Unsupported process type")

        if to_delete:
            for i in sorted(to_delete, reverse=True):
                del processes[i]
            st.rerun()

    return processes


__all__ = [
    "render_continuous_processes_editor",
    "get_default_appreciation",
    "get_default_gbm",
    "get_default_mean_reverting",
]
