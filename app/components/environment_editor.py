"""
Market & macro environment — three required variables in collapsible subsections.

Each of interest rates, housing, and stock market supports constant, growth/decline,
or stochastic evolution with an inline path preview.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from financial_simulator.scenarios.macro_environment import (
    SLOT_DEFS,
    MacroEnvironment,
    MacroSlot,
    MacroVariableConfig,
    default_macro_environment,
    ensure_macro_environment,
    macro_summary_label,
    sample_macro_paths,
)
from financial_simulator.scenarios.models import ScenarioConfig

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    go = None  # type: ignore
    HAS_PLOTLY = False


MODE_LABELS = {
    "constant": "Constant",
    "growth": "Growth / decline",
    "stochastic": "Stochastic",
}


def _plot_macro_paths(
    var: MacroVariableConfig,
    start: datetime,
    end: datetime,
    *,
    n_paths: int = 5,
    seed: int = 42,
    height: int = 280,
) -> Any:
    if go is None:
        raise RuntimeError("plotly is required for macro previews")

    data = sample_macro_paths(var, start, end, n_paths=n_paths, seed=seed)
    times = [datetime.fromisoformat(t) if isinstance(t, str) else t for t in data["times"]]
    paths = data["paths"]

    fig = go.Figure()
    for i, path in enumerate(paths):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=path,
                mode="lines",
                name=f"Path {i + 1}" if n_paths <= 4 else None,
                line=dict(width=1.2, color="rgba(70,130,180,0.55)"),
                showlegend=(n_paths <= 4 and var.mode == "stochastic"),
            )
        )

    if paths:
        import numpy as np

        mean_path = np.mean(paths, axis=0).tolist()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=mean_path,
                mode="lines",
                name="Expected path" if var.mode != "stochastic" else "Mean path",
                line=dict(width=2.5, color="#1f77b4"),
            )
        )

    meta = SLOT_DEFS[var.slot]
    fig.update_layout(
        title=f"`{var.state_key}` over simulation horizon",
        xaxis_title="Time",
        yaxis_title="Rate" if meta["value_kind"] == "rate" else "Value",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=var.mode == "stochastic",
    )
    return fig


def _render_mode_selector(
    st, key_prefix: str, slot: MacroSlot, current: MacroVariableConfig
) -> str:
    modes = list(MODE_LABELS.keys())
    labels = [MODE_LABELS[m] for m in modes]
    try:
        idx = modes.index(current.mode)
    except ValueError:
        idx = 0
    choice = st.radio(
        "How should this variable change over time?",
        options=labels,
        index=idx,
        horizontal=True,
        key=f"{key_prefix}_{slot}_mode",
    )
    return modes[labels.index(choice)]


def _render_variable_form(
    st,
    key_prefix: str,
    slot: MacroSlot,
    current: MacroVariableConfig,
) -> MacroVariableConfig:
    meta = SLOT_DEFS[slot]
    is_rate = meta["value_kind"] == "rate"
    mode = _render_mode_selector(st, key_prefix, slot, current)

    st.caption(f"Simulation state key: `{current.state_key}`")

    if mode == "constant":
        if is_rate:
            value = st.number_input(
                "Fixed rate (e.g. 0.05 = 5%)",
                value=float(current.value),
                step=0.0025,
                format="%.4f",
                key=f"{key_prefix}_{slot}_const",
            )
        else:
            value = st.number_input(
                "Fixed value ($)",
                value=float(current.value),
                step=1000.0,
                key=f"{key_prefix}_{slot}_const",
            )
        return current.model_copy(update={"mode": "constant", "value": float(value)})

    if mode == "growth":
        c1, c2 = st.columns(2)
        if is_rate:
            value = c1.number_input(
                "Starting rate",
                value=float(current.value),
                step=0.0025,
                format="%.4f",
                key=f"{key_prefix}_{slot}_grow_start",
            )
            annual_rate = c2.number_input(
                "Annual change (e.g. -0.005 = −0.5%/yr)",
                value=float(current.annual_rate),
                step=0.0025,
                format="%.4f",
                key=f"{key_prefix}_{slot}_grow_rate",
            )
        else:
            value = c1.number_input(
                "Starting value ($)",
                value=float(current.value),
                step=5000.0,
                key=f"{key_prefix}_{slot}_grow_start",
            )
            annual_rate = c2.number_input(
                "Annual growth rate (e.g. 0.04 = 4%/yr, negative = decline)",
                value=float(current.annual_rate),
                step=0.005,
                format="%.3f",
                key=f"{key_prefix}_{slot}_grow_rate",
            )
        return current.model_copy(
            update={
                "mode": "growth",
                "value": float(value),
                "annual_rate": float(annual_rate),
            }
        )

    # stochastic
    default_stoch = meta["default_stochastic"]
    stoch_options = ["gbm", "mean_reverting"]
    stoch_labels = {
        "gbm": "Random walk with drift (GBM)",
        "mean_reverting": "Mean-reverting (rates / indices)",
    }
    try:
        stoch_idx = stoch_options.index(current.stochastic_type)
    except ValueError:
        stoch_idx = stoch_options.index(default_stoch)
    stoch_choice = st.radio(
        "Stochastic model",
        options=[stoch_labels[k] for k in stoch_options],
        index=stoch_idx,
        horizontal=True,
        key=f"{key_prefix}_{slot}_stoch_type",
    )
    stoch_type = stoch_options[[stoch_labels[k] for k in stoch_options].index(stoch_choice)]

    if is_rate:
        value = st.number_input(
            "Starting rate",
            value=float(current.value),
            step=0.0025,
            format="%.4f",
            key=f"{key_prefix}_{slot}_stoch_start",
        )
    else:
        value = st.number_input(
            "Starting value ($)",
            value=float(current.value),
            step=5000.0,
            key=f"{key_prefix}_{slot}_stoch_start",
        )

    updates: dict[str, Any] = {
        "mode": "stochastic",
        "value": float(value),
        "stochastic_type": stoch_type,
    }

    if stoch_type == "gbm":
        c1, c2 = st.columns(2)
        updates["drift"] = c1.number_input(
            "Drift (annual)",
            value=float(current.drift),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_{slot}_drift",
        )
        updates["volatility"] = c2.number_input(
            "Volatility (annual)",
            value=float(current.volatility),
            min_value=0.001,
            step=0.01,
            format="%.3f",
            key=f"{key_prefix}_{slot}_vol",
        )
    else:
        c1, c2, c3 = st.columns(3)
        updates["long_term_mean"] = c1.number_input(
            "Long-term mean",
            value=float(current.long_term_mean),
            step=0.005 if is_rate else 5000.0,
            format="%.4f" if is_rate else None,
            key=f"{key_prefix}_{slot}_ltm",
        )
        updates["reversion_speed"] = c2.number_input(
            "Reversion speed",
            value=float(current.reversion_speed),
            min_value=0.01,
            step=0.1,
            key=f"{key_prefix}_{slot}_speed",
        )
        updates["volatility"] = c3.number_input(
            "Volatility",
            value=float(current.volatility),
            min_value=0.0001,
            step=0.001 if is_rate else 0.01,
            format="%.4f" if is_rate else "%.3f",
            key=f"{key_prefix}_{slot}_stoch_vol",
        )

    return current.model_copy(update=updates)


def _render_slot_expander(
    st,
    key_prefix: str,
    slot: MacroSlot,
    current: MacroVariableConfig,
    scenario_start: datetime | None,
    scenario_end: datetime | None,
    *,
    expanded: bool,
) -> MacroVariableConfig:
    label = macro_summary_label(current)
    with st.expander(label, expanded=expanded):
        updated = _render_variable_form(st, key_prefix, slot, current)

        eff_start = scenario_start or datetime(2026, 1, 1)
        eff_end = scenario_end or datetime(2036, 1, 1)
        if HAS_PLOTLY:
            try:
                fig = _plot_macro_paths(updated, eff_start, eff_end, n_paths=6, seed=42)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{slot}_chart")
                summary = sample_macro_paths(updated, eff_start, eff_end, n_paths=200, seed=123)[
                    "summary"
                ]
                m1, m2, m3 = st.columns(3)
                m1.metric("End (mean)", f"{summary['mean_terminal']:.4f}")
                m2.metric("End (std)", f"{summary['std_terminal']:.4f}")
                m3.metric(
                    "End range", f"{summary['min_terminal']:.2f} – {summary['max_terminal']:.2f}"
                )
            except Exception as ex:
                st.caption(f"Preview unavailable: {ex}")
        else:
            st.caption("Install plotly for path previews.")

        return updated


def render_environment_editor(
    key_prefix: str = "environment",
    macro: MacroEnvironment | None = None,
    scenario: ScenarioConfig | None = None,
    scenario_start: datetime | None = None,
    scenario_end: datetime | None = None,
    generators: list[Any] | None = None,
) -> MacroEnvironment:
    """
    Three collapsible macro subsections. Always returns a complete MacroEnvironment.
    """
    import streamlit as st

    from app.components.scenario_links import format_macro_links

    if macro is None and scenario is not None:
        macro = ensure_macro_environment(scenario)
    if macro is None:
        macro = default_macro_environment()

    st.subheader("Market & macro environment")
    st.caption(
        "Every simulation includes **interest rates**, **housing**, and **stock market** "
        "state variables. Keep them constant for a simple baseline, or add growth and "
        "volatility when macro factors matter."
    )

    slot_vars = {
        "interest_rates": macro.interest_rates,
        "housing": macro.housing,
        "stock_market": macro.stock_market,
    }
    updated_slots: dict[MacroSlot, MacroVariableConfig] = {}
    for i, slot in enumerate(("interest_rates", "housing", "stock_market")):
        updated_slots[slot] = _render_slot_expander(
            st,
            key_prefix,
            slot,
            slot_vars[slot],
            scenario_start,
            scenario_end,
            expanded=(i == 0),
        )

    result = MacroEnvironment(
        interest_rates=updated_slots["interest_rates"],
        housing=updated_slots["housing"],
        stock_market=updated_slots["stock_market"],
    )

    if generators:
        linked = []
        for gen in generators:
            label = format_macro_links(gen, result)
            if label:
                gname = getattr(gen, "name", None) or "Unnamed generator"
                linked.append(f"**{gname}** → {label}")
        if linked:
            with st.expander("Generator ↔ environment links", expanded=False):
                st.markdown("\n".join(f"- {line}" for line in linked))
                st.caption(f"State keys: `{', '.join(result.state_keys())}`")

    return result


__all__ = ["render_environment_editor"]
