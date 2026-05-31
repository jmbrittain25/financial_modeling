"""
Interactive External Driver Editor with Live Path Preview (Plotly).

This component is the "wow" UI for External Drivers in the Scenario Builder.
Users can:
- Choose driver type (Discrete Rate, Constant, GBM Continuous, Mean-Revert Continuous)
- Edit name, target state key, and type-specific parameters live
- For DiscreteRateDriver: embed the existing distribution picker for the inner dist
- See an immediate multi-path time-series preview (fan chart style)
- Get quick stats on the simulated paths

Designed to be dropped into streamlit_app.py exactly like render_distribution_picker.

Pure helpers (plot_driver_preview, sample_stats, etc.) are importable/testable
without Streamlit (see tests/test_driver_viz.py).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from financial_simulator.core.distributions import NormalDistribution
from financial_simulator.core.event import IntervalTiming
from financial_simulator.scenarios.drivers import (
    AnyExternalDriver,
    DiscreteRateDriver,
    create_external_driver,
    sample_driver_path,
)

# Lazy plotly (consistent with distribution_viz)
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None  # type: ignore


# -----------------------------------------------------------------------------
# Pure helpers (testable, no Streamlit)
# -----------------------------------------------------------------------------


def plot_driver_preview(
    driver: AnyExternalDriver,
    start: datetime | None = None,
    end: datetime | None = None,
    n_paths: int = 5,
    seed: int = 42,
    height: int = 380,
    title: str | None = None,
) -> Any:
    """Return a Plotly Figure showing multiple sampled paths for the driver."""
    if go is None:
        raise RuntimeError("plotly is required for driver previews")

    start = start or datetime(2026, 1, 1)
    end = end or datetime(2027, 1, 1)

    data = sample_driver_path(driver, start, end, freq="MS", n_paths=n_paths, seed=seed)
    times = [datetime.fromisoformat(t) if isinstance(t, str) else t for t in data["times"]]
    paths = data["paths"]

    fig = go.Figure()

    # Individual paths (light)
    for i, path in enumerate(paths):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=path,
                mode="lines",
                name=f"Path {i+1}" if n_paths <= 4 else None,
                line=dict(width=1.2, color="rgba(70,130,180,0.6)"),
                showlegend=(n_paths <= 4),
            )
        )

    # Mean path (bold)
    if paths:
        mean_path = np.mean(paths, axis=0).tolist()
        fig.add_trace(
            go.Scatter(
                x=times,
                y=mean_path,
                mode="lines",
                name="Mean path",
                line=dict(width=3, color="#1f77b4"),
            )
        )

    driver_name = getattr(driver, "name", "Driver")
    target = getattr(driver, "target_state_key", "")
    fig_title = title or f"{driver_name} → {target} (live paths)"

    fig.update_layout(
        title=fig_title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def get_driver_stats(
    driver: AnyExternalDriver, n_paths: int = 200, seed: int = 42
) -> dict[str, float]:
    """Quick terminal-value statistics across many paths (for metrics row)."""
    start = datetime(2026, 1, 1)
    end = datetime(2027, 1, 1)
    data = sample_driver_path(driver, start, end, freq="M", n_paths=n_paths, seed=seed)
    terminals = [p[-1] for p in data["paths"]]
    arr = np.asarray(terminals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


# -----------------------------------------------------------------------------
# Parameter defaults (drive the dynamic form)
# -----------------------------------------------------------------------------

DRIVER_TYPE_LABELS: dict[str, str] = {
    "discrete_rate": "Discrete Rate (samples distribution on schedule)",
    "constant": "Constant (one-time state injection)",
    "gbm_continuous": "GBM Continuous (equity-style growth with vol)",
    "mean_revert_continuous": "Mean-Reverting Continuous (rates, inflation)",
}

DRIVER_TYPE_ORDER = list(DRIVER_TYPE_LABELS.keys())


def get_default_driver_params(driver_type: str) -> dict[str, Any]:
    if driver_type == "discrete_rate":
        return {
            "name": "interest_rate_driver",
            "target_state_key": "market_rate",
            "dist": {"type": "normal", "mean": 0.055, "std": 0.012},
            "timing": {"type": "Interval", "interval": "P90D"},
        }
    if driver_type == "constant":
        return {
            "name": "initial_cash_inject",
            "target_state_key": "cash",
            "value": 25000.0,
        }
    if driver_type == "gbm_continuous":
        return {
            "name": "equity_market",
            "target_state_key": "portfolio_value",
            "drift": 0.08,
            "volatility": 0.18,
            "initial_value": 500_000.0,
        }
    if driver_type == "mean_revert_continuous":
        return {
            "name": "inflation_index",
            "target_state_key": "inflation",
            "long_term_mean": 0.025,
            "speed": 0.8,
            "volatility": 0.006,
            "initial_value": 0.03,
        }
    return {}


# -----------------------------------------------------------------------------
# Main render component (the one called from Streamlit pages)
# -----------------------------------------------------------------------------


def render_external_driver_editor(
    key_prefix: str = "driver",
    initial: AnyExternalDriver | None = None,
    show_save_section: bool = True,
    on_save_callback: Callable[[AnyExternalDriver], None] | None = None,
    height: int = 420,
) -> AnyExternalDriver:
    """Live, self-contained editor for any External Driver type.

    Returns a freshly constructed (or edited) AnyExternalDriver on every interaction.
    Mirrors the API and UX idioms of render_distribution_picker exactly.
    """
    # Determine starting type
    if initial is not None:
        driver_type = getattr(initial, "type", "discrete_rate")
    else:
        driver_type = "discrete_rate"

    import streamlit as st  # lazy — allows test collection in environments without Streamlit

    # Type selector (stable ordering)
    type_labels = [DRIVER_TYPE_LABELS[t] for t in DRIVER_TYPE_ORDER]
    type_values = DRIVER_TYPE_ORDER
    current_label = DRIVER_TYPE_LABELS.get(driver_type, type_labels[0])
    try:
        type_idx = type_labels.index(current_label)
    except ValueError:
        type_idx = 0

    selected_label = st.selectbox(
        "Driver Type",
        type_labels,
        index=type_idx,
        key=f"{key_prefix}_type",
        help="Choose the stochastic behavior. Discrete samples a distribution on a schedule; Continuous evolve smoothly between events.",
    )
    selected_type = type_values[type_labels.index(selected_label)]

    # Seed params from initial (if type matches) or defaults
    defaults = get_default_driver_params(selected_type)
    if initial is not None and getattr(initial, "type", None) == selected_type:
        # Pull current values (best effort)
        for fld in ("name", "target_state_key", "value", "drift", "volatility", "initial_value",
                    "long_term_mean", "speed"):
            if hasattr(initial, fld):
                defaults[fld] = getattr(initial, fld)
        if hasattr(initial, "dist"):
            defaults["dist"] = initial.dist.model_dump(mode="json") if hasattr(initial.dist, "model_dump") else initial.dist
        if hasattr(initial, "timing"):
            defaults["timing"] = initial.timing.model_dump(mode="json") if hasattr(initial.timing, "model_dump") else initial.timing

    # Common fields
    name = st.text_input("Driver Name", value=defaults.get("name", "my_driver"), key=f"{key_prefix}_name")
    target_key = st.text_input(
        "Target State Key (what the simulation reads)",
        value=defaults.get("target_state_key", "rate"),
        key=f"{key_prefix}_target",
        help="Other parts of your model (loans, expenses, growth rules) read this key from simulation state.",
    )

    # Type-specific parameter widgets
    params: dict[str, Any] = {"type": selected_type, "name": name, "target_state_key": target_key}

    if selected_type == "discrete_rate":
        # Embed the distribution picker for the inner dist (beautiful reuse)
        from app.components.distribution_viz import render_distribution_picker

        st.caption("Distribution sampled on each timing tick (embedded)")
        dist = render_distribution_picker(
            key_prefix=f"{key_prefix}_dist",
            initial=defaults.get("dist"),
            show_save_section=False,
            height=320,
        )
        params["dist"] = dist.model_dump(mode="json") if hasattr(dist, "model_dump") else dist

        # Simple timing for MVP (Interval only; full AnyTiming can be added later)
        st.caption("Sampling Schedule (MVP: fixed interval)")
        days = st.number_input(
            "Interval (days)", min_value=1, max_value=3650, value=90, step=30, key=f"{key_prefix}_interval_days"
        )
        params["timing"] = {"type": "Interval", "interval": f"P{int(days)}D"}

    elif selected_type == "constant":
        params["value"] = st.number_input(
            "Constant Value", value=float(defaults.get("value", 1000.0)), step=100.0, key=f"{key_prefix}_value"
        )

    elif selected_type == "gbm_continuous":
        col1, col2, col3 = st.columns(3)
        with col1:
            params["drift"] = col1.number_input("Drift (annual)", value=float(defaults.get("drift", 0.08)), step=0.01, format="%.3f", key=f"{key_prefix}_drift")
        with col2:
            params["volatility"] = col2.number_input("Volatility (annual)", value=float(defaults.get("volatility", 0.18)), min_value=0.001, step=0.01, format="%.3f", key=f"{key_prefix}_vol")
        with col3:
            params["initial_value"] = col3.number_input("Initial Value", value=float(defaults.get("initial_value", 100000.0)), step=1000.0, key=f"{key_prefix}_init")

    elif selected_type == "mean_revert_continuous":
        col1, col2 = st.columns(2)
        with col1:
            params["long_term_mean"] = col1.number_input("Long-term Mean", value=float(defaults.get("long_term_mean", 0.03)), step=0.005, format="%.4f", key=f"{key_prefix}_ltm")
            params["speed"] = col1.number_input("Reversion Speed (theta)", value=float(defaults.get("speed", 0.8)), min_value=0.01, step=0.1, key=f"{key_prefix}_speed")
        with col2:
            params["volatility"] = col2.number_input("Volatility", value=float(defaults.get("volatility", 0.006)), min_value=0.0001, step=0.001, format="%.4f", key=f"{key_prefix}_vol2")
            params["initial_value"] = col2.number_input("Initial Value", value=float(defaults.get("initial_value", 0.03)), step=0.001, format="%.4f", key=f"{key_prefix}_init2")

    # Build the driver (validation happens inside create_)
    try:
        driver = create_external_driver(params)
    except Exception as e:
        st.error(f"Invalid driver configuration: {e}")
        # Fallback to a safe default
        driver = DiscreteRateDriver(
            name=name or "fallback",
            target_state_key=target_key or "rate",
            dist=NormalDistribution(mean=0.05, std=0.01),
            timing=IntervalTiming(interval=timedelta(days=90)),
        )

    # Live preview
    st.markdown("**Live Path Preview** (Monte Carlo samples of the driver over 12 months)")
    try:
        fig = plot_driver_preview(driver, n_paths=6, seed=42, height=height)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_preview")
    except Exception as ex:
        st.warning(f"Preview unavailable: {ex}")

    # Stats row
    try:
        stats = get_driver_stats(driver, n_paths=300, seed=123)
        cols = st.columns(5)
        cols[0].metric("Mean (terminal)", f"{stats['mean']:.4f}")
        cols[1].metric("Std Dev", f"{stats['std']:.4f}")
        cols[2].metric("5th %ile", f"{stats['p5']:.4f}")
        cols[3].metric("95th %ile", f"{stats['p95']:.4f}")
        cols[4].metric("Range", f"{stats['min']:.2f} → {stats['max']:.2f}")
    except Exception:
        pass

    # Optional save-to-library style callback (for future SavedDriver or just "use this driver")
    if show_save_section:
        with st.expander("Save / Reuse this driver definition"):
            st.caption("Drivers are saved inside the Scenario JSON. A reusable driver library can be added later (same pattern as distributions).")
            if st.button("Use this driver", key=f"{key_prefix}_use_btn"):
                if on_save_callback:
                    on_save_callback(driver)
                else:
                    st.json(driver.model_dump(mode="json"), expanded=False)
                    st.success("Driver configuration ready (copy the JSON above or use the returned object).")

    return driver


__all__ = [
    "render_external_driver_editor",
    "plot_driver_preview",
    "get_driver_stats",
    "DRIVER_TYPE_LABELS",
]
