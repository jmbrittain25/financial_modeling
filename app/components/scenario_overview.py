"""
Scenario Overview Component.

Provides a rich, visual summary of a ScenarioConfig so users can understand
the structure and "shape" of their scenario before running simulations.

Includes:
- Key stats (horizon, counts, risk profile)
- Plotly event schedule preview (deterministic sample using distribution means)
- Summary of stochastic elements
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from financial_simulator.core.distributions import AnyDistribution
from financial_simulator.scenarios import ScenarioConfig


def _get_mean_of_dist(dist: AnyDistribution) -> float:
    """Best-effort mean for preview purposes."""
    try:
        if hasattr(dist, "mean"):
            return float(dist.mean)
        if hasattr(dist, "value"):
            return float(dist.value)
        if hasattr(dist, "low") and hasattr(dist, "high"):
            # Uniform or Triangular approximation
            if hasattr(dist, "mode"):
                return float((dist.low + dist.mode + dist.high) / 3)
            return float((dist.low + dist.high) / 2)
        if hasattr(dist, "rate"):  # Exponential
            return float(1.0 / dist.rate)
        return 0.0
    except Exception:
        return 0.0


def _sample_deterministic_events(cfg: ScenarioConfig, max_events: int = 40) -> list[dict]:
    """
    Create a simplified deterministic preview of when events would fire
    and their approximate values (using means for distributions).
    """
    events = []
    start = cfg.start
    end = cfg.end

    for builder in cfg.event_builders[:max_events]:
        try:
            timing = builder.timing
            value_gen = builder.value_gen

            # Approximate value
            approx_value = 0.0
            if hasattr(value_gen, "value"):
                approx_value = float(value_gen.value)
            elif hasattr(value_gen, "dist") and value_gen.dist is not None:
                approx_value = _get_mean_of_dist(value_gen.dist)
            elif hasattr(value_gen, "initial_value"):
                approx_value = float(value_gen.initial_value)
            elif hasattr(value_gen, "amount"):
                approx_value = float(value_gen.amount)

            # Sample a few representative times
            t = start
            count = 0
            while t <= end and count < 6:
                events.append({
                    "time": t,
                    "value": approx_value,
                    "name": builder.name or builder.metadata.get("type", "event"),
                })
                # Advance roughly
                if hasattr(timing, "interval"):
                    t = t + timing.interval
                else:
                    t = t + (end - start) / 5
                count += 1
        except Exception:
            continue

    # Sort by time
    events.sort(key=lambda x: x["time"])
    return events[:max_events]


def render_scenario_overview(cfg: ScenarioConfig, key_prefix: str = "overview"):
    """Render a nice visual + statistical overview of a scenario."""

    st.markdown("### 🔍 Scenario Overview")

    # --- Basic Stats ---
    horizon_days = (cfg.end - cfg.start).days if cfg.end and cfg.start else 0
    horizon_years = round(horizon_days / 365.25, 1) if horizon_days > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Horizon", f"{horizon_years} years")
    col2.metric("Event Sources", len(cfg.event_builders))
    col3.metric("Continuous Processes", len(cfg.continuous_processes))
    col4.metric("External Drivers", len(cfg.external_drivers))
    col5.metric("Custom Metrics", len(cfg.custom_metrics))

    # Risk / complexity badge
    has_stochastic = any(
        hasattr(eb.value_gen, "dist") and eb.value_gen.dist is not None
        for eb in cfg.event_builders
    ) or len(cfg.external_drivers) > 0

    complexity = "Low"
    if len(cfg.event_builders) > 4 or has_stochastic:
        complexity = "Medium"
    if len(cfg.event_builders) > 8 or len(cfg.external_drivers) > 1:
        complexity = "High"

    st.caption(f"**Complexity:** {complexity}  |  **Stochastic elements:** {'Yes' if has_stochastic else 'No'}")

    # --- Event Schedule Preview (Plotly) ---
    st.markdown("**Event Schedule Preview** (approximate, using mean values)")

    preview_events = _sample_deterministic_events(cfg)

    if preview_events:
        times = [e["time"] for e in preview_events]
        values = [e["value"] for e in preview_events]
        names = [e["name"] for e in preview_events]

        # Convert to years from start
        t0 = times[0]
        x_years = [(t - t0).days / 365.25 for t in times]

        fig = go.Figure()

        # Scatter of events
        fig.add_trace(go.Scatter(
            x=x_years,
            y=values,
            mode="markers+text",
            marker=dict(size=10, color="#3b82f6"),
            text=names,
            textposition="top center",
            name="Approximate Events",
            hovertemplate="%{text}<br>Year %{x:.1f}<br>Value: %{y:,.0f}<extra></extra>",
        ))

        # Cumulative impact line (very rough)
        cum = np.cumsum(values)
        fig.add_trace(go.Scatter(
            x=x_years,
            y=cum,
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="Rough Cumulative Impact",
            hovertemplate="Cumulative: %{y:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            height=340,
            xaxis_title="Years from Start",
            yaxis_title="Approximate Value per Event",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No event builders to preview.")

    # --- Key State Variables ---
    if cfg.initial_state:
        st.markdown("**Initial State Variables**")
        cols = st.columns(min(len(cfg.initial_state), 5))
        for i, (k, v) in enumerate(list(cfg.initial_state.items())[:5]):
            cols[i].metric(k, f"{v:,.0f}" if isinstance(v, (int, float)) else str(v))

    # --- Stochastic Sources Summary ---
    stochastic_sources = []
    for eb in cfg.event_builders:
        if hasattr(eb.value_gen, "dist") and eb.value_gen.dist is not None:
            stochastic_sources.append(f"{eb.name or 'Event'}: {type(eb.value_gen.dist).__name__}")

    for drv in cfg.external_drivers:
        if hasattr(drv, "dist") and drv.dist is not None:
            stochastic_sources.append(f"Driver '{drv.name}': {type(drv.dist).__name__}")

    if stochastic_sources:
        with st.expander("Stochastic Sources in this Scenario"):
            for s in stochastic_sources:
                st.write(f"• {s}")

    # --- Quick Actions ---
    st.markdown("**Quick Actions**")
    c1, c2, c3 = st.columns(3)
    if c1.button("📋 Copy Scenario JSON", key=f"{key_prefix}_copy_json"):
        st.code(cfg.to_json(indent=2), language="json")
    if c2.button("🔄 Duplicate Scenario", key=f"{key_prefix}_duplicate"):
        st.session_state[f"{key_prefix}_duplicate_requested"] = True
    if c3.button("📤 Export as File", key=f"{key_prefix}_export"):
        st.download_button(
            "Download scenario.json",
            data=cfg.to_json(),
            file_name=f"{cfg.name.replace(' ', '_')}.json",
            mime="application/json",
            key=f"{key_prefix}_download",
        )


__all__ = ["render_scenario_overview"]
