"""
Results Dashboard — Rich post-simulation analysis using Plotly.

Key visualizations:
- Summary metric cards (mean/median/p5/p95)
- Outcome distribution histogram
- Time-series percentile bands (fan chart) for any state variable
- Custom metrics analysis
- Risk metrics via RiskAnalyzer (when available)
- Download options
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.scenario_io import build_results_bundle, export_scenario_json
from financial_simulator.analytics.risk import RiskAnalyzer
from financial_simulator.core.simulation import SimulationResult
from financial_simulator.scenarios import ScenarioConfig


def _extract_state_series(
    results: list[SimulationResult], key: str
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Return (times, p10/p50/p90 matrix, list of final values) for a state key."""
    all_times: set = set()
    series_by_sim: list[dict] = []
    final_values: list[float] = []

    for r in results:
        if not r.state_history:
            # Fallback to final state only
            val = r.final_state.get(key, 0.0)
            final_values.append(val)
            continue

        times = sorted(r.state_history.keys())
        vals = [r.state_history[t].get(key, 0.0) for t in times]
        series_by_sim.append(dict(zip(times, vals, strict=False)))
        all_times.update(times)
        final_values.append(r.final_state.get(key, vals[-1] if vals else 0.0))

    if not all_times:
        return np.array([]), np.array([]), final_values

    sorted_times = np.array(sorted(all_times))
    n = len(results)
    m = len(sorted_times)

    matrix = np.full((n, m), np.nan)
    for i, sim_series in enumerate(series_by_sim):
        for j, t in enumerate(sorted_times):
            if t in sim_series:
                matrix[i, j] = sim_series[t]

    # Percentiles (ignore NaNs)
    p10 = np.nanpercentile(matrix, 10, axis=0)
    p50 = np.nanpercentile(matrix, 50, axis=0)
    p90 = np.nanpercentile(matrix, 90, axis=0)

    return sorted_times, np.vstack([p10, p50, p90]), final_values


def plot_state_bands(
    results: list[SimulationResult],
    state_key: str = "cumulative_cash",
    title: str | None = None,
) -> go.Figure:
    """Create a nice percentile band (fan) chart for a state variable over time."""
    times, bands, finals = _extract_state_series(results, state_key)

    fig = go.Figure()

    if len(times) == 0:
        fig.add_annotation(text="No time-series data available for this key", showarrow=False)
        fig.update_layout(height=320, title=title or f"{state_key} — No Path Data")
        return fig

    # Convert times to years from start for cleaner x-axis
    t0 = times[0]
    x_years = [(t - t0).days / 365.25 for t in times]

    # Bands
    fig.add_trace(
        go.Scatter(
            x=x_years,
            y=bands[2],
            mode="lines",
            line=dict(color="rgba(59,130,246,0.2)", width=0),
            name="90th %ile",
            showlegend=False,
            hovertemplate="%{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_years,
            y=bands[0],
            mode="lines",
            line=dict(color="rgba(59,130,246,0.2)", width=0),
            name="10th %ile",
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.25)",
            showlegend=False,
            hovertemplate="%{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_years,
            y=bands[1],
            mode="lines",
            line=dict(color="#1e40af", width=2.5),
            name="Median",
            hovertemplate="%{y:.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title or f"{state_key} — Percentile Bands (10 / 50 / 90)",
        xaxis_title="Years from Start",
        yaxis_title=state_key,
        height=380,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def plot_outcome_histogram(
    final_values: list[float], title: str = "Distribution of Final Outcomes"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=40,
            marker_color="#3b82f6",
            opacity=0.75,
            name="Final Value",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Final Value",
        yaxis_title="Count",
        height=320,
        template="plotly_white",
        bargap=0.05,
    )
    return fig


def render_results_dashboard(
    results: list[SimulationResult],
    scenario_name: str = "Scenario",
    scenario_config: ScenarioConfig | None = None,
    run_summary: dict | None = None,
):
    """Main entry point for the rich results view."""
    if not results:
        st.warning("No results to display.")
        return

    st.header(f"📊 Analysis — {scenario_name}")
    st.caption(f"{len(results)} simulations completed")

    # --- 1. Final value statistics ---
    # Try to find the most interesting final value key
    candidate_keys = ["cumulative_cash", "portfolio_value", "cash", "home_value", "savings"]
    final_key = "cumulative_cash"
    for k in candidate_keys:
        if k in results[0].final_state:
            final_key = k
            break

    finals = []
    for r in results:
        val = r.final_state.get(final_key)
        if val is None:
            val = r.final_state.get("cash", 0)
        finals.append(float(val))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Final", f"${np.mean(finals):,.0f}")
    col2.metric("Median", f"${np.median(finals):,.0f}")
    col3.metric("5th %ile (Downside)", f"${np.percentile(finals, 5):,.0f}")
    col4.metric("95th %ile (Upside)", f"${np.percentile(finals, 95):,.0f}")

    # --- 2. Distribution ---
    st.plotly_chart(plot_outcome_histogram(finals), use_container_width=True)

    # --- 3. Time-series bands (the star feature) ---
    st.subheader("Path Analysis (Percentile Bands)")
    available_keys = set()
    for r in results[:5]:  # sample a few
        if r.state_history:
            available_keys.update(r.state_history[list(r.state_history.keys())[0]].keys())

    if available_keys:
        chosen_key = st.selectbox(
            "State variable to plot",
            sorted(available_keys),
            index=0
            if final_key not in available_keys
            else list(sorted(available_keys)).index(final_key),
            key="band_key_selector",
        )
        fig = plot_state_bands(results, state_key=chosen_key)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No detailed state history recorded in these runs (common for very simple scenarios)."
        )

    # --- 4. Custom Metrics (if present) ---
    if results and "__custom_metrics__" in results[0].final_state:
        st.subheader("Custom Metrics Summary")
        all_cm = [r.final_state.get("__custom_metrics__", {}) for r in results]
        if all_cm and all_cm[0]:
            keys = list(all_cm[0].keys())
            cols = st.columns(min(len(keys), 4))
            for i, k in enumerate(keys):
                vals = [m.get(k, 0) for m in all_cm if isinstance(m, dict)]
                if vals:
                    cols[i % 4].metric(
                        k, f"{np.mean(vals):,.2f}", f"avg across {len(results)} sims"
                    )

    # --- 5. Risk Analysis (best effort) ---
    try:
        analyzer = RiskAnalyzer()
        risk = analyzer.compute_var(np.array(finals), 0.95)
        st.caption(f"VaR 95%: ${risk:,.0f} (simple historical)")
    except Exception:
        pass

    # --- 6. Exports ---
    st.divider()
    st.subheader("Export")

    export_rows = []
    for i, r in enumerate(results):
        row = {"sim_idx": i, "final_value": float(finals[i]) if i < len(finals) else None}
        cm = r.final_state.get("__custom_metrics__", {}) if isinstance(r.final_state, dict) else {}
        for k, v in cm.items():
            if isinstance(v, (int, float)):
                row[f"custom:{k}"] = float(v)
        for state_k, state_v in (r.final_state or {}).items():
            if state_k != "__custom_metrics__" and isinstance(state_v, (int, float)):
                row[f"final:{state_k}"] = float(state_v)
        export_rows.append(row)
    export_df = pd.DataFrame(export_rows)

    safe_name = scenario_name.replace(" ", "_")
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Download results (CSV)",
            data=csv_bytes,
            file_name=f"{safe_name}_results.csv",
            mime="text/csv",
            key=f"dl_csv_{safe_name[:20]}",
            use_container_width=True,
        )
    with dl2:
        if scenario_config is not None:
            st.download_button(
                "Download scenario config (JSON)",
                data=export_scenario_json(scenario_config),
                file_name=f"{safe_name}_config.json",
                mime="application/json",
                key=f"dl_cfg_{safe_name[:20]}",
                use_container_width=True,
            )
    with dl3:
        if scenario_config is not None:
            summary_payload = {
                **(run_summary or {}),
                "scenario_name": scenario_name,
                "n_results": len(results),
                "mean_final": float(np.mean(finals)),
                "median_final": float(np.median(finals)),
                "p5_final": float(np.percentile(finals, 5)),
                "p95_final": float(np.percentile(finals, 95)),
            }
            st.download_button(
                "Download full bundle (JSON)",
                data=build_results_bundle(scenario_config, summary_payload),
                file_name=f"{safe_name}_bundle.json",
                mime="application/json",
                key=f"dl_bundle_{safe_name[:20]}",
                use_container_width=True,
            )

    custom_cols = [c for c in export_df.columns if c.startswith("custom:")]
    st.caption(
        f"CSV: {len(export_df)} simulations, {len(custom_cols)} custom metric(s). "
        "JSON bundle includes the scenario config used for this run plus summary statistics."
    )


__all__ = [
    "render_results_dashboard",
    "plot_state_bands",
    "plot_outcome_histogram",
]
