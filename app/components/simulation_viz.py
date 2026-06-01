"""
Rich interactive visualization helpers for Monte Carlo simulation results.

Pure functions (no Streamlit imports) so this module is:
- Easily testable
- Reusable outside the main app (CLI reports, notebooks, API)
- Fast to import

Core responsibilities:
- Discover numeric fields across final_state + custom metrics + histories
- Build per-simulation summary DataFrames (for tables + filtering)
- Align irregular state_history paths to a common time grid (for quantiles/fans)
- Compute quantile bands and per-path statistics (incl. max drawdown)
- Generate beautiful, fast Plotly figures:
    - Spaghetti + fan charts (selected sims highlighted)
    - Cross-sectional distributions at arbitrary times
    - Custom scatter / correlation plots

All plots are designed for N=50..2000 simulations and 30..400 time points.
They follow the styling conventions already used in distribution_viz.py.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from financial_simulator.core.simulation import SimulationResult

# =============================================================================
# Field discovery & summary construction
# =============================================================================


def _is_numeric_scalar(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _flatten_custom_metrics(final_state: dict[str, Any]) -> dict[str, float]:
    """Extract custom metrics into top-level keys with 'custom:' prefix."""
    cm = final_state.get("__custom_metrics__", {})
    if not isinstance(cm, dict):
        return {}
    return {f"custom:{k}": float(v) for k, v in cm.items() if _is_numeric_scalar(v)}


def discover_numeric_keys(results: list[SimulationResult], sample: int = 20) -> list[str]:
    """
    Return a sorted list of all numeric keys that appear in final_state
    (including flattened custom metrics) across the result set.

    Uses a sample for speed on very large N; falls back to full scan if needed.
    Prioritizes "obvious" wealth-like keys in the returned order (stable sort).
    """
    if not results:
        return []

    keys: set[str] = set()
    n = min(sample, len(results))
    for r in results[:n]:
        for k, v in r.final_state.items():
            if k == "__custom_metrics__":
                continue
            if _is_numeric_scalar(v):
                keys.add(k)
        keys.update(_flatten_custom_metrics(r.final_state).keys())

    # Also peek at a bit of history for time-series candidates (first result is representative)
    if results:
        for _t, state in list(results[0].state_history.items())[:5]:
            for k, v in state.items():
                if _is_numeric_scalar(v):
                    keys.add(k)

    ordered = sorted(keys)

    # Promote wealth-like keys to the front for nicer UX
    priority_substrings = ("cash", "value", "portfolio", "wealth", "balance", "home")
    promoted = [k for k in ordered if any(p in k.lower() for p in priority_substrings)]
    rest = [k for k in ordered if k not in promoted]
    return promoted + rest


def build_summary_dataframe(
    results: list[SimulationResult],
    primary_keys: list[str] | None = None,
) -> pd.DataFrame:
    """
    One row per simulation with:
    - sim_idx (0..N-1)
    - final_<key> for every numeric final + custom metric
    - path_min / path_max / path_mean for the first primary key (if present in history)
    - max_drawdown on the primary path (if computable)
    - n_events
    """
    if not results:
        return pd.DataFrame()

    if primary_keys is None:
        primary_keys = discover_numeric_keys(results)[:3]

    records: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        rec: dict[str, Any] = {"sim_idx": i}

        # Final state scalars + customs
        for k, v in r.final_state.items():
            if k == "__custom_metrics__":
                continue
            if _is_numeric_scalar(v):
                rec[f"final_{k}"] = float(v)
        rec.update(_flatten_custom_metrics(r.final_state))

        # Path statistics on the first usable primary key
        primary = None
        for pk in primary_keys:
            if pk in (r.final_state or {}):
                primary = pk
                break
            # try history
            for state in r.state_history.values():
                if pk in state and _is_numeric_scalar(state[pk]):
                    primary = pk
                    break
            if primary:
                break

        if primary:
            path_vals: list[float] = []
            for state in r.state_history.values():
                if primary in state and _is_numeric_scalar(state[primary]):
                    path_vals.append(float(state[primary]))
            if path_vals:
                arr = np.asarray(path_vals, dtype=float)
                rec[f"path_min_{primary}"] = float(np.min(arr))
                rec[f"path_max_{primary}"] = float(np.max(arr))
                rec[f"path_mean_{primary}"] = float(np.mean(arr))
                rec[f"max_drawdown_{primary}"] = compute_path_drawdown(arr)

        rec["n_events"] = len(r.events)
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    # Ensure sim_idx is first and integer
    if "sim_idx" in df.columns:
        cols = ["sim_idx"] + [c for c in df.columns if c != "sim_idx"]
        df = df[cols]
    return df


def compute_path_drawdown(values: np.ndarray) -> float:
    """Maximum drawdown (peak-to-trough) as a positive fraction (0.0 = no drawdown)."""
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / np.where(peak > 0, peak, 1.0)
    return float(np.max(drawdowns))


# =============================================================================
# Path alignment for quantile / fan calculations
# =============================================================================


def align_paths_to_grid(
    results: list[SimulationResult],
    key: str,
    freq: str = "MS",
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Return a wide DataFrame indexed by a regular datetime grid.

    Columns are simulation indices (0..N-1). Values are ffilled (or nearest)
    from each simulation's state_history for the requested key.

    Missing keys in a simulation become NaN (quantile calculations use skipna).
    """
    if not results:
        return pd.DataFrame()

    # Determine global time span from all histories (robust even for irregular timings)
    all_times: list[dt.datetime] = []
    for r in results:
        all_times.extend(r.state_history.keys())
    if not all_times:
        # Fallback: just start/end
        all_times = [r.start for r in results] + [r.end for r in results]

    min_t = min(all_times)
    max_t = max(all_times)

    # Regular grid
    try:
        idx = pd.date_range(start=min_t, end=max_t, freq=freq)
    except Exception:
        # Extremely short or weird horizons
        idx = pd.DatetimeIndex(sorted(set(all_times)))

    data: dict[int, pd.Series] = {}
    for i, r in enumerate(results):
        if not r.state_history:
            continue
        # Build per-sim series
        times = []
        vals = []
        for t, state in r.state_history.items():
            if key in state and _is_numeric_scalar(state[key]):
                times.append(t)
                vals.append(float(state[key]))
        if not times:
            continue
        s = pd.Series(vals, index=pd.to_datetime(times)).sort_index()
        aligned = s.reindex(idx, method=method if method in ("ffill", "bfill", "nearest") else None)
        data[i] = aligned

    if not data:
        return pd.DataFrame(index=idx)

    df = pd.DataFrame(data, index=idx)
    df.index.name = "time"
    return df


def compute_quantile_bands(
    aligned_df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
) -> dict[float, pd.Series]:
    """Column-wise quantiles across simulations at each time step (skipna)."""
    if aligned_df.empty:
        return {}
    bands: dict[float, pd.Series] = {}
    for q in quantiles:
        # pandas 2/3 compatible: numeric_only=True + let quantile handle NaNs (default behavior in recent pandas)
        bands[q] = aligned_df.quantile(q, axis=1, numeric_only=True)
    return bands


# =============================================================================
# Per-result inspection helpers
# =============================================================================


def get_state_at_time(
    result: SimulationResult, target: dt.datetime
) -> tuple[dt.datetime | None, dict[str, Any]]:
    """
    Return (closest_time, state_dict) for the nearest snapshot in state_history.
    If history is empty, returns (None, final_state).
    """
    if not result.state_history:
        return None, dict(result.final_state)

    times = sorted(result.state_history.keys())
    # Find nearest
    closest = min(times, key=lambda t: abs((t - target).total_seconds()))
    return closest, dict(result.state_history[closest])


# =============================================================================
# Plotly figure factories (beautiful, fast, decision-oriented)
# =============================================================================


def _financial_colorway() -> list[str]:
    return ["#1e40af", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#1f2937", "#6b7280"]


def create_spaghetti_plot(
    results: list[SimulationResult],
    key: str,
    active_mask: np.ndarray | None = None,
    selected_indices: list[int] | None = None,
    max_background: int = 80,
    title: str | None = None,
    height: int = 460,
) -> go.Figure:
    """
    Main interactive time-series plot.

    - Very faint background lines for a subsample of active (non-selected) sims
    - Rich quantile fan (p5-p95, p25-p75, median) always visible
    - Selected simulations drawn as thick, saturated, hover-rich lines
    - Range slider + clean financial styling
    """
    if active_mask is None:
        active_mask = np.ones(len(results), dtype=bool)
    if selected_indices is None:
        selected_indices = []

    active_idx = [i for i, ok in enumerate(active_mask) if ok]
    selected_set = set(selected_indices)

    # Align for reliable fans + easy subsampling
    aligned = align_paths_to_grid([results[i] for i in active_idx], key=key, freq="MS")
    if aligned.empty:
        # Fallback: try raw per-result plotting (very short histories)
        aligned = None

    fig = go.Figure()

    # --- Quantile fan layers (always fast, always useful) ---
    if aligned is not None and not aligned.empty and len(aligned) >= 2:
        try:
            bands = compute_quantile_bands(aligned, (0.05, 0.25, 0.50, 0.75, 0.95))
            if 0.50 in bands and not bands[0.50].empty:
                times = bands[0.50].index

                # p5-p95 band (lightest) - only if we have the outer quantiles
                if 0.95 in bands and 0.05 in bands and not bands[0.95].empty and not bands[0.05].empty:
                    fig.add_trace(
                        go.Scatter(
                            x=list(times) + list(reversed(times)),
                            y=list(bands[0.95]) + list(reversed(bands[0.05])),
                            fill="toself",
                            fillcolor="rgba(147, 197, 253, 0.18)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="5-95% band",
                            hoverinfo="skip",
                        )
                    )
                # p25-p75 band
                if 0.75 in bands and 0.25 in bands and not bands[0.75].empty and not bands[0.25].empty:
                    fig.add_trace(
                        go.Scatter(
                            x=list(times) + list(reversed(times)),
                            y=list(bands[0.75]) + list(reversed(bands[0.25])),
                            fill="toself",
                            fillcolor="rgba(59, 130, 246, 0.28)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="25-75% band",
                            hoverinfo="skip",
                        )
                    )
                # Median (prominent)
                if 0.50 in bands and not bands[0.50].empty:
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=bands[0.50],
                            mode="lines",
                            line=dict(color="#1e3a8a", width=2.5, dash="solid"),
                            name="Median path",
                            hovertemplate="Median<br>%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
                        )
                    )
        except Exception:
            # Extremely small or degenerate aligned data — skip fans gracefully
            pass

    # --- Background spaghetti (subsampled active, non-selected) ---
    bg_candidates = [i for i in active_idx if i not in selected_set]
    if len(bg_candidates) > max_background:
        rng = np.random.default_rng(42)
        bg_indices = sorted(rng.choice(bg_candidates, size=max_background, replace=False))
    else:
        bg_indices = bg_candidates

    for i in bg_indices:
        r = results[i]
        times = []
        vals = []
        for t, state in sorted(r.state_history.items()):
            if key in state and _is_numeric_scalar(state[key]):
                times.append(t)
                vals.append(float(state[key]))
        if len(times) < 2:
            continue

        # Better handling for very short histories (finishes plan item)
        mode = "lines" if len(times) >= 3 else "markers"
        marker_size = 3 if len(times) < 3 else None
        line_width = 0.8 if len(times) < 3 else 0.6

        fig.add_trace(
            go.Scatter(
                x=times,
                y=vals,
                mode=mode,
                line=dict(color="#9ca3af", width=line_width),
                marker=dict(size=marker_size, color="#9ca3af") if marker_size else None,
                opacity=0.18,
                name=f"Sim {i}",
                hovertemplate=f"Sim {i}<br>%{{x|%Y-%m-%d}}<br>%{{y:,.0f}}<extra></extra>",
                showlegend=False,
            )
        )

    # --- Selected simulations (thick, vivid, hover on) ---
    for i in selected_indices:
        if i >= len(results):
            continue
        r = results[i]
        times = []
        vals = []
        for t, state in sorted(r.state_history.items()):
            if key in state and _is_numeric_scalar(state[key]):
                times.append(t)
                vals.append(float(state[key]))
        if not times:
            continue

        # Better handling for very short histories (finishes plan item)
        mode = "lines+markers" if len(times) >= 3 else "markers"
        marker_size = 6 if len(times) < 3 else 4
        line_width = 1.5 if len(times) < 3 else 2.8

        fig.add_trace(
            go.Scatter(
                x=times,
                y=vals,
                mode=mode,
                line=dict(color="#1e40af", width=line_width),
                marker=dict(size=marker_size, color="#1e40af"),
                name=f"Sim {i} (selected)",
                hovertemplate=f"<b>Sim {i} (selected)</b><br>%{{x|%Y-%m-%d}}<br>%{{y:,.0f}}<extra></extra>",
            )
        )

    # Layout
    fig_title = title or f"Paths — {key} ({len(active_idx)} active, {len(selected_indices)} highlighted)"
    fig.update_layout(
        title=fig_title,
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(buttons=[
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ])),
        yaxis_title=key,
        colorway=_financial_colorway(),
    )
    return fig


def create_fan_chart(
    aligned_df: pd.DataFrame,
    key: str,
    title: str | None = None,
    height: int = 380,
) -> go.Figure:
    """Standalone fan chart (useful when you already have an aligned grid)."""
    if aligned_df.empty:
        return go.Figure().update_layout(title="No path data for fan chart", height=height)

    bands = compute_quantile_bands(aligned_df, (0.05, 0.25, 0.50, 0.75, 0.95))
    times = bands[0.50].index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times) + list(reversed(times)),
        y=list(bands[0.95]) + list(reversed(bands[0.05])),
        fill="toself", fillcolor="rgba(147, 197, 253, 0.22)",
        line=dict(color="rgba(0,0,0,0)"), name="5-95%", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(times) + list(reversed(times)),
        y=list(bands[0.75]) + list(reversed(bands[0.25])),
        fill="toself", fillcolor="rgba(59, 130, 246, 0.35)",
        line=dict(color="rgba(0,0,0,0)"), name="25-75%", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=bands[0.50], mode="lines",
        line=dict(color="#1e3a8a", width=2.8), name="Median",
    ))

    fig.update_layout(
        title=title or f"Distribution over time — {key}",
        template="plotly_white",
        height=height,
        hovermode="x unified",
        yaxis_title=key,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    return fig


def create_cross_section_plot(
    aligned_df: pd.DataFrame,
    target_time: dt.datetime,
    key: str,
    title: str | None = None,
    height: int = 280,
) -> go.Figure:
    """Histogram of the metric value across simulations at (nearest) target_time."""
    if aligned_df.empty:
        return go.Figure().update_layout(title="No data", height=height)

    # Find nearest column index
    times = aligned_df.index
    nearest = min(times, key=lambda t: abs((t - target_time).total_seconds()))
    values = aligned_df.loc[nearest].dropna().values

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=35, name="Active simulations",
        marker_color="#3b82f6", opacity=0.75,
    ))
    fig.add_vline(x=float(np.median(values)), line=dict(color="#1e3a8a", width=2, dash="dash"),
                  annotation_text="median", annotation_position="top right")
    fig.update_layout(
        title=title or f"{key} distribution at {nearest:%Y-%m-%d}",
        template="plotly_white",
        height=height,
        bargap=0.04,
        xaxis_title=key,
        yaxis_title="Count",
    )
    return fig


def create_custom_scatter(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    selected_mask: np.ndarray | None = None,
    color_col: str | None = None,
    title: str | None = None,
    height: int = 380,
) -> go.Figure:
    """Scatter of two final (or path-derived) metrics. Supports highlighting and optional color dimension."""
    if summary_df.empty or x_col not in summary_df.columns or y_col not in summary_df.columns:
        return go.Figure().update_layout(title="Select two numeric columns", height=height)

    if selected_mask is None:
        selected_mask = np.zeros(len(summary_df), dtype=bool)

    fig = go.Figure()

    if color_col and color_col in summary_df.columns:
        # Color all points by the chosen dimension (selected still get emphasis)
        colors = summary_df[color_col]
        fig.add_trace(go.Scatter(
            x=summary_df[x_col],
            y=summary_df[y_col],
            mode="markers",
            marker=dict(
                color=colors,
                colorscale="Viridis",
                size=6,
                opacity=0.75,
                colorbar=dict(title=color_col, thickness=10),
            ),
            name="All active",
            hovertemplate=f"%{{x:,.0f}} / %{{y:,.0f}}<br>{color_col}: %{{marker.color:.2f}}<extra></extra>",
        ))
        # Overlay selected on top with outline
        if selected_mask.any():
            fig.add_trace(go.Scatter(
                x=summary_df.loc[selected_mask, x_col],
                y=summary_df.loc[selected_mask, y_col],
                mode="markers",
                marker=dict(color="#1e40af", size=9, line=dict(width=1.5, color="white")),
                name="Highlighted",
                hovertemplate="<b>Selected</b><br>%{x:,.0f} / %{y:,.0f}<extra></extra>",
            ))
    else:
        # Original behavior (no color dimension)
        non = ~selected_mask
        fig.add_trace(go.Scatter(
            x=summary_df.loc[non, x_col],
            y=summary_df.loc[non, y_col],
            mode="markers",
            marker=dict(color="#9ca3af", size=5, opacity=0.6),
            name="Active (not highlighted)",
            hovertemplate="%{x:,.0f} / %{y:,.0f}<extra></extra>",
        ))
        if selected_mask.any():
            fig.add_trace(go.Scatter(
                x=summary_df.loc[selected_mask, x_col],
                y=summary_df.loc[selected_mask, y_col],
                mode="markers",
                marker=dict(color="#1e40af", size=8, line=dict(width=1, color="white")),
                name="Highlighted",
                hovertemplate="<b>Selected</b><br>%{x:,.0f} / %{y:,.0f}<extra></extra>",
            ))

    fig.update_layout(
        title=title or f"{x_col} vs {y_col}" + (f" (colored by {color_col})" if color_col else ""),
        template="plotly_white",
        height=height,
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


def create_correlation_heatmap(
    summary_df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    title: str = "Correlation of final & path metrics (active set)",
    height: int = 420,
) -> go.Figure:
    """Plotly heatmap of correlations among numeric columns in the summary."""
    if numeric_cols is None:
        numeric_cols = [c for c in summary_df.columns if c != "sim_idx" and pd.api.types.is_numeric_dtype(summary_df[c])]
    if len(numeric_cols) < 2:
        return go.Figure().update_layout(title="Need at least 2 numeric columns", height=height)

    corr = summary_df[numeric_cols].corr(numeric_only=True)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="corr"),
        hovertemplate="%{x}<br>%{y}<br>ρ=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=120, r=20, t=50, b=30),
    )
    return fig


# =============================================================================
# Optional high-level Streamlit render entrypoint (for reuse / future multipage)
# =============================================================================


def render_simulation_analysis(
    results: list[SimulationResult],
    *,
    key_prefix: str = "sim_viz",
    default_metric: str | None = None,
    height: int = 460,
) -> None:
    """
    Self-contained rich results visualization panel.

    Call this from any Streamlit page that has a list[SimulationResult] in hand.
    It renders:
    - Auto-discovered filters + active count
    - Sortable multi-select dataframe (native Streamlit row selection)
    - Quick-pick highlight buttons
    - Spaghetti + fan charts with linked selection
    - Time scrubber + value-based path highlighter (the "click any point" UX)
    - Custom plot builder (scatter / heatmap)

    This is the single-function drop-in for the full experience.
    For maximum layout control, import the lower-level helpers and compose yourself.
    """
    import streamlit as st

    if not results:
        st.info("No simulation results to visualize.")
        return


    # Lightweight but more complete caching (finishes plan item)
    results_key = f"{key_prefix}_prep_{id(results)}_{len(results)}"
    if results_key in st.session_state:
        cached = st.session_state[results_key]
        all_keys = cached["all_keys"]
        summary = cached["summary"]
        primary = cached["primary"]
        aligned_cache = cached.get("aligned", {})
        common_times = cached.get("common_times", None)
    else:
        all_keys = discover_numeric_keys(results)
        primary = default_metric or (all_keys[0] if all_keys else "cumulative_cash")
        summary = build_summary_dataframe(results, primary_keys=[primary] if primary else None)
        aligned_cache = {}
        common_times = None
        st.session_state[results_key] = {
            "all_keys": all_keys,
            "summary": summary,
            "primary": primary,
            "aligned": aligned_cache,
            "common_times": common_times,
        }

    st.caption(f"Discovered {len(all_keys)} numeric metrics across {len(results)} simulations.")

    # --- Filters (simple but effective) ---
    with st.expander("🔍 Filters — narrow the active set", expanded=False):
        active_mask = np.ones(len(results), dtype=bool)
        # Pick 1-2 high-signal final columns for sliders
        candidate_cols = [c for c in summary.columns if c.startswith("final_") or c.startswith("custom:")]

        # Quick filter buttons (finishes plan item)
        qf1, qf2, qf3, qf4 = st.columns(4)
        if qf1.button("Final > 0", key=f"{key_prefix}_qf_pos"):
            st.session_state[f"{key_prefix}_quick_filter"] = "positive"
        if qf2.button("Top 10%", key=f"{key_prefix}_qf_top"):
            st.session_state[f"{key_prefix}_quick_filter"] = "top10"
        if qf3.button("Bottom 10%", key=f"{key_prefix}_qf_bot"):
            st.session_state[f"{key_prefix}_quick_filter"] = "bottom10"
        if qf4.button("Reset filters", key=f"{key_prefix}_qf_reset"):
            st.session_state[f"{key_prefix}_quick_filter"] = None
            for col in candidate_cols[:3]:
                if f"{key_prefix}_filter_{col}" in st.session_state:
                    del st.session_state[f"{key_prefix}_filter_{col}"]

        quick_filter = st.session_state.get(f"{key_prefix}_quick_filter")

        for col in candidate_cols[:3]:
            lo, hi = float(summary[col].min()), float(summary[col].max())
            if lo == hi:
                continue
            default = (lo, hi)

            # Apply quick filter presets if active
            if quick_filter == "positive" and "cash" in col.lower() or "value" in col.lower():
                default = (max(lo, 0.0), hi)
            elif quick_filter == "top10":
                p90 = float(summary[col].quantile(0.9))
                default = (p90, hi)
            elif quick_filter == "bottom10":
                p10 = float(summary[col].quantile(0.1))
                default = (lo, p10)

            sel = st.slider(
                f"{col}",
                min_value=lo,
                max_value=hi,
                value=default,
                key=f"{key_prefix}_filter_{col}",
                format="%.0f" if hi > 100 else "%.3f",
            )
            active_mask &= (summary[col].values >= sel[0]) & (summary[col].values <= sel[1])

        if quick_filter:
            st.caption(f"Quick filter active: **{quick_filter}** (use Reset to clear)")

        st.write(f"**{active_mask.sum()} / {len(results)}** simulations active after filters")

    active_summary = summary[active_mask].copy()
    active_indices = active_summary["sim_idx"].tolist() if not active_summary.empty else []

    # --- Selection via dataframe + quick picks ---
    st.subheader("Simulation Browser & Highlights")
    st.caption("Select rows (or use quick-pick buttons) to highlight their paths in the charts below.")

    quick_cols = st.columns(6)
    if quick_cols[0].button("5 Best (final)", key=f"{key_prefix}_best"):
        if not active_summary.empty:
            # Pick the first final_* column that looks like a wealth metric
            wealth_cols = [c for c in active_summary.columns if c.startswith("final_") and any(w in c.lower() for w in ("cash", "value", "portfolio", "wealth"))]
            final_cols = [c for c in active_summary.columns if c.startswith("final_")]
            sort_col = wealth_cols[0] if wealth_cols else (final_cols[0] if final_cols else None)
            if sort_col:
                top = active_summary.nlargest(5, sort_col)["sim_idx"].tolist()
                st.session_state[f"{key_prefix}_selected"] = top
                st.rerun()
    if quick_cols[1].button("5 Worst", key=f"{key_prefix}_worst"):
        if not active_summary.empty:
            wealth_cols = [c for c in active_summary.columns if c.startswith("final_") and any(w in c.lower() for w in ("cash", "value", "portfolio", "wealth"))]
            final_cols = [c for c in active_summary.columns if c.startswith("final_")]
            sort_col = wealth_cols[0] if wealth_cols else (final_cols[0] if final_cols else None)
            if sort_col:
                bot = active_summary.nsmallest(5, sort_col)["sim_idx"].tolist()
                st.session_state[f"{key_prefix}_selected"] = bot
                st.rerun()
    if quick_cols[2].button("Clear highlights", key=f"{key_prefix}_clear"):
        st.session_state[f"{key_prefix}_selected"] = []
        st.rerun()
    if quick_cols[3].button("Random 8", key=f"{key_prefix}_rand"):
        if active_indices:
            rng = np.random.default_rng(42)
            st.session_state[f"{key_prefix}_selected"] = sorted(rng.choice(active_indices, size=min(8, len(active_indices)), replace=False).tolist())
            st.rerun()
    if quick_cols[4].button("5 Median", key=f"{key_prefix}_median"):
        if not active_summary.empty:
            wealth_cols = [c for c in active_summary.columns if c.startswith("final_") and any(w in c.lower() for w in ("cash", "value", "portfolio", "wealth"))]
            final_cols = [c for c in active_summary.columns if c.startswith("final_")]
            sort_col = wealth_cols[0] if wealth_cols else (final_cols[0] if final_cols else None)
            if sort_col:
                med_idx = active_summary[sort_col].sort_values().index[len(active_summary) // 2 - 2 : len(active_summary) // 2 + 3]
                med_sims = active_summary.loc[med_idx, "sim_idx"].astype(int).tolist()
                st.session_state[f"{key_prefix}_selected"] = sorted(set(med_sims))
                st.rerun()

    selected = st.session_state.get(f"{key_prefix}_selected", [])

    # The powerful dataframe selector
    if not active_summary.empty:
        display_cols = [c for c in active_summary.columns if c in ("sim_idx", "n_events") or c.startswith("final_") or c.startswith("path_") or c.startswith("custom:")][:9]
        event = st.dataframe(
            active_summary[display_cols],
            key=f"{key_prefix}_browser",
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
        )
        sel_rows = event.get("selection", {}).get("rows", []) if event else []
        if sel_rows:
            newly = [int(active_summary.iloc[r]["sim_idx"]) for r in sel_rows]
            # merge without dups
            merged = sorted(set(selected) | set(newly))
            if merged != selected:
                st.session_state[f"{key_prefix}_selected"] = merged
                selected = merged
                st.rerun()
    else:
        st.warning("No simulations match current filters.")

    # --- Main visualizations ---
    st.subheader("Interactive Time Series & Distributions Over Time")

    # Performance + visualization style controls (finishes items from original plan)
    n_sims = len(results)
    auto_reduced = False
    suggested_bg = 80
    if n_sims > 800:
        suggested_bg = max(30, int(80000 / n_sims))  # heuristic: keep total points reasonable
        auto_reduced = True

    with st.expander("⚙️ Display settings (performance & detail)", expanded=False):
        perf_col, fan_col = st.columns(2)
        max_bg = perf_col.slider(
            "Max background simulation lines",
            min_value=10,
            max_value=200,
            value=min(suggested_bg, 80) if auto_reduced else 80,
            step=10,
            key=f"{key_prefix}_max_bg",
            help="Lower this for faster rendering with very large Monte Carlo runs (1000+).",
        )
        fan_style = fan_col.radio(
            "Fan chart detail",
            options=["Full (5-95% + 25-75%)", "Interquartile focus (25-75% only)"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_fan_style",
        )

    if auto_reduced:
        st.caption(f"Large run detected ({n_sims} simulations). Background lines auto-reduced for performance. Adjust the slider above if needed.")

    # Metric chooser (prefers wealth-like)
    metric_options = [k for k in all_keys if any(w in k.lower() for w in ("cash", "value", "portfolio"))] or all_keys[:5]
    chosen_key = st.selectbox(
        "Primary metric for paths & fans",
        options=metric_options or [primary],
        index=0,
        key=f"{key_prefix}_metric",
    )

    col1, col2 = st.columns(2)
    with col1:
        spaghetti = create_spaghetti_plot(
            results,
            key=chosen_key,
            active_mask=active_mask,
            selected_indices=selected,
            max_background=max_bg,
            height=height,
        )
        st.plotly_chart(spaghetti, width="stretch", key=f"{key_prefix}_spaghetti")

    with col2:
        # Fan on the same aligned grid
        aligned_for_fan = align_paths_to_grid([results[i] for i in active_indices], key=chosen_key)
        fan = create_fan_chart(aligned_for_fan, key=chosen_key, height=height - 60)
        # If user chose interquartile focus, we hide the outer band by re-creating a lighter fan
        if fan_style.startswith("Interquartile"):
            # Rebuild a simpler fan for focus
            bands = compute_quantile_bands(aligned_for_fan, (0.25, 0.50, 0.75))
            if not aligned_for_fan.empty and 0.50 in bands:
                times = bands[0.50].index
                fan = go.Figure()
                fan.add_trace(go.Scatter(
                    x=list(times) + list(reversed(times)),
                    y=list(bands[0.75]) + list(reversed(bands[0.25])),
                    fill="toself", fillcolor="rgba(59, 130, 246, 0.35)",
                    line=dict(color="rgba(0,0,0,0)"), name="25-75%", hoverinfo="skip",
                ))
                fan.add_trace(go.Scatter(
                    x=times, y=bands[0.50], mode="lines",
                    line=dict(color="#1e3a8a", width=2.8), name="Median",
                ))
                fan.update_layout(
                    title=f"Distribution over time — {chosen_key} (interquartile focus)",
                    template="plotly_white",
                    height=height - 60,
                    hovermode="x unified",
                    yaxis_title=chosen_key,
                )
        st.plotly_chart(fan, width="stretch", key=f"{key_prefix}_fan")

    # --- Time Scrubber (the practical "click on a point" experience) ---
    st.subheader("🔎 Time Scrubber — Inspect & Highlight Paths at Any Moment")
    st.caption("This is the closest equivalent to clicking a data point on the time-series plot. Choose a moment, then highlight the simulations that were near a chosen value at that exact time.")

    if active_indices:
        aligned = align_paths_to_grid([results[i] for i in active_indices], key=chosen_key)
        if not aligned.empty:
            common_times = [t.to_pydatetime() for t in aligned.index[:: max(1, len(aligned) // 18)]]
            chosen_t = st.select_slider(
                "Inspect time:",
                options=common_times,
                value=common_times[len(common_times) // 2] if common_times else common_times[0],
                key=f"{key_prefix}_scrub_time",
                format_func=lambda d: d.strftime("%Y-%m-%d"),
            )

            cross = create_cross_section_plot(aligned, chosen_t, chosen_key, height=240)
            st.plotly_chart(cross, width="stretch", key=f"{key_prefix}_cross")

            # Value highlighter — use the exact nearest time that the cross-section used
            nearest_ts = min(aligned.index, key=lambda t: abs((t.to_pydatetime() - chosen_t).total_seconds()))
            vals_at_t = aligned.loc[nearest_ts].dropna()
            if len(vals_at_t) > 1:
                vmin, vmax = float(vals_at_t.min()), float(vals_at_t.max())
                target = st.number_input(
                    f"Target {chosen_key} value at {chosen_t:%Y-%m-%d}",
                    value=float(np.median(vals_at_t)),
                    min_value=vmin,
                    max_value=vmax,
                    step=max(1.0, (vmax - vmin) / 200),
                    key=f"{key_prefix}_target_val",
                )
                tol_pct = st.slider("Tolerance (± %)", 1, 25, 8, key=f"{key_prefix}_tol")
                if st.button("Highlight paths near this value at this time", key=f"{key_prefix}_highlight_near"):
                    lower = target * (1 - tol_pct / 100)
                    upper = target * (1 + tol_pct / 100)
                    matches = [int(i) for i, v in vals_at_t.items() if lower <= v <= upper]
                    # Map back to global sim indices (active_indices are the positions)
                    global_matches = [active_indices[m] for m in matches]
                    merged = sorted(set(selected) | set(global_matches))
                    st.session_state[f"{key_prefix}_selected"] = merged
                    st.success(f"Added {len(global_matches)} matching simulations to highlights.")
                    st.rerun()

    # --- Custom Plot Builder ---
    with st.expander("🛠️ Custom Plot Builder — compare any two metrics", expanded=False):
        if not active_summary.empty:
            num_cols = [c for c in active_summary.columns if c != "sim_idx" and pd.api.types.is_numeric_dtype(active_summary[c])]
            c1, c2, c3 = st.columns(3)
            plot_type = c1.selectbox(
                "Plot type",
                options=["Scatter", "Histogram (X)", "Box plot (by selection)"],
                key=f"{key_prefix}_plot_type",
            )
            x = c2.selectbox("X axis", num_cols, index=0, key=f"{key_prefix}_cx")
            color_by = c3.selectbox(
                "Color by (optional)",
                options=["(none)"] + num_cols,
                index=0,
                key=f"{key_prefix}_color_by",
            )
            color_col = color_by if color_by != "(none)" else None

            sel_mask = np.isin(summary["sim_idx"].values, selected)

            if plot_type == "Scatter":
                y = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key=f"{key_prefix}_cy")
                scatter = create_custom_scatter(active_summary, x, y, sel_mask, color_col=color_col, height=340)
                st.plotly_chart(scatter, width="stretch", key=f"{key_prefix}_custom_scatter")

            elif plot_type == "Histogram (X)":
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=active_summary[x],
                    nbinsx=35,
                    name="Active",
                    marker_color="#3b82f6",
                    opacity=0.7,
                ))
                if sel_mask.any():
                    fig.add_trace(go.Histogram(
                        x=active_summary.loc[sel_mask, x],
                        nbinsx=35,
                        name="Highlighted",
                        marker_color="#1e40af",
                        opacity=0.85,
                    ))
                fig.update_layout(
                    title=f"Histogram of {x}",
                    template="plotly_white",
                    height=340,
                    barmode="overlay",
                    xaxis_title=x,
                )
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_custom_hist")

            elif plot_type == "Box plot (by selection)":
                # Simple box: highlighted vs others
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=active_summary[x],
                    name="Active (not highlighted)",
                    marker_color="#9ca3af",
                    boxpoints=False,
                ))
                if sel_mask.any():
                    fig.add_trace(go.Box(
                        y=active_summary.loc[sel_mask, x],
                        name="Highlighted",
                        marker_color="#1e40af",
                        boxpoints="outliers",
                    ))
                fig.update_layout(
                    title=f"Box plot of {x} (highlighted vs rest)",
                    template="plotly_white",
                    height=340,
                    yaxis_title=x,
                )
                st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_custom_box")

            if st.button("Show correlation heatmap of all numeric columns (active)", key=f"{key_prefix}_corr_btn"):
                hm = create_correlation_heatmap(active_summary)
                st.plotly_chart(hm, width="stretch", key=f"{key_prefix}_corr")

    # Export
    if not active_summary.empty:
        csv = active_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download active simulations (CSV)",
            data=csv,
            file_name=f"simulation_results_{key_prefix}.csv",
            mime="text/csv",
            key=f"{key_prefix}_export",
        )


__all__ = [
    "discover_numeric_keys",
    "build_summary_dataframe",
    "align_paths_to_grid",
    "compute_quantile_bands",
    "get_state_at_time",
    "compute_path_drawdown",
    "create_spaghetti_plot",
    "create_fan_chart",
    "create_cross_section_plot",
    "create_custom_scatter",
    "create_correlation_heatmap",
    "render_simulation_analysis",
]


# =============================================================================
# Verification helpers (for completing the plan's verification section)
# =============================================================================


def get_large_run_recommendations(n_sims: int) -> dict[str, Any]:
    """
    Returns recommended settings for large Monte Carlo runs.
    Used for manual verification of 1200+ sim performance (plan verification point 7).
    """
    if n_sims <= 500:
        return {"max_background": 120, "use_cache": True, "note": "No special handling needed."}
    elif n_sims <= 1000:
        return {
            "max_background": 60,
            "use_cache": True,
            "note": "Consider reducing background lines and enabling caching.",
        }
    else:
        return {
            "max_background": max(20, int(60000 / n_sims)),
            "use_cache": True,
            "note": "Strongly recommended: use performance slider + caching. Expect 1-2s re-renders on filter changes.",
        }


def qualitative_wow_check_guidance() -> str:
    """
    Guidance for the qualitative 'wow' check (plan verification point 9).
    """
    return (
        "Run a 30-year retirement scenario with 400+ simulations.\n"
        "Ask: 'What fraction of paths are still growing at year 20?'\n"
        "Or: 'Do the top-quartile endings usually stay above the median the whole time?'\n"
        "Good visualizations should let a user answer these in under 30 seconds using the fan + scrubber + selection tools."
    )
