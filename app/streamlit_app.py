"""
Financial Simulator — Interactive Scenario Builder
==================================================

Major upgrade: Full interactive scenario builder with live distributions,
templates, external drivers, custom metrics, save/load, and Plotly-first UX.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from financial_simulator.core import SimulationResult
from financial_simulator.core.event import ComposedEventBuilder, IntervalTiming, FixedValue
from financial_simulator.monte_carlo import MonteCarloRunner
from financial_simulator.scenarios import (
    ScenarioConfig, list_templates, load_template,
    build_engine, run_single, run_monte_carlo,
)
from app.components.distribution_viz import render_distribution_picker

# Optional legacy examples (still supported)
try:
    from examples.retirement import create_retirement_engine
    from examples.business_cashflow import create_business_engine
    HAS_LEGACY = True
except Exception:
    HAS_LEGACY = False


st.set_page_config(
    page_title="Financial Simulator — Scenario Builder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Financial Simulation Platform")
st.caption("Monte Carlo Analysis & Interactive Scenario Builder (Phase 4+ foundation)")

# =============================================================================
# NAVIGATION
# =============================================================================
nav = st.radio(
    "Mode",
    ["🛠️ Scenario Builder", "📊 Run & Analyze", "🎲 Distribution Library", "📁 My Scenarios"],
    horizontal=True,
    label_visibility="collapsed",
)

# =============================================================================
# SESSION STATE
# =============================================================================
if "current_scenario" not in st.session_state:
    # Start with a nice default template
    try:
        st.session_state.current_scenario = load_template("retirement_30yr")
    except Exception:
        st.session_state.current_scenario = ScenarioConfig(
            name="Quick Start",
            start=datetime(2026, 1, 1),
            end=datetime(2027, 1, 1),
            initial_state={"cumulative_cash": 0.0},
        )

if "results" not in st.session_state:
    st.session_state.results = None

if "lib" not in st.session_state:
    from financial_simulator.scenarios import DistributionLibrary
    st.session_state.lib = DistributionLibrary()

current: ScenarioConfig = st.session_state.current_scenario

# =============================================================================
# MODE: SCENARIO BUILDER (the main new experience)
# =============================================================================
if nav == "🛠️ Scenario Builder":
    st.header("🛠️ Scenario Builder")
    st.markdown("Build, tweak, preview, save, and run complex financial simulations — no code required.")

    # Template loader
    with st.expander("📥 Load from Template (recommended starting point)", expanded=True):
        templates = list_templates()
        if templates:
            choice = st.selectbox("Choose a template", templates, index=0)
            if st.button("Load Template into Builder", type="primary"):
                st.session_state.current_scenario = load_template(choice)
                st.rerun()
        else:
            st.info("No templates found.")

    # Basic metadata
    col1, col2 = st.columns(2)
    with col1:
        current.name = st.text_input("Scenario Name", value=current.name)
        current.description = st.text_area("Description", value=current.description or "", height=80)
    with col2:
        current.start = st.date_input("Start Date", value=current.start)
        current.end = st.date_input("End Date", value=current.end)
        if isinstance(current.start, datetime):
            current.start = datetime.combine(current.start, datetime.min.time())
        if isinstance(current.end, datetime):
            current.end = datetime.combine(current.end, datetime.min.time())

    # Initial State editor (simple key-value)
    st.subheader("Initial State")
    init_state = current.initial_state or {}
    new_init = {}
    for k, v in list(init_state.items()):
        new_init[k] = st.number_input(f"Initial {k}", value=float(v) if isinstance(v, (int, float)) else 0.0, key=f"init_{k}")
    # Allow adding new keys
    new_key = st.text_input("Add new state variable name", key="new_init_key")
    if new_key:
        new_init[new_key] = st.number_input(f"Initial value for {new_key}", value=0.0, key="new_init_val")
    current.initial_state = new_init

    # Event Builders (very basic list for Phase 4 MVP)
    st.subheader("Event Sources")
    st.caption("Full event editor coming in later refinement. For now you can load rich templates or add simple fixed monthly items.")

    # Show existing
    for i, eb in enumerate(current.event_builders):
        st.write(f"**{eb.name or f'Event {i+1}'}** — {eb.metadata}")

    # Simple "add monthly fixed" helper
    with st.expander("➕ Add simple monthly fixed event"):
        name = st.text_input("Event name", "monthly_income")
        value = st.number_input("Monthly value (positive = inflow)", value=1000.0)
        if st.button("Add Monthly Fixed Event"):
            current.event_builders.append(
                ComposedEventBuilder(
                    timing=IntervalTiming(interval=timedelta(days=30)),
                    value_gen=FixedValue(value=value),
                    metadata={"type": name},
                    name=name,
                )
            )
            st.success("Added. Re-run preview to see effect.")
            st.rerun()

    # Distribution picker demo (live)
    st.subheader("🎲 Quick Distribution Explorer (embedded)")
    render_distribution_picker(key_prefix="builder_dist", show_save_section=False)

    # Preview & Run controls
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔍 Preview Single Run", use_container_width=True):
            with st.spinner("Running single simulation..."):
                res = run_single(current, seed=42)
                st.session_state.preview_result = res
            st.success("Preview complete.")
    with c2:
        n_sims = st.slider("Monte Carlo runs", 50, 2000, 300, 50)
    with c3:
        if st.button("🚀 Run Monte Carlo from Builder", type="primary", use_container_width=True):
            with st.spinner(f"Running {n_sims} simulations..."):
                results = run_monte_carlo(current, n_sims=n_sims, base_seed=42, n_jobs=4)
            st.session_state.results = results
            st.session_state.scenario_name = current.name
            st.success(f"Completed {len(results)} simulations!")
            st.switch_page("Run & Analyze") if hasattr(st, "switch_page") else None

    if "preview_result" in st.session_state:
        st.markdown("**Single Run Preview**")
        st.json(st.session_state.preview_result.final_state)

# =============================================================================
# MODE: RUN & ANALYZE (enhanced results view)
# =============================================================================
elif nav == "📊 Run & Analyze":
    st.header("📊 Run & Analyze")

    # Legacy quick selector (still useful)
    if HAS_LEGACY:
        with st.sidebar:
            st.header("Quick Legacy Scenarios")
            legacy = st.selectbox("Legacy", ["None", "Retirement Planning", "Small Business Cash Flow"])
            if st.button("Run Legacy"):
                factory = create_retirement_engine if legacy == "Retirement Planning" else create_business_engine
                with st.spinner("Running legacy..."):
                    results = MonteCarloRunner(n_jobs=4).run(200, factory, base_seed=42)
                st.session_state.results = results
                st.session_state.scenario_name = legacy

    results = st.session_state.results
    if not results:
        st.info("Run a simulation from the **Scenario Builder** tab or the legacy quick selector in the sidebar.")
        st.stop()

    st.subheader(f"Results — {st.session_state.get('scenario_name', 'Custom Scenario')}")

    # Basic overview (reused + enhanced)
    final_values = []
    for r in results:
        val = r.final_state.get("cumulative_cash")
        if val is None:
            val = r.final_state.get("cash", 0)
        final_values.append(val)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Final Value", f"${pd.Series(final_values).mean():,.0f}")
    col2.metric("Median", f"${pd.Series(final_values).median():,.0f}")
    col3.metric("5th %ile (Downside)", f"${pd.Series(final_values).quantile(0.05):,.0f}")
    col4.metric("95th %ile (Upside)", f"${pd.Series(final_values).quantile(0.95):,.0f}")

    # Distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=final_values, nbinsx=35, name="Final Outcomes"))
    fig.update_layout(title="Distribution of Final Outcomes", height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Custom metrics (new in Phase 5 spirit)
    if results and "__custom_metrics__" in results[0].final_state:
        st.subheader("Custom Metrics (from scenario)")
        cm = results[0].final_state["__custom_metrics__"]
        st.json(cm)

# =============================================================================
# MODE: DISTRIBUTION LIBRARY
# =============================================================================
elif nav == "🎲 Distribution Library":
    st.header("🎲 Distribution Library")
    st.markdown("This is powered by the Phase 2 interactive component.")

    def save_to_lib(saved):
        try:
            st.session_state.lib.add(saved)
            st.toast(f"Saved {saved.name}")
        except Exception as e:
            st.error(str(e))

    render_distribution_picker(
        key_prefix="global_lib",
        library=st.session_state.lib,
        on_save_callback=save_to_lib,
    )

    st.divider()
    st.subheader("Currently in Session Library")
    for d in st.session_state.lib.distributions:
        st.write(f"**{d.name}** — {d.description}")

# =============================================================================
# MODE: MY SCENARIOS (placeholder for Phase 3+ full persistence)
# =============================================================================
else:
    st.header("📁 My Scenarios")
    st.info("Full file-based save/load UI coming in the next iteration. For now, use the Scenario Builder + Export/Import JSON, or the committed templates.")
    st.json(current.to_dict(), expanded=False)

    if st.button("Download current scenario as JSON"):
        st.download_button(
            "Download scenario.json",
            data=current.to_json(),
            file_name=f"{current.name.replace(' ', '_')}.json",
            mime="application/json",
        )

# =============================================================================
# Footer
# =============================================================================
st.caption("Financial Simulator • Scenario Builder foundation (Phases 1-4 complete) • Streamlit + Plotly + Pydantic")

# -----------------------------
# Main Logic
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.scenario_name = None

if run_button:
    factory = create_retirement_engine if scenario_name == "Retirement Planning" else create_business_engine

    with st.spinner(f"Running {n_sims} simulations... this may take a moment"):
        runner = MonteCarloRunner(n_jobs=4)
        results = runner.run(n_sims, factory, base_seed=base_seed)

    st.session_state.results = results
    st.session_state.scenario_name = scenario_name
    st.success(f"Completed {len(results)} simulations!")

# -----------------------------
# Display Results
# -----------------------------
results: list[SimulationResult] | None = st.session_state.results

if results is None:
    st.info("Configure your simulation on the left and click **Run Monte Carlo** to begin.")
    st.stop()

# Tabs
tab_overview, tab_timeseries, tab_risk, tab_details = st.tabs([
    "📊 Overview",
    "📈 Time Series",
    "📉 Risk Metrics",
    "🔍 Selected Simulation"
])

# =====================
# TAB 1: Overview
# =====================
with tab_overview:
    st.subheader(f"Results Overview — {st.session_state.scenario_name}")

    final_values = []
    for r in results:
        val = r.final_state.get("cumulative_cash")
        if val is None:
            val = r.final_state.get("cash", 0)
        final_values.append(val)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Final Value", f"${pd.Series(final_values).mean():,.0f}")
    col2.metric("Median", f"${pd.Series(final_values).median():,.0f}")
    col3.metric("5th Percentile (Downside)", f"${pd.Series(final_values).quantile(0.05):,.0f}")
    col4.metric("95th Percentile (Upside)", f"${pd.Series(final_values).quantile(0.95):,.0f}")

    # Distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=final_values, nbinsx=30, name="Final Value Distribution"))
    fig.update_layout(title="Distribution of Final Outcomes", xaxis_title="Final Value ($)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 2: Time Series (Interactive)
# =====================
with tab_timeseries:
    st.subheader("Individual Simulation Paths")

    # Build time series data for all simulations
    all_dfs = []
    for i, res in enumerate(results):
        if not res.state_history:
            continue
        times = list(res.state_history.keys())
        values = [res.state_history[t].get(y_axis, res.state_history[t].get("cash", 0)) for t in times]
        df = pd.DataFrame({"time": times, "value": values, "sim_id": i})
        all_dfs.append(df)

    if not all_dfs:
        st.warning("No time series data available for this scenario.")
    else:
        combined = pd.concat(all_dfs, ignore_index=True)

        # Highlight controls
        highlight_sims = st.multiselect(
            "Highlight specific simulations (by index)",
            options=list(range(len(results))),
            default=[],
            help="Selected simulations will be drawn thicker and more opaque"
        )

        # Create Plotly figure
        fig = go.Figure()

        for sim_id in range(len(results)):
            sim_df = combined[combined["sim_id"] == sim_id]
            is_highlighted = sim_id in highlight_sims

            fig.add_trace(go.Scatter(
                x=sim_df["time"],
                y=sim_df["value"],
                mode="lines",
                name=f"Sim {sim_id}",
                line=dict(width=2.5 if is_highlighted else 0.8),
                opacity=0.85 if is_highlighted else 0.25,
                showlegend=is_highlighted
            ))

        fig.update_layout(
            title=f"Time Series: {y_axis} across all simulations",
            xaxis_title="Date",
            yaxis_title=y_axis,
            hovermode="x unified",
            height=550
        )
        st.plotly_chart(fig, use_container_width=True, key="timeseries")

# =====================
# TAB 3: Risk Metrics
# =====================
with tab_risk:
    st.subheader("Risk Analysis")

    from financial_simulator.analytics.risk import MonteCarloAnalyzer

    analyzer = MonteCarloAnalyzer()
    report = analyzer.analyze_results(results)

    st.json(report.metrics)

# =====================
# TAB 4: Selected Simulation Details
# =====================
with tab_details:
    st.subheader("Inspect Individual Simulation")

    selected_id = st.selectbox(
        "Select simulation to inspect",
        options=list(range(len(results))),
        format_func=lambda x: f"Simulation #{x}"
    )

    selected_result: SimulationResult = results[selected_id]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Simulation Parameters**")
        st.write(f"**Seed used:** {selected_result.final_state.get('__seed_used__', 'N/A')}")
        st.write(f"**Name:** {selected_result.name}")
        st.write(f"**Period:** {selected_result.start.date()} → {selected_result.end.date()}")
        st.write("**Final State:**")
        st.json(selected_result.final_state)

    with col2:
        st.markdown("**Events in this simulation**")
        if selected_result.events:
            events_df = pd.DataFrame([
                {
                    "Time": e.time.strftime("%Y-%m-%d"),
                    "Value": round(e.value, 2),
                    "Metadata": e.metadata
                }
                for e in selected_result.events
            ])
            st.dataframe(events_df, use_container_width=True, height=400)
        else:
            st.info("No events recorded for this simulation.")

st.caption("Built with the Financial Simulation Platform • Streamlit + Plotly")
