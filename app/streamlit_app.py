"""
Professional Interactive Dashboard for the Financial Simulation Platform

Run with:
    streamlit run app/streamlit_app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from examples.business_cashflow import create_business_engine
from examples.retirement import create_retirement_engine
from financial_simulator.core import SimulationResult
from financial_simulator.monte_carlo import MonteCarloRunner

st.set_page_config(
    page_title="Financial Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Simulation Platform")
st.caption("Monte Carlo Analysis & Risk Dashboard")

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Simulation Controls")

    scenario_name = st.selectbox(
        "Scenario",
        ["Retirement Planning", "Small Business Cash Flow"],
        help="Choose a pre-built scenario",
    )

    n_sims = st.slider(
        "Number of Simulations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Higher values give better statistics but take longer",
    )

    base_seed = st.number_input(
        "Base Random Seed", value=42, step=1, help="Change this to get different random outcomes"
    )

    y_axis = st.selectbox(
        "Time Series Metric",
        options=["cumulative_cash", "cash", "portfolio_value", "property_value"],
        index=0,
        help="Which state variable to plot over time",
    )

    run_button = st.button("🚀 Run Monte Carlo", type="primary", use_container_width=True)

# -----------------------------
# Main Logic
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.scenario_name = None

if run_button:
    factory = (
        create_retirement_engine
        if scenario_name == "Retirement Planning"
        else create_business_engine
    )

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
tab_overview, tab_timeseries, tab_risk, tab_details = st.tabs(
    ["📊 Overview", "📈 Time Series", "📉 Risk Metrics", "🔍 Selected Simulation"]
)

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
    fig.update_layout(
        title="Distribution of Final Outcomes", xaxis_title="Final Value ($)", yaxis_title="Count"
    )
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
        values = [
            res.state_history[t].get(y_axis, res.state_history[t].get("cash", 0)) for t in times
        ]
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
            help="Selected simulations will be drawn thicker and more opaque",
        )

        # Create Plotly figure
        fig = go.Figure()

        for sim_id in range(len(results)):
            sim_df = combined[combined["sim_id"] == sim_id]
            is_highlighted = sim_id in highlight_sims

            fig.add_trace(
                go.Scatter(
                    x=sim_df["time"],
                    y=sim_df["value"],
                    mode="lines",
                    name=f"Sim {sim_id}",
                    line=dict(width=2.5 if is_highlighted else 0.8),
                    opacity=0.85 if is_highlighted else 0.25,
                    showlegend=is_highlighted,
                )
            )

        fig.update_layout(
            title=f"Time Series: {y_axis} across all simulations",
            xaxis_title="Date",
            yaxis_title=y_axis,
            hovermode="x unified",
            height=550,
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
        format_func=lambda x: f"Simulation #{x}",
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
            events_df = pd.DataFrame(
                [
                    {
                        "Time": e.time.strftime("%Y-%m-%d"),
                        "Value": round(e.value, 2),
                        "Metadata": e.metadata,
                    }
                    for e in selected_result.events
                ]
            )
            st.dataframe(events_df, use_container_width=True, height=400)
        else:
            st.info("No events recorded for this simulation.")

st.caption("Built with the Financial Simulation Platform • Streamlit + Plotly")
