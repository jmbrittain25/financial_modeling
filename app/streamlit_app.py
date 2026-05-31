"""
Modern interactive web interface for the financial simulator using Streamlit.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px

from financial_simulator.core import SimulationEngine
from financial_simulator.monte_carlo import MonteCarloRunner
from examples.retirement import create_retirement_engine
from examples.business_cashflow import create_business_engine


st.set_page_config(page_title="Financial Simulator", layout="wide")
st.title("Financial Simulation Platform")

st.sidebar.header("Simulation Controls")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Retirement Planning", "Small Business Cash Flow"]
)

n_sims = st.sidebar.slider("Number of Simulations", 10, 2000, 200, 50)
base_seed = st.sidebar.number_input("Base Seed", value=42, step=1)

run_button = st.sidebar.button("Run Monte Carlo", type="primary")

if run_button:
    with st.spinner(f"Running {n_sims} simulations..."):
        factory = create_retirement_engine if scenario == "Retirement Planning" else create_business_engine
        runner = MonteCarloRunner(n_jobs=4)
        results = runner.run(n_sims, factory, base_seed=base_seed)

    # Extract final values
    final_values = [r.final_state.get("cumulative_cash", r.final_state.get("cash", 0)) for r in results]

    st.success(f"Completed {len(results)} simulations")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Final Value", f"${pd.Series(final_values).mean():,.0f}")
    col2.metric("Median Final Value", f"${pd.Series(final_values).median():,.0f}")
    col3.metric("5th Percentile (Bad Case)", f"${pd.Series(final_values).quantile(0.05):,.0f}")

    # Distribution plot
    fig = px.histogram(
        final_values,
        nbins=40,
        title=f"Distribution of Final Outcomes — {scenario}",
        labels={"value": "Final Value ($)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show a few sample paths (simplified)
    st.subheader("Sample Simulation Paths (first 5)")
    for i, res in enumerate(results[:5]):
        st.write(f"Sim {i}: Final = ${res.final_state.get('cumulative_cash', 0):,.0f}")

else:
    st.info("Configure parameters on the left and click **Run Monte Carlo** to begin.")
