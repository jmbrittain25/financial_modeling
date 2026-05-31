"""
Financial Simulator — Interactive Scenario Builder
==================================================

Major upgrade: Full interactive scenario builder with live distributions,
templates, external drivers, custom metrics, save/load, and Plotly-first UX.

Run with:
    streamlit run app/streamlit_app.py
"""

from datetime import datetime

import streamlit as st

from app.components.continuous_processes_editor import render_continuous_processes_editor
from app.components.custom_metrics_editor import render_custom_metrics_editor
from app.components.distribution_viz import render_distribution_gallery, render_distribution_picker
from app.components.event_builder_editor import render_event_builder_list_editor
from app.components.external_drivers_editor import render_external_drivers_editor
from app.components.library_manager import render_library_manager
from app.components.results_dashboard import render_results_dashboard
from app.components.scenario_overview import render_scenario_overview
from app.components.template_gallery import render_template_gallery
from financial_simulator.monte_carlo import MonteCarloRunner
from financial_simulator.scenarios import (
    ScenarioConfig,
    load_template,
    load_user_scenario_library,
    run_monte_carlo,
    run_single,
    save_user_scenario,
)

# Optional legacy examples (still supported)
try:
    from examples.business_cashflow import create_business_engine
    from examples.retirement import create_retirement_engine

    HAS_LEGACY = True
except Exception:
    HAS_LEGACY = False


st.set_page_config(
    page_title="Financial Simulator — Scenario Builder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Simulation Platform")
st.caption("Powerful Interactive Scenario Builder + Monte Carlo Analysis (Full Phase 5 Experience)")

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

# First-run seeding of user library with high-quality templates (Step 8)
if "user_libs_seeded" not in st.session_state:
    try:
        user_scen_lib = load_user_scenario_library()
        if len(user_scen_lib.scenarios) == 0:
            for template_name in [
                "retirement_30yr",
                "variable_rate_mortgage",
                "business_variable_costs",
                "tax_planning_optimized",
                "savings_with_growth",
            ]:
                try:
                    tmpl = load_template(template_name)
                    # Save copies so user can modify without affecting originals
                    save_user_scenario(tmpl)
                except Exception:
                    pass
            st.toast("Seeded your personal library with the 5 high-quality templates", icon="📚")
    except Exception:
        pass
    st.session_state.user_libs_seeded = True

current: ScenarioConfig = st.session_state.current_scenario

# =============================================================================
# SIDEBAR: Current Scenario Summary (plan Step 8 polish)
# =============================================================================
with st.sidebar:
    st.header("Current Scenario")

    summary = current.summary()
    st.markdown(f"**{current.name}**")
    st.caption(current.description or "No description yet")

    cols = st.columns(2)
    cols[0].metric("Horizon", f"{summary['horizon_years']}y")
    cols[1].metric("Events", summary["num_event_builders"])

    st.caption(
        f"Drivers: {summary['num_drivers']} • Processes: {summary['num_continuous']} • Metrics: {summary['num_custom_metrics']}"
    )

    if st.button("💾 Save Now", use_container_width=True, key="sidebar_save"):
        try:
            save_user_scenario(current)
            st.success("Saved to your library!")
        except Exception as e:
            st.error(str(e))

    if st.button("📋 Duplicate", use_container_width=True, key="sidebar_dup"):
        dup = current.clone()
        dup.name = f"{current.name} (Copy)"
        st.session_state.current_scenario = dup
        st.rerun()

    st.divider()
    st.caption("Tip: Use the Template Gallery in Builder mode for great starting points.")

# =============================================================================
# MODE: SCENARIO BUILDER (the main new experience)
# =============================================================================
if nav == "🛠️ Scenario Builder":
    st.header("🛠️ Scenario Builder")
    st.markdown(
        "Build, tweak, preview, save, and run complex financial simulations — no code required."
    )
    st.caption(
        "Tip: Start with a template from the gallery below, then use the presets in the Event Editor to rapidly construct realistic cash flows."
    )

    badges = []
    if current.external_drivers:
        badges.append(f"🔗 {len(current.external_drivers)} external driver(s)")
    if current.custom_metrics:
        badges.append(f"📊 {len(current.custom_metrics)} custom metric(s)")
    if badges:
        st.caption(" | ".join(badges))

    # Template Gallery (new nice UI)
    with st.expander("📥 Load from Template Gallery (recommended)", expanded=True):
        loaded = render_template_gallery(key_prefix="builder_templates")
        if loaded:
            st.session_state.current_scenario = loaded
            st.rerun()

    # Basic metadata
    col1, col2 = st.columns(2)
    with col1:
        current.name = st.text_input("Scenario Name", value=current.name)
        current.description = st.text_area(
            "Description", value=current.description or "", height=80
        )
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
        new_init[k] = st.number_input(
            f"Initial {k}", value=float(v) if isinstance(v, (int, float)) else 0.0, key=f"init_{k}"
        )
    # Allow adding new keys
    new_key = st.text_input("Add new state variable name", key="new_init_key")
    if new_key:
        new_init[new_key] = st.number_input(
            f"Initial value for {new_key}", value=0.0, key="new_init_val"
        )
    current.initial_state = new_init

    # === NEW POWERFUL EVENT EDITOR (Step 7 integration) ===
    current.event_builders = render_event_builder_list_editor(
        key_prefix="main_builder_events",
        builders=current.event_builders,
    )

    # === NEW EDITORS (Step 7 integration) ===
    st.divider()
    with st.expander(
        "📊 Custom Metrics (optional but powerful)", expanded=len(current.custom_metrics) > 0
    ):
        current.custom_metrics = render_custom_metrics_editor(
            key_prefix="main_builder_metrics",
            metrics=current.custom_metrics,
        )

    with st.expander(
        "🔗 External Drivers (for variable rates, inflation, etc.)",
        expanded=len(current.external_drivers) > 0,
    ):
        current.external_drivers = render_external_drivers_editor(
            key_prefix="main_builder_drivers",
            drivers=current.external_drivers,
        )

    with st.expander(
        "📈 Continuous Processes (growth & stochastic paths)",
        expanded=len(current.continuous_processes) > 0,
    ):
        current.continuous_processes = render_continuous_processes_editor(
            key_prefix="main_builder_processes",
            processes=current.continuous_processes,
        )

    # Scenario Overview (visual summary)
    st.divider()
    render_scenario_overview(current, key_prefix="builder_overview")

    # Distribution picker (still useful)
    st.subheader("🎲 Quick Distribution Explorer")
    st.caption(
        "Use this to explore or create distributions, then embed them via the Event Editor above."
    )
    render_distribution_picker(key_prefix="builder_dist", show_save_section=False)

    # Preview & Run controls + Save
    st.divider()
    st.markdown("**Run & Persist**")
    st.caption(
        "Always save your work. Single-run previews are fast and great for sanity checks before launching Monte Carlo."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🔍 Preview Single Run", use_container_width=True):
            with st.spinner("Running single simulation..."):
                res = run_single(current, seed=42)
                st.session_state.preview_result = res
            st.success("Preview complete. Check the JSON below.")
    with c2:
        n_sims = st.slider(
            "Monte Carlo runs",
            50,
            2000,
            300,
            50,
            help="More runs = better statistics but longer wait",
        )
    with c3:
        if st.button("🚀 Run Monte Carlo from Builder", type="primary", use_container_width=True):
            with st.spinner(f"Running {n_sims} simulations..."):
                results = run_monte_carlo(current, n_sims=n_sims, base_seed=42, n_jobs=4)
            st.session_state.results = results
            st.session_state.scenario_name = current.name
            st.success(f"Completed {len(results)} simulations! Switch to 'Run & Analyze' mode.")
    with c4:
        if st.button("💾 Save to My Library", use_container_width=True):
            try:
                save_user_scenario(current)
                st.success(f"Saved '{current.name}' to your personal library!")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    if "preview_result" in st.session_state:
        with st.expander("Single Run Preview (final state)", expanded=False):
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
            legacy = st.selectbox(
                "Legacy", ["None", "Retirement Planning", "Small Business Cash Flow"]
            )
            if st.button("Run Legacy"):
                factory = (
                    create_retirement_engine
                    if legacy == "Retirement Planning"
                    else create_business_engine
                )
                with st.spinner("Running legacy..."):
                    results = MonteCarloRunner(n_jobs=4).run(200, factory, base_seed=42)
                st.session_state.results = results
                st.session_state.scenario_name = legacy

    results = st.session_state.results
    if not results:
        st.info(
            "Run a simulation from the **Scenario Builder** tab or the legacy quick selector in the sidebar."
        )
        st.stop()

    # Use the new rich results dashboard
    render_results_dashboard(
        results,
        scenario_name=st.session_state.get("scenario_name", "Custom Scenario"),
    )

# =============================================================================
# MODE: DISTRIBUTION LIBRARY
# =============================================================================
elif nav == "🎲 Distribution Library":
    st.header("🎲 Distribution Library")
    st.markdown("Create, visualize, and save reusable distributions with live Plotly previews.")

    # Quick preset gallery
    with st.expander("Quick Financial Presets", expanded=False):
        chosen = render_distribution_gallery(key_prefix="global_gallery")
        if chosen:
            st.session_state["pending_dist"] = chosen

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
    if st.session_state.lib.distributions:
        for d in st.session_state.lib.distributions:
            st.write(f"**{d.name}** — {d.description or 'No description'}")
    else:
        st.info(
            "Save distributions from the picker above — they will appear here and be available when building scenarios."
        )

# =============================================================================
# MODE: MY SCENARIOS + DISTRIBUTION LIBRARY (now powered by real persistence)
# =============================================================================
else:
    render_library_manager(key_prefix="global_library")

# =============================================================================
# Footer
# =============================================================================
st.caption(
    "Financial Simulator • Scenario Builder foundation (Phases 1-4 complete) • Streamlit + Plotly + Pydantic"
)
