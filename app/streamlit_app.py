"""
Financial Simulator — Interactive Scenario Builder
==================================================

Workflow:
  1. Setup       — dates and starting assets / cash
  2. Generators  — custom event generators with per-generator distributions
  3. Run         — configure and execute Monte Carlo simulations
  4. Results     — visualize outcomes and export configs + results

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import streamlit as st

from app.components.continuous_processes_editor import render_continuous_processes_editor
from app.components.custom_metrics_editor import render_custom_metrics_editor
from app.components.event_builder_editor import render_event_builder_list_editor
from app.components.external_drivers_editor import render_external_drivers_editor
from app.components.library_manager import render_library_manager
from app.components.results_dashboard import render_results_dashboard
from app.components.scenario_io import render_scenario_export_button, render_scenario_import_panel
from financial_simulator.scenarios import (
    ScenarioConfig,
    run_monte_carlo,
    run_single,
    save_user_scenario,
)

try:
    from app.components.simulation_viz import render_simulation_analysis

    HAS_SIM_VIZ = True
except Exception:
    HAS_SIM_VIZ = False
    render_simulation_analysis = None  # type: ignore


def _blank_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="New Scenario",
        description="",
        start=datetime(2026, 1, 1),
        end=datetime(2036, 1, 1),
        initial_state={"cash": 0.0},
    )


def _load_imported_scenario(cfg: ScenarioConfig) -> None:
    st.session_state.current_scenario = cfg
    st.session_state.saved_scenario_name = None
    st.session_state.results = None
    st.toast(f"Imported '{cfg.name}'", icon="📥")
    st.rerun()


st.set_page_config(
    page_title="Financial Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Simulator")
st.caption("Model your finances with custom generators and Monte Carlo analysis.")

# =============================================================================
# NAVIGATION
# =============================================================================
SECTIONS = [
    "1 · Setup",
    "2 · Event Generators",
    "3 · Run",
    "4 · Results",
    "Scenarios",
]
nav = st.radio("Section", SECTIONS, horizontal=True, label_visibility="collapsed")

# =============================================================================
# SESSION STATE
# =============================================================================
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = _blank_scenario()

if "results" not in st.session_state:
    st.session_state.results = None

if "run_config" not in st.session_state:
    st.session_state.run_config = {
        "n_sims": 300,
        "base_seed": 42,
        "n_jobs": 4,
    }

if "saved_scenario_name" not in st.session_state:
    st.session_state.saved_scenario_name = None

if "init_var_form_reset" not in st.session_state:
    st.session_state.init_var_form_reset = 0

current: ScenarioConfig = st.session_state.current_scenario

# =============================================================================
# SIDEBAR — scenario management
# =============================================================================
with st.sidebar:
    st.header("Scenario")

    st.markdown(f"**{current.name}**")
    if current.description:
        st.caption(current.description[:100])

    summary = current.summary()
    st.caption(f"{summary['horizon_years']}y horizon · {summary['num_event_builders']} generators")

    if st.session_state.saved_scenario_name:
        st.caption(f"Last saved as: {st.session_state.saved_scenario_name}")

    st.divider()

    if st.button("New Scenario", use_container_width=True, key="sidebar_new"):
        st.session_state.current_scenario = _blank_scenario()
        st.session_state.saved_scenario_name = None
        st.session_state.results = None
        st.rerun()

    if st.button("Duplicate", use_container_width=True, key="sidebar_dup"):
        dup = current.clone()
        dup.name = f"{current.name} (Copy)"
        st.session_state.current_scenario = dup
        st.session_state.saved_scenario_name = None
        st.rerun()

    st.caption(
        "**Save** updates your library copy (reload from **Scenarios**). "
        "**Save As** writes a new library file under a name you choose. "
        "**Export** on Setup / Results downloads JSON to your computer."
    )

    if st.button("Save", use_container_width=True, key="sidebar_save"):
        try:
            path = save_user_scenario(current, overwrite=True)
            st.session_state.saved_scenario_name = current.name
            st.toast(f"Saved to library: {path.name}", icon="💾")
        except Exception as e:
            st.error(f"Save failed: {e}")

    save_as_name = st.text_input(
        "Save As — file name",
        value=current.name,
        key="sidebar_save_as_name",
        help="Saved to ~/.financial-simulator/v1/scenarios/ (library folder on this machine).",
    )
    if st.button("Save As…", use_container_width=True, key="sidebar_save_as"):
        chosen = (save_as_name or current.name).strip()
        if not chosen:
            st.warning("Enter a file name for Save As.")
        else:
            try:
                to_save = current.clone()
                to_save.name = chosen
                path = save_user_scenario(to_save, overwrite=False, file_name=chosen)
                st.session_state.current_scenario = to_save
                st.session_state.saved_scenario_name = chosen
                st.toast(f"Saved as {path.name}", icon="💾")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")

    st.divider()
    st.caption(
        "Import from **Setup** or **Scenarios**. Load saved library files from **Scenarios**."
    )

# =============================================================================
# SECTION 1: SETUP
# =============================================================================
if nav == "1 · Setup":
    st.header("Setup")
    st.markdown("Define the simulation horizon and your starting assets and cash.")

    imp_col, exp_col = st.columns([3, 1])
    with imp_col:
        render_scenario_import_panel(
            key_prefix="setup_import",
            on_loaded=_load_imported_scenario,
            label="Import scenario",
            button_label="Import scenario from file…",
        )
    with exp_col:
        st.write("")
        render_scenario_export_button(current, key_prefix="setup_export", label="Export")

    col1, col2 = st.columns(2)
    with col1:
        current.name = st.text_input("Scenario name", value=current.name)
        current.description = st.text_area(
            "Description (optional)", value=current.description or "", height=80
        )
    with col2:
        start_date = st.date_input("Start date", value=current.start)
        end_date = st.date_input("End date", value=current.end)
        current.start = datetime.combine(start_date, datetime.min.time())
        current.end = datetime.combine(end_date, datetime.min.time())

    st.subheader("Starting assets & cash")
    st.caption(
        "Add any state variables you need — cash, investments, home value, "
        "mortgage rate, etc. Generators can read and update these during the simulation."
    )

    init_state = dict(current.initial_state or {})
    keys_to_remove: list[str] = []

    if init_state:
        for k in list(init_state.keys()):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.markdown(f"**{k}**")
            with c2:
                init_state[k] = c2.number_input(
                    f"Value for {k}",
                    value=float(init_state[k]) if isinstance(init_state[k], (int, float)) else 0.0,
                    key=f"init_{k}",
                    label_visibility="collapsed",
                )
            with c3:
                if c3.button("Remove", key=f"remove_init_{k}"):
                    keys_to_remove.append(k)

    for k in keys_to_remove:
        init_state.pop(k, None)

    st.markdown("**Add variable**")
    form_reset = st.session_state.init_var_form_reset
    ac1, ac2, ac3 = st.columns([2, 2, 1])
    with ac1:
        new_key = st.text_input(
            "Variable name",
            key=f"new_init_key_{form_reset}",
            placeholder="e.g. stocks",
        )
    with ac2:
        new_val = st.number_input(
            "Starting value",
            value=0.0,
            key=f"new_init_val_{form_reset}",
        )
    with ac3:
        st.write("")
        add_clicked = ac3.button("Add", key="add_init_var", use_container_width=True)

    if add_clicked:
        key_name = new_key.strip()
        if not key_name:
            st.warning("Enter a variable name before adding.")
        elif key_name in init_state:
            st.warning(f"'{key_name}' already exists — edit its value above or pick another name.")
        else:
            init_state[key_name] = new_val
            st.session_state.init_var_form_reset += 1
            st.toast(f"Added '{key_name}'", icon="✅")
            current.initial_state = dict(init_state)
            st.session_state.current_scenario = current
            st.rerun()

    current.initial_state = dict(init_state)
    st.session_state.current_scenario = current

    if not init_state:
        st.info("No starting variables yet. Add at least one (e.g. `cash`) above.")

# =============================================================================
# SECTION 2: EVENT GENERATORS
# =============================================================================
elif nav == "2 · Event Generators":
    st.header("Event Generators")
    st.markdown(
        "Build as many generators as you need. Each one defines **when** something "
        "happens and **how much** it changes — including custom distributions for "
        "stochastic amounts."
    )

    current.event_builders = render_event_builder_list_editor(
        key_prefix="main_generators",
        builders=current.event_builders,
    )

    with st.expander("Advanced — continuous processes (background growth / volatility)"):
        st.caption(
            "Optional. Models slow-moving changes between discrete events — e.g. "
            "portfolio drift, home appreciation, mean-reverting interest rates."
        )
        current.continuous_processes = render_continuous_processes_editor(
            key_prefix="main_processes",
            processes=current.continuous_processes,
        )

    with st.expander("Advanced — external drivers (macro / market variables)"):
        st.caption(
            "Optional. Inject stochastic values into state keys on a schedule — "
            "useful for variable mortgage rates or inflation indices."
        )
        current.external_drivers = render_external_drivers_editor(
            key_prefix="main_drivers",
            drivers=current.external_drivers,
            scenario_start=current.start,
            scenario_end=current.end,
        )

    with st.expander("Advanced — custom metrics (tracked after each run)"):
        current.custom_metrics = render_custom_metrics_editor(
            key_prefix="main_metrics",
            metrics=current.custom_metrics,
        )

# =============================================================================
# SECTION 3: RUN
# =============================================================================
elif nav == "3 · Run":
    st.header("Run Configuration")
    st.markdown("Configure and execute your Monte Carlo simulation.")

    rc = st.session_state.run_config

    col1, col2, col3 = st.columns(3)
    with col1:
        rc["n_sims"] = st.slider(
            "Number of simulations",
            min_value=50,
            max_value=5000,
            value=rc["n_sims"],
            step=50,
            help="More runs give smoother statistics but take longer.",
        )
    with col2:
        rc["base_seed"] = st.number_input(
            "Random seed",
            min_value=0,
            value=rc["base_seed"],
            help="Same seed + same config = reproducible results.",
        )
    with col3:
        rc["n_jobs"] = st.selectbox(
            "Parallel workers",
            options=[1, 2, 4, 8],
            index=[1, 2, 4, 8].index(rc["n_jobs"]) if rc["n_jobs"] in [1, 2, 4, 8] else 2,
        )

    st.session_state.run_config = rc

    st.divider()

    summary = current.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horizon", f"{summary['horizon_years']} years")
    c2.metric("Generators", summary["num_event_builders"])
    c3.metric("State variables", len(summary["state_keys"]))
    c4.metric("Simulations", rc["n_sims"])

    if summary["num_event_builders"] == 0:
        st.warning(
            "No event generators defined. Add some in **2 · Event Generators** before running."
        )

    col_preview, col_run = st.columns(2)
    with col_preview:
        if st.button("Preview single run", use_container_width=True):
            with st.spinner("Running…"):
                res = run_single(current, seed=rc["base_seed"])
                st.session_state.preview_result = res
            st.success("Preview complete.")

    with col_run:
        if st.button("Run Monte Carlo", type="primary", use_container_width=True):
            with st.spinner(f"Running {rc['n_sims']} simulations…"):
                results = run_monte_carlo(
                    current,
                    n_sims=rc["n_sims"],
                    base_seed=rc["base_seed"],
                    n_jobs=rc["n_jobs"],
                )
            st.session_state.results = results
            st.session_state.results_scenario = current.clone()
            st.session_state.scenario_name = current.name
            st.session_state.run_summary = {
                "n_sims": rc["n_sims"],
                "base_seed": rc["base_seed"],
                "completed_at": datetime.now().isoformat(),
            }
            st.success(f"Completed {len(results)} simulations. Open **4 · Results** to analyze.")

    if "preview_result" in st.session_state:
        with st.expander("Single-run preview (final state)", expanded=False):
            st.json(st.session_state.preview_result.final_state)

# =============================================================================
# SECTION 4: RESULTS
# =============================================================================
elif nav == "4 · Results":
    st.header("Results")

    results = st.session_state.results
    if not results:
        st.info("No results yet. Configure and run a simulation in **3 · Run**.")
        st.stop()

    results_scenario = st.session_state.get("results_scenario", current)
    run_summary = st.session_state.get("run_summary", {})

    render_results_dashboard(
        results,
        scenario_name=st.session_state.get("scenario_name", current.name),
        scenario_config=results_scenario,
        run_summary=run_summary,
    )

    st.divider()
    with st.expander("Deep path analysis", expanded=False):
        if HAS_SIM_VIZ and render_simulation_analysis is not None:
            default_key = None
            if results and results[0].state_history:
                keys = list(next(iter(results[0].state_history.values())).keys())
                for candidate in ("cash", "cumulative_cash", "portfolio_value"):
                    if candidate in keys:
                        default_key = candidate
                        break
                if default_key is None and keys:
                    default_key = keys[0]
            render_simulation_analysis(
                results,
                key_prefix="results_viz",
                default_metric=default_key,
                height=480,
            )
        else:
            st.info("Install pandas and plotly for interactive path analysis.")

# =============================================================================
# SECTION: SCENARIOS (library)
# =============================================================================
else:
    st.header("Scenarios")
    st.markdown(
        "Save, load, duplicate, and import/export scenario configurations. "
        "Use **Save As…** in the sidebar to keep variants without overwriting."
    )
    render_library_manager(key_prefix="scenarios_lib")

st.caption("Financial Simulator · Streamlit + Plotly + Pydantic")
