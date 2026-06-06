"""
Library Manager — Browse, load, duplicate, and import/export saved scenarios.
"""

from __future__ import annotations

import streamlit as st

from app.components.scenario_io import render_scenario_export_button, render_scenario_import_panel
from financial_simulator.scenarios import (
    USER_SCENARIOS_DIR,
    ScenarioLibrary,
    delete_user_scenario,
    load_user_scenario_library,
)


def render_library_manager(key_prefix: str = "lib"):
    """Browser for user-saved scenarios with import/export."""

    st.markdown("#### Saved scenarios")
    st.caption(f"Library folder: `{USER_SCENARIOS_DIR}`")

    def _on_file_import(cfg):
        st.session_state["current_scenario"] = cfg
        st.session_state["saved_scenario_name"] = None
        st.toast(f"Imported '{cfg.name}' — open **1 · Setup** to review.", icon="📥")

    render_scenario_import_panel(
        key_prefix=f"{key_prefix}_import",
        on_loaded=_on_file_import,
        label="Import scenario from JSON file",
    )

    st.divider()

    try:
        lib: ScenarioLibrary = load_user_scenario_library()
    except Exception:
        lib = ScenarioLibrary()

    if not lib.scenarios:
        st.info(
            "No saved scenarios yet. Build one in **Setup** and **Event Generators**, "
            "then use **Save** or **Save As…** in the sidebar."
        )
        return

    search = st.text_input("Search", key=f"{key_prefix}_search", placeholder="Filter by name…")

    filtered = [
        s
        for s in lib.scenarios
        if not search
        or search.lower() in (s.name or "").lower()
        or search.lower() in (s.description or "").lower()
    ]

    for i, scenario in enumerate(filtered):
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])

            c1.markdown(f"**{scenario.name}**")
            if scenario.description:
                c1.caption(scenario.description[:120])

            summary = scenario.summary()
            c1.caption(f"{summary['horizon_years']}y · {summary['num_event_builders']} generators")

            if c2.button("Load", key=f"{key_prefix}_load_{i}", use_container_width=True):
                st.session_state["current_scenario"] = scenario.clone()
                st.session_state["saved_scenario_name"] = scenario.name
                st.toast(f"Loaded '{scenario.name}'", icon="📥")
                st.rerun()

            if c3.button("Copy", key=f"{key_prefix}_copy_{i}", use_container_width=True):
                copy = scenario.clone()
                copy.name = f"{scenario.name} (Copy)"
                st.session_state["current_scenario"] = copy
                st.session_state["saved_scenario_name"] = None
                st.toast("Loaded as copy — edit and Save As… to keep both versions.", icon="📋")
                st.rerun()

            with c4:
                render_scenario_export_button(
                    scenario,
                    key_prefix=f"{key_prefix}_export_{i}",
                    label="Export",
                )

            if c5.button("🗑️", key=f"{key_prefix}_del_{i}"):
                deleted = delete_user_scenario(scenario.name) or lib.remove(scenario.name)
                if deleted:
                    st.toast(f"Deleted '{scenario.name}'", icon="🗑️")
                    st.rerun()


__all__ = ["render_library_manager"]
