"""
Library Manager — Unified browser for user-saved Scenarios and Distributions.

Supports:
- Loading from persisted user library (via new persistence helpers)
- Search + filtering
- Delete
- Load as copy
- Preview
"""

from __future__ import annotations

import streamlit as st

from financial_simulator.scenarios import (
    DistributionLibrary,
    ScenarioLibrary,
    load_user_distribution_library,
    load_user_scenario_library,
    save_user_distribution_library,
    save_user_scenario,
)


def render_library_manager(key_prefix: str = "lib"):
    """Main unified library browser for both scenarios and distributions."""

    tab1, tab2 = st.tabs(["📁 My Scenarios", "🎲 My Distributions"])

    with tab1:
        _render_scenario_library(key_prefix + "_scenarios")

    with tab2:
        _render_distribution_library(key_prefix + "_dists")


def _render_scenario_library(key_prefix: str):
    st.markdown("#### Your Saved Scenarios")

    try:
        lib: ScenarioLibrary = load_user_scenario_library()
    except Exception:
        lib = ScenarioLibrary()

    if not lib.scenarios:
        st.info("You haven't saved any scenarios yet. Build something in the Scenario Builder and click **Save to My Library**.")
        return

    # Simple search
    search = st.text_input("Search your scenarios", key=f"{key_prefix}_search")

    filtered = [
        s for s in lib.scenarios
        if not search or search.lower() in (s.name or "").lower() or search.lower() in (s.description or "").lower()
    ]

    for i, scenario in enumerate(filtered):
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])

            c1.markdown(f"**{scenario.name}**")
            if scenario.description:
                c1.caption(scenario.description[:120])

            summary = scenario.summary()
            c1.caption(f"{summary['horizon_years']}y • {summary['num_event_builders']} events • {summary['num_custom_metrics']} metrics")

            if c2.button("Load", key=f"{key_prefix}_load_{i}"):
                st.session_state["current_scenario"] = scenario.clone()
                st.success(f"Loaded '{scenario.name}' into the builder.")
                st.rerun()

            if c3.button("Load as Copy", key=f"{key_prefix}_copy_{i}"):
                copy = scenario.clone()
                copy.name = f"{scenario.name} (Copy)"
                st.session_state["current_scenario"] = copy
                st.success("Loaded copy.")
                st.rerun()

            if c4.button("🗑️", key=f"{key_prefix}_del_{i}"):
                if lib.remove(scenario.name):
                    # Persist deletion
                    # Simple approach: resave the remaining ones
                    for s in lib.scenarios:
                        save_user_scenario(s, overwrite=True)
                    st.warning(f"Deleted '{scenario.name}'")
                    st.rerun()


def _render_distribution_library(key_prefix: str):
    st.markdown("#### Your Saved Distributions")

    try:
        lib: DistributionLibrary = load_user_distribution_library()
    except Exception:
        lib = DistributionLibrary()

    if not lib.distributions:
        st.info("You haven't saved any custom distributions yet. Use the Distribution picker and click **Save to Library**.")
        return

    search = st.text_input("Search distributions", key=f"{key_prefix}_search")

    filtered = [
        d for d in lib.distributions
        if not search or search.lower() in d.name.lower()
    ]

    for i, dist in enumerate(filtered):
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 1, 1])

            c1.markdown(f"**{dist.name}**")
            if dist.description:
                c1.caption(dist.description)
            if dist.units:
                c1.caption(f"Units: {dist.units}")

            if c2.button("Load into Editor", key=f"{key_prefix}_load_dist_{i}"):
                st.session_state[f"{key_prefix}_selected_dist"] = dist.dist
                st.success(f"Loaded '{dist.name}' — switch to Distribution Library mode to tweak it.")
                st.rerun()

            if c3.button("🗑️", key=f"{key_prefix}_del_dist_{i}"):
                if lib.remove(dist.id):
                    save_user_distribution_library(lib)
                    st.warning(f"Deleted '{dist.name}'")
                    st.rerun()


__all__ = ["render_library_manager"]
