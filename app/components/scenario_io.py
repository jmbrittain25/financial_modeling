"""
Scenario import / export helpers for JSON configuration files.

Used in the sidebar, library manager, and results section so users can
iterate on scenarios outside the app.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import streamlit as st

from financial_simulator.scenarios import ScenarioConfig, scenario_from_json, scenario_to_json


def export_scenario_json(cfg: ScenarioConfig) -> str:
    """Serialize a scenario to a pretty-printed JSON string."""
    return scenario_to_json(cfg)


def import_scenario_json(text: str) -> ScenarioConfig:
    """Parse and validate a scenario JSON string."""
    return scenario_from_json(text)


def _scenario_from_upload(uploaded) -> ScenarioConfig:
    """Parse an uploaded JSON file, including full result bundles."""
    text = uploaded.getvalue().decode("utf-8")
    data = json.loads(text)
    if isinstance(data, dict) and "scenario" in data:
        return ScenarioConfig.from_dict(data["scenario"])
    return import_scenario_json(text)


def render_scenario_import_panel(
    key_prefix: str = "scenario_io",
    *,
    on_loaded: Callable[[ScenarioConfig], None] | None = None,
    label: str = "Import scenario configuration",
    button_label: str | None = None,
) -> ScenarioConfig | None:
    """
    File picker for scenario JSON. Imports once per selected file and calls
    on_loaded(cfg) on success.
    """
    uploaded = st.file_uploader(
        button_label or label,
        type=["json"],
        key=f"{key_prefix}_upload",
        help="Choose a .json scenario file exported from this app.",
        label_visibility="visible" if button_label is None else "collapsed",
    )

    if uploaded is None:
        return None

    file_id = f"{uploaded.file_id}:{uploaded.name}:{uploaded.size}"
    seen_key = f"{key_prefix}_imported_file_id"
    if st.session_state.get(seen_key) == file_id:
        return None

    try:
        cfg = _scenario_from_upload(uploaded)
        st.session_state[seen_key] = file_id
        if on_loaded:
            on_loaded(cfg)
        return cfg
    except Exception as e:
        st.error(f"Could not parse scenario file: {e}")
        return None


def render_scenario_export_button(
    cfg: ScenarioConfig,
    key_prefix: str = "scenario_io",
    *,
    file_name: str | None = None,
    label: str = "Export scenario (JSON)",
) -> None:
    """Download button for the full scenario configuration."""
    safe_name = (cfg.name or "scenario").replace(" ", "_")
    fname = file_name or f"{safe_name}.json"
    st.download_button(
        label,
        data=export_scenario_json(cfg),
        file_name=fname,
        mime="application/json",
        key=f"{key_prefix}_export",
    )


def render_scenario_import_export_row(
    cfg: ScenarioConfig,
    key_prefix: str = "scenario_io",
    *,
    on_loaded: Callable[[ScenarioConfig], None] | None = None,
) -> ScenarioConfig | None:
    """Side-by-side import uploader and export download for compact layouts."""
    col_import, col_export = st.columns(2)
    loaded: ScenarioConfig | None = None
    with col_import:
        loaded = render_scenario_import_panel(
            key_prefix=f"{key_prefix}_import",
            on_loaded=on_loaded,
            label="Import JSON",
        )
    with col_export:
        render_scenario_export_button(cfg, key_prefix=f"{key_prefix}_export")
    return loaded


def build_results_bundle(
    cfg: ScenarioConfig,
    results_summary: dict,
) -> str:
    """Package scenario config + run metadata into one JSON document for archival."""
    bundle = {
        "scenario": json.loads(export_scenario_json(cfg)),
        "run_summary": results_summary,
    }
    return json.dumps(bundle, indent=2)


__all__ = [
    "export_scenario_json",
    "import_scenario_json",
    "render_scenario_import_panel",
    "render_scenario_export_button",
    "render_scenario_import_export_row",
    "build_results_bundle",
]
