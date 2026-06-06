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


def render_scenario_import_panel(
    key_prefix: str = "scenario_io",
    *,
    on_loaded: Callable[[ScenarioConfig], None] | None = None,
    label: str = "Import scenario configuration",
) -> ScenarioConfig | None:
    """
    File uploader + optional paste area. Returns loaded ScenarioConfig or None.
    Calls on_loaded(cfg) when a file is successfully parsed.
    """
    uploaded = st.file_uploader(
        label,
        type=["json"],
        key=f"{key_prefix}_upload",
        help="Upload a .json scenario file exported from this app.",
    )

    if uploaded is not None:
        try:
            text = uploaded.getvalue().decode("utf-8")
            cfg = import_scenario_json(text)
            if on_loaded:
                on_loaded(cfg)
            return cfg
        except Exception as e:
            st.error(f"Could not parse scenario file: {e}")
            return None

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
