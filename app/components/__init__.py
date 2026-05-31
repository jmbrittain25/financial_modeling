"""
Streamlit UI components for the Financial Simulator Scenario Builder.

These are reusable, self-contained render functions that can be composed
in streamlit_app.py or future multipage apps.

Key components:
- distribution_viz: interactive distribution configuration with live Plotly
- scenario_forms: horizon, initial state, event builders, drivers, metrics
- persistence_ui: save/load, templates, import/export
"""

from __future__ import annotations

from .distribution_viz import render_distribution_picker
from .driver_viz import render_external_driver_editor

# Heavy visualization helpers (Plotly + pandas) are optional at import time so that
# tests and minimal environments that only need the distribution picker continue to work.
try:
    from .simulation_viz import (
        build_summary_dataframe,
        create_spaghetti_plot,
        discover_numeric_keys,
        render_simulation_analysis,
    )

    _HAS_SIM_VIZ = True
except Exception:  # pandas/plotly not installed or other import failure
    _HAS_SIM_VIZ = False

# New Phase 5 interactive scenario builder components (import only when Streamlit context exists)
# These are commented here but will be enabled as the editor components are integrated.
# from .timing_editor import render_timing_editor
# from .value_generator_editor import render_value_generator_editor
# from .event_builder_editor import render_event_builder_list_editor
# etc.

__all__ = [
    "render_distribution_picker",
    "render_external_driver_editor",
]

if _HAS_SIM_VIZ:
    __all__.extend([
        "render_simulation_analysis",
        "discover_numeric_keys",
        "build_summary_dataframe",
        "create_spaghetti_plot",
    ])
