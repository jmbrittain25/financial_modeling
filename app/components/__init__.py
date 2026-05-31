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

# New Phase 5 interactive scenario builder components (import only when Streamlit context exists)
# from .timing_editor import render_timing_editor
# from .value_generator_editor import render_value_generator_editor
# from .event_builder_editor import render_event_builder_list_editor

__all__ = [
    "render_distribution_picker",
]
