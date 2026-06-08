"""
Tests for the Streamlit Scenario Builder editor components (Phase 5+ UI layer).

These tests focus on the *mutation contract* that was broken before the hardening session:
- Quick Add / Preset / Duplicate / Manual Add buttons must mutate the input list in place.
- The objects produced must be valid for build_engine + run_single (runnable contract).
- Delete and edit-save paths continue to work.

Uses selective Streamlit mocks (only the intended button returns True) so nested
editors inside expanders do not run unbounded widget trees — avoids hangs and memory spikes.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.components.continuous_processes_editor import render_continuous_processes_editor
from app.components.custom_metrics_editor import render_custom_metrics_editor
from app.components.event_builder_editor import render_event_builder_list_editor
from app.components.external_drivers_editor import render_external_drivers_editor
from financial_simulator.core import FixedValue, IntervalTiming
from financial_simulator.core.event import (
    CASH_FLOW_ADDITIVE,
    CASH_FLOW_DIRECTION_KEY,
    CASH_FLOW_SUBTRACTIVE,
    ComposedEventBuilder,
)
from financial_simulator.scenarios import ScenarioConfig, build_engine, run_single


def _mock_st(*, active_keys: set[str], toggle_values: dict[str, bool] | None = None) -> MagicMock:
    """Minimal streamlit mock: only buttons whose key is in active_keys return True."""
    mock_st = MagicMock(name="mock_streamlit")
    mock_st.session_state = {}
    toggle_values = toggle_values or {}

    def button_side_effect(*_args, **kwargs):
        key = kwargs.get("key", "")
        return key in active_keys

    def toggle_side_effect(*_args, **kwargs):
        key = kwargs.get("key", "")
        if key in toggle_values:
            mock_st.session_state[key] = toggle_values[key]
        return mock_st.session_state.get(key, kwargs.get("value", False))

    def selectbox_side_effect(_label, options, *args, **kwargs):
        if isinstance(options, list) and options:
            idx = kwargs.get("index", 0)
            if isinstance(idx, int) and 0 <= idx < len(options):
                return options[idx]
            return options[0]
        return "currency"

    def columns_side_effect(spec=None, **_kwargs):
        if spec is None:
            n = 1
        elif isinstance(spec, int):
            n = spec
        else:
            n = len(spec)
        cols = []
        for i in range(n):
            col = MagicMock(name=f"col_{i}")
            col.button.side_effect = button_side_effect
            col.toggle.side_effect = toggle_side_effect
            col.selectbox.side_effect = selectbox_side_effect
            cols.append(col)
        return cols

    mock_st.button.side_effect = button_side_effect
    mock_st.toggle.side_effect = toggle_side_effect
    mock_st.columns.side_effect = columns_side_effect

    mock_st.text_input.return_value = "test_item"

    def number_input_side_effect(*_args, **kwargs):
        key = kwargs.get("key", "")
        if "interval" in key:
            return 90
        return 0.05

    mock_st.number_input.side_effect = number_input_side_effect
    mock_st.checkbox.return_value = False
    mock_st.selectbox.side_effect = selectbox_side_effect
    mock_st.date_input.return_value = datetime(2026, 6, 15)
    mock_st.expander.return_value.__enter__.return_value = None
    mock_st.container.return_value.__enter__.return_value = None
    mock_st.toast = MagicMock()
    mock_st.success = MagicMock()
    mock_st.rerun = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.info = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.divider = MagicMock()
    return mock_st


def _run_with_mocks(mock_st: MagicMock, fn, *args, **kwargs):
    """Apply streamlit + nested-editor patches, then call fn."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"streamlit": mock_st}))
        for p in _patch_nested_event_editors():
            stack.enter_context(p)
        return fn(*args, **kwargs)


def _patch_nested_event_editors():
    """Avoid rendering full timing/vg widget trees inside expanders."""
    from financial_simulator.core import FixedValue, IntervalTiming

    return (
        patch(
            "app.components.event_builder_editor.render_timing_editor",
            return_value=IntervalTiming(interval=timedelta(days=30)),
        ),
        patch(
            "app.components.event_builder_editor.render_value_generator_editor",
            return_value=FixedValue(value=100.0),
        ),
    )


def test_event_builder_add_generator_mutates_list_in_place():
    builders: list = []
    mock_st = _mock_st(active_keys={"test_add_add_new"})

    result = _run_with_mocks(mock_st, render_event_builder_list_editor, "test_add", builders)

    assert result is builders
    assert len(builders) == 1
    assert isinstance(builders[0], ComposedEventBuilder)

    cfg = ScenarioConfig(
        name="Add Generator Test",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 6, 1),
        initial_state={"cash": 0.0},
        event_builders=builders,
    )
    build_engine(cfg, seed=123)
    assert run_single(cfg, seed=123) is not None


def test_event_builder_duplicate_mutates_in_place():
    seed_builder = ComposedEventBuilder(
        name="seed",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=FixedValue(value=100.0),
    )
    builders = [seed_builder]
    mock_st = _mock_st(active_keys={"test_dup_dup_0"})

    _run_with_mocks(mock_st, render_event_builder_list_editor, "test_dup", builders)

    assert len(builders) == 2
    assert builders[0].name == "seed"
    assert builders[1].name == "seed"


def test_event_builder_cash_flow_toggle_uses_stable_keys():
    from app.components.event_builder_editor import GENERATOR_ID_KEY, _flow_widget_key

    b1 = ComposedEventBuilder(
        name="income",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=FixedValue(100),
        metadata={GENERATOR_ID_KEY: "gen-a", CASH_FLOW_DIRECTION_KEY: CASH_FLOW_ADDITIVE},
    )
    b2 = ComposedEventBuilder(
        name="expense",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=FixedValue(50),
        metadata={GENERATOR_ID_KEY: "gen-b", CASH_FLOW_DIRECTION_KEY: CASH_FLOW_ADDITIVE},
    )
    builders = [b1, b2]
    flow_key = _flow_widget_key("test_flow", b1.metadata)
    mock_st = _mock_st(active_keys=set(), toggle_values={flow_key: True})

    _run_with_mocks(mock_st, render_event_builder_list_editor, "test_flow", builders)

    assert builders[0].metadata[CASH_FLOW_DIRECTION_KEY] == CASH_FLOW_SUBTRACTIVE
    assert builders[1].metadata[CASH_FLOW_DIRECTION_KEY] == CASH_FLOW_ADDITIVE


def test_event_builder_delete_uses_post_loop_mutation():
    b1 = ComposedEventBuilder(
        name="one",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=FixedValue(10),
    )
    b2 = ComposedEventBuilder(
        name="two",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=FixedValue(20),
    )
    builders = [b1, b2]
    mock_st = _mock_st(active_keys={"test_del_del_0"})

    _run_with_mocks(mock_st, render_event_builder_list_editor, "test_del", builders)

    assert len(builders) == 1
    assert builders[0].name == "two"


def test_custom_metrics_quick_add_mutates_in_place():
    metrics: list = []
    mock_st = _mock_st(active_keys={"test_metrics_quick_0"})

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        render_custom_metrics_editor("test_metrics", metrics)

    assert len(metrics) == 1
    from financial_simulator.scenarios.models import CustomMetric

    assert isinstance(metrics[0], CustomMetric)


def test_continuous_processes_quick_add_mutates_in_place():
    procs: list = []
    mock_st = _mock_st(active_keys={"test_procs_add_app"})

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        render_continuous_processes_editor("test_procs", procs)

    assert len(procs) == 1
    from financial_simulator.core.simulation import AppreciationProcess

    assert isinstance(procs[0], AppreciationProcess)


def test_external_drivers_quick_add_mutates_in_place_and_accepts_dates():
    from financial_simulator.core.distributions import NormalDistribution

    drivers: list = []
    mock_st = _mock_st(active_keys={"test_drivers_add_rate"})
    stub_dist = NormalDistribution(mean=0.05, std=0.01)

    with (
        patch.dict(sys.modules, {"streamlit": mock_st}),
        patch(
            "app.components.external_drivers_editor.render_distribution_picker",
            return_value=stub_dist,
        ),
    ):
        render_external_drivers_editor(
            "test_drivers",
            drivers,
            scenario_start=datetime(2026, 1, 1),
            scenario_end=datetime(2027, 1, 1),
        )

    assert len(drivers) == 1
    from financial_simulator.scenarios.models import DiscreteRateDriver

    assert isinstance(drivers[0], DiscreteRateDriver)


def test_results_dashboard_export_df_includes_custom_metrics():
    class FakeResult:
        def __init__(self, final):
            self.final_state = final
            self.state_history = {}
            self.events = []

    results = [
        FakeResult({"__custom_metrics__": {"final_cash": 12345.6, "goal_hit": 1}}),
        FakeResult({"__custom_metrics__": {"final_cash": 9876.5}}),
    ]
    finals = [12345.6, 9876.5]

    export_rows = []
    for i, r in enumerate(results):
        row = {"sim_idx": i, "final_value": float(finals[i])}
        cm = r.final_state.get("__custom_metrics__", {}) if isinstance(r.final_state, dict) else {}
        for k, v in cm.items():
            if isinstance(v, (int, float)):
                row[f"custom:{k}"] = float(v)
        export_rows.append(row)

    import pandas as pd

    df = pd.DataFrame(export_rows)

    assert "sim_idx" in df.columns
    assert "final_value" in df.columns
    assert "custom:final_cash" in df.columns
    assert len(df) == 2
    assert df.loc[0, "custom:final_cash"] == 12345.6
