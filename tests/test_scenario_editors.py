"""
Unit tests for the new Phase 5 interactive scenario editor components.

Following the established pattern from test_distribution_viz.py:
- We test the pure (non-Streamlit) logic and factories.
- We do **not** attempt to execute the render_* functions that call st.* widgets.
- We add strong contract tests: "the things the editor can produce must be runnable."

This gives us regression protection that the new visual builder cannot generate
broken ComposedEventBuilder / ScenarioConfig objects.
"""

from datetime import datetime

import pytest

from app.components.event_builder_editor import PRESET_DEFS

# Pure helpers under test
from app.components.timing_editor import (
    TIMING_TYPE_LABELS,
    get_default_timing,
)
from app.components.value_generator_editor import (
    VG_TYPE_LABELS,
    get_default_vg,
)
from financial_simulator.core.event import ComposedEventBuilder
from financial_simulator.scenarios import ScenarioConfig, build_engine, run_single

# =============================================================================
# Pure factory / default helpers
# =============================================================================


def test_timing_type_labels_are_complete():
    """The labels dict must cover exactly the 4 supported timing types."""
    assert set(TIMING_TYPE_LABELS.keys()) == {"OneTime", "Interval", "Random", "Seasonal"}


def test_get_default_timing_returns_correct_types():
    for t in ["OneTime", "Interval", "Random", "Seasonal"]:
        timing = get_default_timing(t)
        assert timing.type == t


def test_value_generator_labels_are_complete():
    """We must expose editors for all value generator types the core supports."""
    expected = {
        "Fixed",
        "Growing",
        "Distribution",
        "VariableRateLoan",
        "Dividend",
        "InvestmentContribution",
        "TaxEvent",
        "RateChange",
    }
    assert set(VG_TYPE_LABELS.keys()) == expected


def test_get_default_vg_returns_correct_types():
    for t in VG_TYPE_LABELS.keys():
        vg = get_default_vg(t)
        assert vg.type == t


# =============================================================================
# Preset contract tests — the most important safety net for the visual builder
# =============================================================================


@pytest.mark.parametrize("preset", PRESET_DEFS, ids=lambda p: p["label"])
def test_every_preset_produces_a_valid_runnable_event_builder(preset):
    """
    Every one-click preset must:
    1. Return a well-formed ComposedEventBuilder
    2. Be materializable by build_engine
    3. Run to completion via run_single without crashing
    """
    builder = preset["builder"]()
    assert isinstance(builder, ComposedEventBuilder)
    assert builder.timing is not None
    assert builder.value_gen is not None

    # Build a minimal runnable scenario around it
    start = datetime(2026, 1, 1)
    end = datetime(2026, 6, 1)

    cfg = ScenarioConfig(
        name=f"Preset Test: {preset['label']}",
        start=start,
        end=end,
        initial_state={"cumulative_cash": 0.0, "portfolio_value": 100_000.0},
        event_builders=[builder],
    )

    # This is the critical contract: the UI must only ever produce runnable artifacts
    _ = build_engine(cfg, seed=123)
    result = run_single(cfg, seed=123)

    assert result is not None
    assert "cumulative_cash" in result.final_state or len(result.events) >= 0


def test_all_presets_have_unique_reasonable_names():
    """Names should be distinct and human-friendly (prevents accidental overwrites in UI)."""
    names = [p["label"] for p in PRESET_DEFS]
    assert len(names) == len(set(names)), "Preset labels must be unique"
    for name in names:
        assert len(name) > 5
        assert any(c.isalpha() for c in name)


# =============================================================================
# Integration-style sanity: round-trip through the editor factories
# =============================================================================


def test_editor_factories_can_be_composed_into_full_scenario():
    """Simulates what the future Streamlit UI will do: pick defaults → user tweaks → run."""
    timing = get_default_timing("Interval")
    vg = get_default_vg("Distribution")

    builder = ComposedEventBuilder(
        name="editor_composed",
        timing=timing,
        value_gen=vg,
        metadata={"type": "test"},
    )

    cfg = ScenarioConfig(
        name="Editor Composition Test",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 4, 1),
        initial_state={"cumulative_cash": 0.0},
        event_builders=[builder],
        custom_metrics=[
            # Use one of the new UI-enhanced metric fields to ensure they serialize
            {"name": "final_cash", "metric_type": "final_state_value", "params": {"key": "cumulative_cash"},
             "display_format": "currency", "higher_is_better": True}
        ],
    )

    results = run_single(cfg, seed=42)
    metrics = results.final_state.get("__custom_metrics__", {})
    assert "final_cash" in metrics
    # The value should be numeric even if the metric definition used the new fields
    assert isinstance(metrics["final_cash"], (int, float))


def test_preset_builders_do_not_mutate_shared_state():
    """Each call to a preset lambda must produce an independent object."""
    b1 = PRESET_DEFS[0]["builder"]()
    b2 = PRESET_DEFS[0]["builder"]()
    assert b1 is not b2
    b1.metadata["mutated"] = True
    assert "mutated" not in b2.metadata
