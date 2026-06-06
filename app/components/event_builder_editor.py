"""
Event Builder Editor — Manage custom event generators for a scenario.

Each generator combines timing (when it fires) with a value generator
(including custom distributions). Users add as many generators as needed.
"""

from __future__ import annotations

from datetime import timedelta

from app.components.timing_editor import render_timing_editor
from app.components.value_generator_editor import render_value_generator_editor
from financial_simulator.core.distributions import TriangularDistribution
from financial_simulator.core.event import (
    CASH_FLOW_ADDITIVE,
    CASH_FLOW_DIRECTION_KEY,
    CASH_FLOW_SUBTRACTIVE,
    ComposedEventBuilder,
    DistributionValue,
    IntervalTiming,
)


def _flow_label(metadata: dict) -> str:
    if metadata.get(CASH_FLOW_DIRECTION_KEY) == CASH_FLOW_SUBTRACTIVE:
        return "Subtracts from cash"
    return "Adds to cash"


def _default_generator() -> ComposedEventBuilder:
    """Blank-slate generator — user configures timing and distribution."""
    return ComposedEventBuilder(
        name="generator",
        timing=IntervalTiming(interval=timedelta(days=30)),
        value_gen=DistributionValue(dist=TriangularDistribution(low=50.0, mode=80.0, high=150.0)),
        metadata={"type": "custom", CASH_FLOW_DIRECTION_KEY: CASH_FLOW_ADDITIVE},
    )


def render_event_builder_list_editor(
    key_prefix: str = "events",
    builders: list[ComposedEventBuilder] | None = None,
) -> list[ComposedEventBuilder]:
    """
    Interactive list manager for event generators.

    Users add generators one at a time, each with custom timing and value
    logic (including per-generator distributions).
    """
    import streamlit as st

    if builders is None:
        builders = []

    st.caption(
        "Enter positive amounts only. Use **Add to cash** / **Subtract from cash** on each "
        "generator to control whether events increase or decrease your cash balance."
    )

    if st.button("➕ Add Generator", type="primary", key=f"{key_prefix}_add_new"):
        builders.append(_default_generator())
        st.session_state[f"{key_prefix}_editing_idx"] = len(builders) - 1
        st.toast("New generator added — configure it below.", icon="✅")
        st.rerun()

    if not builders:
        st.info("No generators yet. Click **Add Generator** to create your first one.")
    else:
        st.markdown(f"**Generators ({len(builders)})**")
        to_delete: list[int] = []
        for idx, eb in enumerate(builders):
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
                c1.markdown(
                    f"**{eb.name or f'Generator {idx + 1}'}** — `{eb.metadata.get('type', 'custom')}`"
                )
                c1.caption(_flow_label(eb.metadata))

                is_subtractive = eb.metadata.get(CASH_FLOW_DIRECTION_KEY) == CASH_FLOW_SUBTRACTIVE
                if c2.toggle(
                    "Subtract from cash",
                    value=is_subtractive,
                    key=f"{key_prefix}_flow_{idx}",
                    help="Off = add to cash. On = subtract from cash.",
                ):
                    eb.metadata[CASH_FLOW_DIRECTION_KEY] = CASH_FLOW_SUBTRACTIVE
                else:
                    eb.metadata[CASH_FLOW_DIRECTION_KEY] = CASH_FLOW_ADDITIVE

                if c3.button("✏️ Edit", key=f"{key_prefix}_edit_{idx}", use_container_width=True):
                    st.session_state[f"{key_prefix}_editing_idx"] = idx
                    st.rerun()
                if c4.button("📋 Dup", key=f"{key_prefix}_dup_{idx}", use_container_width=True):
                    builders.append(eb.model_copy(deep=True))
                    st.toast("Duplicated", icon="📋")
                    st.rerun()
                if c5.button("🗑️", key=f"{key_prefix}_del_{idx}", use_container_width=True):
                    to_delete.append(idx)

        if to_delete:
            for i in sorted(to_delete, reverse=True):
                del builders[i]
            editing = st.session_state.get(f"{key_prefix}_editing_idx")
            if editing is not None and editing >= len(builders):
                st.session_state.pop(f"{key_prefix}_editing_idx", None)
            st.rerun()

    editing_idx = st.session_state.get(f"{key_prefix}_editing_idx")
    if editing_idx is not None and 0 <= editing_idx < len(builders):
        st.markdown("---")
        st.markdown(f"### Editing Generator #{editing_idx + 1}")
        current = builders[editing_idx]

        name = st.text_input(
            "Name",
            value=current.name or "",
            key=f"{key_prefix}_edit_name",
        )
        meta_type = st.text_input(
            "Category tag (optional, for your reference)",
            value=current.metadata.get("type", "custom"),
            key=f"{key_prefix}_edit_meta",
        )

        current_direction = current.metadata.get(CASH_FLOW_DIRECTION_KEY, CASH_FLOW_ADDITIVE)
        flow_choice = st.radio(
            "Cash impact",
            options=[CASH_FLOW_ADDITIVE, CASH_FLOW_SUBTRACTIVE],
            format_func=lambda x: (
                "Add to cash" if x == CASH_FLOW_ADDITIVE else "Subtract from cash"
            ),
            index=0 if current_direction != CASH_FLOW_SUBTRACTIVE else 1,
            horizontal=True,
            key=f"{key_prefix}_edit_flow_{editing_idx}",
        )

        new_timing = render_timing_editor(
            key_prefix=f"{key_prefix}_edit_timing_{editing_idx}",
            initial=current.timing,
            help_text="When does this generator fire?",
        )
        new_vg = render_value_generator_editor(
            key_prefix=f"{key_prefix}_edit_vg_{editing_idx}",
            initial=current.value_gen,
        )

        c1, c2 = st.columns(2)
        if c1.button(
            "✅ Save Changes", type="primary", key=f"{key_prefix}_save_edit_{editing_idx}"
        ):
            builders[editing_idx] = ComposedEventBuilder(
                name=name or None,
                timing=new_timing,
                value_gen=new_vg,
                metadata={
                    **current.metadata,
                    "type": meta_type,
                    CASH_FLOW_DIRECTION_KEY: flow_choice,
                },
            )
            st.session_state.pop(f"{key_prefix}_editing_idx", None)
            st.success("Generator updated.")
            st.rerun()

        if c2.button("Cancel", key=f"{key_prefix}_cancel_edit_{editing_idx}"):
            st.session_state.pop(f"{key_prefix}_editing_idx", None)
            st.rerun()

    return builders


# Kept for backward compatibility with tests that referenced presets
PRESET_DEFS: list = []


__all__ = [
    "render_event_builder_list_editor",
    "PRESET_DEFS",
]
