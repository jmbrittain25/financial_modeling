"""
Event Builder Editor — Manage custom event generators for a scenario.

Each generator combines timing (when it fires) with a value generator
(including custom distributions). Users add as many generators as needed.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from app.components.scenario_links import (
    CONTINUOUS_PROCESS_META_KEY,
    config_from_process,
    find_linked_process,
    format_macro_links,
    get_continuous_process_config,
    sync_continuous_processes,
)
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

GENERATOR_ID_KEY = "_generator_id"


def _flow_label(metadata: dict) -> str:
    if metadata.get(CASH_FLOW_DIRECTION_KEY) == CASH_FLOW_SUBTRACTIVE:
        return "Subtracts from cash"
    return "Adds to cash"


def _ensure_generator_id(metadata: dict) -> str:
    """Stable per-generator id for Streamlit widget keys (survives reorder/delete)."""
    gen_id = metadata.get(GENERATOR_ID_KEY)
    if not gen_id:
        gen_id = str(uuid.uuid4())
        metadata[GENERATOR_ID_KEY] = gen_id
    return gen_id


def _flow_widget_key(key_prefix: str, metadata: dict) -> str:
    return f"{key_prefix}_flow_{_ensure_generator_id(metadata)}"


def _sync_flow_widget_state(st, flow_key: str, metadata: dict) -> None:
    """Seed toggle session state from persisted metadata when the widget is new."""
    if flow_key not in st.session_state:
        st.session_state[flow_key] = metadata.get(CASH_FLOW_DIRECTION_KEY) == CASH_FLOW_SUBTRACTIVE


def _render_continuous_process_editor(
    st,
    key_prefix: str,
    gen_id: str,
    metadata: dict,
    linked_process: Any | None,
) -> dict | None:
    """Optional background evolution stored on generator metadata."""
    from app.components.continuous_processes_editor import PROCESS_TYPE_INFO

    existing = get_continuous_process_config(metadata)
    if existing is None and linked_process is not None:
        existing = config_from_process(linked_process)

    enabled = existing is not None and existing.get("enabled", False)
    enable = st.checkbox(
        "Evolve a state variable between events",
        value=enabled,
        key=f"{key_prefix}_bg_enable_{gen_id}",
        help="Background growth or volatility applied continuously (e.g. portfolio drift).",
    )
    if not enable:
        return None

    type_labels = {k: v["label"] for k, v in PROCESS_TYPE_INFO.items()}
    current_type = (existing or {}).get("type", "appreciation")
    type_keys = list(PROCESS_TYPE_INFO.keys())
    chosen_label = st.selectbox(
        "Background process type",
        options=[type_labels[k] for k in type_keys],
        index=type_keys.index(current_type) if current_type in type_keys else 0,
        key=f"{key_prefix}_bg_type_{gen_id}",
    )
    label_to_key = {type_labels[k]: k for k in type_keys}
    ptype = label_to_key[chosen_label]

    var = st.text_input(
        "State variable to evolve",
        value=(existing or {}).get("var", "cash"),
        key=f"{key_prefix}_bg_var_{gen_id}",
    )

    config: dict = {"enabled": True, "type": ptype, "var": var}
    if ptype == "appreciation":
        config["rate"] = st.number_input(
            "Annual growth rate",
            value=float((existing or {}).get("rate", 0.04)),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_bg_rate_{gen_id}",
        )
    elif ptype == "gbm":
        c1, c2 = st.columns(2)
        config["drift"] = c1.number_input(
            "Drift",
            value=float((existing or {}).get("drift", 0.08)),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_bg_drift_{gen_id}",
        )
        config["volatility"] = c2.number_input(
            "Volatility",
            value=float((existing or {}).get("volatility", 0.16)),
            min_value=0.001,
            step=0.01,
            format="%.3f",
            key=f"{key_prefix}_bg_vol_{gen_id}",
        )
    elif ptype == "mean_reverting":
        c1, c2, c3 = st.columns(3)
        config["long_term_mean"] = c1.number_input(
            "Long-term mean",
            value=float((existing or {}).get("long_term_mean", 0.045)),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_bg_ltm_{gen_id}",
        )
        config["speed"] = c2.number_input(
            "Reversion speed",
            value=float((existing or {}).get("speed", 1.2)),
            min_value=0.01,
            step=0.1,
            key=f"{key_prefix}_bg_speed_{gen_id}",
        )
        config["volatility"] = c3.number_input(
            "Volatility",
            value=float((existing or {}).get("volatility", 0.008)),
            min_value=0.001,
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_bg_mrvol_{gen_id}",
        )
    return config


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
    continuous_processes: list[Any] | None = None,
    macro_environment: Any | None = None,
) -> tuple[list[ComposedEventBuilder], list[Any]]:
    """
    Interactive list manager for event generators.

    Users add generators one at a time, each with custom timing and value
    logic (including per-generator distributions).
    """
    import streamlit as st

    if builders is None:
        builders = []
    if continuous_processes is None:
        continuous_processes = []
    environment_keys = list(macro_environment.state_keys()) if macro_environment else []

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
                flow_key = _flow_widget_key(key_prefix, eb.metadata)
                _sync_flow_widget_state(st, flow_key, eb.metadata)

                c1.markdown(
                    f"**{eb.name or f'Generator {idx + 1}'}** — `{eb.metadata.get('type', 'custom')}`"
                )

                subtract_on = c2.toggle(
                    "Subtract from cash",
                    key=flow_key,
                    help="Off = add to cash. On = subtract from cash.",
                )
                eb.metadata[CASH_FLOW_DIRECTION_KEY] = (
                    CASH_FLOW_SUBTRACTIVE if subtract_on else CASH_FLOW_ADDITIVE
                )
                c1.caption(_flow_label(eb.metadata))
                driver_link = format_macro_links(eb, macro_environment)
                if driver_link:
                    c1.caption(f"Environment: {driver_link}")
                bg_cfg = get_continuous_process_config(eb.metadata)
                if bg_cfg and bg_cfg.get("enabled"):
                    c1.caption(
                        f"Background: `{bg_cfg.get('type', 'process')}` on `{bg_cfg.get('var', '?')}`"
                    )

                if c3.button("✏️ Edit", key=f"{key_prefix}_edit_{idx}", use_container_width=True):
                    st.session_state[f"{key_prefix}_editing_idx"] = idx
                    st.rerun()
                if c4.button("📋 Dup", key=f"{key_prefix}_dup_{idx}", use_container_width=True):
                    dup = eb.model_copy(deep=True)
                    dup.metadata.pop(GENERATOR_ID_KEY, None)
                    _ensure_generator_id(dup.metadata)
                    builders.append(dup)
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
        gen_id = _ensure_generator_id(current.metadata)
        linked_proc = find_linked_process(gen_id, continuous_processes)

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
        if environment_keys:
            st.caption(
                f"Environment state keys available: `{', '.join(environment_keys)}` — "
                "use these in loan rate / dividend keys to link generators to the macro environment."
            )

        new_vg = render_value_generator_editor(
            key_prefix=f"{key_prefix}_edit_vg_{editing_idx}",
            initial=current.value_gen,
            environment_keys=environment_keys,
        )

        st.markdown("#### Background evolution")
        bg_config = _render_continuous_process_editor(
            st,
            key_prefix,
            gen_id,
            current.metadata,
            linked_proc,
        )

        c1, c2 = st.columns(2)
        if c1.button(
            "✅ Save Changes", type="primary", key=f"{key_prefix}_save_edit_{editing_idx}"
        ):
            saved_metadata = {
                **current.metadata,
                "type": meta_type,
                CASH_FLOW_DIRECTION_KEY: flow_choice,
            }
            if bg_config:
                saved_metadata[CONTINUOUS_PROCESS_META_KEY] = bg_config
            else:
                saved_metadata.pop(CONTINUOUS_PROCESS_META_KEY, None)
            builders[editing_idx] = ComposedEventBuilder(
                name=name or None,
                timing=new_timing,
                value_gen=new_vg,
                metadata=saved_metadata,
            )
            flow_key = _flow_widget_key(key_prefix, saved_metadata)
            st.session_state.pop(flow_key, None)
            st.session_state.pop(f"{key_prefix}_editing_idx", None)
            continuous_processes = sync_continuous_processes(builders, continuous_processes)
            st.success("Generator updated.")
            st.rerun()

        if c2.button("Cancel", key=f"{key_prefix}_cancel_edit_{editing_idx}"):
            st.session_state.pop(f"{key_prefix}_editing_idx", None)
            st.rerun()

    continuous_processes = sync_continuous_processes(builders, continuous_processes)
    return builders, continuous_processes


# Kept for backward compatibility with tests that referenced presets
PRESET_DEFS: list = []


__all__ = [
    "render_event_builder_list_editor",
    "PRESET_DEFS",
]
