"""
Event Builder Editor — The heart of the interactive Scenario Builder.

Composes timing + value generator editors, provides a "Quick Add Presets" bar with
realistic, one-click financial patterns, and manages a list of ComposedEventBuilders
with edit / delete / duplicate actions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from app.components.timing_editor import render_timing_editor
from app.components.value_generator_editor import render_value_generator_editor
from financial_simulator.core.distributions import TriangularDistribution
from financial_simulator.core.event import (
    ComposedEventBuilder,
    DistributionValue,
    DividendValue,
    FixedValue,
    GrowingValue,
    IntervalTiming,
    OneTimeTiming,
    SeasonalTiming,
    TaxEventValue,
    VariableRateLoanValue,
)

PRESET_DEFS: list[dict[str, Any]] = [
    {
        "label": "📈 Monthly Salary / Income (+)",
        "builder": lambda: ComposedEventBuilder(
            name="monthly_income",
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=FixedValue(value=6500.0),
            metadata={"type": "income"},
        ),
    },
    {
        "label": "📉 Monthly Variable Opex / Expenses",
        "builder": lambda: ComposedEventBuilder(
            name="monthly_opex",
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=DistributionValue(
                dist=TriangularDistribution(low=-18500, mode=-16200, high=-14100)
            ),
            metadata={"type": "opex"},
        ),
    },
    {
        "label": "🏠 30-yr Fixed Mortgage Payment",
        "builder": lambda: ComposedEventBuilder(
            name="mortgage_payment",
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=VariableRateLoanValue(
                principal=320000.0, initial_rate=0.065, term_months=360, rate_key="mortgage_rate"
            ),
            metadata={"type": "loan_payment"},
        ),
    },
    {
        "label": "💰 Quarterly Dividend Income",
        "builder": lambda: ComposedEventBuilder(
            name="quarterly_dividend",
            timing=IntervalTiming(interval=timedelta(days=90)),
            value_gen=DividendValue(annual_yield=0.028, investment_value_key="portfolio_value"),
            metadata={"type": "dividend"},
        ),
    },
    {
        "label": "📊 One-time Bonus / Windfall",
        "builder": lambda: ComposedEventBuilder(
            name="one_time_bonus",
            timing=OneTimeTiming(time=datetime(2026, 9, 15)),
            value_gen=FixedValue(value=18000.0),
            metadata={"type": "bonus"},
        ),
    },
    {
        "label": "📈 Growing Monthly Contribution (3%/yr)",
        "builder": lambda: ComposedEventBuilder(
            name="growing_contribution",
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=GrowingValue(initial=1200.0, growth_rate=0.03),
            metadata={"type": "contribution"},
        ),
    },
    {
        "label": "🧾 Monthly Tax Drag (22% on taxable income)",
        "builder": lambda: ComposedEventBuilder(
            name="tax_event",
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=TaxEventValue(rate=0.22, base_key="taxable_income", tax_key="tax_paid"),
            metadata={"type": "tax"},
        ),
    },
    {
        "label": "🌊 Seasonal Revenue (Q4 spike)",
        "builder": lambda: ComposedEventBuilder(
            name="seasonal_revenue",
            timing=SeasonalTiming(
                inner=IntervalTiming(interval=timedelta(days=30)), months=[10, 11, 12]
            ),
            value_gen=FixedValue(value=28500.0),
            metadata={"type": "revenue"},
        ),
    },
]


def render_event_builder_list_editor(
    key_prefix: str = "events",
    builders: list[ComposedEventBuilder] | None = None,
) -> list[ComposedEventBuilder]:
    """
    Full interactive list manager for event sources.

    Shows a beautiful preset bar, the current list as editable cards, and an
    "Add custom" flow that opens the full timing + vg editors.
    """
    import streamlit as st

    if builders is None:
        builders = []

    st.markdown("### 📅 Event Sources")
    st.caption(
        "These are the discrete cash-flow generators. Use the quick presets for the most common patterns, or build a fully custom one below."
    )

    # Preset bar (the magic for non-technical users)
    st.markdown("**Quick Add Presets** — click any to append a realistic, ready-to-run event")
    cols = st.columns(4)
    for i, preset in enumerate(PRESET_DEFS[:8]):
        col = cols[i % 4]
        if col.button(preset["label"], key=f"{key_prefix}_preset_{i}", use_container_width=True):
            new_b = preset["builder"]()
            builders.append(new_b)
            st.toast(f"Added: {new_b.name or 'event'}", icon="✅")
            st.rerun()

    st.divider()

    # Current list
    if not builders:
        st.info("No event sources yet. Click a preset above or use **Add Custom Event** below.")
    else:
        st.markdown(f"**Current event sources ({len(builders)})**")
        to_delete: list[int] = []
        for idx, eb in enumerate(builders):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.markdown(
                    f"**{eb.name or f'Event {idx + 1}'}** — `{eb.metadata.get('type', 'custom')}`"
                )
                if c2.button("✏️ Edit", key=f"{key_prefix}_edit_{idx}", use_container_width=True):
                    st.session_state[f"{key_prefix}_editing_idx"] = idx
                    st.rerun()
                if c3.button("📋 Dup", key=f"{key_prefix}_dup_{idx}", use_container_width=True):
                    builders.append(eb.model_copy(deep=True))
                    st.toast("Duplicated", icon="📋")
                    st.rerun()
                if c4.button("🗑️", key=f"{key_prefix}_del_{idx}", use_container_width=True):
                    to_delete.append(idx)

        # Handle deletions after the loop (safe)
        if to_delete:
            for i in sorted(to_delete, reverse=True):
                del builders[i]
            st.rerun()

    # Editing pane (only one at a time)
    editing_idx = st.session_state.get(f"{key_prefix}_editing_idx")
    if editing_idx is not None and 0 <= editing_idx < len(builders):
        st.markdown("---")
        st.markdown(f"### Editing Event #{editing_idx + 1}")
        current = builders[editing_idx]

        name = st.text_input(
            "Event name (for your reference)",
            value=current.name or "",
            key=f"{key_prefix}_edit_name",
        )
        meta_type = st.text_input(
            "Type tag (used by metrics & reports)",
            value=current.metadata.get("type", "custom"),
            key=f"{key_prefix}_edit_meta",
        )

        # Sub-editors
        new_timing = render_timing_editor(
            key_prefix=f"{key_prefix}_edit_timing_{editing_idx}",
            initial=current.timing,
            help_text="When does this event fire?",
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
                metadata={**current.metadata, "type": meta_type},
            )
            st.session_state.pop(f"{key_prefix}_editing_idx", None)
            st.success("Event updated.")
            st.rerun()

        if c2.button("Cancel", key=f"{key_prefix}_cancel_edit_{editing_idx}"):
            st.session_state.pop(f"{key_prefix}_editing_idx", None)
            st.rerun()

    # Add custom (advanced) flow
    with st.expander("➕ Add Fully Custom Event (advanced)", expanded=False):
        st.caption("For power users or unusual patterns not covered by presets.")
        custom_name = st.text_input("Name", value="custom_event", key=f"{key_prefix}_custom_name")
        custom_meta = st.text_input("Type tag", value="custom", key=f"{key_prefix}_custom_meta")

        t = render_timing_editor(key_prefix=f"{key_prefix}_custom_timing", initial=None)
        v = render_value_generator_editor(key_prefix=f"{key_prefix}_custom_vg", initial=None)

        if st.button("Add Custom Event", type="primary", key=f"{key_prefix}_add_custom"):
            builders.append(
                ComposedEventBuilder(
                    name=custom_name or None,
                    timing=t,
                    value_gen=v,
                    metadata={"type": custom_meta},
                )
            )
            st.toast("Custom event added.", icon="✅")
            st.rerun()

    return builders


__all__ = [
    "render_event_builder_list_editor",
    "PRESET_DEFS",
]
