"""
Custom Metrics Editor — Allows users to define and track custom metrics
beyond the built-in cumulative_cash and property_value.

Supports all 5 metric types from the model with guided forms and good defaults.
"""

from __future__ import annotations

from financial_simulator.scenarios import CustomMetric

METRIC_TYPE_INFO: dict[str, dict[str, str]] = {
    "final_state_value": {
        "label": "Final State Value",
        "description": "Read a single value from the final state (e.g. portfolio_value, cash).",
        "example_params": {"key": "portfolio_value"},
    },
    "sum_positive_events": {
        "label": "Sum of Positive Events",
        "description": "Total of all positive cash flows, optionally filtered by metadata.type.",
        "example_params": {"metadata_type": "revenue"},
    },
    "max_drawdown_on_path": {
        "label": "Maximum Drawdown",
        "description": "Largest peak-to-trough decline on a state variable's path.",
        "example_params": {"state_key": "cumulative_cash"},
    },
    "event_count_by_type": {
        "label": "Event Count by Type",
        "description": "How many events of a certain type fired (or total events if no type).",
        "example_params": {"metadata_type": "contribution"},
    },
    "time_to_threshold": {
        "label": "Time to Threshold",
        "description": "Years until a state variable crosses a threshold (or 999 if never).",
        "example_params": {
            "state_key": "portfolio_value",
            "threshold": 500000,
            "direction": "above",
        },
    },
}


def get_default_metric(metric_type: str) -> CustomMetric:
    info = METRIC_TYPE_INFO.get(metric_type, {})
    params = info.get("example_params", {}).copy()

    if metric_type == "final_state_value":
        return CustomMetric(
            name="final_value",
            metric_type=metric_type,
            params=params,
            display_format="currency",
            higher_is_better=True,
        )
    if metric_type == "sum_positive_events":
        return CustomMetric(
            name="total_inflows",
            metric_type=metric_type,
            params=params,
            display_format="currency",
            higher_is_better=True,
        )
    if metric_type == "max_drawdown_on_path":
        return CustomMetric(
            name="max_drawdown",
            metric_type=metric_type,
            params=params,
            display_format="percent",
            higher_is_better=False,
        )
    if metric_type == "event_count_by_type":
        return CustomMetric(
            name="event_count",
            metric_type=metric_type,
            params=params,
            display_format="count",
            higher_is_better=True,
        )
    if metric_type == "time_to_threshold":
        return CustomMetric(
            name="years_to_goal",
            metric_type=metric_type,
            params=params,
            display_format="years",
            higher_is_better=False,
        )

    return CustomMetric(name="custom_metric", metric_type=metric_type, params=params)


def render_custom_metrics_editor(
    key_prefix: str = "metrics",
    metrics: list[CustomMetric] | None = None,
) -> list[CustomMetric]:
    """
    Editor for custom metrics list.
    """
    import streamlit as st

    if metrics is None:
        metrics = []

    st.markdown("### 📊 Custom Metrics")
    st.caption(
        "Define extra quantities you want tracked on every simulation run. "
        "These appear in results and can be used for decision criteria (e.g. 'probability of reaching goal')."
    )

    # Quick add buttons
    st.markdown("**Quick Add Common Metrics**")
    cols = st.columns(4)
    quick_adds = [
        ("final_state_value", "Final Portfolio Value"),
        ("max_drawdown_on_path", "Max Drawdown"),
        ("sum_positive_events", "Total Inflows"),
        ("time_to_threshold", "Time to Goal"),
    ]

    for i, (mtype, label) in enumerate(quick_adds):
        if cols[i % 4].button(label, key=f"{key_prefix}_quick_{i}", use_container_width=True):
            new_metric = get_default_metric(mtype)
            metrics = metrics + [new_metric]
            st.rerun()

    st.divider()

    if not metrics:
        st.info(
            "No custom metrics defined yet. Use the quick add buttons above or create one manually."
        )
    else:
        to_delete = []
        for idx, m in enumerate(metrics):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{m.name}** — `{m.metric_type}`")

                if c2.button("🗑️ Remove", key=f"{key_prefix}_del_{idx}", use_container_width=True):
                    to_delete.append(idx)

                # Editable fields
                new_name = st.text_input(
                    "Metric Name", value=m.name, key=f"{key_prefix}_name_{idx}"
                )
                new_desc = st.text_input(
                    "Description (optional)", value=m.description, key=f"{key_prefix}_desc_{idx}"
                )

                # Type (read-only for now to keep simple; changing type is advanced)
                st.caption(f"Type: **{m.metric_type}**")

                # Params editor (simple key-value for MVP)
                st.markdown("**Parameters**")
                new_params = {}
                for k, v in (m.params or {}).items():
                    if isinstance(v, (int, float)):
                        new_params[k] = st.number_input(
                            k, value=float(v), key=f"{key_prefix}_param_{idx}_{k}"
                        )
                    else:
                        new_params[k] = st.text_input(
                            k, value=str(v), key=f"{key_prefix}_param_{idx}_{k}"
                        )

                # UI display hints
                c1, c2, c3 = st.columns(3)
                new_format = c1.selectbox(
                    "Display Format",
                    ["currency", "percent", "number", "years", "count"],
                    index=["currency", "percent", "number", "years", "count"].index(
                        m.display_format
                    ),
                    key=f"{key_prefix}_format_{idx}",
                )
                new_higher = c2.selectbox(
                    "Higher is Better?",
                    [True, False, None],
                    index={True: 0, False: 1, None: 2}.get(m.higher_is_better, 2),
                    key=f"{key_prefix}_higher_{idx}",
                )
                new_goal = (
                    c3.number_input(
                        "Goal / Target (optional)",
                        value=m.goal_value if m.goal_value is not None else 0.0,
                        key=f"{key_prefix}_goal_{idx}",
                    )
                    if m.goal_value is not None
                    or st.checkbox("Set a goal value", key=f"{key_prefix}_hasgoal_{idx}")
                    else None
                )

                # Apply updates
                metrics[idx] = CustomMetric(
                    name=new_name,
                    description=new_desc,
                    metric_type=m.metric_type,
                    params=new_params,
                    display_format=new_format,
                    higher_is_better=new_higher,
                    goal_value=new_goal if new_goal not in (0.0, None) else None,
                )

        if to_delete:
            for i in sorted(to_delete, reverse=True):
                del metrics[i]
            st.rerun()

    # Manual add
    with st.expander("➕ Add Custom Metric Manually", expanded=False):
        mtype = st.selectbox(
            "Metric Type",
            list(METRIC_TYPE_INFO.keys()),
            format_func=lambda x: METRIC_TYPE_INFO[x]["label"],
            key=f"{key_prefix}_manual_type",
        )
        info = METRIC_TYPE_INFO[mtype]
        st.caption(info["description"])

        name = st.text_input("Metric Name", value=f"my_{mtype}", key=f"{key_prefix}_manual_name")

        if st.button("Add Metric", key=f"{key_prefix}_manual_add"):
            new_m = get_default_metric(mtype)
            new_m.name = name
            metrics = metrics + [new_m]
            st.success(f"Added {name}")
            st.rerun()

    return metrics


__all__ = ["render_custom_metrics_editor", "METRIC_TYPE_INFO", "get_default_metric"]
