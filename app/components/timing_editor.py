"""
Timing Editor Component — Interactive editor for AnyTiming (OneTime, Interval, Random, Seasonal).

Used inside the Event Builder editor. Returns a validated timing instance on every change
so the parent can immediately compose a ComposedEventBuilder.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from financial_simulator.core.event import (
    AnyTiming,
    IntervalTiming,
    OneTimeTiming,
    RandomTiming,
    SeasonalTiming,
)

TIMING_TYPE_LABELS: dict[str, str] = {
    "OneTime": "One-time (fires exactly once)",
    "Interval": "Recurring Interval (e.g. monthly)",
    "Random": "Random N times in window",
    "Seasonal": "Seasonal wrapper (only certain months)",
}

TIMING_TYPE_ORDER = list(TIMING_TYPE_LABELS.keys())


def _default_one_time() -> OneTimeTiming:
    return OneTimeTiming(time=datetime(2026, 6, 15))


def _default_interval() -> IntervalTiming:
    return IntervalTiming(interval=timedelta(days=30))


def _default_random() -> RandomTiming:
    return RandomTiming(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 12, 31),
        n=4,
    )


def _default_seasonal() -> SeasonalTiming:
    inner = IntervalTiming(interval=timedelta(days=30))
    return SeasonalTiming(inner=inner, months=[1, 4, 7, 10])


def get_default_timing(timing_type: str) -> AnyTiming:
    if timing_type == "OneTime":
        return _default_one_time()
    if timing_type == "Interval":
        return _default_interval()
    if timing_type == "Random":
        return _default_random()
    if timing_type == "Seasonal":
        return _default_seasonal()
    return _default_interval()


def render_timing_editor(
    key_prefix: str = "timing",
    initial: AnyTiming | None = None,
    help_text: str = "Controls exactly when the event fires during the simulation.",
) -> AnyTiming:
    """
    Render a complete, live timing configuration form.

    Returns a fully validated AnyTiming that can be dropped straight into a
    ComposedEventBuilder.
    """
    import streamlit as st

    st.markdown("#### ⏱️ Timing")
    st.caption(help_text)

    current_type = getattr(initial, "type", "Interval") if initial else "Interval"

    # Type selector
    type_label_to_key = {v: k for k, v in TIMING_TYPE_LABELS.items()}
    selected_label = st.selectbox(
        "Timing Pattern",
        options=list(TIMING_TYPE_LABELS.values()),
        index=TIMING_TYPE_ORDER.index(current_type) if current_type in TIMING_TYPE_ORDER else 1,
        key=f"{key_prefix}_type",
        help="Choose how often / when this cash flow occurs.",
    )
    timing_type = type_label_to_key[selected_label]

    # Seed params from initial if the type matches
    params: dict[str, Any] = {}
    if initial is not None and getattr(initial, "type", None) == timing_type:
        if timing_type == "OneTime":
            params["time"] = getattr(initial, "time", datetime(2026, 6, 15))
        elif timing_type == "Interval":
            params["interval"] = getattr(initial, "interval", timedelta(days=30))
            params["start_time"] = getattr(initial, "start_time", None)
        elif timing_type == "Random":
            params["start"] = getattr(initial, "start", datetime(2026, 1, 1))
            params["end"] = getattr(initial, "end", datetime(2026, 12, 31))
            params["n"] = getattr(initial, "n", 3)
        elif timing_type == "Seasonal":
            params["months"] = getattr(initial, "months", [1, 4, 7, 10])
            # inner is handled below

    # Dynamic form per type
    if timing_type == "OneTime":
        t = params.get("time", datetime(2026, 6, 15))
        if isinstance(t, datetime):
            t = t.date()
        chosen = st.date_input(
            "Fire Date",
            value=t,
            key=f"{key_prefix}_one_time_date",
            help="The exact calendar day this event occurs (must be inside scenario [start, end]).",
        )
        timing = OneTimeTiming(time=datetime.combine(chosen, datetime.min.time()))

    elif timing_type == "Interval":
        col1, col2 = st.columns(2)
        days = (params.get("interval") or timedelta(days=30)).days
        interval_days = col1.number_input(
            "Every N days (or use 30/90/365 for common periods)",
            min_value=1,
            max_value=3650,
            value=days,
            step=1,
            key=f"{key_prefix}_interval_days",
        )
        start_val = params.get("start_time")
        if start_val is not None and isinstance(start_val, datetime):
            start_val = start_val.date()
        start_date = col2.date_input(
            "First occurrence (optional — defaults to scenario start + interval)",
            value=start_val,
            key=f"{key_prefix}_interval_start",
        )
        timing = IntervalTiming(
            interval=timedelta(days=int(interval_days)),
            start_time=datetime.combine(start_date, datetime.min.time()) if start_date else None,
        )

    elif timing_type == "Random":
        col1, col2, col3 = st.columns(3)
        start_d = (
            params.get("start", datetime(2026, 1, 1)).date()
            if hasattr(params.get("start", datetime(2026, 1, 1)), "date")
            else datetime(2026, 1, 1).date()
        )
        end_d = (
            params.get("end", datetime(2026, 12, 31)).date()
            if hasattr(params.get("end", datetime(2026, 12, 31)), "date")
            else datetime(2026, 12, 31).date()
        )
        n_val = int(params.get("n", 3))

        s = col1.date_input("Window start", value=start_d, key=f"{key_prefix}_rand_start")
        e = col2.date_input("Window end", value=end_d, key=f"{key_prefix}_rand_end")
        n = col3.number_input(
            "How many times?", min_value=1, max_value=500, value=n_val, key=f"{key_prefix}_rand_n"
        )

        timing = RandomTiming(
            start=datetime.combine(s, datetime.min.time()),
            end=datetime.combine(e, datetime.min.time()),
            n=int(n),
        )

    elif timing_type == "Seasonal":
        st.caption("Wraps another timing and suppresses events outside the allowed months.")
        months = st.multiselect(
            "Allowed months (1=Jan … 12=Dec)",
            options=list(range(1, 13)),
            default=params.get("months", [1, 4, 7, 10]),
            key=f"{key_prefix}_seasonal_months",
            help="Common patterns: quarterly (1,4,7,10), summer (5-8), year-end (10-12).",
        )
        if not months:
            months = [1]
        # For simplicity we always wrap a fresh monthly interval. Advanced users can edit the JSON.
        inner = IntervalTiming(interval=timedelta(days=30))
        timing = SeasonalTiming(inner=inner, months=[int(m) for m in months])

    else:
        timing = _default_interval()

    # Live mini-preview (next 5 candidate dates)
    with st.expander("Preview next fire dates (sample)", expanded=False):
        try:
            from datetime import datetime as _dt

            preview_start = _dt(2026, 1, 1)
            preview_end = _dt(2027, 1, 1)
            t_copy = timing.model_copy(deep=True) if hasattr(timing, "model_copy") else timing
            t_copy.reset()
            dates = []
            cur = preview_start
            for _ in range(6):
                nxt = t_copy.next_time(cur, preview_end, {})
                if nxt is None:
                    break
                dates.append(nxt.date().isoformat())
                t_copy.advance()
                cur = nxt + timedelta(days=1)
            if dates:
                st.code(" → ".join(dates[:5]), language=None)
            else:
                st.caption("No firings in the sample window (check scenario dates).")
        except Exception as ex:
            st.caption(f"Preview unavailable: {ex}")

    return timing


__all__ = [
    "render_timing_editor",
    "TIMING_TYPE_LABELS",
    "get_default_timing",
]
