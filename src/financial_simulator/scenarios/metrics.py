"""
Computation of CustomMetric instances against SimulationResult objects.

These are pure post-processing functions. They walk events or state_history
and return a single float per metric per simulation. Used by the materialization
layer and the UI results display.
"""

from __future__ import annotations

import numpy as np

from ..core.simulation import SimulationResult
from .models import CustomMetric


def compute_metric(metric: CustomMetric, result: SimulationResult) -> float:
    """Compute a single custom metric from a completed simulation result."""
    mt = metric.metric_type
    p = metric.params or {}

    if mt == "final_state_value":
        key = p.get("key", "cumulative_cash")
        return float(result.final_state.get(key, 0.0))

    if mt == "sum_positive_events":
        mtype = p.get("metadata_type")
        total = 0.0
        for ev in result.events:
            if ev.value > 0:
                if mtype is None or ev.metadata.get("type") == mtype:
                    total += ev.value
        return total

    if mt == "max_drawdown_on_path":
        key = p.get("state_key", "cumulative_cash")
        path = _extract_state_path(result, key)
        if len(path) < 2:
            return 0.0
        peak = np.maximum.accumulate(path)
        drawdown = (peak - path) / np.maximum(peak, 1e-12)
        return float(np.max(drawdown))

    if mt == "event_count_by_type":
        mtype = p.get("metadata_type")
        if mtype is None:
            return float(len(result.events))
        return float(sum(1 for ev in result.events if ev.metadata.get("type") == mtype))

    if mt == "time_to_threshold":
        key = p.get("state_key")
        threshold = float(p.get("threshold", 0.0))
        direction = p.get("direction", "above")  # "above" or "below"
        if not key or key not in (result.state_history or {}):
            # fall back to final state
            val = result.final_state.get(key, 0.0)
            crossed = (val >= threshold) if direction == "above" else (val <= threshold)
            return 0.0 if crossed else 999.0  # sentinel for "never"
        times = sorted(result.state_history.keys())
        for t in times:
            val = result.state_history[t].get(key, 0.0)
            crossed = (val >= threshold) if direction == "above" else (val <= threshold)
            if crossed:
                # return years since start as a convenient scalar
                years = (t - result.start).total_seconds() / (365.25 * 24 * 3600)
                return max(0.0, years)
        return 999.0

    # Unknown metric type — return NaN so UI can surface it
    return float("nan")


def compute_all_metrics(metrics: list[CustomMetric], result: SimulationResult) -> dict[str, float]:
    """Compute all custom metrics for one simulation result."""
    out: dict[str, float] = {}
    for m in metrics:
        try:
            out[m.name] = compute_metric(m, result)
        except Exception:
            out[m.name] = float("nan")
    return out


def _extract_state_path(result: SimulationResult, key: str) -> np.ndarray:
    """Helper: pull a numeric series from state_history for a given key."""
    if not result.state_history:
        val = result.final_state.get(key, 0.0)
        return np.array([val], dtype=float)
    times = sorted(result.state_history.keys())
    vals = [result.state_history[t].get(key, 0.0) for t in times]
    return np.asarray(vals, dtype=float)


__all__ = ["compute_metric", "compute_all_metrics"]
