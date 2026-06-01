"""
External Drivers — first-class stochastic processes for macro factors.

This module provides:
- create_external_driver: factory (parallel to create_distribution etc.)
- sample_driver_path: generate time-series realizations for visualization / analysis
- Example driver factories (Interest Rate, Inflation, Stock Market Returns)

Drivers remain declarative Pydantic models (in models.py). All driver types now
include an optional `description: str` field for documentation.

All sampling is reproducible when a seed (or rng) is supplied.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from ..core.distributions import create_distribution
from ..core.event import create_timing
from .models import (
    AnyExternalDriver,
    ConstantDriver,
    ContinuousGBMDriver,
    ContinuousMeanRevertDriver,
    DiscreteRateDriver,
)

# -----------------------------------------------------------------------------
# Factory (the single entry point for constructing drivers from dicts / JSON)
# -----------------------------------------------------------------------------

_external_driver_adapter: TypeAdapter[AnyExternalDriver] = TypeAdapter(AnyExternalDriver)


def create_external_driver(data: dict[str, Any] | AnyExternalDriver) -> AnyExternalDriver:
    """Create any supported ExternalDriver from a config dict or existing object.

    This is the primary entry point when loading drivers from ScenarioConfig JSON,
    templates, or UI output.

    Example:
        driver = create_external_driver({
            "type": "discrete_rate",
            "name": "mortgage_rate",
            "target_state_key": "mortgage_rate",
            "dist": {"type": "normal", "mean": 0.065, "std": 0.012},
            "timing": {"type": "Interval", "interval": "P90D"},
        })
    """
    if isinstance(
        data, (DiscreteRateDriver, ConstantDriver, ContinuousGBMDriver, ContinuousMeanRevertDriver)
    ):
        return data
    if isinstance(data, dict):
        # Normalize nested dist / timing using core factories (defensive, cheap)
        d = dict(data)
        if "dist" in d and isinstance(d["dist"], dict):
            d["dist"] = create_distribution(d["dist"])
        if "timing" in d and isinstance(d["timing"], dict):
            d["timing"] = create_timing(d["timing"])
        return _external_driver_adapter.validate_python(d)
    raise TypeError(f"Cannot create external driver from type {type(data)}")


# -----------------------------------------------------------------------------
# Path Sampling (the "sample over time" capability)
# -----------------------------------------------------------------------------


def sample_driver_path(
    driver: AnyExternalDriver,
    start: dt.datetime,
    end: dt.datetime,
    *,
    freq: str | dt.timedelta = "M",
    n_paths: int = 3,
    seed: int | None = None,
    num_points: int | None = None,
) -> dict[str, Any]:
    """Generate one or more sample paths for a driver over [start, end].

    This is the key API for UI previews, calibration, and driver analysis.
    It does NOT mutate any simulation state — it only produces synthetic histories.

    Supported freq values (passed to pandas date_range when available, otherwise
    approximated):
        "M", "MS", "Q", "QS", "Y", "YS", "W", or a raw timedelta.

    Returns a dict with:
        driver_name, target_state_key, driver_type,
        times: list[datetime],
        paths: list[list[float]] (n_paths x n_times),
        summary: simple terminal stats across paths.
    """
    rng = np.random.default_rng(seed)

    # Resolve time grid
    times = _resolve_time_grid(start, end, freq, num_points)

    if isinstance(driver, DiscreteRateDriver):
        paths = [_sample_discrete_path(driver, times, rng) for _ in range(n_paths)]
    elif isinstance(driver, ConstantDriver):
        val = driver.value
        paths = [[val] * len(times) for _ in range(n_paths)]
    elif isinstance(driver, ContinuousGBMDriver):
        paths = [
            _sample_gbm_path(driver.initial_value, driver.drift, driver.volatility, times, rng)
            for _ in range(n_paths)
        ]
    elif isinstance(driver, ContinuousMeanRevertDriver):
        paths = [
            _sample_mean_revert_path(
                driver.initial_value,
                driver.long_term_mean,
                driver.speed,
                driver.volatility,
                times,
                rng,
            )
            for _ in range(n_paths)
        ]
    else:
        raise TypeError(f"Unsupported driver type for sampling: {type(driver)}")

    # Terminal statistics across paths (last value of each path)
    terminals = [p[-1] for p in paths]
    summary = {
        "mean_terminal": float(np.mean(terminals)),
        "std_terminal": float(np.std(terminals)),
        "min_terminal": float(np.min(terminals)),
        "max_terminal": float(np.max(terminals)),
        "n_paths": n_paths,
    }

    return {
        "driver_name": getattr(driver, "name", "unnamed"),
        "target_state_key": getattr(driver, "target_state_key", None),
        "driver_type": getattr(driver, "type", type(driver).__name__),
        "times": [t.isoformat() if isinstance(t, dt.datetime) else t for t in times],
        "paths": paths,
        "summary": summary,
    }


def _resolve_time_grid(
    start: dt.datetime, end: dt.datetime, freq: str | dt.timedelta, num_points: int | None
) -> list[dt.datetime]:
    """Produce a list of observation times between start and end (inclusive)."""
    if num_points is not None and num_points > 0:
        # Evenly spaced in time (simple linear)
        delta = (end - start) / (num_points - 1) if num_points > 1 else dt.timedelta(0)
        return [start + i * delta for i in range(num_points)]

    # Try pandas for rich freq support (graceful fallback if pandas missing)
    try:
        import pandas as pd  # type: ignore

        idx = pd.date_range(start=start, end=end, freq=freq, inclusive="both")
        # Ensure we have at least start and end
        if len(idx) == 0:
            return [start, end]
        times = list(idx.to_pydatetime())
        # Guarantee exact start/end
        if times[0] != start:
            times = [start] + times
        if times[-1] != end:
            times = times + [end]
        return times
    except Exception:
        pass

    # Fallback: monthly steps
    if isinstance(freq, dt.timedelta):
        step = freq
    else:
        step = dt.timedelta(days=30)  # reasonable default

    times: list[dt.datetime] = []
    t = start
    while t <= end:
        times.append(t)
        t = t + step
    if times[-1] != end:
        times.append(end)
    return times


def _sample_discrete_path(
    driver: DiscreteRateDriver, times: list[dt.datetime], rng: np.random.Generator
) -> list[float]:
    """Simulate the discrete driver by walking its Timing and sampling the dist."""
    # We simulate the timing machinery without a full engine
    timing = driver.timing.model_copy(deep=True)
    timing.reset(rng)

    values: list[float] = []
    current = times[0]
    dist = driver.dist

    for t in times:
        # Advance timing until we would have fired at or before t
        nt = timing.next_time(current, t, {})
        while nt is not None and nt <= t:
            val = dist.sample(rng)
            values.append(val)  # last sample wins for that observation point
            timing.advance()
            current = nt
            nt = timing.next_time(current, t, {})

        if not values:
            # No samples yet — draw one on demand for the first point(s)
            val = dist.sample(rng)
            values.append(val)
        else:
            # Hold last observed value forward (step-function semantics)
            values.append(values[-1])

    # Trim/pad to exact length of times
    if len(values) < len(times):
        values += [values[-1]] * (len(times) - len(values))
    return values[: len(times)]


def _sample_gbm_path(
    initial: float, drift: float, vol: float, times: list[dt.datetime], rng: np.random.Generator
) -> list[float]:
    """Exact GBM sampling at arbitrary observation times (log-space)."""
    if not times:
        return []
    path = [float(initial)]
    prev_t = times[0]
    prev_val = initial
    for t in times[1:]:
        years = (t - prev_t).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            path.append(prev_val)
            continue
        drift_term = (drift - 0.5 * vol**2) * years
        diffusion = vol * np.sqrt(years) * rng.normal()
        new_val = prev_val * np.exp(drift_term + diffusion)
        path.append(float(new_val))
        prev_val = new_val
        prev_t = t
    return path


def _sample_mean_revert_path(
    initial: float,
    long_term: float,
    speed: float,
    vol: float,
    times: list[dt.datetime],
    rng: np.random.Generator,
) -> list[float]:
    """Euler–Maruyama discretization of the mean-reverting OU process."""
    if not times:
        return []
    path = [float(initial)]
    prev_t = times[0]
    prev_val = initial
    for t in times[1:]:
        years = (t - prev_t).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            path.append(prev_val)
            continue
        mean_rev = speed * (long_term - prev_val) * years
        noise = vol * np.sqrt(years) * rng.normal()
        new_val = prev_val + mean_rev + noise
        path.append(float(new_val))
        prev_val = new_val
        prev_t = t
    return path


# -----------------------------------------------------------------------------
# Example Driver Factories (the "Add example drivers" requirement)
# -----------------------------------------------------------------------------


def make_interest_rate_driver(
    name: str = "interest_rate_path",
    target_state_key: str = "market_rate",
    long_term_mean: float = 0.045,
    speed: float = 1.2,
    volatility: float = 0.008,
    initial_value: float = 0.052,
) -> ContinuousMeanRevertDriver:
    """Realistic mean-reverting short-rate style driver (good for mortgages, loans)."""
    return ContinuousMeanRevertDriver(
        name=name,
        target_state_key=target_state_key,
        long_term_mean=long_term_mean,
        speed=speed,
        volatility=volatility,
        initial_value=initial_value,
        metadata={"example": "interest_rate", "notes": "Mean-reverting rates"},
    )


def make_inflation_driver(
    name: str = "inflation_path",
    target_state_key: str = "inflation_index",
    long_term_mean: float = 0.025,
    speed: float = 0.6,
    volatility: float = 0.006,
    initial_value: float = 0.028,
) -> ContinuousMeanRevertDriver:
    """Mean-reverting inflation driver. Use to grow expenses or index-linked cash flows."""
    return ContinuousMeanRevertDriver(
        name=name,
        target_state_key=target_state_key,
        long_term_mean=long_term_mean,
        speed=speed,
        volatility=volatility,
        initial_value=initial_value,
        metadata={"example": "inflation"},
    )


def make_stock_market_driver(
    name: str = "equity_market_returns",
    target_state_key: str = "portfolio_value",
    drift: float = 0.08,
    volatility: float = 0.18,
    initial_value: float = 1_000_000.0,
) -> ContinuousGBMDriver:
    """Classic equity-market GBM driver. Pair with a continuous process consumer or
    custom logic that reads the state key for portfolio growth.
    """
    return ContinuousGBMDriver(
        name=name,
        target_state_key=target_state_key,
        drift=drift,
        volatility=volatility,
        initial_value=initial_value,
        metadata={"example": "stock_market"},
    )


# Convenience re-export of the public surface
__all__ = [
    "create_external_driver",
    "sample_driver_path",
    "make_interest_rate_driver",
    "make_inflation_driver",
    "make_stock_market_driver",
    # Re-export the concrete classes for ergonomic imports
    "DiscreteRateDriver",
    "ConstantDriver",
    "ContinuousGBMDriver",
    "ContinuousMeanRevertDriver",
    "AnyExternalDriver",
]
