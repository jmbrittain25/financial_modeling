"""
Required macro environment — interest rates, housing, and stock market.

Each variable supports constant, deterministic growth/decline, or stochastic
evolution. Materialized into simulation initial state and continuous processes.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from ..core.simulation import (
    AppreciationProcess,
    GBMContinuousProcess,
    GeometricBrownianMotion,
    MeanRevertingContinuousProcess,
    MeanRevertingProcess,
    SimulationEngine,
)
from .models import (
    ConstantDriver,
    ContinuousGBMDriver,
    ContinuousMeanRevertDriver,
    MacroEnvironment,
    MacroSlot,
    MacroVariableConfig,
    ScenarioConfig,
)

MACRO_STATE_KEYS = frozenset({"market_rate", "home_value", "portfolio_value"})

SLOT_DEFS: dict[MacroSlot, dict[str, Any]] = {
    "interest_rates": {
        "title": "Interest rates",
        "state_key": "market_rate",
        "value_kind": "rate",
        "default_stochastic": "mean_reverting",
    },
    "housing": {
        "title": "Housing market",
        "state_key": "home_value",
        "value_kind": "currency",
        "default_stochastic": "gbm",
    },
    "stock_market": {
        "title": "Stock market",
        "state_key": "portfolio_value",
        "value_kind": "currency",
        "default_stochastic": "gbm",
    },
}


def default_macro_variable(slot: MacroSlot) -> MacroVariableConfig:
    meta = SLOT_DEFS[slot]
    key = meta["state_key"]
    if slot == "interest_rates":
        return MacroVariableConfig(
            slot=slot,
            state_key=key,
            mode="constant",
            value=0.05,
            annual_rate=0.0,
            stochastic_type="mean_reverting",
            drift=0.0,
            volatility=0.008,
            long_term_mean=0.045,
            reversion_speed=1.2,
        )
    if slot == "housing":
        return MacroVariableConfig(
            slot=slot,
            state_key=key,
            mode="constant",
            value=500_000.0,
            annual_rate=0.04,
            stochastic_type="gbm",
            drift=0.04,
            volatility=0.08,
            long_term_mean=500_000.0,
            reversion_speed=0.5,
        )
    return MacroVariableConfig(
        slot=slot,
        state_key=key,
        mode="constant",
        value=100_000.0,
        annual_rate=0.07,
        stochastic_type="gbm",
        drift=0.08,
        volatility=0.18,
        long_term_mean=100_000.0,
        reversion_speed=0.8,
    )


def default_macro_environment() -> MacroEnvironment:
    return MacroEnvironment(
        interest_rates=default_macro_variable("interest_rates"),
        housing=default_macro_variable("housing"),
        stock_market=default_macro_variable("stock_market"),
    )


def _infer_mode_from_driver(drv: Any) -> str | None:
    dtype = getattr(drv, "type", None)
    if dtype == "constant":
        return "constant"
    if dtype in ("gbm_continuous", "mean_revert_continuous"):
        return "stochastic"
    return None


def _macro_from_driver(slot: MacroSlot, drv: Any) -> MacroVariableConfig:
    base = default_macro_variable(slot)
    base.mode = _infer_mode_from_driver(drv) or "constant"
    target = getattr(drv, "target_state_key", base.state_key)
    base.state_key = target

    if base.mode == "constant":
        base.value = float(getattr(drv, "value", base.value))
        return base

    base.value = float(getattr(drv, "initial_value", base.value))
    if isinstance(drv, ContinuousGBMDriver) or getattr(drv, "type", None) == "gbm_continuous":
        base.stochastic_type = "gbm"
        base.drift = float(getattr(drv, "drift", base.drift))
        base.volatility = float(getattr(drv, "volatility", base.volatility))
    elif (
        isinstance(drv, ContinuousMeanRevertDriver)
        or getattr(drv, "type", None) == "mean_revert_continuous"
    ):
        base.stochastic_type = "mean_reverting"
        base.long_term_mean = float(getattr(drv, "long_term_mean", base.long_term_mean))
        base.reversion_speed = float(getattr(drv, "speed", base.reversion_speed))
        base.volatility = float(getattr(drv, "volatility", base.volatility))
    return base


def migrate_macro_from_external_drivers(drivers: list[Any]) -> MacroEnvironment | None:
    """Best-effort migration from legacy external_drivers list."""
    found: dict[MacroSlot, Any] = {}
    for slot, meta in SLOT_DEFS.items():
        key = meta["state_key"]
        for drv in drivers:
            if getattr(drv, "target_state_key", None) == key:
                found[slot] = drv
                break
    if not found:
        return None
    env = default_macro_environment()
    updates = {
        "interest_rates": env.interest_rates,
        "housing": env.housing,
        "stock_market": env.stock_market,
    }
    for slot, drv in found.items():
        updates[slot] = _macro_from_driver(slot, drv)
    return MacroEnvironment(**updates)


def ensure_macro_environment(cfg: ScenarioConfig) -> MacroEnvironment:
    """Return macro environment, migrating legacy drivers when needed."""
    macro = getattr(cfg, "macro_environment", None)
    if macro is None:
        migrated = migrate_macro_from_external_drivers(cfg.external_drivers)
        return migrated or default_macro_environment()
    return macro


def macro_variable_to_preview_driver(var: MacroVariableConfig) -> Any:
    """Convert a macro variable into a driver object for path sampling."""
    name = f"macro_{var.slot}"
    if var.mode == "constant":
        return ConstantDriver(name=name, target_state_key=var.state_key, value=var.value)
    if var.mode == "growth":
        return ConstantDriver(name=name, target_state_key=var.state_key, value=var.value)
    if var.stochastic_type == "mean_reverting":
        return ContinuousMeanRevertDriver(
            name=name,
            target_state_key=var.state_key,
            long_term_mean=var.long_term_mean,
            speed=var.reversion_speed,
            volatility=var.volatility,
            initial_value=var.value,
        )
    return ContinuousGBMDriver(
        name=name,
        target_state_key=var.state_key,
        drift=var.drift,
        volatility=var.volatility,
        initial_value=var.value,
    )


def sample_growth_path(
    initial: float,
    annual_rate: float,
    times: list[dt.datetime],
) -> list[float]:
    if not times:
        return []
    path = [float(initial)]
    year_seconds = 365.25 * 24 * 3600
    prev_t = times[0]
    for t in times[1:]:
        years = (t - prev_t).total_seconds() / year_seconds
        if years <= 0:
            path.append(path[-1])
        else:
            path.append(path[-1] * (1.0 + annual_rate) ** years)
        prev_t = t
    return path


def sample_macro_paths(
    var: MacroVariableConfig,
    start: dt.datetime,
    end: dt.datetime,
    *,
    n_paths: int = 5,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Sample paths for UI preview (constant, growth, or stochastic)."""
    from .drivers import _resolve_time_grid, sample_driver_path

    times = _resolve_time_grid(start, end, "MS", None, max_points=120)
    iso_times = [t.isoformat() for t in times]

    if var.mode == "constant":
        paths = [[var.value] * len(times) for _ in range(n_paths)]
    elif var.mode == "growth":
        single = sample_growth_path(var.value, var.annual_rate, times)
        paths = [single for _ in range(n_paths)]
    else:
        driver = macro_variable_to_preview_driver(var)
        data = sample_driver_path(driver, start, end, freq="MS", n_paths=n_paths, seed=seed)
        return data

    terminals = [p[-1] for p in paths]
    return {
        "driver_name": f"macro_{var.slot}",
        "target_state_key": var.state_key,
        "driver_type": var.mode,
        "times": iso_times,
        "paths": paths,
        "summary": {
            "mean_terminal": float(np.mean(terminals)),
            "std_terminal": float(np.std(terminals)),
            "min_terminal": float(np.min(terminals)),
            "max_terminal": float(np.max(terminals)),
            "n_paths": n_paths,
        },
    }


def apply_macro_environment(eng: SimulationEngine, macro: MacroEnvironment) -> None:
    """Wire the three macro variables into engine state and continuous processes."""
    for var in macro.slots():
        if var.mode == "constant":
            eng.initial_state[var.state_key] = var.value
            continue

        eng.initial_state.setdefault(var.state_key, var.value)

        if var.mode == "growth":
            eng.add_continuous_process(
                AppreciationProcess(
                    rate=var.annual_rate,
                    var=var.state_key,
                    name=f"macro:{var.slot}",
                )
            )
            continue

        if var.stochastic_type == "mean_reverting":
            eng.add_continuous_process(
                MeanRevertingContinuousProcess(
                    var=var.state_key,
                    process=MeanRevertingProcess(
                        long_term_mean=var.long_term_mean,
                        speed=var.reversion_speed,
                        volatility=var.volatility,
                    ),
                    name=f"macro:{var.slot}",
                )
            )
        else:
            eng.add_continuous_process(
                GBMContinuousProcess(
                    var=var.state_key,
                    process=GeometricBrownianMotion(
                        drift=var.drift,
                        volatility=var.volatility,
                    ),
                    name=f"macro:{var.slot}",
                )
            )


def macro_summary_label(var: MacroVariableConfig) -> str:
    """One-line summary for collapsed expander titles."""
    meta = SLOT_DEFS[var.slot]
    title = meta["title"]
    if var.mode == "constant":
        if meta["value_kind"] == "rate":
            return f"{title} — Constant ({var.value:.2%})"
        return f"{title} — Constant (${var.value:,.0f})"
    if var.mode == "growth":
        direction = "growth" if var.annual_rate >= 0 else "decline"
        if meta["value_kind"] == "rate":
            return f"{title} — {direction.title()} ({var.annual_rate:+.2%}/yr from {var.value:.2%})"
        return f"{title} — {direction.title()} ({var.annual_rate:+.1%}/yr from ${var.value:,.0f})"
    stype = "GBM" if var.stochastic_type == "gbm" else "Mean-reverting"
    if meta["value_kind"] == "rate":
        return f"{title} — Stochastic {stype} (start {var.value:.2%})"
    return f"{title} — Stochastic {stype} (start ${var.value:,.0f})"


__all__ = [
    "MACRO_STATE_KEYS",
    "SLOT_DEFS",
    "MacroEnvironment",
    "MacroSlot",
    "MacroVariableConfig",
    "apply_macro_environment",
    "default_macro_environment",
    "default_macro_variable",
    "ensure_macro_environment",
    "macro_summary_label",
    "macro_variable_to_preview_driver",
    "migrate_macro_from_external_drivers",
    "sample_growth_path",
    "sample_macro_paths",
]