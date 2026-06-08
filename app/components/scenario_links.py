"""
Helpers linking generators, continuous processes, and environmental drivers.

Generators connect to the macro environment through shared state keys (e.g. a loan
reads ``market_rate`` that an interest-rate driver writes). Continuous background
evolution is stored on generator metadata and synced into ``continuous_processes``.
"""

from __future__ import annotations

from typing import Any

from financial_simulator.core.event import ComposedEventBuilder
from financial_simulator.core.simulation import (
    AppreciationProcess,
    GBMContinuousProcess,
    GeometricBrownianMotion,
    MeanRevertingContinuousProcess,
    MeanRevertingProcess,
)

CONTINUOUS_PROCESS_META_KEY = "_continuous_process"
LINKED_PROCESS_NAME_PREFIX = "@gen:"


def continuous_process_name(gen_id: str) -> str:
    return f"{LINKED_PROCESS_NAME_PREFIX}{gen_id}"


def is_generator_linked_process(proc: Any) -> bool:
    name = getattr(proc, "name", None) or ""
    return str(name).startswith(LINKED_PROCESS_NAME_PREFIX)


def get_continuous_process_config(metadata: dict) -> dict[str, Any] | None:
    raw = metadata.get(CONTINUOUS_PROCESS_META_KEY)
    return raw if isinstance(raw, dict) else None


def get_generator_state_keys(builder: ComposedEventBuilder) -> set[str]:
    """State keys this generator reads when producing cash flows."""
    vg = builder.value_gen
    vtype = getattr(vg, "type", None)
    keys: set[str] = set()
    if vtype == "VariableRateLoan":
        keys.add(vg.rate_key)
    elif vtype == "Dividend":
        keys.add(vg.investment_value_key)
    elif vtype == "InvestmentContribution" and getattr(vg, "growth_key", None):
        keys.add(vg.growth_key)
    elif vtype == "TaxEvent":
        keys.add(vg.base_key)
        keys.add(vg.tax_key)
    elif vtype == "RateChange":
        keys.add(vg.update_key)
    return keys


def get_driver_target_keys(drivers: list[Any]) -> list[str]:
    keys: list[str] = []
    for drv in drivers:
        key = getattr(drv, "target_state_key", None)
        if key and key not in keys:
            keys.append(key)
    return keys


def format_driver_links(builder: ComposedEventBuilder, drivers: list[Any]) -> str:
    """Comma-separated names of environmental drivers this generator depends on."""
    gen_keys = get_generator_state_keys(builder)
    if not gen_keys or not drivers:
        return ""
    names: list[str] = []
    for drv in drivers:
        target = getattr(drv, "target_state_key", None)
        if target in gen_keys:
            names.append(getattr(drv, "name", None) or target)
    return ", ".join(names)


def format_macro_links(builder: ComposedEventBuilder, macro: Any) -> str:
    """Comma-separated macro slot titles this generator reads from."""
    from financial_simulator.scenarios.macro_environment import SLOT_DEFS

    gen_keys = get_generator_state_keys(builder)
    if not gen_keys or macro is None:
        return ""
    names: list[str] = []
    for var in macro.slots():
        if var.state_key in gen_keys:
            names.append(SLOT_DEFS[var.slot]["title"])
    return ", ".join(names)


def build_process_from_config(
    config: dict[str, Any],
    gen_id: str,
    gen_name: str | None,
) -> Any:
    ptype = config.get("type", "appreciation")
    var = config.get("var") or "cash"
    proc_name = continuous_process_name(gen_id)
    display = gen_name or f"Generator {gen_id[:8]}"

    if ptype == "appreciation":
        return AppreciationProcess(
            rate=float(config.get("rate", 0.04)),
            var=var,
            name=proc_name,
        )
    if ptype == "gbm":
        return GBMContinuousProcess(
            var=var,
            process=GeometricBrownianMotion(
                drift=float(config.get("drift", 0.08)),
                volatility=float(config.get("volatility", 0.16)),
            ),
            name=proc_name,
        )
    if ptype == "mean_reverting":
        return MeanRevertingContinuousProcess(
            var=var,
            process=MeanRevertingProcess(
                long_term_mean=float(config.get("long_term_mean", 0.045)),
                speed=float(config.get("speed", 1.2)),
                volatility=float(config.get("volatility", 0.008)),
            ),
            name=proc_name,
        )
    raise ValueError(f"Unsupported continuous process type: {ptype} ({display})")


def sync_continuous_processes(
    builders: list[ComposedEventBuilder],
    processes: list[Any],
) -> list[Any]:
    """Rebuild generator-linked processes; preserve standalone processes."""
    standalone = [p for p in processes if not is_generator_linked_process(p)]
    linked: list[Any] = []
    for eb in builders:
        cfg = get_continuous_process_config(eb.metadata)
        if not cfg or not cfg.get("enabled"):
            continue
        gen_id = eb.metadata.get("_generator_id")
        if not gen_id:
            continue
        linked.append(build_process_from_config(cfg, gen_id, eb.name))
    return standalone + linked


def config_from_process(proc: Any) -> dict[str, Any] | None:
    """Reverse a linked continuous process into editor metadata."""
    if isinstance(proc, AppreciationProcess):
        return {"enabled": True, "type": "appreciation", "var": proc.var, "rate": proc.rate}
    if isinstance(proc, GBMContinuousProcess):
        return {
            "enabled": True,
            "type": "gbm",
            "var": proc.var,
            "drift": proc.process.drift,
            "volatility": proc.process.volatility,
        }
    if isinstance(proc, MeanRevertingContinuousProcess):
        return {
            "enabled": True,
            "type": "mean_reverting",
            "var": proc.var,
            "long_term_mean": proc.process.long_term_mean,
            "speed": proc.process.speed,
            "volatility": proc.process.volatility,
        }
    return None


def find_linked_process(gen_id: str, processes: list[Any]) -> Any | None:
    target = continuous_process_name(gen_id)
    for proc in processes:
        if getattr(proc, "name", None) == target:
            return proc
    return None


__all__ = [
    "CONTINUOUS_PROCESS_META_KEY",
    "LINKED_PROCESS_NAME_PREFIX",
    "build_process_from_config",
    "config_from_process",
    "continuous_process_name",
    "find_linked_process",
    "format_driver_links",
    "format_macro_links",
    "get_continuous_process_config",
    "get_driver_target_keys",
    "get_generator_state_keys",
    "is_generator_linked_process",
    "sync_continuous_processes",
]
