"""
Materialization: turn a ScenarioConfig into a runnable SimulationEngine
(and run it, with driver expansion and custom metric attachment).

This is the canonical "builder" for the UI and for saved scenarios.
It reuses the existing core factories wherever possible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core import (
    SimulationEngine,
    ComposedEventBuilder,
    RateChangeValue,
    FixedValue,  # for constant driver
    IntervalTiming,
)
from ..core.distributions import create_distribution
from ..core.event import create_timing, create_value_generator
from ..monte_carlo.runner import MonteCarloRunner
from .models import ScenarioConfig, CustomMetric, DiscreteRateDriver, ConstantDriver
from .metrics import compute_all_metrics


def build_engine(
    cfg: ScenarioConfig,
    seed: Optional[int] = None,
) -> SimulationEngine:
    """Create a fully configured SimulationEngine from a ScenarioConfig.

    - Starts from the declarative event_builders + continuous_processes
    - Expands external_drivers into the appropriate builders / processes
    - Uses the provided seed (or cfg.seed)
    """
    effective_seed = seed if seed is not None else cfg.seed

    eng = SimulationEngine(
        name=cfg.name,
        start=cfg.start,
        end=cfg.end,
        initial_state=dict(cfg.initial_state),  # shallow copy
        seed=effective_seed,
        event_builders=[b.model_copy(deep=True) for b in cfg.event_builders],
        continuous_processes=[p.model_copy(deep=True) for p in cfg.continuous_processes],
    )

    # Expand external drivers
    for driver in cfg.external_drivers:
        if isinstance(driver, DiscreteRateDriver):
            # Turn into a RateChangeValue event builder (the classic variable-rate pattern)
            vg = RateChangeValue(dist=driver.dist, update_key=driver.target_state_key)
            builder = ComposedEventBuilder(
                timing=driver.timing.model_copy(deep=True),
                value_gen=vg,
                metadata={**driver.metadata, "driver": driver.name, "is_external_driver": True},
                name=driver.name,
            )
            eng.add_event_builder(builder)

        elif isinstance(driver, ConstantDriver):
            # Inject once at t=start via a one-time zero-value event that updates state
            # Simpler: just put it in initial_state (most predictable)
            if driver.target_state_key not in eng.initial_state:
                eng.initial_state[driver.target_state_key] = driver.value
            # Also ensure it is present at t=start even if user overrode
            eng.initial_state.setdefault(driver.target_state_key, driver.value)

        elif hasattr(driver, "type") and driver.type in ("gbm_continuous", "mean_revert_continuous"):
            # Stubs for Phase 5+ — create the appropriate ContinuousProcess
            # For now we log a warning by not crashing and do nothing (UI will gate them)
            pass

    return eng


def run_single(
    cfg: ScenarioConfig,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Materialize and run one simulation. Returns the result with custom metrics attached."""
    eng = build_engine(cfg, seed=seed)
    eng.run()
    result = eng.get_result()

    # Attach custom metrics (non-destructive)
    if cfg.custom_metrics:
        metrics = compute_all_metrics(cfg.custom_metrics, result)
        # Store under a conventional key so UI and analyzers can find them
        result.final_state.setdefault("__custom_metrics__", {}).update(metrics)

    return result


def run_monte_carlo(
    cfg: ScenarioConfig,
    n_sims: int,
    base_seed: Optional[int] = None,
    n_jobs: int = 4,
) -> List[SimulationResult]:
    """Run many simulations and attach custom metrics to every result."""
    def factory(i: int) -> SimulationEngine:
        seed = (base_seed + i) if base_seed is not None else None
        return build_engine(cfg, seed=seed)

    runner = MonteCarloRunner(n_jobs=n_jobs)
    results = runner.run(n_sims, factory, base_seed=base_seed)

    # Post-attach custom metrics (MonteCarloRunner doesn't know about ScenarioConfig)
    if cfg.custom_metrics:
        for r in results:
            metrics = compute_all_metrics(cfg.custom_metrics, r)
            r.final_state.setdefault("__custom_metrics__", {}).update(metrics)

    return results


__all__ = ["build_engine", "run_single", "run_monte_carlo"]
