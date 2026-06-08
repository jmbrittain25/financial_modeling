"""
Materialization: turn a ScenarioConfig into a runnable SimulationEngine
(and run it, with driver expansion and custom metric attachment).

This is the canonical "builder" for the UI and for saved scenarios.
It reuses the existing core factories wherever possible.
"""

from __future__ import annotations

from ..core import (
    ComposedEventBuilder,
    GBMContinuousProcess,
    GeometricBrownianMotion,
    MeanRevertingContinuousProcess,
    MeanRevertingProcess,
    RateChangeValue,
    SimulationEngine,
    SimulationResult,
)
from ..monte_carlo.runner import MonteCarloRunner
from .macro_environment import (
    MACRO_STATE_KEYS,
    apply_macro_environment,
    ensure_macro_environment,
)
from .metrics import compute_all_metrics
from .models import (
    ConstantDriver,
    ContinuousGBMDriver,
    ContinuousMeanRevertDriver,
    DiscreteRateDriver,
    ScenarioConfig,
)


def build_engine(
    cfg: ScenarioConfig,
    seed: int | None = None,
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

    macro = ensure_macro_environment(cfg)
    apply_macro_environment(eng, macro)

    # Expand legacy / extra external drivers (skip macro keys — handled above)
    for driver in cfg.external_drivers:
        if getattr(driver, "target_state_key", None) in MACRO_STATE_KEYS:
            continue
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

        elif isinstance(driver, ContinuousGBMDriver):
            # Wire as a true continuous process (smooth stochastic evolution)
            proc = GBMContinuousProcess(
                process=GeometricBrownianMotion(drift=driver.drift, volatility=driver.volatility),
                var=driver.target_state_key,
                name=driver.name or f"external:{driver.target_state_key}",
            )
            # Seed the initial value if the key is absent (non-destructive)
            eng.initial_state.setdefault(driver.target_state_key, driver.initial_value)
            # Mark the driver object itself (harmless and useful for introspection)
            driver.metadata.setdefault("is_external_driver", True)
            eng.add_continuous_process(proc)

        elif isinstance(driver, ContinuousMeanRevertDriver):
            proc = MeanRevertingContinuousProcess(
                process=MeanRevertingProcess(
                    long_term_mean=driver.long_term_mean,
                    speed=driver.speed,
                    volatility=driver.volatility,
                ),
                var=driver.target_state_key,
                name=driver.name or f"external:{driver.target_state_key}",
            )
            eng.initial_state.setdefault(driver.target_state_key, driver.initial_value)
            driver.metadata.setdefault("is_external_driver", True)
            eng.add_continuous_process(proc)

    return eng


def run_single(
    cfg: ScenarioConfig,
    seed: int | None = None,
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
    base_seed: int | None = None,
    n_jobs: int = 4,
) -> list[SimulationResult]:
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
