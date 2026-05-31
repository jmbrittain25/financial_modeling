"""
scenarios
=========

High-level, user-facing, fully serializable configuration layer for the
Financial Simulator. Designed for the interactive Scenario Builder UI,
template library, saved user artifacts, and future AI-agent consumption.

Key exports:
- ScenarioConfig: the primary serializable "scenario" document
- SavedDistribution + DistributionLibrary: user-defined reusable distributions
- CustomMetric + compute helpers
- External driver types (DiscreteRateDriver, etc.)
- Materialization functions (build_engine, run_single, run_monte_carlo)

All models use the core discriminated unions and round-trip cleanly to JSON.
"""

from __future__ import annotations

from .models import (
    SavedDistribution,
    DistributionLibrary,
    CustomMetric,
    DiscreteRateDriver,
    AnyExternalDriver,
    ScenarioConfig,
)
from .metrics import compute_metric, compute_all_metrics
from .materialization import build_engine, run_single, run_monte_carlo
from .persistence import (
    scenario_to_json,
    scenario_from_json,
    load_scenario,
    save_scenario,
    load_distribution_library,
    save_distribution_library,
)

__all__ = [
    "SavedDistribution",
    "DistributionLibrary",
    "CustomMetric",
    "DiscreteRateDriver",
    "AnyExternalDriver",
    "ScenarioConfig",
    "compute_metric",
    "compute_all_metrics",
    "build_engine",
    "run_single",
    "run_monte_carlo",
    "scenario_to_json",
    "scenario_from_json",
    "load_scenario",
    "save_scenario",
    "load_distribution_library",
    "save_distribution_library",
]

__version__ = "1.0.0"  # scenario builder foundation
