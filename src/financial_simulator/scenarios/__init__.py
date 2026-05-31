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

from .materialization import build_engine, run_monte_carlo, run_single
from .metrics import compute_all_metrics, compute_metric
from .models import (
    AnyExternalDriver,
    CustomMetric,
    DiscreteRateDriver,
    DistributionLibrary,
    SavedDistribution,
    ScenarioConfig,
)
from .persistence import (
    TEMPLATES_DIR,
    list_templates,
    load_distribution_library,
    load_scenario,
    load_template,
    save_distribution_library,
    save_scenario,
    scenario_from_json,
    scenario_to_json,
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
    "list_templates",
    "load_template",
    "TEMPLATES_DIR",
]

__version__ = "1.0.0"  # scenario builder foundation
