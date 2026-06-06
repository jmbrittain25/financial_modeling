"""
scenarios
=========

High-level, user-facing, fully serializable configuration layer for the
Financial Simulator. Designed for the interactive Scenario Builder UI,
template library, saved user artifacts, and future AI-agent consumption.

Key exports:
- ScenarioConfig: the primary serializable "scenario" document
- SavedDistribution + DistributionLibrary + ScenarioLibrary: user-defined reusable distributions and scenarios
- CustomMetric + compute helpers
- External driver types (DiscreteRateDriver, ConstantDriver, Continuous*Driver, AnyExternalDriver)
- Materialization functions (build_engine, run_single, run_monte_carlo)

All models use the core discriminated unions and round-trip cleanly to JSON.
"""

from __future__ import annotations

from .drivers import (
    AnyExternalDriver,
    ConstantDriver,
    ContinuousGBMDriver,
    ContinuousMeanRevertDriver,
    DiscreteRateDriver,
    create_external_driver,
    make_inflation_driver,
    make_interest_rate_driver,
    make_stock_market_driver,
    sample_driver_path,
)
from .materialization import build_engine, run_monte_carlo, run_single
from .metrics import compute_all_metrics, compute_metric
from .models import (
    CustomMetric,
    DistributionLibrary,
    SavedDistribution,
    ScenarioConfig,
    ScenarioLibrary,
)
from .persistence import (
    PROJECT_ROOT,
    TEMPLATES_DIR,
    USER_SCENARIOS_DIR,
    delete_user_scenario,
    ensure_user_dirs,
    get_user_data_dir,
    list_templates,
    list_user_scenarios,
    load_distribution_library,
    load_scenario,
    load_template,
    load_user_distribution_library,
    load_user_scenario,
    load_user_scenario_library,
    save_distribution_library,
    save_scenario,
    save_user_distribution_library,
    save_user_scenario,
    scenario_from_json,
    scenario_to_json,
)

__all__ = [
    "SavedDistribution",
    "DistributionLibrary",
    "ScenarioLibrary",
    "CustomMetric",
    "DiscreteRateDriver",
    "ConstantDriver",
    "ContinuousGBMDriver",
    "ContinuousMeanRevertDriver",
    "AnyExternalDriver",
    "ScenarioConfig",
    "create_external_driver",
    "sample_driver_path",
    "make_interest_rate_driver",
    "make_inflation_driver",
    "make_stock_market_driver",
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
    "PROJECT_ROOT",
    "TEMPLATES_DIR",
    "USER_SCENARIOS_DIR",
    # User persistence
    "get_user_data_dir",
    "ensure_user_dirs",
    "load_user_distribution_library",
    "save_user_distribution_library",
    "load_user_scenario_library",
    "save_user_scenario",
    "list_user_scenarios",
    "load_user_scenario",
    "delete_user_scenario",
]

__version__ = "1.0.0"  # scenario builder foundation
