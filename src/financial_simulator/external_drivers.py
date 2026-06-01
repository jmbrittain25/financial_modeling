"""Public API for External Drivers.

This module provides convenient top-level access to external driver types
and helpers without needing to go through `financial_simulator.scenarios`.

Example:
    from financial_simulator.external_drivers import (
        DiscreteRateDriver,
        ContinuousGBMDriver,
        make_interest_rate_driver,
        sample_driver_path,
    )

    driver = make_interest_rate_driver()
    paths = sample_driver_path(driver, start=..., end=...)
"""

from .scenarios import (
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

__all__ = [
    "AnyExternalDriver",
    "DiscreteRateDriver",
    "ConstantDriver",
    "ContinuousGBMDriver",
    "ContinuousMeanRevertDriver",
    "create_external_driver",
    "sample_driver_path",
    "make_interest_rate_driver",
    "make_inflation_driver",
    "make_stock_market_driver",
]
