"""Stochastic processes for financial modeling (Geometric Brownian Motion, mean reversion, etc.).

These can be used either as ContinuousProcess implementations or inside
ValueGenerators for more sophisticated cash flow modeling.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class StochasticProcess(BaseModel, ABC):
    """Abstract base for stochastic processes used in simulations."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    name: Optional[str] = None

    @abstractmethod
    def step(
        self,
        current_value: float,
        delta: dt.timedelta,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Advance the process by one time step and return the new value."""
        ...


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion (GBM).

    Commonly used for modeling stock prices, revenue, or other growing
    quantities with volatility.

    dS = mu * S * dt + sigma * S * dW
    """

    drift: float = Field(description="Expected annual drift (mu)")
    volatility: float = Field(gt=0, description="Annual volatility (sigma)")

    def step(
        self,
        current_value: float,
        delta: dt.timedelta,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        rng = rng or np.random.default_rng()
        years = delta.total_seconds() / (365.25 * 24 * 3600)

        if years <= 0:
            return current_value

        drift_term = (self.drift - 0.5 * self.volatility**2) * years
        diffusion_term = self.volatility * np.sqrt(years) * rng.normal()

        return current_value * np.exp(drift_term + diffusion_term)


class MeanRevertingProcess(StochasticProcess):
    """
    Simple mean-reverting (Ornstein-Uhlenbeck style) process.

    Useful for modeling interest rates, inflation, or commodity prices
    that tend to revert to a long-term mean.
    """

    long_term_mean: float
    speed: float = Field(gt=0, description="Speed of reversion (theta)")
    volatility: float = Field(gt=0)

    def step(
        self,
        current_value: float,
        delta: dt.timedelta,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        rng = rng or np.random.default_rng()
        years = delta.total_seconds() / (365.25 * 24 * 3600)

        if years <= 0:
            return current_value

        mean_reversion = self.speed * (self.long_term_mean - current_value) * years
        noise = self.volatility * np.sqrt(years) * rng.normal()

        return current_value + mean_reversion + noise
