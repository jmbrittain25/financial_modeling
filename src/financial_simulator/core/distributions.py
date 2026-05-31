"""Probability distributions for Monte Carlo parameters and stochastic events.

All distributions are Pydantic v2 models supporting validation, JSON/YAML
serialization, and easy extension. Use the discriminated union for config-driven
instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

try:
    from typing import Annotated  # py >= 3.9
except ImportError:
    from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class Distribution(BaseModel, ABC):
    """Abstract base for all parameter distributions.

    Subclasses implement sample() and declare a literal 'type' discriminator.
    All instances are (de)serializable via model_dump / model_validate.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=False,
    )

    @abstractmethod
    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Return a single sample.

        If an rng (numpy Generator) is supplied, sampling is deterministic
        relative to that generator's state. Otherwise a fresh default_rng()
        is used (non-reproducible across calls).
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # type: ignore[override]
        fields = self.model_dump(exclude_unset=True)
        fields.pop("type", None)
        return f"{self.__class__.__name__}({fields})"


class NormalDistribution(Distribution):
    """Normal / Gaussian distribution.

    Typical uses: interest rates, returns, asset values subject to symmetric uncertainty.
    """

    type: Literal["normal"] = "normal"
    mean: float
    std: float = Field(gt=0, description="Standard deviation (> 0)")

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.normal(self.mean, self.std))


class UniformDistribution(Distribution):
    """Continuous uniform distribution on [low, high]."""

    type: Literal["uniform"] = "uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def _validate_bounds(self) -> UniformDistribution:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")
        return self

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.uniform(self.low, self.high))


class TriangularDistribution(Distribution):
    """Triangular distribution (low, mode, high).

    Ideal for three-point estimates commonly elicited from domain experts
    (pessimistic, most-likely, optimistic).
    """

    type: Literal["triangular"] = "triangular"
    low: float
    mode: float
    high: float

    @model_validator(mode="after")
    def _validate_triangular(self) -> TriangularDistribution:
        if not (self.low <= self.mode <= self.high):
            raise ValueError(
                f"Triangular requires low <= mode <= high, got {self.low}, {self.mode}, {self.high}"
            )
        return self

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.triangular(self.low, self.mode, self.high))


class LogNormalDistribution(Distribution):
    """Log-normal distribution.

    The sampled value is exp(normal(mean, sigma)). Always positive.
    Excellent for costs, home prices, or other right-skewed positive quantities.
    """

    type: Literal["lognormal"] = "lognormal"
    mean: float = Field(description="Mean of the underlying normal (log-space)")
    sigma: float = Field(gt=0, description="Standard deviation of the underlying normal (shape)")

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.lognormal(mean=self.mean, sigma=self.sigma))


class ExponentialDistribution(Distribution):
    """Exponential distribution with given rate (lambda).

    Memoryless; useful for modeling time-to-event or certain tail risks.
    Mean = 1/rate.
    """

    type: Literal["exponential"] = "exponential"
    rate: float = Field(gt=0, description="Rate parameter lambda (> 0)")

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        # numpy uses scale = 1/rate
        return float(rng.exponential(scale=1.0 / self.rate))


class ConstantDistribution(Distribution):
    """Degenerate distribution that always returns the same value.

    Useful inside Monte-Carlo configs for fixed-but-named parameters.
    """

    type: Literal["constant"] = "constant"
    value: float

    def sample(self, rng: np.random.Generator | None = None) -> float:
        return float(self.value)


class BetaDistribution(Distribution):
    """Beta distribution (values strictly between 0 and 1).

    Very useful for modeling probabilities, allocation weights,
    or normalized financial ratios.
    """

    type: Literal["beta"] = "beta"
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.beta(self.alpha, self.beta))


# -----------------------------------------------------------------------------
# Discriminated union + adapter for polymorphic construction from dicts
# -----------------------------------------------------------------------------

AnyDistribution = Annotated[
    NormalDistribution
    | UniformDistribution
    | TriangularDistribution
    | LogNormalDistribution
    | ExponentialDistribution
    | ConstantDistribution
    | BetaDistribution,
    Field(discriminator="type"),
]

# TypeAdapter gives us a reusable validator for the union
_distribution_adapter: TypeAdapter[AnyDistribution] = TypeAdapter(AnyDistribution)


def create_distribution(data: dict[str, Any] | AnyDistribution) -> AnyDistribution:
    """Create a Distribution instance from a config dict or existing object.

    This is the primary entry point when loading from JSON/YAML simulation configs.

    Example:
        dist = create_distribution({
            "type": "triangular",
            "low": -75000,
            "mode": -40000,
            "high": -25000
        })
    """
    if isinstance(data, Distribution):
        return data
    if isinstance(data, dict):
        return _distribution_adapter.validate_python(data)
    raise TypeError(f"Cannot create distribution from type {type(data)}")


__all__ = [
    "Distribution",
    "NormalDistribution",
    "UniformDistribution",
    "TriangularDistribution",
    "LogNormalDistribution",
    "ExponentialDistribution",
    "ConstantDistribution",
    "BetaDistribution",
    "AnyDistribution",
    "create_distribution",
]
