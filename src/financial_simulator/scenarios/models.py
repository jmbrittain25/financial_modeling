"""
Pydantic data models for the interactive scenario builder.

These models are the single source of truth for:
- What users build and save in the UI
- Template library
- JSON/YAML round-tripping
- Materialization into runnable SimulationEngine instances

They deliberately reuse the core discriminated unions (AnyDistribution,
ComposedEventBuilder, AnyTiming, AnyValueGenerator, AnyContinuousProcess)
so there is zero duplication of validation or serialization logic.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

try:
    from typing import Annotated  # py >= 3.9
except ImportError:
    from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Core imports (reused heavily)
from ..core.distributions import AnyDistribution
from ..core.event import (
    AnyTiming,
    ComposedEventBuilder,
)
from ..core.simulation import (
    AnyContinuousProcess,
)

# =============================================================================
# Saved / Reusable Distributions (the "build and save custom distributions" feature)
# =============================================================================


class SavedDistribution(BaseModel):
    """A named, reusable distribution that users can save in their personal library."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier (slug or uuid)")
    name: str = Field(min_length=1, description="Human-friendly name shown in UI pickers")
    description: str = ""
    dist: AnyDistribution
    tags: list[str] = Field(default_factory=list)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class DistributionLibrary(BaseModel):
    """Collection of SavedDistribution entries with persistence helpers."""

    model_config = ConfigDict(extra="forbid")

    distributions: list[SavedDistribution] = Field(default_factory=list)

    def add(self, saved: SavedDistribution) -> None:
        if any(d.id == saved.id for d in self.distributions):
            raise ValueError(f"Distribution with id {saved.id} already exists")
        self.distributions.append(saved)

    def remove(self, dist_id: str) -> bool:
        before = len(self.distributions)
        self.distributions = [d for d in self.distributions if d.id != dist_id]
        return len(self.distributions) < before

    def get(self, dist_id: str) -> SavedDistribution | None:
        for d in self.distributions:
            if d.id == dist_id:
                return d
        return None

    def get_by_name(self, name: str) -> SavedDistribution | None:
        for d in self.distributions:
            if d.name.lower() == name.lower():
                return d
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributionLibrary:
        return cls.model_validate(data)


# =============================================================================
# Custom Metrics (user-defined quantities tracked across Monte Carlo runs)
# =============================================================================


class CustomMetric(BaseModel):
    """Definition of a scalar metric computed from a completed SimulationResult."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Short identifier used in reports")
    description: str = ""
    metric_type: Literal[
        "final_state_value",  # params: {"key": "portfolio_value"}
        "sum_positive_events",  # optional params: {"metadata_type": "revenue"}
        "max_drawdown_on_path",  # params: {"state_key": "portfolio_value"}
        "event_count_by_type",  # params: {"metadata_type": "contribution"}
        "time_to_threshold",  # params: {"state_key": "...", "threshold": 0.0, "direction": "above"}
    ]
    params: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# External Drivers (e.g. stochastic interest rate paths that affect loans)
# =============================================================================


class DiscreteRateDriver(BaseModel):
    """A driver that periodically samples a distribution and writes it into state.

    Materializes as a ComposedEventBuilder using RateChangeValue + the supplied timing.
    Perfect for variable-rate loans (pairs with VariableRateLoanValue(rate_key=...)).
    """

    type: Literal["discrete_rate"] = "discrete_rate"
    name: str
    target_state_key: str
    dist: AnyDistribution
    timing: AnyTiming
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstantDriver(BaseModel):
    """Simple constant value injected once at start into the target state key."""

    type: Literal["constant"] = "constant"
    name: str
    target_state_key: str
    value: float


# Future driver kinds (stubs for UI extensibility; full support added in later phases)
class ContinuousGBMDriver(BaseModel):
    type: Literal["gbm_continuous"] = "gbm_continuous"
    name: str
    target_state_key: str
    drift: float
    volatility: float
    initial_value: float


class ContinuousMeanRevertDriver(BaseModel):
    type: Literal["mean_revert_continuous"] = "mean_revert_continuous"
    name: str
    target_state_key: str
    long_term_mean: float
    speed: float
    volatility: float
    initial_value: float


AnyExternalDriver = Annotated[
    DiscreteRateDriver | ConstantDriver | ContinuousGBMDriver | ContinuousMeanRevertDriver,
    Field(discriminator="type"),
]


# =============================================================================
# Top-Level Scenario Configuration
# =============================================================================


class ScenarioConfig(BaseModel):
    """The primary user-facing, savable, loadable, and runnable scenario document.

    This is what the Streamlit scenario builder edits, what gets saved as JSON,
    what templates are made of, and what the materialization layer turns into
    a runnable SimulationEngine (plus custom metrics and driver expansion).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_encoders={
            # Pydantic v2 handles most via mode="json"; keep for explicitness
            dt.datetime: lambda v: v.isoformat(),
        },
    )

    version: Literal["1.0"] = "1.0"
    name: str = "Untitled Scenario"
    description: str | None = None
    created_at: dt.datetime | None = None
    tags: list[str] = Field(default_factory=list)
    is_template: bool = False

    start: dt.datetime
    end: dt.datetime
    initial_state: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None

    # Core simulation declarative content (reuse battle-tested core models)
    event_builders: list[ComposedEventBuilder] = Field(default_factory=list)
    continuous_processes: list[AnyContinuousProcess] = Field(default_factory=list)

    # New scenario-builder power features
    external_drivers: list[AnyExternalDriver] = Field(default_factory=list)
    custom_metrics: list[CustomMetric] = Field(default_factory=list)

    notes: str | None = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioConfig:
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True)

    @classmethod
    def from_json(cls, text: str) -> ScenarioConfig:
        return cls.model_validate_json(text)


__all__ = [
    "SavedDistribution",
    "DistributionLibrary",
    "CustomMetric",
    "DiscreteRateDriver",
    "ConstantDriver",
    "ContinuousGBMDriver",
    "ContinuousMeanRevertDriver",
    "AnyExternalDriver",
    "ScenarioConfig",
]
