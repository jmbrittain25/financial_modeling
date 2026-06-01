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
    """A named, reusable distribution that users can save in their personal library.

    Extra UI fields (units, domain_hint, last_used) are optional and used only by
    the interactive scenario builder / library browser for better filtering and display.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier (slug or uuid)")
    name: str = Field(min_length=1, description="Human-friendly name shown in UI pickers")
    description: str = ""
    dist: AnyDistribution
    tags: list[str] = Field(default_factory=list)
    units: str | None = Field(default=None, description="E.g. 'USD', '%', 'rate'")
    domain_hint: Literal["absolute", "rate", "fraction", "time"] | None = Field(
        default=None, description="Helps UI pick sensible defaults and validation"
    )
    last_used: dt.datetime | None = None
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


class ScenarioLibrary(BaseModel):
    """Collection of user-saved ScenarioConfig documents.

    Symmetric to DistributionLibrary. Used by the 'My Scenarios' browser and
    persistence layer for disk-backed personal libraries.
    """

    model_config = ConfigDict(extra="forbid")

    scenarios: list[ScenarioConfig] = Field(default_factory=list)

    def add(self, scenario: ScenarioConfig) -> None:
        # Use name as natural key for simplicity (user can rename to avoid collisions)
        if any(s.name == scenario.name for s in self.scenarios):
            raise ValueError(f"Scenario named '{scenario.name}' already exists in library")
        self.scenarios.append(scenario)

    def remove(self, name: str) -> bool:
        before = len(self.scenarios)
        self.scenarios = [s for s in self.scenarios if s.name != name]
        return len(self.scenarios) < before

    def get(self, name: str) -> ScenarioConfig | None:
        for s in self.scenarios:
            if s.name == name:
                return s
        return None

    def get_by_name(self, name: str) -> ScenarioConfig | None:
        return self.get(name)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioLibrary:
        return cls.model_validate(data)


# =============================================================================
# Custom Metrics (user-defined quantities tracked across Monte Carlo runs)
# =============================================================================


class CustomMetric(BaseModel):
    """Definition of a scalar metric computed from a completed SimulationResult.

    The UI-only fields (display_format, unit_label, higher_is_better, goal_value) do not
    affect computation in metrics.py — they only improve presentation in the Streamlit
    results dashboard and custom metrics editor.
    """

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

    # UI / presentation hints (additive, ignored by compute_metric)
    display_format: Literal["currency", "percent", "number", "years", "count"] = "number"
    unit_label: str = ""
    higher_is_better: bool | None = None
    goal_value: float | None = None


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
    """Geometric Brownian Motion driver (for equity-style paths)."""

    type: Literal["gbm_continuous"] = "gbm_continuous"
    name: str
    description: str = ""
    target_state_key: str
    drift: float
    volatility: float = Field(gt=0)
    initial_value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContinuousMeanRevertDriver(BaseModel):
    """Mean-reverting driver (for inflation, interest rates, etc.)."""

    type: Literal["mean_revert_continuous"] = "mean_revert_continuous"
    name: str
    description: str = ""
    target_state_key: str
    long_term_mean: float
    speed: float = Field(gt=0)
    volatility: float = Field(gt=0)
    initial_value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


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

    # Lightweight UI-only hints (e.g. last open panel, favorite flag). Never used by
    # materialization or the core engine — safe to ignore on load.
    ui_hints: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors & UI helpers
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

    # ------------------------------------------------------------------
    # UI / builder helpers (pure, no side effects)
    # ------------------------------------------------------------------

    def clone(self) -> ScenarioConfig:
        """Return a deep copy suitable for 'Save as' / duplication in the UI."""
        return self.model_copy(deep=True)

    def get_all_referenced_distributions(self) -> list[AnyDistribution]:
        """Harvest every distribution embedded anywhere in the scenario (for library views)."""
        dists: list[AnyDistribution] = []
        # From event builders (DistributionValue, RateChangeValue, etc.)
        for eb in self.event_builders:
            vg = getattr(eb, "value_gen", None)
            d = getattr(vg, "dist", None) if vg is not None else None
            if d is not None:
                dists.append(d)
        # From external drivers (DiscreteRateDriver etc.)
        for drv in self.external_drivers:
            d = getattr(drv, "dist", None)
            if d is not None:
                dists.append(d)
        return dists

    def summary(self) -> dict[str, Any]:
        """Compact stats for sidebar cards and gallery previews."""
        horizon_days = (self.end - self.start).days if self.end and self.start else 0
        return {
            "name": self.name,
            "horizon_years": round(horizon_days / 365.25, 1) if horizon_days > 0 else 0.0,
            "num_event_builders": len(self.event_builders),
            "num_continuous": len(self.continuous_processes),
            "num_drivers": len(self.external_drivers),
            "num_custom_metrics": len(self.custom_metrics),
            "has_stochastic": any(
                "Distribution" in str(type(getattr(eb.value_gen, "dist", None)))
                or hasattr(getattr(eb.value_gen, "dist", None), "sample")
                for eb in self.event_builders
            ),
            "state_keys": sorted(self.initial_state.keys()),
        }


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
]
