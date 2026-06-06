"""Event system for discrete financial events.

Core abstractions:
- Event: immutable record of a cash flow at a point in time
- Timing: when events occur (one-time, recurring, random, seasonal)
- ValueGenerator: how large the cash flow is (fixed, growing, stochastic, loan amortizing)
- EventBuilder: combines timing + value generation + metadata into a source of events

All major classes are Pydantic models for easy configuration, serialization,
and validation. Stateful objects (timings, generators, builders) use PrivateAttr
for run-time mutable state (current_next, fired flags, etc.) while remaining
serializable via their declarative fields.

The design deliberately decouples from the SimulationEngine by accepting
`state: dict` and optional `rng` explicitly.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Literal

try:
    from typing import Annotated  # py >= 3.9
except ImportError:
    from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, field_validator

from .distributions import AnyDistribution, create_distribution

# =============================================================================
# Event
# =============================================================================


CASH_FLOW_DIRECTION_KEY = "cash_flow_direction"
CASH_FLOW_ADDITIVE = "additive"
CASH_FLOW_SUBTRACTIVE = "subtractive"


class Event(BaseModel):
    """A single financial event (cash flow) at a specific time.

    Attributes:
        time: When the event occurs.
        value: Cash impact. Positive = inflow, negative = outflow (expense or payment).
        metadata: Arbitrary extra information (e.g. {'type': 'rent_income', 'interest': 123.4}).
    """

    time: dt.datetime
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="allow",  # allow extra runtime metadata if useful
        frozen=True,  # events are records; immutable once created
    )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (datetimes become ISO strings)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        return cls.model_validate(data)


def signed_cash_flow_amount(event: Event) -> float:
    """Apply the generator's explicit direction to the event magnitude.

    Generator value inputs are treated as non-negative magnitudes; direction is
    controlled by ``metadata[cash_flow_direction]`` (additive or subtractive).
    """
    magnitude = abs(float(event.value))
    direction = str(event.metadata.get(CASH_FLOW_DIRECTION_KEY, CASH_FLOW_ADDITIVE)).lower()
    if direction == CASH_FLOW_SUBTRACTIVE:
        return -magnitude
    return magnitude


# =============================================================================
# Timing strategies (when events fire)
# =============================================================================


class Timing(BaseModel, ABC):
    """Abstract base for event timing policies."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    @abstractmethod
    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset internal state machine. Call before each simulation run."""
        ...

    @abstractmethod
    def next_time(
        self, current: dt.datetime, end: dt.datetime, state: dict[str, Any]
    ) -> dt.datetime | None:
        """Return the next candidate event time >= current (or None if exhausted)."""
        ...

    @abstractmethod
    def advance(self) -> None:
        """Advance the internal cursor after an event at the previously returned time has fired."""
        ...


class OneTimeTiming(Timing):
    """Fires exactly once at a fixed (or pre-sampled) time."""

    type: Literal["OneTime"] = "OneTime"
    time: dt.datetime
    _fired: bool = PrivateAttr(default=False)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._fired = False

    def next_time(
        self, current: dt.datetime, end: dt.datetime, state: dict[str, Any]
    ) -> dt.datetime | None:
        if not self._fired and current <= self.time <= end:
            return self.time
        return None

    def advance(self) -> None:
        self._fired = True


class IntervalTiming(Timing):
    """Recurring fixed-interval events (e.g. monthly rent or loan payments)."""

    type: Literal["Interval"] = "Interval"
    interval: dt.timedelta
    start_time: dt.datetime | None = None

    _current_next: dt.datetime | None = PrivateAttr(default=None)

    @field_validator("interval")
    @classmethod
    def interval_must_be_positive(cls, v: dt.timedelta) -> dt.timedelta:
        if v <= dt.timedelta(0):
            raise ValueError(
                "interval must be strictly positive (zero/negative causes infinite loops)"
            )
        return v

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._current_next = self.start_time

    def next_time(
        self, current: dt.datetime, end: dt.datetime, state: dict[str, Any]
    ) -> dt.datetime | None:
        if self._current_next is None:
            self._current_next = (
                current + self.interval if self.start_time is None else self.start_time
            )
        # Advance until we are at or past 'current'
        while self._current_next < current:
            self._current_next += self.interval
        if self._current_next > end:
            return None
        return self._current_next

    def advance(self) -> None:
        if self._current_next is not None:
            self._current_next += self.interval


class RandomTiming(Timing):
    """Fires a fixed number (n) of times at randomly sampled instants within [start, end].

    Sampling occurs once on reset (using supplied rng for reproducibility).
    Times are sorted chronologically.
    """

    type: Literal["Random"] = "Random"
    start: dt.datetime
    end: dt.datetime
    n: int = Field(ge=1)
    distribution: Literal["uniform"] = "uniform"  # extensible later

    _times: list[dt.datetime] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        delta_days = (self.end - self.start).days
        if self.distribution == "uniform":
            random_days = sorted(rng.integers(0, delta_days + 1, size=self.n))
            self._times = [self.start + dt.timedelta(days=int(d)) for d in random_days]
        else:
            raise ValueError(f"Unsupported distribution for RandomTiming: {self.distribution}")
        self._index = 0

    def next_time(
        self, current: dt.datetime, end: dt.datetime, state: dict[str, Any]
    ) -> dt.datetime | None:
        while self._index < len(self._times) and self._times[self._index] < current:
            self._index += 1
        if self._index < len(self._times) and self._times[self._index] <= end:
            return self._times[self._index]
        return None

    def advance(self) -> None:
        self._index += 1


class SeasonalTiming(Timing):
    """Wraps another Timing and only allows events in the given months (1-12)."""

    type: Literal["Seasonal"] = "Seasonal"
    inner: AnyTiming
    months: list[int] = Field(min_length=1)

    _allowed_months: set[int] = PrivateAttr(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        self._allowed_months = set(self.months)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self.inner.reset(rng)
        # months set already populated

    def next_time(
        self, current: dt.datetime, end: dt.datetime, state: dict[str, Any]
    ) -> dt.datetime | None:
        nt = self.inner.next_time(current, end, state)
        while nt is not None and nt.month not in self._allowed_months:
            self.inner.advance()
            nt = self.inner.next_time(nt, end, state)
        return nt

    def advance(self) -> None:
        self.inner.advance()


# Discriminated union for Timing (used for fields + validation)
AnyTiming = Annotated[
    OneTimeTiming | IntervalTiming | RandomTiming | SeasonalTiming,
    Field(discriminator="type"),
]

_timing_adapter: TypeAdapter[AnyTiming] = TypeAdapter(AnyTiming)


def create_timing(data: dict[str, Any] | Timing) -> Timing:
    """Factory used by legacy config loaders. Delegates to Pydantic validation.

    Accepts both the modern Pydantic field names and the legacy shapes used
    by the existing simulation_server / config.json (e.g. "interval_days").
    """
    if isinstance(data, Timing):
        return data

    d = dict(data)  # shallow copy so we can mutate safely

    # Legacy -> modern normalization for IntervalTiming
    if d.get("type") == "Interval" and "interval_days" in d:
        days = d.pop("interval_days")
        d["interval"] = dt.timedelta(days=days)

    # Recursively materialize nested Seasonal.inner (supports legacy dicts)
    if d.get("type") == "Seasonal" and isinstance(d.get("inner"), dict):
        d["inner"] = create_timing(d["inner"])

    return _timing_adapter.validate_python(d)


# =============================================================================
# Value Generators (how large is the cash flow / what side effects)
# =============================================================================


class ValueGenerator(BaseModel, ABC):
    """Produces the numeric value (and optional side effects) for an event."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    @abstractmethod
    def reset(self, rng: np.random.Generator | None = None) -> None: ...

    @abstractmethod
    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Return (cash_value, extra_metadata).

        extra_metadata may contain the special key 'state_update' (dict) which the
        SimulationEngine will merge into the simulation state after the event.
        """
        ...


class FixedValue(ValueGenerator):
    """Always emits the same constant cash value.

    Can be constructed as FixedValue(1234) or FixedValue(value=1234) for convenience.
    """

    type: Literal["Fixed"] = "Fixed"
    value: float

    def __init__(self, value: float = None, **data):
        if value is not None and "value" not in data:
            data["value"] = value
        super().__init__(**data)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        return self.value, {}


class GrowingValue(ValueGenerator):
    """Emits a value that compounds continuously (exponentially) between calls.

    Typical use: growing rental income or expense inflation.
    """

    type: Literal["Growing"] = "Growing"
    initial: float
    growth_rate: float  # annual continuous compounding rate, e.g. 0.03

    _current: float = PrivateAttr(default=0.0)
    _last_time: dt.datetime | None = PrivateAttr(default=None)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._current = self.initial
        self._last_time = None

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        if self._last_time is not None:
            delta_years = (time - self._last_time).days / 365.25
            if delta_years > 0:
                self._current *= (1 + self.growth_rate) ** delta_years
        self._last_time = time
        return self._current, {}


class DistributionValue(ValueGenerator):
    """Samples a fresh value from a distribution on every event."""

    type: Literal["Distribution"] = "Distribution"
    dist: AnyDistribution

    def reset(self, rng: np.random.Generator | None = None) -> None:
        # Distributions are stateless
        pass

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        val = self.dist.sample(rng)
        return val, {}


class RateChangeValue(ValueGenerator):
    """Emits a zero cash flow whose only purpose is to update a named state variable.

    The sampled value is placed under 'state_update'.
    """

    type: Literal["RateChange"] = "RateChange"
    dist: AnyDistribution
    update_key: str

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        new_val = self.dist.sample(rng)
        return 0.0, {"state_update": {self.update_key: new_val}}


class VariableRateLoanValue(ValueGenerator):
    """Amortizing loan payment calculator with variable interest rate.

    Reads current rate from state[rate_key] (falls back to initial_rate).
    Returns negative payment (outflow) + rich metadata (interest, principal, remaining_balance).
    """

    type: Literal["VariableRateLoan"] = "VariableRateLoan"
    principal: float
    initial_rate: float
    term_months: int
    rate_key: str

    _balance: float = PrivateAttr(default=0.0)
    _month: int = PrivateAttr(default=0)
    _last_time: dt.datetime | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        # Ensure internal state is initialized from declarative fields on construction
        self._balance = self.principal
        self._month = 0
        self._last_time = None

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._balance = self.principal
        self._month = 0
        self._last_time = None

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        if self._month >= self.term_months or self._balance <= 0:
            return 0.0, {}

        current_rate = state.get(self.rate_key, self.initial_rate)
        monthly_rate = current_rate / 12.0
        remaining = self.term_months - self._month

        if monthly_rate == 0:
            payment = self._balance / remaining if remaining > 0 else 0.0
        else:
            r = (1 + monthly_rate) ** remaining
            payment = self._balance * monthly_rate * r / (r - 1)

        interest = self._balance * monthly_rate
        principal_paid = min(payment - interest, self._balance)
        if principal_paid < 0:
            principal_paid = 0.0
            payment = interest

        self._balance -= principal_paid
        self._month += 1
        self._last_time = time

        extra: dict[str, Any] = {
            "interest": interest,
            "principal": principal_paid,
            "rate": current_rate,
            "remaining_balance": max(self._balance, 0.0),
        }
        return -payment, extra


class DividendValue(ValueGenerator):
    """Models dividend or distribution income from an investment."""

    type: Literal["Dividend"] = "Dividend"
    annual_yield: float
    investment_value_key: str = "portfolio_value"

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        portfolio_value = state.get(self.investment_value_key, 0.0)
        dividend = portfolio_value * self.annual_yield / 12  # monthly approximation
        return dividend, {"source": "dividend"}


class InvestmentContributionValue(ValueGenerator):
    """Models regular contributions into an investment account."""

    type: Literal["InvestmentContribution"] = "InvestmentContribution"
    amount: float
    growth_key: str | None = None  # if set, contribution grows with this state variable

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        amount = self.amount
        if self.growth_key and self.growth_key in state:
            amount *= state[self.growth_key]
        return -amount, {"type": "investment_contribution"}


class TaxEventValue(ValueGenerator):
    """
    Simple tax event generator.
    Applies a tax rate to a base amount stored in state.
    """

    type: Literal["TaxEvent"] = "TaxEvent"
    rate: float
    base_key: str
    tax_key: str = "tax_paid"

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass

    def get_value(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> tuple[float, dict[str, Any]]:
        base = state.get(self.base_key, 0.0)
        tax = base * self.rate
        return -tax, {
            "tax": tax,
            "state_update": {self.tax_key: state.get(self.tax_key, 0.0) + tax},
        }


# ValueGenerator discriminated union + factory
AnyValueGenerator = Annotated[
    FixedValue
    | GrowingValue
    | DistributionValue
    | RateChangeValue
    | VariableRateLoanValue
    | DividendValue
    | InvestmentContributionValue
    | TaxEventValue,
    Field(discriminator="type"),
]

_vg_adapter: TypeAdapter[AnyValueGenerator] = TypeAdapter(AnyValueGenerator)


def create_value_generator(data: dict[str, Any] | ValueGenerator) -> ValueGenerator:
    """Factory for ValueGenerator from config dicts (supports nested distributions)."""
    if isinstance(data, ValueGenerator):
        return data
    d = dict(data)
    if "dist" in d and isinstance(d["dist"], dict):
        d["dist"] = create_distribution(d["dist"])
    return _vg_adapter.validate_python(d)


# =============================================================================
# EventBuilder (the primary extension point)
# =============================================================================


class EventBuilder(BaseModel, ABC):
    """Produces a stream of Events during a simulation run.

    Concrete implementations are typically composed (Timing + ValueGenerator).
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def reset(self, rng: np.random.Generator | None = None) -> None: ...

    @abstractmethod
    def next_event_time(
        self, current: dt.datetime, end: dt.datetime, state: dict[str, Any]
    ) -> dt.datetime | None: ...

    @abstractmethod
    def generate_event(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> Event | None:
        """Return an Event if one is due at exactly this time, else None.

        The engine is responsible for calling next_event_time first to decide the
        global next time, then calling generate_event only on builders that match.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventBuilder:
        return cls.model_validate(data)


class ComposedEventBuilder(EventBuilder):
    """The standard concrete implementation: Timing + ValueGenerator + metadata."""

    timing: AnyTiming
    value_gen: AnyValueGenerator

    # Internal cursor used by next_event_time to avoid recomputing
    _current_next: dt.datetime | None = PrivateAttr(default=None)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self.timing.reset(rng)
        self.value_gen.reset(rng)
        self._current_next = None

    def next_event_time(
        self, current: dt.datetime, end: dt.datetime, state: dict[str, Any]
    ) -> dt.datetime | None:
        if self._current_next is None or self._current_next <= current:
            nt = self.timing.next_time(current, end, state)
            # Skip any stale times that are in the past (defensive)
            while nt is not None and nt < current:
                self.timing.advance()
                nt = self.timing.next_time(nt, end, state)
            self._current_next = nt
        return self._current_next

    def generate_event(
        self, time: dt.datetime, state: dict[str, Any], rng: np.random.Generator | None = None
    ) -> Event | None:
        scheduled = self.next_event_time(time, time, state)  # cheap check using cached
        if scheduled != time:
            return None

        cash_value, extra_meta = self.value_gen.get_value(time, state, rng)
        merged_meta = {**self.metadata, **extra_meta}

        # Support the convention used by RateChangeValue etc.
        if "state_update" in extra_meta:
            # Do NOT mutate state here — return the info so engine can do it
            # after collecting all events at this timestamp (deterministic order).
            pass

        event = Event(time=time, value=cash_value, metadata=merged_meta)

        # Advance only after successful generation
        self.timing.advance()
        self._current_next = None  # force recalculation on next call
        return event


# Top-level factory used by legacy simulation_server / config loaders
def create_event_builder(data: dict[str, Any]) -> EventBuilder:
    """Create a ComposedEventBuilder (or future subclasses) from a full config dict."""
    timing = create_timing(data["timing"])
    value_gen = create_value_generator(data["value_gen"])
    metadata = data.get("metadata", {})
    name = data.get("name")
    return ComposedEventBuilder(timing=timing, value_gen=value_gen, metadata=metadata, name=name)


__all__ = [
    "CASH_FLOW_DIRECTION_KEY",
    "CASH_FLOW_ADDITIVE",
    "CASH_FLOW_SUBTRACTIVE",
    "signed_cash_flow_amount",
    "Event",
    "Timing",
    "OneTimeTiming",
    "IntervalTiming",
    "RandomTiming",
    "SeasonalTiming",
    "ValueGenerator",
    "FixedValue",
    "GrowingValue",
    "DistributionValue",
    "RateChangeValue",
    "VariableRateLoanValue",
    "DividendValue",
    "InvestmentContributionValue",
    "TaxEventValue",
    "EventBuilder",
    "ComposedEventBuilder",
    "create_timing",
    "create_value_generator",
    "create_event_builder",
]
