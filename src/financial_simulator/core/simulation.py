"""Core simulation engine for discrete-event financial Monte Carlo simulations.

The SimulationEngine is the central orchestrator. It advances time from start to end,
executing the following at each step:

1. Determine the next global event time across all registered EventBuilders.
2. Advance any ContinuousProcess objects over the intervening delta (e.g. property appreciation).
3. Generate and collect all Events scheduled for that exact timestamp.
4. Apply any state updates returned by ValueGenerators (rate changes, etc.).
5. Record state snapshots for later analysis / IRR calculations.

Key design goals:
- Strong typing + Pydantic v2 models
- Full reproducibility via explicit numpy Generator + seed
- Decoupled: EventBuilders and ContinuousProcesses only see `state` dict + optional rng
- Extensible: subclass EventBuilder or ContinuousProcess for custom behavior
- Serializable configuration (the declarative parts)

The legacy `Simulation` dataclass + builder pattern can be migrated to use this engine.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

try:
    from typing import Annotated  # py >= 3.9
except ImportError:
    from typing_extensions import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter

from .event import ComposedEventBuilder, Event, EventBuilder
from .stochastic import GeometricBrownianMotion, MeanRevertingProcess

# =============================================================================
# Continuous Processes (state evolution between discrete events)
# =============================================================================


class ContinuousProcess(BaseModel, ABC):
    """Base class for processes that continuously mutate simulation state.

    These are evaluated between event times using the exact timedelta delta.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str | None = None

    @abstractmethod
    def advance(
        self,
        state: dict[str, Any],
        delta: dt.timedelta,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Mutate the provided state dict in place for the elapsed time.

        The optional rng enables reproducible stochastic continuous processes
        (GBM, MeanReverting). Backward-compatible: existing calls without rng
        continue to work and fall back to non-deterministic sampling.
        """
        ...


class AppreciationProcess(ContinuousProcess):
    """Geometric growth process (e.g. home value appreciation, investment growth).

    state[var] *= (1 + rate) ** (delta_years)
    """

    type: Literal["appreciation"] = "appreciation"
    rate: float = Field(description="Annual growth rate (e.g. 0.04 for 4%)")
    var: str = Field(default="property_value", description="State key to compound")

    def advance(
        self,
        state: dict[str, Any],
        delta: dt.timedelta,
        rng: np.random.Generator | None = None,
    ) -> None:
        if self.var not in state:
            return
        years = delta.total_seconds() / (365.25 * 24 * 3600)
        if years > 0:
            state[self.var] *= (1.0 + self.rate) ** years


class GBMContinuousProcess(ContinuousProcess):
    """Continuous process driven by Geometric Brownian Motion."""

    type: Literal["gbm"] = "gbm"
    process: GeometricBrownianMotion
    var: str

    def advance(
        self,
        state: dict[str, Any],
        delta: dt.timedelta,
        rng: np.random.Generator | None = None,
    ) -> None:
        if self.var not in state:
            return
        current = state[self.var]
        new_val = self.process.step(current, delta, rng=rng)
        state[self.var] = new_val


class MeanRevertingContinuousProcess(ContinuousProcess):
    """Continuous process driven by a mean-reverting stochastic process."""

    type: Literal["mean_reverting"] = "mean_reverting"
    process: MeanRevertingProcess
    var: str

    def advance(
        self,
        state: dict[str, Any],
        delta: dt.timedelta,
        rng: np.random.Generator | None = None,
    ) -> None:
        if self.var not in state:
            return
        current = state[self.var]
        new_val = self.process.step(current, delta, rng=rng)
        state[self.var] = new_val


# Discriminated union for ContinuousProcess (used for serialization, UI, materialization)
AnyContinuousProcess = Annotated[
    Union[AppreciationProcess, GBMContinuousProcess, MeanRevertingContinuousProcess],
    Field(discriminator="type"),
]

_continuous_adapter: TypeAdapter[AnyContinuousProcess] = TypeAdapter(AnyContinuousProcess)


# =============================================================================
# Result object (what you get after engine.run())
# =============================================================================


class SimulationResult(BaseModel):
    """Immutable snapshot of a completed simulation run."""

    name: str
    start: dt.datetime
    end: dt.datetime
    events: list[Event] = Field(default_factory=list)
    final_state: dict[str, Any]
    state_history: dict[dt.datetime, dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationResult:
        return cls.model_validate(data)


# =============================================================================
# Main SimulationEngine
# =============================================================================


class SimulationEngine(BaseModel):
    """The primary entry point for running financial simulations.

    Example usage (simple case):

        from datetime import datetime, timedelta
        from financial_simulator.core import (
            SimulationEngine, ComposedEventBuilder, IntervalTiming, FixedValue
        )

        engine = SimulationEngine(
            name="Rent only",
            start=datetime(2026, 1, 1),
            end=datetime(2027, 1, 1),
            initial_state={"cumulative_cash": 0.0},
        )
        engine.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=timedelta(days=30), start_time=datetime(2026, 2, 1)),
                value_gen=FixedValue(value=2000.0),
                metadata={"type": "rent_income"},
            )
        )
        engine.run()
        result = engine.get_result()
        print(result.final_state["cumulative_cash"])
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # for np.random.Generator if ever exposed
    )

    name: str = "financial-simulation"
    start: dt.datetime
    end: dt.datetime

    # Declarative configuration (persisted / serialized)
    event_builders: list[EventBuilder] = Field(default_factory=list)
    continuous_processes: list[ContinuousProcess] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None

    # Runtime-only (excluded from model_dump by default via exclude)
    events: list[Event] = Field(default_factory=list, exclude=True)
    state: dict[str, Any] = Field(default_factory=dict, exclude=True)
    state_history: dict[dt.datetime, dict[str, Any]] = Field(default_factory=dict, exclude=True)

    _rng: np.random.Generator | None = PrivateAttr(default=None)
    _is_finished: bool = PrivateAttr(default=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_event_builder(self, builder: EventBuilder) -> None:
        """Append an event source. Call before run()."""
        self.event_builders.append(builder)

    def add_continuous_process(self, process: ContinuousProcess) -> None:
        """Append a continuous process. Call before run()."""
        self.continuous_processes.append(process)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Prepare for a fresh run. Re-seeds the RNG and resets all sub-components."""
        self._rng = np.random.default_rng(self.seed)
        self.events = []
        self.state = dict(self.initial_state)  # shallow copy is usually sufficient
        self.state_history = {}
        self._is_finished = False

        for builder in self.event_builders:
            builder.reset(self._rng)

        # Continuous processes are stateless by default; subclasses may override
        # if they need per-run initialization.

    def run(self) -> None:
        """Execute the full simulation.

        After completion, use get_result() or inspect .events / .state_history directly.
        """
        self.reset()

        current = self.start
        self.state_history[current] = dict(self.state)  # snapshot

        while True:
            # 1. Ask every builder for its next event time
            candidates: list[tuple[dt.datetime, EventBuilder]] = []
            for builder in self.event_builders:
                nt = builder.next_event_time(current, self.end, self.state)
                if nt is not None and nt <= self.end:
                    candidates.append((nt, builder))

            if not candidates:
                break

            # 2. Advance global clock to the soonest next event
            next_time = min(nt for nt, _ in candidates)

            # 3. Let continuous processes evolve over the interval
            delta = next_time - current
            for proc in self.continuous_processes:
                proc.advance(self.state, delta, self._rng)

            # 4. Generate events from all builders that fire exactly at next_time
            events_at_time: list[Event] = []
            state_updates: dict[str, Any] = {}

            for nt, builder in candidates:
                if nt == next_time:
                    event = builder.generate_event(next_time, self.state, self._rng)
                    if event is not None:
                        events_at_time.append(event)
                        # Collect any state updates requested by value generators
                        if "state_update" in event.metadata:
                            state_updates.update(event.metadata["state_update"])

            # 5. Commit collected events
            self.events.extend(events_at_time)

            # 6. Apply state updates (after all events at this tick; deterministic)
            if state_updates:
                self.state.update(state_updates)

            # 7. Convenience: auto-track cumulative cash when the key exists
            if "cumulative_cash" in self.state:
                self.state["cumulative_cash"] += sum(e.value for e in events_at_time)

            # 8. Record history and advance
            self.state_history[next_time] = dict(self.state)
            current = next_time

        self._is_finished = True

    # ------------------------------------------------------------------
    # Introspection & results
    # ------------------------------------------------------------------

    def get_result(self) -> SimulationResult:
        """Return an immutable, serializable result object for the last completed run."""
        if not self._is_finished:
            # Allow calling before run for partial results (useful in debugging)
            pass
        return SimulationResult(
            name=self.name,
            start=self.start,
            end=self.end,
            events=list(self.events),
            final_state=dict(self.state),
            state_history={t: dict(s) for t, s in self.state_history.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the declarative configuration (not runtime state)."""
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "initial_state": self.initial_state,
            "seed": self.seed,
            "event_builders": [b.to_dict() for b in self.event_builders],
            "continuous_processes": [p.model_dump(mode="json") for p in self.continuous_processes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationEngine:
        """Rehydrate a SimulationEngine from a serialized configuration.

        Note: runtime state (events, current state) is not restored; call run() again.
        Supports the modern discriminated "type" fields for continuous processes.
        """
        builders = [ComposedEventBuilder.from_dict(b) for b in data.get("event_builders", [])]
        # Use the discriminated union + adapter for robust deserialization of all known process types
        procs: list[ContinuousProcess] = []
        for p in data.get("continuous_processes", []):
            if isinstance(p, ContinuousProcess):
                procs.append(p)
            else:
                procs.append(_continuous_adapter.validate_python(p))

        eng = cls(
            name=data.get("name", "restored-simulation"),
            start=dt.datetime.fromisoformat(data["start"]),
            end=dt.datetime.fromisoformat(data["end"]),
            initial_state=data.get("initial_state", {}),
            seed=data.get("seed"),
            event_builders=builders,
            continuous_processes=procs,
        )
        return eng


__all__ = [
    "ContinuousProcess",
    "AppreciationProcess",
    "GBMContinuousProcess",
    "MeanRevertingContinuousProcess",
    "AnyContinuousProcess",
    "SimulationResult",
    "SimulationEngine",
]
