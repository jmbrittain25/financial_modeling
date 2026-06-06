"""Tests for SimulationEngine core behavior, edge cases, continuous processes, and state management.

MonteCarloRunner tests live in test_monte_carlo.py.
"""

import datetime as dt

import pytest

from financial_simulator.core import (
    AppreciationProcess,
    ComposedEventBuilder,
    SimulationEngine,
    SimulationResult,
)
from financial_simulator.core.distributions import NormalDistribution
from financial_simulator.core.event import (
    FixedValue,
    IntervalTiming,
    OneTimeTiming,
    RateChangeValue,
)
from financial_simulator.core.simulation import SimulationStuckError

# --- Basic functionality ---


def test_basic_engine_run(simple_engine: SimulationEngine):
    simple_engine.run()
    result = simple_engine.get_result()
    assert len(result.events) >= 2
    assert result.final_state["cumulative_cash"] > 0


def test_engine_reproducibility():
    def make_engine(seed: int) -> SimulationEngine:
        eng = SimulationEngine(
            name="Repro",
            start=dt.datetime(2026, 1, 1),
            end=dt.datetime(2026, 7, 1),
            initial_state={"cumulative_cash": 0.0},
            seed=seed,
        )
        eng.add_event_builder(
            ComposedEventBuilder(
                timing=IntervalTiming(interval=dt.timedelta(days=30)),
                value_gen=FixedValue(value=500.0),
            )
        )
        return eng

    e1 = make_engine(123)
    e1.run()
    r1 = e1.get_result()

    e2 = make_engine(123)
    e2.run()
    r2 = e2.get_result()

    assert r1.final_state["cumulative_cash"] == r2.final_state["cumulative_cash"]


# --- Edge cases ---


def test_engine_raises_simulation_stuck_error_on_non_advancing_clock():
    """Engine guard: misconfigured zero-interval timing must not spin forever."""
    bad_timing = IntervalTiming.model_construct(interval=dt.timedelta(0))
    eng = SimulationEngine(
        name="stuck",
        start=dt.datetime(2026, 1, 1),
        end=dt.datetime(2026, 12, 31),
        initial_state={"cumulative_cash": 0.0},
        seed=1,
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=bad_timing,
            value_gen=FixedValue(value=100.0),
        )
    )
    with pytest.raises(SimulationStuckError, match="stuck"):
        eng.run()


def test_engine_with_no_builders_produces_single_state_snapshot(
    start_2026: dt.datetime, end_2026_mid: dt.datetime
):
    eng = SimulationEngine(
        name="empty",
        start=start_2026,
        end=end_2026_mid,
        initial_state={"x": 42.0},
    )
    eng.run()
    result = eng.get_result()
    assert len(result.events) == 0
    assert len(result.state_history) == 1  # only the initial snapshot
    assert result.final_state["x"] == 42.0


def test_engine_zero_duration_still_records_start(start_2026: dt.datetime):
    eng = SimulationEngine(
        name="zero",
        start=start_2026,
        end=start_2026,
        initial_state={"cumulative_cash": 0.0},
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=OneTimeTiming(time=start_2026),
            value_gen=FixedValue(value=100),
        )
    )
    eng.run()
    result = eng.get_result()
    assert len(result.state_history) == 1
    assert len(result.events) == 1


def test_engine_events_entirely_before_window_still_generate_when_catchup_logic_fires():
    # IntervalTiming with start far in the past will fast-forward
    eng = SimulationEngine(
        name="past",
        start=dt.datetime(2027, 1, 1),
        end=dt.datetime(2027, 3, 1),
        initial_state={"cumulative_cash": 0.0},
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(
                interval=dt.timedelta(days=30), start_time=dt.datetime(2020, 1, 1)
            ),
            value_gen=FixedValue(value=100),
        )
    )
    eng.run()
    result = eng.get_result()
    # Should still produce events after fast-forward
    assert len(result.events) > 0


def test_get_result_before_run_returns_partial_state():
    eng = SimulationEngine(
        name="pre",
        start=dt.datetime(2026, 1, 1),
        end=dt.datetime(2026, 2, 1),
        initial_state={"foo": 1},
    )
    result = eng.get_result()
    assert result.name == "pre"
    # Before run, events/state are empty but object is valid
    assert isinstance(result, SimulationResult)


# --- State updates from ValueGenerators ---


def test_rate_change_value_updates_state_deterministically():
    eng = SimulationEngine(
        name="rate-change",
        start=dt.datetime(2026, 1, 1),
        end=dt.datetime(2026, 4, 1),
        initial_state={"cumulative_cash": 0.0, "interest_rate": 0.03},
        seed=55,
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=OneTimeTiming(time=dt.datetime(2026, 2, 15)),
            value_gen=RateChangeValue(
                dist=NormalDistribution(mean=0.07, std=1e-9),  # effectively deterministic
                update_key="interest_rate",
            ),
        )
    )
    eng.run()
    result = eng.get_result()
    # The state at/after the event should reflect the update
    post = [s for t, s in result.state_history.items() if t >= dt.datetime(2026, 2, 15)]
    assert any(abs(s.get("interest_rate", 0) - 0.07) < 1e-9 for s in post)


def test_cumulative_cash_auto_tracking_with_mixed_events():
    eng = SimulationEngine(
        name="cash-track",
        start=dt.datetime(2026, 1, 1),
        end=dt.datetime(2026, 3, 1),
        initial_state={"cumulative_cash": 0.0},
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(
                interval=dt.timedelta(days=30), start_time=dt.datetime(2026, 1, 1)
            ),
            value_gen=FixedValue(value=1000),
        )
    )
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(
                interval=dt.timedelta(days=30), start_time=dt.datetime(2026, 1, 15)
            ),
            value_gen=FixedValue(value=-300),
        )
    )
    eng.run()
    result = eng.get_result()
    # Final cumulative should be net of all events
    assert (
        result.final_state["cumulative_cash"] == 1400.0
    )  # 2*1000 - 2*300 (approx, depending on exact dates)


# --- Continuous processes ---


def test_appreciation_process_compounds_between_events(engine_with_continuous: SimulationEngine):
    engine_with_continuous.run()
    result = engine_with_continuous.get_result()
    # Portfolio should have grown beyond initial 100k minus contributions
    final_port = result.final_state["portfolio"]
    assert final_port > 100_000.0 * 1.03  # at least some growth


def test_appreciation_process_no_op_when_var_missing():
    eng = SimulationEngine(
        name="no-app",
        start=dt.datetime(2026, 1, 1),
        end=dt.datetime(2026, 2, 1),
        initial_state={"other": 123.0},
    )
    eng.add_continuous_process(AppreciationProcess(rate=0.10, var="missing"))
    eng.run()
    # Should not crash; state unchanged
    assert eng.state["other"] == 123.0


# --- Result object ---


def test_simulation_result_is_frozen_and_serializable(simple_engine: SimulationEngine):
    simple_engine.run()
    result = simple_engine.get_result()
    assert result.model_config.get("frozen") is True or result.model_config.get("frozen") is True
    data = result.model_dump(mode="json")
    assert "events" in data
    assert "final_state" in data


def test_to_dict_and_from_dict_roundtrip(simple_engine: SimulationEngine):
    simple_engine.run()
    d = simple_engine.to_dict()
    eng2 = SimulationEngine.from_dict(d)
    eng2.run()
    r1 = simple_engine.get_result()
    r2 = eng2.get_result()
    # Declarative parts match; runtime events may differ slightly due to RNG but structure same
    assert r1.name == r2.name
    assert len(r1.events) == len(r2.events)  # deterministic in this case (no stochastic builders)
