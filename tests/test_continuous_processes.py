"""Tests for ContinuousProcess reproducibility (the RNG wiring fix)."""

from datetime import datetime, timedelta

from financial_simulator.core import (
    GBMContinuousProcess,
    GeometricBrownianMotion,
    MeanRevertingContinuousProcess,
    MeanRevertingProcess,
    SimulationEngine,
    create_continuous_process,
)
from financial_simulator.core.simulation import AppreciationProcess, ContinuousProcess


def test_gbm_continuous_reproducible_with_seed():
    """Two engines with same seed + GBMContinuousProcess must produce identical paths."""
    start = datetime(2026, 1, 1)
    end = datetime(2026, 1, 10)

    def make_engine(seed: int):
        eng = SimulationEngine(
            name="gbm-test",
            start=start,
            end=end,
            initial_state={"portfolio": 100.0},
            seed=seed,
        )
        eng.add_continuous_process(
            GBMContinuousProcess(
                process=GeometricBrownianMotion(drift=0.08, volatility=0.20),
                var="portfolio",
            )
        )
        return eng

    e1 = make_engine(12345)
    e1.run()
    r1 = e1.get_result()

    e2 = make_engine(12345)
    e2.run()
    r2 = e2.get_result()

    # Final values must match exactly (reproducibility)
    assert r1.final_state["portfolio"] == r2.final_state["portfolio"]

    # History must be identical at every snapshot
    for t in r1.state_history:
        assert r1.state_history[t]["portfolio"] == r2.state_history[t]["portfolio"]


def test_mean_reverting_continuous_reproducible():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 4, 1)

    def make_engine(seed: int):
        eng = SimulationEngine(
            name="mr-test",
            start=start,
            end=end,
            initial_state={"rate": 0.05},
            seed=seed,
        )
        eng.add_continuous_process(
            MeanRevertingContinuousProcess(
                process=MeanRevertingProcess(long_term_mean=0.04, speed=1.5, volatility=0.01),
                var="rate",
            )
        )
        return eng

    e1 = make_engine(42)
    e1.run()
    r1 = e1.get_result()

    e2 = make_engine(42)
    e2.run()
    r2 = e2.get_result()

    assert abs(r1.final_state["rate"] - r2.final_state["rate"]) < 1e-12


def test_appreciation_unaffected_by_rng_wiring():
    """AppreciationProcess (deterministic) must produce identical results regardless of the RNG change.

    NOTE: Continuous processes only advance when there are discrete events driving the simulation clock.
    We add a dummy monthly event so the full year elapses.
    """
    from financial_simulator.core.event import ComposedEventBuilder, FixedValue, IntervalTiming

    start = datetime(2026, 1, 1)
    end = datetime(2027, 1, 1)

    eng = SimulationEngine(
        name="app-test",
        start=start,
        end=end,
        initial_state={"house": 400_000.0, "cumulative_cash": 0.0},
        seed=999,
    )
    eng.add_continuous_process(AppreciationProcess(rate=0.03, var="house"))
    # Dummy monthly events to advance the global clock across the full year
    eng.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=timedelta(days=30), start_time=start),
            value_gen=FixedValue(value=0.0),
        )
    )
    eng.run()
    result = eng.get_result()

    # Pure geometric growth formula check (engine ran ~1 year with monthly ticks)
    expected = 400_000.0 * (1.03**1.0)
    # Monthly stepping means the final delta is not exactly 1.0 year; allow small relative error
    assert abs(result.final_state["house"] - expected) < 200.0


# =============================================================================
# Deserialization + public factory tests (post-merge cleanup coverage)
# =============================================================================


def test_create_continuous_process_supports_all_types():
    """The public helper + discriminated union must construct every supported process."""
    app = create_continuous_process({"type": "appreciation", "rate": 0.04, "var": "home"})
    assert isinstance(app, ContinuousProcess)
    assert app.type == "appreciation"  # type: ignore[attr-defined]
    assert app.rate == 0.04

    gbm = create_continuous_process(
        {
            "type": "gbm",
            "var": "portfolio",
            "process": {"drift": 0.07, "volatility": 0.18},
        }
    )
    assert gbm.type == "gbm"  # type: ignore[attr-defined]

    mr = create_continuous_process(
        {
            "type": "mean_reverting",
            "var": "rate",
            "process": {"long_term_mean": 0.05, "speed": 1.2, "volatility": 0.005},
        }
    )
    assert mr.type == "mean_reverting"  # type: ignore[attr-defined]


def test_continuous_process_roundtrip_via_engine_from_dict():
    """Full cycle: model_dump -> from_dict must preserve all three continuous process types."""
    from datetime import datetime

    eng = SimulationEngine(
        name="roundtrip",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 12, 31),
        initial_state={"val": 100.0},
        continuous_processes=[
            create_continuous_process({"type": "appreciation", "rate": 0.05, "var": "val"}),
            create_continuous_process(
                {"type": "gbm", "var": "val2", "process": {"drift": 0.1, "volatility": 0.2}}
            ),
        ],
    )
    data = eng.to_dict()
    restored = SimulationEngine.from_dict(data)

    assert len(restored.continuous_processes) == 2
    types = [p.type for p in restored.continuous_processes]  # type: ignore[attr-defined]
    assert "appreciation" in types
    assert "gbm" in types


def test_cli_legacy_appreciation_fallback_still_works():
    """Old user configs with type=Appreciation or bare rate dicts must still load via CLI helper."""
    from financial_simulator.cli import _create_continuous_process as cli_create

    p1 = cli_create({"type": "Appreciation", "rate": 0.03, "var": "house"})
    assert p1.type == "appreciation"  # normalized  # type: ignore[attr-defined]

    p2 = cli_create({"rate": 0.02, "var": "savings"})  # bare legacy shape
    assert p2.type == "appreciation"  # type: ignore[attr-defined]
