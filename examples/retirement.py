"""
Retirement Planning Monte Carlo Example
"""

from datetime import datetime, timedelta

from financial_simulator.core import (
    AppreciationProcess,
    ComposedEventBuilder,
    FixedValue,
    IntervalTiming,
    SimulationEngine,
)


def create_retirement_engine(seed: int = None) -> SimulationEngine:
    start = datetime(2026, 1, 1)
    end = datetime(2056, 1, 1)  # 30-year horizon

    engine = SimulationEngine(
        name="Retirement Planning",
        start=start,
        end=end,
        initial_state={
            "portfolio_value": 450_000,
            "cumulative_cash": 0.0,
        },
        seed=seed,
    )

    # Monthly retirement contributions (growing with salary)
    engine.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=FixedValue(value=-2500.0),
            metadata={"type": "contribution"},
        )
    )

    # Portfolio growth (continuous)
    engine.add_continuous_process(AppreciationProcess(rate=0.065, var="portfolio_value"))

    # Safe withdrawal phase (starting at year 20)
    # For simplicity in v1 we model it as a large negative event series
    # (a more advanced version would use conditional logic)

    return engine
