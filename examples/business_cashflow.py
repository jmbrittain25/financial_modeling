"""
Business Cash Flow Monte Carlo Example
"""

from datetime import datetime, timedelta

from financial_simulator.core import (
    ComposedEventBuilder,
    DistributionValue,
    FixedValue,
    IntervalTiming,
    SimulationEngine,
    TriangularDistribution,
)


def create_business_engine(seed: int = None) -> SimulationEngine:
    start = datetime(2026, 1, 1)
    end = datetime(2028, 1, 1)

    engine = SimulationEngine(
        name="Small Business Cash Flow",
        start=start,
        end=end,
        initial_state={"cash": 120_000, "cumulative_cash": 0.0},
        seed=seed,
    )

    # Recurring revenue
    engine.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=FixedValue(value=28_000),
            metadata={"type": "revenue"},
        )
    )

    # Variable operating expenses (realistic monthly range with variation)
    engine.add_event_builder(
        ComposedEventBuilder(
            timing=IntervalTiming(interval=timedelta(days=30)),
            value_gen=DistributionValue(
                dist=TriangularDistribution(low=-31000, mode=-26000, high=-22000)
            ),
            metadata={"type": "opex"},
        )
    )

    return engine
