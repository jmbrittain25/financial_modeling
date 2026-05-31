"""Core module of the financial simulation platform.

This package provides the foundational, strongly-typed, Pydantic-powered
building blocks:

- distributions: reusable stochastic distributions (Normal, Triangular, LogNormal, ...)
- event: Event, Timing strategies, ValueGenerators, and the EventBuilder abstraction
- simulation: SimulationEngine + ContinuousProcess + SimulationResult

All components are designed for:
- Monte Carlo reproducibility (explicit RNG)
- Easy serialization to/from JSON configs
- Extensibility via subclassing or composition

Typical import:

    from financial_simulator.core import (
        SimulationEngine,
        ComposedEventBuilder,
        IntervalTiming,
        FixedValue,
        NormalDistribution,
        create_distribution,
    )
"""

from .distributions import (
    Distribution,
    NormalDistribution,
    UniformDistribution,
    TriangularDistribution,
    LogNormalDistribution,
    ExponentialDistribution,
    ConstantDistribution,
    BetaDistribution,
    AnyDistribution,
    create_distribution,
)

from .event import (
    Event,
    Timing,
    OneTimeTiming,
    IntervalTiming,
    RandomTiming,
    SeasonalTiming,
    ValueGenerator,
    FixedValue,
    GrowingValue,
    DistributionValue,
    RateChangeValue,
    VariableRateLoanValue,
    DividendValue,
    InvestmentContributionValue,
    TaxEventValue,
    EventBuilder,
    ComposedEventBuilder,
    create_timing,
    create_value_generator,
    create_event_builder,
)

from .simulation import (
    ContinuousProcess,
    AppreciationProcess,
    GBMContinuousProcess,
    MeanRevertingContinuousProcess,
    SimulationResult,
    SimulationEngine,
)

from .stochastic import (
    StochasticProcess,
    GeometricBrownianMotion,
    MeanRevertingProcess,
)

from .financial_models import (
    Loan,
    TaxBracket,
    TaxSchedule,
    Portfolio,
)

__all__ = [
    # distributions
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
    # event system
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
    # simulation
    "ContinuousProcess",
    "AppreciationProcess",
    "GBMContinuousProcess",
    "MeanRevertingContinuousProcess",
    "SimulationResult",
    "SimulationEngine",
    # stochastic processes
    "StochasticProcess",
    "GeometricBrownianMotion",
    "MeanRevertingProcess",
    # financial domain models
    "Loan",
    "TaxBracket",
    "TaxSchedule",
    "Portfolio",
]

__version__ = "0.2.0"  # core rewrite
