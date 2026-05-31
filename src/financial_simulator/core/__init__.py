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
    AnyDistribution,
    BetaDistribution,
    ConstantDistribution,
    Distribution,
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    TriangularDistribution,
    UniformDistribution,
    create_distribution,
)
from .event import (
    ComposedEventBuilder,
    DistributionValue,
    DividendValue,
    Event,
    EventBuilder,
    FixedValue,
    GrowingValue,
    IntervalTiming,
    InvestmentContributionValue,
    OneTimeTiming,
    RandomTiming,
    RateChangeValue,
    SeasonalTiming,
    TaxEventValue,
    Timing,
    ValueGenerator,
    VariableRateLoanValue,
    create_event_builder,
    create_timing,
    create_value_generator,
)
from .financial_models import (
    Loan,
    Portfolio,
    TaxBracket,
    TaxSchedule,
)
from .simulation import (
    AppreciationProcess,
    ContinuousProcess,
    GBMContinuousProcess,
    MeanRevertingContinuousProcess,
    SimulationEngine,
    SimulationResult,
)
from .stochastic import (
    GeometricBrownianMotion,
    MeanRevertingProcess,
    StochasticProcess,
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
