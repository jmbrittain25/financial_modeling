"""
financial_simulator

A professional-grade financial simulation platform with Monte Carlo capabilities,
rich quantitative analytics, and support for future autonomous AI agents.
"""

from .core import (
    AnyContinuousProcess,
    BetaDistribution,
    Event,
    GeometricBrownianMotion,
    Loan,
    MeanRevertingProcess,
    Portfolio,
    SimulationEngine,
    SimulationResult,
    TaxSchedule,
    create_event_builder,
)

# Analytics (Phase 2)
try:
    from .analytics.risk import MonteCarloAnalyzer, RiskAnalyzer, RiskReport
except ImportError:
    RiskAnalyzer = MonteCarloAnalyzer = RiskReport = None  # type: ignore

__version__ = "0.3.0"

__all__ = [
    "SimulationEngine",
    "SimulationResult",
    "Event",
    "create_event_builder",
    "BetaDistribution",
    "GeometricBrownianMotion",
    "MeanRevertingProcess",
    "AnyContinuousProcess",
    "Loan",
    "TaxSchedule",
    "Portfolio",
    "RiskAnalyzer",
    "MonteCarloAnalyzer",
    "RiskReport",
]
