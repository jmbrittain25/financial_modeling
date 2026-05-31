"""
financial_simulator

A professional-grade financial simulation platform with Monte Carlo capabilities,
rich quantitative analytics, and support for future autonomous AI agents.
"""

from .core import (
    SimulationEngine,
    SimulationResult,
    Event,
    create_event_builder,
    BetaDistribution,
    GeometricBrownianMotion,
    MeanRevertingProcess,
    Loan,
    TaxSchedule,
    Portfolio,
)

# Analytics (Phase 2)
try:
    from .analytics.risk import RiskAnalyzer, MonteCarloAnalyzer, RiskReport
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
    "Loan",
    "TaxSchedule",
    "Portfolio",
    "RiskAnalyzer",
    "MonteCarloAnalyzer",
    "RiskReport",
]
