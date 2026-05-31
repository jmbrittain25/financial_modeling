"""
Professional-grade risk and performance analytics for financial simulations.

Designed to work with lists of SimulationResult objects or raw outcome arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..core.simulation import SimulationResult


@dataclass
class RiskMetrics:
    """Container for common risk metrics."""

    var_95: float
    cvar_95: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    probability_of_ruin: Optional[float] = None
    mean_return: float = 0.0
    std_dev: float = 0.0


class RiskReport(BaseModel):
    """Structured, serializable risk report."""

    model_config = ConfigDict(extra="forbid")

    n_simulations: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskAnalyzer:
    """
    Computes risk and performance metrics across many simulation outcomes.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    def compute_var(self, outcomes: np.ndarray, confidence: float = 0.95) -> float:
        """Historical Value at Risk. Returns nan for empty input."""
        if len(outcomes) == 0:
            return float("nan")
        return float(np.percentile(outcomes, (1 - confidence) * 100))

    def compute_cvar(self, outcomes: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall). Returns nan for empty input."""
        if len(outcomes) == 0:
            return float("nan")
        var = self.compute_var(outcomes, confidence)
        tail = outcomes[outcomes <= var]
        return float(np.mean(tail)) if len(tail) > 0 else var

    def compute_sharpe(self, returns: np.ndarray) -> float:
        excess = returns - self.risk_free_rate
        std = np.std(excess)
        if std < 1e-12:  # treat as zero to avoid huge/inf values
            return 0.0
        return float(np.mean(excess) / std)

    def compute_sortino(self, returns: np.ndarray, target: float = 0.0) -> float:
        excess = returns - target
        downside = excess[excess < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0.0
        return float(np.mean(excess) / downside_std) if downside_std > 0 else 0.0

    def max_drawdown(self, path: np.ndarray) -> float:
        """Maximum drawdown for a single wealth path."""
        peak = np.maximum.accumulate(path)
        drawdown = (peak - path) / peak
        return float(np.max(drawdown))

    def probability_of_ruin(self, outcomes: np.ndarray, threshold: float = 0.0) -> float:
        return float(np.mean(outcomes <= threshold))

    def analyze_outcomes(
        self,
        outcomes: np.ndarray,
        returns: Optional[np.ndarray] = None,
        paths: Optional[List[np.ndarray]] = None,
    ) -> RiskReport:
        """Produce a comprehensive risk report from final outcomes (and optionally returns/paths)."""
        metrics: Dict[str, float] = {
            "var_95": self.compute_var(outcomes, 0.95),
            "cvar_95": self.compute_cvar(outcomes, 0.95),
            "mean": float(np.mean(outcomes)),
            "std": float(np.std(outcomes)),
            "median": float(np.median(outcomes)),
            "p5": float(np.percentile(outcomes, 5)),
            "p95": float(np.percentile(outcomes, 95)),
        }

        if returns is not None and len(returns) > 0:
            metrics["sharpe"] = self.compute_sharpe(returns)
            metrics["sortino"] = self.compute_sortino(returns)

        if paths:
            mdds = [self.max_drawdown(p) for p in paths if len(p) > 1]
            if mdds:
                metrics["avg_max_drawdown"] = float(np.mean(mdds))
                metrics["worst_max_drawdown"] = float(np.max(mdds))

        metrics["prob_ruin"] = self.probability_of_ruin(outcomes, threshold=0.0)

        return RiskReport(n_simulations=len(outcomes), metrics=metrics)


class MonteCarloAnalyzer:
    """
    Higher-level analyzer that works directly with lists of SimulationResult.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk = RiskAnalyzer(risk_free_rate=risk_free_rate)

    def analyze_results(self, results: List[SimulationResult]) -> RiskReport:
        """Analyze a collection of completed simulations."""
        if not results:
            raise ValueError("No simulation results provided")

        final_values = np.array([r.final_state.get("cumulative_cash", 0.0) for r in results])

        # Simple returns approximation (can be made more sophisticated later)
        returns = None
        if len(final_values) > 1:
            # Use log returns of final wealth as proxy
            returns = np.diff(np.log(np.maximum(final_values, 1e-9)))

        return self.risk.analyze_outcomes(final_values, returns=returns)
