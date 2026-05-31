"""
Monte Carlo simulation runner.

Runs many independent simulations efficiently and collects results.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, List, Optional

import numpy as np

from ..core import SimulationEngine, SimulationResult


class MonteCarloRunner:
    """
    Runs multiple simulations (Monte Carlo) and returns aggregated results.
    """

    def __init__(self, n_jobs: int = 4):
        self.n_jobs = n_jobs

    def run(
        self,
        n_sims: int,
        factory: Callable[[int], SimulationEngine],
        base_seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        """
        Run n_sims simulations.

        Args:
            n_sims: Number of simulations to run.
            factory: Callable that takes a simulation index and returns a configured SimulationEngine.
            base_seed: If provided, seeds will be base_seed + i for reproducibility.
        """
        results: List[SimulationResult] = []

        def _run_one(i: int) -> SimulationResult:
            seed = base_seed + i if base_seed is not None else None
            engine = factory(i)
            if seed is not None:
                engine.seed = seed  # override
            engine.run()
            return engine.get_result()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(_run_one, i) for i in range(n_sims)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results
