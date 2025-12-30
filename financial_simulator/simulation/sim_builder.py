from typing import List, Dict, Callable, Any

from .sim import Simulation
from ..utils import Distribution, DateDistribution


class SimulationBuilder:
    def __init__(self, factory: Callable[[Dict[str, Any]], Simulation], param_distributions: Dict[str, Distribution]):
        self.factory = factory
        self.param_distributions = param_distributions

    def build_simulations(self, num: int) -> List[Simulation]:
        sims = []
        for i in range(num):
            params = {k: v.sample() for k, v in self.param_distributions.items() if not isinstance(v, DateDistribution)}
            # Handle date samples separately
            for k, v in self.param_distributions.items():
                if isinstance(v, DateDistribution):
                    params[k] = v.sample()
            sim = self.factory(params)
            sim.name = f"Sim_{i}"
            sim.run()
            sims.append(sim)
        return sims
