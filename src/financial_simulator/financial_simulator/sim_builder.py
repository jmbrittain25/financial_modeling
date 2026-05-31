from typing import List, Dict, Callable, Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .sim import Simulation
from .utils import Distribution


def _build_one(factory: Callable[[Dict[str, Any]], Simulation], param_distributions: Dict[str, Distribution], base_seed: Optional[int], i: int) -> Simulation:
    if base_seed is not None:
        np.random.seed(base_seed + i)
    params = {k: v.sample() for k, v in param_distributions.items()}
    sim = factory(params)
    sim.name = f"Sim_{i}"
    sim.run()
    return sim

class SimulationBuilder:
    def __init__(self, factory: Callable[[Dict[str, Any]], Simulation], param_distributions: Dict[str, Distribution]):
        self.factory = factory
        self.param_distributions = param_distributions

    def build_simulations(self, num: int, seed: Optional[int] = None) -> List[Simulation]:
        build_func = partial(_build_one, self.factory, self.param_distributions, seed)
        with ProcessPoolExecutor() as executor:
            sims = list(executor.map(build_func, range(num)))
        return sims