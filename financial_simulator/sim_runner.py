from typing import List

from .sim import Simulation
from .sim_builder import SimulationBuilder


class SimulationRunner:
    def __init__(self, builder: SimulationBuilder):
        self.builder = builder

    def run(self, num_simulations: int) -> List[Simulation]:
        return self.builder.build_simulations(num_simulations)