import datetime as dt
from abc import ABC, abstractmethod
from typing import Dict


class ContinuousProcess(ABC):
    @abstractmethod
    def advance(self, state: Dict[str, float], delta: dt.timedelta):
        pass


class AppreciationProcess(ContinuousProcess):
    def __init__(self, rate: float, var: str = 'property_value'):
        self.rate = rate
        self.var = var

    def advance(self, state: Dict[str, float], delta: dt.timedelta):
        delta_years = delta.days / 365.25
        if self.var in state:
            state[self.var] *= (1 + self.rate) ** delta_years


def create_continuous_process(d: Dict) -> ContinuousProcess:
    typ = d['type']
    if typ == 'Appreciation':
        return AppreciationProcess(d['rate'], d.get('var', 'property_value'))
    raise ValueError(f"Unknown continuous process type: {typ}")