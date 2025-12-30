from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import datetime as dt
from typing import Dict


class Distribution(ABC):
    """Abstract base for parameter distributions in Monte Carlo."""
    @abstractmethod
    def sample(self) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

@dataclass
class NormalDistribution(Distribution):
    mean: float
    std: float

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)

    def to_dict(self) -> Dict:
        return {'type': 'NormalDistribution', 'mean': self.mean, 'std': self.std}

    @classmethod
    def from_dict(cls, d: Dict) -> 'NormalDistribution':
        return cls(d['mean'], d['std'])

@dataclass
class UniformDistribution(Distribution):
    low: float
    high: float

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

    def to_dict(self) -> Dict:
        return {'type': 'UniformDistribution', 'low': self.low, 'high': self.high}

    @classmethod
    def from_dict(cls, d: Dict) -> 'UniformDistribution':
        return cls(d['low'], d['high'])

@dataclass
class TriangularDistribution(Distribution):
    """For costs with low/mode/high estimates."""
    low: float
    mode: float
    high: float

    def sample(self) -> float:
        return np.random.triangular(self.low, self.mode, self.high)

    def to_dict(self) -> Dict:
        return {'type': 'TriangularDistribution', 'low': self.low, 'mode': self.mode, 'high': self.high}

    @classmethod
    def from_dict(cls, d: Dict) -> 'TriangularDistribution':
        return cls(d['low'], d['mode'], d['high'])

class DateDistribution(Distribution):
    """Samples dates between start and end."""
    def __init__(self, start: dt.datetime, end: dt.datetime):
        self.start = start
        self.end = end

    def sample(self) -> dt.datetime:
        delta = (self.end - self.start).days
        random_days = np.random.randint(0, delta + 1)
        return self.start + dt.timedelta(days=random_days)

    def to_dict(self) -> Dict:
        return {'type': 'DateDistribution', 'start': self.start.isoformat(), 'end': self.end.isoformat()}

    @classmethod
    def from_dict(cls, d: Dict) -> 'DateDistribution':
        return cls(dt.datetime.fromisoformat(d['start']), dt.datetime.fromisoformat(d['end']))

def create_distribution(d: Dict) -> Distribution:
    typ = d['type']
    if typ == 'NormalDistribution':
        return NormalDistribution.from_dict(d)
    elif typ == 'UniformDistribution':
        return UniformDistribution.from_dict(d)
    elif typ == 'TriangularDistribution':
        return TriangularDistribution.from_dict(d)
    elif typ == 'DateDistribution':
        return DateDistribution.from_dict(d)
    raise ValueError(f"Unknown distribution type: {typ}")
