from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import datetime


class Distribution(ABC):
    """Abstract base for parameter distributions in Monte Carlo."""
    @abstractmethod
    def sample(self) -> float:
        pass

@dataclass
class NormalDistribution(Distribution):
    mean: float
    std: float

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)

@dataclass
class UniformDistribution(Distribution):
    low: float
    high: float

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

@dataclass
class TriangularDistribution(Distribution):
    """For costs with low/mode/high estimates."""
    low: float
    mode: float
    high: float

    def sample(self) -> float:
        return np.random.triangular(self.low, self.mode, self.high)

class DateDistribution(Distribution):
    """Samples dates between start and end."""
    def __init__(self, start: datetime.datetime, end: datetime.datetime):
        self.start = start
        self.end = end

    def sample(self) -> datetime.datetime:
        delta = (self.end - self.start).days
        random_days = np.random.randint(0, delta + 1)
        return self.start + datetime.timedelta(days=random_days)
