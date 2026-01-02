import datetime as dt
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Timing(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def next_time(self, current: dt.datetime, end: dt.datetime) -> Optional[dt.datetime]:
        pass


class OneTimeTiming(Timing):
    def __init__(self, time: dt.datetime):
        self.time = time
        self.fired = False

    def reset(self):
        self.fired = False

    def next_time(self, current: dt.datetime, end: dt.datetime) -> Optional[dt.datetime]:
        if not self.fired and current <= self.time <= end:
            return self.time
        return None


class IntervalTiming(Timing):
    def __init__(self, interval: dt.timedelta, start_time: Optional[dt.datetime] = None):
        self.interval = interval
        self.start_time = start_time
        self.current_next = None

    def reset(self):
        self.current_next = self.start_time

    def next_time(self, current: dt.datetime, end: dt.datetime) -> Optional[dt.datetime]:
        if self.current_next is None:
            self.current_next = current + self.interval if self.start_time is None else self.start_time
        while self.current_next < current:
            self.current_next += self.interval
        if self.current_next > end:
            return None
        return self.current_next

    # Advance is handled in builder after generate


class RandomTiming(Timing):
    def __init__(self, start: dt.datetime, end: dt.datetime, n: int, distribution: str = 'uniform'):
        self.start = start
        self.end = end
        self.n = n
        self.distribution = distribution
        self.times = []
        self.index = 0

    def reset(self):
        delta_days = (self.end - self.start).days
        if self.distribution == 'uniform':
            random_days = sorted(np.random.randint(0, delta_days + 1, self.n))
            self.times = [self.start + dt.timedelta(days=d) for d in random_days]
        # Add other distributions if needed
        self.index = 0

    def next_time(self, current: dt.datetime, end: dt.datetime) -> Optional[dt.datetime]:
        while self.index < len(self.times) and self.times[self.index] < current:
            self.index += 1
        if self.index < len(self.times) and self.times[self.index] <= end:
            return self.times[self.index]
        return None


class SeasonalTiming(Timing):
    def __init__(self, inner: Timing, months: list[int]):
        self.inner = inner
        self.months = set(months)

    def reset(self):
        self.inner.reset()

    def next_time(self, current: dt.datetime, end: dt.datetime) -> Optional[dt.datetime]:
        nt = self.inner.next_time(current, end)
        while nt is not None and nt.month not in self.months:
            nt = self.inner.next_time(nt, end)
        return nt