from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import datetime as dt

from .event import Event
from .time_generator import Timing, OneTimeTiming, IntervalTiming, RandomTiming, SeasonalTiming
from .value_generator import ValueGenerator, FixedValue, GrowingValue, DistributionValue, RateChangeValue, VariableRateLoanValue
from .utils import create_distribution
from .sim import Simulation

@dataclass
class Event:
    """Represents a financial event (expense negative, payment positive) at a specific time."""
    time: dt.datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['time'] = self.time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Event':
        data = d.copy()
        data['time'] = dt.datetime.fromisoformat(data['time'])
        return cls(**data)


class EventBuilder(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def next_event_time(self, current: dt.datetime, sim: Simulation) -> Optional[dt.datetime]:
        pass

    @abstractmethod
    def generate_event(self, time: dt.datetime, sim: Simulation) -> Optional[Event]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @classmethod
    def from_dict(cls, d: Dict) -> 'EventBuilder':
        raise NotImplementedError


class ComposedEventBuilder(EventBuilder):
    def __init__(self, timing: Timing, value_gen: ValueGenerator, metadata: Dict = None, name: Optional[str] = None):
        self.timing = timing
        self.value_gen = value_gen
        self.metadata = metadata or {}
        self.name = name
        self.current_next = None

    def reset(self):
        self.timing.reset()
        self.value_gen.reset()
        self.current_next = None

    def next_event_time(self, current: dt.datetime, sim: Simulation) -> Optional[dt.datetime]:
        if self.current_next is None or self.current_next <= current:
            nt = self.timing.next_time(current, sim.end)
            while nt is not None and nt <= current:
                nt = self.timing.next_time(nt, sim.end)
            self.current_next = nt
        return self.current_next

    def generate_event(self, time: dt.datetime, sim: Simulation) -> Optional[Event]:
        if self.next_event_time(time, sim) != time:
            return None
        cash_value, extra_meta = self.value_gen.get_value(time, sim)
        meta = {**self.metadata, **extra_meta}
        event = Event(time, cash_value, meta)
        if 'update_state' in extra_meta:
            sim.state.update(extra_meta['update_state'])
        self.current_next = None  # Force recalc next
        return event

    def to_dict(self) -> Dict:
        return {
            'type': 'ComposedEventBuilder',
            'timing': self.timing.__dict__,  # Simplify, implement proper serialization if needed
            'value_gen': self.value_gen.__dict__,
            'metadata': self.metadata,
            'name': self.name
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ComposedEventBuilder':
        # Implement deserialization if needed, but for now assume not
        raise NotImplementedError


def create_timing(d: Dict) -> Timing:
    typ = d['type']
    if typ == 'OneTime':
        time = dt.datetime.fromisoformat(d['time'])
        return OneTimeTiming(time)
    elif typ == 'Interval':
        interval = dt.timedelta(days=d['interval_days'])
        start_time = dt.datetime.fromisoformat(d['start_time']) if 'start_time' in d else None
        return IntervalTiming(interval, start_time)
    elif typ == 'Random':
        start = dt.datetime.fromisoformat(d['start'])
        end = dt.datetime.fromisoformat(d['end'])
        n = d['n']
        dist = d.get('distribution', 'uniform')
        return RandomTiming(start, end, n, dist)
    elif typ == 'Seasonal':
        months = d['months']
        inner = create_timing(d['inner'])
        return SeasonalTiming(inner, months)
    raise ValueError(f"Unknown timing type: {typ}")


def create_value_generator(d: Dict) -> ValueGenerator:
    typ = d['type']
    if typ == 'Fixed':
        return FixedValue(d['value'])
    elif typ == 'Growing':
        return GrowingValue(d['initial'], d['growth_rate'])
    elif typ == 'Distribution':
        dist = create_distribution(d['dist'])
        return DistributionValue(dist)
    elif typ == 'RateChange':
        dist = create_distribution(d['dist'])
        update_key = d['update_key']
        return RateChangeValue(dist, update_key)
    elif typ == 'VariableRateLoan':
        return VariableRateLoanValue(d['principal'], d['initial_rate'], d['term_months'], d['rate_key'])
    raise ValueError(f"Unknown value generator type: {typ}")


def create_event_builder(d: Dict) -> EventBuilder:
    timing = create_timing(d['timing'])
    value_gen = create_value_generator(d['value_gen'])
    metadata = d.get('metadata', {})
    name = d.get('name')
    return ComposedEventBuilder(timing, value_gen, metadata, name)