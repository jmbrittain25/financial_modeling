import datetime as dt
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from .event import Event
from ..utils import Distribution, create_distribution


class EventBuilder(ABC):
    """Abstract base for generating lists of Events."""
    @abstractmethod
    def build(self, start: dt.datetime, end: dt.datetime) -> List[Event]:
        pass

    def to_dict(self) -> Dict:
        return {'type': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d: Dict) -> 'EventBuilder':
        raise NotImplementedError("Subclasses must implement from_dict.")

class FixedValueGenerator(EventBuilder):
    def __init__(self, value: float, interval: dt.timedelta, metadata: Optional[Dict] = None, start_time: Optional[dt.datetime] = None):
        self.value = value
        self.interval = interval
        self.metadata = metadata or {}
        self.start_time = start_time

    def build(self, start: dt.datetime, end: dt.datetime) -> List[Event]:
        events = []
        current = self.start_time if self.start_time is not None else start
        while current <= end:
            events.append(Event(current, self.value, self.metadata.copy()))
            current += self.interval
        return events

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({'value': self.value, 'interval_days': self.interval.days, 'metadata': self.metadata,
                  'start_time': self.start_time.isoformat() if self.start_time else None})
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'FixedValueGenerator':
        start_time = dt.datetime.fromisoformat(d['start_time']) if d.get('start_time') else None
        return cls(d['value'], dt.timedelta(days=d['interval_days']), d['metadata'], start_time)

class GrowingValueGenerator(EventBuilder):
    def __init__(self, initial_value: float, growth_rate: float, interval: dt.timedelta, metadata: Optional[Dict] = None, start_time: Optional[dt.datetime] = None):
        self.initial_value = initial_value
        self.growth_rate = growth_rate
        self.interval = interval
        self.metadata = metadata or {}
        self.start_time = start_time

    def build(self, start: dt.datetime, end: dt.datetime) -> List[Event]:
        events = []
        current = self.start_time if self.start_time is not None else start
        value = self.initial_value
        while current <= end:
            events.append(Event(current, value, self.metadata.copy()))
            value *= (1 + self.growth_rate)
            current += self.interval
        return events

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({'initial_value': self.initial_value, 'growth_rate': self.growth_rate, 'interval_days': self.interval.days,
                  'metadata': self.metadata, 'start_time': self.start_time.isoformat() if self.start_time else None})
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'GrowingValueGenerator':
        start_time = dt.datetime.fromisoformat(d['start_time']) if d.get('start_time') else None
        return cls(d['initial_value'], d['growth_rate'], dt.timedelta(days=d['interval_days']), d['metadata'], start_time)

class VariableRateLoanBuilder(EventBuilder):
    """Generates loan payments with variable interest rate changing over time."""
    def __init__(self, principal: float, initial_rate: float, term_months: int, start_date: dt.datetime,
                 rate_distribution: Distribution, rate_change_interval: dt.timedelta,
                 metadata: Optional[Dict] = None):
        self.principal = principal
        self.initial_rate = initial_rate
        self.term_months = term_months
        self.start_date = start_date
        self.rate_distribution = rate_distribution
        self.rate_change_interval = rate_change_interval
        self.metadata = metadata or {}
        self.monthly_interval = dt.timedelta(days=30)

    def build(self, start: dt.datetime, end: dt.datetime) -> List[Event]:
        events = []
        current = max(self.start_date, start)
        balance = self.principal
        current_rate = self.initial_rate
        last_rate_change = current
        month = 0
        while current <= end and month < self.term_months and balance > 0:
            if current - last_rate_change >= self.rate_change_interval:
                current_rate = self.rate_distribution.sample()
                last_rate_change = current
            monthly_rate = current_rate / 12
            # Recalculate payment based on current rate and remaining term
            remaining_months = self.term_months - month
            if monthly_rate == 0:
                payment = balance / remaining_months
            else:
                payment = balance * (monthly_rate * (1 + monthly_rate)**remaining_months) / ((1 + monthly_rate)**remaining_months - 1)
            interest = balance * monthly_rate
            principal_pay = min(payment - interest, balance)
            balance -= principal_pay
            meta = self.metadata.copy()
            meta.update({'interest': interest, 'principal': principal_pay, 'rate': current_rate, 'remaining_balance': balance})
            events.append(Event(current, - (interest + principal_pay), meta))
            current += self.monthly_interval
            month += 1
        return events

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({'principal': self.principal, 'initial_rate': self.initial_rate, 'term_months': self.term_months,
                 'start_date': self.start_date.isoformat(), 'rate_change_interval_days': self.rate_change_interval.days,
                 'metadata': self.metadata, 'rate_distribution': self.rate_distribution.to_dict()})
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'VariableRateLoanBuilder':
        start_date = dt.datetime.fromisoformat(d['start_date'])
        rate_dist = create_distribution(d['rate_distribution'])
        return cls(d['principal'], d['initial_rate'], d['term_months'], start_date, rate_dist,
                   dt.timedelta(days=d['rate_change_interval_days']), d['metadata'])

class TriggeredEventBuilder(EventBuilder):
    """Generates a one-time event at/after a trigger time, e.g., renovation after mom leaves."""
    def __init__(self, trigger_time: dt.datetime, value: float, delay: dt.timedelta = dt.timedelta(days=0),
                 metadata: Optional[Dict] = None):
        self.trigger_time = trigger_time
        self.value = value
        self.delay = delay
        self.metadata = metadata or {}

    def build(self, start: dt.datetime, end: dt.datetime) -> List[Event]:
        event_time = max(start, self.trigger_time + self.delay)
        if event_time <= end:
            return [Event(event_time, self.value, self.metadata.copy())]
        return []

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({'trigger_time': self.trigger_time.isoformat(), 'value': self.value, 'delay_days': self.delay.days, 'metadata': self.metadata})
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'TriggeredEventBuilder':
        trigger_time = dt.datetime.fromisoformat(d['trigger_time'])
        return cls(trigger_time, d['value'], dt.timedelta(days=d['delay_days']), d['metadata'])

class SeasonalEventBuilder(EventBuilder):
    """Generates events only in specific months, e.g., lawn mowing in summer."""
    def __init__(self, value: float, interval: dt.timedelta, months: List[int], metadata: Optional[Dict] = None):
        self.value = value
        self.interval = interval
        self.months = months  # e.g., [6,7,8] for June-Aug
        self.metadata = metadata or {}

    def build(self, start: dt.datetime, end: dt.datetime) -> List[Event]:
        events = []
        current = start
        while current <= end:
            if current.month in self.months:
                events.append(Event(current, self.value, self.metadata.copy()))
            current += self.interval
        return events

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({'value': self.value, 'interval_days': self.interval.days, 'months': self.months, 'metadata': self.metadata})
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'SeasonalEventBuilder':
        return cls(d['value'], dt.timedelta(days=d['interval_days']), d['months'], d['metadata'])
