import datetime as dt
from abc import ABC, abstractmethod
from typing import Tuple, Dict

from .utils import Distribution


class ValueGenerator(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_value(self, time: dt.datetime, sim: 'Simulation') -> Tuple[float, Dict]:
        pass


class FixedValue(ValueGenerator):
    def __init__(self, value: float):
        self.value = value

    def reset(self):
        pass

    def get_value(self, time: dt.datetime, sim: 'Simulation') -> Tuple[float, Dict]:
        return self.value, {}


class GrowingValue(ValueGenerator):
    def __init__(self, initial: float, growth_rate: float):
        self.initial = initial
        self.growth_rate = growth_rate
        self.current = initial
        self.last_time = None

    def reset(self):
        self.current = self.initial
        self.last_time = None

    def get_value(self, time: dt.datetime, sim: 'Simulation') -> Tuple[float, Dict]:
        if self.last_time is not None:
            delta_years = (time - self.last_time).days / 365.25
            self.current *= (1 + self.growth_rate) ** delta_years
        self.last_time = time
        return self.current, {}


class DistributionValue(ValueGenerator):
    def __init__(self, dist: Distribution):
        self.dist = dist

    def reset(self):
        pass

    def get_value(self, time: dt.datetime, sim: 'Simulation') -> Tuple[float, Dict]:
        return self.dist.sample(), {}


class RateChangeValue(ValueGenerator):
    def __init__(self, dist: Distribution, update_key: str):
        self.dist = dist
        self.update_key = update_key

    def reset(self):
        pass

    def get_value(self, time: dt.datetime, sim: 'Simulation') -> Tuple[float, Dict]:
        new_rate = self.dist.sample()
        return 0.0, {'update_state': {self.update_key: new_rate}}


class VariableRateLoanValue(ValueGenerator):
    def __init__(self, principal: float, initial_rate: float, term_months: int, rate_key: str):
        self.principal = principal
        self.initial_rate = initial_rate
        self.term_months = term_months
        self.rate_key = rate_key
        self.balance = principal
        self.month = 0
        self.last_time = None

    def reset(self):
        self.balance = self.principal
        self.month = 0
        self.last_time = None

    def get_value(self, time: dt.datetime, sim: 'Simulation') -> Tuple[float, Dict]:
        if self.month >= self.term_months or self.balance <= 0:
            return 0.0, {}
        if self.last_time is not None:
            # Check if time is next interval, but assume called correctly
            pass
        current_rate = sim.state.get(self.rate_key, self.initial_rate)
        monthly_rate = current_rate / 12
        remaining_months = self.term_months - self.month
        if monthly_rate == 0:
            payment = self.balance / remaining_months
        else:
            r = (1 + monthly_rate) ** remaining_months
            payment = self.balance * monthly_rate * r / (r - 1)
        interest = self.balance * monthly_rate
        principal_pay = payment - interest
        if principal_pay > self.balance:
            principal_pay = self.balance
            payment = interest + principal_pay
        self.balance -= principal_pay
        self.month += 1
        self.last_time = time
        extra_meta = {
            'interest': interest,
            'principal': principal_pay,
            'rate': current_rate,
            'remaining_balance': self.balance
        }
        return -payment, extra_meta