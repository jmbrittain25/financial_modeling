import datetime as dt
import json
import pickle
from dataclasses import dataclass, field
from typing import List, Dict

from ..event.event import Event
from ..event.event_builder import EventBuilder, FixedValueGenerator, GrowingValueGenerator, VariableRateLoanBuilder, TriggeredEventBuilder, SeasonalEventBuilder


@dataclass
class Simulation:
    name: str
    start: dt.datetime
    end: dt.datetime
    appreciation_rate: float = 0.04
    event_builders: List[EventBuilder] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    state_history: Dict[dt.datetime, Dict[str, float]] = field(default_factory=dict)

    def add_builder(self, builder: EventBuilder):
        self.event_builders.append(builder)

    def run(self):
        all_events = []
        for builder in self.event_builders:
            all_events.extend(builder.build(self.start, self.end))
        self.events = sorted(all_events, key=lambda e: e.time)

        # Process events and update state: cumulative cash, property value (appreciating)
        cumulative_cash = 0.0
        property_value = 0.0  # Set via metadata or param
        last_time = self.start
        monthly_appreciation = self.appreciation_rate / 12
        for event in self.events:
            if event.time != last_time:
                delta_months = (event.time - last_time).days / 30
                property_value *= (1 + monthly_appreciation) ** delta_months
                self.state_history[last_time] = {'cumulative_cash': cumulative_cash, 'property_value': property_value}
            cumulative_cash += event.value
            # If purchase event, set initial property_value
            if 'type' in event.metadata and event.metadata['type'] == 'purchase':
                property_value = -event.value  # Absolute value
            last_time = event.time
        self.state_history[last_time] = {'cumulative_cash': cumulative_cash, 'property_value': property_value}

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
            'appreciation_rate': self.appreciation_rate,
            'event_builders': [b.to_dict() for b in self.event_builders],
            'events': [e.to_dict() for e in self.events],
            'state_history': {t.isoformat(): v for t, v in self.state_history.items()}
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Simulation':
        sim = cls(
            d['name'],
            dt.datetime.fromisoformat(d['start']),
            dt.datetime.fromisoformat(d['end']),
            d['appreciation_rate']
        )
        builder_types = {
            'FixedValueGenerator': FixedValueGenerator.from_dict,
            'GrowingValueGenerator': GrowingValueGenerator.from_dict,
            'VariableRateLoanBuilder': VariableRateLoanBuilder.from_dict,
            'TriggeredEventBuilder': TriggeredEventBuilder.from_dict,
            'SeasonalEventBuilder': SeasonalEventBuilder.from_dict
        }
        for b_dict in d['event_builders']:
            builder_cls = builder_types.get(b_dict['type'])
            if builder_cls:
                sim.add_builder(builder_cls(b_dict))
        sim.events = [Event.from_dict(e) for e in d['events']]
        sim.state_history = {dt.datetime.fromisoformat(t): v for t, v in d['state_history'].items()}
        return sim

    def save_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, filepath: str) -> 'Simulation':
        with open(filepath, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)

    def save_pickle(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filepath: str) -> 'Simulation':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
