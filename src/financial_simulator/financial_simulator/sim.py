import datetime as dt
import json
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .event import Event, EventBuilder
from .continuous_process import ContinuousProcess


@dataclass
class Simulation:
    name: str
    start: dt.datetime
    end: dt.datetime
    params: Dict[str, any] = field(default_factory=dict)
    event_builders: List[EventBuilder] = field(default_factory=list)
    continuous_processes: List[ContinuousProcess] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    state: Dict[str, float] = field(default_factory=dict)
    state_history: Dict[dt.datetime, Dict[str, float]] = field(default_factory=dict)

    def add_builder(self, builder: EventBuilder):
        self.event_builders.append(builder)

    def add_continuous(self, proc: ContinuousProcess):
        self.continuous_processes.append(proc)

    def run(self):
        for b in self.event_builders:
            b.reset()
        current = self.start
        self.state_history[current] = self.state.copy()
        while True:
            # Pre-compute next times
            builder_next_times = [(b, b.next_event_time(current, self)) for b in self.event_builders]
            valid_next_times = [t for b, t in builder_next_times if t is not None and t <= self.end]
            if not valid_next_times:
                break
            next_time = min(valid_next_times)

            # Advance continuous processes
            delta = next_time - current
            for proc in self.continuous_processes:
                proc.advance(self.state, delta)

            # Generate events only for builders whose next time matches the global next_time
            events_at_time = []
            for b, nt in builder_next_times:
                if nt == next_time:
                    event = b.generate_event(next_time, self)
                    if event:
                        events_at_time.append(event)

            self.events.extend(events_at_time)

            # Update cumulative cash if tracked
            if 'cumulative_cash' in self.state:
                self.state['cumulative_cash'] += sum(e.value for e in events_at_time)

            self.state_history[next_time] = self.state.copy()
            current = next_time

    def to_dict(self) -> Dict:
        params_serialized = {k: v.isoformat() if isinstance(v, dt.datetime) else v for k, v in self.params.items()}
        return {
            'name': self.name,
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
            'params': params_serialized,
            # Serialize builders, processes if needed
            'events': [e.to_dict() for e in self.events],
            'state': self.state,
            'state_history': {t.isoformat(): v for t, v in self.state_history.items()}
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Simulation':
        params = {k: dt.datetime.fromisoformat(v) if isinstance(v, str) and '-' in v else v for k, v in d['params'].items()}
        sim = cls(d['name'], dt.datetime.fromisoformat(d['start']), dt.datetime.fromisoformat(d['end']), params)
        sim.events = [Event.from_dict(e) for e in d['events']]
        sim.state = d['state']
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