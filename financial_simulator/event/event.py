import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class Event:
    """Represents a financial event (expense negative, payment positive) at a specific time."""
    time: datetime.datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['time'] = self.time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Event':
        data = d.copy()
        data['time'] = datetime.datetime.fromisoformat(data['time'])
        return cls(**data)
