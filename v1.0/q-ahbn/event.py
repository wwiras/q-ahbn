from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True, slots=True)
class Event:
    time: float
    priority: int
    event_type: str = field(compare=False)
    payload: Any = field(compare=False, default=None)