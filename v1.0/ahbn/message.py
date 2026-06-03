from dataclasses import dataclass


@dataclass(slots=True)
class Message:
    message_id: str
    source_id: int
    created_at: float