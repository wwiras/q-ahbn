from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ahbn.control import NodeControlState


@dataclass
class NodeStats:
    received_new: int = 0
    received_duplicate: int = 0
    forwarded: int = 0
    dropped: int = 0
    first_receive_time: Dict[str, float] = field(default_factory=dict)
    last_receive_time: Dict[str, float] = field(default_factory=dict)

    @property
    def total_received(self) -> int:
        return self.received_new + self.received_duplicate

    @property
    def duplicate_ratio_raw(self) -> float:
        total = self.total_received
        if total <= 0:
            return 0.0
        return self.received_duplicate / total


@dataclass
class Node:
    node_id: int
    neighbors: List[int] = field(default_factory=list)

    cluster_id: Optional[int] = None
    is_cluster_head: bool = False
    gateway_neighbors: List[int] = field(default_factory=list)

    seen_messages: Set[str] = field(default_factory=set)
    stats: NodeStats = field(default_factory=NodeStats)
    control: NodeControlState = field(default_factory=NodeControlState)

    is_active: bool = True
    is_overloaded: bool = False
    extra_delay: float = 0.0

    original_neighbors: List[int] = field(default_factory=list)

    # Exp12 resource-awareness
    resource_class: str = "medium"
    capacity_score: float = 1.0
    processing_delay: float = 0.0

    def __post_init__(self) -> None:
        if not self.original_neighbors:
            self.original_neighbors = list(self.neighbors)

    def has_seen(self, message_id: str) -> bool:
        return message_id in self.seen_messages

    def mark_seen(self, message_id: str) -> None:
        self.seen_messages.add(message_id)

    def fail(self) -> None:
        self.is_active = False
        self.neighbors = []
        self.gateway_neighbors = []
        self.is_cluster_head = False

    def recover(self) -> None:
        self.is_active = True

    def set_overload(self, extra_delay: float) -> None:
        self.is_overloaded = True
        self.extra_delay = max(0.0, extra_delay)

    def clear_overload(self) -> None:
        self.is_overloaded = False
        self.extra_delay = 0.0

    def leave_network(self) -> None:
        self.fail()

    def rejoin_network(self) -> None:
        self.recover()
