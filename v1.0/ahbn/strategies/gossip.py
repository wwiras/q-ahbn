from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class GossipStrategy(ForwardingStrategy):
    def __init__(self, fanout: int) -> None:
        self.fanout = fanout

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        candidates = list(node.neighbors)
        if not candidates:
            return []

        k = min(self.fanout, len(candidates))
        return simulator.rng.sample(candidates, k)