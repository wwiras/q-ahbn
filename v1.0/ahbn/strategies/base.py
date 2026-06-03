from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ahbn.message import Message
from ahbn.node import Node


class ForwardingStrategy(ABC):
    @abstractmethod
    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        raise NotImplementedError