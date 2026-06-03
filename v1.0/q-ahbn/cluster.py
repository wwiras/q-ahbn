from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClusterManager:
    cluster_to_members: Dict[int, List[int]] = field(default_factory=dict)
    cluster_to_head: Dict[int, int] = field(default_factory=dict)

    def get_cluster_members(self, cluster_id: int | None, exclude: int | None = None) -> List[int]:
        if cluster_id is None:
            return []

        members = self.cluster_to_members.get(cluster_id, [])
        if exclude is None:
            return members[:]
        return [m for m in members if m != exclude]

    def get_cluster_head(self, cluster_id: int | None) -> Optional[int]:
        if cluster_id is None:
            return None
        return self.cluster_to_head.get(cluster_id)