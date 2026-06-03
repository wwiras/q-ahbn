from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class ClusterStrategy(ForwardingStrategy):
    """
    Structured cluster dissemination baseline.

    Interpretation:
    - Member nodes forward upward to their cluster head.
    - Cluster heads disseminate downward to their members and outward
      to gateway/peer heads.

    Exp11-only resilience patch:
    - members may include one same-cluster backup neighbor
    - cluster heads may include one physical backup neighbor

    This preserves earlier experiment behavior because the backup
    paths are enabled only when simulator.experiment_name == "exp11".
    """

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        cluster_mgr = simulator.cluster_manager
        if cluster_mgr is None:
            return []

        exp11_mode = getattr(simulator, "experiment_name", "") == "exp11"

        if node.is_cluster_head:
            members = cluster_mgr.get_cluster_members(
                node.cluster_id,
                exclude=node.node_id,
            )
            gateways = list(node.gateway_neighbors)
            selected = list(dict.fromkeys(members + gateways))

            # Exp11-only: keep one extra physical backup path alive
            if exp11_mode:
                for nbr_id in node.neighbors:
                    if nbr_id != node.node_id and nbr_id not in selected:
                        selected.append(nbr_id)
                        break

            return selected

        ch_id = cluster_mgr.get_cluster_head(node.cluster_id)
        selected: List[int] = []

        if ch_id is not None and ch_id != node.node_id:
            selected.append(ch_id)

        # Exp11-only: add one same-cluster backup peer if available
        if exp11_mode:
            for nbr_id in node.neighbors:
                nbr = simulator.nodes.get(nbr_id)
                if nbr is None:
                    continue
                if nbr_id == ch_id:
                    continue
                if nbr.cluster_id == node.cluster_id and nbr.is_active:
                    selected.append(nbr_id)
                    break

        return list(dict.fromkeys(selected))