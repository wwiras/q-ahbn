from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class HybridFixedStrategy(ForwardingStrategy):
    """
    Fixed-hybrid forwarding for Exp07.
    """

    def __init__(self, fanout: int = 3, external_leakage: int = 1) -> None:
        self.fanout = fanout
        self.external_leakage = external_leakage

    def _sample(self, simulator, candidates: List[int], k: int) -> List[int]:
        if k <= 0 or not candidates:
            return []
        if len(candidates) <= k:
            return candidates[:]
        return simulator.rng.sample(candidates, k)

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        cluster_mgr = simulator.cluster_manager
        if cluster_mgr is None or self.fanout <= 0:
            return []

        valid_neighbors = set(node.neighbors)

        same_cluster_neighbors = [
            n for n in node.neighbors
            if simulator.nodes[n].cluster_id == node.cluster_id and n != node.node_id
        ]
        other_cluster_neighbors = [
            n for n in node.neighbors
            if simulator.nodes[n].cluster_id != node.cluster_id and n != node.node_id
        ]

        targets: List[int] = []
        remaining_budget = self.fanout

        if node.is_cluster_head:
            members = [
                m for m in cluster_mgr.get_cluster_members(node.cluster_id, exclude=node.node_id)
                if m in valid_neighbors
            ]
            gateways = [
                g for g in node.gateway_neighbors
                if g in valid_neighbors and g != node.node_id
            ]

            gateway_budget = min(self.external_leakage, remaining_budget)
            member_budget = max(0, remaining_budget - gateway_budget)

            sampled_members = self._sample(simulator, members, member_budget)
            targets.extend(sampled_members)
            remaining_budget -= len(sampled_members)

            if remaining_budget > 0 and gateways:
                sampled_gateways = self._sample(
                    simulator,
                    gateways,
                    min(gateway_budget, remaining_budget),
                )
                targets.extend(sampled_gateways)
                remaining_budget -= len(sampled_gateways)

        else:
            ch_id = cluster_mgr.get_cluster_head(node.cluster_id)
            if (
                ch_id is not None
                and ch_id != node.node_id
                and ch_id in valid_neighbors
                and remaining_budget > 0
            ):
                targets.append(ch_id)
                remaining_budget -= 1

            if remaining_budget > 0 and same_cluster_neighbors:
                local_budget = remaining_budget
                if other_cluster_neighbors and remaining_budget > 1:
                    local_budget = remaining_budget - min(
                        self.external_leakage, remaining_budget - 1
                    )

                sampled_local = self._sample(simulator, same_cluster_neighbors, local_budget)
                targets.extend(sampled_local)
                remaining_budget -= len(sampled_local)

            if remaining_budget > 0 and other_cluster_neighbors:
                sampled_external = self._sample(
                    simulator,
                    other_cluster_neighbors,
                    min(self.external_leakage, remaining_budget),
                )
                targets.extend(sampled_external)
                remaining_budget -= len(sampled_external)

        unique_targets = [t for t in dict.fromkeys(targets) if t != node.node_id]

        if len(unique_targets) > self.fanout:
            unique_targets = unique_targets[:self.fanout]

        return unique_targets