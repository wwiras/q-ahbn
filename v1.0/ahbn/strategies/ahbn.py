from __future__ import annotations

import hashlib
import math
from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy


def deterministic_hash01(*parts: str) -> float:
    s = ":".join(parts).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:16], 16) / float(16**16)


class AHBNStrategy(ForwardingStrategy):
    """
    AHBN strategy with backward-compatible defaults.

    Old behavior:
    - hybrid_mode=False: hard mode switch
    - hybrid_mode=True but mode_sensitive_mix=False:
      same weighted hybrid behavior as before

    New Exp11 behavior:
    - hybrid_mode=True and mode_sensitive_mix=True:
      control mode actually changes target composition and order
    """

    def __init__(
        self,
        default_fanout: int = 3,
        adaptive_fanout: bool = False,
        hybrid_mode: bool = True,
        use_tau_gate: bool = True,
        min_cluster_targets: int = 1,
        # Exp11-focused options; defaults preserve older behavior
        mode_sensitive_mix: bool = False,
        cluster_mode_bias: float = 0.75,
        gossip_mode_bias: float = 0.75,
        preserve_cluster_path_under_tau: bool = False,
        cluster_reserve_in_gossip_mode: int = 0,
        gossip_reserve_in_cluster_mode: int = 0,
        resource_aware_targeting: bool = False,
    ) -> None:
        self.default_fanout = default_fanout
        self.adaptive_fanout = adaptive_fanout
        self.hybrid_mode = hybrid_mode
        self.use_tau_gate = use_tau_gate
        self.min_cluster_targets = min_cluster_targets

        self.mode_sensitive_mix = mode_sensitive_mix
        self.cluster_mode_bias = cluster_mode_bias
        self.gossip_mode_bias = gossip_mode_bias
        self.preserve_cluster_path_under_tau = preserve_cluster_path_under_tau
        self.cluster_reserve_in_gossip_mode = cluster_reserve_in_gossip_mode
        self.gossip_reserve_in_cluster_mode = gossip_reserve_in_cluster_mode
        self.resource_aware_targeting = resource_aware_targeting

        self._gossip = GossipStrategy(fanout=default_fanout)
        self._cluster = ClusterStrategy()

    def _get_effective_fanout(self, node: Node) -> int:
        if self.adaptive_fanout:
            return max(1, int(node.control.fanout))
        return max(1, int(self.default_fanout))

    def _passes_tau_gate(self, node: Node, message: Message) -> bool:
        tau = float(getattr(node.control, "tau", 1.0))
        score = deterministic_hash01(str(node.node_id), message.message_id)
        return score < tau

    def _dedup_preserve_order(self, targets: List[int], self_id: int) -> List[int]:
        return [t for t in dict.fromkeys(targets) if t != self_id]

    def _allocate_counts_legacy(self, fanout: int, gossip_weight: float) -> tuple[int, int]:
        n_gossip = int(round(fanout * gossip_weight))
        n_cluster = fanout - n_gossip

        if fanout > 1:
            n_cluster = max(self.min_cluster_targets, n_cluster)
            n_cluster = min(n_cluster, fanout)
            n_gossip = max(0, fanout - n_cluster)

        return n_cluster, n_gossip

    def _allocate_counts_mode_sensitive(
        self,
        fanout: int,
        gossip_weight: float,
        mode: str,
    ) -> tuple[int, int]:
        """
        Exp11 logic:
        - cluster mode => cluster-majority composition
        - gossip mode => gossip-majority composition
        - still allows small cross-over reserve so hybrid behavior remains visible
        """
        base_gossip = int(round(fanout * gossip_weight))
        base_cluster = fanout - base_gossip

        if mode == "cluster":
            target_cluster = int(math.ceil(fanout * self.cluster_mode_bias))
            target_cluster = max(self.min_cluster_targets, target_cluster)
            target_cluster = min(target_cluster, fanout)

            reserve_gossip = min(
                self.gossip_reserve_in_cluster_mode,
                max(0, fanout - target_cluster),
            )

            n_cluster = max(base_cluster, target_cluster)
            n_cluster = min(n_cluster, fanout - reserve_gossip)
            n_gossip = fanout - n_cluster

        elif mode == "gossip":
            target_gossip = int(math.ceil(fanout * self.gossip_mode_bias))
            target_gossip = max(1, target_gossip)
            target_gossip = min(target_gossip, fanout)

            reserve_cluster = min(
                self.cluster_reserve_in_gossip_mode,
                max(0, fanout - target_gossip),
            )

            n_gossip = max(base_gossip, target_gossip)
            n_gossip = min(n_gossip, fanout - reserve_cluster)
            n_cluster = fanout - n_gossip

        else:
            n_cluster, n_gossip = self._allocate_counts_legacy(fanout, gossip_weight)

        return n_cluster, n_gossip

    def _sort_targets_by_capacity(self, targets: List[int], simulator) -> List[int]:
        if not self.resource_aware_targeting:
            return targets
        return sorted(
            targets,
            key=lambda nid: (
                -float(getattr(simulator.nodes[nid], "capacity_score", 1.0)),
                float(getattr(simulator.nodes[nid], "processing_delay", 0.0)),
                nid,
            ),
        )

    def _take_unique(self, pool: List[int], selected: List[int], limit: int) -> None:
        for t in pool:
            if t not in selected:
                selected.append(t)
            if len(selected) >= limit:
                break

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        fanout = self._get_effective_fanout(node)
        self._gossip.fanout = fanout

        # ------------------------------------------------------------
        # Legacy hard-switch behavior for Exp07 / Exp08 compatibility
        # ------------------------------------------------------------
        if not self.hybrid_mode:
            if node.control.mode == "gossip":
                return self._gossip.select_targets(node, message, simulator)
            return self._cluster.select_targets(node, message, simulator)

        gossip_weight = float(getattr(node.control, "weight", 0.5))
        gossip_weight = max(0.0, min(1.0, gossip_weight))
        mode = getattr(node.control, "mode", "gossip")

        gossip_targets = self._sort_targets_by_capacity(
            self._dedup_preserve_order(
                self._gossip.select_targets(node, message, simulator),
                node.node_id,
            ),
            simulator,
        )
        cluster_targets = self._sort_targets_by_capacity(
            self._dedup_preserve_order(
                self._cluster.select_targets(node, message, simulator),
                node.node_id,
            ),
            simulator,
        )

        # ------------------------------------------------------------
        # Tau handling
        # Old behavior: tau failure returns []
        # Exp11 fix: if cluster-oriented, preserve at least a structured path
        # ------------------------------------------------------------
        tau_ok = True
        if self.use_tau_gate:
            tau = float(getattr(node.control, "tau", 1.0))

            # PATCH: duplicate-aware suppression for strong nodes
            if getattr(node, "capacity_score", 0.0) >= 0.8:
                tau *= (1.0 - 0.5 * float(getattr(node.control, "d_hat", 0.0)))
                tau = max(0.05, min(1.0, tau))

            score = deterministic_hash01(str(node.node_id), message.message_id)
            tau_ok = score < tau

        if not tau_ok:
            if (
                self.mode_sensitive_mix
                and self.preserve_cluster_path_under_tau
                and mode == "cluster"
                and cluster_targets
            ):
                # preserve minimal structured progress
                keep = max(1, self.min_cluster_targets)
                return cluster_targets[:keep]
            return []
        
        # tau_ok = True
        # if self.use_tau_gate:
        #     tau_ok = self._passes_tau_gate(node, message)

        # if not tau_ok:
        #     if (
        #         self.mode_sensitive_mix
        #         and self.preserve_cluster_path_under_tau
        #         and mode == "cluster"
        #         and cluster_targets
        #     ):
        #         # preserve minimal structured progress
        #         keep = max(1, self.min_cluster_targets)
        #         return cluster_targets[:keep]
        #     return []
        

        # ------------------------------------------------------------
        # Count allocation
        # ------------------------------------------------------------
        if self.mode_sensitive_mix:
            n_cluster, n_gossip = self._allocate_counts_mode_sensitive(
                fanout=fanout,
                gossip_weight=gossip_weight,
                mode=mode,
            )
        else:
            n_cluster, n_gossip = self._allocate_counts_legacy(
                fanout=fanout,
                gossip_weight=gossip_weight,
            )

        selected: List[int] = []

        # ------------------------------------------------------------
        # Order now depends on mode in Exp11 mode-sensitive mix
        # ------------------------------------------------------------
        if self.mode_sensitive_mix and mode == "gossip":
            self._take_unique(gossip_targets, selected, n_gossip)
            self._take_unique(cluster_targets, selected, n_gossip + n_cluster)
        else:
            # legacy order and cluster-mode order: cluster first
            self._take_unique(cluster_targets, selected, n_cluster)
            self._take_unique(gossip_targets, selected, n_cluster + n_gossip)

        # backfill if still short
        if len(selected) < fanout:
            self._take_unique(cluster_targets, selected, fanout)
        if len(selected) < fanout:
            self._take_unique(gossip_targets, selected, fanout)

        return selected[:fanout]