from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChurnConfig:
    enabled: bool = False
    start_time: float = 2.0
    interval: float = 1.5
    num_cycles: int = 4
    down_time: float = 2.0
    target_fraction: float = 0.10
    min_targets: int = 1
    allow_cluster_heads: bool = True
    permanent_leave: bool = False


class ChurnManager:
    def __init__(self, cfg: dict, seed: int = 42) -> None:
        churn_cfg = cfg.get("churn", {})
        self.config = ChurnConfig(
            enabled=churn_cfg.get("enabled", False),
            start_time=float(churn_cfg.get("start_time", 2.0)),
            interval=float(churn_cfg.get("interval", 1.5)),
            num_cycles=int(churn_cfg.get("num_cycles", 4)),
            down_time=float(churn_cfg.get("down_time", 2.0)),
            target_fraction=float(churn_cfg.get("target_fraction", 0.10)),
            min_targets=int(churn_cfg.get("min_targets", 1)),
            allow_cluster_heads=bool(churn_cfg.get("allow_cluster_heads", True)),
            permanent_leave=bool(churn_cfg.get("permanent_leave", False)),
        )
        self.rng = random.Random(seed)

    def is_enabled(self) -> bool:
        return self.config.enabled and self.config.num_cycles > 0 and self.config.target_fraction > 0.0

    def schedule_events(self, simulator) -> None:
        if not self.is_enabled():
            return

        available_at: Dict[int, float] = {}
        t = self.config.start_time

        for _ in range(self.config.num_cycles):
            reusable = [nid for nid, release_t in available_at.items() if release_t <= t]
            for nid in reusable:
                del available_at[nid]

            candidates = self._candidate_nodes(simulator, excluded_ids=set(available_at.keys()))
            if not candidates:
                t += self.config.interval
                continue

            target_count = max(
                self.config.min_targets,
                int(round(self.config.target_fraction * len(candidates))),
            )
            target_count = min(target_count, len(candidates))
            if target_count <= 0:
                t += self.config.interval
                continue

            selected = self.rng.sample(candidates, target_count)
            for node_id in selected:
                simulator.schedule_event(
                    time=t,
                    priority=0,
                    event_type="churn_leave",
                    payload={"node_id": node_id, "churn_rate": self.config.target_fraction},
                )
                if not self.config.permanent_leave:
                    rejoin_t = t + self.config.down_time
                    simulator.schedule_event(
                        time=rejoin_t,
                        priority=0,
                        event_type="churn_join",
                        payload={"node_id": node_id, "churn_rate": self.config.target_fraction},
                    )
                    available_at[node_id] = rejoin_t

            t += self.config.interval

    def _candidate_nodes(self, simulator, excluded_ids: set[int]) -> List[int]:
        out: List[int] = []
        for node_id, node in simulator.nodes.items():
            if node_id == simulator.message_source_id:
                continue
            if node_id in excluded_ids:
                continue
            if not self.config.allow_cluster_heads and node.is_cluster_head:
                continue
            out.append(node_id)
        return out