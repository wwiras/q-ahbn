from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class FailureConfig:
    enabled: bool = False
    mode: str = "node_failure"          # node_failure | ch_failure | overload
    trigger_time: float = 3.0
    target_selection: str = "random"    # currently only random
    target_count: int = 1
    overload_delay_multiplier: float = 3.0
    overload_duration: float = 0.0      # 0.0 means keep overload until end


class FailureInjector:
    def __init__(self, cfg: dict, seed: int = 42) -> None:
        failure_cfg = cfg.get("failure", {})
        self.config = FailureConfig(
            enabled=failure_cfg.get("enabled", False),
            mode=failure_cfg.get("mode", "node_failure"),
            trigger_time=float(failure_cfg.get("trigger_time", 3.0)),
            target_selection=failure_cfg.get("target_selection", "random"),
            target_count=int(failure_cfg.get("target_count", 1)),
            overload_delay_multiplier=float(
                failure_cfg.get("overload_delay_multiplier", 3.0)
            ),
            overload_duration=float(failure_cfg.get("overload_duration", 0.0)),
        )
        self.rng = random.Random(seed)
        self.applied = False
        self.cleared = False
        self.failed_node_id: Optional[int] = None
        self.failure_mode_applied: Optional[str] = None

    def should_trigger(self, now: float) -> bool:
        return (
            self.config.enabled
            and not self.applied
            and now >= self.config.trigger_time
        )

    def should_clear_overload(self, now: float) -> bool:
        if not self.applied or self.cleared:
            return False
        if self.config.mode != "overload":
            return False
        if self.config.overload_duration <= 0:
            return False
        return now >= self.config.trigger_time + self.config.overload_duration

    def apply(self, simulator) -> None:
        mode = self.config.mode

        if mode == "node_failure":
            node_id = self._pick_random_non_source_node(simulator)
            if node_id is not None:
                simulator.nodes[node_id].fail()
                self.failed_node_id = node_id
                self.failure_mode_applied = mode

        elif mode == "ch_failure":
            node_id = self._pick_random_cluster_head(simulator)
            if node_id is not None:
                simulator.nodes[node_id].fail()
                self.failed_node_id = node_id
                self.failure_mode_applied = mode

        elif mode == "overload":
            node_id = self._pick_random_cluster_head(simulator)
            if node_id is not None:
                node = simulator.nodes[node_id]
                extra_delay = simulator.base_delay * max(
                    0.0, self.config.overload_delay_multiplier - 1.0
                )
                node.set_overload(extra_delay)
                self.failed_node_id = node_id
                self.failure_mode_applied = mode

        else:
            raise ValueError(f"Unsupported failure mode: {mode}")

        self.applied = True
        simulator.metrics.record_failure_trigger(
            failure_mode=self.failure_mode_applied or mode,
            trigger_time=self.config.trigger_time,
            failed_node_id=self.failed_node_id,
        )

    def clear(self, simulator) -> None:
        if self.config.mode != "overload":
            return
        if self.failed_node_id is None:
            return
        simulator.nodes[self.failed_node_id].clear_overload()
        self.cleared = True

    def _pick_random_non_source_node(self, simulator) -> Optional[int]:
        ids = [nid for nid in simulator.nodes.keys() if nid != simulator.message_source_id]
        if not ids:
            return None
        return self.rng.choice(ids)

    def _pick_random_cluster_head(self, simulator) -> Optional[int]:
        chs = [nid for nid, n in simulator.nodes.items() if n.is_cluster_head]
        if not chs:
            return None
        return self.rng.choice(chs)