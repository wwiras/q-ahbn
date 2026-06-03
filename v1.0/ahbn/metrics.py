from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MessageRecord:
    message_id: str
    source_id: int
    created_at: float
    first_seen_times: Dict[int, float] = field(default_factory=dict)
    duplicate_count: int = 0
    total_forwards: int = 0


@dataclass
class MetricsCollector:
    messages: Dict[str, MessageRecord] = field(default_factory=dict)

    # Exp10 additions
    failure_mode: Optional[str] = None
    failure_trigger_time: Optional[float] = None
    failed_node_id: Optional[int] = None

    # Exp11 additions
    churn_event_count: int = 0
    churn_leave_count: int = 0
    churn_join_count: int = 0
    cluster_repair_count: int = 0
    mode_switch_count: int = 0
    fanout_change_count: int = 0
    adaptation_event_count: int = 0
    churn_feedback_update_count: int = 0

    def register_message(self, message_id: str, source_id: int, created_at: float) -> None:
        self.messages[message_id] = MessageRecord(
            message_id=message_id,
            source_id=source_id,
            created_at=created_at,
        )

    def record_first_seen(self, node_id: int, message_id: str, time: float) -> None:
        rec = self.messages[message_id]
        rec.first_seen_times.setdefault(node_id, time)

    def record_duplicate(self, message_id: str) -> None:
        self.messages[message_id].duplicate_count += 1

    def record_forward(self, message_id: str, count: int = 1) -> None:
        self.messages[message_id].total_forwards += count

    # Exp10 additions
    def record_failure_trigger(
        self,
        failure_mode: str,
        trigger_time: float,
        failed_node_id: Optional[int],
    ) -> None:
        self.failure_mode = failure_mode
        self.failure_trigger_time = trigger_time
        self.failed_node_id = failed_node_id

    # Exp11 additions
    def record_churn_event(self, event_type: str) -> None:
        self.churn_event_count += 1
        if event_type == "leave":
            self.churn_leave_count += 1
        elif event_type == "join":
            self.churn_join_count += 1

    def record_cluster_repair(self) -> None:
        self.cluster_repair_count += 1

    def record_churn_feedback_update(self) -> None:
        self.churn_feedback_update_count += 1

    def record_adaptation(self, mode_switched: bool, fanout_changed: bool) -> None:
        if mode_switched:
            self.mode_switch_count += 1
        if fanout_changed:
            self.fanout_change_count += 1
        if mode_switched or fanout_changed:
            self.adaptation_event_count += 1

    def summarize_message(self, message_id: str, total_nodes: int) -> dict:
        rec = self.messages[message_id]
        seen = len(rec.first_seen_times)
        delivery_ratio = seen / total_nodes if total_nodes else 0.0

        propagation_delay: Optional[float]
        end_time: Optional[float] = None
        if rec.first_seen_times:
            end_time = max(rec.first_seen_times.values())
            propagation_delay = end_time - rec.created_at
        else:
            propagation_delay = None

        recovery_time: Optional[float] = None
        if self.failure_trigger_time is not None and end_time is not None:
            recovery_time = max(0.0, end_time - self.failure_trigger_time)

        adaptation_rate: float = (
            self.adaptation_event_count / self.churn_feedback_update_count
            if self.churn_feedback_update_count > 0
            else 0.0
        )

        return {
            "message_id": message_id,
            "delivery_ratio": delivery_ratio,
            "propagation_delay": propagation_delay,
            "duplicates": rec.duplicate_count,
            "total_forwards": rec.total_forwards,
            "failure_mode": self.failure_mode,
            "recovery_time": recovery_time,
            "failed_node_id": self.failed_node_id,
            "churn_event_count": self.churn_event_count,
            "churn_leave_count": self.churn_leave_count,
            "churn_join_count": self.churn_join_count,
            "cluster_repair_count": self.cluster_repair_count,
            "mode_switch_count": self.mode_switch_count,
            "fanout_change_count": self.fanout_change_count,
            "adaptation_event_count": self.adaptation_event_count,
            "churn_feedback_update_count": self.churn_feedback_update_count,
            "adaptation_rate": adaptation_rate,
        }