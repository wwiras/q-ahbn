from __future__ import annotations

import heapq
import math
import random
from typing import Dict, Optional

from ahbn.control import AHBNController
from ahbn.event import Event
from ahbn.message import Message
from ahbn.metrics import MetricsCollector
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy
from ahbn.topology import repair_topology_after_churn
from ahbn.utils import AdaptiveTraceRow


class Simulator:
    def __init__(
        self,
        nodes: Dict[int, Node],
        strategy: ForwardingStrategy,
        seed: int = 42,
        base_delay: float = 1.0,
        jitter: float = 0.2,
        cluster_manager=None,
        controller: Optional[AHBNController] = None,
        ch_overload_factor: float = 1.0,
        failure_injector=None,
        churn_manager=None,
        experiment_name: str = "unknown",
        strategy_name: str = "unknown",
        scenario_tag: str = "default",
        enable_adaptive_trace: bool = False,
        resource_aware_heads: bool = False,
    ) -> None:
        self.nodes = nodes
        self.strategy = strategy
        self.seed = seed
        self.rng = random.Random(seed)
        self.clock = 0.0
        self.queue: list[Event] = []
        self.metrics = MetricsCollector()

        self.base_delay = base_delay
        self.jitter = jitter
        self.cluster_manager = cluster_manager
        self.controller = controller
        self.ch_overload_factor = ch_overload_factor
        self.failure_injector = failure_injector
        self.message_source_id: Optional[int] = None
        self.churn_manager = churn_manager
        self.resource_aware_heads = resource_aware_heads

        self.experiment_name = experiment_name
        self.strategy_name = strategy_name
        self.scenario_tag = scenario_tag
        self.enable_adaptive_trace = enable_adaptive_trace
        self.adaptive_trace_rows: list[AdaptiveTraceRow] = []

        if self.churn_manager is not None:
            self.churn_manager.schedule_events(self)

    def schedule_event(self, time: float, priority: int, event_type: str, payload: dict) -> None:
        heapq.heappush(self.queue, Event(time, priority, event_type, payload))

    def send_message(self, src_id: int, dst_id: int, message: Message, now: float) -> None:
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]

        if not src.is_active or not dst.is_active:
            return

        extra = 0.0

        if dst.is_cluster_head:
            extra += self.base_delay * max(0.0, self.ch_overload_factor - 1.0)

        if dst.is_overloaded:
            extra += dst.extra_delay

        # Exp12 resource-aware latency
        extra += 0.5 * max(0.0, src.processing_delay)
        extra += 0.5 * max(0.0, dst.processing_delay)

        delay = self.base_delay + self.rng.uniform(0.0, self.jitter) + extra
        self.schedule_event(
            time=now + delay,
            priority=1,
            event_type="receive",
            payload={"dst_id": dst_id, "src_id": src_id, "message": message},
        )

    def inject_message(self, source_id: int, message_id: str) -> None:
        self.message_source_id = source_id
        msg = Message(message_id=message_id, source_id=source_id, created_at=self.clock)
        self.metrics.register_message(message_id, source_id, self.clock)
        self.schedule_event(
            time=self.clock,
            priority=0,
            event_type="receive",
            payload={"dst_id": source_id, "src_id": source_id, "message": msg},
        )

    def log_adaptive_trace(self, node: Node, event_type: str, message: Optional[Message] = None) -> None:
        if not self.enable_adaptive_trace or self.controller is None:
            return

        snap = self.controller.snapshot_state(node.control)
        self.adaptive_trace_rows.append(
            AdaptiveTraceRow(
                experiment=self.experiment_name,
                strategy=self.strategy_name,
                seed=self.seed,
                scenario_tag=self.scenario_tag,
                time=self.clock,
                node_id=node.node_id,
                message_id=message.message_id if message is not None else None,
                event_type=event_type,
                mode=snap["mode"],
                fanout=snap["fanout"],
                weight=snap["weight"],
                tau=snap["tau"],
                dup_ratio_raw=node.stats.duplicate_ratio_raw,
                d_hat=snap["d_hat"],
                l_hat=snap["l_hat"],
                rho_hat=snap["rho_hat"],
                u_hat=snap["u_hat"],
                deg_hat=snap["deg_hat"],
                ov_hat=snap["ov_hat"],
                r_hat=snap["r_hat"],
                c_hat=snap.get("c_hat", 0.0),
                resource_class=node.resource_class,
                capacity_score=node.capacity_score,
                processing_delay=node.processing_delay,
                received_new=node.stats.received_new,
                received_duplicate=node.stats.received_duplicate,
                forwarded=node.stats.forwarded,
            )
        )

    def get_node_degree(self, node: Node) -> int:
        return len(node.neighbors)

    def get_neighbor_overlap(self, node: Node) -> float:
        nbrs = set(node.neighbors)
        if not nbrs:
            return 0.0

        vals: list[float] = []
        for nbr_id in nbrs:
            nbr_node = self.nodes.get(nbr_id)
            if nbr_node is None:
                continue
            nbr_set = set(nbr_node.neighbors)
            inter = len(nbrs & nbr_set)
            union = len(nbrs | nbr_set)
            vals.append(inter / union if union > 0 else 0.0)

        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def get_redundancy_proxy(self, node: Node) -> float:
        if self.controller is None:
            return 0.0
        degree = self.get_node_degree(node)
        overlap = self.get_neighbor_overlap(node)
        degree_ref = getattr(self.controller, "degree_ref", 10.0)
        b_degree = getattr(self.controller, "b_degree", 0.25)
        b_overlap = getattr(self.controller, "b_overlap", 0.75)
        norm_degree = min(2.0, degree / max(1.0, degree_ref))
        return b_overlap * overlap + b_degree * norm_degree

    def get_capacity_proxy(self, node: Node) -> float:
        capacity_term = max(0.0, (1.0 / max(0.25, node.capacity_score)) - 1.0)
        delay_term = max(0.0, node.processing_delay / max(0.1, self.base_delay))
        return 0.6 * capacity_term + 0.4 * delay_term

    def update_ahbn_state(
        self,
        node: Node,
        now: float,
        receive_lag: float,
        churn_proxy: float = 0.0,
        event_type: str = "control_update",
    ) -> None:
        if self.controller is None:
            return

        total_recv = node.stats.received_new + node.stats.received_duplicate
        duplicate_ratio = node.stats.received_duplicate / total_recv if total_recv > 0 else 0.0
        load_proxy = float(node.stats.forwarded) / max(0.25, node.capacity_score)
        latency_proxy = receive_lag
        degree_proxy = float(self.get_node_degree(node))
        overlap_proxy = self.get_neighbor_overlap(node)
        redundancy_proxy = self.get_redundancy_proxy(node)
        capacity_proxy = self.get_capacity_proxy(node)

        prev_mode = node.control.mode
        prev_fanout = node.control.fanout

        self.controller.update_metrics(
            node.control,
            duplicate_ratio=duplicate_ratio,
            load_proxy=load_proxy,
            latency_proxy=latency_proxy,
            churn_proxy=churn_proxy,
            degree_proxy=degree_proxy,
            overlap_proxy=overlap_proxy,
            redundancy_proxy=redundancy_proxy,
            capacity_proxy=capacity_proxy,
        )
        self.controller.decide_mode_and_fanout(node.control)

        mode_switched = node.control.mode != prev_mode
        fanout_changed = node.control.fanout != prev_fanout
        self.metrics.record_adaptation(mode_switched, fanout_changed)
        self.log_adaptive_trace(node, event_type=event_type)

    def apply_churn_feedback(self, churn_proxy: float) -> None:
        if self.controller is None:
            return

        for node in self.nodes.values():
            if not node.is_active:
                continue
            self.metrics.record_churn_feedback_update()
            self.update_ahbn_state(
                node=node,
                now=self.clock,
                receive_lag=float(node.control.l_hat),
                churn_proxy=churn_proxy,
                event_type="churn_control_update",
            )

    def handle_receive(self, now: float, dst_id: int, src_id: int, message: Message) -> None:
        self.clock = now
        node = self.nodes[dst_id]
        if not node.is_active:
            return

        if node.has_seen(message.message_id):
            node.stats.received_duplicate += 1
            self.metrics.record_duplicate(message.message_id)
            self.update_ahbn_state(node, now, now - message.created_at)
            self.log_adaptive_trace(node, event_type="duplicate_drop", message=message)
            return

        node.mark_seen(message.message_id)
        node.stats.received_new += 1
        node.stats.first_receive_time.setdefault(message.message_id, now)
        node.stats.last_receive_time[message.message_id] = now
        self.metrics.record_first_seen(node.node_id, message.message_id, now)

        self.update_ahbn_state(node, now, now - message.created_at)
        self.log_adaptive_trace(node, event_type="new_receive", message=message)

        targets = self.strategy.select_targets(node, message, self)
        unique_targets = [t for t in dict.fromkeys(targets) if t != node.node_id]

        for t in unique_targets:
            self.send_message(node.node_id, t, message, now)

        node.stats.forwarded += len(unique_targets)
        self.metrics.record_forward(message.message_id, len(unique_targets))
        self.log_adaptive_trace(node, event_type="forward_decision", message=message)

    def handle_churn_leave(self, now: float, node_id: int, churn_rate: float) -> None:
        self.clock = now
        node = self.nodes[node_id]
        if not node.is_active:
            return

        node.leave_network()
        repair_topology_after_churn(
            self.nodes,
            self.cluster_manager,
            resource_aware_heads=self.resource_aware_heads,
        )
        self.metrics.record_churn_event("leave")
        self.metrics.record_cluster_repair()
        self.apply_churn_feedback(churn_proxy=churn_rate)

    def handle_churn_join(self, now: float, node_id: int, churn_rate: float) -> None:
        self.clock = now
        node = self.nodes[node_id]
        if node.is_active:
            return

        node.rejoin_network()
        repair_topology_after_churn(
            self.nodes,
            self.cluster_manager,
            resource_aware_heads=self.resource_aware_heads,
        )
        self.metrics.record_churn_event("join")
        self.metrics.record_cluster_repair()
        self.apply_churn_feedback(churn_proxy=churn_rate)

    def get_resource_metrics(self) -> dict:
        active_nodes = [n for n in self.nodes.values() if n.is_active]
        if not active_nodes:
            return {
                "max_normalized_load": 0.0,
                "load_balance_cv": 0.0,
                "strong_forward_share": 0.0,
                "medium_forward_share": 0.0,
                "weak_forward_share": 0.0,
            }

        norm_loads = [n.stats.forwarded / max(0.25, n.capacity_score) for n in active_nodes]
        mean_load = sum(norm_loads) / len(norm_loads)
        variance = sum((x - mean_load) ** 2 for x in norm_loads) / len(norm_loads)
        std_load = math.sqrt(variance)
        total_forwarded = sum(n.stats.forwarded for n in active_nodes)

        class_totals = {"strong": 0, "medium": 0, "weak": 0}
        for node in active_nodes:
            class_totals[node.resource_class] = class_totals.get(node.resource_class, 0) + node.stats.forwarded

        denom = total_forwarded if total_forwarded > 0 else 1
        return {
            "max_normalized_load": max(norm_loads) if norm_loads else 0.0,
            "load_balance_cv": std_load / mean_load if mean_load > 0 else 0.0,
            "strong_forward_share": class_totals.get("strong", 0) / denom,
            "medium_forward_share": class_totals.get("medium", 0) / denom,
            "weak_forward_share": class_totals.get("weak", 0) / denom,
        }

    def run(self, until: float = 1000.0) -> None:
        while self.queue:
            event = heapq.heappop(self.queue)
            if event.time > until:
                break

            self.clock = event.time

            if self.failure_injector is not None and self.failure_injector.should_trigger(self.clock):
                self.failure_injector.apply(self)

            if self.failure_injector is not None and self.failure_injector.should_clear_overload(self.clock):
                self.failure_injector.clear(self)

            if event.event_type == "receive":
                self.handle_receive(
                    now=event.time,
                    dst_id=event.payload["dst_id"],
                    src_id=event.payload["src_id"],
                    message=event.payload["message"],
                )
            elif event.event_type == "churn_leave":
                self.handle_churn_leave(
                    now=event.time,
                    node_id=event.payload["node_id"],
                    churn_rate=float(event.payload.get("churn_rate", 0.0)),
                )
            elif event.event_type == "churn_join":
                self.handle_churn_join(
                    now=event.time,
                    node_id=event.payload["node_id"],
                    churn_rate=float(event.payload.get("churn_rate", 0.0)),
                )
