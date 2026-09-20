from __future__ import annotations

import copy
import random
import unittest

from ahbn.message import Message
from ahbn.node import Node
from ahbn.simulator import Simulator
from ahbn.strategies.base import ForwardingStrategy


class NoForwardStrategy(ForwardingStrategy):
    def select_targets(self, node, message, simulator):
        return []


def make_sim(observer: bool) -> Simulator:
    nodes = {
        0: Node(node_id=0, neighbors=[1, 2]),
        1: Node(node_id=1, neighbors=[0]),
        2: Node(node_id=2, neighbors=[0]),
    }
    return Simulator(
        nodes=nodes,
        strategy=NoForwardStrategy(),
        seed=42,
        base_delay=1.0,
        jitter=0.0,
        enable_forward_outcome_observer=observer,
    )


def scientific_snapshot(sim: Simulator) -> dict:
    return {
        "clock": sim.clock,
        "queue": copy.deepcopy(sim.queue),
        "rng_state": sim.rng.getstate(),
        "nodes": {
            node_id: {
                "seen_messages": set(node.seen_messages),
                "received_new": node.stats.received_new,
                "received_duplicate": node.stats.received_duplicate,
                "forwarded": node.stats.forwarded,
                "dropped": node.stats.dropped,
                "first_receive_time": dict(node.stats.first_receive_time),
                "last_receive_time": dict(node.stats.last_receive_time),
                "control": copy.deepcopy(node.control),
            }
            for node_id, node in sim.nodes.items()
        },
        "metrics": copy.deepcopy(sim.metrics),
        "adaptive_trace_rows": copy.deepcopy(sim.adaptive_trace_rows),
    }


def run_microcase(observer: bool):
    sim = make_sim(observer)
    msg = Message(message_id="m1", source_id=0, created_at=0.0)

    # Source injection: must never be attributed as a forwarding action.
    sim.handle_receive(now=0.0, dst_id=0, src_id=0, message=msg)

    # Deterministic controlled forwarding interactions.
    sim.handle_receive(now=1.0, dst_id=1, src_id=0, message=msg)
    sim.handle_receive(now=2.0, dst_id=1, src_id=0, message=msg)
    sim.handle_receive(now=3.0, dst_id=2, src_id=0, message=msg)

    return sim


class TestNewAttributionParity(unittest.TestCase):
    def test_exact_attribution_and_source_exclusion(self):
        sim = run_microcase(observer=True)
        self.assertEqual(
            sim.forward_outcome_rows,
            [
                {"time": 1.0, "src_id": 0, "dst_id": 1, "message_id": "m1", "outcome": "NEW"},
                {"time": 2.0, "src_id": 0, "dst_id": 1, "message_id": "m1", "outcome": "DUPLICATE"},
                {"time": 3.0, "src_id": 0, "dst_id": 2, "message_id": "m1", "outcome": "NEW"},
            ],
        )

    def test_observer_has_no_scientific_side_effects(self):
        off = run_microcase(observer=False)
        on = run_microcase(observer=True)

        self.assertEqual(off.forward_outcome_rows, [])
        self.assertEqual(scientific_snapshot(off), scientific_snapshot(on))


if __name__ == "__main__":
    unittest.main()
