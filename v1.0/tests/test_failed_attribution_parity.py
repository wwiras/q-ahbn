from __future__ import annotations

import copy
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
        0: Node(node_id=0, neighbors=[1]),
        1: Node(node_id=1, neighbors=[0]),
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
                "is_active": node.is_active,
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


def run_microcase(observer: bool) -> Simulator:
    sim = make_sim(observer)

    # Register both messages through the normal simulator path, then remove
    # their source-injection events. The controlled interactions below are
    # the only forwarding outcomes under test.
    sim.inject_message(source_id=0, message_id="m1")
    injection_m1 = sim.queue.pop()
    m1 = injection_m1.payload["message"]

    sim.inject_message(source_id=0, message_id="m2")
    injection_m2 = sim.queue.pop()
    m2 = injection_m2.payload["message"]

    # A active -> B active: receiver classifies m1 as NEW.
    sim.handle_receive(now=1.0, dst_id=1, src_id=0, message=m1)

    # A active -> B inactive: send cannot schedule a receive event and is
    # observed as FAILED.
    sim.nodes[1].is_active = False
    sim.send_message(src_id=0, dst_id=1, message=m2, now=2.0)

    # A inactive -> B active: excluded by the frozen FAILED semantic contract.
    sim.nodes[1].is_active = True
    sim.nodes[0].is_active = False
    m3 = Message(message_id="m3", source_id=0, created_at=3.0)
    sim.send_message(src_id=0, dst_id=1, message=m3, now=3.0)

    return sim


class TestFailedAttributionParity(unittest.TestCase):
    def test_exact_failed_attribution_and_inactive_sender_exclusion(self):
        sim = run_microcase(observer=True)
        self.assertEqual(
            sim.forward_outcome_rows,
            [
                {"time": 1.0, "src_id": 0, "dst_id": 1, "message_id": "m1", "outcome": "NEW"},
                {"time": 2.0, "src_id": 0, "dst_id": 1, "message_id": "m2", "outcome": "FAILED"},
            ],
        )

    def test_failed_observer_has_no_scientific_side_effects(self):
        off = run_microcase(observer=False)
        on = run_microcase(observer=True)

        self.assertEqual(off.forward_outcome_rows, [])
        self.assertEqual(scientific_snapshot(off), scientific_snapshot(on))


if __name__ == "__main__":
    unittest.main()
