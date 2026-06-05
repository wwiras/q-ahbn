from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Tuple


StateKey = Tuple[str, str, str, str, str, str]
ActionName = str


@dataclass(frozen=True)
class QAction:
    """A small meta-control adjustment applied after AHBN's normal decision."""
    name: str
    fanout_delta: int = 0
    weight_delta: float = 0.0
    tau_multiplier: float = 1.0


class QAHBNController:
    """
    Q-AHBN meta-controller.

    AHBN remains intact.
    Q-learning only learns small meta-actions that adjust AHBN's output.
    """

    def __init__(self, base_controller: Any, cfg: dict | None = None, seed: int = 42) -> None:
        self.base = base_controller
        self.params = base_controller.params
        self.cfg = cfg or {}
        self.rng = random.Random(seed)

        self.alpha: float = float(self.cfg.get("alpha", 0.25))
        self.gamma: float = float(self.cfg.get("gamma", 0.90))
        self.epsilon: float = float(self.cfg.get("epsilon", 0.30))
        self.epsilon_min: float = float(self.cfg.get("epsilon_min", 0.03))
        self.epsilon_decay: float = float(self.cfg.get("epsilon_decay", 0.995))

        # Reward weights.
        # Important change:
        # Delivery / recovery pressure is now stronger than duplicate suppression.
        self.w_delivery_proxy: float = float(self.cfg.get("w_delivery_proxy", 5.00))
        self.w_dup: float = float(self.cfg.get("w_dup", 0.50))
        self.w_latency: float = float(self.cfg.get("w_latency", 0.30))
        self.w_load: float = float(self.cfg.get("w_load", 0.20))
        self.w_redundancy: float = float(self.cfg.get("w_redundancy", 0.20))
        self.w_churn: float = float(self.cfg.get("w_churn", 0.20))
        self.w_capacity: float = float(self.cfg.get("w_capacity", 0.20))
        self.w_forwarding = float(self.cfg.get("w_forwarding", 0.50))

        # Bonus terms for unstable situations.
        self.w_recovery_bonus: float = float(self.cfg.get("w_recovery_bonus", 2.00))
        self.high_churn_threshold: float = float(self.cfg.get("high_churn_threshold", 0.20))
        self.high_latency_threshold: float = float(self.cfg.get("high_latency_threshold", 1.20))

        # Normalizers prevent large raw values from dominating the reward.
        self.latency_ref: float = float(self.cfg.get("latency_ref", max(1.0, self.params.l0)))
        self.load_ref: float = float(self.cfg.get("load_ref", max(1.0, self.params.u0)))

        self.actions: List[QAction] = [
            QAction("ahbn_base", fanout_delta=0, weight_delta=0.00, tau_multiplier=1.00),
            QAction("more_structured", fanout_delta=-1, weight_delta=-0.12, tau_multiplier=0.90),
            QAction("more_gossip", fanout_delta=1, weight_delta=0.12, tau_multiplier=1.08),
            # QAction("duplicate_suppression", fanout_delta=-1, weight_delta=-0.18, tau_multiplier=0.75),
            QAction("duplicate_suppression",fanout_delta=0,weight_delta=-0.08,tau_multiplier=0.95),
            QAction("recovery_push", fanout_delta=1, weight_delta=0.18, tau_multiplier=1.15),
            QAction("resource_conservative", fanout_delta=-1, weight_delta=-0.10, tau_multiplier=0.85),
        ]

        self.q_table: DefaultDict[StateKey, Dict[ActionName, float]] = defaultdict(
            lambda: {a.name: 0.0 for a in self.actions}
        )

        self.prev: Dict[int, Tuple[StateKey, ActionName]] = {}
        self.reward_history: List[float] = []
        self.action_history: List[ActionName] = []

        self.update_count: int = 0
        self.decision_count: int = 0

    # ------------------------------------------------------------------
    # Delegate AHBN behavior that must remain unchanged
    # ------------------------------------------------------------------
    def update_metrics(self, *args, **kwargs) -> None:
        self.base.update_metrics(*args, **kwargs)

    def snapshot_state(self, state) -> dict:
        snap = self.base.snapshot_state(state)
        snap.update(
            {
                "q_state": getattr(state, "q_state", None),
                "q_action": getattr(state, "q_action", None),
                "q_reward": getattr(state, "q_reward", 0.0),
                "q_epsilon": self.epsilon,
            }
        )
        return snap

    # ------------------------------------------------------------------
    # Q-learning core
    # ------------------------------------------------------------------
    def decide_mode_and_fanout(self, state) -> None:
        # 1) Let original AHBN decide first.
        self.base.decide_mode_and_fanout(state)

        # 2) Convert current AHBN-observed metrics into a compact RL state.
        s = self.discretize_state(state)
        sid = id(state)

        # 3) Update Q-value from previous decision for this node.
        if sid in self.prev:
            prev_s, prev_a = self.prev[sid]
            reward = self.compute_reward(state)

            best_next = max(self.q_table[s].values())
            old_q = self.q_table[prev_s][prev_a]
            new_q = old_q + self.alpha * (reward + self.gamma * best_next - old_q)

            self.q_table[prev_s][prev_a] = new_q
            self.reward_history.append(reward)
            self.update_count += 1
            state.q_reward = reward
        else:
            state.q_reward = 0.0

        # 4) Choose and apply next meta-action.
        action_name = self.choose_action(s)
        self.apply_action(state, action_name)

        self.prev[sid] = (s, action_name)
        self.action_history.append(action_name)
        self.decision_count += 1

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        state.q_state = "|".join(s)
        state.q_action = action_name
        state.q_epsilon = self.epsilon

    def bucket3(self, value: float, low: float, high: float) -> str:
        if value < low:
            return "L"
        if value < high:
            return "M"
        return "H"

    def discretize_state(self, state) -> StateKey:
        return (
            self.bucket3(float(state.d_hat), 0.10, 0.35),
            self.bucket3(float(state.l_hat), self.params.l0 * 0.75, self.params.l0 * 1.50),
            self.bucket3(float(state.u_hat), self.params.u0 * 0.50, self.params.u0 * 1.25),
            self.bucket3(float(state.rho_hat), self.params.rho0 * 0.75, max(0.20, self.params.rho0 * 1.75)),
            self.bucket3(float(state.r_hat), self.params.r0 * 0.75, self.params.r0 * 1.50),
            self.bucket3(float(getattr(state, "c_hat", 0.0)), 0.20, 0.65),
        )

    def choose_action(self, s: StateKey) -> ActionName:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions).name

        qvals = self.q_table[s]
        max_q = max(qvals.values())
        best = [a for a, q in qvals.items() if q == max_q]
        return self.rng.choice(best)

    def compute_reward(self, state) -> float:
        """
        Reward redesign.

        Previous behavior:
            Q-AHBN reduced duplicates but also reduced delivery.

        New behavior:
            Prioritize delivery / recovery pressure first.
            Penalize duplicates, latency, forwarding pressure, churn, and capacity pressure second.

        Interpretation:
            The learner should not simply minimize forwarding.
            It should maintain dissemination effectiveness while reducing unnecessary overhead.
        """

        dup = min(1.0, max(0.0, float(state.d_hat)))
        lat = min(2.0, max(0.0, float(state.l_hat) / max(0.1, self.latency_ref)))
        load = min(2.0, max(0.0, float(state.u_hat) / max(0.1, self.load_ref)))
        red = min(2.0, max(0.0, float(state.r_hat)))
        churn = min(2.0, max(0.0, float(state.rho_hat)))
        cap = min(2.0, max(0.0, float(getattr(state, "c_hat", 0.0))))
        
        # fanout_norm = (
        #     float(state.fanout) /
        #     max(1.0, self.params.max_fanout)
        # )
        
        # fanout_norm = min(
        #     1.0,
        #     max(0.0, float(state.fanout) / max(1.0, float(self.params.max_fanout)))
        # )
        
        

        # Local delivery proxy:
        # If duplication is low and latency is controlled, dissemination is likely healthier.
        # delivery_proxy = max(0.0, 1.0 - (0.50 * dup + 0.35 * min(1.0, lat) + 0.15 * min(1.0, churn)))
        
        delivery_estimate = min(
            1.0,
            max(0.0, float(getattr(state, "delivery_estimate", 0.0)))
        )

        # Recovery pressure:
        # When churn or latency is high, the system should avoid becoming too conservative.
        recovery_pressure = min(1.0, 0.5 * min(1.0, lat) + 0.5 * min(1.0, churn))

        # reward = (
        #     self.w_delivery_proxy * delivery_proxy
        #     - self.w_dup * dup
        #     - self.w_latency * lat
        #     - self.w_load * load
        #     - self.w_redundancy * red
        #     - self.w_churn * churn
        #     - self.w_capacity * cap
        # )
        
        # reward = (
        #     self.w_delivery_proxy * delivery_proxy 
        #     + self.w_forwarding * fanout_norm
        #     - self.w_dup * dup
        #     - self.w_latency * lat
        #     - self.w_load * load
        #     - self.w_redundancy * red
        #     - self.w_churn * churn
        #     - self.w_capacity * cap
        # )
        
        reward = (
            self.w_delivery_proxy * delivery_estimate
            - self.w_dup * dup
            - self.w_latency * lat
            - self.w_load * load
            - self.w_redundancy * red
            - self.w_churn * churn
            - self.w_capacity * cap
        )

        # Bonus under dynamic conditions.
        # This encourages the learner to maintain dissemination capability during failure/churn.
        if churn >= self.high_churn_threshold or lat >= self.high_latency_threshold:
            reward += self.w_recovery_bonus * recovery_pressure

        return reward

    def apply_action(self, state, action_name: ActionName) -> None:
        action = next(a for a in self.actions if a.name == action_name)
        p = self.params

        state.weight = max(
            p.min_weight,
            min(p.max_weight, float(state.weight) + action.weight_delta),
        )

        state.mode = "gossip" if state.weight >= p.mode_threshold else "cluster"

        state.fanout = max(
            p.min_fanout,
            min(p.max_fanout, int(state.fanout) + action.fanout_delta),
        )

        state.tau = max(
            p.tau_min,
            min(p.tau_max, float(state.tau) * action.tau_multiplier),
        )

    def get_learning_summary(self) -> dict:
        if self.reward_history:
            mean_reward = sum(self.reward_history) / len(self.reward_history)
            recent = self.reward_history[-50:]
            recent_reward = sum(recent) / len(recent)
            cumulative_reward = sum(self.reward_history)
        else:
            mean_reward = 0.0
            recent_reward = 0.0
            cumulative_reward = 0.0

        counts: Dict[str, int] = {}
        for a in self.action_history:
            counts[a] = counts.get(a, 0) + 1

        total_actions = max(1, sum(counts.values()))

        return {
            "q_table_states": len(self.q_table),
            "q_updates": self.update_count,
            "q_decisions": self.decision_count,
            "q_mean_reward": mean_reward,
            "q_recent_reward": recent_reward,
            "q_cumulative_reward": cumulative_reward,
            "q_epsilon_final": self.epsilon,
            "q_unique_actions": len(counts),

            # Raw action counts
            "q_ahbn_base": counts.get("ahbn_base", 0),
            "q_more_structured": counts.get("more_structured", 0),
            "q_more_gossip": counts.get("more_gossip", 0),
            "q_duplicate_suppression": counts.get("duplicate_suppression", 0),
            "q_recovery_push": counts.get("recovery_push", 0),
            "q_resource_conservative": counts.get("resource_conservative", 0),

            # Action percentages
            "q_pct_ahbn_base": counts.get("ahbn_base", 0) / total_actions,
            "q_pct_more_structured": counts.get("more_structured", 0) / total_actions,
            "q_pct_more_gossip": counts.get("more_gossip", 0) / total_actions,
            "q_pct_duplicate_suppression": counts.get("duplicate_suppression", 0) / total_actions,
            "q_pct_recovery_push": counts.get("recovery_push", 0) / total_actions,
            "q_pct_resource_conservative": counts.get("resource_conservative", 0) / total_actions,
        }