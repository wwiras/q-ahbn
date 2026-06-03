from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AHBNParams:
    ewma_alpha: float = 0.25
    d0: float = 0.20
    u0: float = 4.0
    l0: float = 2.2
    rho0: float = 0.0
    deg0: float = 8.0
    ov0: float = 0.25
    r0: float = 0.35
    c0: float = 0.35
    a_dup: float = -2.2
    a_load: float = -1.6
    a_lat: float = 1.1
    a_churn: float = 0.0
    a_deg: float = -0.3
    a_ov: float = -1.0
    a_red: float = -1.6
    a_cap: float = -2.0
    b_degree: float = 0.25
    b_overlap: float = 0.75
    min_fanout: int = 1
    max_fanout: int = 5
    mode_threshold: float = 0.55
    fanout_dup_penalty: float = 2.5
    fanout_load_penalty: float = 1.0
    fanout_lat_reward: float = 0.4
    fanout_red_penalty: float = 2.0
    fanout_cap_penalty: float = 1.6
    tau_max: float = 0.85
    tau_min: float = 0.25
    tau_dup_penalty: float = 1.2
    tau_red_penalty: float = 1.6
    tau_cap_penalty: float = 0.9
    min_weight: float = 0.25
    max_weight: float = 0.75
    weight_center_pull: float = 0.60
    mode_hysteresis: float = 0.06


class AHBNController:
    def __init__(self, params: AHBNParams) -> None:
        self.params = params
        self.degree_ref = params.deg0
        self.b_degree = params.b_degree
        self.b_overlap = params.b_overlap

    def ewma(self, old: float, new: float) -> float:
        a = self.params.ewma_alpha
        return a * new + (1.0 - a) * old

    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def update_metrics(
        self,
        state,
        duplicate_ratio: float,
        load_proxy: float,
        latency_proxy: float,
        churn_proxy: float = 0.0,
        degree_proxy: float = 0.0,
        overlap_proxy: float = 0.0,
        redundancy_proxy: float = 0.0,
        capacity_proxy: float = 0.0,
    ) -> None:
        state.d_hat = self.ewma(state.d_hat, duplicate_ratio)
        state.u_hat = self.ewma(state.u_hat, load_proxy)
        state.l_hat = self.ewma(state.l_hat, latency_proxy)
        state.rho_hat = self.ewma(state.rho_hat, churn_proxy)
        state.deg_hat = self.ewma(state.deg_hat, degree_proxy)
        state.ov_hat = self.ewma(state.ov_hat, overlap_proxy)

        if redundancy_proxy <= 0.0:
            redundancy_proxy = (
                self.params.b_overlap * overlap_proxy
                + self.params.b_degree * (degree_proxy / max(1.0, self.params.deg0))
            )
        state.r_hat = self.ewma(state.r_hat, redundancy_proxy)
        state.c_hat = self.ewma(getattr(state, "c_hat", 0.0), capacity_proxy)

    def compute_weight(self, state) -> float:
        p = self.params
        x_struct = (
            p.a_lat * (state.l_hat - p.l0)
            + p.a_dup * (state.d_hat - p.d0)
            + p.a_load * (state.u_hat - p.u0)
            + p.a_deg * (state.deg_hat - p.deg0)
            + p.a_ov * (state.ov_hat - p.ov0)
            + p.a_red * (state.r_hat - p.r0)
            + p.a_cap * (state.c_hat - p.c0)
        )
        w_struct = self.sigmoid(x_struct)
        w_centered = 0.5 + p.weight_center_pull * (w_struct - 0.5)
        return self.clamp(w_centered, p.min_weight, p.max_weight)

    def compute_tau(self, state) -> float:
        p = self.params
        tau = p.tau_max * math.exp(
            -p.tau_dup_penalty * max(0.0, state.d_hat)
            -p.tau_red_penalty * max(0.0, state.r_hat)
            -p.tau_cap_penalty * max(0.0, state.c_hat - p.c0)
        )
        return self.clamp(tau, p.tau_min, p.tau_max)

    def decide_mode_and_fanout(self, state) -> None:
        p = self.params
        state.weight = self.compute_weight(state)

        upper = p.mode_threshold + p.mode_hysteresis
        lower = p.mode_threshold - p.mode_hysteresis

        if state.mode == "cluster":
            if state.weight >= upper:
                state.mode = "gossip"
        elif state.mode == "gossip":
            if state.weight <= lower:
                state.mode = "cluster"
        else:
            state.mode = "gossip" if state.weight >= p.mode_threshold else "cluster"

        raw = round(
            p.max_fanout
            - p.fanout_dup_penalty * state.d_hat
            - p.fanout_load_penalty * state.u_hat
            + p.fanout_lat_reward * state.l_hat
            - p.fanout_red_penalty * max(0.0, state.r_hat - p.r0)
            - p.fanout_cap_penalty * max(0.0, state.c_hat - p.c0)
        )
        
        # PATCH: extra duplicate suppression for strong nodes
        # if getattr(state, "capacity_score", 0.0) >= 0.8:
            # raw -= round(0.8 * state.d_hat)
        
        # PATCH: extra duplicate suppression for strong nodes
        # if getattr(state, "capacity_score", 0.0) >= 0.8 and state.d_hat > p.d0:
        #     raw -= 1
        
        # STRONG NODE DUPLICATE CONTROL (FINAL FIX)
        if getattr(state, "capacity_score", 0.0) >= 0.8:
            raw -= int(1 + 2 * state.d_hat)
        
        state.fanout = max(p.min_fanout, min(p.max_fanout, raw))
        state.tau = self.compute_tau(state)

    def snapshot_state(self, state) -> dict:
        return {
            "mode": state.mode,
            "weight": state.weight,
            "fanout": state.fanout,
            "tau": state.tau,
            "d_hat": state.d_hat,
            "u_hat": state.u_hat,
            "l_hat": state.l_hat,
            "rho_hat": state.rho_hat,
            "deg_hat": state.deg_hat,
            "ov_hat": state.ov_hat,
            "r_hat": state.r_hat,
            "c_hat": getattr(state, "c_hat", 0.0),
        }
