from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.control_exp11 import AHBNController as AHBNControllerExp11
from ahbn.control_exp11 import AHBNParams as AHBNParamsExp11
from ahbn.control_exp12 import AHBNController as AHBNControllerExp12
from ahbn.control_exp12 import AHBNParams as AHBNParamsExp12
from ahbn.churn_manager import ChurnManager
from ahbn.failure_injector import FailureInjector
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.strategies.hybrid_fixed import HybridFixedStrategy
from ahbn.topology import (
    assign_mixed_resources,
    assign_static_clusters,
    build_nodes_from_graph,
    get_or_build_topology,
)
from ahbn.utils import ResultRow, save_results_csv, save_adaptive_trace_csv


def build_ahbn_params(cfg: dict) -> AHBNParams:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParams(
        ewma_alpha=ahbn_cfg.get("ewma_alpha", 0.3),
        d0=ahbn_cfg.get("d0", 0.2),
        u0=ahbn_cfg.get("u0", 5.0),
        l0=ahbn_cfg.get("l0", 2.0),
        rho0=ahbn_cfg.get("rho0", 0.1),
        deg0=ahbn_cfg.get("deg0", 8.0),
        ov0=ahbn_cfg.get("ov0", 0.25),
        r0=ahbn_cfg.get("r0", 0.35),
        a_dup=ahbn_cfg.get("a_dup", -2.0),
        a_load=ahbn_cfg.get("a_load", -1.5),
        a_lat=ahbn_cfg.get("a_lat", 1.5),
        a_churn=ahbn_cfg.get("a_churn", 1.0),
        a_deg=ahbn_cfg.get("a_deg", -0.4),
        a_ov=ahbn_cfg.get("a_ov", -1.2),
        a_red=ahbn_cfg.get("a_red", -1.8),
        b_degree=ahbn_cfg.get("b_degree", 0.25),
        b_overlap=ahbn_cfg.get("b_overlap", 0.75),
        min_fanout=ahbn_cfg.get("min_fanout", 1),
        max_fanout=ahbn_cfg.get("max_fanout", 6),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.5),
        fanout_dup_penalty=ahbn_cfg.get("fanout_dup_penalty", 2.0),
        fanout_load_penalty=ahbn_cfg.get("fanout_load_penalty", 0.5),
        fanout_lat_reward=ahbn_cfg.get("fanout_lat_reward", 0.8),
        fanout_red_penalty=ahbn_cfg.get("fanout_red_penalty", 1.5),
        tau_max=ahbn_cfg.get("tau_max", 0.90),
        tau_min=ahbn_cfg.get("tau_min", 0.25),
        tau_dup_penalty=ahbn_cfg.get("tau_dup_penalty", 1.0),
        tau_red_penalty=ahbn_cfg.get("tau_red_penalty", 1.5),
        min_weight=ahbn_cfg.get("min_weight", 0.20),
        max_weight=ahbn_cfg.get("max_weight", 0.80),
    )


def build_ahbn_params_exp11(cfg: dict) -> AHBNParamsExp11:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParamsExp11(
        ewma_alpha=ahbn_cfg.get("ewma_alpha", 0.3),
        d0=ahbn_cfg.get("d0", 0.2),
        u0=ahbn_cfg.get("u0", 5.0),
        l0=ahbn_cfg.get("l0", 2.0),
        rho0=ahbn_cfg.get("rho0", 0.1),
        deg0=ahbn_cfg.get("deg0", 8.0),
        ov0=ahbn_cfg.get("ov0", 0.25),
        r0=ahbn_cfg.get("r0", 0.35),
        a_dup=ahbn_cfg.get("a_dup", -2.0),
        a_load=ahbn_cfg.get("a_load", -1.5),
        a_lat=ahbn_cfg.get("a_lat", 1.5),
        a_churn=ahbn_cfg.get("a_churn", 1.0),
        a_deg=ahbn_cfg.get("a_deg", -0.4),
        a_ov=ahbn_cfg.get("a_ov", -1.2),
        a_red=ahbn_cfg.get("a_red", -1.8),
        b_degree=ahbn_cfg.get("b_degree", 0.25),
        b_overlap=ahbn_cfg.get("b_overlap", 0.75),
        min_fanout=ahbn_cfg.get("min_fanout", 1),
        max_fanout=ahbn_cfg.get("max_fanout", 6),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.5),
        fanout_dup_penalty=ahbn_cfg.get("fanout_dup_penalty", 2.0),
        fanout_load_penalty=ahbn_cfg.get("fanout_load_penalty", 0.5),
        fanout_lat_reward=ahbn_cfg.get("fanout_lat_reward", 0.8),
        fanout_red_penalty=ahbn_cfg.get("fanout_red_penalty", 1.5),
        tau_max=ahbn_cfg.get("tau_max", 0.90),
        tau_min=ahbn_cfg.get("tau_min", 0.25),
        tau_dup_penalty=ahbn_cfg.get("tau_dup_penalty", 1.0),
        tau_red_penalty=ahbn_cfg.get("tau_red_penalty", 1.5),
        min_weight=ahbn_cfg.get("min_weight", 0.20),
        max_weight=ahbn_cfg.get("max_weight", 0.80),
        weight_center_pull=ahbn_cfg.get("weight_center_pull", 0.70),
        churn_weight_cap=ahbn_cfg.get("churn_weight_cap", 0.08),
        mode_hysteresis=ahbn_cfg.get("mode_hysteresis", 0.06),
        tau_churn_boost=ahbn_cfg.get("tau_churn_boost", 0.60),
        fanout_churn_boost=ahbn_cfg.get("fanout_churn_boost", 1.0),
    )


def build_ahbn_params_exp12(cfg: dict) -> AHBNParamsExp12:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParamsExp12(
        ewma_alpha=ahbn_cfg.get("ewma_alpha", 0.25),
        d0=ahbn_cfg.get("d0", 0.20),
        u0=ahbn_cfg.get("u0", 4.0),
        l0=ahbn_cfg.get("l0", 2.2),
        rho0=ahbn_cfg.get("rho0", 0.0),
        deg0=ahbn_cfg.get("deg0", 8.0),
        ov0=ahbn_cfg.get("ov0", 0.25),
        r0=ahbn_cfg.get("r0", 0.35),
        c0=ahbn_cfg.get("c0", 0.35),
        a_dup=ahbn_cfg.get("a_dup", -2.2),
        a_load=ahbn_cfg.get("a_load", -1.6),
        a_lat=ahbn_cfg.get("a_lat", 1.1),
        a_churn=ahbn_cfg.get("a_churn", 0.0),
        a_deg=ahbn_cfg.get("a_deg", -0.3),
        a_ov=ahbn_cfg.get("a_ov", -1.0),
        a_red=ahbn_cfg.get("a_red", -1.6),
        a_cap=ahbn_cfg.get("a_cap", -2.0),
        b_degree=ahbn_cfg.get("b_degree", 0.25),
        b_overlap=ahbn_cfg.get("b_overlap", 0.75),
        min_fanout=ahbn_cfg.get("min_fanout", 1),
        max_fanout=ahbn_cfg.get("max_fanout", 5),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.55),
        fanout_dup_penalty=ahbn_cfg.get("fanout_dup_penalty", 2.5),
        fanout_load_penalty=ahbn_cfg.get("fanout_load_penalty", 1.0),
        fanout_lat_reward=ahbn_cfg.get("fanout_lat_reward", 0.4),
        fanout_red_penalty=ahbn_cfg.get("fanout_red_penalty", 2.0),
        fanout_cap_penalty=ahbn_cfg.get("fanout_cap_penalty", 1.6),
        tau_max=ahbn_cfg.get("tau_max", 0.85),
        tau_min=ahbn_cfg.get("tau_min", 0.25),
        tau_dup_penalty=ahbn_cfg.get("tau_dup_penalty", 1.2),
        tau_red_penalty=ahbn_cfg.get("tau_red_penalty", 1.6),
        tau_cap_penalty=ahbn_cfg.get("tau_cap_penalty", 0.9),
        min_weight=ahbn_cfg.get("min_weight", 0.25),
        max_weight=ahbn_cfg.get("max_weight", 0.75),
        weight_center_pull=ahbn_cfg.get("weight_center_pull", 0.60),
        mode_hysteresis=ahbn_cfg.get("mode_hysteresis", 0.06),
    )


def build_ahbn_strategy(cfg: dict, fanout: int | None = None) -> AHBNStrategy:
    ahbn_cfg = cfg.get("ahbn", {})
    default_fanout = fanout if fanout is not None else ahbn_cfg.get("default_fanout", 3)

    experiment_name = cfg.get("experiment", "")
    is_exp11 = experiment_name == "exp11"

    return AHBNStrategy(
        default_fanout=default_fanout,
        adaptive_fanout=ahbn_cfg.get("adaptive_fanout", False),
        hybrid_mode=ahbn_cfg.get("hybrid_mode", True),
        use_tau_gate=ahbn_cfg.get("use_tau_gate", True),
        min_cluster_targets=ahbn_cfg.get("min_cluster_targets", 1),

        # Exp11-only strategy shaping
        mode_sensitive_mix=ahbn_cfg.get("mode_sensitive_mix", is_exp11),
        cluster_mode_bias=ahbn_cfg.get("cluster_mode_bias", 0.75 if is_exp11 else 0.50),
        gossip_mode_bias=ahbn_cfg.get("gossip_mode_bias", 0.75 if is_exp11 else 0.50),
        preserve_cluster_path_under_tau=ahbn_cfg.get(
            "preserve_cluster_path_under_tau",
            is_exp11,
        ),
        cluster_reserve_in_gossip_mode=ahbn_cfg.get(
            "cluster_reserve_in_gossip_mode",
            1 if is_exp11 else 0,
        ),
        gossip_reserve_in_cluster_mode=ahbn_cfg.get(
            "gossip_reserve_in_cluster_mode",
            1 if is_exp11 else 0,
        ),
        resource_aware_targeting=ahbn_cfg.get("resource_aware_targeting", cfg.get("experiment", "") == "exp12"),
    )


def select_ahbn_controller(cfg: dict):
    experiment_name = cfg.get("experiment", "")

    if experiment_name == "exp11":
        return AHBNControllerExp11(build_ahbn_params_exp11(cfg))
    if experiment_name == "exp12":
        return AHBNControllerExp12(build_ahbn_params_exp12(cfg))
    return AHBNController(build_ahbn_params(cfg))


def run_single(
    cfg: dict,
    strategy_name: str,
    seed: int,
    topology_type: str,
    num_nodes: int,
    use_topology_cache: bool,
    base_delay: float,
    jitter: float,
    message_source: int,
    fanout: int | None = None,
    num_clusters: int | None = None,
    ch_overload_factor: float | None = None,
    edge_prob: float | None = None,
    ba_m: int | None = None,
    failure_mode: str | None = None,
    enable_adaptive_trace: bool = False,
    churn_rate: float | None = None,
    resource_scenario: str | None = None,
) -> dict:
    graph = get_or_build_topology(
        topology_type=topology_type,
        num_nodes=num_nodes,
        seed=seed,
        use_cache=use_topology_cache,
        edge_prob=edge_prob,
        ba_m=ba_m,
    )
    nodes = build_nodes_from_graph(graph)

    experiment_name = cfg.get("experiment", "")
    if experiment_name == "exp12":
        assign_mixed_resources(nodes, cfg, seed=seed, scenario_name=resource_scenario)

    cluster_manager = None
    controller = None

    if strategy_name == "gossip":
        strategy = GossipStrategy(fanout=fanout if fanout is not None else 3)

    elif strategy_name == "cluster":
        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
            resource_aware_heads=False,
        )
        strategy = ClusterStrategy()

    elif strategy_name == "ahbn":
        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
            resource_aware_heads=(experiment_name == "exp12"),
        )
        controller = select_ahbn_controller(cfg)
        strategy = build_ahbn_strategy(cfg, fanout=fanout)

    elif strategy_name == "hybrid_fixed":
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters or 4)
        strategy = HybridFixedStrategy(fanout=fanout if fanout is not None else 3)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    local_cfg = dict(cfg)
    if failure_mode is not None:
        local_failure = dict(cfg.get("failure", {}))
        local_failure["mode"] = failure_mode
        local_cfg["failure"] = local_failure

    if churn_rate is not None:
        local_churn = dict(cfg.get("churn", {}))
        local_churn["target_fraction"] = churn_rate
        local_cfg["churn"] = local_churn

    failure_injector = FailureInjector(local_cfg, seed=seed)
    churn_manager = ChurnManager(local_cfg, seed=seed)

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=cluster_manager,
        controller=controller,
        ch_overload_factor=ch_overload_factor if ch_overload_factor is not None else 1.0,
        failure_injector=failure_injector,
        churn_manager=churn_manager,
        experiment_name=cfg.get("experiment", "unknown"),
        strategy_name=strategy_name,
        scenario_tag=(
            resource_scenario
            if resource_scenario is not None
            else (failure_mode if failure_mode is not None else topology_type)
        ),
        enable_adaptive_trace=enable_adaptive_trace,
        resource_aware_heads=(experiment_name == "exp12" and strategy_name == "ahbn"),
    )

    sim.inject_message(source_id=message_source, message_id="m1")
    sim.run()

    summary = sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))
    summary.update(sim.get_resource_metrics())
    if enable_adaptive_trace:
        summary["adaptive_trace_rows"] = sim.adaptive_trace_rows
    return summary


def exp07(cfg: dict) -> list[ResultRow]:
    rows: list[ResultRow] = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    fanouts = cfg["fanouts"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "ahbn"])

    for fanout in fanouts:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                )
                rows.append(
                    ResultRow(
                        experiment="exp07",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type=topology_type,
                        topology_param=edge_prob if topology_type == "er" else ba_m,
                        fanout=fanout,
                        num_clusters=num_clusters,
                        ch_overload_factor=None,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )
    return rows


def exp08(cfg: dict) -> list[ResultRow]:
    rows: list[ResultRow] = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    overload_values = cfg["ch_overload_factor"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    num_clusters = cfg["num_clusters"]

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["cluster", "ahbn"])

    for overload in overload_values:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    num_clusters=num_clusters,
                    ch_overload_factor=overload,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                )
                rows.append(
                    ResultRow(
                        experiment="exp08",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type=topology_type,
                        topology_param=edge_prob if topology_type == "er" else ba_m,
                        fanout=None,
                        num_clusters=num_clusters,
                        ch_overload_factor=overload,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )
    return rows


def exp09(cfg: dict) -> list[ResultRow]:
    rows: list[ResultRow] = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    if topology_type != "er":
        raise ValueError("Exp09 density sweep is intended for ER topology.")

    edge_probs = cfg["edge_probs"]
    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])

    for edge_prob in edge_probs:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type="er",
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                )
                rows.append(
                    ResultRow(
                        experiment="exp09",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type="er",
                        topology_param=edge_prob,
                        fanout=fanout if strategy_name != "cluster" else None,
                        num_clusters=num_clusters,
                        ch_overload_factor=None,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )
    return rows


def exp10(cfg: dict) -> tuple[list[dict], list]:
    rows: list[dict] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])
    failure_modes = cfg.get("failure_modes", ["node_failure", "ch_failure", "overload"])

    for failure_mode in failure_modes:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    failure_mode=failure_mode,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                )

                rows.append(
                    {
                        "experiment": "exp10",
                        "strategy": strategy_name,
                        "seed": seed,
                        "num_nodes": num_nodes,
                        "topology_type": topology_type,
                        "topology_param": edge_prob if topology_type == "er" else ba_m,
                        "fanout": fanout if strategy_name != "cluster" else None,
                        "num_clusters": num_clusters,
                        "ch_overload_factor": None,
                        "failure_mode": summary["failure_mode"],
                        "failed_node_id": summary["failed_node_id"],
                        "delivery_ratio": summary["delivery_ratio"],
                        "propagation_delay": summary["propagation_delay"],
                        "duplicates": summary["duplicates"],
                        "total_forwards": summary["total_forwards"],
                        "recovery_time": summary["recovery_time"],
                    }
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp11(cfg: dict) -> tuple[list[dict], list]:
    rows: list[dict] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])
    churn_rates = cfg.get("churn_rates", [0.0, 0.05, 0.10, 0.20, 0.30])

    for churn_rate in churn_rates:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    churn_rate=churn_rate,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                )

                rows.append(
                    {
                        "experiment": "exp11",
                        "strategy": strategy_name,
                        "seed": seed,
                        "num_nodes": num_nodes,
                        "topology_type": topology_type,
                        "topology_param": edge_prob if topology_type == "er" else ba_m,
                        "fanout": fanout if strategy_name != "cluster" else None,
                        "num_clusters": num_clusters,
                        "churn_rate": churn_rate,
                        "delivery_ratio": summary["delivery_ratio"],
                        "propagation_delay": summary["propagation_delay"],
                        "duplicates": summary["duplicates"],
                        "total_forwards": summary["total_forwards"],
                        "churn_event_count": summary["churn_event_count"],
                        "churn_leave_count": summary["churn_leave_count"],
                        "churn_join_count": summary["churn_join_count"],
                        "cluster_repair_count": summary["cluster_repair_count"],
                        "mode_switch_count": summary["mode_switch_count"],
                        "fanout_change_count": summary["fanout_change_count"],
                        "adaptation_event_count": summary["adaptation_event_count"],
                        "adaptation_rate": summary["adaptation_rate"],
                    }
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp12(cfg: dict) -> tuple[list[dict], list]:
    rows: list[dict] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])
    resource_scenarios = cfg.get("resource_scenarios", ["balanced", "weak_heavy"])

    for resource_scenario in resource_scenarios:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    resource_scenario=resource_scenario,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                )

                rows.append(
                    {
                        "experiment": "exp12",
                        "strategy": strategy_name,
                        "seed": seed,
                        "num_nodes": num_nodes,
                        "topology_type": topology_type,
                        "topology_param": edge_prob if topology_type == "er" else ba_m,
                        "fanout": fanout if strategy_name != "cluster" else None,
                        "num_clusters": num_clusters,
                        "resource_scenario": resource_scenario,
                        "delivery_ratio": summary["delivery_ratio"],
                        "propagation_delay": summary["propagation_delay"],
                        "duplicates": summary["duplicates"],
                        "total_forwards": summary["total_forwards"],
                        "max_normalized_load": summary["max_normalized_load"],
                        "load_balance_cv": summary["load_balance_cv"],
                        "strong_forward_share": summary["strong_forward_share"],
                        "medium_forward_share": summary["medium_forward_share"],
                        "weak_forward_share": summary["weak_forward_share"],
                    }
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    experiment = cfg["experiment"]

    if experiment == "exp07":
        rows = exp07(cfg)
        path = save_results_csv(rows, "outputs/csv/exp07_results.csv")
        print(f"Saved {path}")

    elif experiment == "exp08":
        rows = exp08(cfg)
        path = save_results_csv(rows, "outputs/csv/exp08_results.csv")
        print(f"Saved {path}")

    elif experiment == "exp09":
        rows = exp09(cfg)
        path = save_results_csv(rows, "outputs/csv/exp09_results.csv")
        print(f"Saved {path}")

    elif experiment == "exp10":
        import pandas as pd
        from pathlib import Path
        from ahbn.utils import current_timestamp

        rows, trace_rows = exp10(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        ts = current_timestamp()
        path = out / f"exp10_results_{ts}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp10_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp11":
        import pandas as pd
        from pathlib import Path
        from ahbn.utils import current_timestamp

        rows, trace_rows = exp11(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        ts = current_timestamp()
        path = out / f"exp11_results_{ts}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp11_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp12":
        import pandas as pd
        from pathlib import Path
        from ahbn.utils import current_timestamp

        rows, trace_rows = exp12(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        ts = current_timestamp()
        path = out / f"exp12_results_{ts}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp12_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    else:
        raise ValueError(f"Unsupported experiment: {experiment}")


if __name__ == "__main__":
    main()