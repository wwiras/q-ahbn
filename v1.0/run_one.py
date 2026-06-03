from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.topology import assign_static_clusters, build_nodes_from_graph, get_or_build_topology


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


def build_ahbn_strategy(cfg: dict, fanout_override: int | None = None) -> AHBNStrategy:
    ahbn_cfg = cfg.get("ahbn", {})

    default_fanout = (
        fanout_override
        if fanout_override is not None
        else ahbn_cfg.get("default_fanout", cfg.get("fanout", 3))
    )

    return AHBNStrategy(
        default_fanout=default_fanout,
        adaptive_fanout=ahbn_cfg.get("adaptive_fanout", False),
        hybrid_mode=ahbn_cfg.get("hybrid_mode", True),
        use_tau_gate=ahbn_cfg.get("use_tau_gate", True),
        min_cluster_targets=ahbn_cfg.get("min_cluster_targets", 1),
    )


def build_simulation_from_config(cfg: dict, strategy_name: str):
    seed = cfg["seed"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    message_source = cfg.get("message_source", 0)

    graph = get_or_build_topology(
        topology_type=topology_type,
        num_nodes=num_nodes,
        seed=seed,
        use_cache=use_cache,
        edge_prob=cfg.get("edge_prob"),
        ba_m=cfg.get("ba_m"),
    )
    nodes = build_nodes_from_graph(graph)

    cluster_manager = None
    controller = None
    ch_overload_factor = cfg.get("ch_overload_factor", 1.0)

    if strategy_name == "gossip":
        fanout = cfg.get("fanout", 3)
        strategy = GossipStrategy(fanout=fanout)

    elif strategy_name == "cluster":
        num_clusters = cfg.get("num_clusters", 4)
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
        strategy = ClusterStrategy()

    elif strategy_name == "ahbn":
        num_clusters = cfg.get("num_clusters", 4)
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
        controller = AHBNController(build_ahbn_params(cfg))
        strategy = build_ahbn_strategy(cfg, fanout_override=cfg.get("fanout"))

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=cluster_manager,
        controller=controller,
        ch_overload_factor=ch_overload_factor,
    )
    return sim, message_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", required=True, choices=["gossip", "cluster", "ahbn"])
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    sim, source_id = build_simulation_from_config(cfg, args.strategy)

    sim.inject_message(source_id=source_id, message_id="m1")
    sim.run()

    summary = sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))
    print(f"Strategy: {args.strategy}")
    print(summary)


if __name__ == "__main__":
    main()