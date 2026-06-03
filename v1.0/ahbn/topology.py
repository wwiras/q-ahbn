from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import networkx as nx

from ahbn.cluster import ClusterManager
from ahbn.node import Node


TOPOLOGY_CACHE_DIR = Path("outputs/topologies")


def ensure_cache_dir() -> None:
    TOPOLOGY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def relabel_graph_compact(g: nx.Graph) -> nx.Graph:
    mapping = {old: new for new, old in enumerate(sorted(g.nodes()))}
    return nx.relabel_nodes(g, mapping)


def largest_connected_subgraph(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        raise ValueError("Generated graph has zero nodes.")
    if nx.is_connected(g):
        return g.copy()
    largest = max(nx.connected_components(g), key=len)
    return g.subgraph(largest).copy()


def build_er_graph(num_nodes: int, edge_prob: float, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    graph_seed = rng.randint(0, 10_000_000)
    g = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=graph_seed)
    g = largest_connected_subgraph(g)
    g = relabel_graph_compact(g)
    return g


def build_ba_graph(num_nodes: int, m: int, seed: int) -> nx.Graph:
    if m <= 0:
        raise ValueError("BA parameter m must be > 0")
    if m >= num_nodes:
        raise ValueError("BA parameter m must be less than num_nodes")

    rng = random.Random(seed)
    graph_seed = rng.randint(0, 10_000_000)
    g = nx.barabasi_albert_graph(num_nodes, m, seed=graph_seed)
    g = largest_connected_subgraph(g)
    g = relabel_graph_compact(g)
    return g


def topology_cache_path(topology_type: str, num_nodes: int, param_name: str, param_value: float | int, seed: int) -> Path:
    filename = f"{topology_type}_n{num_nodes}_{param_name}{param_value}_seed{seed}.json"
    return TOPOLOGY_CACHE_DIR / filename


def save_graph_to_cache(graph: nx.Graph, path: Path) -> None:
    data = {
        "nodes": sorted(graph.nodes()),
        "edges": sorted([sorted([u, v]) for u, v in graph.edges()]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph_from_cache(path: Path) -> nx.Graph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g = nx.Graph()
    g.add_nodes_from(data["nodes"])
    g.add_edges_from([tuple(edge) for edge in data["edges"]])
    return g


def get_or_build_topology(
    topology_type: str,
    num_nodes: int,
    seed: int,
    use_cache: bool = True,
    edge_prob: float | None = None,
    ba_m: int | None = None,
) -> nx.Graph:
    ensure_cache_dir()

    if topology_type == "er":
        if edge_prob is None:
            raise ValueError("edge_prob is required for ER topology")
        cache_path = topology_cache_path("er", num_nodes, "p", edge_prob, seed)

        if use_cache and cache_path.exists():
            return load_graph_from_cache(cache_path)

        graph = build_er_graph(num_nodes=num_nodes, edge_prob=edge_prob, seed=seed)

        if use_cache:
            save_graph_to_cache(graph, cache_path)

        return graph

    if topology_type == "ba":
        if ba_m is None:
            raise ValueError("ba_m is required for BA topology")
        cache_path = topology_cache_path("ba", num_nodes, "m", ba_m, seed)

        if use_cache and cache_path.exists():
            return load_graph_from_cache(cache_path)

        graph = build_ba_graph(num_nodes=num_nodes, m=ba_m, seed=seed)

        if use_cache:
            save_graph_to_cache(graph, cache_path)

        return graph

    raise ValueError(f"Unsupported topology_type: {topology_type}")


def build_nodes_from_graph(graph: nx.Graph) -> Dict[int, Node]:
    nodes: Dict[int, Node] = {}
    for n in sorted(graph.nodes()):
        nodes[n] = Node(node_id=n, neighbors=list(sorted(graph.neighbors(n))))
    return nodes


def assign_mixed_resources(nodes: Dict[int, Node], cfg: dict, seed: int, scenario_name: str | None = None) -> None:
    resources_cfg = cfg.get("resources", {})
    classes_cfg = resources_cfg.get("classes", {})
    profiles_cfg = resources_cfg.get("profiles", {})

    if not classes_cfg:
        return

    if scenario_name is None:
        fractions = resources_cfg.get("fractions", {"strong": 0.2, "medium": 0.5, "weak": 0.3})
    else:
        fractions = profiles_cfg.get(scenario_name)
        if fractions is None:
            raise ValueError(f"Unknown resource scenario: {scenario_name}")

    node_ids = sorted(nodes.keys())
    rng = random.Random(seed)
    rng.shuffle(node_ids)

    total = len(node_ids)
    remaining_ids = node_ids[:]
    allocated: dict[str, list[int]] = {}
    classes = list(fractions.keys())

    for idx, cls_name in enumerate(classes):
        if idx == len(classes) - 1:
            selected = remaining_ids[:]
        else:
            count = int(round(float(fractions.get(cls_name, 0.0)) * total))
            count = max(0, min(count, len(remaining_ids)))
            selected = remaining_ids[:count]
            remaining_ids = remaining_ids[count:]
        allocated[cls_name] = selected

    for cls_name, ids in allocated.items():
        cls_cfg = classes_cfg.get(cls_name, {})
        processing_delay = float(cls_cfg.get("processing_delay", 0.0))
        capacity_score = float(cls_cfg.get("capacity_score", 1.0))
        for node_id in ids:
            node = nodes[node_id]
            node.resource_class = cls_name
            node.processing_delay = processing_delay
            node.capacity_score = capacity_score


def _select_cluster_head(member_ids: list[int], nodes: Dict[int, Node], resource_aware_heads: bool) -> int:
    if not resource_aware_heads:
        return min(member_ids)
    return max(
        member_ids,
        key=lambda nid: (
            nodes[nid].capacity_score,
            len(nodes[nid].neighbors),
            -nid,
        ),
    )


def assign_static_clusters(
    nodes: Dict[int, Node],
    num_clusters: int,
    resource_aware_heads: bool = False,
) -> ClusterManager:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be > 0")

    node_ids = sorted(nodes.keys())
    cluster_mgr = ClusterManager()

    for idx, node_id in enumerate(node_ids):
        cluster_id = idx % num_clusters
        nodes[node_id].cluster_id = cluster_id
        cluster_mgr.cluster_to_members.setdefault(cluster_id, []).append(node_id)

    for cluster_id, members in cluster_mgr.cluster_to_members.items():
        head_id = _select_cluster_head(members, nodes, resource_aware_heads)
        cluster_mgr.cluster_to_head[cluster_id] = head_id
        nodes[head_id].is_cluster_head = True

    cluster_ids = sorted(cluster_mgr.cluster_to_head.keys())
    for i in range(len(cluster_ids) - 1):
        left = cluster_mgr.cluster_to_head[cluster_ids[i]]
        right = cluster_mgr.cluster_to_head[cluster_ids[i + 1]]
        nodes[left].gateway_neighbors.append(right)
        nodes[right].gateway_neighbors.append(left)

    return cluster_mgr


def refresh_active_neighbors(nodes: Dict[int, Node]) -> None:
    for node in nodes.values():
        if not node.is_active:
            node.neighbors = []
            continue
        node.neighbors = [
            nbr_id
            for nbr_id in node.original_neighbors
            if nbr_id in nodes and nodes[nbr_id].is_active
        ]


def refresh_cluster_overlay(
    nodes: Dict[int, Node],
    cluster_mgr: ClusterManager | None,
    resource_aware_heads: bool = False,
) -> None:
    if cluster_mgr is None:
        return

    for node in nodes.values():
        node.is_cluster_head = False
        node.gateway_neighbors = []

    cluster_mgr.cluster_to_members = {}
    cluster_mgr.cluster_to_head = {}

    for node in nodes.values():
        if not node.is_active or node.cluster_id is None:
            continue
        cluster_mgr.cluster_to_members.setdefault(node.cluster_id, []).append(node.node_id)

    for cluster_id, members in cluster_mgr.cluster_to_members.items():
        members = sorted(members)
        cluster_mgr.cluster_to_members[cluster_id] = members
        if not members:
            continue
        head_id = _select_cluster_head(members, nodes, resource_aware_heads)
        cluster_mgr.cluster_to_head[cluster_id] = head_id
        nodes[head_id].is_cluster_head = True

    cluster_ids = sorted(cluster_mgr.cluster_to_head.keys())
    for i in range(len(cluster_ids) - 1):
        left = cluster_mgr.cluster_to_head[cluster_ids[i]]
        right = cluster_mgr.cluster_to_head[cluster_ids[i + 1]]
        nodes[left].gateway_neighbors.append(right)
        nodes[right].gateway_neighbors.append(left)


def repair_topology_after_churn(
    nodes: Dict[int, Node],
    cluster_mgr: ClusterManager | None,
    resource_aware_heads: bool = False,
) -> None:
    refresh_active_neighbors(nodes)
    refresh_cluster_overlay(nodes, cluster_mgr, resource_aware_heads=resource_aware_heads)
