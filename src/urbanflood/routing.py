"""SYNTHETIC routing module (Phase 4).

Pure functions for risk-aware routing: depth-threshold closures, penalised
impedance, and Dijkstra shortest paths on synthetic road graphs. Imports only
numpy, networkx, dataclasses, typing, collections.abc, and __future__ -- no
urbanflood imports at all. All inputs are plain arrays/scalars; nothing takes
an environment object or a cell-level array. Results are synthetic recovery
testing only; no route is ever described as "safe".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class RouteResult:
    """Result of routing one OD pair.

    reachable: whether a path exists under the policy.
    edge_ids: edge ids traversed, in order (empty if unreachable).
    node_path: node indices traversed, in order (empty if unreachable).
    total_cost: summed cost along the path (inf if unreachable).
    """

    reachable: bool
    edge_ids: tuple[int, ...]
    node_path: tuple[int, ...]
    total_cost: float


def effective_edge_depth(
    peak_samples: np.ndarray,
    surface_samples: np.ndarray,
    is_bridge: np.ndarray,
    deck_elevation_m: np.ndarray,
) -> np.ndarray:
    """Effective edge depth d_e in metres.

    Non-bridge edges: max over the three peak samples (start, midpoint, end).
    Bridge edges: max(peak_start, peak_end, max(0, surface_mid - deck)).

    peak_samples: (n_edges, 3) peak depth samples at start/mid/end.
    surface_samples: (n_edges, 3) peak water-surface elevation samples at
        start/mid/end (elevation_m + H_peak sampled with the same rule).
    is_bridge: bool (n_edges,), True for bridge edges.
    deck_elevation_m: float (n_edges,), bridge deck elevation above MSL;
        NaN for non-bridge edges.

    Returns a fresh (n_edges,) float64 array. Never emits NaN warnings for
    non-bridge edges (bridge-only computation is masked).
    """
    peak_samples = np.asarray(peak_samples, dtype=np.float64)
    surface_samples = np.asarray(surface_samples, dtype=np.float64)
    is_bridge = np.asarray(is_bridge, dtype=bool)
    deck_elevation_m = np.asarray(deck_elevation_m, dtype=np.float64)

    n_edges = peak_samples.shape[0]
    if surface_samples.shape != (n_edges, 3):
        raise ValueError(
            "surface_samples shape "
            f"{surface_samples.shape} does not match (n_edges, 3) "
            f"= ({n_edges}, 3)"
        )
    if is_bridge.shape != (n_edges,):
        raise ValueError(
            f"is_bridge shape {is_bridge.shape} does not match (n_edges,) = "
            f"({n_edges},)"
        )
    if deck_elevation_m.shape != (n_edges,):
        raise ValueError(
            f"deck_elevation_m shape {deck_elevation_m.shape} does not match "
            f"(n_edges,) = ({n_edges},)"
        )
    if not np.all(np.isfinite(peak_samples)):
        raise ValueError("peak_samples must contain only finite values")
    if not np.all(np.isfinite(surface_samples)):
        raise ValueError("surface_samples must contain only finite values")

    d_e = np.full(n_edges, np.nan, dtype=np.float64)

    # Non-bridge edges: max over the three peak samples.
    non_bridge = ~is_bridge
    d_e[non_bridge] = np.max(peak_samples[non_bridge], axis=1)

    # Bridge edges: max(peak start, peak end, max(0, surface_mid - deck)).
    bridge = is_bridge
    if np.any(bridge):
        bridge_deck = deck_elevation_m[bridge]
        if not np.all(np.isfinite(bridge_deck)):
            raise ValueError(
                "deck_elevation_m must be finite for all bridge edges"
            )
        surface_mid = surface_samples[bridge, 1]
        clearance = np.maximum(0.0, surface_mid - bridge_deck)
        d_e[bridge] = np.maximum(
            peak_samples[bridge, 0],
            np.maximum(peak_samples[bridge, 2], clearance),
        )

    return d_e


def sample_od_pairs(
    node_rows: np.ndarray,
    node_cols: np.ndarray,
    cell_size_m: float,
    n_pairs: int,
    min_distance_m: float,
    seed: int,
) -> np.ndarray:
    """Deterministically sample OD pairs of node indices.

    node_rows: int (n_nodes,), row coordinate of each node.
    node_cols: int (n_cols,), column coordinate of each node.
    cell_size_m: float, grid cell size in metres.
    n_pairs: int, number of OD pairs to draw.
    min_distance_m: float, minimum straight-line distance between origin and
        destination (in metres).
    seed: int, RNG seed; deterministic for a fixed seed.

    Returns (n_pairs, 2) int array of node index pairs (origin, destination).
    Draw order is preserved; duplicates (including reversed pairs) are
    rejected. Raises RuntimeError if not enough valid pairs after a bounded
    number of attempts. Independent of any depth array by signature.
    """
    node_rows = np.asarray(node_rows, dtype=np.int64).ravel()
    node_cols = np.asarray(node_cols, dtype=np.int64).ravel()
    n_nodes = int(node_rows.shape[0])
    if n_nodes < 2:
        raise ValueError("need at least 2 nodes to sample OD pairs")

    rng = np.random.default_rng(seed)
    min_dist_cells = min_distance_m / float(cell_size_m)

    chosen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    max_attempts = max(1000, 50 * n_pairs)
    attempts = 0

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in chosen:
            continue
        dr = float(node_rows[a] - node_rows[b])
        dc = float(node_cols[a] - node_cols[b])
        dist_m = float(np.hypot(dr, dc)) * cell_size_m
        if dist_m < min_distance_m:
            continue
        chosen.add(key)
        pairs.append((a, b))

    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"could not sample {n_pairs} valid OD pairs after {attempts} "
            f"attempts (only {len(pairs)} found); increase n_nodes or reduce "
            f"min_distance_m"
        )

    return np.array(pairs, dtype=np.int64)


def policy_costs(
    edge_time_s: np.ndarray,
    d_e: np.ndarray,
    closure_threshold_m: float | None,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge travel cost and open/closed mask.

    edge_time_s: float (n_edges,), free-flow travel time in seconds.
    d_e: float (n_edges,), effective edge depth in metres.
    closure_threshold_m: float or None. If None (policy P0), no closure and
        lam is ignored; cost = edge_time_s for all edges.
    lam: float, penalised impedance weight (lambda >= 0).

    Returns (cost, open_mask):
    - cost: float (n_edges,), edge_time_s * (1 + lam * d_e / threshold) for
      open edges, np.inf for closed edges.
    - open_mask: bool (n_edges,), True where d_e < threshold (all True if
      threshold is None).

    Raises ValueError if lam < 0 or (threshold is not None and
    threshold <= 0).
    """
    edge_time_s = np.asarray(edge_time_s, dtype=np.float64)
    d_e = np.asarray(d_e, dtype=np.float64)

    if edge_time_s.shape != d_e.shape:
        raise ValueError(
            "edge_time_s shape "
            f"{edge_time_s.shape} does not match d_e shape {d_e.shape}"
        )
    if lam < 0.0:
        raise ValueError("lambda must be non-negative")

    n_edges = int(edge_time_s.shape[0])
    open_mask = np.ones(n_edges, dtype=bool)
    cost = edge_time_s.copy()

    if closure_threshold_m is not None:
        if closure_threshold_m <= 0.0:
            raise ValueError("closure_threshold_m must be positive")
        open_mask = d_e < closure_threshold_m
        cost = np.where(
            open_mask,
            edge_time_s * (1.0 + lam * d_e / closure_threshold_m),
            np.inf,
        )

    return cost, open_mask


def solve_routes(
    n_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    cost: np.ndarray,
    open_mask: np.ndarray,
    od_pairs: np.ndarray,
) -> list[RouteResult]:
    """Run Dijkstra per OD pair over the open-edge graph.

    Builds ONE networkx Graph with all n_nodes added in index order, then adds
    open edges in ascending edge id order with weight=cost and attribute
    edge_id. Parallel edges keep the lower-cost one (ties: lower edge id).
    Uses nx.dijkstra_path with weight; catches nx.NetworkXNoPath ->
    reachable False with empty tuples and total_cost inf. Recovers edge ids
    along the node path from an edge-lookup dict. Deterministic.

    od_pairs: (n_pairs, 2) int array of (origin, destination) node indices.
    Returns one RouteResult per OD pair, in input order.
    """
    edge_u = np.asarray(edge_u, dtype=np.int64).ravel()
    edge_v = np.asarray(edge_v, dtype=np.int64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    open_mask = np.asarray(open_mask, dtype=bool).ravel()
    od_pairs = np.asarray(od_pairs, dtype=np.int64)

    n_edges = int(cost.shape[0])
    if edge_u.shape != (n_edges,) or edge_v.shape != (n_edges,):
        raise ValueError("edge_u and edge_v must match cost shape (n_edges,)")
    if open_mask.shape != (n_edges,):
        raise ValueError("open_mask must match cost shape (n_edges,)")

    graph = nx.Graph()
    graph.add_nodes_from(range(int(n_nodes)))

    # Edge lookup: (min, max) node pair -> (edge_id, cost). Keep lowest cost;
    # ties resolved by lower edge id (edges are processed in ascending id
    # order, so the first-seen lower-cost edge wins; equal-cost later edges
    # are skipped).
    edge_lookup: dict[tuple[int, int], tuple[int, float]] = {}
    for eid in range(n_edges):
        if not open_mask[eid]:
            continue
        u = int(edge_u[eid])
        v = int(edge_v[eid])
        key = (u, v) if u < v else (v, u)
        c = float(cost[eid])
        if key not in edge_lookup or c < edge_lookup[key][1]:
            edge_lookup[key] = (eid, c)

    for (u, v), (eid, c) in edge_lookup.items():
        graph.add_edge(u, v, weight=c, edge_id=eid)

    results: list[RouteResult] = []
    for pair in od_pairs:
        src = int(pair[0])
        dst = int(pair[1])
        try:
            node_path = list(nx.dijkstra_path(graph, src, dst, weight="weight"))
        except nx.NetworkXNoPath:
            results.append(
                RouteResult(
                    reachable=False,
                    edge_ids=(),
                    node_path=(),
                    total_cost=float("inf"),
                )
            )
            continue

        edge_ids: list[int] = []
        total_cost = 0.0
        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i + 1]
            key = (u, v) if u < v else (v, u)
            eid, c = edge_lookup[key]
            edge_ids.append(eid)
            total_cost += c

        results.append(
            RouteResult(
                reachable=True,
                edge_ids=tuple(edge_ids),
                node_path=tuple(node_path),
                total_cost=float(total_cost),
            )
        )

    return results


def route_metrics(
    edge_ids: tuple[int, ...],
    edge_length_m: np.ndarray,
    edge_time_s: np.ndarray,
    d_e: np.ndarray,
    label_threshold_m: float,
    closure_threshold_m: float | None,
) -> dict[str, Any]:
    """Per-route summary metrics.

    edge_ids: tuple of edge indices traversed (empty for unreachable route).
    edge_length_m: float (n_edges,), edge length in metres.
    edge_time_s: float (n_edges,), free-flow travel time in seconds.
    d_e: float (n_edges,), effective edge depth in metres.
    label_threshold_m: float, depth label threshold in metres.
    closure_threshold_m: float or None; if None, n_edges_ge_closure is 0.

    Returns dict with:
    - n_edges: number of edges on the route.
    - length_m: sum of edge_length_m along the route (metres).
    - free_flow_time_s: sum of edge_time_s along the route (seconds).
    - max_d_e: maximum effective edge depth along the route (metres).
    - mean_d_e_length_weighted: length-weighted mean d_e (metres).
    - share_length_ge_label: fraction of route length where d_e >= label
      threshold.
    - n_edges_ge_closure: number of route edges with d_e >= closure threshold
        (0 if closure_threshold_m is None).

    For an empty route returns zeros/NaN consistently: n_edges=0,
    length_m=0.0, free_flow_time_s=0.0, max_d_e=NaN,
    mean_d_e_length_weighted=NaN, share_length_ge_label=NaN,
    n_edges_ge_closure=0.
    """
    edge_length_m = np.asarray(edge_length_m, dtype=np.float64).ravel()
    edge_time_s = np.asarray(edge_time_s, dtype=np.float64).ravel()
    d_e = np.asarray(d_e, dtype=np.float64).ravel()

    result: dict[str, Any] = {
        "n_edges": 0,
        "length_m": 0.0,
        "free_flow_time_s": 0.0,
        "max_d_e": float("nan"),
        "mean_d_e_length_weighted": float("nan"),
        "share_length_ge_label": float("nan"),
        "n_edges_ge_closure": 0,
    }

    if len(edge_ids) == 0:
        return result

    lengths = edge_length_m[list(edge_ids)]
    times = edge_time_s[list(edge_ids)]
    depths = d_e[list(edge_ids)]

    total_length = float(np.sum(lengths))
    total_time = float(np.sum(times))
    max_depth = float(np.max(depths))

    if total_length > 0.0:
        mean_depth = float(np.sum(depths * lengths) / total_length)
        label_length = float(
            np.sum(lengths[depths >= label_threshold_m])
        )
        share_label = label_length / total_length
    else:
        mean_depth = float("nan")
        share_label = float("nan")

    if closure_threshold_m is not None:
        n_ge_closure = int(np.sum(depths >= closure_threshold_m))
    else:
        n_ge_closure = 0

    result["n_edges"] = int(len(edge_ids))
    result["length_m"] = total_length
    result["free_flow_time_s"] = total_time
    result["max_d_e"] = max_depth
    result["mean_d_e_length_weighted"] = mean_depth
    result["share_length_ge_label"] = share_label
    result["n_edges_ge_closure"] = n_ge_closure

    return result
