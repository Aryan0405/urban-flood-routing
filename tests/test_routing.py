"""Tests for src/urbanflood/routing.py (Phase 4 routing module).

Fast tests only: toy data, no simulation, no experiment runner.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import time

import networkx as nx
import numpy as np
import pytest

from urbanflood.routing import (
    RouteResult,
    effective_edge_depth,
    policy_costs,
    route_metrics,
    sample_od_pairs,
    solve_routes,
)

# ---------------------------------------------------------------------------
# effective_edge_depth
# ---------------------------------------------------------------------------


def test_effective_edge_depth_non_bridge_max_of_three():
    peak = np.array([[1.0, 2.0, 3.0], [5.0, 0.0, 4.0]], dtype=np.float64)
    surface = np.zeros((2, 3), dtype=np.float64)
    is_bridge = np.array([False, False], dtype=bool)
    deck = np.full(2, np.nan, dtype=np.float64)
    d = effective_edge_depth(peak, surface, is_bridge, deck)
    assert d.shape == (2,)
    assert d[0] == 3.0
    assert d[1] == 5.0


def test_effective_edge_depth_bridge_deck_above_surface():
    # Deck above surface: clearance = 0, d_e = max(peak start, peak end).
    peak = np.array([[1.0, 5.0, 2.0]], dtype=np.float64)
    surface = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
    is_bridge = np.array([True], dtype=bool)
    deck = np.array([20.0], dtype=np.float64)  # deck above surface -> no clearance
    d = effective_edge_depth(peak, surface, is_bridge, deck)
    assert d.shape == (1,)
    assert d[0] == max(1.0, 2.0, 0.0)  # = 2.0


def test_effective_edge_depth_bridge_surface_above_deck():
    # Surface mid above deck by 0.4: clearance = 0.4.
    peak = np.array([[1.0, 5.0, 2.0]], dtype=np.float64)
    surface = np.array([[10.0, 10.4, 10.0]], dtype=np.float64)
    is_bridge = np.array([True], dtype=bool)
    deck = np.array([10.0], dtype=np.float64)
    d = effective_edge_depth(peak, surface, is_bridge, deck)
    assert d[0] == max(1.0, 2.0, 0.4)  # = 2.0 (peak end still largest)


def test_effective_edge_depth_bridge_clearance_dominates():
    # Clearance dominates: surface_mid - deck = 5.0 > peak samples.
    peak = np.array([[1.0, 5.0, 2.0]], dtype=np.float64)
    surface = np.array([[10.0, 15.0, 10.0]], dtype=np.float64)
    is_bridge = np.array([True], dtype=bool)
    deck = np.array([10.0], dtype=np.float64)
    d = effective_edge_depth(peak, surface, is_bridge, deck)
    assert d[0] == max(1.0, 2.0, 5.0)  # = 5.0


def test_effective_edge_depth_bridge_nan_deck_raises():
    peak = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    surface = np.zeros((1, 3), dtype=np.float64)
    is_bridge = np.array([True], dtype=bool)
    deck = np.array([np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="deck_elevation_m must be finite"):
        effective_edge_depth(peak, surface, is_bridge, deck)


def test_effective_edge_depth_non_bridge_nan_deck_no_raise():
    # Non-bridge edges must not raise or produce NaN even with NaN deck.
    peak = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    surface = np.zeros((1, 3), dtype=np.float64)
    is_bridge = np.array([False], dtype=bool)
    deck = np.array([np.nan], dtype=np.float64)
    d = effective_edge_depth(peak, surface, is_bridge, deck)
    assert d[0] == 3.0
    assert np.isfinite(d[0])


def test_effective_edge_depth_wrong_shape_raises():
    peak = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    surface = np.zeros((1, 2), dtype=np.float64)  # wrong shape
    is_bridge = np.array([False], dtype=bool)
    deck = np.full(1, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="surface_samples shape"):
        effective_edge_depth(peak, surface, is_bridge, deck)


def test_effective_edge_depth_non_finite_peak_raises():
    peak = np.array([[1.0, 2.0, np.nan]], dtype=np.float64)
    surface = np.zeros((1, 3), dtype=np.float64)
    is_bridge = np.array([False], dtype=bool)
    deck = np.full(1, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="peak_samples must contain only finite"):
        effective_edge_depth(peak, surface, is_bridge, deck)


# ---------------------------------------------------------------------------
# sample_od_pairs
# ---------------------------------------------------------------------------


def test_sample_od_pairs_deterministic():
    node_rows = np.arange(10, dtype=np.int64)
    node_cols = np.arange(10, dtype=np.int64)
    p1 = sample_od_pairs(node_rows, node_cols, 50.0, 5, 100.0, seed=42)
    p2 = sample_od_pairs(node_rows, node_cols, 50.0, 5, 100.0, seed=42)
    assert p1.shape == (5, 2)
    assert p1.dtype == np.int64
    np.testing.assert_array_equal(p1, p2)


def test_sample_od_pairs_different_seed():
    node_rows = np.arange(10, dtype=np.int64)
    node_cols = np.arange(10, dtype=np.int64)
    p1 = sample_od_pairs(node_rows, node_cols, 50.0, 5, 100.0, seed=1)
    p2 = sample_od_pairs(node_rows, node_cols, 50.0, 5, 100.0, seed=2)
    assert not np.array_equal(p1, p2)


def test_sample_od_pairs_no_self_loops():
    node_rows = np.arange(10, dtype=np.int64)
    node_cols = np.arange(10, dtype=np.int64)
    p = sample_od_pairs(node_rows, node_cols, 50.0, 20, 100.0, seed=7)
    assert np.all(p[:, 0] != p[:, 1])


def test_sample_od_pairs_min_distance():
    node_rows = np.arange(20, dtype=np.int64)
    node_cols = np.arange(20, dtype=np.int64)
    cell_size = 50.0
    min_dist = 500.0  # 10 cells
    p = sample_od_pairs(node_rows, node_cols, cell_size, 15, min_dist, seed=3)
    # Recompute distances independently.
    for a, b in p:
        dr = float(node_rows[a] - node_rows[b])
        dc = float(node_cols[a] - node_cols[b])
        dist = float(np.hypot(dr, dc)) * cell_size
        assert dist >= min_dist - 1e-9


def test_sample_od_pairs_no_duplicates_including_reversed():
    node_rows = np.arange(20, dtype=np.int64)
    node_cols = np.arange(20, dtype=np.int64)
    p = sample_od_pairs(node_rows, node_cols, 50.0, 20, 100.0, seed=11)
    keys = set()
    for a, b in p:
        key = (int(a), int(b)) if a < b else (int(b), int(a))
        assert key not in keys
        keys.add(key)


def test_sample_od_pairs_shape_dtype():
    node_rows = np.arange(20, dtype=np.int64)
    node_cols = np.arange(20, dtype=np.int64)
    p = sample_od_pairs(node_rows, node_cols, 50.0, 10, 100.0, seed=5)
    assert p.shape == (10, 2)
    assert p.dtype == np.int64


def test_sample_od_pairs_impossible_raises():
    # 2 nodes only, min distance larger than the only possible distance.
    node_rows = np.array([0, 1], dtype=np.int64)
    node_cols = np.array([0, 0], dtype=np.int64)
    cell_size = 50.0  # only possible distance is 50 m
    with pytest.raises(RuntimeError, match="could not sample"):
        sample_od_pairs(node_rows, node_cols, cell_size, 5, 1000.0, seed=1)


def test_sample_od_pairs_no_depth_parameter():
    sig = inspect.signature(sample_od_pairs)
    names = set(sig.parameters.keys())
    forbidden = {
        "drainage", "outfall", "spill", "lake_spec", "initial_state",
        "simulator", "orchestrator", "h_final", "h_initial", "h_peak", "env",
    }
    for name in names:
        assert name not in forbidden, f"forbidden parameter {name!r}"
    # Also confirm no depth-like parameter at all.
    for name in names:
        assert "depth" not in name, f"depth-like parameter {name!r}"


# ---------------------------------------------------------------------------
# policy_costs
# ---------------------------------------------------------------------------


def test_policy_costs_p0_none():
    edge_time = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    d_e = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    cost, open_mask = policy_costs(edge_time, d_e, None, lam=2.0)
    np.testing.assert_array_equal(cost, edge_time)
    assert np.all(open_mask)


def test_policy_costs_threshold_closes_at_boundary():
    edge_time = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    d_e = np.array([0.1, 0.3, 0.5], dtype=np.float64)
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=1.0)
    # d_e == threshold -> closed.
    assert bool(open_mask[0]) is True   # 0.1 < 0.3
    assert bool(open_mask[1]) is False  # 0.3 == 0.3 -> closed
    assert bool(open_mask[2]) is False  # 0.5 > 0.3 -> closed
    assert np.isfinite(cost[0])
    assert np.isinf(cost[1])
    assert np.isinf(cost[2])


def test_policy_costs_formula_hand_checked():
    # edge_time=10, d_e=0.1, threshold=0.3, lam=1.0
    # cost = 10 * (1 + 1.0 * 0.1 / 0.3) = 10 * (1 + 0.3333...) = 13.333...
    edge_time = np.array([10.0], dtype=np.float64)
    d_e = np.array([0.1], dtype=np.float64)
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=1.0)
    expected = 10.0 * (1.0 + 1.0 * 0.1 / 0.3)
    assert bool(open_mask[0]) is True
    assert cost[0] == pytest.approx(expected, abs=1e-12)


def test_policy_costs_lam_zero():
    edge_time = np.array([10.0, 20.0], dtype=np.float64)
    d_e = np.array([0.5, 0.1], dtype=np.float64)
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=0.0)
    # lam=0 -> cost == time on open edges; d_e=0.5 closed, d_e=0.1 open.
    assert bool(open_mask[0]) is False
    assert bool(open_mask[1]) is True
    assert np.isinf(cost[0])
    assert cost[1] == 20.0


def test_policy_costs_bad_args():
    edge_time = np.array([10.0], dtype=np.float64)
    d_e = np.array([0.1], dtype=np.float64)
    with pytest.raises(ValueError, match="lambda must be non-negative"):
        policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=-0.1)
    with pytest.raises(ValueError, match="closure_threshold_m must be positive"):
        policy_costs(edge_time, d_e, closure_threshold_m=0.0, lam=1.0)
    with pytest.raises(ValueError, match="shape"):
        policy_costs(np.array([10.0, 20.0]), d_e, closure_threshold_m=0.3, lam=1.0)


# ---------------------------------------------------------------------------
# solve_routes
# ---------------------------------------------------------------------------


def _diamond_graph():
    # Nodes 0..3. Edges:
    #   0: 0-1, time=1.0, depth=0.05 (fast but shallow flood)
    #   1: 0-2, time=2.0, depth=0.0 (slow but dry)
    #   2: 1-3, time=1.0, depth=0.0
    #   3: 2-3, time=1.0, depth=0.0
    edge_u = np.array([0, 0, 1, 2], dtype=np.int64)
    edge_v = np.array([1, 2, 3, 3], dtype=np.int64)
    edge_time = np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float64)
    d_e = np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    return edge_u, edge_v, edge_time, d_e


def test_solve_routes_diamond_lam_zero_takes_fast_route():
    edge_u, edge_v, edge_time, d_e = _diamond_graph()
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=0.0)
    od = np.array([[0, 3]], dtype=np.int64)
    results = solve_routes(4, edge_u, edge_v, cost, open_mask, od)
    assert len(results) == 1
    r = results[0]
    assert r.reachable is True
    # Path 0-1-3 cost = 1+1 = 2; path 0-2-3 cost = 2+1 = 3. Fast route wins.
    assert r.total_cost == pytest.approx(2.0, abs=1e-12)
    assert r.node_path == (0, 1, 3)
    assert r.edge_ids == (0, 2)


def test_solve_routes_diamond_large_lam_takes_slow_route():
    edge_u, edge_v, edge_time, d_e = _diamond_graph()
    # lam large -> penalized cost of edge 0 (0.05 deep) dominates.
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=100.0)
    od = np.array([[0, 3]], dtype=np.int64)
    results = solve_routes(4, edge_u, edge_v, cost, open_mask, od)
    r = results[0]
    assert r.reachable is True
    # Path 0-1-3: cost = 1*(1+100*0.05/0.3) + 1 = 1+17.67+1 = 19.67
    # Path 0-2-3: cost = 2*(1+0) + 1 = 3. Slow dry route wins.
    assert r.total_cost == pytest.approx(3.0, abs=1e-12)
    assert r.node_path == (0, 2, 3)
    assert r.edge_ids == (1, 3)


def test_solve_routes_closing_only_route():
    edge_u = np.array([0, 1], dtype=np.int64)
    edge_v = np.array([1, 2], dtype=np.int64)
    edge_time = np.array([1.0, 1.0], dtype=np.float64)
    d_e = np.array([0.5, 0.5], dtype=np.float64)  # both above threshold
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=1.0)
    od = np.array([[0, 2]], dtype=np.int64)
    results = solve_routes(3, edge_u, edge_v, cost, open_mask, od)
    assert results[0].reachable is False
    assert results[0].edge_ids == ()
    assert results[0].node_path == ()
    assert results[0].total_cost == float("inf")


def test_solve_routes_disconnected_graph():
    edge_u = np.array([0], dtype=np.int64)
    edge_v = np.array([1], dtype=np.int64)
    edge_time = np.array([1.0], dtype=np.float64)
    d_e = np.array([0.0], dtype=np.float64)
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=1.0)
    od = np.array([[0, 2]], dtype=np.int64)  # node 2 isolated
    results = solve_routes(3, edge_u, edge_v, cost, open_mask, od)
    assert results[0].reachable is False


def test_solve_routes_edge_ids_in_order():
    edge_u = np.array([0, 1, 2], dtype=np.int64)
    edge_v = np.array([1, 2, 3], dtype=np.int64)
    edge_time = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    d_e = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=1.0)
    od = np.array([[0, 3]], dtype=np.int64)
    results = solve_routes(4, edge_u, edge_v, cost, open_mask, od)
    r = results[0]
    assert r.reachable is True
    assert r.edge_ids == (0, 1, 2)
    assert r.node_path == (0, 1, 2, 3)


def test_solve_routes_determinism_repeated():
    edge_u, edge_v, edge_time, d_e = _diamond_graph()
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=1.0)
    od = np.array([[0, 3]], dtype=np.int64)
    r1 = solve_routes(4, edge_u, edge_v, cost, open_mask, od)[0]
    r2 = solve_routes(4, edge_u, edge_v, cost, open_mask, od)[0]
    assert r1.edge_ids == r2.edge_ids
    assert r1.node_path == r2.node_path
    assert r1.total_cost == r2.total_cost


def test_solve_routes_parallel_edge_keeps_lower_cost():
    # Two edges between 0 and 1: edge 0 cost=5, edge 1 cost=3.
    edge_u = np.array([0, 0], dtype=np.int64)
    edge_v = np.array([1, 1], dtype=np.int64)
    edge_time = np.array([5.0, 3.0], dtype=np.float64)
    d_e = np.array([0.0, 0.0], dtype=np.float64)
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=1.0)
    od = np.array([[0, 1]], dtype=np.int64)
    results = solve_routes(2, edge_u, edge_v, cost, open_mask, od)
    r = results[0]
    assert r.reachable is True
    assert r.edge_ids == (1,)  # lower cost edge kept
    assert r.total_cost == pytest.approx(3.0, abs=1e-12)


# ---------------------------------------------------------------------------
# route_metrics
# ---------------------------------------------------------------------------


def test_route_metrics_hand_computed():
    # 3-edge route: lengths [10, 20, 30], times [5, 10, 15], depths [0.1, 0.4, 0.2]
    edge_ids = (0, 1, 2)
    edge_length = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    edge_time = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    d_e = np.array([0.1, 0.4, 0.2], dtype=np.float64)
    m = route_metrics(edge_ids, edge_length, edge_time, d_e, label_threshold_m=0.3, closure_threshold_m=0.3)
    assert m["n_edges"] == 3
    assert m["length_m"] == 60.0
    assert m["free_flow_time_s"] == 30.0
    assert m["max_d_e"] == 0.4
    # length-weighted mean = (0.1*10 + 0.4*20 + 0.2*30) / 60 = (1+8+6)/60 = 15/60 = 0.25
    assert m["mean_d_e_length_weighted"] == pytest.approx(0.25, abs=1e-12)
    # share length >= 0.3: only edge 1 (20 m of 60 m) = 1/3
    assert m["share_length_ge_label"] == pytest.approx(20.0 / 60.0, abs=1e-12)
    # edges >= 0.3: only edge 1 -> 1
    assert m["n_edges_ge_closure"] == 1


def test_route_metrics_closure_none():
    edge_ids = (0, 1)
    edge_length = np.array([10.0, 20.0], dtype=np.float64)
    edge_time = np.array([5.0, 10.0], dtype=np.float64)
    d_e = np.array([0.5, 0.6], dtype=np.float64)
    m = route_metrics(edge_ids, edge_length, edge_time, d_e, label_threshold_m=0.3, closure_threshold_m=None)
    assert m["n_edges_ge_closure"] == 0


def test_route_metrics_empty_route():
    edge_ids = ()
    edge_length = np.array([10.0], dtype=np.float64)
    edge_time = np.array([5.0], dtype=np.float64)
    d_e = np.array([0.5], dtype=np.float64)
    m = route_metrics(edge_ids, edge_length, edge_time, d_e, label_threshold_m=0.3, closure_threshold_m=0.3)
    assert m["n_edges"] == 0
    assert m["length_m"] == 0.0
    assert m["free_flow_time_s"] == 0.0
    assert np.isnan(m["max_d_e"])
    assert np.isnan(m["mean_d_e_length_weighted"])
    assert np.isnan(m["share_length_ge_label"])
    assert m["n_edges_ge_closure"] == 0


# ---------------------------------------------------------------------------
# P0 vs policy consistency
# ---------------------------------------------------------------------------


def test_p0_equals_lam_zero_high_threshold():
    edge_u, edge_v, edge_time, d_e = _diamond_graph()
    cost_p0, _ = policy_costs(edge_time, d_e, closure_threshold_m=None, lam=0.0)
    # lam=0 with threshold above all depths -> same as P0.
    cost_pol, _ = policy_costs(edge_time, d_e, closure_threshold_m=1.0, lam=0.0)
    od = np.array([[0, 3]], dtype=np.int64)
    r_p0 = solve_routes(4, edge_u, edge_v, cost_p0, np.ones(4, dtype=bool), od)[0]
    r_pol = solve_routes(4, edge_u, edge_v, cost_pol, np.ones(4, dtype=bool), od)[0]
    assert r_p0.reachable is True
    assert r_pol.reachable is True
    assert r_p0.total_cost == pytest.approx(r_pol.total_cost, abs=1e-12)
    assert r_p0.edge_ids == r_pol.edge_ids


# ---------------------------------------------------------------------------
# Layer structure
# ---------------------------------------------------------------------------


def test_routing_import_isolation():
    spec = importlib.util.find_spec("urbanflood.routing")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        tree = ast.parse(f.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    allowed = {
        "numpy", "networkx", "dataclasses", "typing",
        "collections.abc", "__future__",
    }
    for name in imported:
        assert any(
            name == a or name.startswith(a + ".") for a in allowed
        ), f"disallowed import: {name}"
    assert not any("urbanflood" in n for n in imported)

    # No public function has a forbidden parameter name; none named 'env'.
    import urbanflood.routing as routing_module
    forbidden = {
        "suscept", "drainage", "outfall", "spill", "lake_spec",
        "initial_state", "simulator", "orchestrator",
        "h_final", "h_initial", "h_peak", "env",
    }
    for name, func in inspect.getmembers(routing_module, inspect.isfunction):
        for param_name in inspect.signature(func).parameters:
            for bad in forbidden:
                assert bad not in param_name, (
                    f"forbidden parameter {param_name!r} in {name}"
                )


# ---------------------------------------------------------------------------
# Determinism on larger graph
# ---------------------------------------------------------------------------


def test_solve_routes_determinism_large_graph():
    rng = np.random.default_rng(123)
    n_nodes = 200
    # Random grid-like graph: connect each node to a few neighbors.
    edge_list = []
    for i in range(n_nodes):
        for j in range(i + 1, min(i + 5, n_nodes)):
            if rng.random() < 0.3:
                edge_list.append((i, j))
    edge_u = np.array([e[0] for e in edge_list], dtype=np.int64)
    edge_v = np.array([e[1] for e in edge_list], dtype=np.int64)
    edge_time = rng.uniform(1.0, 10.0, size=len(edge_u))
    d_e = rng.uniform(0.0, 0.5, size=len(edge_u))
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=0.5)
    od_pairs = sample_od_pairs(
        np.arange(n_nodes, dtype=np.int64),
        np.zeros(n_nodes, dtype=np.int64),
        50.0, 20, 500.0, seed=7,
    )
    r1 = solve_routes(n_nodes, edge_u, edge_v, cost, open_mask, od_pairs)
    r2 = solve_routes(n_nodes, edge_u, edge_v, cost, open_mask, od_pairs)
    for a, b in zip(r1, r2):
        assert a.reachable == b.reachable
        assert a.edge_ids == b.edge_ids
        assert a.node_path == b.node_path
        assert a.total_cost == b.total_cost


# ---------------------------------------------------------------------------
# Timing print (not an assertion)
# ---------------------------------------------------------------------------


def test_solve_routes_timing_print(capsys):
    n_nodes = 2200
    n_edges = 2350
    rng = np.random.default_rng(99)
    edge_u = rng.integers(0, n_nodes, size=n_edges)
    edge_v = rng.integers(0, n_nodes, size=n_edges)
    # Ensure distinct endpoints.
    same = edge_u == edge_v
    edge_v[same] = (edge_v[same] + 1) % n_nodes
    edge_time = rng.uniform(1.0, 10.0, size=n_edges)
    d_e = rng.uniform(0.0, 0.5, size=n_edges)
    cost, open_mask = policy_costs(edge_time, d_e, closure_threshold_m=0.3, lam=0.5)
    od_pairs = sample_od_pairs(
        np.arange(n_nodes, dtype=np.int64),
        np.zeros(n_nodes, dtype=np.int64),
        50.0, 20, 500.0, seed=3,
    )
    t0 = time.perf_counter()
    results = solve_routes(n_nodes, edge_u, edge_v, cost, open_mask, od_pairs)
    elapsed = time.perf_counter() - t0
    with capsys.disabled():
        print(
            f"solve_routes timing: {n_nodes} nodes, {n_edges} edges, "
            f"{len(od_pairs)} OD pairs: {elapsed:.4f} s"
        )
    assert len(results) == 20
