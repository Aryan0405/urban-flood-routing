"""Tests for peak node/edge depth outputs in orchestrator.py.

Quick tests only: no run over 120 simulation steps. Uses short runs
(25 steps or 150 steps) and toy fixtures.
"""

from __future__ import annotations

from types import SimpleNamespace

import ast
import importlib
import inspect
import numpy as np
import pytest

from urbanflood.config import build_default_config
from urbanflood.drainage import generate_drainage
from urbanflood.environment import build_environment
from urbanflood.initial_state import build_initial_depth
from urbanflood.orchestrator import (
    edge_depths,
    node_depths,
    peak_edge_depths,
    peak_node_depths,
    run_simulation,
)
from urbanflood.config import SCENARIOS

MASTER = 2027
CONFIG = build_default_config()
ENV = build_environment(MASTER, CONFIG)
N_CELLS = CONFIG.grid.n_rows * CONFIG.grid.n_cols

DRAINAGE = generate_drainage(
    master_seed=MASTER,
    n_cells=N_CELLS,
    blocked_fraction=0.20,
    drainage_config=CONFIG.drainage,
)
RAIN = SCENARIOS["R"].rainfall
TIDE = SCENARIOS["R"].tide
EXCH = CONFIG.exchange
RUNOFF = CONFIG.runoff


def _run_short(dt_s: float, n_steps: int):
    h_initial = build_initial_depth(ENV)
    return run_simulation(
        h_initial, ENV, DRAINAGE, RAIN, TIDE, EXCH, RUNOFF,
        dt=dt_s, n_steps=n_steps,
    )


def _imported_module_names(module_name: str) -> list[str]:
    """Return all module names imported by a module (ast.Import and ast.ImportFrom)."""
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


# ---------------------------------------------------------------------------
# Step 2a: refactor safety - node_depths and edge_depths unchanged
# ---------------------------------------------------------------------------

def _ref_edge_samples(h_final, network):
    """Independent reference implementation of edge_depths sampling."""
    fractions = np.array([0.0, 0.5, 1.0])
    n_edges = network.n_edges
    n_rows, n_cols = ENV.elevation_m.shape
    output = np.empty((n_edges, 3), dtype=h_final.dtype)
    for k in range(n_edges):
        path = network.edge_path_cells[k]
        n_cells = len(path)
        rows = [c[0] for c in path]
        cols = [c[1] for c in path]
        depths = h_final[rows, cols]
        for j, f in enumerate(fractions):
            x = f * (n_cells - 1)
            i = int(np.floor(x))
            alpha = x - i
            if alpha == 0.0:
                output[k, j] = depths[i]
            else:
                output[k, j] = (1.0 - alpha) * depths[i] + alpha * depths[i + 1]
    return output


def test_node_depths_match_independent_recomputation():
    result = _run_short(30.0, 25)
    depths = node_depths(result, ENV)
    node_rows = ENV.road_network.node_rows.astype(np.int64)
    node_cols = ENV.road_network.node_cols.astype(np.int64)
    expected = result.h_final[node_rows, node_cols]
    assert depths.shape == (ENV.road_network.n_nodes,)
    assert depths.dtype == np.float64
    assert depths.tobytes() == expected.tobytes()


def test_edge_depths_match_independent_recomputation():
    result = _run_short(30.0, 25)
    samples = edge_depths(result, ENV)
    expected = _ref_edge_samples(result.h_final, ENV.road_network)
    assert samples.shape == (ENV.road_network.n_edges, 3)
    assert samples.dtype == np.float64
    assert samples.tobytes() == expected.tobytes()


# ---------------------------------------------------------------------------
# Step 2b: peak_node_depths
# ---------------------------------------------------------------------------

def test_peak_node_depths_shape_dtype_fresh_copy():
    result = _run_short(30.0, 25)
    h_peak_before = result.H_peak.copy()
    depths = peak_node_depths(result, ENV)
    node_rows = ENV.road_network.node_rows.astype(np.int64)
    node_cols = ENV.road_network.node_cols.astype(np.int64)
    expected = result.H_peak[node_rows, node_cols]
    assert depths.shape == (ENV.road_network.n_nodes,)
    assert depths.dtype == result.H_peak.dtype
    assert depths.tobytes() == expected.tobytes()
    # Fresh copy: mutating the returned array in place does not affect H_peak
    depths[0] = 999.0
    assert np.array_equal(result.H_peak, h_peak_before)
    assert not np.shares_memory(depths, result.H_peak)


def test_peak_node_depths_equals_h_peak_indexing():
    result = _run_short(30.0, 25)
    depths = peak_node_depths(result, ENV)
    node_rows = ENV.road_network.node_rows.astype(np.int64)
    node_cols = ENV.road_network.node_cols.astype(np.int64)
    expected = result.H_peak[node_rows, node_cols]
    np.testing.assert_array_equal(depths, expected)


def test_node_depths_fresh_copy():
    result = _run_short(30.0, 25)
    node_rows = ENV.road_network.node_rows.astype(np.int64)
    node_cols = ENV.road_network.node_cols.astype(np.int64)
    h_final_before = result.h_final.copy()
    depths = node_depths(result, ENV)
    expected = result.h_final[node_rows, node_cols]
    assert depths.shape == (ENV.road_network.n_nodes,)
    assert depths.dtype == np.float64
    assert depths.tobytes() == expected.tobytes()
    depths[0] = 999.0
    assert np.array_equal(result.h_final, h_final_before)
    assert not np.shares_memory(depths, result.h_final)


def test_edge_depths_fresh_copy():
    result = _run_short(30.0, 25)
    h_final_before = result.h_final.copy()
    samples = edge_depths(result, ENV)
    assert samples.shape == (ENV.road_network.n_edges, 3)
    assert samples.dtype == np.float64
    samples[0, 0] = 999.0
    assert np.array_equal(result.h_final, h_final_before)
    assert not np.shares_memory(samples, result.h_final)


# ---------------------------------------------------------------------------
# Step 2c: peak_edge_depths
# ---------------------------------------------------------------------------

def test_peak_edge_depths_shape_dtype_fresh_copy():
    result = _run_short(30.0, 25)
    h_peak_before = result.H_peak.copy()
    samples = peak_edge_depths(result, ENV)
    assert samples.shape == (ENV.road_network.n_edges, 3)
    assert samples.dtype == result.H_peak.dtype
    # Fresh copy: mutating the returned array in place does not affect H_peak
    samples[0, 0] = 999.0
    assert np.array_equal(result.H_peak, h_peak_before)
    assert not np.shares_memory(samples, result.H_peak)


def test_peak_edge_depths_endpoint_columns_exact():
    result = _run_short(30.0, 25)
    samples = peak_edge_depths(result, ENV)
    network = ENV.road_network
    node_rows = network.node_rows.astype(np.int64)
    node_cols = network.node_cols.astype(np.int64)
    for k in range(network.n_edges):
        u_cell = (int(node_rows[network.edge_u[k]]), int(node_cols[network.edge_u[k]]))
        v_cell = (int(node_rows[network.edge_v[k]]), int(node_cols[network.edge_v[k]]))
        path = network.edge_path_cells[k]
        assert path[0] == u_cell, f"edge {k}: path[0] != edge_u cell"
        assert path[-1] == v_cell, f"edge {k}: path[-1] != edge_v cell"
        assert samples[k, 0] == result.H_peak[u_cell], f"edge {k} start mismatch"
        assert samples[k, 2] == result.H_peak[v_cell], f"edge {k} end mismatch"


def test_peak_edge_depths_midpoint_interpolation():
    result = _run_short(30.0, 25)
    samples = peak_edge_depths(result, ENV)
    network = ENV.road_network
    h_peak = result.H_peak
    for k in range(network.n_edges):
        path = network.edge_path_cells[k]
        n_cells = len(path)
        rows = [c[0] for c in path]
        cols = [c[1] for c in path]
        depths = h_peak[rows, cols]
        f = 0.5
        x = f * (n_cells - 1)
        i = int(np.floor(x))
        alpha = x - i
        if alpha == 0.0:
            expected_mid = depths[i]
        else:
            expected_mid = (1.0 - alpha) * depths[i] + alpha * depths[i + 1]
        assert samples[k, 1] == expected_mid, f"edge {k} midpoint mismatch"


# ---------------------------------------------------------------------------
# Step 2d: dominance (peak >= final)
# ---------------------------------------------------------------------------

def test_peak_node_depths_dominate_final():
    result = _run_short(30.0, 25)
    final = node_depths(result, ENV)
    peak = peak_node_depths(result, ENV)
    assert np.all(peak >= final - 0.0), "peak node depths must be >= final"


def test_peak_edge_depths_dominate_final():
    result = _run_short(30.0, 25)
    final = edge_depths(result, ENV)
    peak = peak_edge_depths(result, ENV)
    assert np.all(peak >= final - 0.0), "peak edge depths must be >= final"


# ---------------------------------------------------------------------------
# Step 2e: validation
# ---------------------------------------------------------------------------

def test_peak_outputs_reject_wrong_shape():
    wrong_shape = np.zeros((ENV.grid.n_rows + 1, ENV.grid.n_cols), dtype=np.float64)
    result = SimpleNamespace(H_peak=wrong_shape)
    with pytest.raises(ValueError, match="does not match environment grid shape"):
        peak_node_depths(result, ENV)
    with pytest.raises(ValueError, match="does not match environment grid shape"):
        peak_edge_depths(result, ENV)


# ---------------------------------------------------------------------------
# Step 2f: rain-sensitive case (150 steps, covers 30 steps of rain)
# ---------------------------------------------------------------------------

def test_peak_edge_depths_rain_sensitive_report(capsys):
    result = _run_short(30.0, 150)
    final = edge_depths(result, ENV)
    peak = peak_edge_depths(result, ENV)
    mid_diff = peak[:, 1] - final[:, 1]
    n_mid_gt_05 = int(np.sum(mid_diff > 0.05))
    final_max = final.max(axis=1)
    peak_max = peak.max(axis=1)
    n_peak_ge_030 = int(np.sum(peak_max >= 0.30))
    n_final_ge_030 = int(np.sum(final_max >= 0.30))
    with capsys.disabled():
        print("")
        print("=== Peak vs final edge samples (150 steps, rain active) ===")
        print(f"Edges whose peak midpoint exceeds final midpoint by > 0.05 m: {n_mid_gt_05}")
        print(f"Edges with peak max-sample >= 0.30 m: {n_peak_ge_030}")
        print(f"Edges with final max-sample >= 0.30 m: {n_final_ge_030}")


# ---------------------------------------------------------------------------
# Step 2g: layer separation
# ---------------------------------------------------------------------------

def test_peak_outputs_no_susceptibility_reference():
    names = _imported_module_names("urbanflood.orchestrator")
    assert not any("suscept" in n for n in names), (
        f"orchestrator.py imports susceptibility: {names}"
    )
    func_source = inspect.getsource(peak_node_depths) + inspect.getsource(peak_edge_depths)
    assert "suscept" not in func_source.lower(), "peak functions reference susceptibility"
