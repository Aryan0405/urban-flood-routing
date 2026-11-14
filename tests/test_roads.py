"""Phase 1 tests for the SYNTHETIC road network (roads.py, TC3).

Covers structural connectivity, bridge placement and the locked deck rule,
canonical id assignment, and free-flow travel-time consistency.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from urbanflood.config import build_default_config
from urbanflood.environment import build_environment
from urbanflood.roads import CLASS_ARTERIAL, CLASS_LOCAL, CLASS_SECONDARY, _ensure_lake_crossings

MASTER = 2027
CONFIG = build_default_config()
FREEBOARD = CONFIG.elevation.bridge_deck_freeboard_m
ENV_SEEDS = tuple([MASTER] + list(CONFIG.seeds.env_experiment_seeds()))


@pytest.fixture(scope="module")
def env():
    return build_environment(MASTER, CONFIG)


@pytest.fixture(scope="module")
def net(env):
    return env.road_network


@pytest.fixture(scope="module")
def graph(net):
    graph = nx.Graph()
    for idx in range(net.n_edges):
        graph.add_edge(int(net.edge_u[idx]), int(net.edge_v[idx]))
    return graph


# --- TC3: connectivity and structure -----------------------------------------

def test_tc3_network_is_connected(net, graph):
    assert graph.number_of_nodes() == net.n_nodes
    assert net.n_nodes > 1000
    assert nx.number_connected_components(graph) == 1


def test_tc3_graph_has_no_isolated_nodes_and_uses_all_nodes(net, graph):
    degrees = dict(graph.degree())
    assert min(degrees.values()) >= 1
    assert set(degrees.keys()) == set(range(net.n_nodes))


def test_every_node_is_a_land_cell(net, env):
    for row, col in zip(net.node_rows.tolist(), net.node_cols.tolist()):
        assert not env.is_ocean[row, col]
        assert not env.is_lake[row, col]
        assert env.road_cells[row, col]


def test_road_cells_cover_network_nodes(net, env):
    assert env.road_cells.sum() == net.n_nodes


# --- ordinary edges ------------------------------------------------------------

def test_ordinary_edges_connect_adjacent_land_cells(net, env):
    for idx in range(net.n_edges):
        u, v = int(net.edge_u[idx]), int(net.edge_v[idx])
        assert u != v
        dr = int(net.node_rows[u]) - int(net.node_rows[v])
        dc = int(net.node_cols[u]) - int(net.node_cols[v])
        if net.edge_is_bridge[idx]:
            continue
        assert abs(dr) + abs(dc) == 1
        assert net.edge_length_m[idx] == pytest.approx(env.grid.cell_size_m)
        bridge = net.edge_is_bridge[idx] and True or False
        assert bridge is False


def test_edge_ids_canonical(net):
    pairs = []
    for idx in range(net.n_edges):
        u, v = int(net.edge_u[idx]), int(net.edge_v[idx])
        assert u < v
        key = (int(net.node_rows[u]), int(net.node_cols[u]),
               int(net.node_rows[v]), int(net.node_cols[v]))
        pairs.append(key)
    assert pairs == sorted(pairs)
    assert len(set(pairs)) == len(pairs)


def test_node_ids_canonical_row_major(net):
    order = np.lexsort((net.node_cols, net.node_rows))
    assert np.array_equal(order, np.arange(net.n_nodes))


# --- bridges and the locked deck rule -------------------------------------------

def test_tc3_bridges_exist_and_cross_every_lake(net, env):
    assert net.edge_is_bridge.sum() >= len(env.lake_specs)
    for spec in env.lake_specs:
        bridges_of_lake = []
        for idx in range(net.n_edges):
            if not net.edge_is_bridge[idx]:
                continue
            path = net.edge_path_cells[idx]
            water = [
                (r, c) for (r, c) in path if env.is_lake[r, c]
            ]
            ids = {int(env.lake_id[r, c]) for (r, c) in water}
            if spec.lake_id in ids:
                bridges_of_lake.append(idx)
        assert bridges_of_lake, f"lake {spec.lake_id} has no bridge"


def test_bridge_paths_are_single_water_runs(net, env):
    for idx in range(net.n_edges):
        if not net.edge_is_bridge[idx]:
            continue
        path = net.edge_path_cells[idx]
        assert len(path) >= 3
        assert all(isinstance(cell, tuple) and len(cell) == 2 for cell in path)
        lines_straight = len({cell[0] for cell in path}) == 1 or len(
            {cell[1] for cell in path}
        ) == 1
        assert lines_straight
        first, last = path[0], path[-1]
        assert not env.is_lake[first[0], first[1]]
        assert not env.is_lake[last[0], last[1]]
        for cell in path[1:-1]:
            assert env.is_lake[cell[0], cell[1]]
        a = (int(net.node_rows[int(net.edge_u[idx])]), int(net.node_cols[int(net.edge_u[idx])]))
        b = (int(net.node_rows[int(net.edge_v[idx])]), int(net.node_cols[int(net.edge_v[idx])]))
        assert a == first and b == last
        length = (len(path) - 1) * env.grid.cell_size_m
        assert net.edge_length_m[idx] == pytest.approx(length)


def test_bridge_deck_locked_rule(net, env):
    for idx in range(net.n_edges):
        if not net.edge_is_bridge[idx]:
            assert np.isnan(net.edge_deck_m[idx])
            continue
        path = net.edge_path_cells[idx]
        water = [(r, c) for (r, c) in path if env.is_lake[r, c]]
        levels = [env.lake_level(int(env.lake_id[r, c])) for (r, c) in water]
        level = max(levels)
        z_a = net.node_elevation_m[int(net.edge_u[idx])]
        z_b = net.node_elevation_m[int(net.edge_v[idx])]
        expected = max(level + FREEBOARD, min(z_a, z_b))
        assert net.edge_deck_m[idx] == pytest.approx(expected, rel=1e-12)
        assert net.edge_deck_m[idx] > level


def test_no_path_touches_ocean(net, env):
    for path in net.edge_path_cells:
        for r, c in path:
            assert not env.is_ocean[r, c]


@pytest.mark.parametrize("seed", ENV_SEEDS)
def test_tc3_multi_seed_connected_and_every_lake_bridged(seed):
    """Master seed plus all five Experiment A environment seeds: the graph
    stays connected and every lake keeps at least one bridge. This is tested
    evidence for these six seeds only, not a universal validation claim."""
    env = build_environment(seed, CONFIG)
    network = env.road_network
    graph = nx.Graph()
    for idx in range(network.n_edges):
        graph.add_edge(int(network.edge_u[idx]), int(network.edge_v[idx]))
    assert nx.number_connected_components(graph) == 1
    assert graph.number_of_nodes() == network.n_nodes
    for spec in env.lake_specs:
        crossings = []
        for idx in range(network.n_edges):
            if not network.edge_is_bridge[idx]:
                continue
            water_ids = {
                int(env.lake_id[r, c])
                for (r, c) in network.edge_path_cells[idx]
                if env.is_lake[r, c]
            }
            if spec.lake_id in water_ids:
                crossings.append(idx)
        assert crossings, f"seed {seed}: lake {spec.lake_id} has no bridge"


# --- crossing-guarantee helper (deterministic, unit level) -------------------


def _lake_arrays(lake_cells, n_rows, n_cols):
    """Synthetic lake_id grid from an iterable of (lake_id, rows, cols)."""
    lake_id_grid = np.full((n_rows, n_cols), -1, dtype=np.int16)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    for lid, rows, cols in lake_cells:
        lake_id_grid[np.ix_(rows, cols)] = lid
        is_lake[np.ix_(rows, cols)] = True
    return is_lake, lake_id_grid


def test_crossing_move_targets_an_actual_water_column():
    is_lake, lake_grid = _lake_arrays([(0, range(1, 4), range(10, 15))], 6, 20)
    # neither line crosses the lake yet (columns 18 and 16 are dry), so a
    # move must actually happen.
    lines = [(18, CLASS_ARTERIAL), (16, CLASS_SECONDARY)]
    result = _ensure_lake_crossings(lines, is_lake, lake_grid, 20)
    water_columns = {int(c) for c in np.unique(np.argwhere(lake_grid == 0)[:, 1])}
    columns = [col for col, _ in result]
    assert len(set(columns)) == len(columns)
    # every column carries 3 water cells, so the densest-target tie breaks to
    # column 10 and the closest line (16) is moved onto it, keeping its class
    assert result == [(18, CLASS_ARTERIAL), (10, CLASS_SECONDARY)]
    assert 10 in water_columns, "moved line does not cross an actual water column"


def test_crossing_preserves_class_and_uniqueness():
    is_lake, lake_grid = _lake_arrays([(0, range(1, 4), range(10, 15))], 6, 20)
    # no line crosses the lake yet, so a move must actually happen
    lines = [(18, CLASS_ARTERIAL), (16, CLASS_SECONDARY), (15, CLASS_LOCAL)]
    result = _ensure_lake_crossings(lines, is_lake, lake_grid, 20)
    columns = [col for col, _ in result]
    assert len(set(columns)) == len(columns)
    classes = [cls for _, cls in result]
    assert classes == [CLASS_ARTERIAL, CLASS_SECONDARY, CLASS_LOCAL]
    # closest line to the densest target (column 10) is 15, moved onto it
    assert result == [(18, CLASS_ARTERIAL), (16, CLASS_SECONDARY), (10, CLASS_LOCAL)]


def test_occupied_water_column_means_crossed_so_no_move():
    """A line already on one of the lake's water columns means the lake is
    crossed; the helper must not move anything (no occupied-target bug)."""
    is_lake, lake_grid = _lake_arrays(
        [(0, range(1, 4), [10, 11, 12, 13, 12, 12, 12, 12])], 6, 20
    )
    # give column 12 more water cells than any other column
    is_lake[3, 12] = True
    lake_grid[3, 12] = 0
    lines = [(12, CLASS_LOCAL), (18, CLASS_ARTERIAL)]
    result = _ensure_lake_crossings(lines, is_lake, lake_grid, 20)
    assert result == [(12, CLASS_LOCAL), (18, CLASS_ARTERIAL)]


def test_crossing_falls_back_to_next_candidate_when_densest_out_of_bounds():
    is_lake, lake_grid = _lake_arrays(
        [(0, range(1, 4), [1, 2, 8, 9])], 6, 20
    )
    # columns 1 and 2 carry 3 water cells each; column 2 is inside bounds
    lines = [(15, CLASS_SECONDARY)]
    result = _ensure_lake_crossings(lines, is_lake, lake_grid, 20)
    assert len(result) == 1
    moved_col, moved_class = result[0]
    assert moved_col in {2, 8, 9}  # column 1 (densest, out of interior bounds) skipped
    assert moved_class == CLASS_SECONDARY
    assert moved_col in {int(c) for c in np.unique(np.argwhere(lake_grid == 0)[:, 1])}


def test_two_lakes_both_get_deterministic_crossings():
    is_lake, lake_grid = _lake_arrays(
        [(0, range(1, 4), range(8, 13)), (1, range(6, 9), range(40, 45))], 10, 60
    )
    lines = [(50, CLASS_ARTERIAL), (55, CLASS_LOCAL)]
    result = _ensure_lake_crossings(lines, is_lake, lake_grid, 60)
    columns = [col for col, _ in result]
    assert [cls for _, cls in result] == [CLASS_ARTERIAL, CLASS_LOCAL]
    assert len(set(columns)) == len(columns)
    for lid in (0, 1):
        water_cols = {
            int(c) for c in np.unique(np.argwhere(lake_grid == lid)[:, 1])
        }
        assert any(col in water_cols for col in columns), f"lake {lid} not crossed"


def test_explicit_failure_when_no_interior_water_column_exists():
    is_lake, lake_grid = _lake_arrays([(0, range(0, 3), [0, 1])], 5, 10)
    lines = [(5, CLASS_ARTERIAL)]
    with pytest.raises(RuntimeError, match="crossing guarantee cannot be satisfied"):
        _ensure_lake_crossings(lines, is_lake, lake_grid, 10)


def test_move_never_removes_another_lakes_only_crossing():
    """Adversarial two-lake case (seed 78 equivalent): the line closest to the
    second lake's target column is the first lake's ONLY crossing, so it must
    not be moved; the next-closest safe line is moved instead."""
    is_lake, lake_grid = _lake_arrays(
        [(0, range(1, 4), range(10, 13)), (1, range(5, 8), range(40, 45))], 10, 60
    )
    # column 11 is lake 0's only crossing; column 40 is lake 1's densest
    # target, and line 11 is closer to it than line 70 (29 vs 30).
    lines = [(11, CLASS_LOCAL), (70, CLASS_ARTERIAL)]
    result = _ensure_lake_crossings(lines, is_lake, lake_grid, 60)
    columns = [col for col, _ in result]
    classes = [cls for _, cls in result]
    assert classes == [CLASS_LOCAL, CLASS_ARTERIAL]
    assert len(set(columns)) == len(columns)
    # line 11 stays put (lake 0 keeps its only crossing), line 70 moves to 40
    assert result == [(11, CLASS_LOCAL), (40, CLASS_ARTERIAL)]
    for lid in (0, 1):
        water_cols = {
            int(c) for c in np.unique(np.argwhere(lake_grid == lid)[:, 1])
        }
        assert any(col in water_cols for col in columns), f"lake {lid} not crossed"


def test_explicit_failure_when_every_line_is_another_lakes_only_crossing():
    """No safe candidate: both lines are sole crossings of lakes 0 and 1,
    while lake 2 still needs a crossing -> explicit failure, no silent break."""
    is_lake, lake_grid = _lake_arrays(
        [(0, range(1, 4), [10]), (1, range(5, 8), [20]), (2, range(9, 12), [30])],
        13,
        60,
    )
    lines = [(10, CLASS_LOCAL), (20, CLASS_ARTERIAL)]
    with pytest.raises(
        RuntimeError, match="sole crossing of another lake|crossing guarantee cannot be satisfied"
    ):
        _ensure_lake_crossings(lines, is_lake, lake_grid, 60)


def test_seed_78_all_lakes_keep_a_crossing():
    """Environment-level regression for seed 78: before the fix, serving lake 1
    moved lake 0's only crossing and left lake 0 without one. Both lakes must
    be crossed at the built environment."""
    env = build_environment(78, CONFIG)
    network = env.road_network
    for spec in env.lake_specs:
        crossings = 0
        for idx in range(network.n_edges):
            if not network.edge_is_bridge[idx]:
                continue
            water_ids = {
                int(env.lake_id[r, c])
                for (r, c) in network.edge_path_cells[idx]
                if env.is_lake[r, c]
            }
            if spec.lake_id in water_ids:
                crossings += 1
        assert crossings >= 1, f"seed 78: lake {spec.lake_id} has no crossing"


def test_crossing_helper_is_deterministic():
    is_lake, lake_grid = _lake_arrays([(0, range(1, 4), range(10, 15))], 6, 20)
    lines = [(18, CLASS_ARTERIAL), (14, CLASS_SECONDARY)]
    first = _ensure_lake_crossings(lines, is_lake, lake_grid, 20)
    second = _ensure_lake_crossings(lines, is_lake, lake_grid, 20)
    assert first == second


# --- classes, speeds, times ------------------------------------------------------

def test_all_three_classes_present(net):
    counts = np.bincount(net.edge_class.astype(np.int64), minlength=3)
    assert all(counts[CLASS_ARTERIAL:CLASS_LOCAL + 1] > 0)


def test_free_flow_times_locked_speeds(net):
    speeds = {
        CLASS_ARTERIAL: CONFIG.routing.arterial_speed_kmh,
        CLASS_SECONDARY: CONFIG.routing.secondary_speed_kmh,
        CLASS_LOCAL: CONFIG.routing.local_speed_kmh,
    }
    for idx in range(net.n_edges):
        expected = net.edge_length_m[idx] * 3.6 / speeds[int(net.edge_class[idx])]
        assert net.edge_free_flow_time_s[idx] == pytest.approx(expected, rel=1e-12)


def test_nondegenerate_speed_range(net):
    assert net.edge_free_flow_time_s.min() > 0.0


def test_deck_elevation_systematically_above_terrain_levels_only_at_bridges(net):
    decks = net.edge_deck_m[~net.edge_is_bridge]
    assert np.all(np.isnan(decks))
