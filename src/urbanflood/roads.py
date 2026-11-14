"""SYNTHETIC road network generation (Phase 1, TC3).

The road layer is a synthetic connected grid network on the 96 x 96 land
cells. Roads follow grid lines (single rows and columns) at three locked
service classes with locked free-flow speeds:

  arterial  50 km/h   (the coastal arterial is additionally fixed on the
                       shoreline row by construction, with no RNG draw)
  secondary 30 km/h
  local     20 km/h

Because every horizontal line spans all land columns and every vertical line
spans all land rows (row 0 to the shoreline row), lines intersect each other
and the network is connected by construction; the test suite asserts this on
the built graph rather than trusting the argument (TC3).

Water faces: road lines are restricted to land rows/columns, so no road edge
ever touches the ocean band. Where a line crosses an inland lake, the water
run is replaced by a single BRIDGE edge spanning it:

  geometry (path cells) = the water cells between the two land abutments
  length  = (number of path cells - 1) * cell size
  deck    = max(max crossed waterbody normal level + 1.0 m freeboard,
                min(land elevation of the two abutments))

This implements the locked bridge rule "decks sit at the crossed waterbody
level plus 1.0 m, floored by the lower approach road elevation". Deck
elevation is a separate attribute; the terrain elevation under a bridge is
the lake bed and is never used as a road surface.

Each lake is guaranteed at least one crossing: after the RNG-selected lines
are drawn, a deterministic (no-RNG) adjustment moves the vertical line closest
to the densest water column onto the column holding the most lake cells if no
selected vertical line already passes through actual lake cells of that lake.
Bridges over lakes are then an invariant of the environment both for the road graph and for the later
routing experiments.

Canonical order conventions:
  nodes = land cells on any line, sorted row-major; node ids index that
          sorted array
  edges = collected per line in construction order, then sorted once by the
          unordered (u, v) cell pair in lexico-graphic order, assigned
          edge ids 0..E-1
RNG consumption (fixed; part of the environment draw order):
  1. rng.permutation over horizontal candidate rows
  2. rng.permutation over vertical candidate columns
No other draw occurs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbanflood.arrays import freeze_array
from urbanflood.config import ElevationConfig, RoadConfig


@dataclass(frozen=True)
class RoadParams:
    """Generation hyperparameters (Phase 1 design choices, not locked V7 values)."""

    n_arterial_horizontal: int = 2
    n_arterial_vertical: int = 3
    n_secondary_horizontal: int = 4
    n_secondary_vertical: int = 4
    n_local_horizontal: int = 6
    n_local_vertical: int = 6
    candidate_margin_cells: int = 4


CLASS_ARTERIAL = 0
CLASS_SECONDARY = 1
CLASS_LOCAL = 2
CLASS_NAMES = ("arterial", "secondary", "local")


@dataclass(frozen=True)
class RoadNetwork:
    """Canonical arrays describing the SYNTHETIC road network.

    Node ids index the row-major sorted land-cell node list. Edge ids index
    the canonically sorted edge list. Bridge edges carry their deck
    elevation and their full path cells; ordinary edges carry just their two
    endpoint cells. Arrays must be treated as immutable by callers.
    """

    node_rows: np.ndarray
    node_cols: np.ndarray
    node_elevation_m: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_class: np.ndarray
    edge_length_m: np.ndarray
    edge_free_flow_time_s: np.ndarray
    edge_is_bridge: np.ndarray
    edge_deck_m: np.ndarray
    edge_path_cells: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def n_nodes(self) -> int:
        return int(self.node_rows.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_u.shape[0])


def build_road_network(
    rng,
    elevation_m: np.ndarray,
    is_lake: np.ndarray,
    lake_id: np.ndarray,
    lake_levels: dict[int, float],
    shore_row: int,
    cell_size_m: float,
    road_cfg: RoadConfig,
    elevation_cfg: ElevationConfig,
    params: RoadParams | None = None,
) -> RoadNetwork:
    """Build the connected SYNTHETIC road network with bridges over lakes.

    Consumes exactly two RNG draws (horizontal then vertical permutation) in
    the documented canonical order; every other decision is deterministic.
    """
    if params is None:
        params = RoadParams()
    n_rows, n_cols = elevation_m.shape

    speeds = (
        road_cfg.arterial_speed_kmh,
        road_cfg.secondary_speed_kmh,
        road_cfg.local_speed_kmh,
    )

    # --- line selection (canonical RNG draw order) ---------------------------
    horizontal_candidates = np.arange(
        params.candidate_margin_cells, shore_row - params.candidate_margin_cells
    )
    vertical_candidates = np.arange(
        params.candidate_margin_cells, n_cols - params.candidate_margin_cells
    )
    required_h = (
        params.n_arterial_horizontal
        + params.n_secondary_horizontal
        + params.n_local_horizontal
    )
    required_v = (
        params.n_arterial_vertical + params.n_secondary_vertical + params.n_local_vertical
    )
    if horizontal_candidates.size < required_h + 1:
        raise ValueError("horizontal candidate range too small for the requested lines")
    if vertical_candidates.size < required_v + 1:
        raise ValueError("vertical candidate range too small for the requested lines")

    # Fixed coastal arterial first (no RNG draw); the rest are RNG-selected.
    h_lines: list[tuple[int, int]] = [(shore_row, CLASS_ARTERIAL)]
    h_classes = (
        [CLASS_ARTERIAL] * params.n_arterial_horizontal
        + [CLASS_SECONDARY] * params.n_secondary_horizontal
        + [CLASS_LOCAL] * params.n_local_horizontal
    )
    order_h = rng.permutation(horizontal_candidates)
    for row, cls in zip(order_h[:required_h], h_classes):
        h_lines.append((int(row), cls))

    v_classes = (
        [CLASS_ARTERIAL] * params.n_arterial_vertical
        + [CLASS_SECONDARY] * params.n_secondary_vertical
        + [CLASS_LOCAL] * params.n_local_vertical
    )
    order_v = rng.permutation(vertical_candidates)
    v_lines: list[tuple[int, int]] = []
    for col, cls in zip(order_v[:required_v], v_classes):
        v_lines.append((int(col), cls))

    # --- deterministic lake-crossing guarantee (no RNG) ----------------------
    v_lines = _ensure_lake_crossings(v_lines, is_lake, lake_id, n_cols)

    # --- edges per line -------------------------------------------------------
    edge_records: list[tuple[tuple[int, int], tuple[int, int], int, tuple[tuple[int, int], ...], bool]] = []
    for row, cls in sorted(h_lines):
        cells = [(row, col) for col in range(n_cols)]
        edge_records.extend(
            _line_edges(cells, cls, is_lake, lake_id, lake_levels, elevation_m,
                        cell_size_m, elevation_cfg.bridge_deck_freeboard_m)
        )
    for col, cls in sorted(v_lines):
        cells = [(row, col) for row in range(shore_row + 1)]
        edge_records.extend(
            _line_edges(cells, cls, is_lake, lake_id, lake_levels, elevation_m,
                        cell_size_m, elevation_cfg.bridge_deck_freeboard_m)
        )

    # --- canonical node and edge ids -----------------------------------------
    node_cell_set = set()
    for row, cls in h_lines:
        for col in range(n_cols):
            if not is_lake[row, col]:
                node_cell_set.add((row, col))
    for col, cls in v_lines:
        for row in range(shore_row + 1):
            if not is_lake[row, col]:
                node_cell_set.add((row, col))
    node_cells = sorted(node_cell_set)
    node_index = {cell: idx for idx, cell in enumerate(node_cells)}

    canon = sorted(
        edge_records,
        key=lambda rec: (min(rec[0], rec[1]), max(rec[0], rec[1])),
    )
    n_edges = len(canon)
    edge_u = np.empty(n_edges, dtype=np.int32)
    edge_v = np.empty(n_edges, dtype=np.int32)
    edge_class = np.empty(n_edges, dtype=np.int8)
    edge_length = np.empty(n_edges, dtype=np.float64)
    edge_time = np.empty(n_edges, dtype=np.float64)
    edge_is_bridge = np.empty(n_edges, dtype=np.bool_)
    edge_deck = np.full(n_edges, np.nan, dtype=np.float64)
    paths: list[tuple[tuple[int, int], ...]] = []
    for idx, (a, b, cls, path, _) in enumerate(canon):
        cell_a, cell_b = min(a, b), max(a, b)
        length = (len(path) - 1) * cell_size_m
        edge_u[idx] = node_index[cell_a]
        edge_v[idx] = node_index[cell_b]
        edge_class[idx] = cls
        edge_length[idx] = length
        edge_time[idx] = length * 3.6 / speeds[cls]
        if _:
            water_lids = {int(lake_id[c[0], c[1]]) for c in path if is_lake[c[0], c[1]]}
            level = max(lake_levels[lid] for lid in water_lids)
            z_a = elevation_m[cell_a[0], cell_a[1]]
            z_b = elevation_m[cell_b[0], cell_b[1]]
            deck = max(level + elevation_cfg.bridge_deck_freeboard_m, min(z_a, z_b))
            edge_is_bridge[idx] = True
            edge_deck[idx] = deck
        else:
            edge_is_bridge[idx] = False
        paths.append(tuple(path))

    return RoadNetwork(
        node_rows=freeze_array(np.array([c[0] for c in node_cells], dtype=np.int16)),
        node_cols=freeze_array(np.array([c[1] for c in node_cells], dtype=np.int16)),
        node_elevation_m=freeze_array(elevation_m[
            np.array([c[0] for c in node_cells], dtype=np.int64),
            np.array([c[1] for c in node_cells], dtype=np.int64),
        ]),
        edge_u=freeze_array(edge_u),
        edge_v=freeze_array(edge_v),
        edge_class=freeze_array(edge_class),
        edge_length_m=freeze_array(edge_length),
        edge_free_flow_time_s=freeze_array(edge_time),
        edge_is_bridge=freeze_array(edge_is_bridge),
        edge_deck_m=freeze_array(edge_deck),
        edge_path_cells=tuple(paths),
    )


def _line_is_sole_crossing_of_another_lake(
    v_lines: list[tuple[int, int]],
    lake_water_columns: dict[int, set[int]],
    index: int,
    serving_lid: int,
) -> bool:
    """True if moving the line at `index` would remove another lake's only crossing.

    The line's current column is checked against every other lake's water
    columns; the move is unsafe when that column is among them and no other
    selected line still crosses that lake. No RNG is consumed.
    """
    old_col = v_lines[index][0]
    for other_lid, other_columns in lake_water_columns.items():
        if other_lid == serving_lid:
            continue
        if old_col not in other_columns:
            continue
        still_crossed = any(
            col != old_col and col in other_columns for col, _ in v_lines
        )
        if not still_crossed:
            return True
    return False


def _select_line_to_move(
    v_lines: list[tuple[int, int]],
    lake_water_columns: dict[int, set[int]],
    serving_lid: int,
    target: int,
) -> int:
    """Index of the line moved onto the target crossing column.

    The closest line to the target (ties broken by line order) is chosen,
    but a line is never moved off a column that is another lake's only
    crossing. The moved line keeps its service class and columns stay unique.
    Raises RuntimeError when no safely movable line exists, so the helper
    never silently breaks another lake's crossing invariant. No RNG draw is
    consumed.
    """
    safe = [
        i
        for i in range(len(v_lines))
        if not _line_is_sole_crossing_of_another_lake(
            v_lines, lake_water_columns, i, serving_lid
        )
    ]
    if not safe:
        raise RuntimeError(
            "every line is the sole crossing of another lake; moving any of "
            f"them would break a lake's crossing guarantee (explicit failure, "
            "not a silent violation)"
        )
    return min(safe, key=lambda i: (abs(v_lines[i][0] - target), i))


def _ensure_lake_crossings(
    v_lines: list[tuple[int, int]],
    is_lake: np.ndarray,
    lake_id: np.ndarray,
    n_cols: int,
) -> list[tuple[int, int]]:
    """Deterministically guarantee at least one vertical line per lake.

    For each lake in ascending id order: when no selected vertical line
    passes through actual lake cells of that lake, a crossing column is
    chosen among the lake's OWN water columns only, candidates ordered
    deterministically by descending water cell count and then by column id.
    Columns already occupied by other lines or outside the allowed interior
    range [2, n_cols - 3] are skipped explicitly; the line closest to the
    chosen target (ties broken by line order) is moved onto it, preserving
    that line's service class. A line is never moved off a column that is
    another lake's only crossing; if no safely movable line exists, the
    helper fails explicitly instead of silently breaking that lake's
    invariant. No line is ever moved to a column without lake water of the
    target lake, and no extra RNG draw is consumed.
    """
    v_lines = list(v_lines)
    used = {col for col, _ in v_lines}
    lake_ids = [int(v) for v in np.unique(lake_id[is_lake])]
    lake_cells: dict[int, np.ndarray] = {}
    lake_water_columns: dict[int, set[int]] = {}
    for lid in lake_ids:
        cells = np.argwhere(lake_id == lid)
        if cells.size == 0:
            raise RuntimeError(f"lake {lid} has no cells; environment generator error")
        lake_cells[lid] = cells
        lake_water_columns[lid] = {int(c) for c in np.unique(cells[:, 1])}
    for lid in lake_ids:
        cells = lake_cells[lid]
        water_columns = lake_water_columns[lid]
        crossed = any(col in water_columns for col, _ in v_lines)
        if crossed:
            continue
        counts = {int(c): int((cells[:, 1] == c).sum()) for c in water_columns}
        candidates = sorted(counts, key=lambda c: (-counts[c], c))
        target = next(
            (c for c in candidates if c not in used and 2 <= c <= n_cols - 3),
            None,
        )
        if target is None:
            raise RuntimeError(
                f"lake {lid} has no unoccupied interior water column for a road "
                "crossing; the crossing guarantee cannot be satisfied "
                "(explicit failure, not a silent violation)"
            )
        best = _select_line_to_move(v_lines, lake_water_columns, lid, target)
        old_col, old_class = v_lines[best]
        used.discard(old_col)
        v_lines[best] = (target, old_class)
        used.add(target)
        if counts[target] <= 0:
            raise RuntimeError("crossing column lost its lake water; invariant violated")
    return v_lines


def _line_edges(
    cells: list[tuple[int, int]],
    cls: int,
    is_lake: np.ndarray,
    lake_id: np.ndarray,
    lake_levels: dict[int, float],
    elevation_m: np.ndarray,
    cell_size_m: float,
    deck_freeboard_m: float,
):
    """Edges along one straight line of cells, merging water runs into bridges.

    Yields (cell_a, cell_b, class, path_cells, is_bridge) records. Path cells
    of an ordinary edge are its two endpoints; path cells of a bridge edge
    are the land abutments plus every water cell of the run, in line order.
    """
    n = len(cells)
    first, last = cells[0], cells[-1]
    if is_lake[first[0], first[1]] or is_lake[last[0], last[1]]:
        raise RuntimeError("water run touches a line end; generator invariant violated")
    out = []
    i = 0
    while i < n - 1:
        a = cells[i]
        if is_lake[a[0], a[1]]:
            raise RuntimeError("land/water walk started inside a water run")
        if not is_lake[cells[i + 1][0], cells[i + 1][1]]:
            out.append((a, cells[i + 1], cls, (a, cells[i + 1]), False))
            i += 1
            continue
        j = i + 1
        while j < n and is_lake[cells[j][0], cells[j][1]]:
            j += 1
        if j == n:
            raise RuntimeError("water run touches a line end; generator invariant violated")
        b = cells[j]
        run = tuple(cells[i : j + 1])
        out.append((a, b, cls, run, True))
        i = j
    return out
