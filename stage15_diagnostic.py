"""Stage 15 integration diagnostic: full pipeline verification.

Runs the complete pipeline:
  environment generation -> road generation -> six-hour simulation ->
  SimulationResult -> node_depths -> edge_depths.

Verifies:
- Array shapes, dtypes, finiteness
- Mass conservation (relative error < 1e-9)
- H_peak includes initial depths and is >= final depths
- Node-depth and edge-depth contracts (shape, dtype, fresh copy, no aliasing)
- Bitwise reproducibility across two identical runs (optional)
- Runtime and approximate memory usage

EXECUTION MODES (default = single six-hour simulation + extraction):

  python stage15_diagnostic.py
      Run environment generation, road extraction, ONE six-hour
      simulation (dt=30s, 720 steps), node_depths, edge_depths.
      No reproducibility second run. tracemalloc is disabled to avoid
      overhead. This is the fast single-run mode.

  python stage15_diagnostic.py --repro
      Same as default, plus a SECOND six-hour simulation for bitwise
      reproducibility verification. This takes roughly 2x the simulation time.

  python stage15_diagnostic.py --no-sim
      Skip the six-hour simulation. Runs environment + road generation
      only, plus lightweight node_depths/edge_depths contract checks
      using a synthetic h_final array (no simulation). Useful for
      syntax/import validation without the expensive simulation.

EXIT CODE:
    0 if all applicable checks pass.
    1 if any applicable check fails.

No simulation source code, numerical specification, or existing test is
modified by this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from urbanflood.config import build_default_config
from urbanflood.drainage import generate_drainage
from urbanflood.environment import build_environment
from urbanflood.initial_state import build_initial_depth
from urbanflood.orchestrator import (
    run_simulation,
    node_depths,
    edge_depths,
)


def log(message: str, **kwargs) -> None:
    """Print a progress message with immediate flush by default."""
    kwargs.setdefault("flush", True)
    print(message, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 15 integration diagnostic for the flood simulator.",
    )
    parser.add_argument(
        "--repro",
        action="store_true",
        help="Run a second six-hour simulation for bitwise reproducibility "
        "verification (roughly doubles simulation time).",
    )
    parser.add_argument(
        "--no-sim",
        action="store_true",
        help="Skip the six-hour simulation. Runs environment + road "
        "generation only, plus lightweight node_depths/edge_depths "
        "contract checks using a synthetic h_final array.",
    )
    args = parser.parse_args()

    if args.no_sim and args.repro:
        log("ERROR: --no-sim and --repro are incompatible: --repro requires "
            "a six-hour simulation result, which --no-sim skips.")
        return 2

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))
        status = "PASS" if ok else "FAIL"
        log("  [%s] %s" % (status, name))

    log("=" * 72)
    log("STAGE 15 INTEGRATION DIAGNOSTIC")
    log("=" * 72)
    log("")
    log("Mode: repro=%s, no_sim=%s" % (args.repro, args.no_sim))
    log("")

    # ------------------------------------------------------------------
    # 1. Environment generation
    # ------------------------------------------------------------------
    log("1. ENVIRONMENT GENERATION")
    log("  -> Building synthetic environment (master_seed=2027)...", flush=True)
    t0 = time.perf_counter()
    config = build_default_config()
    env = build_environment(master_seed=2027, config=config)
    t_env = time.perf_counter() - t0

    log("  Grid: %d x %d = %d cells" % (env.grid.n_rows, env.grid.n_cols,
                                        env.grid.n_cells))
    log("  Cell size: %s m, area: %s m^2" % (env.grid.cell_size_m,
          env.grid.cell_area_m2))
    log("  Domain: %s m" % env.grid.domain_size_m)
    log("  Time: %.3f s" % t_env)
    log("  is_ocean dtype: %s, shape: %s" % (env.is_ocean.dtype,
          env.is_ocean.shape))
    log("  elevation dtype: %s, shape: %s" % (env.elevation_m.dtype,
          env.elevation_m.shape))
    log("  land_cover dtype: %s, shape: %s" % (env.land_cover.dtype,
          env.land_cover.shape))
    log("  is_lake dtype: %s, shape: %s" % (env.is_lake.dtype,
          env.is_lake.shape))
    log("  outfall_mask dtype: %s, shape: %s" % (env.outfall_mask.dtype,
          env.outfall_mask.shape))
    log("  n_ocean cells: %d" % int(np.count_nonzero(env.is_ocean)))
    log("  n_lake cells: %d" % int(np.count_nonzero(env.is_lake)))
    log("  n_outfall cells: %d" % int(np.count_nonzero(env.outfall_mask)))
    log("  terrain min/max: %.6f / %.6f m" % (env.elevation_m.min(),
          env.elevation_m.max()))
    log("  terrain finite: %s" % np.all(np.isfinite(env.elevation_m)))
    check("terrain finite", bool(np.all(np.isfinite(env.elevation_m))))
    # Ocean cells are bathymetry (below MSL); land terrain is within [0.2, 60.0].
    land_terrain = env.elevation_m[~env.is_ocean]
    check("land terrain within [0.2, 60.0]",
          bool(np.all(land_terrain >= 0.2) and np.all(land_terrain <= 60.0)))
    check("ocean cells present", bool(int(np.count_nonzero(env.is_ocean)) > 0))
    check("lake cells present", bool(int(np.count_nonzero(env.is_lake)) > 0))
    check("outfall cells present", bool(int(np.count_nonzero(env.outfall_mask)) > 0))
    log("")

    # ------------------------------------------------------------------
    # 2. Road network generation
    # ------------------------------------------------------------------
    log("2. ROAD NETWORK")
    log("  -> Accessing road_network from environment...", flush=True)
    t0 = time.perf_counter()
    network = env.road_network
    t_road = time.perf_counter() - t0

    log("  n_nodes: %d" % network.n_nodes)
    log("  n_edges: %d" % network.n_edges)
    log("  node_rows dtype: %s, shape: %s" % (network.node_rows.dtype,
          network.node_rows.shape))
    log("  node_cols dtype: %s, shape: %s" % (network.node_cols.dtype,
          network.node_cols.shape))
    log("  edge_u dtype: %s, shape: %s" % (network.edge_u.dtype,
          network.edge_u.shape))
    log("  edge_v dtype: %s, shape: %s" % (network.edge_v.dtype,
          network.edge_v.shape))
    log("  edge_path_cells: %d edges" % len(network.edge_path_cells))
    log("  Time (property access): %.3f s" % t_road)

    # Node ordering: verify row-major
    node_flat = network.node_rows * env.grid.n_cols + network.node_cols
    is_row_major = bool(np.all(np.diff(node_flat) > 0))
    log("  node ids row-major sorted: %s" % is_row_major)
    check("node ids row-major sorted", is_row_major)

    # Edge ordering: verify canonical
    edge_pairs = []
    for k in range(network.n_edges):
        u = (int(network.node_rows[network.edge_u[k]]),
             int(network.node_cols[network.edge_u[k]]))
        v = (int(network.node_rows[network.edge_v[k]]),
             int(network.node_cols[network.edge_v[k]]))
        edge_pairs.append((min(u, v), max(u, v)))
    canon_sorted = sorted(edge_pairs, key=lambda p: p)
    edge_ids_canonical = all(edge_pairs[k] == canon_sorted[k]
                             for k in range(network.n_edges))
    log("  edge ids canonical order: %s" % edge_ids_canonical)
    check("edge ids canonical order", edge_ids_canonical)

    # Path endpoint correspondence
    path_endpoint_ok = True
    for k in range(network.n_edges):
        path = network.edge_path_cells[k]
        u_cell = (int(network.node_rows[network.edge_u[k]]),
                  int(network.node_cols[network.edge_u[k]]))
        v_cell = (int(network.node_rows[network.edge_v[k]]),
                  int(network.node_cols[network.edge_v[k]]))
        if path[0] != u_cell or path[-1] != v_cell:
            path_endpoint_ok = False
            break
    log("  path endpoint correspondence: %s" % path_endpoint_ok)
    check("edge path endpoint correspondence", path_endpoint_ok)

    # Path lengths distribution
    path_lens = [len(p) for p in network.edge_path_cells]
    log("  path length min/max: %d / %d cells" % (min(path_lens),
          max(path_lens)))
    log("  two-cell paths: %d" % sum(1 for l in path_lens if l == 2))
    log("  three-cell paths: %d" % sum(1 for l in path_lens if l == 3))
    log("  longer paths (>=4): %d" % sum(1 for l in path_lens if l >= 4))
    check("all paths non-empty", bool(all(l >= 1 for l in path_lens)))

    # Real inspection: verify no edge path contains an ocean cell.
    no_ocean_in_paths = True
    for k in range(network.n_edges):
        path = network.edge_path_cells[k]
        for cell in path:
            r, c = int(cell[0]), int(cell[1])
            if env.is_ocean[r, c]:
                no_ocean_in_paths = False
                break
        if not no_ocean_in_paths:
            break
    check("no ocean cells in any edge path", no_ocean_in_paths)
    log("")

    # ------------------------------------------------------------------
    # 3. Six-hour simulation (dt=30s, 720 steps)
    # ------------------------------------------------------------------
    result = None
    h_initial = None
    t_sim = None

    if not args.no_sim:
        log("3. SIX-HOUR SIMULATION (dt=30s, 720 steps)")
        n_cells = env.grid.n_rows * env.grid.n_cols
        log("  -> Generating drainage fields...", flush=True)
        drainage = generate_drainage(
            master_seed=2027,
            n_cells=n_cells,
            blocked_fraction=0.20,
            drainage_config=config.drainage,
        )
        rain = config.scenarios["R"].rainfall
        tide = config.scenarios["R"].tide
        exch = config.exchange
        runoff = config.runoff
        h_initial = build_initial_depth(env)

        log("  -> Running six-hour simulation (720 steps, dt=30s)...",
            flush=True)
        t0 = time.perf_counter()
        result = run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff,
            dt=30.0, n_steps=720,
        )
        t_sim = time.perf_counter() - t0

        log("  n_steps: %d" % result.n_steps)
        log("  dt: %s s" % result.dt)
        log("  Duration: %s s (%d steps x %s s)" % (result.n_steps * result.dt,
              result.n_steps, result.dt))
        log("  Time: %.3f s" % t_sim)
        log("")
        log("  SimulationResult fields:")
        log("    h_final shape: %s, dtype: %s" % (result.h_final.shape,
              result.h_final.dtype))
        log("    H_peak shape: %s, dtype: %s" % (result.H_peak.shape,
              result.H_peak.dtype))
        log("    h_final finite: %s" % np.all(np.isfinite(result.h_final)))
        log("    H_peak finite: %s" % np.all(np.isfinite(result.H_peak)))
        log("    h_final min/max: %.6f / %.6f m" % (result.h_final.min(),
              result.h_final.max()))
        log("    H_peak min/max: %.6f / %.6f m" % (result.H_peak.min(),
              result.H_peak.max()))
        log("    h_final non-negative: %s" % np.all(result.h_final >= -1e-12))
        log("    H_peak >= h_initial: %s" %
              np.all(result.H_peak >= h_initial))
        log("    H_peak >= h_final: %s" %
              np.all(result.H_peak >= result.h_final))
        log("    S_initial: %.6f m3" % result.S_initial)
        log("    S_final: %.6f m3" % result.S_final)
        log("    V_precip_total: %.6f m3" % result.V_precip_total)
        log("    V_runoff_total: %.6f m3" % result.V_runoff_total)
        log("    V_loss_total: %.6f m3" % result.V_loss_total)
        log("    V_drain_total: %.6f m3" % result.V_drain_total)
        log("    SEA_IN_total: %.6f m3" % result.SEA_IN_total)
        log("    SEA_OUT_total: %.6f m3" % result.SEA_OUT_total)
        log("    internal_face_eval_total: %d" %
              result.internal_face_eval_total)
        log("    IN_total: %.6f m3" % result.IN_total)
        log("    OUT_total: %.6f m3" % result.OUT_total)
        log("    Residual_total: %.6e m3" % result.Residual_total)
        log("    relative_error: %.3e" % result.relative_error)
        log("    Mass conservation (rel err < 1e-9): %s" %
              (result.relative_error < 1e-9))

        # TC12: rainfall normalization
        n_receiving = int(np.count_nonzero(~env.is_ocean))
        rainfall_receiving_area = n_receiving * env.grid.cell_area_m2
        analytical_total_m = rain.total_mm / 1000.0
        p_sim = result.V_precip_total / rainfall_receiving_area
        abs_error = abs(p_sim - analytical_total_m)
        log("")
        log("  TC12 rainfall normalization:")
        log("    rainfall-receiving cells: %d" % n_receiving)
        log("    simulated precipitation depth: %.15f m" % p_sim)
        log("    analytical storm total: %.15f m" % analytical_total_m)
        log("    absolute error: %.3e m (tolerance: 1e-12 m)" % abs_error)
        log("    TC12 PASS: %s" % (abs_error <= 1e-12))

        # Ocean cells excluded from storage
        log("")
        log("  Ocean cells in h_final: all zero = %s" %
              np.all(result.h_final[env.is_ocean] == 0.0))
        log("  H_peak at ocean cells: %s" %
              np.all(result.H_peak[env.is_ocean] == 0.0))
        log("")

        # Simulation-level checks
        check("h_final shape (96, 96)", result.h_final.shape == (96, 96))
        check("H_peak shape (96, 96)", result.H_peak.shape == (96, 96))
        check("h_final dtype float64", result.h_final.dtype == np.float64)
        check("H_peak dtype float64", result.H_peak.dtype == np.float64)
        check("h_final finite", bool(np.all(np.isfinite(result.h_final))))
        check("H_peak finite", bool(np.all(np.isfinite(result.H_peak))))
        check("h_final non-negative (>= -1e-12 round-off tolerance)",
              bool(np.all(result.h_final >= -1e-12)))
        check("H_peak >= h_initial (initial depths included)",
              bool(np.all(result.H_peak >= h_initial)))
        check("H_peak >= h_final (peak covers final)",
              bool(np.all(result.H_peak >= result.h_final)))
        check("ocean cells h_final == 0",
              bool(np.all(result.h_final[env.is_ocean] == 0.0)))
        check("ocean cells H_peak == 0",
              bool(np.all(result.H_peak[env.is_ocean] == 0.0)))
        check("mass conservation (rel err < 1e-9)",
              bool(result.relative_error < 1e-9))
        check("TC12 rainfall normalization (abs err <= 1e-12)",
              bool(abs_error <= 1e-12))
        log("")

    # ------------------------------------------------------------------
    # 4. Node depths
    # ------------------------------------------------------------------
    log("4. NODE DEPTHS")
    if result is not None:
        log("  -> Calling node_depths(result, env)...", flush=True)
        t0 = time.perf_counter()
        node_depth_array = node_depths(result, env)
        t_node = time.perf_counter() - t0
        log("  Time: %.6f s" % t_node)
        log("  shape: %s (expected: (%d,))" % (node_depth_array.shape,
              network.n_nodes))
        log("  dtype: %s (expected: %s)" % (node_depth_array.dtype,
              result.h_final.dtype))
        log("  finite: %s" % np.all(np.isfinite(node_depth_array)))
        log("  non-negative: %s" % np.all(node_depth_array >= -1e-12))

        # Verify against h_final
        node_rows = network.node_rows.astype(np.int64)
        node_cols = network.node_cols.astype(np.int64)
        expected = result.h_final[node_rows, node_cols]
        node_matches = node_depth_array.tobytes() == expected.tobytes()
        log("  matches h_final indexing: %s" % node_matches)
        node_no_alias = not np.shares_memory(node_depth_array, result.h_final)
        log("  fresh copy (no shared memory): %s" % node_no_alias)

        # Caller mutation isolation: capture source and extracted values,
        # mutate the output, verify the output changed and the source did not.
        original_source_value = result.h_final[node_rows[0], node_cols[0]]
        original_extracted_value = node_depth_array[0]
        node_depth_array[0] = 999.0
        extracted_changed = node_depth_array[0] != original_extracted_value
        source_unchanged = (
            result.h_final[node_rows[0], node_cols[0]] == original_source_value
        )
        node_mutation_isolated = extracted_changed and source_unchanged
        log("  extracted value changed: %s" % extracted_changed)
        log("  source cell unchanged: %s" % source_unchanged)
        log("  caller mutation isolated: %s" % node_mutation_isolated)
        # Restore
        node_depth_array = node_depths(result, env)

        check("node_depths shape == (n_nodes,)",
              node_depth_array.shape == (network.n_nodes,))
        check("node_depths dtype follows h_final",
              node_depth_array.dtype == result.h_final.dtype)
        check("node_depths finite", bool(np.all(np.isfinite(node_depth_array))))
        check("node_depths matches h_final indexing", node_matches)
        check("node_depths fresh copy (no aliasing)", node_no_alias)
        check("node_depths caller mutation isolated", node_mutation_isolated)
    else:
        # Lightweight contract check with synthetic h_final
        log("  (simulation skipped; lightweight contract check)")
        h_final = np.arange(9, dtype=np.float64).reshape(3, 3)
        from types import SimpleNamespace
        result_syn = SimpleNamespace(h_final=h_final)
        env_syn = SimpleNamespace(
            elevation_m=np.zeros((3, 3), dtype=np.float64),
            road_network=SimpleNamespace(
                node_rows=np.array([0, 1, 2], dtype=np.int16),
                node_cols=np.array([0, 1, 2], dtype=np.int16),
            ),
        )
        node_depth_array = node_depths(result_syn, env_syn)
        log("  shape: %s" % (node_depth_array.shape,))
        log("  dtype: %s" % node_depth_array.dtype)
        log("  values: %s" % node_depth_array)
        node_ok = (
            node_depth_array.shape == (3,) and
            node_depth_array.dtype == np.float64 and
            node_depth_array[0] == 0.0 and
            node_depth_array[1] == 4.0 and
            node_depth_array[2] == 8.0 and
            not np.shares_memory(node_depth_array, h_final)
        )
        check("node_depths lightweight contract", node_ok)
    log("")

    # ------------------------------------------------------------------
    # 5. Edge depths
    # ------------------------------------------------------------------
    log("5. EDGE DEPTHS")
    if result is not None:
        log("  -> Calling edge_depths(result, env)...", flush=True)
        t0 = time.perf_counter()
        edge_depth_array = edge_depths(result, env)
        t_edge = time.perf_counter() - t0
        log("  Time: %.6f s" % t_edge)
        log("  shape: %s (expected: (%d, 3))" % (edge_depth_array.shape,
              network.n_edges))
        log("  dtype: %s (expected: %s)" % (edge_depth_array.dtype,
              result.h_final.dtype))
        log("  finite: %s" % np.all(np.isfinite(edge_depth_array)))

        # Verify against direct computation
        edge_match = True
        for k in range(network.n_edges):
            path = network.edge_path_cells[k]
            n_cells_path = len(path)
            rows = np.fromiter((c[0] for c in path), dtype=np.int64,
                               count=n_cells_path)
            cols = np.fromiter((c[1] for c in path), dtype=np.int64,
                               count=n_cells_path)
            depths = result.h_final[rows, cols]
            for j, f in enumerate((0.0, 0.5, 1.0)):
                x = f * (n_cells_path - 1)
                i = int(np.floor(x))
                alpha = x - i
                if alpha == 0.0:
                    expected_val = depths[i]
                else:
                    expected_val = (1.0 - alpha) * depths[i] + alpha * depths[i + 1]
                if edge_depth_array[k, j] != expected_val:
                    edge_match = False
                    log("  MISMATCH edge %d col %d: %r != %r" %
                        (k, j, edge_depth_array[k, j], expected_val))
        log("  matches direct computation: %s" % edge_match)
        edge_no_alias = not np.shares_memory(edge_depth_array, result.h_final)
        log("  fresh copy (no shared memory): %s" % edge_no_alias)

        # Caller mutation isolation: use an actual cell belonging to edge 0's
        # path. Column 0 is the start cell = path[0].
        first_path = network.edge_path_cells[0]
        first_row, first_col = first_path[0]
        original_source_value = result.h_final[first_row, first_col]
        original_extracted_value = edge_depth_array[0, 0]
        edge_depth_array[0, 0] = 999.0
        extracted_changed = edge_depth_array[0, 0] != original_extracted_value
        source_unchanged = (
            result.h_final[first_row, first_col] == original_source_value
        )
        edge_mutation_isolated = extracted_changed and source_unchanged
        log("  extracted value changed: %s" % extracted_changed)
        log("  source cell unchanged: %s" % source_unchanged)
        log("  caller mutation isolated: %s" % edge_mutation_isolated)

        check("edge_depths shape == (n_edges, 3)",
              edge_depth_array.shape == (network.n_edges, 3))
        check("edge_depths dtype follows h_final",
              edge_depth_array.dtype == result.h_final.dtype)
        check("edge_depths finite", bool(np.all(np.isfinite(edge_depth_array))))
        check("edge_depths matches direct computation", edge_match)
        check("edge_depths fresh copy (no aliasing)", edge_no_alias)
        check("edge_depths caller mutation isolated", edge_mutation_isolated)
    else:
        # Lightweight contract check with synthetic h_final
        log("  (simulation skipped; lightweight contract check)")
        h_final = np.arange(9, dtype=np.float64).reshape(3, 3)
        from types import SimpleNamespace
        result_syn = SimpleNamespace(h_final=h_final)
        env_syn = SimpleNamespace(
            elevation_m=np.zeros((3, 3), dtype=np.float64),
            road_network=SimpleNamespace(
                edge_path_cells=(
                    ((0, 0), (0, 1)),
                    ((0, 1), (0, 2)),
                ),
                n_edges=2,
            ),
        )
        edge_depth_array = edge_depths(result_syn, env_syn)
        log("  shape: %s" % (edge_depth_array.shape,))
        log("  dtype: %s" % edge_depth_array.dtype)
        # Edge 0: path [(0,0),(0,1)] -> depths [0.0, 1.0] -> [0.0, 0.5, 1.0]
        # Edge 1: path [(0,1),(0,2)] -> depths [1.0, 2.0] -> [1.0, 1.5, 2.0]
        edge_ok = (
            edge_depth_array.shape == (2, 3) and
            edge_depth_array.dtype == np.float64 and
            edge_depth_array[0, 0] == 0.0 and
            abs(edge_depth_array[0, 1] - 0.5) < 1e-12 and
            edge_depth_array[0, 2] == 1.0 and
            edge_depth_array[1, 0] == 1.0 and
            abs(edge_depth_array[1, 1] - 1.5) < 1e-12 and
            edge_depth_array[1, 2] == 2.0 and
            not np.shares_memory(edge_depth_array, h_final)
        )
        check("edge_depths lightweight contract", edge_ok)
    log("")

    # ------------------------------------------------------------------
    # 6. Reproducibility (bitwise) - optional second simulation
    # ------------------------------------------------------------------
    h_peak_bitwise = None
    h_final_bitwise = None
    node_bitwise = None
    edge_bitwise = None
    t_sim2 = None

    if args.repro:
        if result is None:
            log("6. REPRODUCIBILITY (BITWISE)")
            log("  SKIPPED: --no-sim is set; cannot reproduce without a simulation.")
            log("")
        else:
            log("6. REPRODUCIBILITY (BITWISE)")
            log("  -> Running SECOND six-hour simulation (720 steps, dt=30s)...",
                flush=True)
            t0 = time.perf_counter()
            result2 = run_simulation(
                h_initial, env, drainage, rain, tide, exch, runoff,
                dt=30.0, n_steps=720,
            )
            t_sim2 = time.perf_counter() - t0

            log("  Second simulation time: %.3f s" % t_sim2)
            h_peak_bitwise = result.H_peak.tobytes() == result2.H_peak.tobytes()
            h_final_bitwise = result.h_final.tobytes() == result2.h_final.tobytes()
            node_bitwise = node_depths(result, env).tobytes() == \
                node_depths(result2, env).tobytes()
            edge_bitwise = edge_depths(result, env).tobytes() == \
                edge_depths(result2, env).tobytes()

            log("  H_peak bitwise identical: %s" % h_peak_bitwise)
            log("  h_final bitwise identical: %s" % h_final_bitwise)
            log("  node_depths bitwise identical: %s" % node_bitwise)
            log("  edge_depths bitwise identical: %s" % edge_bitwise)
            log("  S_final bitwise identical: %s" %
                  (result.S_final == result2.S_final))
            log("  relative_error bitwise identical: %s" %
                  (result.relative_error == result2.relative_error))
            log("")

            check("H_peak bitwise reproducible", h_peak_bitwise)
            check("h_final bitwise reproducible", h_final_bitwise)
            check("node_depths bitwise reproducible", node_bitwise)
            check("edge_depths bitwise reproducible", edge_bitwise)
            check("S_final bitwise reproducible",
                  bool(result.S_final == result2.S_final))
            check("relative_error bitwise reproducible",
                  bool(result.relative_error == result2.relative_error))
    else:
        log("6. REPRODUCIBILITY (BITWISE)")
        log("  SKIPPED: pass --repro to run the second six-hour simulation.")
        log("")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)

    failed = [name for name, ok in checks if not ok]
    passed = [name for name, ok in checks if ok]

    log("  Total checks: %d" % len(checks))
    log("  Passed: %d" % len(passed))
    log("  Failed: %d" % len(failed))
    for name in failed:
        log("    FAILED: %s" % name)

    all_pass = len(failed) == 0
    log("  ALL CHECKS PASS: %s" % all_pass)
    log("")
    log("  Environment gen:  %.3f s" % t_env)
    if t_sim is not None:
        log("  Simulation:       %.3f s" % t_sim)
    if t_sim2 is not None:
        log("  Simulation (rep): %.3f s" % t_sim2)
    log("")

    if all_pass:
        log("RESULT: ALL DIAGNOSTIC CHECKS PASSED")
        return 0
    else:
        log("RESULT: %d DIAGNOSTIC CHECK(S) FAILED" % len(failed))
        return 1


if __name__ == "__main__":
    sys.exit(main())
