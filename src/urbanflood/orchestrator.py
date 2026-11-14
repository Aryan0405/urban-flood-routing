"""SYNTHETIC multi-timestep orchestrator for Phase 2 flood simulator.

Wraps the existing step_timestep() numerical engine to execute a sequence
of timesteps, tracking cumulative state and H_peak.

Does not duplicate or modify any numerical calculations from step_timestep.
All flux calculations, donor-availability limiting, and synchronous updates
are performed exclusively by step_timestep().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbanflood.config import (
    ExchangeConfig,
    RainScenario,
    RunoffConfig,
    TideConfig,
)
from urbanflood.drainage import DrainageFields
from urbanflood.environment import SyntheticEnvironment
from urbanflood.simulator import step_timestep


@dataclass(frozen=True)
class SimulationResult:
    """Result of a multi-timestep flood simulation.

    All volume terms are cumulative over all executed timesteps, in m3.
    H_peak is the element-wise maximum water depth over the simulation,
    initialized from the initial depth.
    internal_face_eval_total is the cumulative count of internal face
    evaluations across all timesteps. Sea faces excluded.
    """

    h_final: np.ndarray
    H_peak: np.ndarray
    S_initial: float
    S_final: float
    V_precip_total: float
    V_runoff_total: float
    V_loss_total: float
    V_drain_total: float
    SEA_IN_total: float
    SEA_OUT_total: float
    internal_inflow_total: float
    internal_outflow_total: float
    internal_face_eval_total: int
    IN_total: float
    OUT_total: float
    Residual_total: float
    relative_error: float
    n_steps: int
    dt: float


def run_simulation(
    h_initial: np.ndarray,
    env: SyntheticEnvironment,
    drainage: DrainageFields,
    rain: RainScenario,
    tide: TideConfig,
    exchange_config: ExchangeConfig,
    runoff_config: RunoffConfig,
    dt: float,
    n_steps: int,
) -> SimulationResult:
    """Execute a multi-timestep flood simulation.

    Repeatedly calls step_timestep(), propagating state between steps.
    H_peak is initialized from h_initial and updated element-wise as the
    maximum of the previous peak and each new depth.

    Parameters
    ----------
    h_initial : np.ndarray
        Initial water depths, shape (n_rows, n_cols), metres.
    env : SyntheticEnvironment
        The synthetic environment (terrain, masks, outfalls).
    drainage : DrainageFields
        Hidden drainage fields (capacity, efficiency, blocked).
    rain : RainScenario
        Rainfall scenario (used for interval integration).
    tide : TideConfig
        Tide configuration (midpoint evaluation).
    exchange_config : ExchangeConfig
        Exchange model parameters (exchange velocity c).
    runoff_config : RunoffConfig
        Runoff coefficients by land cover.
    dt : float
        Timestep in seconds. Must be positive.
    n_steps : int
        Number of timesteps to execute. Must be at least 1.

    Returns
    -------
    SimulationResult
        Cumulative simulation result over all timesteps.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    n_rows, n_cols = env.elevation_m.shape
    if h_initial.shape != (n_rows, n_cols):
        raise ValueError(
            f"h_initial shape {h_initial.shape} does not match "
            f"environment grid shape {(n_rows, n_cols)}"
        )

    h_now = np.array(h_initial, dtype=np.float64, copy=True)
    H_peak = np.array(h_initial, dtype=np.float64, copy=True)

    V_precip_total = 0.0
    V_runoff_total = 0.0
    V_loss_total = 0.0
    V_drain_total = 0.0
    SEA_IN_total = 0.0
    SEA_OUT_total = 0.0
    internal_inflow_total = 0.0
    internal_outflow_total = 0.0
    internal_face_eval_total = 0

    S_initial = None
    S_final = None

    for k in range(n_steps):
        result = step_timestep(
            h_now,
            env,
            drainage,
            rain,
            tide,
            exchange_config,
            runoff_config,
            dt,
            k,
        )

        if S_initial is None:
            S_initial = result.S_before

        S_final = result.S_after

        V_precip_total += result.V_precip_total
        V_runoff_total += result.V_runoff_total
        V_loss_total += result.V_loss_total
        V_drain_total += result.V_drain_total
        SEA_IN_total += result.SEA_IN_total
        SEA_OUT_total += result.SEA_OUT_total
        internal_inflow_total += result.internal_inflow_total
        internal_outflow_total += result.internal_outflow_total
        internal_face_eval_total += result.internal_face_eval_count

        H_peak = np.maximum(H_peak, result.h_new)
        h_now = result.h_new

    # Global mass ledger (spec section 10):
    #   IN      = sum(V_precip) + SEA_IN
    #   OUT     = sum(V_loss) + sum(V_drain) + SEA_OUT
    #   dS      = S_final - S_initial
    #   Residual = IN - OUT - dS
    # Internal transfers (internal_inflow/outflow) cancel internally and are
    # NOT counted as external inputs or outputs.
    IN_total = V_precip_total + SEA_IN_total
    OUT_total = V_loss_total + V_drain_total + SEA_OUT_total
    Residual_total = IN_total - OUT_total - (S_final - S_initial)
    relative_error = abs(Residual_total) / max(IN_total, OUT_total, 1.0)

    return SimulationResult(
        h_final=h_now,
        H_peak=H_peak,
        S_initial=S_initial,
        S_final=S_final,
        V_precip_total=V_precip_total,
        V_runoff_total=V_runoff_total,
        V_loss_total=V_loss_total,
        V_drain_total=V_drain_total,
        SEA_IN_total=SEA_IN_total,
        SEA_OUT_total=SEA_OUT_total,
        internal_inflow_total=internal_inflow_total,
        internal_outflow_total=internal_outflow_total,
        internal_face_eval_total=internal_face_eval_total,
        IN_total=IN_total,
        OUT_total=OUT_total,
        Residual_total=Residual_total,
        relative_error=relative_error,
        n_steps=n_steps,
        dt=dt,
    )


def compute_tc10_metrics(
    h_peak_30: np.ndarray,
    h_peak_15: np.ndarray,
    is_ocean: np.ndarray,
    S30: float,
    S15: float,
) -> tuple[float, float, float]:
    """Compute TC10 timestep sensitivity metrics.
    
    Compares H_peak arrays from 30s and 15s simulations over all simulated
    surface-water cells excluding ocean cells.
    
    Returns (D_max, RMSE, D_storage).
    
    Raises ValueError if arrays have incompatible shapes or no valid cells.
    """
    if h_peak_30.shape != h_peak_15.shape:
        raise ValueError(
            f"H_peak shape mismatch: {h_peak_30.shape} vs {h_peak_15.shape}"
        )
    if h_peak_30.shape != is_ocean.shape:
        raise ValueError(
            f"H_peak shape {h_peak_30.shape} does not match "
            f"is_ocean shape {is_ocean.shape}"
        )
    
    valid_mask = ~is_ocean
    hp_30 = h_peak_30[valid_mask]
    hp_15 = h_peak_15[valid_mask]
    
    if hp_30.size == 0:
        raise ValueError("No valid (non-ocean) cells for comparison")
    
    if not np.all(np.isfinite(hp_30)) or not np.all(np.isfinite(hp_15)):
        raise ValueError("H_peak arrays contain non-finite values")
    
    difference = hp_30 - hp_15
    D_max = float(np.max(np.abs(difference)))
    RMSE = float(np.sqrt(np.mean(difference ** 2)))
    
    if abs(S30) == 0.0 and abs(S15) == 0.0:
        D_storage = 0.0
    else:
        D_storage = abs(S30 - S15) / max(abs(S30), abs(S15))
    
    return D_max, RMSE, D_storage


def evaluate_tc10_invariance(
    h_peak_30: np.ndarray,
    h_peak_15: np.ndarray,
    is_ocean: np.ndarray,
    S30: float,
    S15: float,
) -> dict:
    """Evaluate TC10 canonical-order invariance metrics.

    Compares H_peak arrays from 30s and 15s simulations over all simulated
    surface-water cells excluding ocean cells.

    Returns a dict with D_max, RMSE, D_storage, and per-criterion pass flags.
    """
    D_max, RMSE, D_storage = compute_tc10_metrics(
        h_peak_30, h_peak_15, is_ocean, S30, S15
    )

    return {
        'D_max': D_max,
        'RMSE': RMSE,
        'D_storage': D_storage,
        'd_max_pass': D_max <= 0.05,
        'rmse_pass': RMSE <= 0.01,
        'd_storage_pass': D_storage <= 0.01,
        'all_pass': D_max <= 0.05 and RMSE <= 0.01 and D_storage <= 0.01,
    }


def evaluate_tc10_sensitivity(
    h_initial: np.ndarray,
    env: SyntheticEnvironment,
    drainage: DrainageFields,
    rain: RainScenario,
    tide: TideConfig,
    exchange_config: ExchangeConfig,
    runoff_config: RunoffConfig,
    result_30: SimulationResult | None = None,
) -> dict:
    """Run TC10 timestep-sensitivity orchestration: 30s vs 15s, with 7.5s fallback.

    Executes the initial 30-second (720-step) and 15-second (1440-step)
    simulations, evaluates all three TC10 criteria, and - only if any
    criterion fails - runs exactly one fallback at dt=7.5s with
    n_steps=2880 comparing the 15s result against the 7.5s result.

    Parameters
    ----------
    h_initial, env, drainage, rain, tide, exchange_config, runoff_config :
        Shared simulation inputs passed unchanged to every run_simulation()
        call.
    result_30 : SimulationResult or None
        Pre-computed 30s result. If None, the 30s simulation is executed
        internally. Provided by callers that already hold the 30s result
        (e.g. the six-hour Scenario R test).

    Returns
    -------
    dict
        initial_metrics : dict from evaluate_tc10_invariance (30s vs 15s)
        fallback_metrics : dict or None from evaluate_tc10_invariance
            (15s vs 7.5s); None when no fallback was attempted.
        fallback_attempted : bool
        fallback_passed : bool or None
        result_30 : SimulationResult
        result_15 : SimulationResult
        result_7_5 : SimulationResult or None
    """
    if result_30 is None:
        result_30 = run_simulation(
            h_initial, env, drainage, rain, tide,
            exchange_config, runoff_config,
            dt=30.0, n_steps=720,
        )

    result_15 = run_simulation(
        h_initial, env, drainage, rain, tide,
        exchange_config, runoff_config,
        dt=15.0, n_steps=1440,
    )

    initial_metrics = evaluate_tc10_invariance(
        result_30.H_peak,
        result_15.H_peak,
        env.is_ocean,
        result_30.S_final,
        result_15.S_final,
    )

    if initial_metrics['all_pass']:
        return {
            'initial_metrics': initial_metrics,
            'fallback_metrics': None,
            'fallback_attempted': False,
            'fallback_passed': None,
            'result_30': result_30,
            'result_15': result_15,
            'result_7_5': None,
        }

    result_7_5 = run_simulation(
        h_initial, env, drainage, rain, tide,
        exchange_config, runoff_config,
        dt=7.5, n_steps=2880,
    )

    fallback_metrics = evaluate_tc10_invariance(
        result_15.H_peak,
        result_7_5.H_peak,
        env.is_ocean,
        result_15.S_final,
        result_7_5.S_final,
    )

    return {
        'initial_metrics': initial_metrics,
        'fallback_metrics': fallback_metrics,
        'fallback_attempted': True,
        'fallback_passed': fallback_metrics['all_pass'],
        'result_30': result_30,
        'result_15': result_15,
        'result_7_5': result_7_5,
    }


def node_depths(
    result: SimulationResult,
    env: SyntheticEnvironment,
) -> np.ndarray:
    """Final water depth at each road-network node cell.

    Node-depth output contract (docs/PHASE2_NUMERICAL_SPEC.md section 14):

    * What: final water depth in metres at the end of the simulation at each
      road-network node cell. This is a final-depth output only, not a time
      series; for peak node depths see peak_node_depths().
    * What is a node: a land cell on any road line, as defined by
      env.road_network.node_rows / node_cols. Lake cells and ocean cells are
      never nodes (see roads.py), so by construction every node cell is a
      simulated surface-water cell with a defined final depth. No ocean or
      lake special-casing is required beyond correct cell indexing.
    * Identity and ordering: node id k indexes
      env.road_network.node_rows[k] / node_cols[k], the row-major sorted node
      list from roads.py. Ordering is canonical and deterministic, derived
      only from grid geometry. Output order follows node id order.
    * Derivation: depth[k] = result.h_final[node_rows[k], node_cols[k]].
      Values are taken from h_final (final cell depths), not H_peak.
    * Shape and dtype: (n_nodes,), metres. The output dtype follows
      result.h_final exactly (no independent dtype conversion is applied);
      the broader output container and dtype decision remains deferred
      (spec section 17). The returned array is a fresh copy;
      result.h_final is not modified and no aliasing is exposed.

    Parameters
    ----------
    result : SimulationResult
        Completed multi-timestep simulation result; h_final is used.
    env : SyntheticEnvironment
        Environment providing the road-network node geometry.

    Returns
    -------
    np.ndarray
        Final node depths in metres, shape (n_nodes,), indexed by node id.
        The dtype follows result.h_final.

    Raises
    ------
    ValueError
        If result.h_final does not match the environment grid shape.
    """
    h_final = np.asarray(result.h_final)
    return _node_depths_from_cells(h_final, env)


def _node_depths_from_cells(
    cell_depths: np.ndarray,
    env: SyntheticEnvironment,
) -> np.ndarray:
    """Sample cell depths at road-network node cells.

    Private helper shared by node_depths() and peak_node_depths(). Returns a
    fresh copy of cell_depths indexed by the canonical node rows/cols.

    Parameters
    ----------
    cell_depths : np.ndarray
        Cell depths in metres, shape (n_rows, n_cols), any dtype.
    env : SyntheticEnvironment
        Environment providing the road-network node geometry.

    Returns
    -------
    np.ndarray
        Depths at node cells, shape (n_nodes,), dtype follows cell_depths.
        Fresh copy; no aliasing into cell_depths.
    """
    if cell_depths.shape != env.elevation_m.shape:
        raise ValueError(
            "cell_depths shape "
            f"{cell_depths.shape} does not match environment grid shape "
            f"{env.elevation_m.shape}"
        )
    node_rows = env.road_network.node_rows
    node_cols = env.road_network.node_cols
    # Advanced indexing always returns a copy: caller mutation cannot affect
    # cell_depths, and no view into internal state is exposed.
    return cell_depths[node_rows, node_cols]


def edge_depths(
    result: SimulationResult,
    env: SyntheticEnvironment,
) -> np.ndarray:
    """Final water-depth samples at three positions along each road-network edge.

    Edge-depth output contract (docs/PHASE2_NUMERICAL_SPEC.md section 14):

    * What: final water depth in metres at three positions along each
      road-network edge: column 0 is the canonical start (0% of edge length),
      column 1 is the geometric midpoint (50% of edge length), and column 2
      is the canonical end (100% of edge length). This is a final-depth
      output only, not a time series; for peak edge depths see
      peak_edge_depths().
    * What is an edge: a road-network edge as defined by
      env.road_network.edge_u / edge_v / edge_path_cells. Edge ids index the
      canonically sorted edge list from roads.py. Ordering is canonical and
      deterministic, derived only from grid geometry. Output row order
      follows edge id order.
    * Canonical orientation: each edge path begins at the edge_u endpoint
      cell and ends at the edge_v endpoint cell, as guaranteed by the road
      generator's increasing-coordinate line traversal. Path orientation is
      preserved exactly; paths are never sorted or reversed based on
      coordinates.
    * Derivation: for a path containing N cells with final depths
      h_0, ..., h_{N-1} taken from result.h_final, a fractional edge
      position f in {0, 0.5, 1} maps to the path coordinate
      x = f * (N - 1). Let i = floor(x) and alpha = x - i. The sample is
      h(f) = (1 - alpha) * h_i + alpha * h_{i+1} when x is not an integer,
      and h(f) = h_i when x is an integer. Consecutive path cells are
      equally spaced at one cell size (50 m), so fractional positions
      correspond to equal fractions of edge length.
    * Bridge edges: bridge paths contain land abutments and intervening
      lake cells. Samples falling on lake cells represent water depth in the
      underlying lake terrain cells, not water depth on the bridge deck. The
      current model does not simulate bridge-deck elevation, bridge
      clearance, or bridge-deck inundation. These samples must not be used
      to claim that a bridge is passable or impassable based solely on
      their values.
    * Shape and dtype: (n_edges, 3), metres. The output dtype follows
      result.h_final exactly (no independent dtype conversion is applied);
      the broader output container and dtype decision remains deferred
      (spec section 17). The returned array is a fresh copy;
      result.h_final is not modified and no aliasing is exposed.

    Parameters
    ----------
    result : SimulationResult
        Completed multi-timestep simulation result; h_final is used.
    env : SyntheticEnvironment
        Environment providing the road-network edge geometry.

    Returns
    -------
    np.ndarray
        Final edge-depth samples in metres, shape (n_edges, 3), indexed by
        edge id. Column 0 is start, column 1 is midpoint, column 2 is end.
        The dtype follows result.h_final.

    Raises
    ------
    ValueError
        If result.h_final does not match the environment grid shape, or if
        any edge path is empty or contains out-of-bounds cell indices.
    """
    h_final = np.asarray(result.h_final)
    return _edge_depths_from_cells(h_final, env)


def _edge_depths_from_cells(
    cell_depths: np.ndarray,
    env: SyntheticEnvironment,
) -> np.ndarray:
    """Sample cell depths at three positions along each road-network edge.

    Private helper shared by edge_depths() and peak_edge_depths(). Column 0 is
    the canonical start (0% of edge length), column 1 is the geometric midpoint
    (50%), column 2 is the canonical end (100%). Uses piecewise-linear
    interpolation along the ordered edge_path_cells sequence.

    Parameters
    ----------
    cell_depths : np.ndarray
        Cell depths in metres, shape (n_rows, n_cols), any dtype.
    env : SyntheticEnvironment
        Environment providing the road-network edge geometry.

    Returns
    -------
    np.ndarray
        Edge-depth samples, shape (n_edges, 3), dtype follows cell_depths.
        Fresh copy; no aliasing into cell_depths.
    """
    if cell_depths.shape != env.elevation_m.shape:
        raise ValueError(
            "cell_depths shape "
            f"{cell_depths.shape} does not match environment grid shape "
            f"{env.elevation_m.shape}"
        )

    network = env.road_network
    n_edges = network.n_edges
    n_rows, n_cols = env.elevation_m.shape

    # Fractional edge positions: start (0%), midpoint (50%), end (100%).
    fractions = np.array([0.0, 0.5, 1.0])

    # Output array preserves the dtype of cell_depths; no independent
    # dtype conversion is applied (spec section 17 defers that decision).
    output = np.empty((n_edges, 3), dtype=cell_depths.dtype)

    for k in range(n_edges):
        path = network.edge_path_cells[k]
        n_cells = len(path)
        if n_cells == 0:
            raise ValueError(
                f"edge {k} has an empty path; generator invariant violated"
            )

        # Extract depths along the path, preserving canonical orientation.
        # path cells are (row, col) tuples ordered from edge_u to edge_v.
        rows = np.fromiter((cell[0] for cell in path), dtype=np.int64, count=n_cells)
        cols = np.fromiter((cell[1] for cell in path), dtype=np.int64, count=n_cells)

        # Validate cell indices are within grid bounds.
        if np.any(rows < 0) or np.any(rows >= n_rows) or np.any(cols < 0) or np.any(cols >= n_cols):
            raise ValueError(
                f"edge {k} path contains cell indices outside grid bounds; "
                "generator invariant violated"
            )

        depths = cell_depths[rows, cols]

        for j, f in enumerate(fractions):
            x = f * (n_cells - 1)
            i = int(np.floor(x))
            alpha = x - i
            if alpha == 0.0:
                output[k, j] = depths[i]
            else:
                output[k, j] = (1.0 - alpha) * depths[i] + alpha * depths[i + 1]

    return output


def peak_node_depths(
    result: SimulationResult,
    env: SyntheticEnvironment,
) -> np.ndarray:
    """Peak water depth at each road-network node cell.

    Peak = running maximum of cell depth over the whole simulation, including
    the initial depth (SimulationResult.H_peak is initialized from h_initial
    and updated as the element-wise maximum with each new depth).

    Sampled from result.H_peak at the canonical node cells, indexed by node id
    order. Shape (n_nodes,), metres. Dtype follows result.H_peak exactly; no
    independent dtype conversion is applied. The returned array is a fresh
    copy; result.H_peak is not modified and no aliasing is exposed.

    The three edge samples (see peak_edge_depths) are the same start/mid/end
    positions as edge_depths, but sampled from H_peak. This is a SAMPLED peak
    (start/mid/end interpolated from cell peaks), not the maximum over every
    cell on the path.

    Bridge edges follow the same caveat as edge_depths: samples falling on
    lake cells represent peak depth in the underlying lake terrain cells, not
    peak depth on the bridge deck. The current model does not simulate
    bridge-deck elevation, bridge clearance, or bridge-deck inundation.

    Parameters
    ----------
    result : SimulationResult
        Completed multi-timestep simulation result; H_peak is used.
    env : SyntheticEnvironment
        Environment providing the road-network node geometry.

    Returns
    -------
    np.ndarray
        Peak node depths in metres, shape (n_nodes,), indexed by node id.
        The dtype follows result.H_peak.

    Raises
    ------
    ValueError
        If result.H_peak does not match the environment grid shape.
    """
    h_peak = np.asarray(result.H_peak)
    return _node_depths_from_cells(h_peak, env)


def peak_edge_depths(
    result: SimulationResult,
    env: SyntheticEnvironment,
) -> np.ndarray:
    """Peak water-depth samples at three positions along each road-network edge.

    Sampled from result.H_peak at start (column 0), geometric midpoint
    (column 1), and canonical end (column 2) using the same piecewise-linear
    interpolation rule as edge_depths. Shape (n_edges, 3), metres. Dtype
    follows result.H_peak exactly; no independent dtype conversion is applied.
    The returned array is a fresh copy; result.H_peak is not modified and no
    aliasing is exposed.

    This is a SAMPLED peak (start/mid/end interpolated from cell peaks), not
    the maximum over every cell on the path. Bridge edges follow the same caveat
    as edge_depths.

    Parameters
    ----------
    result : SimulationResult
        Completed multi-timestep simulation result; H_peak is used.
    env : SyntheticEnvironment
        Environment providing the road-network edge geometry.

    Returns
    -------
    np.ndarray
        Peak edge-depth samples in metres, shape (n_edges, 3), indexed by edge
        id. Column 0 is start, column 1 is midpoint, column 2 is end.
        The dtype follows result.H_peak.

    Raises
    ------
    ValueError
        If result.H_peak does not match the environment grid shape, or if any
        edge path is empty or contains out-of-bounds cell indices.
    """
    h_peak = np.asarray(result.H_peak)
    return _edge_depths_from_cells(h_peak, env)
