r"""App core: pure data loading and routing helpers for the Streamlit app.

Imports only numpy, pathlib, typing, csv, __future__ and urbanflood modules.
No streamlit, no matplotlib. Nothing is cached here (the app uses
st.cache_data). Everything is synthetic recovery testing data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from urbanflood.app_data import (
    SCENARIO_NAMES,
    hpeak_filename,
    read_hpeak_text,
    read_manifest,
)
from urbanflood.config import build_default_config
from urbanflood.environment import build_environment
from urbanflood.orchestrator import peak_edge_depths
from urbanflood.rng import STREAM_OD_PAIRS, derive_stream_seed
from urbanflood.routing import (
    effective_edge_depth,
    policy_costs,
    route_metrics,
    sample_od_pairs,
    solve_routes,
)
from types import SimpleNamespace

APP_DATA_DIR = Path(__file__).resolve().parents[2] / "app_data"
OD_MIN_DISTANCE_M = 1000.0


def load_manifest(app_data_dir: Optional[Path] = None) -> Dict[int, int]:
    """Return {index: seed} from the app_data manifest."""
    if app_data_dir is None:
        app_data_dir = APP_DATA_DIR
    return read_manifest(app_data_dir / "manifest.txt")


def load_hpeak(seed_index: int, scenario: str, app_data_dir: Optional[Path] = None) -> np.ndarray:
    """Load a H_peak text array (float64, 2D) from app_data."""
    if app_data_dir is None:
        app_data_dir = APP_DATA_DIR
    name = hpeak_filename(seed_index, scenario)
    return read_hpeak_text(app_data_dir / name)


def build_env(seed_index: int):
    """Build (env, cfg) for seed_index using config env_experiment_seeds order."""
    cfg = build_default_config()
    seeds = list(cfg.seeds.env_experiment_seeds())
    seed = seeds[int(seed_index)]
    env = build_environment(seed, cfg)
    return env, cfg


def build_d_e(env, hpeak: np.ndarray) -> np.ndarray:
    """Build d_e exactly as run_experiment_c.py does."""
    network = env.road_network
    peak = peak_edge_depths(SimpleNamespace(H_peak=hpeak), env)
    surface = peak_edge_depths(
        SimpleNamespace(H_peak=env.elevation_m + hpeak), env
    )
    return effective_edge_depth(
        peak, surface, network.edge_is_bridge, network.edge_deck_m
    )


def _od_pairs(env, cfg) -> np.ndarray:
    """Derive the 20 OD pairs for the environment's seed (same as run_experiment_c.py)."""
    network = env.road_network
    cell_size_m = float(env.grid.cell_size_m)
    od_seed = derive_stream_seed(int(env.master_seed), STREAM_OD_PAIRS)
    return sample_od_pairs(
        network.node_rows,
        network.node_cols,
        cell_size_m,
        int(cfg.routing.n_od_pairs),
        OD_MIN_DISTANCE_M,
        od_seed,
    )


def compute_routes(
    env, d_e: np.ndarray, threshold_m: float, lam: float, pair_index: int
) -> dict:
    """Route one OD pair under P0 and the closure policy.

    Returns dict with keys:
      p0_edges, policy_edges (lists of edge ids or None if unreachable),
      p0_reachable, policy_reachable, closed_edge_ids (list), od (u, v).
    """
    network = env.road_network
    cfg = build_default_config()
    od_pairs = _od_pairs(env, cfg)
    n_nodes = int(network.n_nodes)
    edge_u = network.edge_u
    edge_v = network.edge_v
    times = network.edge_free_flow_time_s

    cost_p0, open_p0 = policy_costs(times, d_e, None, 0.0)
    cost_pol, open_pol = policy_costs(times, d_e, threshold_m, lam)

    routes_p0 = solve_routes(n_nodes, edge_u, edge_v, cost_p0, open_p0, od_pairs)
    routes_pol = solve_routes(n_nodes, edge_u, edge_v, cost_pol, open_pol, od_pairs)

    r_p0 = routes_p0[int(pair_index)]
    r_pol = routes_pol[int(pair_index)]

    closed_edge_ids = [
        int(eid) for eid in range(len(edge_u)) if not bool(open_pol[eid])
    ]
    u = int(od_pairs[int(pair_index), 0])
    v = int(od_pairs[int(pair_index), 1])

    return {
        "p0_edges": list(r_p0.edge_ids) if r_p0.reachable else None,
        "policy_edges": list(r_pol.edge_ids) if r_pol.reachable else None,
        "p0_reachable": bool(r_p0.reachable),
        "policy_reachable": bool(r_pol.reachable),
        "closed_edge_ids": closed_edge_ids,
        "od": (u, v),
    }


def reachable_count(env, d_e: np.ndarray, threshold_m: float, lam: float) -> int:
    """Number of reachable OD pairs (out of 20) under the closure policy."""
    network = env.road_network
    cfg = build_default_config()
    od_pairs = _od_pairs(env, cfg)
    cost, open_mask = policy_costs(
        network.edge_free_flow_time_s, d_e, threshold_m, lam
    )
    routes = solve_routes(
        int(network.n_nodes), network.edge_u, network.edge_v,
        cost, open_mask, od_pairs,
    )
    return sum(1 for r in routes if r.reachable)


def load_experiment_a_rows(path: Optional[Path] = None) -> List[dict]:
    """Read docs/experiment_a/results.csv rows as dicts (string values kept)."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "docs" / "experiment_a" / "results.csv"
    with open(path, newline="", encoding="ascii") as f:
        return list(csv.DictReader(f))


_METHODS = ("index", "elevation_only", "random_permutation")


def summarize_experiment_a(rows: List[dict]) -> List[dict]:
    """Summarize Experiment A rows: one dict per (scenario, method).

    Keys: scenario, method, n_seeds, mean_auc_030, min_auc_030, max_auc_030,
    mean_lift_030. Values are floats (n_seeds is int).
    """
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        key = (row["scenario"], row["method"])
        grouped.setdefault(key, []).append(row)

    out: List[dict] = []
    for (scenario, method), group in sorted(grouped.items()):
        auc_vals = [float(r["auc_030"]) for r in group]
        lift_vals = [float(r["lift_030"]) for r in group]
        out.append(
            {
                "scenario": scenario,
                "method": method,
                "n_seeds": len(group),
                "mean_auc_030": sum(auc_vals) / len(auc_vals),
                "min_auc_030": min(auc_vals),
                "max_auc_030": max(auc_vals),
                "mean_lift_030": sum(lift_vals) / len(lift_vals),
            }
        )
    return out


def count_pairs(rows: List[dict]) -> dict:
    """Count Experiment A pair-level wins.

    Keys: n_pairs (seed x scenario combinations), index_auc_wins_030
    (index auc_030 > elevation_only auc_030),
    elevation_lift_wins_030 (elevation_only lift_030 > index lift_030).
    """
    by_ss: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for row in rows:
        key = (row["scenario"], row["seed"])
        by_ss.setdefault(key, {})[row["method"]] = row

    index_auc_wins = 0
    elevation_lift_wins = 0
    for group in by_ss.values():
        idx = group.get("index")
        elev = group.get("elevation_only")
        if idx is None or elev is None:
            continue
        if float(idx["auc_030"]) > float(elev["auc_030"]):
            index_auc_wins += 1
        if float(elev["lift_030"]) > float(idx["lift_030"]):
            elevation_lift_wins += 1

    return {
        "n_pairs": len(by_ss),
        "index_auc_wins_030": index_auc_wins,
        "elevation_lift_wins_030": elevation_lift_wins,
    }


def depth_stats(env, hpeak: np.ndarray, d_e: np.ndarray) -> dict:
    """Depth and closure statistics over land cells.

    Land = not ocean and not lake. Returns dict with:
    max_peak_m (max over land cells), max_peak_lake_m (max over lake cells,
    0.0 if none), share_land_ge_010, share_land_ge_030, share_land_ge_045,
    closed_020, closed_030, closed_045.
    """
    land_mask = ~(env.is_ocean | env.is_lake)
    lake_mask = env.is_lake
    land_cells = int(np.sum(land_mask))
    lake_cells = int(np.sum(lake_mask))
    hpeak_land = hpeak[land_mask]
    if land_cells > 0:
        max_peak_m = float(np.max(hpeak_land))
        share_010 = float(np.sum(hpeak_land >= 0.10) / land_cells)
        share_030 = float(np.sum(hpeak_land >= 0.30) / land_cells)
        share_045 = float(np.sum(hpeak_land >= 0.45) / land_cells)
    else:
        max_peak_m = 0.0
        share_010 = 0.0
        share_030 = 0.0
        share_045 = 0.0
    if lake_cells > 0:
        max_peak_lake_m = float(np.max(hpeak[lake_mask]))
    else:
        max_peak_lake_m = 0.0
    return {
        "max_peak_m": max_peak_m,
        "max_peak_lake_m": max_peak_lake_m,
        "share_land_ge_010": share_010,
        "share_land_ge_030": share_030,
        "share_land_ge_045": share_045,
        "closed_020": int(np.sum(d_e >= 0.20)),
        "closed_030": int(np.sum(d_e >= 0.30)),
        "closed_045": int(np.sum(d_e >= 0.45)),
    }


def route_summary(env, d_e: np.ndarray, edge_ids) -> dict | None:
    """Route summary: length_m, free_flow_time_s, mean_d_e (length weighted),
    max_d_e, n_edges. None input returns None. Uses routing.route_metrics.
    """
    if edge_ids is None:
        return None
    network = env.road_network
    edge_ids_t = tuple(int(e) for e in edge_ids)
    m = route_metrics(
        edge_ids_t,
        network.edge_length_m,
        network.edge_free_flow_time_s,
        d_e,
        0.10,
        None,
    )
    return {
        "length_m": float(m["length_m"]),
        "free_flow_time_s": float(m["free_flow_time_s"]),
        "mean_d_e": float(m["mean_d_e_length_weighted"]),
        "max_d_e": float(m["max_d_e"]),
        "n_edges": int(m["n_edges"]),
    }


