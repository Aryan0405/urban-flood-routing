"""Run Experiment C: risk-aware routing comparison (Phase 4).

Usage:
    python run_experiment_c.py --expa DIR --out DIR [--seeds S1 S2 ...]
                               [--scenarios R S1 S2 S3]
                               [--thresholds T1 T2 ...]
                               [--lambdas L1 L2 ...]
                               [--label-thresholds L1 L2 ...]

The --expa directory must contain cached Experiment A outputs
(hpeak_seed<seed>_<scenario>.npy and run_meta.txt). The --out directory must
be outside the repository directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from urbanflood.config import build_default_config, SCENARIOS
from urbanflood.environment import build_environment
from urbanflood.orchestrator import peak_edge_depths, run_simulation
from urbanflood.rng import STREAM_OD_PAIRS, derive_stream_seed
from urbanflood.routing import (
    effective_edge_depth,
    policy_costs,
    route_metrics,
    sample_od_pairs,
    solve_routes,
)
from urbanflood.experiment_c import aggregate_policy_rows, check_policy_invariants
from types import SimpleNamespace

OD_MIN_DISTANCE_M = 1000.0


def _is_inside_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment C.")
    parser.add_argument("--expa", required=True, help="Experiment A output dir (cached hpeak + run_meta.txt).")
    parser.add_argument("--out", required=True, help="Output directory (must be outside repo).")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Experiment seeds (default: the five from config).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Scenario names (default: R S1 S2 S3).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Closure thresholds (default: RoadConfig closure sweep).",
    )
    parser.add_argument(
        "--lambdas",
        nargs="*",
        type=float,
        default=None,
        help="Lambda values (default: RoadConfig lambda sweep).",
    )
    parser.add_argument(
        "--label-thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Label thresholds (default: RoadConfig label sweep).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    expa_dir = Path(args.expa).resolve()
    out_dir = Path(args.out).resolve()

    cfg = build_default_config()
    if args.seeds is None:
        seeds = list(cfg.seeds.env_experiment_seeds())
    else:
        seeds = list(args.seeds)
    if args.scenarios is None:
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = list(args.scenarios)
    thresholds = (
        list(args.thresholds)
        if args.thresholds is not None
        else list(cfg.routing.closure_thresholds_sweep_m)
    )
    lambdas = (
        list(args.lambdas)
        if args.lambdas is not None
        else list(cfg.routing.lambda_sweep)
    )
    label_thresholds = (
        list(args.label_thresholds)
        if args.label_thresholds is not None
        else list(cfg.routing.label_thresholds_sweep_m)
    )

    if _is_inside_repo(out_dir, repo_root):
        print("ERROR: --out must be outside the repository directory.", file=sys.stderr)
        return 1

    # Cache guard: expA run_meta.txt must exist and match steps/dt.
    meta_path = expa_dir / "run_meta.txt"
    if not meta_path.exists():
        print(
            "ERROR: expA run_meta.txt not found at "
            f"{meta_path}. Run Experiment A first.",
            file=sys.stderr,
        )
        return 1
    meta_lines = meta_path.read_text(encoding="ascii").splitlines()
    meta_dict = {}
    for line in meta_lines:
        if "=" in line:
            k, v = line.split("=", 1)
            meta_dict[k.strip()] = v.strip()
    expa_steps = meta_dict.get("steps")
    expa_dt = meta_dict.get("dt")
    if expa_steps != "720" or expa_dt != "30.0":
        print(
            "ERROR: expA run_meta.txt records steps="
            f"{expa_steps}, dt={expa_dt}, but Experiment C requires "
            "steps=720, dt=30.0.",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta_c.txt").write_text(
        f"seeds={','.join(map(str, seeds))}\n"
        f"scenarios={','.join(scenario_names)}\n"
        f"thresholds={','.join(map(str, thresholds))}\n"
        f"lambdas={','.join(map(str, lambdas))}\n"
        f"label_thresholds={','.join(map(str, label_thresholds))}\n"
        f"steps=720\ndt=30.0\nod_min_distance_m={OD_MIN_DISTANCE_M}\n",
        encoding="ascii",
    )

    all_rows: list[dict] = []

    for seed in seeds:
        env = build_environment(seed, cfg)
        network = env.road_network
        cell_size_m = float(env.grid.cell_size_m)

        # OD pairs sampled once per seed using the od_pairs stream seed.
        od_seed = derive_stream_seed(seed, STREAM_OD_PAIRS)
        od_pairs = sample_od_pairs(
            network.node_rows,
            network.node_cols,
            cell_size_m,
            int(cfg.routing.n_od_pairs),
            OD_MIN_DISTANCE_M,
            od_seed,
        )

        hpeak_path = expa_dir / f"hpeak_seed{seed}_R.npy"
        missing = [
            f"hpeak_seed{seed}_{s}.npy"
            for s in scenario_names
            if not (expa_dir / f"hpeak_seed{seed}_{s}.npy").exists()
        ]
        if missing:
            print(
                "ERROR: missing Experiment A cache files: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 1

        t0 = time.perf_counter()
        print(
            f"seed={seed} nodes={network.n_nodes} edges={network.n_edges} "
            f"od_pairs={len(od_pairs)}",
            flush=True,
        )

        for sname in scenario_names:
            scen = SCENARIOS[sname]
            hpeak_path = expa_dir / f"hpeak_seed{seed}_{sname}.npy"
            hpeak = np.load(hpeak_path)
            if hpeak.shape != env.elevation_m.shape:
                print(
                    f"ERROR: hpeak shape {hpeak.shape} != env shape "
                    f"{env.elevation_m.shape} for seed={seed} scenario={sname}",
                    file=sys.stderr,
                )
                return 1

            peak = peak_edge_depths(SimpleNamespace(H_peak=hpeak), env)
            surface = peak_edge_depths(
                SimpleNamespace(H_peak=env.elevation_m + hpeak), env
            )
            d_e = effective_edge_depth(
                peak, surface, network.edge_is_bridge, network.edge_deck_m
            )
            if not np.all(np.isfinite(d_e)):
                print(
                    f"ERROR: non-finite d_e for seed={seed} scenario={sname}",
                    file=sys.stderr,
                )
                return 1

            # P0: no closure, lam=0.
            cost_p0, open_p0 = policy_costs(
                network.edge_free_flow_time_s, d_e, None, 0.0
            )
            routes_p0 = solve_routes(
                int(network.n_nodes),
                network.edge_u,
                network.edge_v,
                cost_p0,
                open_p0,
                od_pairs,
            )

            for threshold_m in thresholds:
                for lam in lambdas:
                    cost_pol, open_pol = policy_costs(
                        network.edge_free_flow_time_s, d_e, threshold_m, lam
                    )
                    routes_pol = solve_routes(
                        int(network.n_nodes),
                        network.edge_u,
                        network.edge_v,
                        cost_pol,
                        open_pol,
                        od_pairs,
                    )

                    for pair_index, (r_pol, r_p0) in enumerate(
                        zip(routes_pol, routes_p0)
                    ):
                        u, v = int(od_pairs[pair_index, 0]), int(od_pairs[pair_index, 1])
                        if r_pol.reachable:
                            assert r_pol.node_path[0] == u
                            assert r_pol.node_path[-1] == v
                        if r_p0.reachable:
                            assert r_p0.node_path[0] == u
                            assert r_p0.node_path[-1] == v

                        row = {
                            "seed": seed,
                            "scenario": sname,
                            "pair_index": pair_index,
                            "threshold_m": float(threshold_m),
                            "lam": float(lam),
                            "reachable": bool(r_pol.reachable),
                            "ff_time_s": (
                                float(r_pol.total_cost)
                                if r_pol.reachable
                                else 0.0
                            ),
                            "length_m": 0.0,
                            "max_d_e": 0.0,
                            "mean_d_e": 0.0,
                            "share_ge_005": 0.0,
                            "share_ge_010": 0.0,
                            "share_ge_020": 0.0,
                            "share_ge_030": 0.0,
                            "n_edges": 0,
                            "n_ge_closure": 0,
                            "p0_reachable": bool(r_p0.reachable),
                            "p0_ff_time_s": (
                                float(r_p0.total_cost) if r_p0.reachable else 0.0
                            ),
                            "p0_length_m": 0.0,
                            "p0_max_d_e": 0.0,
                            "p0_mean_d_e": 0.0,
                            "p0_share_ge_010": 0.0,
                            "p0_n_ge_closure": 0,
                            "detour_ratio": float("nan"),
                            "exposure_reduction": float("nan"),
                            "max_depth_reduction": float("nan"),
                        }

                        if r_pol.reachable:
                            m = route_metrics(
                                r_pol.edge_ids,
                                network.edge_length_m,
                                network.edge_free_flow_time_s,
                                d_e,
                                float(cfg.routing.label_threshold_default_m),
                                threshold_m,
                            )
                            row["ff_time_s"] = float(
                                np.sum(
                                    network.edge_free_flow_time_s[
                                        list(r_pol.edge_ids)
                                    ]
                                )
                            )
                            row["length_m"] = float(m["length_m"])
                            row["max_d_e"] = float(m["max_d_e"])
                            row["mean_d_e"] = float(m["mean_d_e_length_weighted"])
                            row["share_ge_005"] = float(
                                route_metrics(
                                    r_pol.edge_ids,
                                    network.edge_length_m,
                                    network.edge_free_flow_time_s,
                                    d_e,
                                    float(label_thresholds[0]),
                                    threshold_m,
                                )["share_length_ge_label"]
                            )
                            row["share_ge_010"] = float(
                                route_metrics(
                                    r_pol.edge_ids,
                                    network.edge_length_m,
                                    network.edge_free_flow_time_s,
                                    d_e,
                                    float(label_thresholds[1]),
                                    threshold_m,
                                )["share_length_ge_label"]
                            )
                            row["share_ge_020"] = float(
                                route_metrics(
                                    r_pol.edge_ids,
                                    network.edge_length_m,
                                    network.edge_free_flow_time_s,
                                    d_e,
                                    float(label_thresholds[2]),
                                    threshold_m,
                                )["share_length_ge_label"]
                            )
                            row["share_ge_030"] = float(
                                route_metrics(
                                    r_pol.edge_ids,
                                    network.edge_length_m,
                                    network.edge_free_flow_time_s,
                                    d_e,
                                    float(label_thresholds[3]),
                                    threshold_m,
                                )["share_length_ge_label"]
                            )
                            row["n_edges"] = int(m["n_edges"])
                            row["n_ge_closure"] = int(m["n_edges_ge_closure"])

                        if r_p0.reachable:
                            m_p0 = route_metrics(
                                r_p0.edge_ids,
                                network.edge_length_m,
                                network.edge_free_flow_time_s,
                                d_e,
                                float(cfg.routing.label_threshold_default_m),
                                threshold_m,
                            )
                            row["p0_ff_time_s"] = float(
                                np.sum(
                                    network.edge_free_flow_time_s[
                                        list(r_p0.edge_ids)
                                    ]
                                )
                            )
                            row["p0_length_m"] = float(m_p0["length_m"])
                            row["p0_max_d_e"] = float(m_p0["max_d_e"])
                            row["p0_mean_d_e"] = float(
                                m_p0["mean_d_e_length_weighted"]
                            )
                            row["p0_share_ge_010"] = float(
                                route_metrics(
                                    r_p0.edge_ids,
                                    network.edge_length_m,
                                    network.edge_free_flow_time_s,
                                    d_e,
                                    float(label_thresholds[1]),
                                    threshold_m,
                                )["share_length_ge_label"]
                            )
                            row["p0_n_ge_closure"] = int(
                                m_p0["n_edges_ge_closure"]
                            )

                        if r_pol.reachable and r_p0.reachable:
                            if row["p0_ff_time_s"] > 0.0:
                                row["detour_ratio"] = (
                                    row["ff_time_s"] / row["p0_ff_time_s"]
                                )
                            row["exposure_reduction"] = (
                                row["p0_mean_d_e"] - row["mean_d_e"]
                            )
                            row["max_depth_reduction"] = (
                                row["p0_max_d_e"] - row["max_d_e"]
                            )

                        all_rows.append(row)

        t_elapsed = time.perf_counter() - t0
        print(f"  elapsed={t_elapsed:.2f}s", flush=True)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(out_dir / "pairs.csv", "w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        # edge_summary.csv: one row per seed x scenario
        edge_rows: list[dict] = []
        for seed in seeds:
            env = build_environment(seed, cfg)
            network = env.road_network
            for sname in scenario_names:
                hpeak = np.load(expa_dir / f"hpeak_seed{seed}_{sname}.npy")
                peak = peak_edge_depths(SimpleNamespace(H_peak=hpeak), env)
                surface = peak_edge_depths(
                    SimpleNamespace(H_peak=env.elevation_m + hpeak), env
                )
                d_e = effective_edge_depth(
                    peak, surface, network.edge_is_bridge, network.edge_deck_m
                )
                n_bridge = int(np.sum(network.edge_is_bridge))
                row = {
                    "seed": seed,
                    "scenario": sname,
                    "n_edges": int(network.n_edges),
                    "n_bridge_edges": n_bridge,
                    "mean_d_e": float(np.mean(d_e)),
                    "max_d_e": float(np.max(d_e)),
                }
                for t in thresholds:
                    _, open_mask = policy_costs(
                        network.edge_free_flow_time_s, d_e, t, 0.0
                    )
                    row[f"n_closed_at_{t}"] = int(np.sum(~open_mask))
                    row[f"n_bridge_closed_at_{t}"] = int(
                        np.sum(~open_mask & network.edge_is_bridge)
                    )
                edge_rows.append(row)

        edge_fieldnames = list(edge_rows[0].keys())
        with open(out_dir / "edge_summary.csv", "w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=edge_fieldnames)
            writer.writeheader()
            for row in edge_rows:
                writer.writerow(row)

        aggregated = aggregate_policy_rows(all_rows)
        agg_fieldnames = list(aggregated[0].keys())
        with open(out_dir / "summary.csv", "w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
            writer.writeheader()
            for row in aggregated:
                writer.writerow(row)

        # summary.txt
        with open(out_dir / "summary.txt", "w", encoding="ascii") as f:
            f.write("Experiment C summary\n\n")
            f.write("Aggregated results (pooled over all seed x OD-pair rows):\n\n")
            for scenario in scenario_names:
                f.write(f"  Scenario {scenario}:\n")
                for t in thresholds:
                    f.write(f"    Threshold {t}:\n")
                    for lam in lambdas:
                        row = next(
                            (
                                r
                                for r in aggregated
                                if r["scenario"] == scenario
                                and r["threshold_m"] == t
                                and r["lam"] == lam
                            ),
                            None,
                        )
                        if row is None:
                            continue
                        f.write(
                            f"      lambda={lam}: "
                            f"reachable_fraction={row['reachable_fraction']:.4f}, "
                            f"p0_blocked_fraction={row['p0_blocked_fraction']:.4f}, "
                            f"mean_detour_ratio={row['mean_detour_ratio']:.4f}, "
                            f"mean_exposure_reduction={row['mean_exposure_reduction']:.4f}, "
                            f"frac_lower_exposure={row['frac_lower_exposure']:.4f}\n"
                        )
            f.write("\n  Headline (threshold 0.30):\n")
            for scenario in scenario_names:
                f.write(f"    Scenario {scenario}:\n")
                for lam in (0.0, 1.0):
                    row = next(
                        (
                            r
                            for r in aggregated
                            if r["scenario"] == scenario
                            and r["threshold_m"] == 0.30
                            and r["lam"] == lam
                        ),
                        None,
                    )
                    if row is None:
                        continue
                    f.write(
                        f"      lambda={lam}: "
                        f"reachable_fraction={row['reachable_fraction']:.4f}, "
                        f"mean_detour_ratio={row['mean_detour_ratio']:.4f}\n"
                    )

        violations = check_policy_invariants(all_rows)
        if violations:
            print(
                f"ERROR: {len(violations)} policy invariant violation(s):",
                file=sys.stderr,
            )
            for v in violations[:10]:
                print(f"  {v}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
