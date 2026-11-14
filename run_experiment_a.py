"""Run Experiment A: susceptibility vs simulated peak depth comparison.

Usage:
    python run_experiment_a.py --out DIR [--seeds S1 S2 ...] [--scenarios R S1 S2 S3]
                               [--steps 720] [--dt 30.0]

The --out directory must be outside the repository directory. Results are
resumable: cached .npy files are reused if present.
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
from urbanflood.drainage import generate_drainage
from urbanflood.environment import build_environment
from urbanflood.initial_state import build_initial_depth
from urbanflood.orchestrator import run_simulation
from urbanflood.susceptibility import build_susceptibility
from urbanflood.experiment_a import (
    ablated_index,
    evaluate_methods,
    permuted_scores,
    summarize,
)


def _is_inside_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment A.")
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
    parser.add_argument("--steps", type=int, default=720, help="Number of timesteps.")
    parser.add_argument("--dt", type=float, default=30.0, help="Timestep in seconds.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    out_dir = Path(args.out).resolve()
    if _is_inside_repo(out_dir, repo_root):
        print("ERROR: --out must be outside the repository directory.", file=sys.stderr)
        return 1

    cfg = build_default_config()
    if args.seeds is None:
        seeds = list(cfg.seeds.env_experiment_seeds())
    else:
        seeds = list(args.seeds)
    if args.scenarios is None:
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = list(args.scenarios)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache guard: if run_meta.txt exists and steps/dt differ, refuse.
    meta_path = out_dir / "run_meta.txt"
    meta_exists = meta_path.exists()
    if meta_exists:
        meta_lines = meta_path.read_text(encoding="ascii").splitlines()
        meta_dict = {}
        for line in meta_lines:
            if "=" in line:
                k, v = line.split("=", 1)
                meta_dict[k.strip()] = v.strip()
        if (
            meta_dict.get("steps") != str(args.steps)
            or meta_dict.get("dt") != str(args.dt)
        ):
            print(
                "ERROR: run_meta.txt in --out was written with "
                f"steps={meta_dict.get('steps')}, dt={meta_dict.get('dt')}, "
                f"but current arguments are steps={args.steps}, dt={args.dt}. "
                "Use a fresh --out directory.",
                file=sys.stderr,
            )
            return 1
    else:
        meta_path.write_text(
            f"steps={args.steps}\ndt={args.dt}\nscenarios={','.join(scenario_names)}\n",
            encoding="ascii",
        )

    all_rows: list[dict] = []
    elapsed_times: list[float] = []

    for seed in seeds:
        env = build_environment(seed, cfg)
        susc = build_susceptibility(env)
        index_arr = susc.index
        np.save(out_dir / f"index_seed{seed}.npy", index_arr)

        land_mask = (~env.is_ocean) & (~env.is_lake)
        h0 = build_initial_depth(env)
        elev_score = susc.feature_scores["elevation"]

        for sname in scenario_names:
            scen = SCENARIOS[sname]
            hpeak_path = out_dir / f"hpeak_seed{seed}_{sname}.npy"
            if hpeak_path.exists():
                h_peak = np.load(hpeak_path)
            else:
                drainage = generate_drainage(
                    seed,
                    env.grid.n_cells,
                    scen.blocked_fraction,
                    cfg.drainage,
                )
                t0 = time.perf_counter()
                result = run_simulation(
                    h0,
                    env,
                    drainage,
                    scen.rainfall,
                    scen.tide,
                    cfg.exchange,
                    cfg.runoff,
                    args.dt,
                    args.steps,
                )
                t_elapsed = time.perf_counter() - t0
                h_peak = result.H_peak
                np.save(hpeak_path, h_peak)
                elapsed_times.append(t_elapsed)
                print(
                    f"seed={seed} scenario={sname} steps={args.steps} "
                    f"elapsed={t_elapsed:.2f}s",
                    flush=True,
                )

            target = (h_peak - h0)[land_mask]
            index_land = index_arr[land_mask]
            elev_land = elev_score[land_mask]

            methods: dict[str, np.ndarray] = {
                "index": index_land,
                "elevation_only": elev_land,
                "random_permutation": permuted_scores(index_land, seed=1000 + seed),
            }
            for feat_name in susc.feature_scores:
                methods[f"drop_{feat_name}"] = ablated_index(
                    susc.feature_scores, susc.weights, drop=feat_name
                )[land_mask]

            rows = evaluate_methods(methods, target)
            for row in rows:
                row["seed"] = seed
                row["scenario"] = sname
            all_rows.extend(rows)

    if all_rows:
        fieldnames = ["seed", "scenario"] + [
            k for k in all_rows[0].keys() if k not in ("seed", "scenario")
        ]
        with open(out_dir / "results.csv", "w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

    aggregated, comparison = summarize(all_rows)
    mean_elapsed = float(np.mean(elapsed_times)) if elapsed_times else float("nan")
    with open(out_dir / "summary.txt", "w", encoding="ascii") as f:
        f.write("Experiment A summary\n")
        f.write(f"Mean elapsed time per run: {mean_elapsed:.2f} s\n")
        f.write(
            "\nAggregated results (mean over seeds per scenario x method):\n"
        )
        for row in aggregated:
            f.write(
                f"  scenario={row['scenario']} method={row['method']} "
                f"n_rows={row['n_rows']} "
                f"mean_spearman={row['mean_spearman']:.6f} "
                f"n_finite_spearman={row['n_finite_spearman']} "
                f"mean_auc_010={row['mean_auc_010']:.6f} "
                f"n_finite_auc_010={row['n_finite_auc_010']} "
                f"mean_auc_030={row['mean_auc_030']:.6f} "
                f"n_finite_auc_030={row['n_finite_auc_030']} "
                f"mean_auc_050={row['mean_auc_050']:.6f} "
                f"n_finite_auc_050={row['n_finite_auc_050']} "
                f"mean_lift_010={row['mean_lift_010']:.6f} "
                f"n_finite_lift_010={row['n_finite_lift_010']} "
                f"mean_lift_030={row['mean_lift_030']:.6f} "
                f"n_finite_lift_030={row['n_finite_lift_030']} "
                f"mean_lift_050={row['mean_lift_050']:.6f} "
                f"n_finite_lift_050={row['n_finite_lift_050']}\n"
            )
        f.write("\nComparison: index vs elevation_only on auc_030 (per scenario):\n")
        for scenario in sorted(comparison.keys()):
            c = comparison[scenario]
            f.write(
                f"  {scenario}: wins={c['wins']} of n_pairs={c['n_pairs']} "
                f"(ties={c['ties']}, losses={c['losses']}, "
                f"skipped={c['n_skipped_nan']}), "
                f"mean_diff_auc_030={c['mean_diff_auc_030']:.6f}\n"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
