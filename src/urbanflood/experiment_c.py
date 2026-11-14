"""SYNTHETIC Experiment C aggregation module (Phase 4).

Pure functions for aggregating risk-aware routing experiment results. Imports
only numpy and __future__/typing/collections/collections.abc/dataclasses -- no
urbanflood imports at all. All inputs are plain data structures.

Architecture: V7 (approved); Experiment C additions adopted (author-delegated,
Stage 16). Synthetic recovery testing only; no route is ever described as
"safe".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_REQUIRED_KEYS = (
    "seed",
    "scenario",
    "pair_index",
    "threshold_m",
    "lam",
    "reachable",
    "ff_time_s",
    "length_m",
    "max_d_e",
    "mean_d_e",
    "share_ge_005",
    "share_ge_010",
    "share_ge_020",
    "share_ge_030",
    "n_edges",
    "n_ge_closure",
    "p0_reachable",
    "p0_ff_time_s",
    "p0_length_m",
    "p0_max_d_e",
    "p0_mean_d_e",
    "p0_share_ge_010",
    "p0_n_ge_closure",
    "detour_ratio",
    "exposure_reduction",
    "max_depth_reduction",
)


def _require_keys(row: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(
            "row is missing required key(s): " + ", ".join(sorted(missing))
        )


def _finite(val: Any) -> bool:
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError):
        return False


def _nanmean(vals: list[float]) -> float:
    finite = [float(v) for v in vals if _finite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def aggregate_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate one row per (scenario, threshold_m, lam).

    P0 is not a row of its own; its fields are attached to every policy row.
    Returns a list of dicts, one per (scenario, threshold_m, lam), sorted by
    scenario then threshold_m then lam. Fields:

    n_pairs: number of (seed, pair) rows in the group.
    n_seeds: number of distinct seeds in the group.
    reachable_fraction: reachable / n_pairs.
    p0_reachable_fraction: share of rows with p0_reachable True.
    p0_blocked_fraction: among rows with p0_reachable, share with
        p0_n_ge_closure > 0; NaN if none.
    seed_reachable_min / seed_reachable_max: min/max over seeds of that seed's
        reachable fraction.
    mean_detour_ratio / n_finite_detour: mean detour_ratio over rows where both
        routes are reachable and detour_ratio is finite; NaN if none.
    mean_exposure_reduction / n_finite_exposure: mean exposure_reduction over
        rows where both routes are reachable and exposure_reduction is finite;
        NaN if none.
    frac_lower_exposure / frac_same_exposure / frac_higher_exposure: among
        finite exposure_reduction rows; lower means > 1e-12, higher means <
        -1e-12, same otherwise; NaN if none.
    mean_max_depth_reduction, mean_max_d_e, mean_p0_max_d_e,
    mean_share_ge_010, mean_p0_share_ge_010: means over rows where both routes
        are reachable; NaN if none.

    Raises ValueError if any row lacks a required key.
    """
    for row in rows:
        _require_keys(row)

    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["scenario"], float(row["threshold_m"]), float(row["lam"]))
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (scenario, threshold_m, lam), group in sorted(groups.items()):
        n_pairs = len(group)
        n_seeds = len({row["seed"] for row in group})
        reachable_rows = [r for r in group if bool(r["reachable"])]
        reachable_fraction = len(reachable_rows) / n_pairs if n_pairs else float("nan")
        p0_reachable_rows = [r for r in group if bool(r["p0_reachable"])]
        p0_reachable_fraction = (
            len(p0_reachable_rows) / n_pairs if n_pairs else float("nan")
        )

        p0_blocked = [
            r for r in p0_reachable_rows if int(r["p0_n_ge_closure"]) > 0
        ]
        p0_blocked_fraction = (
            len(p0_blocked) / len(p0_reachable_rows)
            if p0_reachable_rows
            else float("nan")
        )

        # per-seed reachable fraction
        seed_reachable: dict[Any, tuple[int, int]] = {}
        for r in group:
            s = r["seed"]
            if s not in seed_reachable:
                seed_reachable[s] = [0, 0]
            seed_reachable[s][0] += 1
            if bool(r["reachable"]):
                seed_reachable[s][1] += 1
        seed_fracs = [v[1] / v[0] for v in seed_reachable.values()]
        seed_reachable_min = float(min(seed_fracs)) if seed_fracs else float("nan")
        seed_reachable_max = float(max(seed_fracs)) if seed_fracs else float("nan")

        both_reachable = [r for r in group if bool(r["reachable"]) and bool(r["p0_reachable"])]

        detour_vals = [float(r["detour_ratio"]) for r in both_reachable if _finite(r["detour_ratio"])]
        exposure_vals = [float(r["exposure_reduction"]) for r in both_reachable if _finite(r["exposure_reduction"])]
        max_depth_red_vals = [float(r["max_depth_reduction"]) for r in both_reachable if _finite(r["max_depth_reduction"])]
        max_d_e_vals = [float(r["max_d_e"]) for r in both_reachable if _finite(r["max_d_e"])]
        p0_max_d_e_vals = [float(r["p0_max_d_e"]) for r in both_reachable if _finite(r["p0_max_d_e"])]
        share_ge_010_vals = [float(r["share_ge_010"]) for r in both_reachable if _finite(r["share_ge_010"])]
        p0_share_ge_010_vals = [float(r["p0_share_ge_010"]) for r in both_reachable if _finite(r["p0_share_ge_010"])]

        frac_lower = frac_same = frac_higher = float("nan")
        finite_exposure = [v for v in exposure_vals if _finite(v)]
        if finite_exposure:
            lower = sum(1 for v in finite_exposure if v > 1e-12)
            higher = sum(1 for v in finite_exposure if v < -1e-12)
            same = len(finite_exposure) - lower - higher
            n_finite = len(finite_exposure)
            frac_lower = lower / n_finite
            frac_same = same / n_finite
            frac_higher = higher / n_finite

        aggregated.append(
            {
                "scenario": scenario,
                "threshold_m": float(threshold_m),
                "lam": float(lam),
                "n_pairs": int(n_pairs),
                "n_seeds": int(n_seeds),
                "reachable_fraction": reachable_fraction,
                "p0_reachable_fraction": p0_reachable_fraction,
                "p0_blocked_fraction": p0_blocked_fraction,
                "seed_reachable_min": seed_reachable_min,
                "seed_reachable_max": seed_reachable_max,
                "mean_detour_ratio": _finite_val(_nanmean(detour_vals)),
                "n_finite_detour": len(detour_vals),
                "mean_exposure_reduction": _finite_val(_nanmean(exposure_vals)),
                "n_finite_exposure": len(exposure_vals),
                "frac_lower_exposure": frac_lower,
                "frac_same_exposure": frac_same,
                "frac_higher_exposure": frac_higher,
                "mean_max_depth_reduction": _finite_val(_nanmean(max_depth_red_vals)),
                "mean_max_d_e": _finite_val(_nanmean(max_d_e_vals)),
                "mean_p0_max_d_e": _finite_val(_nanmean(p0_max_d_e_vals)),
                "mean_share_ge_010": _finite_val(_nanmean(share_ge_010_vals)),
                "mean_p0_share_ge_010": _finite_val(_nanmean(p0_share_ge_010_vals)),
            }
        )

    return aggregated


def _finite_val(v: float) -> float:
    return float(v) if _finite(v) else float("nan")


def check_policy_invariants(rows: list[dict[str, Any]]) -> list[str]:
    """Check consistency invariants on aggregate rows.

    Returns a list of violation strings (empty if fine):
    - reachable rows with n_ge_closure != 0.
    - rows with lam == 0 and both reachable where ff_time_s < p0_ff_time_s - 1e-9.
    - rows where exposure/detour fields are inconsistent with reachability flags.

    Raises ValueError if any row lacks a required key.
    """
    for row in rows:
        _require_keys(row)

    violations: list[str] = []
    for row in rows:
        reachable = bool(row["reachable"])
        p0_reachable = bool(row["p0_reachable"])

        if reachable and int(row["n_ge_closure"]) != 0:
            violations.append(
                f"seed={row['seed']} scenario={row['scenario']} "
                f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                f"lam={row['lam']}: reachable but n_ge_closure={row['n_ge_closure']} != 0"
            )

        if (
            float(row["lam"]) == 0.0
            and reachable
            and p0_reachable
            and _finite(row["ff_time_s"])
            and _finite(row["p0_ff_time_s"])
            and row["ff_time_s"] < row["p0_ff_time_s"] - 1e-9
        ):
            violations.append(
                f"seed={row['seed']} scenario={row['scenario']} "
                f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                f"lam=0: ff_time_s={row['ff_time_s']} < p0_ff_time_s={row['p0_ff_time_s']} - 1e-9"
            )

        both = reachable and p0_reachable
        detour_finite = _finite(row["detour_ratio"])
        exposure_finite = _finite(row["exposure_reduction"])
        max_depth_red_finite = _finite(row["max_depth_reduction"])

        if both:
            if detour_finite and row["detour_ratio"] < 1.0 - 1e-9:
                violations.append(
                    f"seed={row['seed']} scenario={row['scenario']} "
                    f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                    f"lam={row['lam']}: both reachable but detour_ratio={row['detour_ratio']} < 1"
                )
            if exposure_finite:
                expected_sign = None
                if row["mean_d_e"] is not None and row["p0_mean_d_e"] is not None:
                    expected = row["p0_mean_d_e"] - row["mean_d_e"]
                    if abs(row["exposure_reduction"] - expected) > 1e-9:
                        violations.append(
                            f"seed={row['seed']} scenario={row['scenario']} "
                            f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                            f"lam={row['lam']}: exposure_reduction={row['exposure_reduction']} "
                            f"!= p0_mean_d_e - mean_d_e={expected}"
                        )
            if max_depth_red_finite:
                expected = row["p0_max_d_e"] - row["max_d_e"]
                if abs(row["max_depth_reduction"] - expected) > 1e-9:
                    violations.append(
                        f"seed={row['seed']} scenario={row['scenario']} "
                        f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                        f"lam={row['lam']}: max_depth_reduction={row['max_depth_reduction']} "
                        f"!= p0_max_d_e - max_d_e={expected}"
                    )
        else:
            if detour_finite:
                violations.append(
                    f"seed={row['seed']} scenario={row['scenario']} "
                    f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                    f"lam={row['lam']}: detour_ratio={row['detour_ratio']} set "
                    f"but not both reachable"
                )
            if exposure_finite:
                violations.append(
                    f"seed={row['seed']} scenario={row['scenario']} "
                    f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                    f"lam={row['lam']}: exposure_reduction={row['exposure_reduction']} "
                    f"set but not both reachable"
                )
            if max_depth_red_finite:
                violations.append(
                    f"seed={row['seed']} scenario={row['scenario']} "
                    f"pair_index={row['pair_index']} threshold_m={row['threshold_m']} "
                    f"lam={row['lam']}: max_depth_reduction={row['max_depth_reduction']} "
                    f"set but not both reachable"
                )

    return violations
