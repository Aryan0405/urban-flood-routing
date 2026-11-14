"""Experiment A metrics module (Phase 3).

Pure functions for synthetic recovery testing of the susceptibility index
against simulated peak depths. Imports only numpy and scipy.stats; no
urbanflood imports, so this module is independent of the simulator package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def spearman(index: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties.

    Computed as Pearson correlation on rankdata-transformed inputs. Returns
    NaN if either input is constant (zero variance in ranks).
    """
    x = np.asarray(index, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    xr = stats.rankdata(x, method="average")
    yr = stats.rankdata(y, method="average")
    xr_std = float(xr.std())
    yr_std = float(yr.std())
    if xr_std == 0.0 or yr_std == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def roc_auc(score: np.ndarray, flooded_bool: np.ndarray) -> float:
    """Mann-Whitney U formulation of ROC-AUC with average ranks.

    Returns NaN if there are no positives or no negatives.
    """
    s = np.asarray(score, dtype=np.float64).ravel()
    f = np.asarray(flooded_bool, dtype=bool).ravel()
    mask = np.isfinite(s) & np.isfinite(f)
    s = s[mask]
    f = f[mask]
    n_pos = int(f.sum())
    n_neg = int((~f).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s, method="average")
    sum_ranks_pos = float(ranks[f].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def top_decile_lift(
    score: np.ndarray, flooded_bool: np.ndarray, frac: float = 0.10
) -> float:
    """Lift of the top frac fraction of cells by score.

    The top ceil(frac * n) cells are selected using a stable sort on -score
    (ties broken by position). Lift = (flooded share in top group) / (overall
    flooded share). Returns NaN if there are no positives.
    """
    s = np.asarray(score, dtype=np.float64).ravel()
    f = np.asarray(flooded_bool, dtype=bool).ravel()
    mask = np.isfinite(s) & np.isfinite(f)
    s = s[mask]
    f = f[mask]
    n = int(s.size)
    if n == 0:
        return float("nan")
    n_pos = int(f.sum())
    if n_pos == 0:
        return float("nan")
    k = int(np.ceil(frac * n))
    if k < 1:
        k = 1
    order = np.argsort(-s, kind="stable")
    top = order[:k]
    flooded_share_top = float(f[top].mean())
    flooded_share_overall = float(f.mean())
    if flooded_share_overall == 0.0:
        return float("nan")
    return float(flooded_share_top / flooded_share_overall)


def ablated_index(
    feature_scores: dict[str, np.ndarray], weights: dict[str, float], drop: str
) -> np.ndarray:
    """Weighted sum of remaining features with weights renormalised to sum to 1.

    Returns NaN where any remaining feature is NaN. Raises ValueError if
    drop is not a known feature name or if no features remain.
    """
    if drop not in feature_scores:
        raise ValueError(f"unknown feature to drop: {drop}")
    remaining = {k: v for k, v in feature_scores.items() if k != drop}
    if len(remaining) == 0:
        raise ValueError("no features remain after drop")
    remaining_weights = {k: weights[k] for k in remaining}
    w_sum = float(sum(remaining_weights.values()))
    if w_sum <= 0.0:
        raise ValueError("remaining weights sum to zero")
    keys = sorted(remaining.keys())
    stacked = np.stack([remaining[k] for k in keys], axis=0)
    any_nan = np.any(np.isnan(stacked), axis=0)
    result = np.full(stacked.shape[1:], np.nan, dtype=np.float64)
    valid = ~any_nan
    if np.any(valid):
        norm_weights = np.array(
            [remaining_weights[k] / w_sum for k in keys], dtype=np.float64
        )
        result[valid] = np.tensordot(norm_weights, stacked[:, valid], axes=1)
    return result


def permuted_scores(values: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic permutation of values using np.random.default_rng(seed)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    return rng.permutation(v)


def evaluate_methods(
    methods: dict[str, np.ndarray],
    target: np.ndarray,
    thresholds: tuple[float, ...] = (0.10, 0.30, 0.50),
) -> list[dict[str, Any]]:
    """Evaluate each method against target across thresholds.

    Returns one row dict per method with fields:
    method, spearman, auc_010, auc_030, auc_050, lift_010, lift_030,
    lift_050, n_land, frac_flooded_010, frac_flooded_030, frac_flooded_050.
    """
    y = np.asarray(target, dtype=np.float64).ravel()
    rows: list[dict[str, Any]] = []
    for method_name, score in methods.items():
        s = np.asarray(score, dtype=np.float64).ravel()
        mask = np.isfinite(s) & np.isfinite(y)
        s_m = s[mask]
        y_m = y[mask]
        n_land = int(s_m.size)
        row: dict[str, Any] = {"method": method_name, "n_land": n_land}
        row["spearman"] = spearman(s_m, y_m)
        for i, thr in enumerate(thresholds):
            flooded = y_m >= thr
            pct = int(round(thr * 100))
            key = f"auc_{pct:03d}"
            row[key] = roc_auc(s_m, flooded)
            key = f"lift_{pct:03d}"
            row[key] = top_decile_lift(s_m, flooded)
            key = f"frac_flooded_{pct:03d}"
            row[key] = float(flooded.mean()) if flooded.size > 0 else float("nan")
        rows.append(row)
    return rows


def summarize(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Aggregate rows per (scenario, method) and compare index vs elevation_only.

    Returns (aggregated, comparison):

    - aggregated: one dict per (scenario, method) with n_rows, and for each of
      spearman, auc_010, auc_030, auc_050, lift_010, lift_030, lift_050 both
      the mean over finite values (fields like mean_auc_030) and the count of
      finite values (fields like n_finite_auc_030). Mean is NaN when there are
      no finite values.
    - comparison: dict scenario -> dict with n_pairs, wins, ties, losses,
      n_skipped_nan, mean_diff_auc_030. Every scenario present in rows appears,
      including scenarios with zero wins.

    Raises ValueError if any row lacks 'seed'.
    """
    from collections import defaultdict

    for row in rows:
        if "seed" not in row:
            raise ValueError("summarize requires 'seed' in every row dict")

    METRICS = (
        "spearman",
        "auc_010",
        "auc_030",
        "auc_050",
        "lift_010",
        "lift_030",
        "lift_050",
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["method"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for (scenario, method), group in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "scenario": scenario,
            "method": method,
            "n_rows": len(group),
        }
        for metric in METRICS:
            vals = [r[metric] for r in group if np.isfinite(r[metric])]
            entry[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"n_finite_{metric}"] = len(vals)
        aggregated.append(entry)

    scenarios = sorted({row["scenario"] for row in rows})
    comparison: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        scenario_rows = [r for r in rows if r["scenario"] == scenario]

        index_auc: dict[int, float] = {}
        elev_auc: dict[int, float] = {}
        all_seeds: set[int] = set()
        for r in scenario_rows:
            seed = r["seed"]
            all_seeds.add(seed)
            if r["method"] == "index" and np.isfinite(r["auc_030"]):
                index_auc[seed] = r["auc_030"]
            elif r["method"] == "elevation_only" and np.isfinite(r["auc_030"]):
                elev_auc[seed] = r["auc_030"]

        n_pairs = 0
        wins = 0
        ties = 0
        losses = 0
        n_skipped_nan = 0
        diffs: list[float] = []

        for seed in sorted(all_seeds):
            idx_has = seed in index_auc
            el_has = seed in elev_auc
            if idx_has and el_has:
                n_pairs += 1
                diff = index_auc[seed] - elev_auc[seed]
                diffs.append(diff)
                if diff > 0:
                    wins += 1
                elif diff == 0:
                    ties += 1
                else:
                    losses += 1
            else:
                n_skipped_nan += 1

        mean_diff = float(np.mean(diffs)) if diffs else float("nan")

        comparison[scenario] = {
            "n_pairs": n_pairs,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "n_skipped_nan": n_skipped_nan,
            "mean_diff_auc_030": mean_diff,
        }

    return aggregated, comparison
