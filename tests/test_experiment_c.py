"""Tests for src/urbanflood/experiment_c.py (Phase 4 Experiment C aggregation).

Fast tests only: toy data, no simulation, no experiment runner.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect

import numpy as np
import pytest

from urbanflood.experiment_c import (
    aggregate_policy_rows,
    check_policy_invariants,
)
from urbanflood.routing import RouteResult


def _row(
    seed,
    scenario,
    pair_index,
    threshold_m,
    lam,
    reachable,
    ff_time_s,
    length_m,
    max_d_e,
    mean_d_e,
    share_ge_005,
    share_ge_010,
    share_ge_020,
    share_ge_030,
    n_edges,
    n_ge_closure,
    p0_reachable,
    p0_ff_time_s,
    p0_length_m,
    p0_max_d_e,
    p0_mean_d_e,
    p0_share_ge_010,
    p0_n_ge_closure,
    detour_ratio,
    exposure_reduction,
    max_depth_reduction,
):
    return {
        "seed": seed,
        "scenario": scenario,
        "pair_index": pair_index,
        "threshold_m": threshold_m,
        "lam": lam,
        "reachable": reachable,
        "ff_time_s": ff_time_s,
        "length_m": length_m,
        "max_d_e": max_d_e,
        "mean_d_e": mean_d_e,
        "share_ge_005": share_ge_005,
        "share_ge_010": share_ge_010,
        "share_ge_020": share_ge_020,
        "share_ge_030": share_ge_030,
        "n_edges": n_edges,
        "n_ge_closure": n_ge_closure,
        "p0_reachable": p0_reachable,
        "p0_ff_time_s": p0_ff_time_s,
        "p0_length_m": p0_length_m,
        "p0_max_d_e": p0_max_d_e,
        "p0_mean_d_e": p0_mean_d_e,
        "p0_share_ge_010": p0_share_ge_010,
        "p0_n_ge_closure": p0_n_ge_closure,
        "detour_ratio": detour_ratio,
        "exposure_reduction": exposure_reduction,
        "max_depth_reduction": max_depth_reduction,
    }


# ---------------------------------------------------------------------------
# aggregate_policy_rows
# ---------------------------------------------------------------------------


def test_aggregate_policy_rows_hand_computed():
    """2 seeds x 2 scenarios x 3 pairs x 2 policies.

    Scenario A:
      seed 1: pair 0 both reachable, detour 1.1, exposure -0.1; pair 1 both
              reachable, detour 1.2, exposure -0.1; pair 2 policy unreachable.
      seed 2: pair 0 both reachable, detour 1.0, exposure 0.0; pair 1 both
              reachable, detour 1.0, exposure 0.0; pair 2 both unreachable.
    Scenario B: all unreachable.
    """
    rows = []
    # Scenario A, seed 1
    rows.append(_row(1, "A", 0, 0.3, 0.5, True, 2.0, 10.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 3, 0, True, 1.0, 9.0, 0.3, 0.1, 0.0, 0, 1.1, -0.1, -0.1))
    rows.append(_row(1, "A", 1, 0.3, 0.5, True, 3.0, 20.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 4, 0, True, 2.5, 18.0, 0.4, 0.2, 0.0, 0, 1.2, -0.1, -0.1))
    rows.append(_row(1, "A", 2, 0.3, 0.5, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0))
    # Scenario A, seed 2
    rows.append(_row(2, "A", 0, 0.3, 0.5, True, 2.0, 10.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 3, 0, True, 1.5, 8.0, 0.1, 0.0, 0.0, 0, 1.0, 0.0, 0.0))
    rows.append(_row(2, "A", 1, 0.3, 0.5, True, 2.0, 15.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 4, 0, True, 1.8, 14.0, 0.2, 0.1, 0.0, 0, 1.0, 0.0, 0.0))
    rows.append(_row(2, "A", 2, 0.3, 0.5, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0))
    # Scenario B, all unreachable
    for s in (1, 2):
        for p in range(3):
            rows.append(_row(s, "B", p, 0.3, 0.5, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0))

    aggregated = aggregate_policy_rows(rows)
    agg = {(r["scenario"], r["threshold_m"], r["lam"]): r for r in aggregated}
    a = agg[("A", 0.3, 0.5)]
    # Scenario A: 6 rows total (2 seeds x 3 pairs), 4 reachable policy rows.
    assert a["n_pairs"] == 6
    assert a["n_seeds"] == 2
    assert a["reachable_fraction"] == pytest.approx(4 / 6, abs=1e-12)
    assert a["p0_reachable_fraction"] == pytest.approx(4 / 6, abs=1e-12)
    # p0_blocked_fraction: p0_reachable rows = 4, all have p0_n_ge_closure=0.
    assert a["p0_blocked_fraction"] == pytest.approx(0.0, abs=1e-12)
    # mean_detour_ratio = (1.1 + 1.2 + 1.0 + 1.0) / 4 = 4.3 / 4 = 1.075
    assert a["mean_detour_ratio"] == pytest.approx(1.075, abs=1e-12)
    assert a["n_finite_detour"] == 4
    # mean_exposure_reduction = (-0.1 + -0.1 + 0.0 + 0.0) / 4 = -0.2 / 4 = -0.05
    assert a["mean_exposure_reduction"] == pytest.approx(-0.05, abs=1e-12)
    assert a["n_finite_exposure"] == 4
    # frac_lower (exposure > 1e-12): none -> 0/4 = 0.0
    assert a["frac_lower_exposure"] == pytest.approx(0.0, abs=1e-12)
    # frac_higher (exposure < -1e-12): seed1 pair0 (-0.1) and seed1 pair1 (-0.1) -> 2/4 = 0.5
    assert a["frac_higher_exposure"] == pytest.approx(0.5, abs=1e-12)
    # frac_same: 2/4 = 0.5
    assert a["frac_same_exposure"] == pytest.approx(0.5, abs=1e-12)
    # mean_max_depth_reduction = (-0.1 + -0.1 + 0.0 + 0.0) / 4 = -0.2/4 = -0.05
    assert a["mean_max_depth_reduction"] == pytest.approx(-0.05, abs=1e-12)
    # mean_max_d_e = (0.4 + 0.5 + 0.2 + 0.3) / 4 = 1.4 / 4 = 0.35
    assert a["mean_max_d_e"] == pytest.approx(0.35, abs=1e-12)
    # mean_p0_max_d_e = (0.3 + 0.4 + 0.1 + 0.2) / 4 = 1.0 / 4 = 0.25
    assert a["mean_p0_max_d_e"] == pytest.approx(0.25, abs=1e-12)
    # mean_share_ge_010 = (0.0 + 0.0 + 0.0 + 0.0) / 4 = 0.0
    assert a["mean_share_ge_010"] == pytest.approx(0.0, abs=1e-12)
    # mean_p0_share_ge_010 = (0.0 + 0.0 + 0.0 + 0.0) / 4 = 0.0
    assert a["mean_p0_share_ge_010"] == pytest.approx(0.0, abs=1e-12)

    # Scenario B: all unreachable
    b = agg[("B", 0.3, 0.5)]
    assert b["n_pairs"] == 6
    assert b["n_seeds"] == 2
    assert b["reachable_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert b["p0_reachable_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(b["p0_blocked_fraction"])
    assert np.isnan(b["mean_detour_ratio"])
    assert b["n_finite_detour"] == 0
    assert np.isnan(b["mean_exposure_reduction"])
    assert b["n_finite_exposure"] == 0
    assert np.isnan(b["frac_lower_exposure"])
    assert np.isnan(b["frac_same_exposure"])
    assert np.isnan(b["frac_higher_exposure"])
    assert np.isnan(b["mean_max_depth_reduction"])
    assert np.isnan(b["mean_max_d_e"])
    assert np.isnan(b["mean_p0_max_d_e"])
    assert np.isnan(b["mean_share_ge_010"])
    assert np.isnan(b["mean_p0_share_ge_010"])


def test_aggregate_policy_rows_nan_handling():
    """Unreachable policy rows excluded from detour/exposure means."""
    rows = [
        _row(1, "A", 0, 0.3, 0.5, True, 2.0, 10.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 3, 0, True, 1.0, 9.0, 0.3, 0.1, 0.0, 0, 1.1, -0.1, -0.1),
        _row(1, "A", 1, 0.3, 0.5, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0),
    ]
    aggregated = aggregate_policy_rows(rows)
    a = aggregated[0]
    assert a["n_pairs"] == 2
    assert a["reachable_fraction"] == pytest.approx(0.5, abs=1e-12)
    # Only one both-reachable row, so detour = 1.1, exposure = -0.1.
    assert a["mean_detour_ratio"] == pytest.approx(1.1, abs=1e-12)
    assert a["n_finite_detour"] == 1
    assert a["mean_exposure_reduction"] == pytest.approx(-0.1, abs=1e-12)
    assert a["n_finite_exposure"] == 1


def test_aggregate_policy_rows_missing_key_raises():
    rows = [{"seed": 1, "scenario": "A"}]
    with pytest.raises(ValueError, match="missing required key"):
        aggregate_policy_rows(rows)


# ---------------------------------------------------------------------------
# check_policy_invariants
# ---------------------------------------------------------------------------


def test_check_policy_invariants_clean():
    rows = [
        _row(1, "A", 0, 0.3, 0.5, True, 2.0, 10.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 3, 0, True, 1.0, 9.0, 0.3, 0.1, 0.0, 0, 1.1, -0.1, -0.1),
        _row(1, "A", 1, 0.3, 0.5, True, 3.0, 20.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 4, 0, True, 2.5, 18.0, 0.4, 0.2, 0.0, 0, 1.2, -0.1, -0.1),
    ]
    violations = check_policy_invariants(rows)
    assert violations == []


def test_check_policy_invariants_reachable_with_closure():
    rows = [
        _row(1, "A", 0, 0.3, 0.5, True, 2.0, 10.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 3, 1, True, 1.0, 9.0, 0.3, 0.1, 0.0, 0, 1.1, -0.1, -0.1),
    ]
    violations = check_policy_invariants(rows)
    assert len(violations) == 1
    assert "n_ge_closure" in violations[0]


def test_check_policy_invariants_lam_zero_faster_than_p0():
    rows = [
        _row(1, "A", 0, 0.3, 0.0, True, 1.5, 10.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 3, 0, True, 2.0, 9.0, 0.3, 0.1, 0.0, 0, 1.1, -0.1, -0.1),
    ]
    violations = check_policy_invariants(rows)
    assert len(violations) == 1
    assert "ff_time_s" in violations[0]


def test_check_policy_invariants_detour_inconsistent():
    rows = [
        _row(1, "A", 0, 0.3, 0.5, True, 2.0, 10.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 3, 0, True, 1.0, 9.0, 0.3, 0.1, 0.0, 0, 0.5, -0.1, -0.1),
    ]
    violations = check_policy_invariants(rows)
    assert len(violations) == 1
    assert "detour_ratio" in violations[0]


def test_check_policy_invariants_exposure_inconsistent():
    rows = [
        _row(1, "A", 0, 0.3, 0.5, True, 2.0, 10.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 3, 0, True, 1.0, 9.0, 0.3, 0.1, 0.0, 0, 1.1, -0.9, -0.1),
    ]
    violations = check_policy_invariants(rows)
    assert len(violations) == 1
    assert "exposure_reduction" in violations[0]


def test_check_policy_invariants_not_both_reachable_with_fields():
    rows = [
        _row(1, "A", 0, 0.3, 0.5, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.1, float("nan"), float("nan")),
    ]
    violations = check_policy_invariants(rows)
    assert len(violations) == 1
    assert "detour_ratio" in violations[0]


def test_check_policy_invariants_missing_key_raises():
    rows = [{"seed": 1, "scenario": "A"}]
    with pytest.raises(ValueError, match="missing required key"):
        check_policy_invariants(rows)


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


def test_experiment_c_import_isolation():
    spec = importlib.util.find_spec("urbanflood.experiment_c")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        tree = ast.parse(f.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    allowed = {
        "numpy",
        "typing",
        "collections",
        "collections.abc",
        "dataclasses",
        "__future__",
    }
    for name in imported:
        assert any(
            name == a or name.startswith(a + ".") for a in allowed
        ), f"disallowed import: {name}"
    assert not any("urbanflood" in n for n in imported)
