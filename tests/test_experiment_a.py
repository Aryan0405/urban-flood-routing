"""Tests for src/urbanflood/experiment_a.py (Phase 3 Experiment A metrics).

Fast tests only: toy data, no simulation.
"""

from __future__ import annotations

import ast
import importlib.util

import numpy as np
import pytest

from urbanflood.experiment_a import (
    ablated_index,
    evaluate_methods,
    permuted_scores,
    roc_auc,
    spearman,
    summarize,
    top_decile_lift,
)


# ---------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------

def test_spearman_perfect_increasing():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, y) == pytest.approx(1.0, abs=1e-12)


def test_spearman_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert spearman(x, y) == pytest.approx(-1.0, abs=1e-12)


def test_spearman_constant_returns_nan():
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    assert np.isnan(spearman(x, y))
    assert np.isnan(spearman(y, x))


def test_spearman_tie_case_hand_computed():
    """x = [1, 2, 2, 3], y = [1, 2, 3, 4].
    ranks x = [1, 2.5, 2.5, 4], ranks y = [1, 2, 3, 4]
    Pearson on ranks = 4.5 / sqrt(4.5 * 5.0) = 0.9486832980505138
    """
    x = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    expected = 4.5 / np.sqrt(4.5 * 5.0)
    assert spearman(x, y) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# roc_auc
# ---------------------------------------------------------------------------

def test_roc_auc_perfect():
    score = np.array([0.1, 0.2, 0.9, 1.0])
    flooded = np.array([False, False, True, True])
    assert roc_auc(score, flooded) == pytest.approx(1.0, abs=1e-12)


def test_roc_auc_reversed():
    score = np.array([0.9, 1.0, 0.1, 0.2])
    flooded = np.array([False, False, True, True])
    assert roc_auc(score, flooded) == pytest.approx(0.0, abs=1e-12)


def test_roc_auc_all_tied():
    score = np.array([0.5, 0.5, 0.5, 0.5])
    flooded = np.array([False, False, True, True])
    assert roc_auc(score, flooded) == pytest.approx(0.5, abs=1e-12)


def test_roc_auc_no_positives_returns_nan():
    score = np.array([0.1, 0.2, 0.3, 0.4])
    flooded = np.array([False, False, False, False])
    assert np.isnan(roc_auc(score, flooded))


def test_roc_auc_hand_computed():
    """score = [0.1, 0.4, 0.2, 0.3], flooded = [F, T, T, F].
    ranks: 0.1->1, 0.2->2, 0.3->3, 0.4->4
    sum_ranks_pos = 4 + 2 = 6, n_pos = 2, n_neg = 2
    u = 6 - 2*3/2 = 3, auc = 3/(2*2) = 0.75
    """
    score = np.array([0.1, 0.4, 0.2, 0.3])
    flooded = np.array([False, True, True, False])
    assert roc_auc(score, flooded) == pytest.approx(0.75, abs=1e-12)


# ---------------------------------------------------------------------------
# top_decile_lift
# ---------------------------------------------------------------------------

def test_top_decile_lift_hand_computed():
    """n=20, 4 positives, both top-2 cells positive.
    top group = ceil(0.10*20) = 2 cells, flooded share top = 2/2 = 1.0
    overall flooded share = 4/20 = 0.2, lift = 1.0/0.2 = 5.0
    """
    n = 20
    score = np.zeros(n, dtype=np.float64)
    flooded = np.zeros(n, dtype=bool)
    score[0] = 100.0  # positive, top rank
    score[1] = 99.0   # positive, top rank
    flooded[0] = True
    flooded[1] = True
    flooded[2] = True
    flooded[3] = True
    assert top_decile_lift(score, flooded, frac=0.10) == pytest.approx(5.0, abs=1e-12)


def test_top_decile_lift_no_positives_returns_nan():
    score = np.array([0.1, 0.2, 0.3])
    flooded = np.array([False, False, False])
    assert np.isnan(top_decile_lift(score, flooded))


# ---------------------------------------------------------------------------
# ablated_index
# ---------------------------------------------------------------------------

def test_ablated_index_hand_computed():
    feature_scores = {
        "elevation": np.array([[1.0, 0.0], [0.5, 0.5]]),
        "tpi": np.array([[0.0, 1.0], [0.5, 0.0]]),
    }
    weights = {"elevation": 0.5, "tpi": 0.5}
    # Drop tpi: result = elevation * 1.0
    result = ablated_index(feature_scores, weights, drop="tpi")
    expected = np.array([[1.0, 0.0], [0.5, 0.5]])
    np.testing.assert_array_equal(result, expected)
    # Drop elevation: result = tpi * 1.0
    result = ablated_index(feature_scores, weights, drop="elevation")
    expected = np.array([[0.0, 1.0], [0.5, 0.0]])
    np.testing.assert_array_equal(result, expected)


def test_ablated_index_renormalised_weights():
    feature_scores = {
        "elevation": np.array([[1.0, 0.0], [0.5, 0.5]]),
        "tpi": np.array([[0.0, 1.0], [0.5, 0.0]]),
        "slope": np.array([[0.2, 0.8], [0.1, 0.9]]),
    }
    weights = {"elevation": 0.35, "tpi": 0.25, "slope": 0.40}
    result = ablated_index(feature_scores, weights, drop="slope")
    # Remaining: elevation (0.35), tpi (0.25), sum = 0.60
    # norm: elevation = 0.35/0.60, tpi = 0.25/0.60
    expected = (
        (0.35 / 0.60) * np.array([[1.0, 0.0], [0.5, 0.5]])
        + (0.25 / 0.60) * np.array([[0.0, 1.0], [0.5, 0.0]])
    )
    np.testing.assert_array_almost_equal(result, expected)


def test_ablated_index_unknown_name_raises():
    feature_scores = {"elevation": np.zeros((2, 2)), "tpi": np.zeros((2, 2))}
    weights = {"elevation": 0.5, "tpi": 0.5}
    with pytest.raises(ValueError, match="unknown feature"):
        ablated_index(feature_scores, weights, drop="nonexistent")


def test_ablated_index_nan_propagates():
    feature_scores = {
        "elevation": np.array([[1.0, np.nan], [0.5, 0.5]]),
        "tpi": np.array([[0.0, 1.0], [0.5, 0.0]]),
    }
    weights = {"elevation": 0.5, "tpi": 0.5}
    # Drop tpi: only elevation remains, elevation[0,1] is NaN
    result = ablated_index(feature_scores, weights, drop="tpi")
    assert result[0, 0] == 1.0
    assert np.isnan(result[0, 1])
    # Drop elevation: only tpi remains, no NaN
    result = ablated_index(feature_scores, weights, drop="elevation")
    assert result[0, 1] == 1.0


# ---------------------------------------------------------------------------
# permuted_scores
# ---------------------------------------------------------------------------

def test_permuted_scores_deterministic():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p1 = permuted_scores(values, seed=42)
    p2 = permuted_scores(values, seed=42)
    np.testing.assert_array_equal(p1, p2)


def test_permuted_scores_preserves_multiset():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p = permuted_scores(values, seed=7)
    assert np.array_equal(np.sort(p), np.sort(values))


def test_permuted_scores_differs_from_input():
    values = np.arange(100, dtype=np.float64)
    p = permuted_scores(values, seed=1)
    assert not np.array_equal(p, values)


# ---------------------------------------------------------------------------
# evaluate_methods
# ---------------------------------------------------------------------------

def test_evaluate_methods_fields():
    n = 50
    rng = np.random.default_rng(0)
    methods = {
        "index": rng.random(n),
        "elevation_only": rng.random(n),
    }
    target = rng.random(n)
    rows = evaluate_methods(methods, target)
    assert len(rows) == 2
    expected_fields = {
        "method", "spearman", "auc_010", "auc_030", "auc_050",
        "lift_010", "lift_030", "lift_050", "n_land",
        "frac_flooded_010", "frac_flooded_030", "frac_flooded_050",
    }
    for row in rows:
        assert set(row.keys()) == expected_fields
        assert row["n_land"] == n


def test_evaluate_methods_constant_target_nan():
    n = 50
    methods = {"index": np.random.default_rng(1).random(n)}
    target = np.ones(n)  # constant
    rows = evaluate_methods(methods, target)
    assert np.isnan(rows[0]["spearman"])


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------

def test_experiment_a_import_isolation():
    spec = importlib.util.find_spec("urbanflood.experiment_a")
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
        "scipy",
        "scipy.stats",
        "collections",
        "collections.abc",
        "__future__",
        "typing",
        "dataclasses",
    }
    for name in imported:
        assert any(
            name == a or name.startswith(a + ".") for a in allowed
        ), f"disallowed import: {name}"
    assert not any("urbanflood" in n for n in imported)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def _make_row(seed, scenario, method, spearman, auc_030):
    return {
        "seed": seed,
        "scenario": scenario,
        "method": method,
        "spearman": spearman,
        "auc_010": 0.5,
        "auc_030": auc_030,
        "auc_050": 0.5,
        "lift_010": 1.0,
        "lift_030": 1.0,
        "lift_050": 1.0,
        "n_land": 100,
        "frac_flooded_010": 0.1,
        "frac_flooded_030": 0.1,
        "frac_flooded_050": 0.1,
    }


def test_summarize_wins_and_losses_by_seed():
    """2 scenarios x 3 seeds. Scenario A: index wins 2 of 3. Scenario B: wins 0 of 3."""
    rows = [
        _make_row(1, "A", "index", 0.5, 0.8),
        _make_row(1, "A", "elevation_only", 0.4, 0.5),
        _make_row(2, "A", "index", 0.3, 0.4),
        _make_row(2, "A", "elevation_only", 0.4, 0.6),
        _make_row(3, "A", "index", 0.6, 0.9),
        _make_row(3, "A", "elevation_only", 0.5, 0.7),
        _make_row(1, "B", "index", 0.3, 0.4),
        _make_row(1, "B", "elevation_only", 0.5, 0.8),
        _make_row(2, "B", "index", 0.2, 0.3),
        _make_row(2, "B", "elevation_only", 0.4, 0.7),
        _make_row(3, "B", "index", 0.1, 0.2),
        _make_row(3, "B", "elevation_only", 0.5, 0.6),
    ]
    aggregated, comparison = summarize(rows)
    assert comparison["A"]["n_pairs"] == 3
    assert comparison["A"]["wins"] == 2
    assert comparison["A"]["ties"] == 0
    assert comparison["A"]["losses"] == 1
    assert comparison["A"]["n_skipped_nan"] == 0
    assert comparison["B"]["n_pairs"] == 3
    assert comparison["B"]["wins"] == 0
    assert comparison["B"]["ties"] == 0
    assert comparison["B"]["losses"] == 3
    assert comparison["B"]["n_skipped_nan"] == 0


def test_summarize_nan_auc_030_skipped():
    """Seed 2 has NaN auc_030 for index: counted as skipped, excluded from n_pairs."""
    rows = [
        _make_row(1, "A", "index", 0.5, 0.8),
        _make_row(1, "A", "elevation_only", 0.4, 0.5),
        _make_row(2, "A", "index", 0.3, float("nan")),
        _make_row(2, "A", "elevation_only", 0.4, 0.6),
        _make_row(3, "A", "index", 0.6, 0.9),
        _make_row(3, "A", "elevation_only", 0.5, 0.7),
    ]
    aggregated, comparison = summarize(rows)
    assert comparison["A"]["n_pairs"] == 2
    assert comparison["A"]["n_skipped_nan"] == 1
    assert comparison["A"]["wins"] == 2


def test_summarize_tie_not_win():
    """Equal auc_030 values count as ties, not wins."""
    rows = [
        _make_row(1, "A", "index", 0.5, 0.8),
        _make_row(1, "A", "elevation_only", 0.4, 0.8),
        _make_row(2, "A", "index", 0.3, 0.4),
        _make_row(2, "A", "elevation_only", 0.4, 0.6),
    ]
    aggregated, comparison = summarize(rows)
    assert comparison["A"]["n_pairs"] == 2
    assert comparison["A"]["wins"] == 0
    assert comparison["A"]["ties"] == 1
    assert comparison["A"]["losses"] == 1


def test_summarize_mean_diff_hand_computed():
    """mean_diff_auc_030 = mean(index - elevation_only) over pairs."""
    rows = [
        _make_row(1, "A", "index", 0.5, 0.8),
        _make_row(1, "A", "elevation_only", 0.4, 0.5),
        _make_row(2, "A", "index", 0.3, 0.4),
        _make_row(2, "A", "elevation_only", 0.4, 0.6),
    ]
    aggregated, comparison = summarize(rows)
    assert comparison["A"]["mean_diff_auc_030"] == pytest.approx(0.05, abs=1e-12)


def test_summarize_pairing_by_seed_not_cross_product():
    """Cross-product would count 2 wins, per-seed pairing counts 1 win."""
    rows = [
        _make_row(1, "A", "index", 0.5, 0.9),
        _make_row(1, "A", "elevation_only", 0.4, 0.5),
        _make_row(2, "A", "index", 0.3, 0.1),
        _make_row(2, "A", "elevation_only", 0.4, 0.8),
    ]
    aggregated, comparison = summarize(rows)
    assert comparison["A"]["n_pairs"] == 2
    assert comparison["A"]["wins"] == 1
    assert comparison["A"]["losses"] == 1


def test_summarize_missing_seed_raises():
    """Rows without 'seed' must raise ValueError."""
    rows = [
        {"scenario": "A", "method": "index", "spearman": 0.5, "auc_030": 0.8},
    ]
    with pytest.raises(ValueError, match="seed"):
        summarize(rows)


def test_summarize_means_ignore_nan_and_n_finite():
    """Means ignore NaN; n_finite counts match."""
    rows = [
        _make_row(1, "A", "index", float("nan"), 0.8),
        _make_row(2, "A", "index", 0.3, float("nan")),
        _make_row(3, "A", "index", 0.6, 0.9),
        _make_row(1, "A", "elevation_only", 0.4, 0.5),
        _make_row(2, "A", "elevation_only", 0.4, 0.6),
        _make_row(3, "A", "elevation_only", 0.5, 0.7),
    ]
    aggregated, comparison = summarize(rows)
    idx_rows = [r for r in aggregated if r["scenario"] == "A" and r["method"] == "index"]
    assert len(idx_rows) == 1
    r = idx_rows[0]
    assert r["n_rows"] == 3
    assert r["mean_spearman"] == pytest.approx(0.45, abs=1e-12)
    assert r["n_finite_spearman"] == 2
    assert r["mean_auc_030"] == pytest.approx(0.85, abs=1e-12)
    assert r["n_finite_auc_030"] == 2


def test_summarize_all_seven_metrics_present():
    """aggregated rows contain all seven metrics with mean and n_finite fields."""
    rows = [
        _make_row(1, "A", "index", 0.5, 0.8),
        _make_row(1, "A", "elevation_only", 0.4, 0.5),
    ]
    aggregated, _ = summarize(rows)
    expected_fields = {
        "scenario", "method", "n_rows",
        "mean_spearman", "n_finite_spearman",
        "mean_auc_010", "n_finite_auc_010",
        "mean_auc_030", "n_finite_auc_030",
        "mean_auc_050", "n_finite_auc_050",
        "mean_lift_010", "n_finite_lift_010",
        "mean_lift_030", "n_finite_lift_030",
        "mean_lift_050", "n_finite_lift_050",
    }
    for r in aggregated:
        assert set(r.keys()) == expected_fields


def test_summarize_all_scenarios_present_even_zero_wins():
    """Scenario B with zero wins still appears in comparison."""
    rows = [
        _make_row(1, "A", "index", 0.5, 0.9),
        _make_row(1, "A", "elevation_only", 0.4, 0.5),
        _make_row(1, "B", "index", 0.1, 0.2),
        _make_row(1, "B", "elevation_only", 0.5, 0.8),
    ]
    aggregated, comparison = summarize(rows)
    assert "A" in comparison
    assert "B" in comparison
    assert comparison["B"]["wins"] == 0
    assert comparison["B"]["n_pairs"] == 1
    assert comparison["B"]["losses"] == 1
