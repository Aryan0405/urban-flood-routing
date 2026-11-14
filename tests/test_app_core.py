r"""Tests for src/urbanflood/app_core (app data layer core).

Fast tests only: uses the committed app_data directory (default), one
environment build at module scope. No simulation, no streamlit, no expA.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from urbanflood.app_core import (
    APP_DATA_DIR,
    build_d_e,
    build_env,
    compute_routes,
    count_pairs,
    load_experiment_a_rows,
    load_hpeak,
    load_manifest,
    reachable_count,
    summarize_experiment_a,
)
from urbanflood.config import build_default_config

SEED_INDEX = 0
SCENARIO = "R"


@pytest.fixture(scope="module")
def env_cfg():
    env, cfg = build_env(SEED_INDEX)
    return env, cfg


# ---------------------------------------------------------------------------
# manifest (committed app_data)
# ---------------------------------------------------------------------------


def test_manifest_matches_config_seeds():
    cfg = build_default_config()
    expected_seeds = list(cfg.seeds.env_experiment_seeds())
    m = load_manifest()
    assert len(m) == 5
    assert [m[i] for i in sorted(m.keys())] == expected_seeds


def test_app_data_files_exist_and_shape():
    """All 20 hpeak files and manifest.txt exist in app_data; each hpeak is
    shape (96, 96)."""
    assert APP_DATA_DIR.is_dir()
    manifest = load_manifest()
    assert len(manifest) == 5
    for i in range(5):
        for scenario in ("R", "S1", "S2", "S3"):
            p = APP_DATA_DIR / ("hpeak_s%d_%s.txt" % (i, scenario))
            assert p.is_file(), "missing %s" % p
            arr = load_hpeak(i, scenario)
            assert arr.shape == (96, 96), "bad shape for %s" % p
            assert np.all(np.isfinite(arr)), "non-finite in %s" % p
    assert (APP_DATA_DIR / "manifest.txt").is_file()


# ---------------------------------------------------------------------------
# hpeak
# ---------------------------------------------------------------------------


def test_hpeak_shape_and_finite(env_cfg):
    env, cfg = env_cfg
    arr = load_hpeak(SEED_INDEX, SCENARIO)
    assert arr.shape == (96, 96)
    assert arr.dtype == np.float64
    assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# build_d_e closed edges
# ---------------------------------------------------------------------------


def test_build_d_e_closed_edges_at_030(env_cfg):
    env, cfg = env_cfg
    arr = load_hpeak(SEED_INDEX, SCENARIO)
    d_e = build_d_e(env, arr)
    closed = int(np.sum(d_e >= 0.30))
    assert closed == 361


# ---------------------------------------------------------------------------
# reachable_count
# ---------------------------------------------------------------------------


def test_reachable_count_030_lam0(env_cfg):
    env, cfg = env_cfg
    arr = load_hpeak(SEED_INDEX, SCENARIO)
    d_e = build_d_e(env, arr)
    n = reachable_count(env, d_e, 0.30, 0.0)
    assert n == 13


# ---------------------------------------------------------------------------
# compute_routes
# ---------------------------------------------------------------------------


def test_compute_routes_pair0_consistent(env_cfg):
    env, cfg = env_cfg
    arr = load_hpeak(SEED_INDEX, SCENARIO)
    d_e = build_d_e(env, arr)
    res = compute_routes(env, d_e, 0.30, 0.0, 0)
    assert set(res.keys()) == {
        "p0_edges", "policy_edges", "p0_reachable",
        "policy_reachable", "closed_edge_ids", "od",
    }
    u, v = res["od"]
    assert u != v
    assert res["p0_reachable"] is True  # P0 has no closures
    assert isinstance(res["p0_edges"], list)
    assert isinstance(res["policy_edges"], (list, type(None)))
    assert isinstance(res["closed_edge_ids"], list)
    assert len(res["closed_edge_ids"]) == 361


def test_compute_routes_unreachable_policy_returns_none(env_cfg):
    env, cfg = env_cfg
    arr = load_hpeak(SEED_INDEX, SCENARIO)
    d_e = build_d_e(env, arr)
    from urbanflood.app_core import _od_pairs
    od_pairs = _od_pairs(env, cfg)
    found = False
    for pair_index in range(len(od_pairs)):
        res = compute_routes(env, d_e, 0.30, 0.0, pair_index)
        if not res["policy_reachable"]:
            assert res["policy_edges"] is None
            found = True
            break
    assert found, "expected at least one unreachable policy pair"


# ---------------------------------------------------------------------------
# experiment_a rows
# ---------------------------------------------------------------------------


def test_load_experiment_a_rows_columns():
    rows = load_experiment_a_rows()
    assert len(rows) > 0
    assert "seed" in rows[0]
    assert "scenario" in rows[0]
    assert "method" in rows[0]
    assert "auc_030" in rows[0]
    assert isinstance(rows[0]["seed"], str)
    assert isinstance(rows[0]["auc_030"], str)


# ---------------------------------------------------------------------------
# summarize_experiment_a / count_pairs on toy rows
# ---------------------------------------------------------------------------


def _toy_rows():
    return [
        {"seed": "1", "scenario": "R", "method": "index", "auc_030": "0.8", "lift_030": "3.0"},
        {"seed": "1", "scenario": "R", "method": "elevation_only", "auc_030": "0.7", "lift_030": "2.5"},
        {"seed": "1", "scenario": "R", "method": "random_permutation", "auc_030": "0.5", "lift_030": "1.0"},
        {"seed": "2", "scenario": "R", "method": "index", "auc_030": "0.6", "lift_030": "2.0"},
        {"seed": "2", "scenario": "R", "method": "elevation_only", "auc_030": "0.65", "lift_030": "3.0"},
        {"seed": "2", "scenario": "R", "method": "random_permutation", "auc_030": "0.5", "lift_030": "1.0"},
    ]


def test_summarize_experiment_a_toy():
    rows = _toy_rows()
    out = summarize_experiment_a(rows)
    by_key = {(d["scenario"], d["method"]): d for d in out}
    r_idx = by_key[("R", "index")]
    assert r_idx["n_seeds"] == 2
    assert r_idx["mean_auc_030"] == pytest.approx(0.7, abs=1e-12)
    assert r_idx["min_auc_030"] == pytest.approx(0.6, abs=1e-12)
    assert r_idx["max_auc_030"] == pytest.approx(0.8, abs=1e-12)
    assert r_idx["mean_lift_030"] == pytest.approx(2.5, abs=1e-12)
    r_elev = by_key[("R", "elevation_only")]
    assert r_elev["mean_auc_030"] == pytest.approx(0.675, abs=1e-12)
    assert r_elev["mean_lift_030"] == pytest.approx(2.75, abs=1e-12)


def test_count_pairs_toy():
    rows = _toy_rows()
    c = count_pairs(rows)
    assert c["n_pairs"] == 2
    # seed 1: index 0.8 > elev 0.7 -> index win; elev lift 2.5 > index 3.0? no
    # seed 2: index 0.6 > elev 0.65? no; elev lift 3.0 > index 2.0 -> elev win
    assert c["index_auc_wins_030"] == 1
    assert c["elevation_lift_wins_030"] == 1


# ---------------------------------------------------------------------------
# summarize_experiment_a / count_pairs on real results.csv
# ---------------------------------------------------------------------------


def test_summarize_experiment_a_real_values():
    rows = load_experiment_a_rows()
    out = summarize_experiment_a(rows)
    by_key = {(d["scenario"], d["method"]): d for d in out}
    expected = {
        ("R", "index"): 0.722,
        ("R", "elevation_only"): 0.669,
        ("S1", "index"): 0.741,
        ("S1", "elevation_only"): 0.701,
        ("S2", "index"): 0.820,
        ("S2", "elevation_only"): 0.782,
        ("S3", "index"): 0.725,
        ("S3", "elevation_only"): 0.673,
    }
    for (scenario, method), exp_mean in expected.items():
        d = by_key[(scenario, method)]
        assert d["n_seeds"] == 5, "%s %s n_seeds" % (scenario, method)
        assert abs(d["mean_auc_030"] - exp_mean) <= 5e-4, (
            "%s %s mean auc_030=%.6f expected ~%.3f"
            % (scenario, method, d["mean_auc_030"], exp_mean)
        )


def test_count_pairs_real_values():
    rows = load_experiment_a_rows()
    c = count_pairs(rows)
    assert c["n_pairs"] == 20, "n_pairs=%d" % c["n_pairs"]
    assert c["index_auc_wins_030"] == 20, "index_auc_wins_030=%d" % c["index_auc_wins_030"]


# ---------------------------------------------------------------------------
# Import isolation: app_core must not import streamlit or matplotlib
# ---------------------------------------------------------------------------


def test_app_core_import_isolation():
    spec = importlib.util.find_spec("urbanflood.app_core")
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
    for name in imported:
        assert "streamlit" not in name, f"disallowed import: {name}"
        assert "matplotlib" not in name, f"disallowed import: {name}"
