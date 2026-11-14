"""Tests for src/urbanflood/susceptibility.py (Phase 3).

Fast tests only: environment generation, no run_simulation.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect

import numpy as np
import pytest

from urbanflood.config import build_default_config, SusceptibilityConfig
from urbanflood.environment import (
    LakeSpec,
    build_environment,
)
from urbanflood.susceptibility import (
    SusceptibilityResult,
    build_susceptibility,
    elevation_score,
    land_cover_score,
    slope_score,
    tpi_score,
    water_distance_score,
)

MASTER = 2027
CONFIG = build_default_config()
ENV = build_environment(MASTER, CONFIG)


# ---------------------------------------------------------------------------
# Toy fixtures with hand-computed expectations
# ---------------------------------------------------------------------------

def _make_toy_env(n_rows: int, n_cols: int, cell_size_m: float = 50.0):
    """Build a minimal environment-like object for toy tests."""
    from types import SimpleNamespace
    grid = SimpleNamespace(cell_size_m=cell_size_m)
    return SimpleNamespace(
        grid=grid,
        elevation_m=np.zeros((n_rows, n_cols), dtype=np.float64),
        is_ocean=np.zeros((n_rows, n_cols), dtype=bool),
        is_lake=np.zeros((n_rows, n_cols), dtype=bool),
        land_cover=np.zeros((n_rows, n_cols), dtype=np.uint8),
    )


def test_elevation_score_toy():
    """z=0 -> 1.0, z=20 -> 0.5, z=40+ -> 0.0."""
    cfg = SusceptibilityConfig()
    elevation = np.array([[0.0, 20.0, 40.0, 60.0]], dtype=np.float64)
    is_ocean = np.zeros((1, 4), dtype=bool)
    score, clip = elevation_score(elevation, is_ocean, cfg)
    assert score[0, 0] == 1.0
    assert score[0, 1] == 0.5
    assert score[0, 2] == 0.0
    assert score[0, 3] == 0.0


def test_slope_score_toy():
    """Linear ramp: slope = 1/50 = 0.02, score = 1 - 0.02/0.3."""
    cfg = SusceptibilityConfig()
    elevation = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float64)
    is_ocean = np.zeros((2, 3), dtype=bool)
    score = slope_score(elevation, is_ocean, 50.0, cfg)
    expected_slope = 1.0 / 50.0
    expected_score = 1.0 - expected_slope / cfg.slope_max
    assert score[0, 1] == pytest.approx(expected_score, rel=1e-12)


def test_water_distance_score_toy():
    """Row of water cells: ocean cell is NaN, next cell is 50 m away."""
    cfg = SusceptibilityConfig()
    is_ocean = np.array([[True, False, False, False]], dtype=bool)
    is_lake = np.zeros((1, 4), dtype=bool)
    score = water_distance_score(is_ocean, is_lake, 50.0, cfg)
    assert np.isnan(score[0, 0])  # ocean cell is NaN
    assert score[0, 1] == pytest.approx(1.0 - 50.0 / 1000.0, rel=1e-12)


def test_land_cover_score_toy():
    """Each class code maps to its runoff_index value."""
    cfg = SusceptibilityConfig()
    land_cover = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    score = land_cover_score(land_cover, cfg)
    assert score[0, 0] == cfg.runoff_index_impervious  # 1.0
    assert score[0, 1] == cfg.runoff_index_open_ground  # 0.6
    assert score[0, 2] == cfg.runoff_index_vegetation  # 0.25
    assert score[0, 3] == cfg.runoff_index_water  # 1.0


def test_tpi_score_flat_plane():
    """Flat plane: TPI = 0 -> score 0.5 everywhere."""
    cfg = SusceptibilityConfig()
    n = 21
    elevation = np.full((n, n), 10.0, dtype=np.float64)
    is_ocean = np.zeros((n, n), dtype=bool)
    score = tpi_score(elevation, is_ocean, 50.0, cfg)
    valid = score[~np.isnan(score)]
    assert np.all(valid == 0.5)


def _disk_count(radius_cells: int) -> int:
    yy, xx = np.meshgrid(
        np.arange(-radius_cells, radius_cells + 1),
        np.arange(-radius_cells, radius_cells + 1),
    )
    return int(((xx**2 + yy**2) <= radius_cells**2).sum())


def test_tpi_score_pit_deeper_than_span():
    """Pit deeper than tpi_span -> centre score = (tpi_span - TPI)/(2*tpi_span) with TPI = -480/49."""
    cfg = SusceptibilityConfig(tpi_span_m=10.0)
    n = 21
    elevation = np.full((n, n), 10.0, dtype=np.float64)
    centre = n // 2
    elevation[centre, centre] = 0.0
    is_ocean = np.zeros((n, n), dtype=bool)
    score = tpi_score(elevation, is_ocean, 50.0, cfg)
    radius_cells = int(np.ceil(cfg.tpi_radius_m / 50.0))
    footprint_count = _disk_count(radius_cells)
    assert footprint_count == 49
    mean_z = 480.0 / footprint_count
    tpi = 0.0 - mean_z
    expected = (cfg.tpi_span_m - tpi) / (2.0 * cfg.tpi_span_m)
    assert score[centre, centre] == pytest.approx(expected, abs=1e-12)


def test_tpi_score_peak_higher_than_span():
    """Peak higher than tpi_span -> centre score clips to exactly 0.0."""
    cfg = SusceptibilityConfig(tpi_span_m=10.0)
    n = 21
    elevation = np.full((n, n), 10.0, dtype=np.float64)
    centre = n // 2
    elevation[centre, centre] = 30.0
    is_ocean = np.zeros((n, n), dtype=bool)
    score = tpi_score(elevation, is_ocean, 50.0, cfg)
    radius_cells = int(np.ceil(cfg.tpi_radius_m / 50.0))
    footprint_count = _disk_count(radius_cells)
    assert footprint_count == 49
    mean_z = 510.0 / footprint_count
    tpi = 30.0 - mean_z
    expected = max(0.0, (cfg.tpi_span_m - tpi) / (2.0 * cfg.tpi_span_m))
    assert score[centre, centre] == pytest.approx(expected, abs=1e-12)


def test_tpi_score_linear_ramp():
    """Linear ramp z = 0.5 * column: TPI = 0 in interior, score 0.5 exactly."""
    cfg = SusceptibilityConfig()
    n = 21
    cols = np.arange(n, dtype=np.float64) * 0.5
    elevation = np.tile(cols, (n, 1))
    is_ocean = np.zeros((n, n), dtype=bool)
    score = tpi_score(elevation, is_ocean, 50.0, cfg)
    radius_cells = int(np.ceil(cfg.tpi_radius_m / 50.0))
    interior = np.zeros((n, n), dtype=bool)
    interior[
        radius_cells : n - radius_cells, radius_cells : n - radius_cells
    ] = True
    valid = score[interior]
    assert np.all(valid == pytest.approx(0.5, abs=1e-12))


# ---------------------------------------------------------------------------
# Weighted sum and range/mask checks
# ---------------------------------------------------------------------------

def test_index_equals_weighted_sum():
    """Index equals explicit weighted sum of the five feature arrays."""
    result = build_susceptibility(ENV)
    cfg = CONFIG.susceptibility
    scored = ~ENV.is_ocean
    expected = (
        cfg.weight_elevation * result.feature_scores["elevation"][scored]
        + cfg.weight_tpi * result.feature_scores["tpi"][scored]
        + cfg.weight_water_distance * result.feature_scores["water_distance"][scored]
        + cfg.weight_land_cover * result.feature_scores["land_cover"][scored]
        + cfg.weight_slope * result.feature_scores["slope"][scored]
    )
    assert np.allclose(result.index[scored], expected, atol=1e-12)


def test_weights_sum_to_one():
    """SusceptibilityConfig weights sum to 1.0."""
    cfg = CONFIG.susceptibility
    total = (
        cfg.weight_elevation
        + cfg.weight_tpi
        + cfg.weight_water_distance
        + cfg.weight_land_cover
        + cfg.weight_slope
    )
    assert abs(total - 1.0) < 1e-9


def test_range_and_masks():
    """All scored cells in [0,1], finite; ocean cells exactly NaN."""
    result = build_susceptibility(ENV)
    scored = ~ENV.is_ocean
    assert np.all(result.index[scored] >= 0.0)
    assert np.all(result.index[scored] <= 1.0)
    assert np.all(np.isfinite(result.index[scored]))
    assert np.all(np.isnan(result.index[ENV.is_ocean]))
    assert not np.any(np.isnan(result.index[scored]))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """Two calls bitwise equal; env unchanged."""
    result1 = build_susceptibility(ENV)
    result2 = build_susceptibility(ENV)
    assert result1.index.tobytes() == result2.index.tobytes()
    assert ENV.digest == build_environment(MASTER, CONFIG).digest


# ---------------------------------------------------------------------------
# TC4 structural: ast import parsing and parameter name check
# ---------------------------------------------------------------------------

def test_tc4_structural_imports():
    """Imported modules limited to allowed set; no forbidden parameter names."""
    spec = importlib.util.find_spec("urbanflood.susceptibility")
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
        "scipy.ndimage",
        "dataclasses",
        "typing",
        "urbanflood.config",
        "urbanflood.environment",
        "__future__",
    }
    for name in imported:
        assert any(
            name == a or name.startswith(a + ".") for a in allowed
        ), f"disallowed import: {name}"

    # No public function has a forbidden parameter name
    import urbanflood.susceptibility as susc_module
    for name, func in inspect.getmembers(susc_module, inspect.isfunction):
        sig = inspect.signature(func)
        for param_name in sig.parameters:
            assert not any(
                forbidden in param_name
                for forbidden in ["drainage", "outfall", "depth", "spill", "lake_spec"]
            ), f"forbidden parameter {param_name!r} in {name}"

    # TC4 structural: build_susceptibility may only access env attributes
    # elevation_m, is_ocean, is_lake, land_cover, and env.grid.cell_size_m.
    spec = importlib.util.find_spec("urbanflood.susceptibility")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        susc_tree = ast.parse(f.read())
    env_attrs: set[str] = set()
    env_grid_attrs: set[str] = set()
    for node in ast.walk(susc_tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "env":
                env_attrs.add(node.attr)
            elif (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "env"
                and node.value.attr == "grid"
            ):
                env_grid_attrs.add(node.attr)
    assert env_attrs <= {
        "elevation_m",
        "is_ocean",
        "is_lake",
        "land_cover",
        "grid",
    }, f"disallowed env attributes: {env_attrs}"
    assert env_grid_attrs <= {"cell_size_m"}, (
        f"disallowed env.grid attributes: {env_grid_attrs}"
    )


# ---------------------------------------------------------------------------
# TC4 behavioural: altered environment
# ---------------------------------------------------------------------------

def test_tc4_behavioural_altered_environment():
    """Index byte-identical when outfall fields and lake spill levels change."""
    result_original = build_susceptibility(ENV)
    altered = dataclasses.replace(
        ENV,
        outfall_mask=~ENV.outfall_mask,
        outfall_invert_m=ENV.outfall_invert_m + 5.0,
        lake_specs=tuple(
            LakeSpec(
                lake_id=s.lake_id,
                spill_level_m=s.spill_level_m + 3.0,
                min_bed_m=s.min_bed_m,
                carved_depth_m=s.carved_depth_m,
                center_row=s.center_row,
                center_col=s.center_col,
                sigma_cells=s.sigma_cells,
            )
            for s in ENV.lake_specs
        ),
    )
    result_altered = build_susceptibility(altered)
    assert result_altered.index.tobytes() == result_original.index.tobytes()


# ---------------------------------------------------------------------------
# Clip fractions
# ---------------------------------------------------------------------------

def test_clip_fractions_print(capsys):
    """Print elevation clip fractions for seed 2027."""
    result = build_susceptibility(ENV)
    with capsys.disabled():
        print(
            f"elevation clip_above_upper_bound: {result.clip_fractions['clip_above_upper_bound']}"
        )
        print(
            f"elevation clip_below_lower_bound: {result.clip_fractions['clip_below_lower_bound']}"
        )
    assert 0.0 <= result.clip_fractions["clip_above_upper_bound"] <= 1.0
    assert 0.0 <= result.clip_fractions["clip_below_lower_bound"] <= 1.0


# ---------------------------------------------------------------------------
# Multi-seed
# ---------------------------------------------------------------------------

def test_multi_seed():
    """Range/mask properties hold on extra seeds."""
    extra_seeds = [2028, 2029]
    for seed in extra_seeds:
        env = build_environment(seed, CONFIG)
        result = build_susceptibility(env)
        scored = ~env.is_ocean
        assert np.all(result.index[scored] >= 0.0)
        assert np.all(result.index[scored] <= 1.0)
        assert np.all(np.isfinite(result.index[scored]))
        assert np.all(np.isnan(result.index[env.is_ocean]))
        assert not np.any(np.isnan(result.index[scored]))


# ---------------------------------------------------------------------------
# Index statistics
# ---------------------------------------------------------------------------

def test_index_stats_print(capsys):
    """Print mean/std of index over land and Pearson(elevation_score, index)."""
    result = build_susceptibility(ENV)
    land_mask = (~ENV.is_ocean) & (~ENV.is_lake)
    index_land = result.index[land_mask]
    elev_score = result.feature_scores["elevation"]
    corr = float(np.corrcoef(elev_score[land_mask], index_land)[0, 1])
    with capsys.disabled():
        print(f"index mean over land: {float(index_land.mean())}")
        print(f"index std over land: {float(index_land.std())}")
        print(f"Pearson(elevation_score, index) over land: {corr}")
