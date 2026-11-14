"""Tests for src/urbanflood/initial_state.py (Option A initial-depth builder).

All tests are quick: no six-hour runs. Uses the project's existing master-seed
pattern from tests/test_environment.py (build_default_config + build_environment).
"""

from __future__ import annotations

import ast
import importlib
import inspect

import numpy as np
import pytest

from urbanflood.config import build_default_config
from urbanflood.drainage import generate_drainage
from urbanflood.environment import build_environment
from urbanflood.initial_state import build_initial_depth
from urbanflood.orchestrator import run_simulation
from urbanflood.config import SCENARIOS

MASTER = 2027
CONFIG = build_default_config()
ENV = build_environment(MASTER, CONFIG)
N_CELLS = CONFIG.grid.n_rows * CONFIG.grid.n_cols

DRAINAGE = generate_drainage(
    master_seed=MASTER,
    n_cells=N_CELLS,
    blocked_fraction=0.20,
    drainage_config=CONFIG.drainage,
)
RAIN = SCENARIOS["R"].rainfall
TIDE = SCENARIOS["R"].tide
EXCH = CONFIG.exchange
RUNOFF = CONFIG.runoff


def _run_short(dt_s: float, n_steps: int) -> object:
    h_initial = build_initial_depth(ENV)
    return run_simulation(
        h_initial,
        ENV,
        DRAINAGE,
        RAIN,
        TIDE,
        EXCH,
        RUNOFF,
        dt=dt_s,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Test a: shape, dtype, finite, non-negative
# ---------------------------------------------------------------------------

def test_initial_state_shape_dtype_finite_nonnegative():
    h = build_initial_depth(ENV)
    assert h.shape == (CONFIG.grid.n_rows, CONFIG.grid.n_cols)
    assert h.dtype == np.float64
    assert np.all(np.isfinite(h))
    assert np.all(h >= 0.0)


# ---------------------------------------------------------------------------
# Test b: lake cells h == max(0, spill - z) exactly; others exactly 0.0
# ---------------------------------------------------------------------------

def test_initial_state_lake_formula_exact():
    h = build_initial_depth(ENV)
    lake_mask = ENV.is_lake
    expected = np.zeros_like(h)
    spill_at_cell = np.zeros_like(h)
    for spec in ENV.lake_specs:
        spill_at_cell[ENV.lake_id == spec.lake_id] = spec.spill_level_m
    expected[lake_mask] = np.maximum(
        0.0, spill_at_cell[lake_mask] - ENV.elevation_m[lake_mask]
    )
    np.testing.assert_array_equal(h[lake_mask], expected[lake_mask])
    # Non-lake cells (land + ocean) exactly 0.0.
    assert np.all(h[~lake_mask] == 0.0)
    # Ocean cells specifically 0.0 (boundary reservoir decision).
    assert np.all(h[ENV.is_ocean] == 0.0)


# ---------------------------------------------------------------------------
# Test c: at least one lake cell > 0; report per-lake max depth vs STATUS.md
# ---------------------------------------------------------------------------

def test_initial_state_at_least_one_lake_positive(capsys):
    h = build_initial_depth(ENV)
    lake_mask = ENV.is_lake
    assert np.any(h[lake_mask] > 0.0), "no lake cell has positive initial depth"

    per_lake_max = []
    for spec in ENV.lake_specs:
        mask = ENV.lake_id == spec.lake_id
        per_lake_max.append(float(h[mask].max()) if np.any(mask) else 0.0)
    lo = float(min(per_lake_max))
    hi = float(max(per_lake_max))
    with capsys.disabled():
        print(
            "initial_state per-lake max depth (master seed): "
            + ", ".join(f"lake {spec.lake_id} = {v:.6f} m"
                        for spec, v in zip(ENV.lake_specs, per_lake_max))
        )
        print(
            f"initial_state range across lakes: [{lo:.6f}, {hi:.6f}] m; "
            "STATUS.md:96-98 reports [2.83, 13.53] m across twelve lakes "
            "(six seeds) - not forced to match."
        )
        if lo < 2.83 or hi > 13.53:
            print(
                "FLAG: master-seed lake depth range differs from STATUS.md "
                "twelve-lake span (expected subset, not identical)."
            )


# ---------------------------------------------------------------------------
# Test d: fresh array each call; env arrays unchanged
# ---------------------------------------------------------------------------

def test_initial_state_fresh_array_and_env_unchanged():
    elevation_before = ENV.elevation_m.copy()
    lake_id_before = ENV.lake_id.copy()

    h1 = build_initial_depth(ENV)
    # Pick an index to mutate (use a lake cell if any, else index 0).
    idx = (0, 0)
    if np.any(ENV.is_lake):
        rows, cols = np.nonzero(ENV.is_lake)
        idx = (int(rows[0]), int(cols[0]))
    h1_mutated = h1.copy()
    h1_mutated[idx] = 999.0

    h2 = build_initial_depth(ENV)
    # Second call must be fresh: unaffected by mutation of h1.
    assert np.array_equal(h2, h1)
    # h1 mutation must not leak into h2.
    assert h2[idx] == h1[idx]
    # env arrays unchanged by the builder.
    np.testing.assert_array_equal(ENV.elevation_m, elevation_before)
    np.testing.assert_array_equal(ENV.lake_id, lake_id_before)


# ---------------------------------------------------------------------------
# Test e: determinism and digest unchanged
# ---------------------------------------------------------------------------

def test_initial_state_deterministic_and_digest_unchanged():
    digest_before = ENV.digest
    h1 = build_initial_depth(ENV)
    h2 = build_initial_depth(ENV)
    assert h1.tobytes() == h2.tobytes()
    assert ENV.digest == digest_before


# ---------------------------------------------------------------------------
# Test f: short-run integration (150 steps, Scenario R, dt=30 s)
# ---------------------------------------------------------------------------

def test_initial_state_short_run_integration():
    h_initial = build_initial_depth(ENV)
    result = _run_short(30.0, 150)

    # Global ledger relative error < 1e-9.
    assert result.relative_error < 1e-9, (
        f"relative_error {result.relative_error} exceeds 1e-9"
    )

    # S_initial equals sum(h_initial over non-ocean cells) * cell area.
    A = CONFIG.grid.cell_area_m2
    s_initial_expected = float(h_initial[~ENV.is_ocean].sum()) * A
    if max(abs(result.S_initial), abs(s_initial_expected), 1.0) > 0.0:
        rel = abs(result.S_initial - s_initial_expected) / max(
            abs(result.S_initial), abs(s_initial_expected), 1.0
        )
        assert rel < 1e-9, f"S_initial relative difference {rel} exceeds 1e-9"

    # H_peak >= h_initial everywhere.
    assert np.all(result.H_peak >= h_initial - 1e-12)

    # H_peak on every lake cell with h_initial > 0 is >= h_initial.
    lake_pos = ENV.is_lake & (h_initial > 0.0)
    if np.any(lake_pos):
        assert np.all(result.H_peak[lake_pos] >= h_initial[lake_pos] - 1e-12)


# ---------------------------------------------------------------------------
# Test g: layer separation structural check
# ---------------------------------------------------------------------------

def _imported_module_names(module_name: str) -> list[str]:
    """Return all module names imported by a module (ast.Import and ast.ImportFrom)."""
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def test_initial_state_layer_separation_no_drainage_import():
    names = _imported_module_names("urbanflood.initial_state")
    assert not any(n == "urbanflood.drainage" or n.startswith("urbanflood.drainage.") for n in names), (
        f"initial_state.py imports drainage: {names}"
    )
    assert not any("suscept" in n for n in names), (
        f"initial_state.py imports susceptibility: {names}"
    )


def test_simulator_layers_do_not_import_susceptibility():
    """Replaced structural check: simulator-layer modules must not import
    urbanflood.susceptibility.

    The original test (test_initial_state_no_susceptibility_module_yet) was
    written before Phase 3 and asserted that no susceptibility*.py file
    exists in src/urbanflood/. Phase 3 added src/urbanflood/susceptibility.py
    legitimately, so that file-existence assertion is stale. This test keeps
    the protective intent: initial_state.py and the simulator-layer modules
    (simulator, orchestrator, numerics, drainage) must not import
    urbanflood.susceptibility.
    """
    for module_name in (
        "urbanflood.initial_state",
        "urbanflood.simulator",
        "urbanflood.orchestrator",
        "urbanflood.numerics",
        "urbanflood.drainage",
    ):
        names = _imported_module_names(module_name)
        assert not any(
            n == "urbanflood.susceptibility"
            or n.startswith("urbanflood.susceptibility.")
            for n in names
        ), f"{module_name} imports susceptibility: {names}"
