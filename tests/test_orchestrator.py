"""Phase 2 Milestone 2 tests for the multi-timestep orchestrator.

Tests cover:

1. Single-step equivalence with direct step_timestep() call.
2. H_peak initialization from initial depth.
3. H_peak update after a timestep.
4. Output shape and dtype conventions.
5. Invalid input handling.

Specification: docs/PHASE2_NUMERICAL_SPEC.md (approved).
Architecture: V7 (approved).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from urbanflood.config import (
    DrainageConfig,
    ExchangeConfig,
    LandCover,
    RainScenario,
    RunoffConfig,
    TideConfig,
    build_default_config,
)
from urbanflood.drainage import DrainageFields, generate_drainage
from urbanflood.environment import build_environment
from urbanflood.orchestrator import (
    run_simulation,
    SimulationResult,
    evaluate_tc10_sensitivity,
)
from urbanflood.simulator import StepResult, step_timestep
from urbanflood.config import SCENARIOS
from urbanflood.weather import interval_mean_rate

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

DX = 50.0  # m
CELL_AREA = 2500.0  # m^2
DT = 30.0  # s
C = 0.1  # m/s

# Land cover codes: 0=impervious, 1=open_ground, 2=vegetation, 3=water
CODE_IMPERVIOUS = 0
CODE_OPEN_GROUND = 1
CODE_VEGETATION = 2
CODE_WATER = 3


def _make_env(
    n_rows: int,
    n_cols: int,
    z: np.ndarray,
    is_ocean: np.ndarray,
    is_lake: np.ndarray,
    land_cover: np.ndarray,
    outfall_mask: np.ndarray,
    outfall_invert_m: np.ndarray,
) -> SimpleNamespace:
    """Build a lightweight environment-like object for timestep testing."""
    grid = SimpleNamespace(cell_size_m=DX, cell_area_m2=CELL_AREA)
    return SimpleNamespace(
        grid=grid,
        elevation_m=z,
        is_ocean=is_ocean,
        is_lake=is_lake,
        land_cover=land_cover,
        outfall_mask=outfall_mask,
        outfall_invert_m=outfall_invert_m,
    )


def _make_drainage(
    n_cells: int,
    capacity_mm_per_h: float | np.ndarray,
    efficiency: float | np.ndarray,
    blocked: bool | np.ndarray,
) -> DrainageFields:
    """Build drainage fields for testing."""
    if np.isscalar(capacity_mm_per_h):
        capacity = np.full(n_cells, capacity_mm_per_h, dtype=np.float64)
    else:
        capacity = np.asarray(capacity_mm_per_h, dtype=np.float64)
    if np.isscalar(efficiency):
        efficiency = np.full(n_cells, efficiency, dtype=np.float64)
    else:
        efficiency = np.asarray(efficiency, dtype=np.float64)
    if np.isscalar(blocked):
        blocked = np.full(n_cells, blocked, dtype=bool)
    else:
        blocked = np.asarray(blocked, dtype=bool)
    return DrainageFields(
        capacity_mm_per_h=capacity,
        efficiency=efficiency,
        blocked=blocked,
        blocked_fraction=0.20,
        master_seed=2027,
    )


def _make_runoff_config() -> RunoffConfig:
    return RunoffConfig()


def _make_exchange_config() -> ExchangeConfig:
    return ExchangeConfig()


def _make_rain_scenario() -> RainScenario:
    return SCENARIOS["R"].rainfall


def _make_tide_config(amplitude_m: float = 0.8) -> TideConfig:
    return TideConfig(amplitude_m=amplitude_m)


# ---------------------------------------------------------------------------
# Test 1: Single-step equivalence
# ---------------------------------------------------------------------------


def test_orchestrator_single_step_equivalence():
    """One orchestrator timestep matches a direct step_timestep() call."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.array([[False, False], [True, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.05, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    # Direct step_timestep call
    result_direct = step_timestep(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, k=0
    )

    # Orchestrator with n_steps=1
    result_orch = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=1
    )

    # Bitwise equality of the final depth array
    assert result_orch.h_final.shape == result_direct.h_new.shape
    assert result_orch.h_final.dtype == result_direct.h_new.dtype
    assert result_orch.h_final.tobytes() == result_direct.h_new.tobytes()

    # Bitwise equality of all single-step ledger values
    assert result_orch.V_precip_total == result_direct.V_precip_total
    assert result_orch.V_runoff_total == result_direct.V_runoff_total
    assert result_orch.V_loss_total == result_direct.V_loss_total
    assert result_orch.V_drain_total == result_direct.V_drain_total
    assert result_orch.SEA_IN_total == result_direct.SEA_IN_total
    assert result_orch.SEA_OUT_total == result_direct.SEA_OUT_total
    assert result_orch.internal_inflow_total == result_direct.internal_inflow_total
    assert result_orch.internal_outflow_total == result_direct.internal_outflow_total
    assert result_orch.S_initial == result_direct.S_before
    assert result_orch.S_final == result_direct.S_after

    # H_peak equals max(h_initial, h_new) after a single step
    assert result_orch.H_peak.shape == result_direct.h_new.shape
    assert result_orch.H_peak.dtype == result_direct.h_new.dtype
    H_peak_expected = np.maximum(h_initial, result_direct.h_new)
    assert result_orch.H_peak.tobytes() == H_peak_expected.tobytes()


# ---------------------------------------------------------------------------
# Test 2: H_peak initialization
# ---------------------------------------------------------------------------


def test_orchestrator_h_peak_initialization():
    """H_peak initialized from h_initial; element-wise max after one step."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.05, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    # Direct step_timestep for one step to compute expected H_peak
    step_result = step_timestep(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, k=0
    )
    H_peak_expected = np.maximum(h_initial, step_result.h_new)

    result = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=1
    )

    # H_peak initialized from h_initial, then max with h_new after step
    assert result.H_peak.shape == h_initial.shape
    assert result.H_peak.dtype == h_initial.dtype
    assert result.H_peak.tobytes() == H_peak_expected.tobytes()

    # H_peak must be >= h_initial element-wise (max only increases or keeps)
    assert np.all(result.H_peak >= h_initial)


# ---------------------------------------------------------------------------
# Test 3: H_peak update
# ---------------------------------------------------------------------------


def test_orchestrator_h_peak_update():
    """H_peak is the element-wise maximum of previous peak and new depth."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.array([[False, True], [True, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.001, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=2
    )

    # Direct step_timestep for each step to compute expected H_peak
    h_now = h_initial.copy()
    H_peak_expected = h_initial.copy()
    for k in range(2):
        step_result = step_timestep(
            h_now, env, drainage, rain, tide, exch, runoff, DT, k
        )
        H_peak_expected = np.maximum(H_peak_expected, step_result.h_new)
        h_now = step_result.h_new

    assert result.H_peak.shape == H_peak_expected.shape
    assert result.H_peak.dtype == H_peak_expected.dtype
    assert result.H_peak.tobytes() == H_peak_expected.tobytes()


# ---------------------------------------------------------------------------
# Test 4: Output shape and dtype
# ---------------------------------------------------------------------------


def test_orchestrator_result_shape_and_dtype():
    """Output arrays follow project conventions."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.05, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=1
    )

    # Shape conventions: (n_rows, n_cols) for depth arrays
    assert result.h_final.shape == (n_rows, n_cols)
    assert result.H_peak.shape == (n_rows, n_cols)
    # Dtype conventions: float64 for depth arrays
    assert result.h_final.dtype == np.float64
    assert result.H_peak.dtype == np.float64
    # Scalar fields are Python floats
    assert isinstance(result.S_initial, float)
    assert isinstance(result.S_final, float)
    assert isinstance(result.V_precip_total, float)
    assert isinstance(result.n_steps, int)
    assert isinstance(result.dt, float)
    assert result.n_steps == 1
    assert result.dt == DT


# ---------------------------------------------------------------------------
# Test 5: Invalid inputs
# ---------------------------------------------------------------------------


def test_orchestrator_invalid_dt():
    """dt must be positive."""
    n_rows, n_cols = 1, 1
    z = np.array([[1.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.1]])
    drainage = _make_drainage(1, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    with pytest.raises(ValueError, match="dt must be positive"):
        run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, dt=0.0, n_steps=1
        )
    with pytest.raises(ValueError, match="dt must be positive"):
        run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, dt=-1.0, n_steps=1
        )


def test_orchestrator_invalid_n_steps():
    """n_steps must be at least 1."""
    n_rows, n_cols = 1, 1
    z = np.array([[1.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.1]])
    drainage = _make_drainage(1, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=0
        )
    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=-1
        )


def test_orchestrator_invalid_h_initial_shape():
    """h_initial shape must match environment grid shape."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3x2, wrong shape
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    with pytest.raises(ValueError, match="does not match"):
        run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=1
        )


# ---------------------------------------------------------------------------
# Test 6: Six-hour Scenario R execution at dt=30s
# ---------------------------------------------------------------------------


def test_orchestrator_six_hour_execution():
    """Complete six-hour Scenario R simulation at dt=30s (720 steps)."""
    config = build_default_config()
    env = build_environment(master_seed=2027, config=config)
    n_cells = env.grid.n_rows * env.grid.n_cols

    drainage = generate_drainage(
        master_seed=2027,
        n_cells=n_cells,
        blocked_fraction=0.20,
        drainage_config=config.drainage,
    )
    rain = config.scenarios["R"].rainfall
    tide = config.scenarios["R"].tide
    exch = config.exchange
    runoff = config.runoff
    h_initial = np.zeros((env.grid.n_rows, env.grid.n_cols), dtype=np.float64)

    # Monkeypatch orchestrator's binding of step_timestep to count
    # every invocation and record its timestep index. Run the six-hour
    # simulation only once.
    captured_k = []

    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    def counting_step_timestep(h_now_arg, env_arg, drainage_arg, rain_arg,
                               tide_arg, exch_arg, runoff_arg, dt_arg, k_arg):
        captured_k.append(k_arg)
        return original_step_timestep(
            h_now_arg, env_arg, drainage_arg, rain_arg,
            tide_arg, exch_arg, runoff_arg, dt_arg, k_arg,
        )

    orch_mod.step_timestep = counting_step_timestep
    try:
        result = run_simulation(
            h_initial,
            env,
            drainage,
            rain,
            tide,
            exch,
            runoff,
            dt=30.0,
            n_steps=720,
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    # Exactly 720 invocations occurred
    assert len(captured_k) == 720
    # Recorded indices equal list(range(720))
    assert captured_k == list(range(720))

    # Timestep count and duration
    assert result.n_steps == 720
    assert result.dt == 30.0

    # Grid shape
    assert result.h_final.shape == (env.grid.n_rows, env.grid.n_cols)
    assert result.H_peak.shape == (env.grid.n_rows, env.grid.n_cols)

    # Finite and non-negative final depths (small negative values from
    # floating-point round-off in the donor limiter are acceptable;
    # anything below -1e-12 m indicates a real numerical problem)
    assert np.all(np.isfinite(result.h_final))
    assert np.all(result.h_final >= -1e-12)
    assert np.all(np.isfinite(result.H_peak))
    assert np.all(result.H_peak >= -1e-12)

    # Storage continuity: S_initial from step 0, S_final from last step
    assert result.S_initial >= 0.0
    assert result.S_final >= 0.0

    # TC7: Global mass balance across the six-hour simulation
    IN_total = result.V_precip_total + result.SEA_IN_total
    OUT_total = result.V_loss_total + result.V_drain_total + result.SEA_OUT_total
    dS = result.S_final - result.S_initial
    Residual_total = IN_total - OUT_total - dS
    relative_error = abs(Residual_total) / max(IN_total, OUT_total, 1.0)

    # Verify stored fields match recomputed values
    assert result.IN_total == IN_total
    assert result.OUT_total == OUT_total
    assert result.Residual_total == Residual_total
    assert result.relative_error == relative_error

    # TC7 acceptance: relative mass-balance error below 1e-9
    assert result.relative_error < 1e-9, (
        f"Relative mass-balance error {result.relative_error} exceeds 1e-9. "
        f"IN={result.IN_total}, OUT={result.OUT_total}, "
        f"S_initial={result.S_initial}, S_final={result.S_final}, "
        f"Residual={result.Residual_total}"
    )

    # TC10: Timestep sensitivity via the shared orchestration helper.
    tc10 = evaluate_tc10_sensitivity(
        h_initial, env, drainage, rain, tide, exch, runoff,
        result_30=result,
    )

    im = tc10['initial_metrics']


    # Print actual values for 30s vs 15s
    print(
        f"\nTC10 Timestep Sensitivity (dt=30s vs dt=15s):\n"
        f"  Grid: {env.grid.n_rows}x{env.grid.n_cols}={env.grid.n_rows * env.grid.n_cols} cells\n"
        f"  D_max = {im['D_max']:.6f} m (tolerance: 0.05 m)\n"
        f"  RMSE  = {im['RMSE']:.6f} m (tolerance: 0.01 m)\n"
        f"  S30   = {result.S_final:.6f} m3\n"
        f"  S15   = {tc10['result_15'].S_final:.6f} m3\n"
        f"  D_storage = {im['D_storage']:.6f} (tolerance: 0.01)"
    )

    if im['all_pass']:
        print(f"  TC10 PASS (30s vs 15s): All criteria met.")
    else:
        fm = tc10['fallback_metrics']

        print(
            f"\nTC10 Timestep Sensitivity (dt=15s vs dt=7.5s - fallback):\n"
            f"  D_max = {fm['D_max']:.6f} m (tolerance: 0.05 m)\n"
            f"  RMSE  = {fm['RMSE']:.6f} m (tolerance: 0.01 m)\n"
            f"  S15   = {tc10['result_15'].S_final:.6f} m3\n"
            f"  S7_5  = {tc10['result_7_5'].S_final:.6f} m3\n"
            f"  D_storage = {fm['D_storage']:.6f} (tolerance: 0.01)"
        )
        # Fallback is the final comparison: no further timestep reduction.
        assert fm['d_max_pass'], (
            f"TC10 fallback D_max={fm['D_max']:.6f} m exceeds 0.05 m tolerance"
        )
        assert fm['rmse_pass'], (
            f"TC10 fallback RMSE={fm['RMSE']:.6f} m exceeds 0.01 m tolerance"
        )
        assert fm['d_storage_pass'], (
            f"TC10 fallback D_storage={fm['D_storage']:.6f} exceeds 0.01 tolerance"
        )
        print(f"  TC10 PASS (15s vs 7.5s fallback): All criteria met.")


# ---------------------------------------------------------------------------
# Test 18: TC12 - Rainfall normalization
# ---------------------------------------------------------------------------


def test_orchestrator_tc12_rainfall_normalization():
    """TC12: Simulated precipitation depth equals analytical storm total.

    Runs complete six-hour Scenario R simulations at dt=30s (720 steps)
    and dt=15s (1440 steps), computes P_sim = V_precip_total /
    rainfall_receiving_area for each, and compares against the declared
    storm total with absolute error <= 1e-12 m.

    Ocean cells are excluded from the rainfall-receiving area because the
    simulator does not apply rainfall to them (simulator.py lines 209-213).
    """
    config = build_default_config()
    env = build_environment(master_seed=2027, config=config)
    n_cells = env.grid.n_rows * env.grid.n_cols

    drainage = generate_drainage(
        master_seed=2027,
        n_cells=n_cells,
        blocked_fraction=0.20,
        drainage_config=config.drainage,
    )
    rain = config.scenarios["R"].rainfall
    tide = config.scenarios["R"].tide
    exch = config.exchange
    runoff = config.runoff
    h_initial = np.zeros((env.grid.n_rows, env.grid.n_cols), dtype=np.float64)

    # Analytical declared storm total in metres (scenario R: 120 mm).
    analytical_total_m = rain.total_mm / 1000.0

    # Rainfall-receiving cells: all non-ocean cells.
    n_receiving = int(np.count_nonzero(~env.is_ocean))
    rainfall_receiving_area = n_receiving * env.grid.cell_area_m2

    for dt, n_steps in ((30.0, 720), (15.0, 1440)):
        result = run_simulation(
            h_initial,
            env,
            drainage,
            rain,
            tide,
            exch,
            runoff,
            dt=dt,
            n_steps=n_steps,
        )

        p_sim = result.V_precip_total / rainfall_receiving_area
        abs_error = abs(p_sim - analytical_total_m)

        print(
            f"\nTC12 Rainfall Normalization (dt={dt}s, n_steps={n_steps}):\n"
            f"  Grid: {env.grid.n_rows}x{env.grid.n_cols}={env.grid.n_rows * env.grid.n_cols} cells\n"
            f"  Rainfall-receiving cells: {n_receiving}\n"
            f"  Rainfall-receiving area: {rainfall_receiving_area:.3f} m2\n"
            f"  Total precipitation volume: {result.V_precip_total:.6f} m3\n"
            f"  Simulated precipitation depth: {p_sim:.15f} m\n"
            f"  Analytical storm total: {analytical_total_m:.15f} m\n"
            f"  Absolute depth error: {abs_error:.3e} m\n"
            f"  Tolerance: 1e-12 m"
        )

        assert abs_error <= 1e-12, (
            f"TC12 FAIL at dt={dt}s: |P_sim - P_analytical| = {abs_error:.3e} m "
            f"exceeds 1e-12 m tolerance. "
            f"P_sim={p_sim:.15f} m, P_analytical={analytical_total_m:.15f} m"
        )


# ---------------------------------------------------------------------------
# Test 7: Sequential state propagation
# ---------------------------------------------------------------------------


def test_orchestrator_state_propagation():
    """Each orchestrator step receives the previous step's h_new as h_now."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.05, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    captured_h_now = []
    captured_k = []

    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    def capturing_step_timestep(h_now_arg, env_arg, drainage_arg, rain_arg,
                                tide_arg, exch_arg, runoff_arg, dt_arg, k_arg):
        captured_h_now.append(np.array(h_now_arg, copy=True))
        captured_k.append(k_arg)
        return original_step_timestep(
            h_now_arg, env_arg, drainage_arg, rain_arg,
            tide_arg, exch_arg, runoff_arg, dt_arg, k_arg,
        )

    orch_mod.step_timestep = capturing_step_timestep
    try:
        result = run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=3
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    # Three calls, sequential k indices starting at 0
    assert len(captured_h_now) == 3
    assert captured_k == [0, 1, 2]

    # First call receives the supplied initial depth
    assert captured_h_now[0].tobytes() == h_initial.tobytes()

    # Verify by re-running step_timestep manually
    h_manual = h_initial.copy()
    for k in range(3):
        manual_result = step_timestep(
            h_manual, env, drainage, rain, tide, exch, runoff, DT, k
        )
        assert captured_h_now[k].tobytes() == h_manual.tobytes()
        h_manual = manual_result.h_new

    # Final h_final matches the manual chain
    assert result.h_final.tobytes() == h_manual.tobytes()


# ---------------------------------------------------------------------------
# Test 8: Storage continuity across steps
# ---------------------------------------------------------------------------


def test_orchestrator_storage_continuity():
    """S_before[k+1] == S_after[k] for consecutive steps."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.05, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    captured_S_before = []
    captured_S_after = []

    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    def capturing_step_timestep(h_now_arg, env_arg, drainage_arg, rain_arg,
                                tide_arg, exch_arg, runoff_arg, dt_arg, k_arg):
        result = original_step_timestep(
            h_now_arg, env_arg, drainage_arg, rain_arg,
            tide_arg, exch_arg, runoff_arg, dt_arg, k_arg,
        )
        captured_S_before.append(result.S_before)
        captured_S_after.append(result.S_after)
        return result

    orch_mod.step_timestep = capturing_step_timestep
    try:
        run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=3
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    assert len(captured_S_before) == 3
    assert len(captured_S_after) == 3

    # Storage continuity: S_before[k+1] == S_after[k]
    for k in range(2):
        assert captured_S_before[k + 1] == captured_S_after[k]


# ---------------------------------------------------------------------------
# Test 9: Non-negative finite depths at every timestep
# ---------------------------------------------------------------------------


def test_orchestrator_non_negative_depths():
    """All h_new values are finite and non-negative at every timestep."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.array([[False, True], [True, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.001, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    def capturing_step_timestep(h_now_arg, env_arg, drainage_arg, rain_arg,
                                tide_arg, exch_arg, runoff_arg, dt_arg, k_arg):
        result = original_step_timestep(
            h_now_arg, env_arg, drainage_arg, rain_arg,
            tide_arg, exch_arg, runoff_arg, dt_arg, k_arg,
        )
        # Check every returned h_new
        assert np.all(np.isfinite(result.h_new)), f"NaN/Inf at step {k_arg}"
        # Small negative values from floating-point round-off in the donor
        # limiter are acceptable; anything below -1e-12 indicates a real issue
        assert np.all(result.h_new[~env_arg.is_ocean] >= -1e-12), \
            f"Negative depth at step {k_arg}"
        assert result.h_new.shape == h_now_arg.shape
        return result

    orch_mod.step_timestep = capturing_step_timestep
    try:
        run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=5
        )
    finally:
        orch_mod.step_timestep = original_step_timestep


# ---------------------------------------------------------------------------
# Test 10: H_peak initial depth stays peak when never exceeded
# ---------------------------------------------------------------------------


def test_orchestrator_h_peak_initial_depth_stays_peak():
    """A cell whose initial depth exceeds all later depths keeps it as peak."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.array([[False, True], [True, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    # Cell (0,0) starts high at 0.9; subsequent depths stay below it.
    h_initial = np.array([[0.9, 0.1], [0.05, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, True)  # blocked: no drainage loss
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=3
    )

    # Cell (0,0) initial depth 0.9 must remain the peak regardless of later changes.
    assert result.H_peak[0, 0] == 0.9
    # Cell (0,0) peak must not be lower than initial.
    assert result.H_peak[0, 0] >= h_initial[0, 0]

    # All other cells: peak >= initial (max only increases or keeps).
    assert np.all(result.H_peak >= h_initial)


# ---------------------------------------------------------------------------
# Test 11: H_peak cells peak at different timesteps
# ---------------------------------------------------------------------------


def test_orchestrator_h_peak_peaks_at_different_timesteps():
    """Different cells reach their maximum at different timesteps."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.1, 0.0], [0.0, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    # Predetermined h_new per step: cell peaks at different times.
    h_new_step1 = np.array([[0.5, 0.0], [0.0, 0.0]])  # cell (0,0) peaks here
    h_new_step2 = np.array([[0.3, 0.7], [0.0, 0.0]])  # cell (0,1) peaks here
    h_new_step3 = np.array([[0.2, 0.5], [0.4, 0.0]])  # cell (1,0) peaks here

    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    captured_h_now = []

    def predetermined_step_timestep(h_now_arg, env_arg, drainage_arg, rain_arg,
                                    tide_arg, exch_arg, runoff_arg, dt_arg, k_arg):
        captured_h_now.append(np.array(h_now_arg, copy=True))
        if k_arg == 0:
            return StepResult(
                h_new=h_new_step1,
                V_precip_total=0.0,
                V_runoff_total=0.0,
                V_loss_total=0.0,
                V_drain_total=0.0,
                SEA_IN_total=0.0,
                SEA_OUT_total=0.0,
                internal_inflow_total=0.0,
                internal_outflow_total=0.0,
                internal_face_eval_count=0,
                S_before=0.0,
                S_after=0.0,
                residual=0.0,
            )
        if k_arg == 1:
            return StepResult(
                h_new=h_new_step2,
                V_precip_total=0.0,
                V_runoff_total=0.0,
                V_loss_total=0.0,
                V_drain_total=0.0,
                SEA_IN_total=0.0,
                SEA_OUT_total=0.0,
                internal_inflow_total=0.0,
                internal_outflow_total=0.0,
                internal_face_eval_count=0,
                S_before=0.0,
                S_after=0.0,
                residual=0.0,
            )
        # k_arg == 2
        return StepResult(
            h_new=h_new_step3,
            V_precip_total=0.0,
            V_runoff_total=0.0,
            V_loss_total=0.0,
            V_drain_total=0.0,
            SEA_IN_total=0.0,
            SEA_OUT_total=0.0,
            internal_inflow_total=0.0,
            internal_outflow_total=0.0,
            internal_face_eval_count=0,
            S_before=0.0,
            S_after=0.0,
            residual=0.0,
        )

    orch_mod.step_timestep = predetermined_step_timestep
    try:
        result = run_simulation(
            h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=3
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    # Expected H_peak = elementwise max over h_initial, h_new_step1, h_new_step2, h_new_step3
    H_peak_expected = np.maximum.reduce([
        h_initial,
        h_new_step1,
        h_new_step2,
        h_new_step3,
    ])

    assert result.H_peak.shape == H_peak_expected.shape
    assert result.H_peak.dtype == H_peak_expected.dtype
    assert result.H_peak.tobytes() == H_peak_expected.tobytes()

    # Each cell's peak occurs at a different timestep:
    # cell (0,0): max = 0.5 at step 1
    # cell (0,1): max = 0.7 at step 2
    # cell (1,0): max = 0.4 at step 3
    # cell (1,1): max = 0.0 everywhere
    assert result.H_peak[0, 0] == 0.5
    assert result.H_peak[0, 1] == 0.7
    assert result.H_peak[1, 0] == 0.4
    assert result.H_peak[1, 1] == 0.0


# ---------------------------------------------------------------------------
# Test 12: H_peak independent memory ownership
# ---------------------------------------------------------------------------


def test_orchestrator_h_peak_independent_memory():
    """H_peak does not share mutable memory with h_initial or h_final."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.05, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=2
    )

    # H_peak must not share memory with the caller's h_initial array.
    assert not np.shares_memory(result.H_peak, h_initial)
    # H_peak must not share memory with h_final.
    assert not np.shares_memory(result.H_peak, result.h_final)

    # Mutating the caller's original h_initial must not change H_peak.
    h_initial[0, 0] = 999.0
    assert result.H_peak[0, 0] != 999.0

    # Mutating h_final must not change H_peak.
    result.h_final[0, 0] = 999.0
    assert result.H_peak[0, 0] != 999.0

    # Restore h_final for cleanliness (not strictly required but avoids
    # surprising a reader who inspects the result afterwards).
    result.h_final[0, 0] = 0.0  # will be overwritten anyway; just reset


# ---------------------------------------------------------------------------
# Test 13: H_peak finiteness across multiple steps
# ---------------------------------------------------------------------------


def test_orchestrator_h_peak_finite():
    """H_peak values remain finite across multiple steps."""
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 0.5], [0.5, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.array([[False, True], [True, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.array([[0.2, 0.1], [0.001, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=4
    )

    assert np.all(np.isfinite(result.H_peak))
    assert np.all(result.H_peak >= -1e-12)


# ---------------------------------------------------------------------------
# Test 14: Global mass ledger - formula correctness (controlled inputs)
# ---------------------------------------------------------------------------


def test_orchestrator_global_mass_ledger_formula():
    """Verify IN_total, OUT_total, Residual_total, relative_error formula
    against hand-calculated values.

    IN      = V_precip_total + SEA_IN_total
    OUT     = V_loss_total + V_drain_total + SEA_OUT_total
    dS      = S_final - S_initial
    Residual = IN - OUT - dS
    relative_error = abs(Residual) / max(IN, OUT, 1.0)
    """
    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    step_result = StepResult(
        h_new=np.array([[0.1, 0.0], [0.0, 0.0]]),
        V_precip_total=100.0,
        V_runoff_total=90.0,
        V_loss_total=10.0,
        V_drain_total=20.0,
        SEA_IN_total=50.0,
        SEA_OUT_total=30.0,
        internal_inflow_total=500.0,  # internal - should NOT count
        internal_outflow_total=500.0,  # internal - should NOT count
        internal_face_eval_count=0,
        S_before=500.0,
        S_after=540.0,
        residual=0.0,
    )

    orch_mod.step_timestep = lambda *args, **kwargs: step_result
    try:
        result = run_simulation(
            h_initial=np.zeros((2, 2)),
            env=_make_env(
                2, 2,
                np.array([[1.0, 0.5], [0.5, 0.0]]),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2), dtype=bool),
                np.full((2, 2), CODE_IMPERVIOUS, dtype=np.uint8),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2)),
            ),
            drainage=_make_drainage(4, 50.0, 0.5, False),
            rain=_make_rain_scenario(),
            tide=_make_tide_config(),
            exchange_config=_make_exchange_config(),
            runoff_config=_make_runoff_config(),
            dt=DT,
            n_steps=1,
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    # IN = V_precip + SEA_IN = 100 + 50 = 150
    assert result.IN_total == 150.0
    # OUT = V_loss + V_drain + SEA_OUT = 10 + 20 + 30 = 60
    assert result.OUT_total == 60.0
    # dS = S_final - S_initial = 540 - 500 = 40
    assert result.S_final - result.S_initial == 40.0
    # Residual = IN - OUT - dS = 150 - 60 - 40 = 50
    assert result.Residual_total == 50.0
    # relative_error = |50| / max(150, 60, 1.0) = 50 / 150 = 1/3
    assert result.relative_error == 50.0 / max(150.0, 60.0, 1.0)


# ---------------------------------------------------------------------------
# Test 15: Global mass ledger - perfectly balanced case (zero residual)
# ---------------------------------------------------------------------------


def test_orchestrator_global_mass_ledger_balanced():
    """Perfectly balanced ledger: Residual_total == 0.0."""
    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    # IN = V_precip + SEA_IN = 100 + 50 = 150
    # OUT = V_loss + V_drain + SEA_OUT = 10 + 20 + 30 = 60
    # Need dS = IN - OUT = 90, so S_after = S_before + 90 = 590
    step_result = StepResult(
        h_new=np.array([[0.1, 0.0], [0.0, 0.0]]),
        V_precip_total=100.0,
        V_runoff_total=90.0,
        V_loss_total=10.0,
        V_drain_total=20.0,
        SEA_IN_total=50.0,
        SEA_OUT_total=30.0,
        internal_inflow_total=999.0,  # internal - must NOT count
        internal_outflow_total=999.0,  # internal - must NOT count
        internal_face_eval_count=0,
        S_before=500.0,
        S_after=590.0,
        residual=0.0,  # 150 - 60 - 90 = 0
    )

    orch_mod.step_timestep = lambda *args, **kwargs: step_result
    try:
        result = run_simulation(
            h_initial=np.zeros((2, 2)),
            env=_make_env(
                2, 2,
                np.array([[1.0, 0.5], [0.5, 0.0]]),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2), dtype=bool),
                np.full((2, 2), CODE_IMPERVIOUS, dtype=np.uint8),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2)),
            ),
            drainage=_make_drainage(4, 50.0, 0.5, False),
            rain=_make_rain_scenario(),
            tide=_make_tide_config(),
            exchange_config=_make_exchange_config(),
            runoff_config=_make_runoff_config(),
            dt=DT,
            n_steps=1,
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    assert result.IN_total == 150.0
    assert result.OUT_total == 60.0
    assert result.Residual_total == 0.0
    assert result.relative_error == 0.0


# ---------------------------------------------------------------------------
# Test 16: Global mass ledger - denominator floor at 1.0
# ---------------------------------------------------------------------------


def test_orchestrator_global_mass_ledger_denominator_floor():
    """When IN and OUT are both zero, denominator falls back to 1.0."""
    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    step_result = StepResult(
        h_new=np.array([[0.0, 0.0], [0.0, 0.0]]),
        V_precip_total=0.0,
        V_runoff_total=0.0,
        V_loss_total=0.0,
        V_drain_total=0.0,
        SEA_IN_total=0.0,
        SEA_OUT_total=0.0,
        internal_inflow_total=0.0,
        internal_outflow_total=0.0,
        internal_face_eval_count=0,
        S_before=0.0,
        S_after=0.0,
        residual=0.0,
    )

    orch_mod.step_timestep = lambda *args, **kwargs: step_result
    try:
        result = run_simulation(
            h_initial=np.zeros((2, 2)),
            env=_make_env(
                2, 2,
                np.array([[1.0, 0.5], [0.5, 0.0]]),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2), dtype=bool),
                np.full((2, 2), CODE_IMPERVIOUS, dtype=np.uint8),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2)),
            ),
            drainage=_make_drainage(4, 50.0, 0.5, False),
            rain=_make_rain_scenario(),
            tide=_make_tide_config(),
            exchange_config=_make_exchange_config(),
            runoff_config=_make_runoff_config(),
            dt=DT,
            n_steps=1,
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    assert result.IN_total == 0.0
    assert result.OUT_total == 0.0
    assert result.Residual_total == 0.0
    # relative_error = |0| / max(0, 0, 1.0) = 0 / 1 = 0
    assert result.relative_error == 0.0


# ---------------------------------------------------------------------------
# Test 17: Internal transfers excluded from global ledger
# ---------------------------------------------------------------------------


def test_orchestrator_internal_transfers_excluded_from_ledger():
    """Internal inflow/outflow must not appear in IN_total or OUT_total."""
    import urbanflood.orchestrator as orch_mod

    original_step_timestep = orch_mod.step_timestep

    step_result = StepResult(
        h_new=np.array([[0.1, 0.0], [0.0, 0.0]]),
        V_precip_total=100.0,
        V_runoff_total=90.0,
        V_loss_total=10.0,
        V_drain_total=0.0,
        SEA_IN_total=0.0,
        SEA_OUT_total=0.0,
        internal_inflow_total=1_000_000.0,  # huge internal - must NOT count
        internal_outflow_total=1_000_000.0,  # huge internal - must NOT count
        internal_face_eval_count=0,
        S_before=100.0,
        S_after=100.0,
        residual=0.0,
    )

    orch_mod.step_timestep = lambda *args, **kwargs: step_result
    try:
        result = run_simulation(
            h_initial=np.zeros((2, 2)),
            env=_make_env(
                2, 2,
                np.array([[1.0, 0.5], [0.5, 0.0]]),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2), dtype=bool),
                np.full((2, 2), CODE_IMPERVIOUS, dtype=np.uint8),
                np.array([[False, False], [False, True]]),
                np.zeros((2, 2)),
            ),
            drainage=_make_drainage(4, 50.0, 0.5, False),
            rain=_make_rain_scenario(),
            tide=_make_tide_config(),
            exchange_config=_make_exchange_config(),
            runoff_config=_make_runoff_config(),
            dt=DT,
            n_steps=1,
        )
    finally:
        orch_mod.step_timestep = original_step_timestep

    # IN = 100 + 0 = 100 (NOT 100 + 1_000_000)
    assert result.IN_total == 100.0
    # OUT = 10 + 0 + 0 = 10 (NOT 10 + 1_000_000)
    assert result.OUT_total == 10.0
    # Residual = 100 - 10 - 0 = 90
    assert result.Residual_total == 90.0


# ---------------------------------------------------------------------------
# Test 19: TC10 metrics - identical arrays produce zero D_max and RMSE
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_identical_arrays_zero_diff():
    """Identical H_peak arrays produce zero D_max and RMSE."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak = np.array([[0.5, 0.2], [0.1, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])

    D_max, RMSE, D_storage = compute_tc10_metrics(
        h_peak, h_peak.copy(), is_ocean, 100.0, 100.0
    )

    assert D_max == 0.0
    assert RMSE == 0.0
    assert D_storage == 0.0


# ---------------------------------------------------------------------------
# Test 20: TC10 metrics - known difference produces expected values
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_known_difference():
    """Known difference produces expected D_max and RMSE."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak_30 = np.array([[1.0, 0.5], [0.0, 0.0]])
    h_peak_15 = np.array([[0.8, 0.3], [0.1, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])

    D_max, RMSE, D_storage = compute_tc10_metrics(
        h_peak_30, h_peak_15, is_ocean, 50.0, 45.0
    )

    # Valid cells: (0,0), (0,1), (1,0); cell (1,1) is ocean -> excluded
    # differences: 1.0-0.8=0.2, 0.5-0.3=0.2, 0.0-0.1=-0.1
    # D_max = max(|0.2|, |0.2|, |-0.1|) = 0.2
    # RMSE = sqrt((0.04 + 0.04 + 0.01) / 3) = sqrt(0.09/3) = sqrt(0.03) ~= 0.1732
    assert D_max == 0.2
    assert abs(RMSE - np.sqrt(0.03)) < 1e-12
    assert D_storage == abs(50.0 - 45.0) / max(abs(50.0), abs(45.0))


# ---------------------------------------------------------------------------
# Test 21: TC10 metrics - ocean cells excluded
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_ocean_cells_excluded():
    """Ocean cells are excluded from D_max and RMSE computation."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak_30 = np.array([[100.0, 0.5], [0.0, 0.0]])
    h_peak_15 = np.array([[0.0, 0.5], [0.0, 0.0]])
    is_ocean = np.array([[True, False], [False, False]])

    D_max, RMSE, _ = compute_tc10_metrics(h_peak_30, h_peak_15, is_ocean, 10.0, 10.0)

    # Only cells (0,1), (1,0), (1,1) are valid:
    # differences: 0.5-0.5=0, 0.0-0.0=0, 0.0-0.0=0
    assert D_max == 0.0
    assert RMSE == 0.0


# ---------------------------------------------------------------------------
# Test 22: TC10 metrics - dry and inland cells included
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_dry_and_inland_cells_included():
    """Dry cells (zero depth) and inland cells are included."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak_30 = np.array([[0.0, 0.3], [0.0, 0.0]])
    h_peak_15 = np.array([[0.0, 0.1], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    D_max, RMSE, _ = compute_tc10_metrics(h_peak_30, h_peak_15, is_ocean, 10.0, 10.0)

    # All 4 cells valid. Differences: 0, 0.2, 0, 0
    assert abs(D_max - 0.2) < 1e-12
    assert abs(RMSE - np.sqrt(0.04 / 4)) < 1e-12  # sqrt(0.01) = 0.1


# ---------------------------------------------------------------------------
# Test 23: TC10 metrics - both-zero storage returns zero D_storage
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_both_zero_storage():
    """Both-zero final storage yields D_storage = 0.0."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak_30 = np.array([[0.1, 0.0], [0.0, 0.0]])
    h_peak_15 = np.array([[0.1, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    D_max, RMSE, D_storage = compute_tc10_metrics(
        h_peak_30, h_peak_15, is_ocean, 0.0, 0.0
    )

    assert D_max == 0.0
    assert RMSE == 0.0
    assert D_storage == 0.0


# ---------------------------------------------------------------------------
# Test 24: TC10 metrics - shape mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_shape_mismatch_raises():
    """Shape mismatch between h_peak arrays raises ValueError."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak_30 = np.array([[0.1, 0.0], [0.0, 0.0]])
    h_peak_15 = np.array([[0.1, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_tc10_metrics(h_peak_30, h_peak_15, is_ocean, 10.0, 10.0)


# ---------------------------------------------------------------------------
# Test 25: TC10 metrics - non-finite data raises ValueError
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_non_finite_raises():
    """Non-finite H_peak values raise ValueError."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak_30 = np.array([[np.nan, 0.0], [0.0, 0.0]])
    h_peak_15 = np.array([[0.1, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    with pytest.raises(ValueError, match="non-finite"):
        compute_tc10_metrics(h_peak_30, h_peak_15, is_ocean, 10.0, 10.0)


# ---------------------------------------------------------------------------
# Test 26: TC10 metrics - all ocean cells raises ValueError
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_all_ocean_raises():
    """All-ocean grid raises ValueError (no valid cells)."""
    from urbanflood.orchestrator import compute_tc10_metrics

    h_peak_30 = np.array([[0.1, 0.0], [0.0, 0.0]])
    h_peak_15 = np.array([[0.1, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[True, True], [True, True]])

    with pytest.raises(ValueError, match="No valid"):
        compute_tc10_metrics(h_peak_30, h_peak_15, is_ocean, 10.0, 10.0)


# ---------------------------------------------------------------------------
# Test 27: TC10 metrics - fallback not triggered when all pass
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_fallback_not_triggered_when_all_pass():
    """All criteria pass: no fallback simulation should be requested."""
    from urbanflood.orchestrator import evaluate_tc10_invariance

    h_peak_30 = np.array([[0.5, 0.2], [0.1, 0.0]])
    h_peak_15 = np.array([[0.501, 0.201], [0.101, 0.001]])
    is_ocean = np.array([[False, False], [False, False]])

    result = evaluate_tc10_invariance(h_peak_30, h_peak_15, is_ocean, 100.0, 100.0)

    assert result['all_pass'] is True


# ---------------------------------------------------------------------------
# Test 28: TC10 fallback - D_max failure triggers fallback
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_fallback_triggered_when_dmax_fails():
    """D_max failure triggers fallback to 7.5s simulation."""
    from urbanflood.orchestrator import evaluate_tc10_invariance

    h_peak_30 = np.array([[1.0, 0.0], [0.0, 0.0]])
    h_peak_15 = np.array([[0.0, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    result = evaluate_tc10_invariance(h_peak_30, h_peak_15, is_ocean, 100.0, 100.0)

    assert result['d_max_pass'] is False
    assert result['all_pass'] is False
    assert result['D_max'] == 1.0  # > 0.05, triggers fallback


# ---------------------------------------------------------------------------
# Test 29: TC10 fallback - RMSE failure triggers fallback
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_fallback_triggered_when_rmse_fails():
    """RMSE failure triggers fallback to 7.5s simulation."""
    from urbanflood.orchestrator import evaluate_tc10_invariance

    h_peak_30 = np.array([[1.0, 0.0], [0.0, 0.0]])
    h_peak_15 = np.array([[0.0, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    result = evaluate_tc10_invariance(h_peak_30, h_peak_15, is_ocean, 100.0, 100.0)

    assert result['rmse_pass'] is False
    assert result['all_pass'] is False


# ---------------------------------------------------------------------------
# Test 30: TC10 fallback - D_storage failure triggers fallback
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_fallback_triggered_when_dstorage_fails():
    """D_storage failure triggers fallback to 7.5s simulation."""
    from urbanflood.orchestrator import evaluate_tc10_invariance

    h_peak_30 = np.array([[0.1, 0.0], [0.0, 0.0]])
    h_peak_15 = np.array([[0.1, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    result = evaluate_tc10_invariance(h_peak_30, h_peak_15, is_ocean, 100.0, 1.0)

    assert result['d_storage_pass'] is False
    assert result['all_pass'] is False
    assert result['D_storage'] == abs(100.0 - 1.0) / max(abs(100.0), abs(1.0))


# ---------------------------------------------------------------------------
# Test 31: TC10 fallback - passing fallback is accepted
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_fallback_passing():
    """Passing fallback (15s vs 7.5s) is accepted."""
    from urbanflood.orchestrator import evaluate_tc10_invariance

    h_peak_15 = np.array([[0.5, 0.2], [0.1, 0.0]])
    h_peak_7_5 = np.array([[0.501, 0.201], [0.101, 0.001]])
    is_ocean = np.array([[False, False], [False, False]])

    result = evaluate_tc10_invariance(h_peak_15, h_peak_7_5, is_ocean, 100.0, 100.0)

    assert result['all_pass'] is True


# ---------------------------------------------------------------------------
# Test 32: TC10 fallback - failing fallback is reported as failing
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_fallback_failing():
    """Failing fallback (15s vs 7.5s) is reported as failing."""
    from urbanflood.orchestrator import evaluate_tc10_invariance

    h_peak_15 = np.array([[1.0, 0.0], [0.0, 0.0]])
    h_peak_7_5 = np.array([[0.0, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    result = evaluate_tc10_invariance(h_peak_15, h_peak_7_5, is_ocean, 100.0, 100.0)

    assert result['all_pass'] is False
    assert result['d_max_pass'] is False


# ---------------------------------------------------------------------------
# Test 33: TC10 fallback - no further timestep reduction
# ---------------------------------------------------------------------------


def test_orchestrator_tc10_no_further_timestep_reduction():
    """No further halving beyond 7.5s - fallback is the final comparison."""
    from urbanflood.orchestrator import evaluate_tc10_invariance

    h_peak_15 = np.array([[1.0, 0.0], [0.0, 0.0]])
    h_peak_7_5 = np.array([[0.0, 0.0], [0.0, 0.0]])
    is_ocean = np.array([[False, False], [False, False]])

    result = evaluate_tc10_invariance(h_peak_15, h_peak_7_5, is_ocean, 100.0, 100.0)

    # The fallback comparison itself can fail - that's the end of the line
    assert result['all_pass'] is False
    assert result['D_max'] == 1.0


# ---------------------------------------------------------------------------
# TC10 mocked fallback orchestration tests
# ---------------------------------------------------------------------------

def _make_mock_result(h_peak, s_final):
    """Build a minimal SimulationResult-like object for mocking run_simulation."""
    return SimpleNamespace(
        H_peak=np.asarray(h_peak, dtype=np.float64),
        S_final=float(s_final),
    )


def _env_two_by_two_all_land():
    """Minimal env-like object with a 2x2 all-land ocean mask."""
    return SimpleNamespace(
        is_ocean=np.array([[False, False], [False, False]], dtype=bool),
    )


def _passing_arrays():
    """H_peak pairs whose differences are within TC10 tolerances."""
    h_peak_30 = np.array([[0.5, 0.2], [0.1, 0.0]])
    h_peak_15 = np.array([[0.501, 0.201], [0.101, 0.001]])
    return h_peak_30, h_peak_15


def test_orchestrator_tc10_fallback_initial_pass_no_fallback(monkeypatch):
    """Case A: Initial 30s vs 15s comparison passes → exactly two calls, no 7.5s."""
    from urbanflood.orchestrator import evaluate_tc10_sensitivity

    calls = []

    def fake_run_simulation(h_initial, env, drainage, rain, tide,
                            exchange_config, runoff_config, dt, n_steps):
        calls.append({'dt': float(dt), 'n_steps': int(n_steps),
                      'h_initial': h_initial, 'env': env,
                      'drainage': drainage, 'rain': rain,
                      'tide': tide, 'exchange_config': exchange_config,
                      'runoff_config': runoff_config})
        if dt == 30.0:
            return _make_mock_result(_passing_arrays()[0], 100.0)
        if dt == 15.0:
            return _make_mock_result(_passing_arrays()[1], 100.0)
        raise AssertionError(
            f"Unexpected run_simulation call: dt={dt}, n_steps={n_steps}")

    monkeypatch.setattr('urbanflood.orchestrator.run_simulation',
                        fake_run_simulation)

    result = evaluate_tc10_sensitivity(
        np.zeros((2, 2)), _env_two_by_two_all_land(),
        None, None, None, None, None,
    )

    assert result['fallback_attempted'] is False
    assert result['fallback_passed'] is None
    assert result['fallback_metrics'] is None
    assert result['initial_metrics']['all_pass'] is True
    assert len(calls) == 2
    assert calls[0]['dt'] == 30.0 and calls[0]['n_steps'] == 720
    assert calls[1]['dt'] == 15.0 and calls[1]['n_steps'] == 1440


def test_orchestrator_tc10_fallback_dmax_failure_runs_fallback(monkeypatch):
    """Case B: D_max failure → exactly three calls, third is dt=7.5, n_steps=2880."""
    from urbanflood.orchestrator import evaluate_tc10_sensitivity

    calls = []

    def fake_run_simulation(h_initial, env, drainage, rain, tide,
                            exchange_config, runoff_config, dt, n_steps):
        calls.append({'dt': float(dt), 'n_steps': int(n_steps)})
        if dt == 30.0:
            return _make_mock_result([[1.0, 0.0], [0.0, 0.0]], 100.0)
        if dt == 15.0:
            return _make_mock_result(_passing_arrays()[0], 100.0)
        if dt == 7.5:
            return _make_mock_result(_passing_arrays()[1], 100.0)
        raise AssertionError(
            f"Unexpected run_simulation call: dt={dt}, n_steps={n_steps}")

    monkeypatch.setattr('urbanflood.orchestrator.run_simulation',
                        fake_run_simulation)

    result = evaluate_tc10_sensitivity(
        np.zeros((2, 2)), _env_two_by_two_all_land(),
        None, None, None, None, None,
    )

    assert result['fallback_attempted'] is True
    assert result['fallback_passed'] is True
    assert result['initial_metrics']['all_pass'] is False
    assert result['initial_metrics']['d_max_pass'] is False
    assert len(calls) == 3
    assert calls[0] == {'dt': 30.0, 'n_steps': 720}
    assert calls[1] == {'dt': 15.0, 'n_steps': 1440}
    assert calls[2] == {'dt': 7.5, 'n_steps': 2880}
    # Fallback compares 15s result against 7.5s result (both have passing arrays).
    assert result['fallback_metrics']['all_pass'] is True


def test_orchestrator_tc10_fallback_rmse_failure_runs_fallback(monkeypatch):
    """Case C: RMSE failure → exactly three calls, third is dt=7.5, n_steps=2880."""
    from urbanflood.orchestrator import evaluate_tc10_sensitivity

    calls = []

    def fake_run_simulation(h_initial, env, drainage, rain, tide,
                            exchange_config, runoff_config, dt, n_steps):
        calls.append({'dt': float(dt), 'n_steps': int(n_steps)})
        if dt == 30.0:
            # Uniform 0.04 offset: D_max=0.04 (pass), RMSE=0.04 (fail).
            return _make_mock_result([[0.04, 0.04], [0.04, 0.04]], 100.0)
        if dt == 15.0:
            return _make_mock_result([[0.0, 0.0], [0.0, 0.0]], 100.0)
        if dt == 7.5:
            # 15s vs 7.5s are close (fallback passes).
            return _make_mock_result([[0.001, 0.001], [0.001, 0.001]], 100.0)
        raise AssertionError(
            f"Unexpected run_simulation call: dt={dt}, n_steps={n_steps}")

    monkeypatch.setattr('urbanflood.orchestrator.run_simulation',
                        fake_run_simulation)

    result = evaluate_tc10_sensitivity(
        np.zeros((2, 2)), _env_two_by_two_all_land(),
        None, None, None, None, None,
    )

    assert result['fallback_attempted'] is True
    assert result['fallback_passed'] is True
    assert result['initial_metrics']['all_pass'] is False
    assert result['initial_metrics']['d_max_pass'] is True
    assert result['initial_metrics']['rmse_pass'] is False
    assert len(calls) == 3
    assert calls[2] == {'dt': 7.5, 'n_steps': 2880}
    assert result['fallback_metrics']['all_pass'] is True


def test_orchestrator_tc10_fallback_dstorage_failure_runs_fallback(monkeypatch):
    """Case D: D_storage failure → exactly three calls, third is dt=7.5, n_steps=2880."""
    from urbanflood.orchestrator import evaluate_tc10_sensitivity

    calls = []

    def fake_run_simulation(h_initial, env, drainage, rain, tide,
                            exchange_config, runoff_config, dt, n_steps):
        calls.append({'dt': float(dt), 'n_steps': int(n_steps)})
        if dt == 30.0:
            return _make_mock_result([[0.5, 0.2], [0.1, 0.0]], 100.0)
        if dt == 15.0:
            return _make_mock_result([[0.5, 0.2], [0.1, 0.0]], 1.0)
        if dt == 7.5:
            # 15s vs 7.5s: H_peak close, S_final both 1.0 → fallback passes.
            return _make_mock_result([[0.501, 0.201], [0.101, 0.001]], 1.0)
        raise AssertionError(
            f"Unexpected run_simulation call: dt={dt}, n_steps={n_steps}")

    monkeypatch.setattr('urbanflood.orchestrator.run_simulation',
                        fake_run_simulation)

    result = evaluate_tc10_sensitivity(
        np.zeros((2, 2)), _env_two_by_two_all_land(),
        None, None, None, None, None,
    )

    assert result['fallback_attempted'] is True
    assert result['fallback_passed'] is True
    assert result['initial_metrics']['all_pass'] is False
    assert result['initial_metrics']['d_max_pass'] is True
    assert result['initial_metrics']['rmse_pass'] is True
    assert result['initial_metrics']['d_storage_pass'] is False
    assert len(calls) == 3
    assert calls[2] == {'dt': 7.5, 'n_steps': 2880}
    assert result['fallback_metrics']['all_pass'] is True


def test_orchestrator_tc10_fallback_failure_reported_no_further_reduction(monkeypatch):
    """Case E: Initial fails and fallback also fails → attempted once, reports failure."""
    from urbanflood.orchestrator import evaluate_tc10_sensitivity

    calls = []

    def fake_run_simulation(h_initial, env, drainage, rain, tide,
                            exchange_config, runoff_config, dt, n_steps):
        calls.append({'dt': float(dt), 'n_steps': int(n_steps)})
        if dt == 30.0:
            return _make_mock_result([[1.0, 0.0], [0.0, 0.0]], 100.0)
        if dt == 15.0:
            return _make_mock_result([[0.0, 0.0], [0.0, 0.0]], 100.0)
        if dt == 7.5:
            # Fallback also fails: 15s vs 7.5s differ by 1.0 m.
            return _make_mock_result([[1.0, 0.0], [0.0, 0.0]], 100.0)
        raise AssertionError(
            f"Unexpected run_simulation call: dt={dt}, n_steps={n_steps}")

    monkeypatch.setattr('urbanflood.orchestrator.run_simulation',
                        fake_run_simulation)

    result = evaluate_tc10_sensitivity(
        np.zeros((2, 2)), _env_two_by_two_all_land(),
        None, None, None, None, None,
    )

    assert result['fallback_attempted'] is True
    assert result['fallback_passed'] is False
    assert result['initial_metrics']['all_pass'] is False
    assert result['fallback_metrics']['all_pass'] is False
    assert result['fallback_metrics']['d_max_pass'] is False
    assert len(calls) == 3
    assert calls[2] == {'dt': 7.5, 'n_steps': 2880}
    # No fourth call: no further timestep reduction.
    assert len(calls) == 3


def test_orchestrator_tc10_fallback_argument_consistency(monkeypatch):
    """Case F: All simulation calls receive identical inputs; full-grid arrays
    and the same ocean mask are supplied to both metric evaluations."""
    from urbanflood.orchestrator import (
        evaluate_tc10_sensitivity,
        evaluate_tc10_invariance as real_evaluate_tc10_invariance,
    )

    h_initial = np.zeros((2, 2), dtype=np.float64)
    env = _env_two_by_two_all_land()
    drainage = SimpleNamespace()
    rain = SimpleNamespace()
    tide = SimpleNamespace()
    exchange_config = SimpleNamespace()
    runoff_config = SimpleNamespace()

    sim_calls = []
    metric_calls = []

    def fake_run_simulation(h_initial_arg, env_arg, drainage_arg, rain_arg,
                            tide_arg, exchange_config_arg, runoff_config_arg,
                            dt, n_steps):
        sim_calls.append({
            'dt': float(dt), 'n_steps': int(n_steps),
            'h_initial': h_initial_arg, 'env': env_arg,
            'drainage': drainage_arg, 'rain': rain_arg,
            'tide': tide_arg,
            'exchange_config': exchange_config_arg,
            'runoff_config': runoff_config_arg,
        })
        if dt == 30.0:
            return _make_mock_result([[1.0, 0.0], [0.0, 0.0]], 100.0)
        if dt == 15.0:
            return _make_mock_result([[0.0, 0.0], [0.0, 0.0]], 100.0)
        if dt == 7.5:
            # 15s vs 7.5s close → fallback passes.
            return _make_mock_result([[0.001, 0.001], [0.001, 0.001]], 100.0)
        raise AssertionError(
            f"Unexpected run_simulation call: dt={dt}, n_steps={n_steps}")

    def recording_metric_spy(h_peak_30, h_peak_15, is_ocean, S30, S15):
        metric_calls.append({
            'h_peak_30': h_peak_30,
            'h_peak_15': h_peak_15,
            'is_ocean': is_ocean,
            'S30': S30,
            'S15': S15,
        })
        return real_evaluate_tc10_invariance(
            h_peak_30, h_peak_15, is_ocean, S30, S15)

    monkeypatch.setattr('urbanflood.orchestrator.run_simulation',
                        fake_run_simulation)
    monkeypatch.setattr('urbanflood.orchestrator.evaluate_tc10_invariance',
                        recording_metric_spy)

    result = evaluate_tc10_sensitivity(
        h_initial, env, drainage, rain, tide,
        exchange_config, runoff_config,
    )

    # Exactly three simulation calls: 30s, 15s, 7.5s fallback.
    assert len(sim_calls) == 3
    assert sim_calls[0] == {'dt': 30.0, 'n_steps': 720,
                            'h_initial': h_initial, 'env': env,
                            'drainage': drainage, 'rain': rain,
                            'tide': tide,
                            'exchange_config': exchange_config,
                            'runoff_config': runoff_config}
    assert sim_calls[1] == {'dt': 15.0, 'n_steps': 1440,
                            'h_initial': h_initial, 'env': env,
                            'drainage': drainage, 'rain': rain,
                            'tide': tide,
                            'exchange_config': exchange_config,
                            'runoff_config': runoff_config}
    assert sim_calls[2] == {'dt': 7.5, 'n_steps': 2880,
                            'h_initial': h_initial, 'env': env,
                            'drainage': drainage, 'rain': rain,
                            'tide': tide,
                            'exchange_config': exchange_config,
                            'runoff_config': runoff_config}

    # Exactly two metric-evaluation calls: initial (30s vs 15s) then fallback (15s vs 7.5s).
    assert len(metric_calls) == 2

    # Initial metric evaluation receives full-grid H_peak arrays from result_30
    # and result_15, plus the exact env.is_ocean mask (object identity).
    assert metric_calls[0]['h_peak_30'] is result['result_30'].H_peak
    assert metric_calls[0]['h_peak_15'] is result['result_15'].H_peak
    assert metric_calls[0]['is_ocean'] is env.is_ocean
    assert np.array_equal(metric_calls[0]['h_peak_30'],
                          [[1.0, 0.0], [0.0, 0.0]])
    assert np.array_equal(metric_calls[0]['h_peak_15'],
                          [[0.0, 0.0], [0.0, 0.0]])
    assert metric_calls[0]['S30'] == 100.0
    assert metric_calls[0]['S15'] == 100.0

    # Fallback metric evaluation receives full-grid H_peak arrays from
    # result_15 and result_7_5, plus the same env.is_ocean mask (object identity).
    assert metric_calls[1]['h_peak_30'] is result['result_15'].H_peak
    assert metric_calls[1]['h_peak_15'] is result['result_7_5'].H_peak
    assert metric_calls[1]['is_ocean'] is env.is_ocean
    assert np.array_equal(metric_calls[1]['h_peak_30'],
                          [[0.0, 0.0], [0.0, 0.0]])
    assert np.array_equal(metric_calls[1]['h_peak_15'],
                          [[0.001, 0.001], [0.001, 0.001]])
    assert metric_calls[1]['S30'] == 100.0
    assert metric_calls[1]['S15'] == 100.0

    # Order: initial comparison first, fallback comparison second.
    assert metric_calls[0]['S30'] == 100.0 and metric_calls[0]['S15'] == 100.0
    assert metric_calls[1]['S30'] == 100.0 and metric_calls[1]['S15'] == 100.0

    # Existing assertions: fallback metrics.
    assert result['initial_metrics']['d_max_pass'] is False
    assert result['fallback_metrics']['d_max_pass'] is True


# ---------------------------------------------------------------------------
# Test 19: TC11 - Face-count proportionality at dt=30s vs dt=15s
# ---------------------------------------------------------------------------


def test_orchestrator_tc11_face_count_proportionality():
    """TC11: Internal face evaluation count scales linearly with timestep count.

    Runs complete six-hour Scenario R simulations at dt=30s (720 steps)
    and dt=15s (1440 steps). Verifies that total face evaluations at
    dt=15s is exactly twice the count at dt=30s, using actual measured
    instrumentation from the simulator (not hardcoded arithmetic).
    """
    config = build_default_config()
    env = build_environment(master_seed=2027, config=config)
    n_cells = env.grid.n_rows * env.grid.n_cols

    drainage = generate_drainage(
        master_seed=2027,
        n_cells=n_cells,
        blocked_fraction=0.20,
        drainage_config=config.drainage,
    )
    rain = config.scenarios["R"].rainfall
    tide = config.scenarios["R"].tide
    exch = config.exchange
    runoff = config.runoff
    h_initial = np.zeros((env.grid.n_rows, env.grid.n_cols), dtype=np.float64)

    result_30 = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff,
        dt=30.0, n_steps=720,
    )

    result_15 = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff,
        dt=15.0, n_steps=1440,
    )

    # Timestep counts match configuration
    assert result_30.n_steps == 720
    assert result_15.n_steps == 1440

    # 15s total evaluations must be exactly twice the 30s total
    assert result_15.internal_face_eval_total == 2 * result_30.internal_face_eval_total

    # Each run's total equals its per-step count times its actual timestep count
    # (per-step count is stable because environment is static)
    assert result_30.internal_face_eval_total % result_30.n_steps == 0
    assert result_15.internal_face_eval_total % result_15.n_steps == 0

    # Per-step face count is consistent across timesteps (static environment)
    per_step_30 = result_30.internal_face_eval_total // result_30.n_steps
    per_step_15 = result_15.internal_face_eval_total // result_15.n_steps
    assert per_step_30 == per_step_15
    assert per_step_30 > 0

    print(
        f"\nTC11 Face-Count Proportionality:\n"
        f"  Grid: {env.grid.n_rows}x{env.grid.n_cols}={env.grid.n_rows * env.grid.n_cols} cells\n"
        f"  dt=30s: {result_30.internal_face_eval_total} total evaluations "
        f"({per_step_30} per step x {result_30.n_steps} steps)\n"
        f"  dt=15s: {result_15.internal_face_eval_total} total evaluations "
        f"({per_step_15} per step x {result_15.n_steps} steps)\n"
        f"  Ratio (15s/30s): {result_15.internal_face_eval_total / result_30.internal_face_eval_total:.6f}\n"
        f"  Expected ratio: 2.0"
    )


# ---------------------------------------------------------------------------
# Test 20: TC11c2 - Aggregate equals per-step count for single-step run
# ---------------------------------------------------------------------------


def test_orchestrator_tc11c_aggregate_equals_per_step_single_step():
    """TC11c2: Single-step run_simulation aggregate equals per-step count.

    Runs run_simulation() with n_steps=1 and verifies that
    internal_face_eval_total equals the per-step count from step_timestep().
    """
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 1.0], [1.0, 1.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_initial = np.full((n_rows, n_cols), 0.1)
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    # Per-step count from direct step_timestep() call
    step_result = step_timestep(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, k=0
    )
    per_step_count = step_result.internal_face_eval_count

    # Aggregate from run_simulation() with n_steps=1
    result = run_simulation(
        h_initial, env, drainage, rain, tide, exch, runoff, DT, n_steps=1
    )

    # Aggregate equals per-step count for single step
    assert result.internal_face_eval_total == per_step_count
    assert result.internal_face_eval_total == 4
    assert result.n_steps == 1


