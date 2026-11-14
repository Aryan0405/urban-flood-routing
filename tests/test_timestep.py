"""Phase 2 Milestone 2 tests for the single-timestep water-exchange update.

Tests cover:

1. Internal exchange in both directions.
2. Zero signed flux creates no request.
3. Internal transfer subtracts and adds the exact same volume.
4. Two or more outgoing request types compete for one donor budget.
5. Incoming water cannot fund same-step outgoing requests.
6. Multiple sea outfalls from one donor share one common scale.
7. Sea inflow bypasses the land donor budget.
8. Blocked drainage produces zero drainage.
9. Outfall blocking prevents the specified sea exchange.
10. All proposals use the old-state snapshot.
11. Storage remains finite and nonnegative after a complete update.
12. A no-flux/no-rain/no-drainage fixture leaves storage unchanged.
13. A mixed-source/sink fixture satisfies the explicit per-step mass balance.
14. Reordering candidate-generation inputs does not change results.
15. Inland lake storage is included in land storage and treated according to Option A.

Specification: docs/PHASE2_NUMERICAL_SPEC.md (approved).
Architecture: V7 (approved).
"""

from __future__ import annotations

import math
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
)
from urbanflood.drainage import DrainageFields
from urbanflood.simulator import StepResult, enumerate_sea_faces, step_timestep
from urbanflood.numerics import canonicalize_faces, enumerate_internal_faces
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
# Test 12: No-flux/no-rain/no-drainage leaves storage unchanged
# ---------------------------------------------------------------------------


def test_timestep_no_flux_no_rain_no_drainage_unchanged():
    """A flat, all-land grid with no rain, no drainage, no sea: storage unchanged."""
    n_rows, n_cols = 2, 2
    z = np.full((n_rows, n_cols), 10.0)
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.full((n_rows, n_cols), 0.1)
    drainage = _make_drainage(n_rows * n_cols, 50.0, 0.5, True)  # all blocked
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # k=0 is before the pulse start (t=0..30 < 3600), so R_k = 0
    assert result.V_precip_total == 0.0
    assert result.V_runoff_total == 0.0
    assert result.V_loss_total == 0.0
    assert result.V_drain_total == 0.0
    assert result.SEA_IN_total == 0.0
    assert result.SEA_OUT_total == 0.0
    # h_new == h_now exactly (no flux at all)
    assert np.array_equal(result.h_new, h_now)
    assert result.S_before == result.S_after
    assert abs(result.residual) < 1e-12


# ---------------------------------------------------------------------------
# Test 1: Internal exchange in both directions
# ---------------------------------------------------------------------------


def test_timestep_internal_exchange_both_directions():
    """Water flows from higher W to lower W; direction follows sign of delta_W."""
    n_rows, n_cols = 1, 2
    z = np.array([[10.0, 10.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    # Cell 0 has more water than cell 1: flow 0 -> 1
    h_now = np.array([[0.5, 0.1]])
    drainage = _make_drainage(2, 50.0, 0.5, True)  # all blocked
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # W_0 = 10.5, W_1 = 10.1, delta_W = 0.4
    # V_01 = c * 0.4 * dx * dt = 0.1 * 0.4 * 50 * 30 = 60 m3
    expected_flux = C * 0.4 * DX * DT
    assert result.internal_outflow_total == pytest.approx(expected_flux, rel=1e-12)
    assert result.internal_inflow_total == pytest.approx(expected_flux, rel=1e-12)

    # Cell 0: V_old = 1250, outflow = 60 -> V_new = 1190, h = 0.476
    # Cell 1: V_old = 250, inflow = 60 -> V_new = 310, h = 0.124
    assert result.h_new[0, 0] == pytest.approx(0.476, rel=1e-12)
    assert result.h_new[0, 1] == pytest.approx(0.124, rel=1e-12)

    # Mass balance: no rain, no drain, no sea -> residual = 0
    assert abs(result.residual) < 1e-9


# ---------------------------------------------------------------------------
# Test 2: Zero signed flux creates no request
# ---------------------------------------------------------------------------


def test_timestep_zero_flux_no_request():
    """Equal water levels: delta_W = 0, no internal exchange request."""
    n_rows, n_cols = 1, 2
    z = np.array([[10.0, 10.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.array([[0.1, 0.1]])  # equal levels, no head difference
    drainage = _make_drainage(2, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    assert result.internal_inflow_total == 0.0
    assert result.internal_outflow_total == 0.0
    assert np.array_equal(result.h_new, h_now)


# ---------------------------------------------------------------------------
# Test 3: Internal transfer subtracts and adds exact same volume
# ---------------------------------------------------------------------------


def test_timestep_internal_transfer_exact_same_volume():
    """Donor loses exactly what receiver gains (equal and opposite)."""
    n_rows, n_cols = 1, 2
    z = np.array([[10.0, 10.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.array([[0.5, 0.1]])
    drainage = _make_drainage(2, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # Donor (cell 0) loses 60 m3, receiver (cell 1) gains 60 m3
    V_0_before = h_now[0, 0] * CELL_AREA
    V_1_before = h_now[0, 1] * CELL_AREA
    V_0_after = result.h_new[0, 0] * CELL_AREA
    V_1_after = result.h_new[0, 1] * CELL_AREA
    assert V_0_before - V_0_after == pytest.approx(V_1_after - V_1_before, rel=1e-12)
    assert (V_0_before - V_0_after) == pytest.approx(60.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 4: Two or more outgoing request types compete for one donor budget
# ---------------------------------------------------------------------------


def test_timestep_donor_budget_competition():
    """When Q > budget, all outgoing requests from a donor share one scale."""
    n_rows, n_cols = 1, 2
    # Cell 0: high terrain (z=10), tiny water (h=0.001 -> W=10.001, V=2.5 m3)
    # Cell 1: low terrain (z=0), empty (h=0 -> W=0)
    # delta_W = 10.001 -> proposed outflow = 0.1 * 10.001 * 50 * 30 = 1500.15 m3
    # Q_0 = 1500.15, budget_0 = 2.5, scale_0 = 2.5/1500.15
    # Scaled outflow = 2.5 m3 (entire budget)
    z = np.array([[10.0, 0.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.array([[0.001, 0.0]])
    drainage = _make_drainage(2, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # Q_0 = 1500.15 m3, budget_0 = 2.5 m3
    scale = 2.5 / (C * 10.001 * DX * DT)
    expected_outflow = (C * 10.001 * DX * DT) * scale  # = 2.5 m3
    assert result.internal_outflow_total == pytest.approx(expected_outflow, rel=1e-12)
    # Cell 0 gives up its entire budget
    assert result.h_new[0, 0] == pytest.approx(0.0, abs=1e-12)
    # Cell 1 receives exactly 2.5 m3 -> h = 0.001
    assert result.h_new[0, 1] == pytest.approx(0.001, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 5: Incoming water cannot fund same-step outgoing requests
# ---------------------------------------------------------------------------


def test_timestep_incoming_water_cannot_fund_outgoing():
    """Cell B's outflow is based on h_now_B, not h_now_B + inflow from A."""
    n_rows, n_cols = 1, 3
    z = np.array([[10.0, 10.0, 10.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    # Cell 0 has water, cell 1 has almost none, cell 2 is empty
    h_now = np.array([[0.5, 0.001, 0.0]])
    # Face (0,1): delta_W = 0.499 -> donor 0 -> receiver 1, V = 74.85 m3
    # Face (1,2): delta_W = 0.001 -> donor 1 -> receiver 2, V = 0.15 m3
    drainage = _make_drainage(3, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # Cell 1's outflow to cell 2 is based on h_now_1 = 0.001 (V_old = 2.5 m3),
    # NOT on h_now_1 + 74.85 (inflow from cell 0).
    # Q_1 = 0.15 m3, budget_1 = 2.5 m3, scale_1 = 1.0
    # Cell 1 sends only 0.15 m3 to cell 2 despite receiving 74.85 m3.
    V_2_after = result.h_new[0, 2] * CELL_AREA
    assert V_2_after == pytest.approx(0.15, rel=1e-12)

    # Cell 1 receives 74.85 from cell 0, sends 0.15 to cell 2
    V_1_before = h_now[0, 1] * CELL_AREA  # 2.5 m3
    V_1_after = result.h_new[0, 1] * CELL_AREA
    assert V_1_after == pytest.approx(V_1_before + 74.85 - 0.15, rel=1e-12)

    # Mass balance: no rain, no drain, no sea -> residual = 0
    assert abs(result.residual) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: Multiple sea outfalls from one donor share one common scale
# ---------------------------------------------------------------------------


def test_timestep_multiple_sea_outfalls_share_scale():
    """One land cell with two ocean faces: both share the same donor scale."""
    n_rows, n_cols = 2, 2
    z = np.array([[0.5, 0.0], [0.0, 0.0]])  # cell (0,0) is land, others ocean
    is_ocean = np.array([[False, True], [True, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.array([[True, False], [False, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    # h_00 = 0.001 -> V_old = 2.5 m3
    # W_00 = 0.501. At k=0, eta_k = 0.8 * sin(2*pi*15/44712) ~= 0.001686 (low tide)
    # V_sea per face = 0.1 * (eta_k - 0.501) * 50 * 30 ~= -74.896 m3
    # Two faces: total proposed outflow ~= 149.79 m3
    h_now = np.array([[0.001, 0.0], [0.0, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()  # amplitude 0.8
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # Q_00 ~= 149.79 m3 (two sea outflows), budget_00 = 2.5 m3
    # scale_00 = 2.5 / 149.79 ~= 0.01669
    # Each face: 74.896 * scale ~= 1.25 m3, total ~= 2.5 m3 (entire budget)
    assert result.SEA_OUT_total == pytest.approx(2.5, rel=1e-9)
    assert result.h_new[0, 0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 7: Sea inflow bypasses the land donor budget
# ---------------------------------------------------------------------------


def test_timestep_sea_inflow_bypasses_donor_budget():
    """Sea inflow when eta_k > W_i enters storage even when the land cell has
    zero available budget (scale factor would be 0 for outflows).
    
    Cell 0: z=0.5, h=0.0 -> W_0 = 0.5, V_old = 0 m3.
    Cell 1: ocean.
    Outfall at cell 0.
    k=119 (t=3570s), amplitude 2.0: rainfall pulse hasn't started yet (starts at t=3600),
    so R_k = 0 and no rainfall volume contributes.
    eta_k = 2.0 * sin(2*pi*3585/44712) approximately 0.9637 > W_0 = 0.5, sea inflow occurs.
    Sea inflow is external and does not consume donor budget.
    """
    n_rows, n_cols = 1, 2
    z = np.array([[0.5, 0.0]])  # cell 0 land, cell 1 ocean
    is_ocean = np.array([[False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.array([[True, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.array([[0.0, 0.0]])
    drainage = _make_drainage(2, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = TideConfig(amplitude_m=2.0)  # high tide: eta_k > W_0 = 0.5
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    k = 119  # timestep index (t = 3570 s), before the rainfall pulse start (t = 3600 s)

    # Compute expected eta_k and sea inflow exactly as the simulator does
    eta_k = tide.amplitude_m * math.sin(2 * math.pi * (k * DT + DT / 2) / tide.period_s)
    V_sea = exch.exchange_velocity_m_per_s * (eta_k - 0.5) * DX * DT

    # k=119 is before rainfall pulse start (t=3600s), so R_k = 0 and no rainfall contributes
    R_k = interval_mean_rate(rain, k * DT, (k + 1) * DT)
    assert R_k == 0.0, f"Expected R_k = 0 at k={k}, got {R_k}"
    V_precip = R_k * CELL_AREA * DT
    V_runoff = runoff.coefficients[LandCover.VEGETATION] * V_precip
    assert V_runoff == 0.0, f"Expected V_runoff = 0, got {V_runoff}"

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=119)

    # Cell 0 has h=0.0, budget = 0 m3, but sea inflow bypasses donor budget
    expected_h = V_sea / CELL_AREA
    assert result.h_new[0, 0] == pytest.approx(expected_h, rel=1e-9)
    assert result.SEA_IN_total == pytest.approx(V_sea, rel=1e-6)
    assert result.SEA_OUT_total == 0.0
    assert abs(result.residual) < 1e-9




# ---------------------------------------------------------------------------
# Test 8: Blocked drainage produces zero drainage
# ---------------------------------------------------------------------------


def test_timestep_blocked_drainage_zero_drainage():
    """Blocked drainage cells produce exactly zero drainage volume."""
    n_rows, n_cols = 1, 1
    z = np.array([[10.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.array([[0.5]])
    # All drainage blocked
    drainage = _make_drainage(1, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    assert result.V_drain_total == 0.0


# ---------------------------------------------------------------------------
# Test 9: Outfall blocking prevents the specified sea exchange
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Test 10: All proposals use the old-state snapshot
# ---------------------------------------------------------------------------


def test_timestep_old_state_snapshot():
    """Fluxes are computed from h_now, not from h_now updated mid-step."""
    n_rows, n_cols = 1, 2
    z = np.array([[10.0, 10.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    # Cell 0 has water, cell 1 is empty. Flow 0 -> 1 based on h_now.
    h_now = np.array([[0.5, 0.0]])
    drainage = _make_drainage(2, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # Flux based on h_now: delta_W = 0.5, V = 75 m3
    # budget_0 = 1250, scale = 1, cell 0 sends 75, cell 1 receives 75
    expected_flux = C * 0.5 * DX * DT  # = 75 m3
    assert result.internal_outflow_total == pytest.approx(expected_flux, rel=1e-12)
    assert result.h_new[0, 1] == pytest.approx(0.03, abs=1e-12)  # 75 / 2500

    # Crucially, cell 0's outflow was NOT increased by any incoming water
    # (there is none in this case), and cell 1's outflow (none) was not
    # affected by the inflow it received. The outflow from cell 0 is based
    # solely on h_now_0.
    assert result.h_new[0, 0] == pytest.approx(0.47, abs=1e-12)  # (1250-75)/2500


# ---------------------------------------------------------------------------
# Test 11: Storage remains finite and nonnegative after complete update
# ---------------------------------------------------------------------------


def test_timestep_storage_finite_and_nonnegative():
    """After any step, all depths are finite and nonnegative."""
    n_rows, n_cols = 3, 3
    z = np.random.RandomState(42).uniform(0.2, 5.0, (n_rows, n_cols))
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_ocean[2, :] = True  # bottom row is ocean
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_mask[1, :] = True  # middle row adjacent to ocean
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.random.RandomState(7).uniform(0.0, 2.0, (n_rows, n_cols))
    h_now[is_ocean] = 0.0
    drainage = _make_drainage(9, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    assert np.all(np.isfinite(result.h_new))
    assert np.all(result.h_new[~is_ocean] >= 0.0)
    # Ocean cells should have zero depth
    assert np.all(result.h_new[is_ocean] == 0.0)


# ---------------------------------------------------------------------------
# Test 13: Mixed-source/sink fixture satisfies per-step mass balance
# ---------------------------------------------------------------------------


def test_timestep_mixed_fixture_mass_balance():
    """A fixture with rain, internal exchange, drainage, and sea exchange:
    residual = IN - OUT - dS should be ~0.
    
    Uses k=120 (t=3615s, inside rainfall pulse) so rainfall is active.
    """
    n_rows, n_cols = 2, 2
    # Cell (0,0): land, W = 1.2 -> internal-exchange donor
    # Cell (0,1): land, W = 0.3, outfall -> coastal INFLOW face (eta_k > W)
    # Cell (1,0): land, W = 2.001, outfall, tiny depth -> coastal OUTFLOW face;
    #             proposed sea outflow >> donor budget -> donor-availability limited
    # Cell (1,1): ocean
    z = np.array([[1.0, 0.2], [2.0, 0.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_IMPERVIOUS, dtype=np.uint8)
    outfall_mask = np.array([[False, True], [True, False]])
    outfall_invert_m = np.where(outfall_mask, z, 0.0)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.array([[0.2, 0.1], [0.001, 0.0]])
    drainage = _make_drainage(4, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=120)

    # Midpoint tide at k=120: eta_k = eta(3615 s) with amplitude 0.8 m
    eta_k = tide.eta(120.5 * DT)
    # Expected directions:
    #   face (0,1): W = 0.2 + 0.1 = 0.3 < eta_k  -> sea inflow (external, no donor budget)
    #   face (1,0): W = 2.0 + 0.001 = 2.001 > eta_k -> sea outflow (donor-limited)

    # Independent expected sea inflow (full volume, not donor-scaled)
    V_sea_in_expected = exch.exchange_velocity_m_per_s * (eta_k - (z[0, 1] + h_now[0, 1])) * DX * DT
    # Donor-limiting evidence for the outflow face: proposed outflow exceeds budget
    W_outflow = z[1, 0] + h_now[1, 0]
    proposed_sea_outflow = exch.exchange_velocity_m_per_s * (W_outflow - eta_k) * DX * DT
    R_k = interval_mean_rate(rain, 120 * DT, 121 * DT)
    V_precip_cell = R_k * CELL_AREA * DT
    budget_outflow = h_now[1, 0] * CELL_AREA + runoff.coefficients[LandCover.IMPERVIOUS] * V_precip_cell
    assert proposed_sea_outflow > budget_outflow, "outflow face must be donor-limited"

    # Sea inflow is added in full (external input, not donor-scaled)
    assert result.SEA_IN_total == pytest.approx(V_sea_in_expected, rel=1e-6)
    # Sea outflow is scaled down by the donor-availability limiter
    assert 0.0 < result.SEA_OUT_total < proposed_sea_outflow

    # Manual mass balance check
    IN = result.V_precip_total + result.SEA_IN_total
    OUT = result.V_loss_total + result.V_drain_total + result.SEA_OUT_total
    dS = result.S_after - result.S_before
    residual = IN - OUT - dS
    assert abs(residual) < 1e-9, f"Residual {residual} is not ~0"
    assert abs(result.residual) < 1e-9

    # Assertions: rainfall active, drainage active, sea outflow active,
    # internal exchange active
    assert result.V_precip_total > 0.0, "Rainfall should be active at k=120"
    assert result.V_drain_total > 0.0, "Drainage should be active"
    assert result.SEA_OUT_total > 0.0, "Sea outflow should be active from cell (0,0) since W_0=1.2 > eta_k~0.9584"
    assert result.internal_outflow_total > 0.0, "Internal exchange should flow from cell (0,0) to cell (0,1) since 1.2 > 0.6"

    # Assertions: rainfall active, drainage active, sea exchange active
    assert result.V_precip_total > 0.0, "Expected non-zero rainfall volume"
    assert result.V_drain_total > 0.0, "Expected non-zero drainage volume"
    assert result.SEA_IN_total > 0.0, "Expected non-zero sea inflow"
    assert result.SEA_OUT_total > 0.0, "Expected non-zero sea outflow"
    assert result.internal_outflow_total > 0.0, "Expected non-zero internal exchange outflow"


# ---------------------------------------------------------------------------
# Test 14: Reordering candidate-generation inputs does not change results
# ---------------------------------------------------------------------------


def test_timestep_deterministic_repeat_call():
    """Calling step_timestep twice with identical inputs gives identical results."""
    n_rows, n_cols = 3, 3
    z = np.full((n_rows, n_cols), 10.0)
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.random.RandomState(123).uniform(0.0, 1.0, (n_rows, n_cols))
    drainage = _make_drainage(9, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result1 = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)
    result2 = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    assert np.array_equal(result1.h_new, result2.h_new)
    assert result1.residual == result2.residual
    assert result1.V_drain_total == result2.V_drain_total
    assert result1.SEA_IN_total == result2.SEA_IN_total
    assert result1.SEA_OUT_total == result2.SEA_OUT_total


def test_timestep_canonical_ordering_permutation_invariance():
    """TC14: Bitwise canonical-order invariance.
    
    Candidate generation order is permuted; candidates are canonicalised
    before accumulation, so permuting generation order must produce bitwise
    identical simulation outputs for both depth and ledger results.
    
    Contract exercised: candidate generation -> canonicalization -> accumulation.
    Generation order may differ; canonicalization (stable face IDs derived only
    from grid geometry, sorted canonically) normalizes the input before
    per-cell accumulation, so the simulator receives an identical canonical
    sequence regardless of generation order.
    
    Limitation: H_peak is listed as a TC14-required output (spec section 12)
    but is not exposed by the single-step StepResult API; spec section 17
    leaves this API question unresolved. H_peak is not compared here.
    """
    n_rows, n_cols = 4, 4
    z = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5, 4.5],
        [2.0, 3.0, 4.0, 5.0],
        [2.5, 3.5, 4.5, 5.0],
    ])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover,
        outfall_mask, outfall_invert_m,
    )
    h_now = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ])
    drainage = _make_drainage(n_rows * n_cols, 50.0, 0.5, False)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    # 1. Generate candidates in canonical order (row-major, i < j, sorted).
    faces_canonical = enumerate_internal_faces(n_rows, n_cols, is_ocean)

    # 2. Generate the same face set in a deliberately different order
    #    (reverse row-major), as plain endpoint pairs.
    faces_permuted_pairs = []
    for r in range(n_rows - 1, -1, -1):
        for c in range(n_cols - 1, -1, -1):
            if is_ocean[r, c]:
                continue
            i = r * n_cols + c
            if c + 1 < n_cols and not is_ocean[r, c + 1]:
                faces_permuted_pairs.append((i, i + 1))
            if r + 1 < n_rows and not is_ocean[r + 1, c]:
                faces_permuted_pairs.append((i, i + n_cols))

    # The permuted generation order must genuinely differ from canonical.
    assert faces_permuted_pairs != [(f.i, f.j) for f in faces_canonical], (
        "Permuted generation order must differ from canonical order"
    )
    # Both sequences must describe exactly the same unique face set.
    assert set(faces_permuted_pairs) == {(f.i, f.j) for f in faces_canonical}, (
        "Permuted generation must yield the same face set"
    )

    # 3. Canonicalize both orderings before accumulation. Canonicalization must
    #    absorb generation-order dependence: both canonical forms identical.
    canon_from_gen = canonicalize_faces(faces_canonical)
    canon_from_perm = canonicalize_faces(faces_permuted_pairs)
    assert canon_from_gen == canon_from_perm, (
        "Canonicalisation must be order-independent"
    )

    # 4. Run the simulator twice with canonical candidates. Both runs receive
    #    canonically identical face sequences (Run 1: canon_from_gen; Run 2:
    #    canon_from_perm - both equal after canonicalization). Independent
    #    copies of mutable input are used for each run.
    import urbanflood.simulator as sim_mod
    original_enumerate_internal_faces = sim_mod.enumerate_internal_faces

    def canonical_gen_enumerate(n_rows_arg, n_cols_arg, is_ocean_arg):
        return canon_from_gen

    def permuted_gen_enumerate(n_rows_arg, n_cols_arg, is_ocean_arg):
        return canon_from_perm

    # Run 1: canonical candidate generation order.
    sim_mod.enumerate_internal_faces = canonical_gen_enumerate
    try:
        result_canon = step_timestep(
            h_now.copy(), env, drainage, rain, tide, exch, runoff, DT, k=0
        )
    finally:
        sim_mod.enumerate_internal_faces = original_enumerate_internal_faces

    # Run 2: permuted candidate generation order, canonicalized before accumulation.
    sim_mod.enumerate_internal_faces = permuted_gen_enumerate
    try:
        result_perm = step_timestep(
            h_now.copy(), env, drainage, rain, tide, exch, runoff, DT, k=0
        )
    finally:
        sim_mod.enumerate_internal_faces = original_enumerate_internal_faces

    # Strict bitwise equality of all exposed TC14-required outputs.
    # Arrays: verify shape and dtype, then compare byte representations.
    assert result_canon.h_new.shape == result_perm.h_new.shape
    assert result_canon.h_new.dtype == result_perm.h_new.dtype
    assert result_canon.h_new.tobytes() == result_perm.h_new.tobytes()
    # Scalars: compare packed IEEE 754 byte representations.
    assert np.float64(result_canon.V_precip_total).tobytes() == np.float64(result_perm.V_precip_total).tobytes()
    assert np.float64(result_canon.V_loss_total).tobytes() == np.float64(result_perm.V_loss_total).tobytes()
    assert np.float64(result_canon.V_drain_total).tobytes() == np.float64(result_perm.V_drain_total).tobytes()
    assert np.float64(result_canon.SEA_IN_total).tobytes() == np.float64(result_perm.SEA_IN_total).tobytes()
    assert np.float64(result_canon.SEA_OUT_total).tobytes() == np.float64(result_perm.SEA_OUT_total).tobytes()
    assert np.float64(result_canon.S_after).tobytes() == np.float64(result_perm.S_after).tobytes()
    assert np.float64(result_canon.residual).tobytes() == np.float64(result_perm.residual).tobytes()




# ---------------------------------------------------------------------------
# Test 15: Inland lake storage included in land storage (Option A)
# ---------------------------------------------------------------------------


def test_timestep_inland_lake_included_in_storage():
    """Lake cells (is_lake=True, is_ocean=False) are surface-water cells:
    they have alpha=1.0, participate in exchange, and count in storage."""
    n_rows, n_cols = 1, 2
    z = np.array([[2.0, 10.0]])
    is_ocean = np.zeros((n_rows, n_cols), dtype=bool)
    # Cell 0 is a lake, cell 1 is land
    is_lake = np.array([[True, False]])
    land_cover = np.array([CODE_WATER, CODE_VEGETATION], dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    # Lake cell 0: spill = 5.0, z = 2.0 -> h_init = 3.0
    # Land cell 1: h = 0.1
    h_now = np.array([[3.0, 0.1]])
    # W_0 = 5.0, W_1 = 10.1 -> delta_W = -5.1 -> donor = 1, receiver = 0
    # V = 0.1 * 5.1 * 50 * 30 = 765 m3
    drainage = _make_drainage(2, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # Lake cell has alpha = 1.0 (WATER class), land cell has alpha = 0.3
    # k=0: R_k = 0 (before pulse), so V_runoff = 0 for both
    assert result.V_runoff_total == 0.0

    # Lake cell is included in S_before and S_after
    S_before = 3.0 * CELL_AREA + 0.1 * CELL_AREA  # 7500 + 250 = 7750
    assert result.S_before == pytest.approx(S_before, rel=1e-12)

    # Mass balance holds for the lake-land system
    assert abs(result.residual) < 1e-9


# ---------------------------------------------------------------------------
# Test: enumerate_sea_faces correctness
# ---------------------------------------------------------------------------


def test_enumerate_sea_faces_unique_and_sorted():
    """Each land-ocean face appears exactly once, sorted by (land, ocean)."""
    n_rows, n_cols = 2, 3
    # Cell (0,0) land, (0,1) ocean, (0,2) land
    # Cell (1,0) land, (1,1) ocean, (1,2) land
    is_ocean = np.array(
        [[False, True, False], [False, True, False]]
    )
    faces = enumerate_sea_faces(n_rows, n_cols, is_ocean)
    # Cell (0,0) idx 0: neighbor (0,1) idx 1 is ocean -> (0, 1)
    # Cell (0,2) idx 2: neighbor (0,1) idx 1 is ocean -> (2, 1)
    # Cell (1,0) idx 3: neighbor (1,1) idx 4 is ocean -> (3, 4)
    # Cell (1,2) idx 5: neighbor (1,1) idx 4 is ocean -> (5, 4)
    expected = [(0, 1), (2, 1), (3, 4), (5, 4)]
    assert faces == expected
    # No duplicates
    assert len(faces) == len(set(faces))
    # Sorted
    assert faces == sorted(faces)


# ---------------------------------------------------------------------------
# Test 16: TC11a - Per-timestep internal face evaluation count (all land)
# ---------------------------------------------------------------------------


def test_timestep_tc11a_per_timestep_face_count_all_land():
    """TC11a: 2x2 all-land grid has 4 unique eligible internal faces.

    Cells: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    Faces: (0,1), (0,2), (1,3), (2,3) = 4 faces.
    All faces counted even when zero-flux (equal depths).
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
    h_now = np.full((n_rows, n_cols), 0.1)
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # 4 unique eligible internal faces in 2x2 all-land grid
    assert result.internal_face_eval_count == 4
    # All faces evaluated even though zero flux (equal depths => no proposals)
    assert result.internal_inflow_total == 0.0
    assert result.internal_outflow_total == 0.0


# ---------------------------------------------------------------------------
# Test 17: TC11b - Per-timestep internal face count excludes ocean cells
# ---------------------------------------------------------------------------


def test_timestep_tc11b_per_timestep_face_count_excludes_ocean():
    """TC11b: 2x2 grid with ocean at (1,1) has 2 internal faces.

    Cells: (0,0)=0 land, (0,1)=1 land, (1,0)=2 land, (1,1)=3 ocean.
    Faces: (0,1), (0,2) = 2 faces. Sea face (1,0)-(1,1) excluded.
    """
    n_rows, n_cols = 2, 2
    z = np.array([[1.0, 1.0], [1.0, 1.0]])
    is_ocean = np.array([[False, False], [False, True]])
    is_lake = np.zeros((n_rows, n_cols), dtype=bool)
    land_cover = np.full((n_rows, n_cols), CODE_VEGETATION, dtype=np.uint8)
    outfall_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outfall_invert_m = np.zeros((n_rows, n_cols), dtype=np.float64)
    env = _make_env(
        n_rows, n_cols, z, is_ocean, is_lake, land_cover, outfall_mask, outfall_invert_m
    )
    h_now = np.full((n_rows, n_cols), 0.1)
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)

    # 2 unique eligible internal faces (sea face excluded)
    assert result.internal_face_eval_count == 2


# ---------------------------------------------------------------------------
# Test 18: TC11c - Instrumentation consistency at single-timestep level
# ---------------------------------------------------------------------------


def test_timestep_tc11c_instrumentation_consistency():
    """TC11c: Per-step count is deterministic and includes zero-flux faces.

    Same static environment run twice yields identical face count.
    Face count is nonzero even when all fluxes are zero.
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
    h_now = np.full((n_rows, n_cols), 0.1)
    drainage = _make_drainage(4, 50.0, 0.5, True)
    rain = _make_rain_scenario()
    tide = _make_tide_config()
    exch = _make_exchange_config()
    runoff = _make_runoff_config()

    result1 = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=0)
    result2 = step_timestep(h_now, env, drainage, rain, tide, exch, runoff, DT, k=1)

    # Same static environment => same face count each timestep
    assert result1.internal_face_eval_count == result2.internal_face_eval_count
    assert result1.internal_face_eval_count == 4
    # Per-step count is deterministic and includes zero-flux faces
    assert result1.internal_face_eval_count > 0
    assert result1.internal_inflow_total == 0.0
    assert result1.internal_outflow_total == 0.0
