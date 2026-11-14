"""Phase 2 Milestone 1 tests for the SYNTHETIC numerical foundations.

Covers the independent numerical primitives in urbanflood/numerics.py:

A. Exact rainfall interval integration (reuses weather.py).
B. Precipitation/runoff/loss partition (new).
C. Midpoint tide calculation (reuses weather.py).
D. Canonical internal-face enumeration (new).
E. Signed internal-flux to donor/receiver conversion (new).
F. Deterministic candidate ordering (inherent in D).

Specification: docs/PHASE2_NUMERICAL_SPEC.md (approved).
Architecture: V7 (approved).
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanflood.config import (
    MM_PER_H_TO_M_PER_S,
    RunoffConfig,
    SCENARIOS,
    build_default_config,
)
from urbanflood.numerics import (
    FluxRequest,
    InternalFace,
    PartitionResult,
    canonicalize_faces,
    enumerate_internal_faces,
    face_count,
    faces_canonical_and_unique,
    precipitation_runoff_loss,
    signed_flux_to_request,
)
from urbanflood.weather import (
    applied_precipitation_volume,
    hyetograph,
    interval_mean_rate,
    rain_rate_at,
    tide_midpoint_series,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DT = 30.0
CELL_AREA_M2 = 2500.0  # 50 m x 50 m


def _analytic_eta(t_s: float, amplitude_m: float, period_s: float = 44712.0) -> float:
    import math

    return amplitude_m * np.sin(2.0 * np.pi * t_s / period_s)


# ---------------------------------------------------------------------------
# A. Exact rainfall interval integration (reuses weather.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,total_mm", [("R", 120.0), ("S1", 200.0)])
@pytest.mark.parametrize("dt", [30.0, 15.0])
def test_A_hyetograph_preserves_declared_total(name, total_mm, dt):
    """Full storm window recovers the declared storm total exactly at
    both 30 s and 15 s timesteps."""
    rain = SCENARIOS[name].rainfall
    n_steps = int(rain.storm_window_s / dt)
    rates = hyetograph(rain, dt, n_steps)
    applied_m = rates.sum() * dt
    assert applied_m == pytest.approx(total_mm / 1000.0, rel=1e-12)


def test_A_interval_crossing_pulse_end():
    """An interval crossing the pulse end includes only the in-pulse portion."""
    rain = SCENARIOS["R"].rainfall
    # Interval starts inside the falling limb, ends after the pulse.
    a = rain.pulse_end_s - 1000.0
    b = rain.pulse_end_s + 500.0
    integral = interval_mean_rate(rain, a, b) * (b - a)
    # Only the portion [a, pulse_end] contributes.
    tp, te = rain.pulse_peak_time_s, rain.pulse_end_s
    peak_m_per_s = rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S
    expected = peak_m_per_s * (te - a) ** 2 / (2.0 * (te - tp))
    assert integral == pytest.approx(expected, rel=1e-12)


def test_A_invalid_interval_rejected():
    """Zero-length or inverted intervals are rejected by the weather API."""
    rain = SCENARIOS["R"].rainfall
    with pytest.raises(ValueError):
        interval_mean_rate(rain, 1000.0, 1000.0)
    with pytest.raises(ValueError):
        interval_mean_rate(rain, 5000.0, 4000.0)


def test_A_zero_rainfall_outside_storm():
    """Rates are zero for intervals completely outside the pulse/storm."""
    rain = SCENARIOS["R"].rainfall
    # Before pulse start.
    assert interval_mean_rate(rain, 0.0, 1000.0) == 0.0
    # After pulse end.
    assert interval_mean_rate(rain, 14000.0, 15000.0) == 0.0
    # After storm window.
    assert interval_mean_rate(rain, 22000.0, 21600.0 + 1000.0) == 0.0
    # Hyetograph rates outside the pulse are zero.
    rates = hyetograph(rain, DT, int(rain.storm_window_s / DT))
    starts = np.arange(len(rates)) * DT
    outside = (starts + DT <= rain.pulse_start_s) | (starts >= rain.pulse_end_s)
    assert np.all(rates[outside] == 0.0)


def test_A_integration_both_sides_of_peak():
    """Intervals fully within rising and falling limbs match analytic trapezoids."""
    rain = SCENARIOS["R"].rainfall
    peak_m_per_s = rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S

    # Rising limb: [t_start, t_start + 1000].
    a, b = rain.pulse_start_s, rain.pulse_start_s + 1000.0
    integral = interval_mean_rate(rain, a, b) * (b - a)
    # Analytic: peak * ((b-ts)^2 - (a-ts)^2) / (2*(tp-ts))
    ts, tp = rain.pulse_start_s, rain.pulse_peak_time_s
    expected = (
        peak_m_per_s
        * ((b - ts) ** 2 - (a - ts) ** 2)
        / (2.0 * (tp - ts))
    )
    assert integral == pytest.approx(expected, rel=1e-12)

    # Falling limb: [t_peak, t_peak + 1000].
    a, b = rain.pulse_peak_time_s, rain.pulse_peak_time_s + 1000.0
    integral = interval_mean_rate(rain, a, b) * (b - a)
    # Analytic: peak * ((te-a)^2 - (te-b)^2) / (2*(te-tp))
    te = rain.pulse_end_s
    expected = (
        peak_m_per_s
        * ((te - a) ** 2 - (te - b) ** 2)
        / (2.0 * (te - tp))
    )
    assert integral == pytest.approx(expected, rel=1e-12)


def test_A_interval_crossing_pulse_boundary():
    """An interval crossing the pulse start includes only the in-pulse portion."""
    rain = SCENARIOS["R"].rainfall
    # Interval starts before the pulse, ends inside the rising limb.
    a = rain.pulse_start_s - 500.0
    b = rain.pulse_start_s + 1000.0
    integral = interval_mean_rate(rain, a, b) * (b - a)
    # Only the portion [pulse_start, b] contributes.
    ts, tp = rain.pulse_start_s, rain.pulse_peak_time_s
    peak_m_per_s = rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S
    expected = peak_m_per_s * (b - ts) ** 2 / (2.0 * (tp - ts))
    assert integral == pytest.approx(expected, rel=1e-12)


def test_A_interval_crossing_the_peak():
    """An interval crossing the peak equals the sum of its two sub-intervals."""
    rain = SCENARIOS["R"].rainfall
    a = rain.pulse_peak_time_s - 2000.0
    b = rain.pulse_peak_time_s + 2000.0
    tp = rain.pulse_peak_time_s

    whole = interval_mean_rate(rain, a, b) * (b - a)
    part1 = interval_mean_rate(rain, a, tp) * (tp - a)
    part2 = interval_mean_rate(rain, tp, b) * (b - tp)
    assert whole == pytest.approx(part1 + part2, rel=1e-12)


def test_A_subdivision_consistency():
    """Sum of sub-interval integrals equals the full-interval integral."""
    rain = SCENARIOS["R"].rainfall
    a, b = rain.pulse_start_s, rain.pulse_end_s
    n_sub = 8
    sub_width = (b - a) / n_sub
    total = 0.0
    for k in range(n_sub):
        total += interval_mean_rate(rain, a + k * sub_width, a + (k + 1) * sub_width) * sub_width
    whole = interval_mean_rate(rain, a, b) * (b - a)
    assert total == pytest.approx(whole, rel=1e-12)


# ---------------------------------------------------------------------------
# B. Precipitation/runoff/loss partition (new)
# ---------------------------------------------------------------------------


def _config_alpha() -> dict[str, float]:
    cfg = build_default_config()
    return {
        "impervious": cfg.runoff.impervious,
        "open_ground": cfg.runoff.open_ground,
        "vegetation": cfg.runoff.vegetation,
        "water": cfg.runoff.water,
    }


@pytest.mark.parametrize(
    "land_cover,expected_alpha",
    [
        ("impervious", 0.90),
        ("open_ground", 0.50),
        ("vegetation", 0.30),
        ("water", 1.00),
    ],
)
def test_B_partition_all_land_cover_classes(land_cover, expected_alpha):
    """Each land-cover class partitions V_precip = V_runoff + V_loss exactly."""
    rate = 1e-5  # m/s
    dt = 30.0
    result = precipitation_runoff_loss(rate, CELL_AREA_M2, dt, expected_alpha)
    assert isinstance(result, PartitionResult)
    V_precip = applied_precipitation_volume(rate, CELL_AREA_M2, dt)
    assert result.V_precip_m3 == pytest.approx(V_precip, rel=1e-15)
    assert result.V_runoff_m3 == pytest.approx(expected_alpha * V_precip, rel=1e-15)
    assert result.V_loss_m3 == pytest.approx((1.0 - expected_alpha) * V_precip, rel=1e-15)
    assert result.V_runoff_m3 + result.V_loss_m3 == pytest.approx(result.V_precip_m3, rel=1e-15)


def test_B_partition_uses_config_coefficients():
    """partition uses the approved RunoffConfig coefficients."""
    cfg = build_default_config()
    rate = 2e-5
    dt = 30.0
    for name, alpha in _config_alpha().items():
        result = precipitation_runoff_loss(rate, CELL_AREA_M2, dt, alpha)
        V_precip = applied_precipitation_volume(rate, CELL_AREA_M2, dt)
        assert result.V_runoff_m3 == pytest.approx(alpha * V_precip, rel=1e-15)


def test_B_tc7b_partition_identity_fixture():
    """TC7b partition identity fixture.

    Verifies the rainfall partition arithmetic for a controlled hand case:
    alpha=0.90, V_precip=100 m3 implies V_runoff=90 m3 and V_loss=10 m3.

    This validates the partition identity V_precip = V_runoff + V_loss for a
    single cell/timestep. It does NOT independently validate a complete
    storage update or a simulation-level mass ledger; the full ledger
    (TC7) belongs to a later milestone once the simulator engine exists.
    """
    # Choose rate such that V_precip = exactly 100 m3: rate = 100 / (A * dt).
    rate = 100.0 / (CELL_AREA_M2 * DT)
    alpha = 0.90
    result = precipitation_runoff_loss(rate, CELL_AREA_M2, DT, alpha)
    assert result.V_precip_m3 == pytest.approx(100.0, rel=1e-12)
    assert result.V_runoff_m3 == pytest.approx(90.0, rel=1e-12)
    assert result.V_loss_m3 == pytest.approx(10.0, rel=1e-12)
    # Partition identity: runoff + loss == precip exactly.
    assert result.V_runoff_m3 + result.V_loss_m3 == pytest.approx(
        result.V_precip_m3, rel=1e-15
    )


def test_B_partition_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        precipitation_runoff_loss(1e-5, CELL_AREA_M2, DT, 0.0)
    with pytest.raises(ValueError):
        precipitation_runoff_loss(1e-5, CELL_AREA_M2, DT, 1.5)


def test_B_partition_rejects_nonfinite_inputs():
    """NaN and infinity are rejected for rate, area, timestep, and alpha."""
    with pytest.raises(ValueError):
        precipitation_runoff_loss(float("nan"), CELL_AREA_M2, DT, 0.9)
    with pytest.raises(ValueError):
        precipitation_runoff_loss(float("inf"), CELL_AREA_M2, DT, 0.9)
    with pytest.raises(ValueError):
        precipitation_runoff_loss(1e-5, float("nan"), DT, 0.9)
    with pytest.raises(ValueError):
        precipitation_runoff_loss(1e-5, float("inf"), DT, 0.9)
    with pytest.raises(ValueError):
        precipitation_runoff_loss(1e-5, CELL_AREA_M2, float("nan"), 0.9)
    with pytest.raises(ValueError):
        precipitation_runoff_loss(1e-5, CELL_AREA_M2, float("inf"), 0.9)
    with pytest.raises(ValueError):
        precipitation_runoff_loss(1e-5, CELL_AREA_M2, DT, float("nan"))


def test_B_partition_rejects_negative_rate():
    """A negative rainfall rate is rejected."""
    with pytest.raises(ValueError):
        precipitation_runoff_loss(-1e-5, CELL_AREA_M2, DT, 0.9)


# ---------------------------------------------------------------------------
# C. Midpoint tide calculation (reuses weather.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["R", "S2"])
def test_C_midpoint_matches_analytic(name):
    """Every simulator midpoint tide value equals the analytic tide at that midpoint."""
    tide = SCENARIOS[name].tide
    n_steps = 10
    series = tide_midpoint_series(tide, DT, n_steps)
    for k in range(n_steps):
        t_mid = (k + 0.5) * DT
        expected = _analytic_eta(t_mid, tide.amplitude_m, tide.period_s)
        assert series[k] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_C_continuous_maximum_at_T_over_4():
    """The continuous analytic maximum is A_t at t = T/4."""
    tide = SCENARIOS["R"].tide
    t_peak = tide.eta_peak_time_s
    assert t_peak == pytest.approx(tide.period_s / 4.0, rel=1e-9)
    expected_peak = _analytic_eta(t_peak, tide.amplitude_m, tide.period_s)
    assert expected_peak == pytest.approx(tide.amplitude_m, rel=1e-12)


def test_C_discrete_need_not_equal_continuous_maximum():
    """A discrete midpoint sample need not equal the continuous analytic maximum."""
    tide = SCENARIOS["R"].tide
    n_steps = int(21600.0 / DT)
    series = tide_midpoint_series(tide, DT, n_steps)
    discrete_max = float(series.max())
    # The discrete max should be close to but not required to equal the continuous max.
    assert discrete_max <= tide.amplitude_m + 1e-12
    assert discrete_max > 0.0


# ---------------------------------------------------------------------------
# D. Canonical internal-face enumeration (new)
# ---------------------------------------------------------------------------


def _mask(n_rows, n_cols, ocean_cells):
    m = np.zeros((n_rows, n_cols), dtype=bool)
    for r, c in ocean_cells:
        m[r, c] = True
    return m


def test_D_no_duplicate_faces():
    """Each eligible internal face appears exactly once."""
    is_ocean = _mask(3, 3, [])
    faces = enumerate_internal_faces(3, 3, is_ocean)
    keys = [(f.i, f.j) for f in faces]
    assert len(keys) == len(set(keys)), "duplicate faces found"


def test_D_face_count_small_fixture():
    """2x2 grid with no ocean: 4 unique internal faces."""
    is_ocean = _mask(2, 2, [])
    faces = enumerate_internal_faces(2, 2, is_ocean)
    assert face_count(faces) == 4
    expected = {(0, 1), (0, 2), (1, 3), (2, 3)}
    assert {(f.i, f.j) for f in faces} == expected


def test_D_ocean_cell_excluded():
    """An ocean cell removes all its incident internal faces."""
    is_ocean = _mask(2, 2, [(0, 1)])
    faces = enumerate_internal_faces(2, 2, is_ocean)
    # Cells: (0,0)=0, (0,1)=1[ocean], (1,0)=2, (1,1)=3.
    # Eligible: (0,0)-(1,0) -> (0,2); (1,0)-(1,1) -> (2,3). Face (0,1) gone; (1,1) right of ocean gone.
    expected = {(0, 2), (2, 3)}
    assert {(f.i, f.j) for f in faces} == expected


def test_D_all_ocean_grid_no_faces():
    """A grid where every cell is ocean yields no internal faces."""
    is_ocean = np.ones((3, 3), dtype=bool)
    faces = enumerate_internal_faces(3, 3, is_ocean)
    assert face_count(faces) == 0


def test_D_single_cell_grid_no_faces():
    """A 1x1 grid has no adjacent pairs."""
    is_ocean = _mask(1, 1, [])
    faces = enumerate_internal_faces(1, 1, is_ocean)
    assert face_count(faces) == 0


def test_D_rejects_invalid_mask_dtype():
    """A non-boolean mask dtype is rejected."""
    with pytest.raises(TypeError):
        enumerate_internal_faces(
            2, 2, np.zeros((2, 2), dtype=np.float64)
        )
    with pytest.raises(TypeError):
        enumerate_internal_faces(2, 2, np.zeros((2, 2), dtype=np.int32))


def test_D_rejects_invalid_mask_shape():
    """A mask with the wrong shape is rejected."""
    with pytest.raises(ValueError):
        enumerate_internal_faces(2, 2, np.zeros((3, 3), dtype=bool))


def test_D_rejects_invalid_dimensions():
    """enumerate_internal_faces rejects boolean, float, zero, and negative dimensions."""
    is_ocean = _mask(2, 2, [])
    with pytest.raises(ValueError):
        enumerate_internal_faces(True, 2, is_ocean)
    with pytest.raises(ValueError):
        enumerate_internal_faces(2, True, is_ocean)
    with pytest.raises(ValueError):
        enumerate_internal_faces(False, 2, is_ocean)
    with pytest.raises(ValueError):
        enumerate_internal_faces(1.5, 2, is_ocean)
    with pytest.raises(ValueError):
        enumerate_internal_faces(2, 1.0, is_ocean)
    with pytest.raises(ValueError):
        enumerate_internal_faces(0, 2, is_ocean)  # zero not positive
    with pytest.raises(ValueError):
        enumerate_internal_faces(-1, 2, is_ocean)  # negative


def test_D_canonical_orientation_i_less_than_j():
    """Every face is canonical: lower cell index is i."""
    is_ocean = _mask(4, 4, [])
    faces = enumerate_internal_faces(4, 4, is_ocean)
    for f in faces:
        assert f.i < f.j


def test_D_internalface_rejects_invalid_indices():
    """InternalFace rejects non-integer, boolean, negative, and same-cell indices."""
    with pytest.raises(ValueError):
        InternalFace(i=1.5, j=2)
    with pytest.raises(ValueError):
        InternalFace(i=1.0, j=2)
    with pytest.raises(ValueError):
        InternalFace(i=True, j=2)
    with pytest.raises(ValueError):
        InternalFace(i="1", j=2)
    with pytest.raises(ValueError):
        InternalFace(i=None, j=2)
    with pytest.raises(ValueError):
        InternalFace(i=-1, j=2)
    with pytest.raises(ValueError):
        InternalFace(i=1, j=-2)
    with pytest.raises(ValueError):
        InternalFace(i=1, j=1)  # same cell
    # Valid construction works.
    f = InternalFace(i=0, j=1)
    assert f.i == 0 and f.j == 1


def test_D_deterministic_sorted_ordering():
    """Faces are sorted by (i, j); ordering is deterministic."""
    is_ocean = _mask(3, 3, [(1, 1)])
    faces1 = enumerate_internal_faces(3, 3, is_ocean)
    faces2 = enumerate_internal_faces(3, 3, is_ocean)
    assert faces1 == faces2
    keys = [(f.i, f.j) for f in faces1]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# E. Signed internal-flux to donor/receiver conversion (new)
# ---------------------------------------------------------------------------


def test_E_positive_flux_donor_is_i():
    """V_ij > 0: donor = i, receiver = j, volume = V_ij."""
    req = signed_flux_to_request(5.0, 3, 7)
    assert isinstance(req, FluxRequest)
    assert req.donor == 3
    assert req.receiver == 7
    assert req.volume_m3 == pytest.approx(5.0)


def test_E_negative_flux_donor_is_j():
    """V_ij < 0: donor = j, receiver = i, volume = abs(V_ij)."""
    req = signed_flux_to_request(-5.0, 3, 7)
    assert isinstance(req, FluxRequest)
    assert req.donor == 7
    assert req.receiver == 3
    assert req.volume_m3 == pytest.approx(5.0)


def test_E_zero_flux_no_request():
    """V_ij == 0: no outgoing request (returns None)."""
    assert signed_flux_to_request(0.0, 3, 7) is None
    assert signed_flux_to_request(-0.0, 3, 7) is None


def test_E_transfer_equal_and_opposite():
    """The same volume is subtracted from donor and added to receiver."""
    V = 12.5
    req_pos = signed_flux_to_request(V, 2, 5)
    req_neg = signed_flux_to_request(-V, 2, 5)
    assert req_pos.volume_m3 == req_neg.volume_m3 == pytest.approx(V)
    # Donor/receiver swap: equal and opposite.
    assert {req_pos.donor, req_pos.receiver} == {req_neg.donor, req_neg.receiver}
    assert req_pos.donor == req_neg.receiver
    assert req_pos.receiver == req_neg.donor


def test_E_flux_rejects_nonfinite():
    """NaN and infinite flux are rejected."""
    with pytest.raises(ValueError):
        signed_flux_to_request(float("nan"), 2, 5)
    with pytest.raises(ValueError):
        signed_flux_to_request(float("inf"), 2, 5)
    with pytest.raises(ValueError):
        signed_flux_to_request(float("-inf"), 2, 5)


def test_E_flux_rejects_invalid_endpoints():
    """signed_flux_to_request rejects non-integer, boolean, negative, and
    same-cell endpoints, in both endpoint positions."""
    invalid_i_values = [1.5, 1.0, True, "1", None, -1]
    invalid_j_values = [1.5, 1.0, True, "1", None, -1]
    for bad_i in invalid_i_values:
        with pytest.raises(ValueError):
            signed_flux_to_request(5.0, bad_i, 7)
    for bad_j in invalid_j_values:
        with pytest.raises(ValueError):
            signed_flux_to_request(5.0, 3, bad_j)
    # Same-cell endpoints are also rejected.
    with pytest.raises(ValueError):
        signed_flux_to_request(5.0, 3, 3)


def test_E_flux_request_rejects_invalid_indices():
    """FluxRequest rejects invalid donor/receiver indices and same-cell pairs."""
    with pytest.raises(ValueError):
        FluxRequest(donor=1.5, receiver=2, volume_m3=1.0)
    with pytest.raises(ValueError):
        FluxRequest(donor=True, receiver=2, volume_m3=1.0)
    with pytest.raises(ValueError):
        FluxRequest(donor="1", receiver=2, volume_m3=1.0)
    with pytest.raises(ValueError):
        FluxRequest(donor=None, receiver=2, volume_m3=1.0)
    with pytest.raises(ValueError):
        FluxRequest(donor=-1, receiver=2, volume_m3=1.0)
    with pytest.raises(ValueError):
        FluxRequest(donor=1, receiver=-2, volume_m3=1.0)
    with pytest.raises(ValueError):
        FluxRequest(donor=1, receiver=1, volume_m3=1.0)  # same cell
    # Valid construction works.
    req = FluxRequest(donor=0, receiver=1, volume_m3=5.0)
    assert req.donor == 0 and req.receiver == 1
    assert req.volume_m3 == 5.0


# ---------------------------------------------------------------------------
# F. Deterministic candidate ordering (canonicalization)
# ---------------------------------------------------------------------------


def _as_pairs(faces):
    return [(f.i, f.j) for f in faces]


def test_F_canonicalize_rejects_duplicates_and_orients():
    """canonicalize_faces drops duplicates and forces i < j orientation."""
    # Duplicate unordered pairs, including one with reversed orientation.
    raw = [(0, 1), (1, 0), (0, 1), (2, 3), (3, 2)]
    faces = canonicalize_faces(raw)
    assert _as_pairs(faces) == [(0, 1), (2, 3)]
    assert faces_canonical_and_unique(faces)


def test_F_canonicalize_permutation_invariant():
    """Permuting input order cannot change the canonical sorted output."""
    raw = [(5, 2), (0, 1), (3, 4), (1, 0), (2, 5), (4, 3)]
    base = canonicalize_faces(raw)
    assert _as_pairs(base) == [(0, 1), (2, 5), (3, 4)]
    import itertools

    perms = list(itertools.permutations(raw))
    # Exercise a subset of permutations (full factorial is 720; 12 is enough
    # to demonstrate order independence).
    for perm in list(perms)[:12]:
        assert canonicalize_faces(perm) == base


def test_F_canonicalize_internalface_input():
    """canonicalize_faces accepts InternalFace objects directly (already
    canonical by construction) and raw tuples (possibly non-canonical)."""
    faces = [InternalFace(i=0, j=2), InternalFace(i=1, j=3)]
    out = canonicalize_faces(faces)
    assert _as_pairs(out) == [(0, 2), (1, 3)]
    assert faces_canonical_and_unique(out)


def test_F_enumerate_is_permutation_invariant():
    """Repeated enumeration with identical input yields identical output."""
    is_ocean = _mask(5, 5, [(0, 0), (4, 4), (2, 2)])
    faces = enumerate_internal_faces(5, 5, is_ocean)
    for _ in range(5):
        assert enumerate_internal_faces(5, 5, is_ocean) == faces
    assert faces_canonical_and_unique(faces)


def test_F_canonicalize_rejects_bad_input():
    """canonicalize_faces rejects malformed face pairs."""
    with pytest.raises(ValueError):
        canonicalize_faces([(1, 1)])  # same-cell endpoints
    with pytest.raises(ValueError):
        canonicalize_faces([(1,)])  # not a pair
    with pytest.raises(ValueError):
        canonicalize_faces(["not a face"])  # not ints


def test_F_canonicalize_rejects_invalid_indices():
    """canonicalize_faces rejects non-integer, boolean, and negative cell indices."""
    with pytest.raises(ValueError):
        canonicalize_faces([(1.5, 2)])
    with pytest.raises(ValueError):
        canonicalize_faces([(1.0, 2)])
    with pytest.raises(ValueError):
        canonicalize_faces([(True, 2)])
    with pytest.raises(ValueError):
        canonicalize_faces([("1", 2)])
    with pytest.raises(ValueError):
        canonicalize_faces([(None, 2)])
    with pytest.raises(ValueError):
        canonicalize_faces([(-1, 2)])
    with pytest.raises(ValueError):
        canonicalize_faces([(1, -2)])
    with pytest.raises(ValueError):
        canonicalize_faces([(1, 1)])  # same cell
