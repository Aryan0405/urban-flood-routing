"""Phase 1 tests for the SYNTHETIC weather layer (weather.py).

Covers exact interval integration (total preservation at any dt), the
triangular pulse shape, the locked S2/S3 inheritance of the R rainfall, and
the midpoint tide convention.

Also asserts the zero-RNG-consumption honesty claim: no weather quantity
touches an RNG stream.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from urbanflood.config import (
    MM_PER_H_TO_M_PER_S,
    SCENARIOS,
    build_default_config,
)
from urbanflood.weather import (
    applied_precipitation_volume,
    hyetograph,
    interval_mean_rate,
    rain_rate_at,
    tide_midpoint_series,
    total_precipitation_volume,
)

DT = 30.0
DT_SENS = 15.0
N_STEPS = 720  # storm window 21600 s / 30 s


@pytest.mark.parametrize("name,total_mm", [("R", 120.0), ("S1", 200.0)])
def test_hyetograph_preserves_declared_total_exact(name, total_mm):
    rain = SCENARIOS[name].rainfall
    rates = hyetograph(rain, DT, N_STEPS)
    applied_m = rates.sum() * DT
    assert applied_m == pytest.approx(total_mm / 1000.0, rel=1e-12)


@pytest.mark.parametrize("dt", [7.5, 10.0, 15.0, 60.0, 300.0])
def test_hyetograph_total_preserved_at_other_timesteps(dt):
    rain = SCENARIOS["R"].rainfall
    n_steps = int(21600.0 // dt)
    applied = hyetograph(rain, dt, n_steps).sum() * dt
    assert applied == pytest.approx(0.120, rel=1e-12)


def test_hyetograph_support_inside_pulse():
    rain = SCENARIOS["R"].rainfall
    rates = hyetograph(rain, DT, N_STEPS)
    ts, te = rain.pulse_start_s, rain.pulse_end_s
    starts = np.arange(N_STEPS) * DT
    outside = (starts + DT <= ts) | (starts >= te)
    inside = ~outside
    assert np.all(rates[outside] == 0.0)
    assert np.all(rates[inside] > 0.0)


def test_hyetograph_bounded_by_peak():
    for name in ("R", "S1"):
        rain = SCENARIOS[name].rainfall
        rates = hyetograph(rain, DT, N_STEPS)
        peak_m_s = rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S
        assert rates.max() <= peak_m_s + 1e-15
        assert rates.max() > 0.8 * peak_m_s


def test_interval_mean_rate_matches_fine_trapezoid():
    rain = SCENARIOS["R"].rainfall
    for k in range(0, N_STEPS, 37):
        a, b = k * DT, (k + 1) * DT
        coarse = interval_mean_rate(rain, a, b)
        ts, tp, te = rain.pulse_start_s, rain.pulse_peak_time_s, rain.pulse_end_s
        if b <= ts or a >= te:
            assert coarse == 0.0
            continue
        t = np.linspace(a, b, 20001)
        rates = np.where(
            t <= ts,
            0.0,
            np.where(
                t <= tp,
                rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S * (t - ts) / (tp - ts),
                np.where(t <= te, rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S * (te - t) / (te - tp), 0.0),
            ),
        )
        # np.trapezoid is NumPy >= 2.0; requirements declare numpy>=1.26, so
        # the trapezoidal rule is applied directly (identical on uniform and
        # non-uniform samples alike).
        fine = float(np.sum(0.5 * (rates[:-1] + rates[1:]) * np.diff(t))) / DT
        assert coarse == pytest.approx(fine, rel=1e-6)


def test_rain_rate_knot_values():
    rain = SCENARIOS["R"].rainfall
    ts, tp, te = rain.pulse_start_s, rain.pulse_peak_time_s, rain.pulse_end_s
    peak_m_s = rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S
    assert rain_rate_at(ts, rain) == 0.0
    assert rain_rate_at(te, rain) == 0.0
    assert rain_rate_at(tp, rain) == pytest.approx(peak_m_s, rel=1e-12)
    assert rain_rate_at(ts - 1.0, rain) == 0.0
    assert rain_rate_at(te + 3600.0, rain) == 0.0
    # locked timings (checked exactly in Phase 0 tests; re-verified here)
    assert ts == pytest.approx(3600.0, abs=1e-6)
    assert tp == pytest.approx(8400.0, abs=1e-6)
    assert te == pytest.approx(13200.0, abs=1e-6)


def test_scenario_rain_inheritance():
    """S2 and S3 inherit the R rainfall profile exactly (bitwise hyetograph)."""
    r_rates = hyetograph(SCENARIOS["R"].rainfall, DT, N_STEPS)
    s2_rates = hyetograph(SCENARIOS["S2"].rainfall, DT, N_STEPS)
    s3_rates = hyetograph(SCENARIOS["S3"].rainfall, DT, N_STEPS)
    assert r_rates.tobytes() == s2_rates.tobytes()
    assert r_rates.tobytes() == s3_rates.tobytes()


@pytest.mark.parametrize("k", [0, 137, 280, 300, 440, 719])
def test_applied_precipitation_volume_per_timestep(k):
    """Per-cell, per-timestep helper semantics: V_precip = R_k * A * dt."""
    rain = SCENARIOS["R"].rainfall
    rates = hyetograph(rain, DT, N_STEPS)
    area = build_default_config().grid.cell_area_m2
    volume = applied_precipitation_volume(rates[k], area, DT)
    assert volume == pytest.approx(rates[k] * area * DT, rel=1e-15)
    assert volume == pytest.approx(
        rates[k] * DT * area, rel=1e-15
    )
    per_cell_depth = volume / area
    assert per_cell_depth == pytest.approx(rates[k] * DT, rel=1e-15)


def test_total_precipitation_volume_aggregate_matches_stepwise_sum():
    """The aggregate helper is exactly the stepwise sum of per-timestep volumes."""
    rain = SCENARIOS["R"].rainfall
    rates = hyetograph(rain, DT, N_STEPS)
    area = build_default_config().grid.cell_area_m2
    stepwise = sum(applied_precipitation_volume(rate, area, DT) for rate in rates)
    aggregate = total_precipitation_volume(rates, area, DT)
    assert aggregate == pytest.approx(stepwise, rel=1e-15)
    assert aggregate == pytest.approx(0.120 * area, rel=1e-12)


def test_hyetograph_rejects_bad_inputs():
    rain = SCENARIOS["R"].rainfall
    with pytest.raises(ValueError):
        hyetograph(rain, 0.0, N_STEPS)
    with pytest.raises(ValueError):
        hyetograph(rain, -1.0, N_STEPS)
    with pytest.raises(ValueError):
        hyetograph(rain, DT, 0)
    with pytest.raises(ValueError):
        interval_mean_rate(rain, 10.0, 10.0)


def test_zero_rng_consumption_is_structural():
    """The weather layer must not consume any RNG stream (locked scenarios
    are deterministic). Asserted structurally: no module-level RNG use and
    no generator parameters in the public weather functions."""
    for function in (rain_rate_at, interval_mean_rate, hyetograph, tide_midpoint_series):
        names = list(inspect.signature(function).parameters)
        for name in ("rng", "seed", "master_seed", "generator", "bit_generator"):
            assert name not in names, f"{function.__name__} must not take {name}"


# --- tide midpoint convention ---------------------------------------------------

def test_tide_midpoint_series_amplitude_bounds():
    for name in ("R", "S2"):
        tide = SCENARIOS[name].tide
        series = tide_midpoint_series(tide, DT, N_STEPS)
        assert series.shape == (N_STEPS,)
        assert np.all(series <= tide.amplitude_m + 1e-12)
        assert np.all(series >= -tide.amplitude_m - 1e-12)


def test_tide_midpoint_hits_near_peak():
    tide = SCENARIOS["R"].tide
    series = tide_midpoint_series(tide, DT, N_STEPS)
    t_peak = tide.eta_peak_time_s
    assert t_peak == pytest.approx(11178.0, rel=1e-9)
    k = int((t_peak - 0.5 * DT) / DT)
    nearest = max(0, min(N_STEPS - 1, k))
    assert series[nearest] > 0.999 * tide.amplitude_m


def test_tide_midpoint_matches_analytic_eta():
    tide = SCENARIOS["R"].tide
    series = tide_midpoint_series(tide, DT, 5)
    for k in range(5):
        expected = tide.eta((k + 0.5) * DT)
        assert series[k] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_tide_at_full_hours_reference():
    tide = SCENARIOS["R"].tide
    assert tide.eta(0.0) == 0.0
    assert tide.eta(tide.period_s / 2.0) == pytest.approx(0.0, abs=1e-12)
