"""SYNTHETIC weather layer: locked rainfall pulses and tide (Phase 1).

All weather in this project is synthetic and fully deterministic: the four
locked scenarios (R, S1, S2, S3 in urbanflood.config) are closed-form
functions of the locked configuration. Rainfall is spatially uniform and the
tide is a closed-form sine, so no locked V7 quantity consumes randomness for
weather. The named "weather" seed stream exists (rng.py) but is deliberately
left unused; inventing spurious random draws would be dishonest. Zero RNG
consumption is asserted by test.

Exact interval integration (locked convention):
    R_k = (1/dt) * integral of R(t) over [t_k, t_k + dt)

so the applied precipitation depth per timestep equals the exact integral of
the triangular pulse over that interval, at ANY dt. Summed applied depth
equals the declared storm total exactly when the requested window covers the
full pulse [pulse_start_s, pulse_end_s]; rainfall is zero outside the pulse,
so timesteps outside it contribute zero and a truncated window under-reports
the total (verified by test here; the simulator-side volume bookkeeping is
TC12 in Phase 2).

Per-step evaluation convention for the tide (locked): exactly one evaluation
per timestep at the interval midpoint, t_mid,k = (k + 0.5) * dt, feeding BOTH
drainage outfall blocking and coastal sea exchange in Phase 2. This module
provides the single midpoint series generator so there is exactly one
implementation of that convention.
"""

from __future__ import annotations

import numpy as np

from urbanflood.config import MM_PER_H_TO_M_PER_S, RainScenario, TideConfig


def rain_rate_at(t_s: float, rain: RainScenario) -> float:
    """Instantaneous rain rate in m/s of the triangular pulse.

    Zero outside the pulse; linear rise from pulse start to the peak and
    linear fall from the peak to the pulse end. Exact at the knots.
    """
    peak_m_per_s = rain.peak_mm_per_h * MM_PER_H_TO_M_PER_S
    t_start = rain.pulse_start_s
    t_peak = rain.pulse_peak_time_s
    t_end = rain.pulse_end_s
    if t_s <= t_start or t_s >= t_end:
        return 0.0
    if t_s <= t_peak:
        return peak_m_per_s * (t_s - t_start) / (t_peak - t_start)
    return peak_m_per_s * (t_end - t_s) / (t_end - t_peak)


def interval_mean_rate(rain: RainScenario, t_start_s: float, t_end_s: float) -> float:
    """Exact interval-mean rain rate in m/s: integral of R over [a, b) / (b-a).

    Exact by construction: the pulse is piecewise linear, the interval is cut
    at the pulse knots, and each linear piece integrates to the trapezoid
    formula with no discretisation error.
    """
    if not t_end_s > t_start_s:
        raise ValueError("interval must satisfy t_end > t_start")
    t_p_start = rain.pulse_start_s
    t_p_peak = rain.pulse_peak_time_s
    t_p_end = rain.pulse_end_s
    internal_knots = {
        knot for knot in (t_p_start, t_p_peak, t_p_end) if t_start_s < knot < t_end_s
    }
    cuts = sorted({t_start_s, t_end_s, *internal_knots})
    total_integral = 0.0
    for x0, x1 in zip(cuts[:-1], cuts[1:]):
        if x1 <= t_p_start or x0 >= t_p_end:
            continue
        f0 = rain_rate_at(x0, rain)
        f1 = rain_rate_at(x1, rain)
        total_integral += 0.5 * (f0 + f1) * (x1 - x0)
    return total_integral / (t_end_s - t_start_s)


def hyetograph(rain: RainScenario, dt_s: float, n_steps: int) -> np.ndarray:
    """Applied rain rates R_k in m/s for steps k = 0..n_steps-1.

    R_k is the exact interval mean over [k*dt, (k+1)*dt). Summed applied
    depth (R.sum() * dt) equals the declared storm total in metres when the
    requested window [0, n_steps * dt_s) covers the full pulse
    [pulse_start_s, pulse_end_s]; rainfall is zero outside the pulse, so
    timesteps outside it contribute zero and a truncated window under-reports
    the total.
    """
    if dt_s <= 0.0:
        raise ValueError("timestep must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    rates = np.empty(n_steps, dtype=np.float64)
    for k in range(n_steps):
        rates[k] = interval_mean_rate(rain, k * dt_s, (k + 1) * dt_s)
    return rates


def applied_precipitation_volume(
    rate_k_m_per_s: float, cell_area_m2: float, dt_s: float
) -> float:
    """V_precip for ONE cell over ONE timestep: R_k * A * dt (m3)."""
    return rate_k_m_per_s * cell_area_m2 * dt_s


def total_precipitation_volume(
    rates_k_m_per_s, cell_area_m2: float, dt_s: float
) -> float:
    """Aggregate storm volume over a cell and a hyetograph (m3).

    Convenience sum of applied_precipitation_volume over every timestep of
    the supplied rates array; the locked partition and the exact per-step
    integration are unchanged by this aggregate view.
    """
    rates = np.asarray(rates_k_m_per_s, dtype=np.float64)
    return float(rates.sum()) * cell_area_m2 * dt_s


def tide_midpoint_series(tide: TideConfig, dt_s: float, n_steps: int) -> np.ndarray:
    """Midpoint tide elevations eta(t_mid,k) in metres above MSL.

    t_mid,k = (k + 0.5) * dt. This single series is THE per-step tide signal:
    in Phase 2 it feeds both outfall blocking and coastal sea exchange, one
    evaluation per step.
    """
    if dt_s <= 0.0:
        raise ValueError("timestep must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    t_mid = (np.arange(n_steps, dtype=np.float64) + 0.5) * dt_s
    return tide.amplitude_m * np.sin(2.0 * np.pi * t_mid / tide.period_s + tide.phase_rad)
