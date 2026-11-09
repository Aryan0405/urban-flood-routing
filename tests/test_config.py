"""Phase 0 tests for the locked configuration of Architecture V7.

Every locked numerical value from V4 through V7 is pinned here so that any
unintentional parameter change fails loudly.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from urbanflood.config import (
    MM_PER_H_TO_M_PER_S,
    REFERENCE_SCENARIO_NAME,
    SCENARIOS,
    SYNTHETIC_LABEL,
    DrainageConfig,
    ElevationConfig,
    ExchangeConfig,
    FloodScenario,
    GridConfig,
    LandCover,
    ProjectConfig,
    RainScenario,
    RoadConfig,
    RunoffConfig,
    SeedConfig,
    SusceptibilityConfig,
    TideConfig,
    build_default_config,
)
from urbanflood.rng import (
    STREAM_DRAINAGE,
    STREAM_ENV,
    STREAM_OD_PAIRS,
    STREAM_ROUTING,
    STREAM_WEATHER,
)


def is_close(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# SYNTHETIC labelling
# ---------------------------------------------------------------------------


def test_synthetic_label_constant_present():
    assert SYNTHETIC_LABEL == "SYNTHETIC"


# ---------------------------------------------------------------------------
# Grid and datum
# ---------------------------------------------------------------------------


def test_grid_locked_geometry():
    grid = GridConfig()
    assert grid.n_rows == 96
    assert grid.n_cols == 96
    assert grid.cell_size_m == 50.0
    assert grid.cell_area_m2 == 2500.0
    assert grid.n_cells == 96 * 96
    assert grid.domain_size_m == 4800.0


def test_elevation_datum_and_pins():
    elevation = ElevationConfig()
    assert elevation.msl_m == 0.0
    assert elevation.terrain_min_m == 0.2
    assert elevation.terrain_max_m == 60.0
    assert elevation.coastal_band_min_m == 0.2
    assert elevation.coastal_band_max_m == 1.2
    assert elevation.bridge_deck_freeboard_m == 1.0
    assert elevation.outfall_freeboard_m == 0.0


def test_elevation_rejects_nonzero_msl_and_outfall_freeboard():
    with pytest.raises(ValueError):
        ElevationConfig(msl_m=1.0)
    with pytest.raises(ValueError):
        ElevationConfig(outfall_freeboard_m=0.5)
    # band starting below the terrain minimum is invalid (band pin [0.2, 1.2]
    # must respect the terrain span), as is a band reaching or passing the max
    with pytest.raises(ValueError):
        ElevationConfig(terrain_min_m=0.5)
    with pytest.raises(ValueError):
        ElevationConfig(coastal_band_max_m=70.0)


# ---------------------------------------------------------------------------
# Runoff partition
# ---------------------------------------------------------------------------


def test_runoff_coefficients_locked():
    runoff = RunoffConfig()
    assert runoff.impervious == 0.90
    assert runoff.open_ground == 0.50
    assert runoff.vegetation == 0.30
    assert runoff.water == 1.00
    coefficients = runoff.coefficients
    assert dict(coefficients) == {
        LandCover.IMPERVIOUS: 0.90,
        LandCover.OPEN_GROUND: 0.50,
        LandCover.VEGETATION: 0.30,
        LandCover.WATER: 1.00,
    }
    assert all(0.0 < a <= 1.0 for a in coefficients.values())


# ---------------------------------------------------------------------------
# Rainfall: locked pulse geometry and totals
# ---------------------------------------------------------------------------


def test_rain_r_locked_timings_and_total():
    rain = SCENARIOS["R"].rainfall
    assert rain.name == "R"
    assert rain.peak_mm_per_h == 90.0
    assert rain.total_mm == 120.0
    assert rain.pulse_start_s == 3600.0
    assert rain.storm_window_s == 21600.0
    assert is_close(rain.pulse_duration_s, 9600.0)
    assert is_close(rain.pulse_peak_time_s, 8400.0)
    assert is_close(rain.pulse_end_s, 13200.0)
    # total = 0.5 * peak * duration, exactly in rational arithmetic
    triangle_total = 0.5 * rain.peak_mm_per_h * (rain.pulse_duration_s / 3600.0)
    assert is_close(triangle_total, 120.0)
    assert is_close(rain.triangle_consistency_error_mm, 0.0)


def test_rain_s1_totals():
    rain = SCENARIOS["S1"].rainfall
    assert rain.name == "S1"
    assert rain.peak_mm_per_h == 150.0
    assert rain.total_mm == 200.0
    assert is_close(rain.pulse_duration_s, 9600.0)
    assert is_close(rain.pulse_peak_time_s, 8400.0)
    triangle_total = 0.5 * rain.peak_mm_per_h * (rain.pulse_duration_s / 3600.0)
    assert is_close(triangle_total, 200.0)


def test_rain_scenario_rejects_inconsistent_total():
    # A full-zero-ended 6 h triangle at 90 mm/h would integrate to 270 mm;
    # declaring total 270 with peak 90 over the locked pulses must be rejected
    # because the implied duration would exceed... it would not; it would span
    # 2*270/90 = 6 h, ending exactly at the storm window: still rejected since
    # the pulse must end strictly inside the window.
    with pytest.raises(ValueError):
        RainScenario(name="BAD", peak_mm_per_h=90.0, total_mm=270.0)


def test_rain_scenario_zero_ends_and_pulse_inside_window():
    rain = SCENARIOS["R"].rainfall
    assert rain.pulse_start_s > 0.0
    assert rain.pulse_end_s < rain.storm_window_s


# ---------------------------------------------------------------------------
# Tide: locked function and sampling property
# ---------------------------------------------------------------------------


def test_tide_locked_function():
    tide = SCENARIOS["R"].tide
    assert tide.amplitude_m == 0.8
    assert tide.period_s == 44712.0
    assert tide.phase_rad == 0.0
    assert tide.surge_m == 0.0
    assert is_close(tide.eta_peak_time_s, 11178.0)
    assert is_close(tide.eta(tide.eta_peak_time_s), 0.8)
    # window-end value: eta(21600) = 0.8 * sin(2*pi*21600/44712)
    expected_end = 0.8 * math.sin(2.0 * math.pi * 21600.0 / 44712.0)
    assert is_close(tide.eta(21600.0), expected_end)
    # tide starts exactly at MSL at t = 0
    assert tide.eta(0.0) == 0.0


def test_tide_rejects_nonzero_surge_and_phase():
    with pytest.raises(ValueError):
        TideConfig(amplitude_m=0.8, surge_m=0.1)
    with pytest.raises(ValueError):
        TideConfig(amplitude_m=0.8, phase_rad=0.1)


# ---------------------------------------------------------------------------
# Locked scenarios
# ---------------------------------------------------------------------------


def test_scenarios_locked_set():
    assert set(SCENARIOS.keys()) == {"R", "S1", "S2", "S3"}
    reference = SCENARIOS["R"]
    assert reference.is_reference is True
    assert SCENARIOS["S1"].is_reference is False
    assert SCENARIOS["S2"].is_reference is False
    assert SCENARIOS["S3"].is_reference is False
    assert REFERENCE_SCENARIO_NAME == "R"


def test_scenario_one_factor_at_a_time_contrasts():
    reference = SCENARIOS["R"]
    s1, s2, s3 = SCENARIOS["S1"], SCENARIOS["S2"], SCENARIOS["S3"]
    # S1 changes only rainfall
    assert s1.rainfall.peak_mm_per_h == 150.0
    assert s1.tide.amplitude_m == reference.tide.amplitude_m
    assert s1.blocked_fraction == reference.blocked_fraction
    # S2 changes only the tide
    assert s2.rainfall is reference.rainfall
    assert s2.tide.amplitude_m == 1.8
    assert s2.blocked_fraction == reference.blocked_fraction
    # S3 changes only blockage
    assert s3.rainfall is reference.rainfall
    assert s3.tide is reference.tide
    assert s3.blocked_fraction == 0.40


# ---------------------------------------------------------------------------
# Drainage
# ---------------------------------------------------------------------------


def test_drainage_locked_parameters():
    drainage = DrainageConfig()
    assert drainage.capacity_min_mm_per_h == 20.0
    assert drainage.capacity_max_mm_per_h == 80.0
    assert is_close(drainage.capacity_min_m_per_s, 20.0 * MM_PER_H_TO_M_PER_S)
    assert is_close(drainage.capacity_max_m_per_s, 80.0 * MM_PER_H_TO_M_PER_S)
    assert drainage.efficiency_min == 0.5
    assert drainage.efficiency_max == 1.0
    assert drainage.ponding_reference_depth_m == 0.01
    assert drainage.outfall_freeboard_m == 0.0
    assert drainage.blocked_fraction_reference == 0.20
    assert drainage.blocked_fraction_increased == 0.40


# ---------------------------------------------------------------------------
# Exchange model
# ---------------------------------------------------------------------------


def test_exchange_design_bound_values():
    exchange = ExchangeConfig()
    assert exchange.exchange_velocity_m_per_s == 0.1
    assert exchange.timestep_s == 30.0
    assert exchange.sensitivity_timestep_s == 15.0
    assert exchange.film_threshold_depth_m == 0.005
    assert exchange.declared_f_step_max == 0.20
    # f_step = 2 * c * dt / dx: 12 percent per step at dt = 30 s
    assert is_close(exchange.f_step(30.0, 50.0), 0.12)
    assert is_close(exchange.f_step(15.0, 50.0), 0.06)


def test_exchange_validate_against_grid_passes_and_fails():
    grid = GridConfig()
    ExchangeConfig().validate_against_grid(grid)  # must not raise
    fast = ExchangeConfig(exchange_velocity_m_per_s=0.5)
    with pytest.raises(ValueError):
        fast.validate_against_grid(grid)


# ---------------------------------------------------------------------------
# Susceptibility declaration
# ---------------------------------------------------------------------------


def test_susceptibility_locked_ranges_and_weights():
    susceptibility = SusceptibilityConfig()
    assert susceptibility.elevation_range_m == (0.0, 40.0)
    assert susceptibility.tpi_radius_m == 200.0
    assert susceptibility.tpi_span_m == 10.0
    assert susceptibility.water_distance_max_m == 1000.0
    assert susceptibility.slope_max == 0.3
    assert susceptibility.weight_elevation == 0.35
    assert susceptibility.weight_tpi == 0.25
    assert susceptibility.weight_water_distance == 0.15
    assert susceptibility.weight_land_cover == 0.15
    assert susceptibility.weight_slope == 0.10
    weights = sum(
        (
            susceptibility.weight_elevation,
            susceptibility.weight_tpi,
            susceptibility.weight_water_distance,
            susceptibility.weight_land_cover,
            susceptibility.weight_slope,
        )
    )
    assert math.isclose(weights, 1.0, abs_tol=1e-9)


def test_susceptibility_rejects_bad_weights():
    with pytest.raises(ValueError):
        SusceptibilityConfig(weight_tpi=0.30)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_routing_locked_parameters():
    routing = RoadConfig()
    assert routing.arterial_speed_kmh == 50.0
    assert routing.secondary_speed_kmh == 30.0
    assert routing.local_speed_kmh == 20.0
    assert routing.closure_threshold_default_m == 0.30
    assert routing.closure_thresholds_sweep_m == (0.20, 0.30, 0.45)
    assert routing.label_threshold_default_m == 0.10
    assert routing.label_thresholds_sweep_m == (0.05, 0.10, 0.20, 0.30)
    assert routing.lambda_sweep == (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
    assert routing.edge_depth_samples == 3
    assert routing.n_od_pairs == 20


# ---------------------------------------------------------------------------
# Seeds and experiment scope
# ---------------------------------------------------------------------------


def test_seed_config_defaults_and_experiment_streams():
    seeds = SeedConfig()
    assert seeds.master_seed == 2027
    experiment_seeds = seeds.env_experiment_seeds(5)
    assert len(experiment_seeds) == 5
    assert len(set(experiment_seeds)) == 5
    assert all(seed >= 0 for seed in experiment_seeds)


def test_seed_config_rejects_bad_inputs():
    with pytest.raises(ValueError):
        SeedConfig(master_seed=-1)


def test_required_stream_names_resolve():
    seeds = SeedConfig()
    for stream in (STREAM_ENV, STREAM_DRAINAGE, STREAM_WEATHER, STREAM_ROUTING, STREAM_OD_PAIRS):
        assert isinstance(seeds.stream_seed(stream), int)
        assert seeds.stream_seed(stream) >= 0


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_project_config_instantiates_and_is_frozen():
    config = build_default_config()
    assert isinstance(config, ProjectConfig)
    assert isinstance(config.reference_scenario, FloodScenario)
    assert config.reference_scenario.name == "R"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.grid = GridConfig(cell_size_m=40.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.grid.n_rows = 128


def test_project_config_scenarios_view():
    config = build_default_config()
    scenarios = config.scenarios
    assert set(scenarios.keys()) == {"R", "S1", "S2", "S3"}
    with pytest.raises(TypeError):
        scenarios["R"] = None
