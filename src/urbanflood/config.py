"""Central configuration for the synthetic urban flood simulation project.

SYNTHETIC: every parameter, seed stream, and scenario defined here drives
synthetically generated data only. No real Mumbai road, terrain, drainage,
tide, or flood-event data is used anywhere in this project.

Specification: Architecture V7 (approved).

Locked values carried by this module (full record in README.md):

Grid: 96 x 96 cells at 50 m resolution (cell area 2500 m2, domain 4.8 km).

Vertical datum (V5): synthetic mean sea level MSL = 0.0 m; all elevations in
metres above this single datum. Terrain is generated in [0.2, 60.0] m, the
coastal land band is pinned to [0.2, 1.2] m, the outfall invert sits at the
cell bed elevation (freeboard 0.0 m), and bridge decks sit at the crossed
waterbody level plus 1.0 m, floored by the lower approach road elevation.

Rainfall partition (V7 correction 1):

  V_precip = R_k * A * dt            total precipitation arriving at a cell [m3]
  V_runoff = alpha_i * V_precip      portion ENTERING the surface-water system
  V_loss   = V_precip - V_runoff     portion removed before entering surface water

  The water-depth update adds V_runoff ONLY. V_precip is a ledger INPUT term
  and V_loss is a ledger OUTPUT term; neither is ever applied to depth. The
  mass ledger is IN - OUT - dS with
    IN  = sum(V_precip) + SEA_IN
    OUT = sum(V_loss) + sum(V_drain) + SEA_OUT
  asserted to 1e-9 relative on every run (TC7).

  Ledger scope and initial state (user decision, Option A, Phase 0 audit):
  Inland permanent-water cells are ORDINARY DYNAMIC surface-water cells, not
  a special simulator category:
    - initial depth comes from the waterbody's defined normal/spill level
      against its bed elevation: h_init,i = max(0, L_normal,i - z_i), with
      L_normal,i the pre-carving rim/spill elevation recorded at generation;
    - they exchange water with neighbouring cells under the ordinary flux
      rules and can overflow onto surrounding land during storms;
    - their storage is included in the total surface-water storage;
    - the mass ledger tracks CHANGES relative to the initial state:
      dS = final total storage - initial total storage, so the initial lake
      volume sits inside the ledger baseline and is never counted as new
      rainfall input;
    - WATER cells retain alpha = 1.0 under the established rainfall
      partition.
  Ocean cells are the only exclusion, acting as the boundary reservoir.

Synchronous update convention (V6 correction 2, retained):
  One old-state snapshot h_now per step feeds ALL candidate quantities
  (drainage, per-face sea exchange, inter-cell pair fluxes). Rainfall is an
  external input applied in the final simultaneous update; it cannot be
  drained, exported, or transported during the same timestep. There are no
  sequential cell updates.

Canonical ordering convention (V7 correction 2, implemented in Phase 2):
  Every unordered adjacent pair and every sea face carries a stable
  identifier derived only from grid geometry. Candidates may be generated in
  any order into a store and are applied with one fixed canonical id order
  governing per-cell accumulation, availability rescaling, and ledger
  accumulation. TC14 asserts bitwise order invariance on one platform.

Exchange model: c = 0.1 m/s is an empirical modelling choice for the linear
head-difference exchange scheme; it has no derivation and no calibration.
The constraint f_step = 2 * c * dt / dx <= 0.20 (0.12 at dt = 30 s) is a
DECLARED NUMERICAL DESIGN BOUND, not a proof of stability. Stability and
accuracy are verified empirically (TC8 toy problems, TC10 dt sensitivity,
TC12 rainfall bookkeeping); c is never tuned to pass the dt test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from urbanflood.rng import (
    STREAM_DRAINAGE,
    STREAM_ENV,
    STREAM_OD_PAIRS,
    STREAM_ROUTING,
    STREAM_WEATHER,
    derive_stream_seed,
)

SYNTHETIC_LABEL = "SYNTHETIC"

#: mm/h to m/s. 1 mm/h = 1e-3 m / 3600 s = 2.777...e-7 m/s.
MM_PER_H_TO_M_PER_S = 1.0 / 3.6e6


class LandCover(str, Enum):
    """SYNTHETIC land cover classes (Phase 1 generation)."""

    IMPERVIOUS = "impervious"
    OPEN_GROUND = "open_ground"
    VEGETATION = "vegetation"
    WATER = "water"


@dataclass(frozen=True)
class GridConfig:
    """SYNTHETIC raster geometry, locked at 96 x 96 cells at 50 m (V5)."""

    n_rows: int = 96
    n_cols: int = 96
    cell_size_m: float = 50.0

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or self.n_cols <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.cell_size_m <= 0.0:
            raise ValueError("cell size must be positive")

    @property
    def cell_area_m2(self) -> float:
        return self.cell_size_m * self.cell_size_m

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def domain_size_m(self) -> float:
        return self.n_rows * self.cell_size_m


@dataclass(frozen=True)
class ElevationConfig:
    """Datum and elevation constraints (V5, unchanged in V6/V7).

    All values in metres above MSL. Terrain spans [0.2, 60.0] m, entirely
    above sea level; the coastal band pin [0.2, 1.2] m makes the outfall
    rule eta_k > Z_out bind at low cells under R and broadly under S2.
    """

    msl_m: float = 0.0
    terrain_min_m: float = 0.2
    terrain_max_m: float = 60.0
    coastal_band_min_m: float = 0.2
    coastal_band_max_m: float = 1.2
    bridge_deck_freeboard_m: float = 1.0
    outfall_freeboard_m: float = 0.0

    def __post_init__(self) -> None:
        if self.msl_m != 0.0:
            raise ValueError("MSL is fixed at 0.0 m as the single vertical datum")
        if not (
            self.terrain_min_m
            <= self.coastal_band_min_m
            <= self.coastal_band_max_m
            < self.terrain_max_m
        ):
            raise ValueError("coastal band must sit inside [terrain_min, terrain_max)")
        if self.bridge_deck_freeboard_m <= 0.0:
            raise ValueError("bridge deck freeboard must be positive")
        if self.outfall_freeboard_m != 0.0:
            raise ValueError("outfall invert is fixed at the cell bed elevation (V5)")


@dataclass(frozen=True)
class RunoffConfig:
    """Runoff coefficients alpha_i by land cover (V3, retained in V7).

    Rational-method-style loss model: rain is partitioned per cell into
    V_runoff = alpha * V_precip (enters the surface-water system) and
    V_loss = V_precip - V_runoff (never enters the surface-water system).
    No soil storage state exists; disclosed as weakness W3 of Architecture V7.
    The WATER class (alpha = 1.0) is treated per the ledger-scope statement
    in the module docstring (Option A decision, Phase 0 audit): waterbody
    cells are ordinary dynamic cells with initial depth set from the spill
    level against the bed, lateral exchange and overflow onto land allowed,
    and initial volume inside the change-relative ledger baseline rather
    than counted as a rainfall input term.
    """

    impervious: float = 0.90
    open_ground: float = 0.50
    vegetation: float = 0.30
    water: float = 1.00

    def __post_init__(self) -> None:
        for name in ("impervious", "open_ground", "vegetation", "water"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"runoff coefficient {name} must lie in (0, 1]")

    @property
    def coefficients(self) -> MappingProxyType:
        return MappingProxyType(
            {
                LandCover.IMPERVIOUS: self.impervious,
                LandCover.OPEN_GROUND: self.open_ground,
                LandCover.VEGETATION: self.vegetation,
                LandCover.WATER: self.water,
            }
        )


@dataclass(frozen=True)
class RainScenario:
    """SYNTHETIC rainfall: a symmetric triangular pulse inside a storm window.

    Exact rational relations (V5 correction 1): pulse duration equals
    2 * total / peak, the peak falls at the pulse midpoint, and rainfall is
    zero outside the pulse. For R and S1 this yields the locked timings
    t_s = 3600 s, t_p = 8400 s, t_e = 13200 s in exact arithmetic; floating
    point evaluation lands within 1e-9 relative of these values.
    Application is by exact interval integration, R_k = (1/dt) * integral of
    R(t) over [t_k, t_k + dt), so applied volume equals the declared total
    at any dt (verified by TC12).
    """

    name: str
    peak_mm_per_h: float
    total_mm: float
    pulse_start_s: float = 3600.0
    storm_window_s: float = 21600.0

    def __post_init__(self) -> None:
        if self.peak_mm_per_h <= 0.0 or self.total_mm <= 0.0:
            raise ValueError("peak intensity and total depth must be positive")
        if self.pulse_start_s < 0.0:
            raise ValueError("pulse start must be non-negative")
        if self.storm_window_s <= self.pulse_start_s:
            raise ValueError("storm window must extend past the pulse start")
        if self.pulse_end_s >= self.storm_window_s:
            raise ValueError("pulse must end inside the storm window")
        if abs(self.triangle_consistency_error_mm) > 1e-6:
            raise ValueError("declared total and peak are inconsistent with a triangular pulse")

    @property
    def pulse_duration_s(self) -> float:
        # Exact rational: 2 * total / peak in hours, converted to seconds.
        return 2.0 * self.total_mm / self.peak_mm_per_h * 3600.0

    @property
    def pulse_peak_time_s(self) -> float:
        return self.pulse_start_s + 0.5 * self.pulse_duration_s

    @property
    def pulse_end_s(self) -> float:
        return self.pulse_start_s + self.pulse_duration_s

    @property
    def triangle_consistency_error_mm(self) -> float:
        return 0.5 * self.peak_mm_per_h * (self.pulse_duration_s / 3600.0) - self.total_mm


@dataclass(frozen=True)
class TideConfig:
    """SYNTHETIC tide on a synthetic coast (V6, retained in V7).

    eta(t) = amplitude * sin(2 * pi * t / T + phase). Period T = 44712 s is
    the M2 period, borrowed as a timescale for a synthetic coast. Phase is
    fixed at zero and surge is fixed at zero; the former ramp term does not
    exist. The analytic peak therefore falls at t = T/4 = 11178 s, inside
    the 6 h storm window. Per-step evaluation uses the interval midpoint
    eta_k = eta(t_k + dt/2); exactly one evaluation per step feeds BOTH
    drainage outfall blocking (E2) and coastal exchange (E3) (V7 correction 3).
    """

    amplitude_m: float
    period_s: float = 44712.0
    phase_rad: float = 0.0
    surge_m: float = 0.0

    def __post_init__(self) -> None:
        if self.amplitude_m <= 0.0:
            raise ValueError("tide amplitude must be positive")
        if self.period_s <= 0.0:
            raise ValueError("tide period must be positive")
        if self.phase_rad != 0.0:
            raise ValueError("tide phase is fixed at zero (V6)")
        if self.surge_m != 0.0:
            raise ValueError("surge is fixed at zero; the ramp term does not exist (V6)")

    def eta(self, t_s: float) -> float:
        """Instantaneous sea surface elevation in metres above MSL."""
        return self.amplitude_m * math.sin(2.0 * math.pi * t_s / self.period_s + self.phase_rad)

    @property
    def eta_peak_time_s(self) -> float:
        """First time inside the window at which the sine argument is pi/2."""
        return (0.5 * math.pi - self.phase_rad) * self.period_s / (2.0 * math.pi)


@dataclass(frozen=True)
class FloodScenario:
    """A complete SYNTHETIC scenario: rainfall profile, tide, blocked drains.

    Locked one-factor-at-a-time set (V5, unchanged in V6/V7), each factor
    manipulated alone against the reference R:

      R:  peak 90 mm/h, total 120 mm, tide A_t = 0.8 m, blocked 0.20
      S1: peak 150 mm/h, total 200 mm, tide A_t = 0.8 m, blocked 0.20
      S2: rain as R, tide A_t = 1.8 m, blocked 0.20
      S3: rain as R, tide A_t = 0.8 m, blocked 0.40

    S3 regenerates the blocked mask on the same drainage stream; capacity D,
    efficiency eta_d, and outfalls are untouched by the manipulation.
    """

    name: str
    rainfall: RainScenario
    tide: TideConfig
    blocked_fraction: float
    is_reference: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.blocked_fraction <= 1.0:
            raise ValueError("blocked fraction must lie in [0, 1]")


RAIN_R = RainScenario(name="R", peak_mm_per_h=90.0, total_mm=120.0)
RAIN_S1 = RainScenario(name="S1", peak_mm_per_h=150.0, total_mm=200.0)
TIDE_R = TideConfig(amplitude_m=0.8)
TIDE_S2 = TideConfig(amplitude_m=1.8)

SCENARIOS: Mapping[str, FloodScenario] = MappingProxyType(
    {
        "R": FloodScenario(name="R", rainfall=RAIN_R, tide=TIDE_R, blocked_fraction=0.20, is_reference=True),
        "S1": FloodScenario(name="S1", rainfall=RAIN_S1, tide=TIDE_R, blocked_fraction=0.20),
        "S2": FloodScenario(name="S2", rainfall=RAIN_R, tide=TIDE_S2, blocked_fraction=0.20),
        "S3": FloodScenario(name="S3", rainfall=RAIN_R, tide=TIDE_R, blocked_fraction=0.40),
    }
)
"""Immutable locked scenario registry.

The mapping view is read-only: insertion, deletion, and value replacement
raise TypeError. Every registered FloodScenario (and its rainfall and tide
members) is a frozen dataclass, so nested mutation also fails. There is no
public mutable handle to this registry.
"""

REFERENCE_SCENARIO_NAME = "R"


@dataclass(frozen=True)
class DrainageConfig:
    """Hidden drainage layer (V4 split, retained; E2 numerics of V6).

    Capacity D_i, efficiency eta_d,i, and blocked status b_i are generated
    from the independent drainage seed stream whose generator receives no
    terrain input. Outfall placement and elevation are terrain-dependent
    deterministic infrastructure: Z_out,i = z_i + outfall_freeboard_m with
    the freeboard locked at 0.0 m (V5), so outfalls block exactly while the
    computed tide eta_k exceeds the cell bed elevation.
    """

    capacity_min_mm_per_h: float = 20.0
    capacity_max_mm_per_h: float = 80.0
    efficiency_min: float = 0.5
    efficiency_max: float = 1.0
    ponding_reference_depth_m: float = 0.01
    outfall_freeboard_m: float = 0.0
    blocked_fraction_reference: float = 0.20
    blocked_fraction_increased: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.capacity_min_mm_per_h < self.capacity_max_mm_per_h:
            raise ValueError("drainage capacity range must satisfy 0 < min < max")
        if not 0.0 < self.efficiency_min <= self.efficiency_max <= 1.0:
            raise ValueError("efficiency range must satisfy 0 < min <= max <= 1")
        if self.ponding_reference_depth_m <= 0.0:
            raise ValueError("ponding reference depth must be positive")
        if self.outfall_freeboard_m != 0.0:
            raise ValueError("outfall freeboard is locked at 0.0 m (Z_out = cell bed, V5)")
        for fraction in (self.blocked_fraction_reference, self.blocked_fraction_increased):
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("blocked fractions must lie in [0, 1]")

    @property
    def capacity_min_m_per_s(self) -> float:
        return self.capacity_min_mm_per_h * MM_PER_H_TO_M_PER_S

    @property
    def capacity_max_m_per_s(self) -> float:
        return self.capacity_max_mm_per_h * MM_PER_H_TO_M_PER_S


@dataclass(frozen=True)
class ExchangeConfig:
    """Linear head-difference exchange model parameters (E4 of V7).

    c is an empirical modelling choice, selected by nothing beyond the two
    declared constraints of V6: the f_step design bound below and a
    minutes-timescale relaxation behaviour. f_step_max is a DECLARED
    NUMERICAL DESIGN BOUND, not a stability proof; empirical behaviour is
    bounded only by TC8, TC10, and TC12. c must never be tuned to pass the
    dt sensitivity test; if that test fails, dt is halved, never c changed.
    """

    exchange_velocity_m_per_s: float = 0.1
    timestep_s: float = 30.0
    sensitivity_timestep_s: float = 15.0
    film_threshold_depth_m: float = 0.005
    declared_f_step_max: float = 0.20

    def __post_init__(self) -> None:
        if self.exchange_velocity_m_per_s <= 0.0:
            raise ValueError("exchange velocity c must be positive")
        if self.timestep_s <= 0.0 or self.sensitivity_timestep_s <= 0.0:
            raise ValueError("time steps must be positive")
        if self.sensitivity_timestep_s >= self.timestep_s:
            raise ValueError("the sensitivity timestep must be smaller than the default")
        if self.film_threshold_depth_m <= 0.0:
            raise ValueError("film threshold depth must be positive")
        if not 0.0 < self.declared_f_step_max <= 1.0:
            raise ValueError("declared f_step bound must lie in (0, 1]")

    def f_step(self, dt_s: float, cell_size_m: float) -> float:
        """Pairwise head-difference decay fraction per step: 2 * c * dt / dx."""
        return 2.0 * self.exchange_velocity_m_per_s * dt_s / cell_size_m

    def validate_against_grid(self, grid: GridConfig) -> None:
        for dt in (self.timestep_s, self.sensitivity_timestep_s):
            f_step = self.f_step(dt, grid.cell_size_m)
            if f_step > self.declared_f_step_max:
                raise ValueError(
                    f"declared design bound violated: f_step = {f_step:.4f} at dt = {dt} s"
                    f" exceeds {self.declared_f_step_max}"
                )


@dataclass(frozen=True)
class SusceptibilityConfig:
    """Static weighted susceptibility index (V5 table T1, unchanged in V7).

    Normalization ranges and weights are declared a priori constants,
    identical across seeds, and are NEVER fitted to simulator output or any
    other label source. The feature builder consumes terrain, land cover,
    and water-distance fields only; drainage capacity, drainage efficiency,
    blocked status, outfall elevations, and simulated depths are
    unreachable from the feature function by construction (TC4).
    Note the disclosed clip (V7 weakness W10): terrain reaches 60 m while
    the declared elevation transform range ends at 40 m, so high inland
    nodes clip to score 0 and clip fractions are reported, not refit.
    """

    elevation_range_m: tuple[float, float] = (0.0, 40.0)
    tpi_radius_m: float = 200.0
    tpi_span_m: float = 10.0
    water_distance_max_m: float = 1000.0
    slope_max: float = 0.3
    runoff_index_impervious: float = 1.0
    runoff_index_open_ground: float = 0.6
    runoff_index_vegetation: float = 0.25
    runoff_index_water: float = 1.0
    weight_elevation: float = 0.35
    weight_tpi: float = 0.25
    weight_water_distance: float = 0.15
    weight_land_cover: float = 0.15
    weight_slope: float = 0.10

    def __post_init__(self) -> None:
        low, high = self.elevation_range_m
        if not 0.0 <= low < high:
            raise ValueError("elevation transform range must satisfy 0 <= low < high")
        if self.tpi_radius_m <= 0.0 or self.tpi_span_m <= 0.0:
            raise ValueError("TPI radius and span must be positive")
        if self.water_distance_max_m <= 0.0 or self.slope_max <= 0.0:
            raise ValueError("water distance cap and slope cap must be positive")
        weights = (
            self.weight_elevation,
            self.weight_tpi,
            self.weight_water_distance,
            self.weight_land_cover,
            self.weight_slope,
        )
        if any(w < 0.0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("susceptibility weights must be non-negative and sum to 1.0")

    @property
    def weight_sum(self) -> float:
        return (
            self.weight_elevation
            + self.weight_tpi
            + self.weight_water_distance
            + self.weight_land_cover
            + self.weight_slope
        )


@dataclass(frozen=True)
class RoadConfig:
    """Road graph and routing parameters (V5, carried into V7).

    Terminology per V5 correction 4: an edge is CLOSED when its effective
    depth d_e >= closure_threshold and closed edges are REMOVED from the
    routing graph copy before path search; removal is the implementation of
    closure and no soft penalty applies to a closed edge.
    """

    arterial_speed_kmh: float = 50.0
    secondary_speed_kmh: float = 30.0
    local_speed_kmh: float = 20.0
    closure_threshold_default_m: float = 0.30
    closure_thresholds_sweep_m: tuple[float, ...] = (0.20, 0.30, 0.45)
    label_threshold_default_m: float = 0.10
    label_thresholds_sweep_m: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    lambda_sweep: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
    edge_depth_samples: int = 3
    n_od_pairs: int = 20

    def __post_init__(self) -> None:
        for name in ("arterial", "secondary", "local"):
            speed = getattr(self, f"{name}_speed_kmh")
            if speed <= 0.0:
                raise ValueError(f"{name} speed must be positive")
        if self.closure_threshold_default_m not in self.closure_thresholds_sweep_m:
            raise ValueError("default closure threshold must appear in its sweep")
        if self.label_threshold_default_m not in self.label_thresholds_sweep_m:
            raise ValueError("default label threshold must appear in its sweep")
        if self.edge_depth_samples != 3:
            raise ValueError("edge depth sampling is locked to endpoints and midpoint (3 samples)")
        if self.n_od_pairs <= 0:
            raise ValueError("OD pair count must be positive")
        if any(lam < 0.0 for lam in self.lambda_sweep):
            raise ValueError("lambda sweep values must be non-negative")


@dataclass(frozen=True)
class SeedConfig:
    """Fixed, configurable master seed with independent named streams.

    Stream roles (V4 provenance split, unchanged in V7), enforced
    structurally by TC2a/TC2b in later phases, never by correlation:

      env        terrain, land cover, water bodies, road geometry
      drainage   hidden drainage properties only; generator receives no terrain
      weather    scenario hyetographs and tide parameters
      routing    OD-pair selection

    Experiment A environment seeds are sub-streams of the env stream.
    """

    master_seed: int = 2027

    def __post_init__(self) -> None:
        if self.master_seed < 0:
            raise ValueError("master seed must be non-negative")

    def stream_seed(self, stream_name: str) -> int:
        return derive_stream_seed(self.master_seed, stream_name)

    def env_experiment_seeds(self, n: int = 5) -> tuple[int, ...]:
        """n independent environment seeds for Experiment A (5 in the MVP)."""
        if n <= 0:
            raise ValueError("experiment seed count must be positive")
        return tuple(self.stream_seed(f"{STREAM_ENV}/experiment/{k}") for k in range(n))


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment scopes (V1-V6, unchanged in V7)."""

    reference_scenario_name: str = REFERENCE_SCENARIO_NAME
    n_env_seeds: int = 5
    externals: tuple[str, ...] = ("R", "S1", "S2", "S3")

    def __post_init__(self) -> None:
        if self.n_env_seeds <= 0:
            raise ValueError("environment seed count must be positive")
        for name in self.externals:
            if name not in SCENARIOS:
                raise ValueError(f"unknown scenario name: {name}")
        if self.reference_scenario_name not in SCENARIOS:
            raise ValueError(f"unknown reference scenario: {self.reference_scenario_name}")


@dataclass(frozen=True)
class ProjectConfig:
    """Assembled SYNTHETIC project configuration (Architecture V7)."""

    grid: GridConfig
    elevation: ElevationConfig
    seeds: SeedConfig
    runoff: RunoffConfig
    drainage: DrainageConfig
    exchange: ExchangeConfig
    susceptibility: SusceptibilityConfig
    routing: RoadConfig
    experiments: ExperimentConfig

    @property
    def scenarios(self) -> MappingProxyType:
        return MappingProxyType(SCENARIOS)

    @property
    def reference_scenario(self) -> FloodScenario:
        return SCENARIOS[self.experiments.reference_scenario_name]


def build_default_config() -> ProjectConfig:
    """Instantiate the locked V7 configuration and validate assembly bounds."""
    config = ProjectConfig(
        grid=GridConfig(),
        elevation=ElevationConfig(),
        seeds=SeedConfig(),
        runoff=RunoffConfig(),
        drainage=DrainageConfig(),
        exchange=ExchangeConfig(),
        susceptibility=SusceptibilityConfig(),
        routing=RoadConfig(),
        experiments=ExperimentConfig(),
    )
    config.exchange.validate_against_grid(config.grid)
    return config
