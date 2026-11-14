"""SYNTHETIC environment generation: terrain, lakes, land cover, outfalls.
Phase 1 of Architecture V7.

All fields are generated synthetically from the independent "env" named RNG
stream (rng.py). No real-world dataset of any kind is read at any point. The
builder cannot see hidden drainage properties: those live in drainage.py from
the separate drainage stream (TC2a/TC2b structural independence is asserted by
test, never by correlation).

Canonical RNG draw order for one environment (FIXED; reordering would
silently change every generated environment):

  1. relief field: rng.standard_normal((2, n_rows, n_cols)) shaped by an
     isotropic power-law amplitude mask and inverse FFT (FFT-based synthesis)
  2. lake placement: for each lake in id order, rng.uniform(size=4) per
     attempt (fractional center row, center column, sigma, minimum depth);
     the effective carve depth is a deterministic function of the draw and
     the terrain (submergence requirement, capped at a hard maximum), so no
     extra draws occur; rejected attempts redraw with the same draw shape,
     so stream consumption depends on the seed in a fully deterministic way
  3. road lines: one permutation over horizontal candidates, then one over
     vertical candidates (roads.py docstring)
  4. land cover: rng.random((n_rows, n_cols)), applied row-major

Vertical datum (V7): synthetic MSL = 0.0 m. Land terrain spans
[terrain_min, terrain_max] = [0.2, 60.0] m above MSL. The coastal land band
(rows shore_row-3 .. shore_row) is pinned to [0.2, 1.2] m so that outfall
blocking binds in low cells under R and broadly under S2 (TC13b checks the
pins exactly). Ocean rows hold synthetic bathymetry below MSL.

Ocean cells are the boundary reservoir of the Phase 2 simulator; they are
excluded from the dynamic surface-water ledger. Inland lake cells follow the
user's Option A decision (recorded in config.py): they are ordinary dynamic
surface-water cells. Their normal level is the recorded depression spill
level L_normal (the minimum rim-ring elevation of the carved bowl); Phase 2
sets h_init = max(0, L_normal - z) there, and their initial volume belongs to
the change-relative ledger baseline, never to rainfall input.

Outfalls are terrain-dependent deterministic infrastructure: land cells with
a 4-neighbour face on the ocean. Invert elevation is the cell bed elevation
(freeboard locked at 0.0 m, no 1 m offset).

The land cover classifier uses road proximity (distance transform to road
nodes) and low-lying terrain; it does NOT see drainage, tide, or simulated
depth - susceptibility features come later (Phase 3) from terrain, land
cover, and water distance only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from urbanflood import roads as _roads
from urbanflood.arrays import freeze_array
from urbanflood.config import ElevationConfig, GridConfig, LandCover, ProjectConfig
from urbanflood.rng import STREAM_ENV, numpy_rng

ENV_DIGEST_VERSION = "urbanflood-environment-v1"

# --- generation hyperparameters (Phase 1 design choices, not locked V7 values) ---
SEA_ROWS = 8
COASTAL_BAND_ROWS = 4
TREND_MAX_M = 52.0
TREND_POWER = 1.35
TREND_FRACTION_OFFSET = 0.06
RELIEF_AMPLITUDE_M = 6.0
RELIEF_AMPLITUDE_RAMP_M = 8.0
SPECTRAL_EXPONENT = 1.6
SPECTRAL_SOFT_WAVELENGTH_M = 1600.0
SPECTRAL_CUTOFF_WAVELENGTH_M = 150.0
OCEAN_BED_TOP_M = -1.5
OCEAN_BED_BOTTOM_M = -7.5

LAKE_COUNT = 2
LAKE_SIGMA_MIN_CELLS = 4.0
LAKE_SIGMA_MAX_CELLS = 7.0
LAKE_DEPTH_MIN_M = 3.0
LAKE_DEPTH_MAX_M = 6.0
LAKE_DEPTH_HARD_MAX_M = 14.0
LAKE_WATER_MARGIN_M = 0.5
LAKE_ROW_MIN = 30
LAKE_ROW_MAX_REL_SHORE = 16
LAKE_COL_MIN = 12
LAKE_COL_MAX_REL = 13
LAKE_MIN_SPILL_M = 2.0
LAKE_MIN_CELLS = 12
LAKE_MIN_DEPTH_M = 0.3
LAKE_MAX_ATTEMPTS = 40
LAKE_MIN_CENTER_SEPARATION_CELLS = 30.0

LAND_COVER_CODES = {
    LandCover.IMPERVIOUS: 0,
    LandCover.OPEN_GROUND: 1,
    LandCover.VEGETATION: 2,
    LandCover.WATER: 3,
}
IMPERVIOUS_BASE = 0.30
IMPERVIOUS_ROAD_TERM = 0.30
IMPERVIOUS_ROAD_LENGTH_SCALE_CELLS = 2.5
IMPERVIOUS_LOWLAND_TERM = 0.15
IMPERVIOUS_LOWLAND_SCALE_M = 10.0
IMPERVIOUS_MAX = 0.85
OPEN_GROUND_FRACTION = 0.15


@dataclass(frozen=True)
class LakeSpec:
    """One SYNTHETIC lake (carved depression) and its Option A state."""

    lake_id: int
    spill_level_m: float
    min_bed_m: float
    carved_depth_m: float
    center_row: int
    center_col: int
    sigma_cells: float


@dataclass(frozen=True)
class SyntheticEnvironment:
    """Assembled SYNTHETIC environment for one seed.

    Arrays are row-major with index = row * n_cols + col for cell ordering.
    Callers must treat all arrays as immutable; nothing recomputes them.
    """

    master_seed: int
    grid: GridConfig
    shore_row: int
    elevation_m: np.ndarray
    is_ocean: np.ndarray
    is_lake: np.ndarray
    lake_id: np.ndarray
    lake_specs: tuple[LakeSpec, ...]
    land_cover: np.ndarray
    road_network: _roads.RoadNetwork
    road_cells: np.ndarray
    outfall_mask: np.ndarray
    outfall_invert_m: np.ndarray
    digest: str

    @property
    def is_water(self) -> np.ndarray:
        # Derived on every access; frozen like the stored arrays so the
        # immutability contract is uniform (stored and derived arrays alike
        # must not be mutated by callers).
        return freeze_array(self.is_ocean | self.is_lake)

    def lake_level(self, lake_id: int) -> float:
        for spec in self.lake_specs:
            if spec.lake_id == lake_id:
                return spec.spill_level_m
        raise KeyError(f"unknown lake id: {lake_id}")


def build_environment(master_seed: int, config: ProjectConfig) -> SyntheticEnvironment:
    """Build the full SYNTHETIC environment for one master (or experiment) seed.

    The seed is fed to the "env" stream derivation; Experiment A environments
    use SeedConfig.env_experiment_seeds() values as their seeds.
    """
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError("master_seed must be an int")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")

    grid = config.grid
    e_cfg = config.elevation
    n_rows, n_cols = grid.n_rows, grid.n_cols
    rng = numpy_rng(master_seed, STREAM_ENV)

    shore_row = n_rows - SEA_ROWS - 1
    if shore_row < 4:
        raise ValueError("grid too small for the configured sea band")
    land_mask = np.zeros((n_rows, n_cols), dtype=np.bool_)
    land_mask[: shore_row + 1, :] = True

    # 1. FFT spectral relief field (first and only large draw).
    field = _spectral_relief(rng, n_rows, n_cols, grid.cell_size_m)

    # Regional inland trend with smooth near-shore approach; ocean beds later.
    rows_idx = np.arange(n_rows)
    frac = np.clip((shore_row - rows_idx) / float(shore_row), 0.0, 1.0)
    scaled = ((frac + TREND_FRACTION_OFFSET) / (1.0 + TREND_FRACTION_OFFSET)) ** TREND_POWER
    base = e_cfg.terrain_min_m + (TREND_MAX_M - e_cfg.terrain_min_m) * scaled
    relief_scale = RELIEF_AMPLITUDE_M * np.clip(
        (base - e_cfg.terrain_min_m) / RELIEF_AMPLITUDE_RAMP_M, 0.0, 1.0
    )
    z = base[:, None] + relief_scale[:, None] * field
    z = np.where(land_mask, np.clip(z, e_cfg.terrain_min_m, e_cfg.terrain_max_m), 0.0)
    n_ocean = n_rows - shore_row - 1
    z[shore_row + 1 :, :] = np.linspace(OCEAN_BED_TOP_M, OCEAN_BED_BOTTOM_M, n_ocean)[:, None]

    # Coastal band pin: rows shore-3 .. shore on [0.2, 1.2] m (locked).
    band = np.linspace(
        e_cfg.coastal_band_max_m, e_cfg.coastal_band_min_m, COASTAL_BAND_ROWS
    )
    z[shore_row - COASTAL_BAND_ROWS + 1 : shore_row + 1, :] = band[:, None]

    # 2. Lakes: carved depressions with recorded spill level (Option A).
    is_lake = np.zeros((n_rows, n_cols), dtype=np.bool_)
    lake_id = np.full((n_rows, n_cols), -1, dtype=np.int16)
    lake_specs: list[LakeSpec] = []
    lake_row_max = shore_row - LAKE_ROW_MAX_REL_SHORE
    lake_row_span = lake_row_max - LAKE_ROW_MIN + 1
    lake_col_span = n_cols - LAKE_COL_MAX_REL - LAKE_COL_MIN
    if lake_row_span < 2 or lake_col_span < 2:
        raise ValueError("grid too small for the configured lake placement ranges")
    for lake_index in range(LAKE_COUNT):
        for _attempt in range(LAKE_MAX_ATTEMPTS):
            frac_row, frac_col, frac_sigma, frac_depth = rng.uniform(size=4)
            center_row = LAKE_ROW_MIN + int(frac_row * lake_row_span)
            center_col = LAKE_COL_MIN + int(frac_col * lake_col_span)
            sigma = LAKE_SIGMA_MIN_CELLS + frac_sigma * (
                LAKE_SIGMA_MAX_CELLS - LAKE_SIGMA_MIN_CELLS
            )
            depth = LAKE_DEPTH_MIN_M + frac_depth * (
                LAKE_DEPTH_MAX_M - LAKE_DEPTH_MIN_M
            )
            if center_row + 2.35 * sigma > shore_row - 1:
                continue
            if center_row - 2.35 * sigma < 2:
                continue
            # Audit guard (user review, Phase 1): the full bowl footprint must
            # sit strictly inside the grid so no lake can be clipped at a
            # lateral boundary and every rim ring is real terrain.
            if center_col - 2.35 * sigma < 1.0 or center_col + 2.35 * sigma > n_cols - 2.0:
                continue
            if any(
                (center_row - spec.center_row) ** 2
                + (center_col - spec.center_col) ** 2
                < LAKE_MIN_CENTER_SEPARATION_CELLS**2
                for spec in lake_specs
            ):
                continue
            result = _try_carve_depression(
                z, land_mask, center_row, center_col, sigma, depth, e_cfg
            )
            if result is None:
                continue
            z, comp, spill, bottom, carved = result
            is_lake |= comp
            lake_id[comp] = lake_index
            lake_specs.append(
                LakeSpec(
                    lake_id=lake_index,
                    spill_level_m=spill,
                    min_bed_m=bottom,
                    carved_depth_m=carved,
                    center_row=center_row,
                    center_col=center_col,
                    sigma_cells=sigma,
                )
            )
            break
        else:
            raise RuntimeError(
                f"could not place lake {lake_index} deterministically in "
                f"{LAKE_MAX_ATTEMPTS} attempts"
            )

    is_ocean = np.zeros((n_rows, n_cols), dtype=np.bool_)
    is_ocean[shore_row + 1 :, :] = True

    # 3. Roads (two RNG permutations consumed inside roads.py).
    lake_levels = {spec.lake_id: spec.spill_level_m for spec in lake_specs}
    network = _roads.build_road_network(
        rng,
        elevation_m=z,
        is_lake=is_lake,
        lake_id=lake_id,
        lake_levels=lake_levels,
        shore_row=shore_row,
        cell_size_m=grid.cell_size_m,
        road_cfg=config.routing,
        elevation_cfg=e_cfg,
    )

    # 4. Land cover (uniforms drawn last, in canonical row-major order).
    road_cells = np.zeros((n_rows, n_cols), dtype=np.bool_)
    road_cells[network.node_rows.astype(np.int64), network.node_cols.astype(np.int64)] = True
    distance_to_road = ndimage.distance_transform_edt(~road_cells)
    lowland = 1.0 - np.clip(
        (z - e_cfg.terrain_min_m) / IMPERVIOUS_LOWLAND_SCALE_M, 0.0, 1.0
    )
    p_impervious = np.clip(
        IMPERVIOUS_BASE
        + IMPERVIOUS_ROAD_TERM
        * np.exp(-distance_to_road / IMPERVIOUS_ROAD_LENGTH_SCALE_CELLS)
        + IMPERVIOUS_LOWLAND_TERM * lowland,
        0.0,
        IMPERVIOUS_MAX,
    )
    draws = rng.random((n_rows, n_cols))
    land = land_mask & ~is_lake
    is_impervious = land & (draws < p_impervious)
    is_open = land & ~is_impervious & (draws < p_impervious + OPEN_GROUND_FRACTION)
    cover = np.full((n_rows, n_cols), LAND_COVER_CODES[LandCover.VEGETATION], dtype=np.uint8)
    cover[is_impervious] = LAND_COVER_CODES[LandCover.IMPERVIOUS]
    cover[is_open] = LAND_COVER_CODES[LandCover.OPEN_GROUND]
    cover[is_ocean | is_lake] = LAND_COVER_CODES[LandCover.WATER]

    # Outfalls: land cells with a 4-neighbour face on the ocean; invert = bed.
    outfall = np.zeros((n_rows, n_cols), dtype=np.bool_)
    outfall[:-1, :] |= is_ocean[1:, :]
    outfall[1:, :] |= is_ocean[:-1, :]
    outfall[:, :-1] |= is_ocean[:, 1:]
    outfall[:, 1:] |= is_ocean[:, :-1]
    outfall &= ~is_ocean & ~is_lake
    invert = np.where(outfall, z, 0.0)

    # Audit correction (user review, Phase 1): every exposed array is made
    # read-only in place before assembly; all construction writes are
    # complete by this point and no copies are made.
    z = freeze_array(z)
    is_ocean = freeze_array(is_ocean)
    is_lake = freeze_array(is_lake)
    lake_id = freeze_array(lake_id)
    road_cells = freeze_array(road_cells)
    cover = freeze_array(cover)
    outfall = freeze_array(outfall)
    invert = freeze_array(invert)

    digest = _environment_digest(
        master_seed,
        shore_row,
        _digest_arrays(z, is_ocean, is_lake, lake_id, cover, outfall, invert, network),
        lake_specs,
    )

    return SyntheticEnvironment(
        master_seed=master_seed,
        grid=grid,
        shore_row=shore_row,
        elevation_m=z,
        is_ocean=is_ocean,
        is_lake=is_lake,
        lake_id=lake_id,
        lake_specs=tuple(lake_specs),
        land_cover=cover,
        road_network=network,
        road_cells=road_cells,
        outfall_mask=outfall,
        outfall_invert_m=invert,
        digest=digest,
    )


def _digest_arrays(
    elevation_m: np.ndarray,
    is_ocean: np.ndarray,
    is_lake: np.ndarray,
    lake_id: np.ndarray,
    land_cover: np.ndarray,
    outfall_mask: np.ndarray,
    outfall_invert_m: np.ndarray,
    network: _roads.RoadNetwork,
) -> list[np.ndarray]:
    """Canonical digest array list (order is part of the digest definition)."""
    return [
        elevation_m,
        is_ocean,
        is_lake,
        lake_id,
        land_cover,
        outfall_mask,
        outfall_invert_m,
        network.node_rows,
        network.node_cols,
        network.node_elevation_m,
        network.edge_u,
        network.edge_v,
        network.edge_class,
        network.edge_length_m,
        network.edge_free_flow_time_s,
        network.edge_is_bridge,
        network.edge_deck_m,
    ]


def _environment_digest(
    master_seed: int,
    shore_row: int,
    arrays: list[np.ndarray],
    lake_specs: list[LakeSpec],
) -> str:
    """SHA-256 over the canonical array sequence in ENV_DIGEST_VERSION order."""
    hasher = hashlib.sha256()
    hasher.update(ENV_DIGEST_VERSION.encode("ascii"))
    hasher.update(f"/{master_seed}/{shore_row}".encode("ascii"))
    for array in arrays:
        hasher.update(str(np.ascontiguousarray(array).shape).encode("ascii"))
        hasher.update(np.ascontiguousarray(array).tobytes())
    hasher.update(str(len(arrays)).encode("ascii"))
    for spec in lake_specs:
        hasher.update(
            f"|lake/{spec.lake_id}/{spec.spill_level_m:.17g}/{spec.min_bed_m:.17g}/"
            f"{spec.carved_depth_m:.17g}/{spec.center_row}/{spec.center_col}/"
            f"{spec.sigma_cells:.17g}".encode("ascii")
        )
    return hasher.hexdigest()


def _spectral_relief(rng, n_rows: int, n_cols: int, cell_size_m: float) -> np.ndarray:
    """FFT-based smooth synthetic relief field, zero mean and unit std.

    White Gaussian noise is coloured by an isotropic power-law amplitude
    mask: features below SPECTRAL_CUTOFF_WAVELENGTH_M are removed and the
    spectrum rolls off around SPECTRAL_SOFT_WAVELENGTH_M. The k = 0 mode is
    zeroed; final standardisation removes any residual mean.
    """
    white = rng.standard_normal((2, n_rows, n_cols))
    fx = 2.0 * np.pi * np.fft.fftfreq(n_cols, d=cell_size_m)
    fy = 2.0 * np.pi * np.fft.fftfreq(n_rows, d=cell_size_m)
    k_mag = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    k_soft = 2.0 * np.pi / SPECTRAL_SOFT_WAVELENGTH_M
    k_cut = 2.0 * np.pi / SPECTRAL_CUTOFF_WAVELENGTH_M
    amplitude = np.where(
        (k_mag > 0.0) & (k_mag <= k_cut),
        (k_mag + k_soft) ** (-SPECTRAL_EXPONENT),
        0.0,
    )
    field = np.fft.ifft2(np.fft.fft2(white[0] + 1j * white[1]) * amplitude).real
    std = field.std()
    if not std > 0.0:
        raise RuntimeError("degenerate spectral relief field (zero variance)")
    return (field - field.mean()) / std


def _try_carve_depression(
    z: np.ndarray,
    land_mask: np.ndarray,
    center_row: int,
    center_col: int,
    sigma_cells: float,
    min_depth_m: float,
    e_cfg: ElevationConfig,
):
    """Attempt one lake carve. Returns (z_new, component_mask, spill, bottom,
    carved_depth) or None when the draw fails acceptance.

    The carve bump has compact support r < 2*sigma (a Gaussian times a
    quartic taper vanishing at exactly 2*sigma), so the entire annulus
    r >= 2*sigma is untouched and the spill level is the minimum over the
    FULL non-carve rim ring [2.0*sigma, 2.35*sigma]. That coverage is
    deliberate (Phase 1 audit correction): the annulus between the carve
    edge and a ring starting above it could otherwise dip below the
    recorded spill and would read as submerged surrounding land. With the
    full-annulus minimum, every surrounding rim cell sits at or above the
    recorded spill level by construction.

    Depth adaptivity (deterministic, no extra RNG draws): the drawn depth is
    a MINIMUM. The effective carve depth is the larger of the draw and the
    submergence requirement (highest pre-carve bowl interior plus a
    LAKE_WATER_MARGIN_M above the recorded spill level), capped at
    LAKE_DEPTH_HARD_MAX_M. Without this, lateral relief variation across the
    bowl span routinely exceeds any fixed small depth and the basin would
    stay above its own spill level. A deep carve clipping to the terrain
    floor is acceptable and produces a flat-bedded basin; acceptance then
    still guarantees real water (spill minus bed at least LAKE_MIN_DEPTH_M).

    The lake is the 4-connected component (containing the bowl center) of
    cells below the spill level; disconnected sub-spill pockets are left as
    land (documented honestly).
    """
    n_rows, n_cols = z.shape
    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
    r = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    outside_r = 2.0 * sigma_cells
    inside = r < outside_r
    ring = (r >= 2.0 * sigma_cells) & (r <= 2.35 * sigma_cells)
    spill = float(z[ring].min())
    if spill < LAKE_MIN_SPILL_M:
        return None
    carved_depth = min(
        max(min_depth_m, float(z[inside].max()) - spill + LAKE_WATER_MARGIN_M),
        LAKE_DEPTH_HARD_MAX_M,
    )
    taper = (1.0 - (r / outside_r) ** 2) ** 2
    carve = carved_depth * np.exp(-0.5 * (r / sigma_cells) ** 2) * np.where(inside, taper, 0.0)
    z_new = z - carve
    z_new = np.where(land_mask, np.clip(z_new, e_cfg.terrain_min_m, e_cfg.terrain_max_m), z_new)
    below = inside & (z_new < spill)
    labels, _ = ndimage.label(below)
    root = labels[center_row, center_col]
    if root == 0:
        return None
    component = labels == root
    if int(component.sum()) < LAKE_MIN_CELLS:
        return None
    bottom = float(z_new[component].min())
    if spill - bottom < LAKE_MIN_DEPTH_M:
        return None
    return z_new, component, spill, bottom, carved_depth
