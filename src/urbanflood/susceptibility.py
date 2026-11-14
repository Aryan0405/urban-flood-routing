"""SYNTHETIC leakage-free susceptibility index (Phase 3).

Implements a static weighted susceptibility index from terrain, land cover,
and water masks only. This module never imports or reads drainage fields,
outfall data, simulated depths, H_peak, the simulator, the orchestrator,
initial_state, or roads. It is a modelling construct, not a flood prediction
model, and makes no claim about real flood behaviour.

Specification: docs/PHASE3_SUSCEPTIBILITY_SPEC.md (adopted (author-delegated, Stage 12)).
Architecture: V7 (approved).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from urbanflood.config import SusceptibilityConfig


@dataclass(frozen=True)
class SusceptibilityResult:
    """Result of build_susceptibility.

    index : (n_rows, n_cols) float64, NaN on ocean cells, in [0, 1] elsewhere.
    feature_scores : dict name -> (n_rows, n_cols) float64, NaN on ocean cells.
    weights : dict name -> float, the five SusceptibilityConfig weights.
    clip_fractions : dict name -> float, from elevation_score.

    All arrays are fresh copies; callers may not mutate internal state through
    them (they are not views into env arrays).
    """

    index: np.ndarray
    feature_scores: dict[str, np.ndarray]
    weights: dict[str, float]
    clip_fractions: dict[str, float]


def _feature_elevation(elevation_m: np.ndarray, is_ocean: np.ndarray) -> np.ndarray:
    """Elevation field with ocean cells set to 0.0 (MSL)."""
    return np.where(is_ocean, 0.0, elevation_m)


def _scored_mask(is_ocean: np.ndarray) -> np.ndarray:
    """Mask of scored cells: all non-ocean cells."""
    return ~is_ocean


def _nan_on_ocean(score: np.ndarray, is_ocean: np.ndarray) -> np.ndarray:
    """Return a fresh copy of score with ocean cells set to NaN."""
    result = score.astype(np.float64, copy=True)
    result[is_ocean] = np.nan
    return result


def elevation_score(
    elevation_m: np.ndarray,
    is_ocean: np.ndarray,
    cfg: SusceptibilityConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Elevation score: 1 - clip((z - low) / (high - low), 0, 1).

    Ocean cells are NaN. Reports clip fractions: share of scored cells with
    z above the upper bound (score 0) and below the lower bound (score 1).
    """
    low, high = cfg.elevation_range_m
    z = _feature_elevation(elevation_m, is_ocean)
    scored = _scored_mask(is_ocean)
    z_scored = z[scored]

    if z_scored.size > 0:
        clip_above = float(np.mean(z_scored > high))
        clip_below = float(np.mean(z_scored < low))
    else:
        clip_above = 0.0
        clip_below = 0.0

    score = np.full(z.shape, np.nan, dtype=np.float64)
    score[scored] = 1.0 - np.clip((z_scored - low) / (high - low), 0.0, 1.0)

    clip_info = {
        "clip_above_upper_bound": clip_above,
        "clip_below_lower_bound": clip_below,
    }
    return score, clip_info


def tpi_score(
    elevation_m: np.ndarray,
    is_ocean: np.ndarray,
    cell_size_m: float,
    cfg: SusceptibilityConfig,
) -> np.ndarray:
    """TPI score: clip((tpi_span - TPI) / (2 * tpi_span), 0, 1).

    TPI = z minus mean z in a circular window of radius tpi_radius_m. The
    window includes the centre cell. Edges are handled by reflect padding.
    The circular footprint radius in cells is ceil(tpi_radius_m / cell_size_m).
    """
    z = _feature_elevation(elevation_m, is_ocean)
    scored = _scored_mask(is_ocean)

    radius_cells = int(np.ceil(cfg.tpi_radius_m / cell_size_m))
    yy, xx = np.ogrid[
        -radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1
    ]
    footprint = (xx**2 + yy**2) <= radius_cells**2
    footprint = footprint.astype(np.float64)
    footprint_sum = float(footprint.sum())

    z_sum = ndimage.convolve(z, footprint, mode="reflect")
    mean_z = z_sum / footprint_sum
    tpi = z - mean_z

    score = np.full(z.shape, np.nan, dtype=np.float64)
    score[scored] = np.clip(
        (cfg.tpi_span_m - tpi[scored]) / (2.0 * cfg.tpi_span_m), 0.0, 1.0
    )
    return score


def water_distance_score(
    is_ocean: np.ndarray,
    is_lake: np.ndarray,
    cell_size_m: float,
    cfg: SusceptibilityConfig,
) -> np.ndarray:
    """Water distance score: 1 - clip(d / water_distance_max_m, 0, 1).

    d = Euclidean distance in metres to the nearest ocean-or-lake cell,
    computed with scipy.ndimage.distance_transform_edt using
    sampling = cell_size_m. Ocean cells are NaN.
    """
    water_mask = is_ocean | is_lake
    # distance_transform_edt computes distance to nearest zero (False) pixel.
    distance = ndimage.distance_transform_edt(~water_mask, sampling=cell_size_m)

    score = np.full(distance.shape, np.nan, dtype=np.float64)
    scored = _scored_mask(is_ocean)
    score[scored] = 1.0 - np.clip(
        distance[scored] / cfg.water_distance_max_m, 0.0, 1.0
    )
    return score


def land_cover_score(
    land_cover: np.ndarray,
    cfg: SusceptibilityConfig,
) -> np.ndarray:
    """Land cover score: lookup of runoff_index_* by class code.

    Codes: 0=impervious, 1=open_ground, 2=vegetation, 3=water. Ocean cells
    are NaN (handled by the caller via _nan_on_ocean).
    """
    lookup = np.array(
        [
            cfg.runoff_index_impervious,
            cfg.runoff_index_open_ground,
            cfg.runoff_index_vegetation,
            cfg.runoff_index_water,
        ],
        dtype=np.float64,
    )
    return lookup[land_cover.astype(np.int64)]


def slope_score(
    elevation_m: np.ndarray,
    is_ocean: np.ndarray,
    cell_size_m: float,
    cfg: SusceptibilityConfig,
) -> np.ndarray:
    """Slope score: 1 - clip(slope / slope_max, 0, 1).

    Slope = gradient magnitude from np.gradient of the feature elevation
    (rise over run, dimensionless, spacing = cell_size_m). Ocean cells are NaN.
    """
    z = _feature_elevation(elevation_m, is_ocean)
    scored = _scored_mask(is_ocean)

    gx, gy = np.gradient(z, cell_size_m)
    slope = np.sqrt(gx**2 + gy**2)

    score = np.full(z.shape, np.nan, dtype=np.float64)
    score[scored] = 1.0 - np.clip(slope[scored] / cfg.slope_max, 0.0, 1.0)
    return score


def build_susceptibility(
    env: object,
    cfg: SusceptibilityConfig | None = None,
) -> SusceptibilityResult:
    """Build the full susceptibility index for one environment.

    Reads ONLY env.elevation_m, env.is_ocean, env.is_lake, env.land_cover,
    and env.grid.cell_size_m. Does not import or read drainage, outfalls,
    simulated depths, H_peak, initial_state, simulator, orchestrator, or
    roads.

    Parameters
    ----------
    env : object
        SyntheticEnvironment-like object with elevation_m, is_ocean, is_lake,
        land_cover, and grid.cell_size_m attributes.
    cfg : SusceptibilityConfig or None
        Configuration; defaults to SusceptibilityConfig().

    Returns
    -------
    SusceptibilityResult
        Index in [0, 1] on non-ocean cells, NaN on ocean cells. All arrays
        are fresh copies.
    """
    if cfg is None:
        cfg = SusceptibilityConfig()

    elevation_m = env.elevation_m
    is_ocean = env.is_ocean
    is_lake = env.is_lake
    land_cover = env.land_cover
    cell_size_m = env.grid.cell_size_m

    elev_score, clip_info = elevation_score(elevation_m, is_ocean, cfg)
    tpi = tpi_score(elevation_m, is_ocean, cell_size_m, cfg)
    wdist = water_distance_score(is_ocean, is_lake, cell_size_m, cfg)
    lc = land_cover_score(land_cover, cfg)
    slope = slope_score(elevation_m, is_ocean, cell_size_m, cfg)

    # Apply NaN on ocean cells to all feature arrays.
    scored = _scored_mask(is_ocean)
    elev_score = _nan_on_ocean(elev_score, is_ocean)
    tpi = _nan_on_ocean(tpi, is_ocean)
    wdist = _nan_on_ocean(wdist, is_ocean)
    lc = _nan_on_ocean(lc, is_ocean)
    slope = _nan_on_ocean(slope, is_ocean)

    index = np.full(elevation_m.shape, np.nan, dtype=np.float64)
    index[scored] = (
        cfg.weight_elevation * elev_score[scored]
        + cfg.weight_tpi * tpi[scored]
        + cfg.weight_water_distance * wdist[scored]
        + cfg.weight_land_cover * lc[scored]
        + cfg.weight_slope * slope[scored]
    )

    feature_scores = {
        "elevation": elev_score,
        "tpi": tpi,
        "water_distance": wdist,
        "land_cover": lc,
        "slope": slope,
    }

    weights = {
        "elevation": cfg.weight_elevation,
        "tpi": cfg.weight_tpi,
        "water_distance": cfg.weight_water_distance,
        "land_cover": cfg.weight_land_cover,
        "slope": cfg.weight_slope,
    }

    return SusceptibilityResult(
        index=index,
        feature_scores=feature_scores,
        weights=weights,
        clip_fractions=clip_info,
    )
