"""SYNTHETIC hidden drainage layer (Phase 1, TC2a structural independence).

Hidden drainage properties (capacity D_i, efficiency eta_d,i, blocked status
b_i) are generated from the independent "drainage" named stream. Structural
independence is enforced by construction, never by correlation:

  - generate_drainage() receives NO terrain input of any kind (no grid, no
    elevation, no land cover, no outfall data). Its signature is exactly
    (master_seed, n_cells, blocked_fraction, drainage_config) and the test
    suite asserts that signature.
  - Terrain, land cover, and outfall infrastructure never consume the
    drainage stream; the environment builder cannot pass terrain into it.

Draw order within one call is canonical and fixed (changing it would
silently change every environment's drainage):

  1. capacity   rng.uniform(D_min, D_max, size=n_cells)
  2. efficiency rng.uniform(eta_min, eta_max, size=n_cells)
  3. blocked    blocked = rng.random(n_cells) < blocked_fraction

This fixed order makes the S3 manipulation exact by construction: the locked
rule "S3 regenerates the blocked mask on the same drainage stream" is
realised by calling this function with the same master seed and the same
n_cells but blocked_fraction = 0.40 instead of 0.20. Steps 1 and 2 reproduce
the identical capacity and efficiency draws, so capacity, efficiency, and
outfall infrastructure are untouched and only the blocked mask changes.
Because the mask threshold is nested, the 0.20 mask is a subset of the 0.40
mask (more drains blocked, no drain unblocked or swapped) exactly.

Outfall placement is deliberately NOT here: it is terrain-dependent
deterministic infrastructure (land cells with a face on the ocean; invert at
the cell bed elevation, freeboard locked at 0.0 m) and belongs to the
environment builder, which may see terrain but never hidden drainage
properties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbanflood.arrays import freeze_array
from urbanflood.config import DrainageConfig
from urbanflood.rng import STREAM_DRAINAGE, numpy_rng

DRAW_ORDER = ("capacity", "efficiency", "blocked")


@dataclass(frozen=True)
class DrainageFields:
    """Hidden SYNTHETIC drainage fields for one environment and one mask state.

    Arrays are in the canonical row-major cell order (index = row * n_cols +
    col). Capacity and efficiency are stateless properties of the stream;
    the blocked mask is scenario state (blocked_fraction), regenerated for
    S3 on the same stream by construction (see module docstring).
    """

    capacity_mm_per_h: np.ndarray
    efficiency: np.ndarray
    blocked: np.ndarray
    blocked_fraction: float
    master_seed: int


def generate_drainage(
    master_seed: int,
    n_cells: int,
    blocked_fraction: float,
    drainage_config: DrainageConfig | None = None,
) -> DrainageFields:
    """Generate hidden drainage fields from the independent drainage stream.

    No terrain argument exists by design (TC2a); the generator is
    structurally blind to the environment.
    """
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError("master_seed must be an int")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    if n_cells < 1:
        raise ValueError("n_cells must be a positive integer")
    if not 0.0 <= blocked_fraction <= 1.0:
        raise ValueError("blocked_fraction must lie in [0, 1]")
    if drainage_config is None:
        drainage_config = DrainageConfig()

    # Canonical draw order: capacity, efficiency, blocked (see DRAW_ORDER).
    rng = numpy_rng(master_seed, STREAM_DRAINAGE)
    capacity = rng.uniform(
        drainage_config.capacity_min_mm_per_h,
        drainage_config.capacity_max_mm_per_h,
        size=n_cells,
    )
    efficiency = rng.uniform(
        drainage_config.efficiency_min,
        drainage_config.efficiency_max,
        size=n_cells,
    )
    uniforms = rng.random(n_cells)
    blocked = uniforms < blocked_fraction

    return DrainageFields(
        capacity_mm_per_h=freeze_array(capacity),
        efficiency=freeze_array(efficiency),
        blocked=freeze_array(blocked),
        blocked_fraction=blocked_fraction,
        master_seed=master_seed,
    )
