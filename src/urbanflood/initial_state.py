"""SYNTHETIC Option A initial-depth builder for the Phase 2 flood simulator.

Implements the documented Option A initial state (docs/PHASE2_NUMERICAL_SPEC.md
section 4): inland lake cells start at h_init = max(0, spill_level - elevation)
where spill_level is the recorded LakeSpec.spill_level_m; non-lake land cells
start at 0.0; ocean cells are set to 0.0 (the ocean is the boundary reservoir
excluded from the dynamic surface-water ledger, so its array entry carries no
ledger-relevant storage).

This builder is a Phase 2 simulator INPUT only. It must NOT be imported by the
susceptibility layer (Phase 3): susceptibility must never see simulated depths
or lake spill levels.

No existing module is modified. run_simulation keeps its caller-supplied
h_initial signature, and the environment digest is unchanged.
"""

from __future__ import annotations

import numpy as np

from urbanflood.environment import SyntheticEnvironment


def build_initial_depth(env: SyntheticEnvironment) -> np.ndarray:
    """Build the Option A initial water-depth array for one environment.

    Returns a fresh float64 array of shape (n_rows, n_cols). Lake cells receive
    max(0, spill_level[lake_id] - elevation_m); every other cell (non-lake
    land and ocean) receives 0.0. No per-cell Python loops: the computation is
    vectorized over NumPy masks. The input env is not mutated.

    Parameters
    ----------
    env : SyntheticEnvironment
        The synthetic environment (terrain, lake masks, lake specs).

    Returns
    -------
    np.ndarray
        Initial water depths in metres, shape (n_rows, n_cols), dtype float64,
        all values >= 0.0.
    """
    n_rows, n_cols = env.elevation_m.shape
    h = np.zeros((n_rows, n_cols), dtype=np.float64)

    lake_mask = env.is_lake
    if not np.any(lake_mask):
        return h

    # Map each lake cell to its recorded spill level. Loop over lake specs
    # (a small fixed count, not per-cell), then vectorize over cells.
    spill_at_cell = np.zeros((n_rows, n_cols), dtype=np.float64)
    for spec in env.lake_specs:
        spill_at_cell[env.lake_id == spec.lake_id] = spec.spill_level_m

    lake_spill = spill_at_cell[lake_mask]
    lake_z = env.elevation_m[lake_mask]
    h[lake_mask] = np.maximum(0.0, lake_spill - lake_z)

    return h
