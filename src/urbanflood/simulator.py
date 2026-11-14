"""SYNTHETIC single-timestep water-exchange update for Phase 2.

Implements a complete, deterministic, single-timestep update integrating:

- Internal surface-water exchange (head-difference flux between 4-connected cells)
- Drainage outflows (capacity * efficiency, bounded by availability)
- Sea-boundary inflows and outflows (tide-driven, at outfall faces)
- A single shared donor-availability limiter across all outgoing water requests
- Synchronous storage updates based on the old-state snapshot

All candidate fluxes are calculated from the old-state snapshot h_now. Rainfall
runoff is added in the final simultaneous update and is available through the
donor budget, but cannot generate new flux proposals during the same timestep.

Specification: docs/PHASE2_NUMERICAL_SPEC.md (approved).
Architecture: V7 (approved).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbanflood.config import (
    ExchangeConfig,
    LandCover,
    RainScenario,
    RunoffConfig,
    TideConfig,
)
from urbanflood.drainage import DrainageFields
from urbanflood.environment import SyntheticEnvironment
from urbanflood.numerics import (
    enumerate_internal_faces,
    signed_flux_to_request,
)
from urbanflood.weather import interval_mean_rate


def enumerate_sea_faces(
    n_rows: int,
    n_cols: int,
    is_ocean: np.ndarray,
) -> list[tuple[int, int]]:
    """Enumerate unique eligible land-ocean sea faces, each exactly once.

    A sea face is a (land_cell_index, ocean_cell_index) pair where the two
    cells are 4-connected neighbours, the land cell is not ocean, and the
    ocean cell is ocean. Faces are enumerated from the land side only, so
    each land-ocean pair appears exactly once. The list is sorted by
    (land_index, ocean_index) for deterministic ordering.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    is_ocean : np.ndarray
        Boolean ocean mask, shape (n_rows, n_cols). True = ocean.

    Returns
    -------
    list of (int, int)
        Sorted unique sea faces as (land_index, ocean_index).
    """
    faces: list[tuple[int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if is_ocean[r, c]:
                continue
            land_idx = r * n_cols + c
            # Check all 4 neighbours in a fixed order.
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < n_rows and 0 <= nc < n_cols and is_ocean[nr, nc]:
                    ocean_idx = nr * n_cols + nc
                    faces.append((land_idx, ocean_idx))
    faces.sort()
    return faces


@dataclass(frozen=True)
class StepResult:
    """Result of one single-timestep water-exchange update.

    All volume terms are in cubic metres (m3) and are summed over all
    simulated surface-water (non-ocean) cells for this step.

    Attributes
    ----------
    h_new : np.ndarray
        Water depths after the step, shape (n_rows, n_cols), metres.
    V_precip_total : float
        Total precipitation volume over all surface-water cells, m3.
    V_runoff_total : float
        Total runoff volume entering storage, m3.
    V_loss_total : float
        Total loss volume (never enters storage), m3.
    V_drain_total : float
        Total drained volume after donor-availability scaling, m3.
    SEA_IN_total : float
        Total positive sea flux into land cells (unscaled, external input), m3.
    SEA_OUT_total : float
        Total magnitude of negative sea flux out of land cells (after scaling), m3.
    internal_inflow_total : float
        Total scaled internal inflow into all cells, m3.
    internal_outflow_total : float
        Total scaled internal outflow from all cells, m3.
    internal_face_eval_count : int
        Number of unique eligible internal faces evaluated this timestep.
        Includes zero-flux faces; excludes sea-boundary faces.
    S_before : float
        Total surface-water storage before the step, m3.
    S_after : float
        Total surface-water storage after the step, m3.
    residual : float
        Per-step mass-balance residual: IN - OUT - dS.
    """

    h_new: np.ndarray
    V_precip_total: float
    V_runoff_total: float
    V_loss_total: float
    V_drain_total: float
    SEA_IN_total: float
    SEA_OUT_total: float
    internal_inflow_total: float
    internal_outflow_total: float
    internal_face_eval_count: int
    S_before: float
    S_after: float
    residual: float


def step_timestep(
    h_now: np.ndarray,
    env: SyntheticEnvironment,
    drainage: DrainageFields,
    rain: RainScenario,
    tide: TideConfig,
    exchange_config: ExchangeConfig,
    runoff_config: RunoffConfig,
    dt: float,
    k: int,
) -> StepResult:
    """Execute one synchronous single-timestep water-exchange update.

    All candidate fluxes are calculated from the old-state snapshot ``h_now``.
    Donor-availability scaling is then applied to all outgoing requests, and
    all transfers are applied synchronously in the final update.

    Parameters
    ----------
    h_now : np.ndarray
        Water depths before the step, shape (n_rows, n_cols), metres.
    env : SyntheticEnvironment
        The synthetic environment (terrain, masks, outfalls).
    drainage : DrainageFields
        Hidden drainage fields (capacity, efficiency, blocked).
    rain : RainScenario
        Rainfall scenario (used for interval integration).
    tide : TideConfig
        Tide configuration (midpoint evaluation).
    exchange_config : ExchangeConfig
        Exchange model parameters (exchange velocity c).
    runoff_config : RunoffConfig
        Runoff coefficients by land cover.
    dt : float
        Timestep in seconds.
    k : int
        Timestep index (0-based), used to compute the rainfall interval
        and midpoint tide.

    Returns
    -------
    StepResult
        New water depths and complete per-step accounting.
    """
    z = env.elevation_m
    is_ocean = env.is_ocean
    is_lake = env.is_lake
    land_cover = env.land_cover
    outfall_mask = env.outfall_mask
    n_rows, n_cols = z.shape
    dx = env.grid.cell_size_m
    A = env.grid.cell_area_m2
    c = exchange_config.exchange_velocity_m_per_s

    # --- Snapshot old state (flat 1-D views for indexed access) ---
    h_old = np.array(h_now, dtype=np.float64, copy=True).ravel()
    V_old = h_old * A  # volume per cell, m3
    S_before = float(V_old[~is_ocean.ravel()].sum())

    # --- Rainfall: exact interval integration from old state ---
    R_k = interval_mean_rate(rain, k * dt, (k + 1) * dt)  # m/s
    V_precip = np.full(n_rows * n_cols, R_k * A * dt, dtype=np.float64)  # m3/cell

    # Runoff coefficient by land cover code:
    #   0 = impervious (0.90), 1 = open_ground (0.50),
    #   2 = vegetation (0.30), 3 = water (1.00)
    alpha_lookup = np.array(
        [
            runoff_config.coefficients[LandCover.IMPERVIOUS],
            runoff_config.coefficients[LandCover.OPEN_GROUND],
            runoff_config.coefficients[LandCover.VEGETATION],
            runoff_config.coefficients[LandCover.WATER],
        ],
        dtype=np.float64,
    )
    alpha_arr = alpha_lookup[land_cover.ravel()]
    V_runoff = alpha_arr * V_precip  # m3/cell, enters storage
    V_loss = V_precip - V_runoff  # m3/cell, never enters storage

    # Ocean cells are excluded from surface-water storage and rainfall input.
    surface_water = ~is_ocean.ravel()
    V_precip = np.where(surface_water, V_precip, 0.0)
    V_runoff = np.where(surface_water, V_runoff, 0.0)
    V_loss = np.where(surface_water, V_loss, 0.0)

    V_precip_total = float(V_precip.sum())
    V_runoff_total = float(V_runoff.sum())
    V_loss_total = float(V_loss.sum())

    # --- Tide: single midpoint evaluation per timestep ---
    eta_k = tide.eta((k + 0.5) * dt)

    # --- Water-surface elevation from old state ---
    W_old = z.ravel() + h_old  # m above MSL

    # --- 1. Internal exchange proposals (from old state) ---
    internal_faces = enumerate_internal_faces(n_rows, n_cols, is_ocean)
    internal_face_eval_count = len(internal_faces)
    internal_proposals: list[tuple[int, int, float]] = []
    for face in internal_faces:
        i, j = face.i, face.j
        delta_W = W_old[i] - W_old[j]
        V_ij = c * delta_W * dx * dt
        req = signed_flux_to_request(V_ij, i, j)
        if req is not None:
            internal_proposals.append((req.donor, req.receiver, req.volume_m3))

    # --- 2. Drainage proposals (from old state) ---
    C = drainage.capacity_mm_per_h.ravel()  # mm/h
    eta_d = drainage.efficiency.ravel()
    blocked = drainage.blocked.ravel()
    V_potential = np.zeros(n_rows * n_cols, dtype=np.float64)
    unblocked = ~blocked
    V_potential[unblocked] = (
        C[unblocked] / (1000.0 * 3600.0)
    ) * A * dt * eta_d[unblocked]

    # --- 3. Sea exchange proposals (from old state) ---
    sea_faces = enumerate_sea_faces(n_rows, n_cols, is_ocean)
    sea_inflow = np.zeros(n_rows * n_cols, dtype=np.float64)  # unscaled, per cell
    sea_outflow_proposed = np.zeros(n_rows * n_cols, dtype=np.float64)  # magnitude, per cell
    SEA_IN_total = 0.0
    SEA_OUT_proposed_total = 0.0

    for land_idx, ocean_idx in sea_faces:
        if not outfall_mask.ravel()[land_idx]:
            continue
        W_i = W_old[land_idx]
        V_sea = c * (eta_k - W_i) * dx * dt
        if V_sea > 0.0:
            # Sea inflow: external input, does not consume donor budget
            sea_inflow[land_idx] += V_sea
            SEA_IN_total += V_sea
        elif V_sea < 0.0:
            # Sea outflow: donor request, will be scaled
            sea_outflow_proposed[land_idx] += -V_sea
            SEA_OUT_proposed_total += -V_sea

    # --- 4. Assemble per-cell proposed outflows (from old state) ---
    internal_outflow_proposed = np.zeros(n_rows * n_cols, dtype=np.float64)
    internal_inflow_proposed = np.zeros(n_rows * n_cols, dtype=np.float64)
    for donor, receiver, volume in internal_proposals:
        internal_outflow_proposed[donor] += volume
        internal_inflow_proposed[receiver] += volume

    Q = (
        internal_outflow_proposed
        + np.where(blocked, 0.0, V_potential)
        + sea_outflow_proposed
    )

    # Donor budget: old storage plus same-timestep runoff
    budget = V_old + V_runoff

    # Donor-availability scale factor (shared across all outgoing request types)
    scale = np.ones(n_rows * n_cols, dtype=np.float64)
    donor_mask = Q > 0.0
    scale[donor_mask] = np.minimum(1.0, budget[donor_mask] / Q[donor_mask])

    # --- 5. Apply scaled transfers ---
    scaled_internal_outflow = np.zeros(n_rows * n_cols, dtype=np.float64)
    scaled_internal_inflow = np.zeros(n_rows * n_cols, dtype=np.float64)
    for donor, receiver, volume in internal_proposals:
        scaled_vol = scale[donor] * volume
        scaled_internal_outflow[donor] += scaled_vol
        scaled_internal_inflow[receiver] += scaled_vol

    scaled_drainage = np.where(blocked, 0.0, scale * V_potential)
    V_drain_total = float(scaled_drainage.sum())

    scaled_sea_outflow = scale * sea_outflow_proposed
    SEA_OUT_total = float(scaled_sea_outflow.sum())

    internal_inflow_total = float(scaled_internal_inflow.sum())
    internal_outflow_total = float(scaled_internal_outflow.sum())

    # --- 6. Synchronous storage update (all fluxes from old state) ---
    V_new = (
        V_old
        + V_runoff
        + scaled_internal_inflow
        + sea_inflow
        - scaled_internal_outflow
        - scaled_sea_outflow
        - scaled_drainage
    )

    # Convert back to depths, ocean cells excluded from storage
    h_new = np.zeros_like(h_old)
    h_new[~is_ocean.ravel()] = V_new[~is_ocean.ravel()] / A
    h_new = h_new.reshape(n_rows, n_cols)

    S_after = float(V_new[~is_ocean.ravel()].sum())

    # --- 7. Per-step mass-balance residual ---
    IN_step = V_precip_total + SEA_IN_total
    OUT_step = V_loss_total + V_drain_total + SEA_OUT_total
    dS = S_after - S_before
    residual = IN_step - OUT_step - dS

    return StepResult(
        h_new=h_new,
        V_precip_total=V_precip_total,
        V_runoff_total=V_runoff_total,
        V_loss_total=V_loss_total,
        V_drain_total=V_drain_total,
        SEA_IN_total=SEA_IN_total,
        SEA_OUT_total=SEA_OUT_total,
        internal_inflow_total=internal_inflow_total,
        internal_outflow_total=internal_outflow_total,
        internal_face_eval_count=internal_face_eval_count,
        S_before=S_before,
        S_after=S_after,
        residual=residual,
    )
