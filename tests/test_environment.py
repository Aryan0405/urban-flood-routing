"""Phase 1 tests for the SYNTHETIC environment builder (environment.py).

Covers TC1 (bitwise determinism of generation), TC13b (datum consistency),
the Option A waterbody invariants, land cover validity, and outfall rules.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from urbanflood.config import build_default_config
from urbanflood.environment import (
    COASTAL_BAND_ROWS,
    SEA_ROWS,
    build_environment,
)
from urbanflood.rng import STREAM_ENV

MASTER = 2027
CONFIG = build_default_config()
ENV_SEEDS = tuple([MASTER] + list(CONFIG.seeds.env_experiment_seeds()))


@pytest.fixture(scope="module")
def env():
    return build_environment(MASTER, CONFIG)


# --- TC1: determinism -------------------------------------------------------

def test_tc1_bitwise_determinism_same_seed():
    first = build_environment(MASTER, CONFIG)
    second = build_environment(MASTER, CONFIG)
    assert first.digest == second.digest
    assert first.elevation_m.tobytes() == second.elevation_m.tobytes()
    assert first.land_cover.tobytes() == second.land_cover.tobytes()
    assert first.road_network.edge_u.tobytes() == second.road_network.edge_u.tobytes()
    assert first.road_network.edge_deck_m.tobytes() == second.road_network.edge_deck_m.tobytes()


def test_tc1_determinism_experiment_env_seed():
    experiment_seed = CONFIG.seeds.env_experiment_seeds()[0]
    assert experiment_seed != MASTER
    first = build_environment(experiment_seed, CONFIG)
    second = build_environment(experiment_seed, CONFIG)
    assert first.digest == second.digest
    assert first.elevation_m.tobytes() == second.elevation_m.tobytes()


def test_digest_changes_with_seed(env):
    other = build_environment(MASTER + 1, CONFIG)
    assert other.digest != env.digest
    assert other.elevation_m.tobytes() != env.elevation_m.tobytes()


def test_build_rejects_bad_seeds():
    with pytest.raises(TypeError):
        build_environment(2027.0, CONFIG)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_environment(-1, CONFIG)


def test_environment_stream_membership():
    """The env build must consume ONLY the env stream; TC2b complement.

    Drainage and weather stream generators are untouched by an env build.

    Note: comparing bit_generator.state of two freshly derived generators is
    vacuous here -- stream derivation is a pure function of (master seed,
    stream name), so fresh generators always start in the identical state no
    matter what the build consumed. The check below instead observes which
    streams the builder actually requests while constructing the environment.
    """
    import urbanflood.environment as env_module

    requested: list[str] = []
    real_numpy_rng = env_module.numpy_rng

    def spy(master_seed: int, stream_name: str):
        requested.append(stream_name)
        return real_numpy_rng(master_seed, stream_name)

    env_module.numpy_rng = spy
    try:
        build_environment(MASTER, CONFIG)
    finally:
        env_module.numpy_rng = real_numpy_rng

    assert requested == [STREAM_ENV], (
        f"env build must request only the env stream, requested: {requested!r}"
    )


# --- TC13b: datum consistency ----------------------------------------------

def test_tc13b_land_terrain_range(env):
    land = ~env.is_ocean
    z_land = env.elevation_m[land]
    assert z_land.min() >= CONFIG.elevation.terrain_min_m - 1e-12
    assert z_land.max() <= CONFIG.elevation.terrain_max_m + 1e-12


def test_tc13b_coastal_band_pin_exact(env):
    band = env.elevation_m[
        env.shore_row - COASTAL_BAND_ROWS + 1 : env.shore_row + 1, :
    ]
    expected = np.linspace(
        CONFIG.elevation.coastal_band_max_m,
        CONFIG.elevation.coastal_band_min_m,
        COASTAL_BAND_ROWS,
    )
    for row in range(COASTAL_BAND_ROWS):
        assert np.allclose(
            band[row, :], np.full(env.grid.n_cols, expected[row]), rtol=0, atol=0
        )
        assert band[row, :].min() >= CONFIG.elevation.coastal_band_min_m
        assert band[row, :].max() <= CONFIG.elevation.coastal_band_max_m


def test_tc13b_ocean_beds_below_msl(env):
    beds = env.elevation_m[env.is_ocean]
    assert beds.max() < 0.0  # below synthetic MSL = 0.0
    assert beds.min() >= -7.5
    assert np.all(np.isfinite(beds))


def test_tc13b_geometry(env):
    grid = env.grid
    assert env.elevation_m.shape == (grid.n_rows, grid.n_cols)
    assert env.shore_row == grid.n_rows - SEA_ROWS - 1
    assert env.is_ocean.sum() == SEA_ROWS * grid.n_cols
    assert env.is_ocean[env.shore_row + 1 :, :].all()
    assert not env.is_ocean[: env.shore_row + 1, :].any()


# --- lakes and Option A ------------------------------------------------------

def test_two_lakes_generated(env):
    ids = sorted(set(env.lake_id[env.is_lake].tolist()))
    assert ids == [spec.lake_id for spec in env.lake_specs]
    assert len(env.lake_specs) == 2


def test_lake_masks_and_ids_consistent(env):
    for spec in env.lake_specs:
        mask = env.lake_id == spec.lake_id
        assert mask.sum() == int(env.is_lake[mask].sum())
        assert int(env.is_lake.sum()) >= mask.sum()
        assert np.all(env.is_lake[mask])
        assert np.all(~env.is_ocean[mask])


def test_lake_components_connected_and_spilled(env):
    z = env.elevation_m
    for spec in env.lake_specs:
        mask = env.lake_id == spec.lake_id
        labels, n = ndimage.label(mask)  # default structure is 4-connectivity
        assert n == 1, "each lake must be exactly one 4-connected component"
        assert mask.sum() >= 12
        assert spec.spill_level_m >= 2.0
        assert spec.spill_level_m - z[mask].min() >= 0.3
        assert np.all(z[mask] < spec.spill_level_m)
        assert spec.spill_level_m <= CONFIG.elevation.terrain_max_m


def test_option_a_recorded_levels_queryable(env):
    for spec in env.lake_specs:
        assert env.lake_level(spec.lake_id) == spec.spill_level_m
    with pytest.raises(KeyError):
        env.lake_level(99)


def test_lakes_separated_and_interior(env):
    specs = env.lake_specs
    (a_row, a_col) = (specs[0].center_row, specs[0].center_col)
    (b_row, b_col) = (specs[1].center_row, specs[1].center_col)
    assert (a_row - b_row) ** 2 + (a_col - b_col) ** 2 >= 30.0**2
    shore = env.shore_row
    for spec in specs:
        assert (spec.center_row + 2.35 * spec.sigma_cells) < shore
        assert (spec.center_row - 2.35 * spec.sigma_cells) >= 2


# --- land cover ---------------------------------------------------------------

def test_land_cover_codes_valid_and_populated(env):
    counts = np.bincount(env.land_cover.ravel(), minlength=4)
    assert set(np.unique(env.land_cover).tolist()) <= {0, 1, 2, 3}
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0 and counts[3] > 0
    total = env.grid.n_cells
    impervious_frac = counts[0] / total
    assert 0.05 <= impervious_frac <= 0.70


def test_land_cover_water_exactly_ocean_plus_lakes(env):
    water = env.is_ocean | env.is_lake
    assert np.all(env.land_cover[water] == 3)
    assert np.all(env.land_cover[~water] != 3)


# --- outfalls -----------------------------------------------------------------

def test_outfalls_are_coastal_land_cells_with_ocean_face(env):
    mask = env.outfall_mask
    assert mask.sum() > 0
    rows, cols = np.nonzero(mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        assert not env.is_ocean[row, col]
        assert not env.is_lake[row, col]
        neighbours = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1),
        ]
        assert any(
            0 <= r < env.grid.n_rows
            and 0 <= c < env.grid.n_cols
            and env.is_ocean[r, c]
            for r, c in neighbours
        )


def test_outfall_invert_sits_at_cell_bed(env):
    mask = env.outfall_mask
    assert np.all(env.outfall_invert_m[mask] == env.elevation_m[mask])
    assert np.all(env.outfall_invert_m[~mask] == 0.0)


def test_every_shoreline_cell_is_an_outfall(env):
    assert env.outfall_mask[env.shore_row, :].all()


# --- elevation finiteness and read-only arrays ----------------------------------

def test_elevation_finite_and_contiguous(env):
    assert np.all(np.isfinite(env.elevation_m))
    assert env.elevation_m.dtype == np.float64


def test_is_water_union(env):
    assert np.all(env.is_water == (env.is_ocean | env.is_lake))


# --- lake invariants across the master and all five Experiment A seeds -----------
# Tested evidence for these six seeds only, not a universal validity claim.

@pytest.mark.parametrize("seed", ENV_SEEDS)
def test_lake_invariants_multi_seed(seed):
    env = build_environment(seed, CONFIG)
    z = env.elevation_m
    ids = sorted(set(env.lake_id[env.is_lake].tolist()))
    assert ids == [0, 1]
    for spec in env.lake_specs:
        mask = env.lake_id == spec.lake_id
        labels, n = ndimage.label(mask)
        assert n == 1, "each lake must be one 4-connected component"
        assert int(mask.sum()) >= 12
        lake_z = z[mask]
        assert spec.spill_level_m >= 2.0
        assert float(lake_z.min()) >= CONFIG.elevation.terrain_min_m - 1e-12
        assert spec.spill_level_m - float(lake_z.min()) >= 0.3
        assert np.all(lake_z < spec.spill_level_m)
        assert spec.carved_depth_m <= 14.0 + 1e-12
        # rim non-submergence: the FULL non-carve annulus [2*sigma, 2.35*sigma]
        # is represented in-grid (the placement guards keep it strictly inside
        # the domain on all four sides) and every ring cell sits at or above
        # the recorded spill level, which IS the ring minimum by construction.
        rows = np.arange(z.shape[0], dtype=np.float64)[:, None]
        cols = np.arange(z.shape[1], dtype=np.float64)[None, :]
        r = np.sqrt(
            (rows - spec.center_row) ** 2 + (cols - spec.center_col) ** 2
        )
        ring = (r >= 2.0 * spec.sigma_cells) & (
            r <= 2.35 * spec.sigma_cells
        )
        ring_cells = np.argwhere(ring)
        assert ring_cells.shape[0] > 0, "annulus must be represented in the grid"
        # Direct geometric extent inequalities on all four sides (not merely
        # ring-cell bounds derived from the in-grid mask).
        assert (spec.center_row + 2.35 * spec.sigma_cells) < env.shore_row
        assert (spec.center_row - 2.35 * spec.sigma_cells) >= 2
        assert (spec.center_col - 2.35 * spec.sigma_cells) >= 1.0
        assert (
            spec.center_col + 2.35 * spec.sigma_cells
        ) <= z.shape[1] - 2.0
        assert int(ring_cells[:, 0].min()) >= 0
        assert int(ring_cells[:, 0].max()) < z.shape[0]
        assert int(ring_cells[:, 1].min()) >= 0
        assert int(ring_cells[:, 1].max()) < z.shape[1]
        assert float(z[ring].min()) >= spec.spill_level_m - 1e-12
        assert float(z[ring].min()) == pytest.approx(spec.spill_level_m, rel=1e-12)


# --- read-only arrays ------------------------------------------------------------

def test_exposed_environment_arrays_are_readonly(env):
    exposed = {
        "elevation_m": env.elevation_m,
        "is_ocean": env.is_ocean,
        "is_lake": env.is_lake,
        "lake_id": env.lake_id,
        "land_cover": env.land_cover,
        "road_cells": env.road_cells,
        "outfall_mask": env.outfall_mask,
        "outfall_invert_m": env.outfall_invert_m,
    }
    for name, array in exposed.items():
        assert isinstance(array, np.ndarray), name
        assert not array.flags.writeable, f"{name} must be read-only"


def test_exposed_road_arrays_are_readonly(env):
    net = env.road_network
    exposed = {
        "node_rows": net.node_rows,
        "node_cols": net.node_cols,
        "node_elevation_m": net.node_elevation_m,
        "edge_u": net.edge_u,
        "edge_v": net.edge_v,
        "edge_class": net.edge_class,
        "edge_length_m": net.edge_length_m,
        "edge_free_flow_time_s": net.edge_free_flow_time_s,
        "edge_is_bridge": net.edge_is_bridge,
        "edge_deck_m": net.edge_deck_m,
    }
    for name, array in exposed.items():
        assert not array.flags.writeable, f"{name} must be read-only"


def test_environment_array_mutation_raises(env):
    with pytest.raises(ValueError):
        env.elevation_m[0, 0] = 123.0
    with pytest.raises(ValueError):
        env.land_cover[0, 0] = 3
    with pytest.raises(ValueError):
        env.is_lake[0, 0] = True
    with pytest.raises(ValueError):
        env.road_network.edge_u[0] = -1
    with pytest.raises(ValueError):
        env.road_network.edge_deck_m[0] = 0.0
    with pytest.raises(ValueError):
        env.road_cells[0, 0] = True


def test_digest_still_works_on_readonly_arrays(env):
    rebuilt = build_environment(MASTER, CONFIG)
    assert rebuilt.digest == env.digest
