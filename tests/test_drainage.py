"""Phase 1 tests for the hidden SYNTHETIC drainage layer (drainage.py).

Covers TC2a (structural independence: the generator receives no terrain
input), locked ranges, determinism, and the S3 blocked-mask regeneration
contract (same stream, capacity and efficiency untouched, nested masks).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from urbanflood.config import DrainageConfig, build_default_config
from urbanflood.drainage import DrainageFields, generate_drainage
from urbanflood.environment import build_environment

MASTER = 2027
N_CELLS = build_default_config().grid.n_cells


def test_tc2a_signature_is_structurally_terrain_free():
    parameters = list(inspect.signature(generate_drainage).parameters)
    assert parameters == ["master_seed", "n_cells", "blocked_fraction", "drainage_config"]
    for name in (
        "terrain",
        "elevation",
        "elev",
        "grid",
        "environment",
        "land_cover",
        "outfall",
        "z",
    ):
        assert name not in parameters


def test_environment_builder_signature_excludes_drainage_state():
    parameters = list(inspect.signature(build_environment).parameters)
    assert parameters == ["master_seed", "config"]
    for name in ("drainage", "blocked", "capacity", "efficiency"):
        assert name not in parameters


def test_ranges_respect_locked_config():
    cfg = DrainageConfig()
    fields = generate_drainage(MASTER, N_CELLS, 0.20)
    assert fields.capacity_mm_per_h.min() >= cfg.capacity_min_mm_per_h
    assert fields.capacity_mm_per_h.max() <= cfg.capacity_max_mm_per_h
    assert fields.efficiency.min() >= cfg.efficiency_min
    assert fields.efficiency.max() <= cfg.efficiency_max
    assert fields.capacity_mm_per_h.dtype == np.float64
    assert fields.efficiency.dtype == np.float64
    assert fields.blocked.dtype == np.bool_


def test_blocked_fraction_matches_nominal():
    fields = generate_drainage(MASTER, N_CELLS, 0.20)
    observed = fields.blocked.sum() / N_CELLS
    assert abs(observed - 0.20) < 0.02


def test_blocked_fraction_rejects_bad_values():
    with pytest.raises(ValueError):
        generate_drainage(MASTER, N_CELLS, 1.5)
    with pytest.raises(ValueError):
        generate_drainage(MASTER, N_CELLS, -0.1)
    with pytest.raises(ValueError):
        generate_drainage(MASTER, 0, 0.2)
    with pytest.raises(TypeError):
        generate_drainage(2027.0, N_CELLS, 0.2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_drainage(-5, N_CELLS, 0.2)


def test_determinism_bitwise():
    first = generate_drainage(MASTER, N_CELLS, 0.20)
    second = generate_drainage(MASTER, N_CELLS, 0.20)
    assert first.capacity_mm_per_h.tobytes() == second.capacity_mm_per_h.tobytes()
    assert first.efficiency.tobytes() == second.efficiency.tobytes()
    assert first.blocked.tobytes() == second.blocked.tobytes()


def test_master_seed_changes_every_field():
    other = generate_drainage(MASTER + 1, N_CELLS, 0.20)
    base = generate_drainage(MASTER, N_CELLS, 0.20)
    assert other.capacity_mm_per_h.tobytes() != base.capacity_mm_per_h.tobytes()
    assert other.efficiency.tobytes() != base.efficiency.tobytes()
    assert other.blocked.tobytes() != base.blocked.tobytes()


def test_s3_mask_regeneration_keeps_capacity_and_efficiency():
    """Locked rule: S3 regenerates the blocked mask on the SAME stream.

    Same seed, same n_cells: steps 1 and 2 of the canonical draw order
    reproduce capacity and efficiency bitwise; only the mask changes, and it
    does so by threshold nesting (0.20 mask is a subset of the 0.40 mask).
    """
    reference = generate_drainage(MASTER, N_CELLS, 0.20)
    s3 = generate_drainage(MASTER, N_CELLS, 0.40)
    assert s3.capacity_mm_per_h.tobytes() == reference.capacity_mm_per_h.tobytes()
    assert s3.efficiency.tobytes() == reference.efficiency.tobytes()
    assert np.all(reference.blocked <= s3.blocked)
    assert s3.blocked.sum() > reference.blocked.sum()
    assert abs(s3.blocked.sum() / N_CELLS - 0.40) < 0.02


def test_mask_not_nested_for_different_seeds_sanity():
    """A different seed draws different uniforms, so mask overlap is partial."""
    a = generate_drainage(MASTER, N_CELLS, 0.20)
    b = generate_drainage(MASTER + 1, N_CELLS, 0.20)
    both = a.blocked & b.blocked
    assert both.sum() < a.blocked.sum()
    assert both.sum() < b.blocked.sum()


def test_fields_is_frozen_container():
    fields = generate_drainage(MASTER, N_CELLS, 0.20)
    with pytest.raises(Exception):
        fields.blocked_fraction = 0.4  # type: ignore[misc]


def test_drainage_arrays_are_readonly():
    fields = generate_drainage(MASTER, N_CELLS, 0.20)
    for name, array in (
        ("capacity_mm_per_h", fields.capacity_mm_per_h),
        ("efficiency", fields.efficiency),
        ("blocked", fields.blocked),
    ):
        assert not array.flags.writeable, f"{name} must be read-only"
    with pytest.raises(ValueError):
        fields.capacity_mm_per_h[0] = 999.0
    with pytest.raises(ValueError):
        fields.efficiency[0] = 0.0
    with pytest.raises(ValueError):
        fields.blocked[0] = not fields.blocked[0]


def test_drainage_digest_relevant_operators_still_work_readonly():
    """min/max/sum and tobytes (digest-relevant reads) work on frozen arrays."""
    fields = generate_drainage(MASTER, N_CELLS, 0.20)
    assert fields.capacity_mm_per_h.min() >= DrainageConfig().capacity_min_mm_per_h
    assert float(fields.efficiency.max()) <= 1.0
    assert fields.blocked.tobytes() == fields.blocked.copy().tobytes()
