"""Phase 0 tests for the independent named seed streams (rng.py, TC2a family).

These tests verify the structural properties required by Architecture V7:
determinism, reproducibility, cross-stream distinctness, and independence of
the master seed. Independence from terrain is asserted structurally in later
phases by TC2a (the drainage generator receives no terrain argument); the
tests here pin the seed arithmetic itself.
"""

from __future__ import annotations

import hashlib

import pytest

from urbanflood.rng import (
    STREAM_DRAINAGE,
    STREAM_ENV,
    STREAM_OD_PAIRS,
    STREAM_ROUTING,
    STREAM_WEATHER,
    all_stream_seeds,
    derive_stream_seed,
    numpy_rng,
    stream_payload,
)

MASTER = 2027
ALL_STREAMS = (STREAM_ENV, STREAM_DRAINAGE, STREAM_WEATHER, STREAM_ROUTING, STREAM_OD_PAIRS)

#: Static golden values captured from the first derivation run of Phase 0.
#: Any change to the payload format shows up here as an intentional failure.
GOLDEN_ENV_2027 = 17843116884876980974
GOLDEN_DRAINAGE_2027 = 753485407558401340


def test_stream_payload_format_is_versioned():
    assert stream_payload(MASTER, STREAM_ENV) == f"synthetic-urban-flood/seed-v1/{MASTER}/env"


def test_derivation_is_deterministic():
    first = derive_stream_seed(MASTER, STREAM_ENV)
    second = derive_stream_seed(MASTER, STREAM_ENV)
    assert first == second


def test_derivation_matches_independent_reimplementation():
    """Recompute the digest directly with hashlib, not via the module."""
    expected = int.from_bytes(
        hashlib.sha256(f"synthetic-urban-flood/seed-v1/{MASTER}/env".encode("ascii")).digest()[:8],
        "big",
    )
    assert derive_stream_seed(MASTER, STREAM_ENV) == expected


def test_static_golden_values():
    assert derive_stream_seed(MASTER, STREAM_ENV) == GOLDEN_ENV_2027
    assert derive_stream_seed(MASTER, STREAM_DRAINAGE) == GOLDEN_DRAINAGE_2027


def test_all_top_level_streams_are_distinct():
    seeds = all_stream_seeds(MASTER)
    assert set(seeds.keys()) == set(ALL_STREAMS)
    values = list(seeds.values())
    assert len(set(values)) == len(values)
    assert all(0 <= value < 2**64 for value in values)


def test_master_seed_changes_every_stream():
    for stream in ALL_STREAMS:
        assert derive_stream_seed(MASTER, stream) != derive_stream_seed(MASTER + 1, stream)


def test_stream_name_changes_output():
    assert derive_stream_seed(MASTER, STREAM_ENV) != derive_stream_seed(MASTER, STREAM_DRAINAGE)
    # sub-streams of one family are distinct from each other and from the base
    sub0 = derive_stream_seed(MASTER, f"{STREAM_ENV}/experiment/0")
    sub1 = derive_stream_seed(MASTER, f"{STREAM_ENV}/experiment/1")
    assert sub0 != sub1
    assert sub0 != derive_stream_seed(MASTER, STREAM_ENV)


def test_stream_payload_rejects_bad_inputs():
    with pytest.raises(TypeError):
        stream_payload("2027", STREAM_ENV)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        stream_payload(MASTER, "")


def test_numpy_rng_is_reproducible():
    first_values = numpy_rng(MASTER, STREAM_ENV).random(5)
    second_values = numpy_rng(MASTER, STREAM_ENV).random(5)
    assert first_values.tobytes() == second_values.tobytes()
    drainage_values = numpy_rng(MASTER, STREAM_DRAINAGE).random(5)
    assert first_values.tobytes() != drainage_values.tobytes()
