"""Deterministic named seed-stream derivation for the synthetic generators.

SYNTHETIC: every seed and stream in this module drives synthetic data only.

Convention (Architecture V7, provenance split adopted in V4):

  env        terrain, land cover, water bodies, road geometry
  drainage   hidden drainage properties ONLY (capacity D_i, efficiency
             eta_d,i, blocked status b_i). The generator that consumes this
             stream receives NO terrain input of any kind; outfall
             infrastructure is a separate deterministic terrain function.
  weather    scenario hyetographs and tide parameters
  routing    OD-pair selection

Streams are derived with SHA-256 over a payload containing the master seed
and the stream name. This makes every stream:

  deterministic      same inputs always produce the same stream seed
  reproducible       the stream seed fully fixes the downstream generator
  name-separated     different stream names map to independently derived seeds.
                      The derivation is SHA-256, truncated to the first 64
                      bits: two distinct named streams CAN in principle
                      collide, though a collision is negligibly unlikely
                      (birthday bound on a 2^64 space). Distinctness across
                      stream names is therefore probabilistic, not
                      mathematically impossible. The structural separation
                      from terrain is exact and by construction: the drainage
                      generator receives no terrain argument at all (asserted
                      by TC2a structurally, never by small-sample
                      correlation).

Payload format (versioned; changing it is an intentional, documented act):
  synthetic-urban-flood/seed-v1/{master_seed}/{stream_name}
"""

from __future__ import annotations

import hashlib

STREAM_ENV = "env"
STREAM_DRAINAGE = "drainage"
STREAM_WEATHER = "weather"
STREAM_ROUTING = "routing"
STREAM_OD_PAIRS = "od_pairs"

_ALL_STREAMS = (
    STREAM_ENV,
    STREAM_DRAINAGE,
    STREAM_WEATHER,
    STREAM_ROUTING,
    STREAM_OD_PAIRS,
)

_PAYLOAD_VERSION = "seed-v1"
_PAYLOAD_PREFIX = "synthetic-urban-flood"


def stream_payload(master_seed: int, stream_name: str) -> str:
    """Return the exact string hashed to derive a stream seed.

    Exposed so that tests (TC2a family) can reimplement the derivation
    independently and compare against this module.
    """
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError("master_seed must be an int")
    if not stream_name or not isinstance(stream_name, str):
        raise ValueError("stream_name must be a non-empty string")
    return f"{_PAYLOAD_PREFIX}/{_PAYLOAD_VERSION}/{int(master_seed)}/{stream_name}"


def derive_stream_seed(master_seed: int, stream_name: str) -> int:
    """Derive a 64-bit non-negative stream seed.

    Deterministic and reproducible by construction: SHA-256 of the versioned
    payload, first 8 bytes read big-endian. Sub-c streams use slash-separated
    names such as "env/experiment/0".
    """
    payload = stream_payload(master_seed, stream_name)
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def all_stream_seeds(master_seed: int) -> dict[str, int]:
    """Derive every required top-level stream seed for one environment."""
    return {name: derive_stream_seed(master_seed, name) for name in _ALL_STREAMS}


def numpy_rng(master_seed: int, stream_name: str):
    """Return a numpy Generator bound to the named stream.

    Reproducible: repeated calls with identical arguments return generators
    that produce identical sequences. numpy is imported lazily so that seed
    arithmetic stays testable without numpy on the import path.
    """
    import numpy as np

    return np.random.default_rng(derive_stream_seed(master_seed, stream_name))
