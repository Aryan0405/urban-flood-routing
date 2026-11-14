"""Shared read-only array helper for SYNTHETIC generated data (Phase 1 audit).

Frozen dataclasses do not make their NumPy arrays immutable, so the
generators explicitly disable write access on every exposed array before
returning it. No copy is made: disabling writeable is a metadata flag
operation, so digest calculations and downstream consumers keep working on
the same buffers.
"""

from __future__ import annotations

import numpy as np


def freeze_array(array: np.ndarray) -> np.ndarray:
    """Return the SAME array with write access disabled (no copy).

    Mutating an attempt afterwards (element assignment, fill, in-place
    arithmetic) raises a ValueError instead of silently corrupting the
    environment. Callers that legitimately need a mutable copy must make
    that copy explicitly.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("freeze_array expects a numpy.ndarray")
    if array.flags.writeable:
        array.flags.writeable = False
    return array
