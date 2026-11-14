"""SYNTHETIC numerical foundations for the Phase 2 flood simulator.

This module provides the independent numerical primitives needed by the
later simulation engine. It deliberately does NOT implement the full
timestep loop, donor limiting, drainage execution, sea exchange, or the
mass-balance ledger -- those belong to later milestones.

All functions are deterministic and avoid hidden rounding or artificial
correction. Numerical calculations use Python floats (IEEE 754 double
precision by default on CPython) and NumPy float64 arrays where arrays are
involved; callers requiring a strict float64 guarantee on scalar inputs
should convert explicitly before calling. Where Phase 1 already provides an
appropriate primitive (rainfall integration, midpoint tide), this module
reuses it rather than duplicating it.

Specification: docs/PHASE2_NUMERICAL_SPEC.md (approved).
Architecture: V7 (approved).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from urbanflood.config import RunoffConfig
from urbanflood.weather import applied_precipitation_volume


def _validate_cell_index(value, name="cell index"):
    """Validate that value is a non-negative integer cell index.

    Rejects booleans, floats (including integral-valued floats), strings,
    None, and negative integers. Does not silently coerce invalid values.

    Parameters
    ----------
    value : object
        The value to validate.
    name : str
        Name used in error messages.

    Returns
    -------
    int
        The validated non-negative integer.

    Raises
    ------
    ValueError
        If value is not a non-negative integer cell index.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a non-negative integer, got bool: {value!r}"
        )
    if not isinstance(value, int):
        raise ValueError(
            f"{name} must be a non-negative integer, "
            f"got {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class PartitionResult:
    """Result of the rainfall partition for one cell over one timestep.

    All volumes in cubic metres (m3).
    """

    V_precip_m3: float
    V_runoff_m3: float
    V_loss_m3: float


def precipitation_runoff_loss(
    rate_k_m_per_s: float,
    cell_area_m2: float,
    dt_s: float,
    alpha: float,
) -> PartitionResult:
    """Partition precipitation into runoff and loss for one cell/timestep.

    V_precip = R_k * A * dt        (total precipitation arriving)
    V_runoff = alpha * V_precip    (enters surface-water storage)
    V_loss   = V_precip - V_runoff (never enters storage)

    Parameters
    ----------
    rate_k_m_per_s : float
        Interval-mean rain rate R_k in m/s (from weather.interval_mean_rate).
    cell_area_m2 : float
        Cell area A in m^2.
    dt_s : float
        Timestep in seconds.
    alpha : float
        Runoff coefficient for the cell's land cover (0 < alpha <= 1).

    Returns
    -------
    PartitionResult
        (V_precip, V_runoff, V_loss) in m3.
    """
    # Validate scalar inputs: reject negative rates, non-finite values, and
    # degenerate geometry/timestep. NaN is never silently interpreted as a
    # valid input.
    for name, value in (
        ("rainfall rate", rate_k_m_per_s),
        ("cell area", cell_area_m2),
        ("timestep", dt_s),
        ("runoff coefficient", alpha),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if rate_k_m_per_s < 0.0:
        raise ValueError("rainfall rate must be non-negative")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("runoff coefficient alpha must lie in (0, 1]")
    if cell_area_m2 <= 0.0:
        raise ValueError("cell area must be positive")
    if dt_s <= 0.0:
        raise ValueError("timestep must be positive")

    V_precip = applied_precipitation_volume(rate_k_m_per_s, cell_area_m2, dt_s)
    V_runoff = alpha * V_precip
    V_loss = V_precip - V_runoff
    return PartitionResult(
        V_precip_m3=float(V_precip),
        V_runoff_m3=float(V_runoff),
        V_loss_m3=float(V_loss),
    )


@dataclass(frozen=True)
class InternalFace:
    """One unique eligible internal face between two adjacent cells.

    Attributes
    ----------
    i, j : int
        Canonical cell indices with i < j. Face ID is derived only from
        grid geometry; ordering is deterministic.
    """

    i: int
    j: int

    def __post_init__(self) -> None:
        _validate_cell_index(self.i, "InternalFace.i")
        _validate_cell_index(self.j, "InternalFace.j")
        if not self.i < self.j:
            raise ValueError("internal face must be canonical: i < j")


def canonicalize_faces(faces) -> list[InternalFace]:
    """Return the canonical ordered representation of an arbitrary face collection.

    Accepts any iterable of ``InternalFace`` objects (or any iterable of
    two-int pairs), canonicalizes each face so that the lower cell index is
    first, removes duplicates, and returns the result sorted by ``(i, j)``.
    The output is independent of input order and of any set/dict iteration
    order.

    Duplicate unordered endpoint pairs are removed: when the same pair
    appears multiple times (possibly with reversed orientation), only one
    canonical ``InternalFace`` is retained in the output. Because the output
    is sorted, the retained copy is indistinguishable from any other.

    Cell indices are validated strictly: booleans, floats (including
    integral-valued floats), strings, None, and negative integers are
    rejected. No silent coercion is performed.

    Parameters
    ----------
    faces : iterable
        An iterable of ``InternalFace`` objects or two-int pairs ``(i, j)``.

    Returns
    -------
    list[InternalFace]
        Sorted, unique, canonical internal faces with ``i < j``.

    Raises
    ------
    ValueError
        If any face is malformed, has same-cell endpoints, or contains
        invalid cell indices.
    """
    canonical: list[InternalFace] = []
    seen: set[tuple[int, int]] = set()
    for face in faces:
        if isinstance(face, InternalFace):
            i, j = face.i, face.j
        elif isinstance(face, (tuple, list)) and len(face) == 2:
            i, j = face[0], face[1]
            _validate_cell_index(i, "face[0]")
            _validate_cell_index(j, "face[1]")
        else:
            raise ValueError(
                "each face must be an InternalFace or a two-int pair"
            )
        if not i < j:
            i, j = j, i
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        canonical.append(InternalFace(i=i, j=j))
    canonical.sort(key=lambda f: (f.i, f.j))
    return canonical


def enumerate_internal_faces(
    n_rows: int,
    n_cols: int,
    is_ocean: np.ndarray,
) -> list[InternalFace]:
    """Enumerate unique eligible internal faces, each exactly once.

    Eligible faces are 4-connected adjacent cell pairs (horizontal and
    vertical) where BOTH cells are simulated surface-water cells
    (non-ocean: land or inland lake). Ocean cells are excluded.

    Faces are canonical: the lower cell index is i, the higher is j
    (i = row * n_cols + col). The list is sorted by (i, j) for
    deterministic accumulation order. No dict/set iteration influences the
    result.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    is_ocean : np.ndarray
        Boolean ocean mask, shape (n_rows, n_cols). True = ocean (excluded).

    Returns
    -------
    list[InternalFace]
        Sorted unique internal faces, i < j.
    """
    _validate_cell_index(n_rows, "n_rows")
    _validate_cell_index(n_cols, "n_cols")
    if n_rows < 1 or n_cols < 1:
        raise ValueError("grid dimensions must be positive")
    if not isinstance(is_ocean, np.ndarray):
        raise TypeError("is_ocean must be a numpy.ndarray")
    if is_ocean.shape != (n_rows, n_cols):
        raise ValueError("is_ocean shape must match (n_rows, n_cols)")
    if is_ocean.dtype != np.bool_:
        raise TypeError("is_ocean must have a boolean dtype")

    faces: list[InternalFace] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if is_ocean[r, c]:
                continue
            i = r * n_cols + c
            # Horizontal neighbour to the right.
            if c + 1 < n_cols and not is_ocean[r, c + 1]:
                faces.append(InternalFace(i=i, j=i + 1))
            # Vertical neighbour below.
            if r + 1 < n_rows and not is_ocean[r + 1, c]:
                faces.append(InternalFace(i=i, j=i + n_cols))
    faces.sort(key=lambda f: (f.i, f.j))
    return faces


@dataclass(frozen=True)
class FluxRequest:
    """A directed outgoing request from one donor cell.

    Attributes
    ----------
    donor, receiver : int
        Cell indices. Water moves from donor to receiver.
    volume_m3 : float
        Proposed transfer volume in m3 (non-negative).
    """

    donor: int
    receiver: int
    volume_m3: float

    def __post_init__(self) -> None:
        _validate_cell_index(self.donor, "FluxRequest.donor")
        _validate_cell_index(self.receiver, "FluxRequest.receiver")
        if self.donor == self.receiver:
            raise ValueError("donor and receiver must be distinct cells")
        if not math.isfinite(self.volume_m3):
            raise ValueError("flux request volume must be finite")
        if self.volume_m3 < 0.0:
            raise ValueError("flux request volume must be non-negative")


def signed_flux_to_request(
    V_ij: float,
    i: int,
    j: int,
) -> FluxRequest | None:
    """Convert a signed internal-face flux into a directed donor request.

    Rules (approved numerical specification, section 5):

    * V_ij > 0: donor = i, receiver = j, volume = V_ij.
    * V_ij < 0: donor = j, receiver = i, volume = abs(V_ij).
    * V_ij == 0: no outgoing request (returns None). The eligible face still
      counts as evaluated for TC11.

    The transfer is equal and opposite: the same volume is subtracted from
    the donor and added to the receiver.

    Parameters
    ----------
    V_ij : float
        Signed flux V_ij = c * delta_W_ij * dx * dt in m3. Positive means
        net flow from i to j.
    i, j : int
        Cell indices of the face endpoints.

    Returns
    -------
    FluxRequest | None
        Directed request, or None when V_ij == 0.
    """
    if not math.isfinite(V_ij):
        raise ValueError("signed flux must be finite")
    _validate_cell_index(i, "signed_flux_to_request.i")
    _validate_cell_index(j, "signed_flux_to_request.j")
    if i == j:
        raise ValueError("face endpoints must be distinct cells")
    if V_ij > 0.0:
        return FluxRequest(donor=i, receiver=j, volume_m3=float(V_ij))
    if V_ij < 0.0:
        return FluxRequest(donor=j, receiver=i, volume_m3=float(-V_ij))
    return None


def face_count(faces: list[InternalFace]) -> int:
    """Return the number of evaluated internal faces.

    Each eligible zero-flux face is still represented and counted.
    """
    return len(faces)


def faces_canonical_and_unique(faces: list[InternalFace]) -> bool:
    """Return True if faces are canonical (i < j), unique, and sorted."""
    seen: set[tuple[int, int]] = set()
    prev: tuple[int, int] | None = None
    for f in faces:
        key = (f.i, f.j)
        if not f.i < f.j:
            return False
        if key in seen:
            return False
        seen.add(key)
        if prev is not None and key <= prev:
            return False
        prev = key
    return True
