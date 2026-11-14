r"""App data layer: text exports of Experiment C H_peak arrays.

Pure text I/O helpers with no urbanflood imports. Everything here is
synthetic recovery testing data; no real-world claim is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

SCENARIO_NAMES = ("R", "S1", "S2", "S3")

HPEAK_FMT = "%.6f"
#: Written without a leading '#'; np.savetxt prepends its comment char so the
#: file's first line is exactly "# SYNTHETIC H_peak (m), 96 x 96, ...".
HPEAK_HEADER = "SYNTHETIC H_peak (m), 96 x 96, rows then columns"

N_SEEDS = 5


def hpeak_filename(seed_index: int, scenario: str) -> str:
    """Return the export filename hpeak_s<i>_<scenario>.txt.

    Raises ValueError if seed_index is outside 0..4 or scenario is not one of
    SCENARIO_NAMES.
    """
    if isinstance(seed_index, bool) or not isinstance(seed_index, (int, np.integer)):
        raise ValueError("seed_index must be an integer in 0..4")
    seed_index = int(seed_index)
    if seed_index < 0 or seed_index >= N_SEEDS:
        raise ValueError(
            "seed_index %r is outside the valid range 0..%d"
            % (seed_index, N_SEEDS - 1)
        )
    if scenario not in SCENARIO_NAMES:
        raise ValueError("scenario %r is not one of %r" % (scenario, SCENARIO_NAMES))
    return "hpeak_s%d_%s.txt" % (seed_index, scenario)


def write_hpeak_text(path, array: np.ndarray) -> None:
    """Write a 2D float64 array as ASCII text.

    Uses np.savetxt with fmt '%.6f' and the header comment line
    '# SYNTHETIC H_peak (m), 96 x 96, rows then columns'.
    """
    arr = np.asarray(array, dtype=np.float64)
    np.savetxt(path, arr, fmt=HPEAK_FMT, header=HPEAK_HEADER)


def read_hpeak_text(path) -> np.ndarray:
    """Read a H_peak text file back into a float64 (n_rows, n_cols) array.

    Raises ValueError if the array is not 2D or contains any non-finite value.
    """
    arr = np.asarray(np.loadtxt(path), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "expected a 2D array, got %d-D array with shape %s"
            % (arr.ndim, arr.shape)
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("array contains non-finite value(s)")
    return arr


def write_manifest(path, seeds: List[int]) -> None:
    """Write a manifest text file with one 'index=seed' line per seed."""
    lines = ["%d=%d" % (i, int(s)) for i, s in enumerate(seeds)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def read_manifest(path) -> Dict[int, int]:
    """Read a manifest text file and return {index: seed}."""
    result: Dict[int, int] = {}
    for raw in Path(path).read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError("malformed manifest line: %r" % line)
        index_s, seed_s = line.split("=", 1)
        result[int(index_s.strip())] = int(seed_s.strip())
    return result
