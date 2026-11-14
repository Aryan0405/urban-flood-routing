r"""Tests for src/urbanflood/app_data.py (app data layer).

Fast tests only: toy data, tmp_path, no simulation, no export runner.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from urbanflood.app_data import (
    SCENARIO_NAMES,
    hpeak_filename,
    read_hpeak_text,
    read_manifest,
    write_hpeak_text,
    write_manifest,
)

N_SEEDS = 5
SEEDS = [
    10406733617104774339,
    2112024614403005981,
    4656971133199668420,
    11861282352235076592,
    6574970189797259754,
]


def _write(tmp_path: Path, name: str, arr: np.ndarray) -> Path:
    p = tmp_path / name
    write_hpeak_text(p, arr)
    return p


# ---------------------------------------------------------------------------
# write_hpeak_text / read_hpeak_text round trip
# ---------------------------------------------------------------------------


def test_hpeak_text_roundtrip(tmp_path: Path) -> None:
    arr = np.array([[0.1234567, 1.5], [2.25, 3.999999]], dtype=np.float64)
    p = _write(tmp_path, "round.txt", arr)
    back = read_hpeak_text(p)
    assert back.shape == arr.shape
    assert back.dtype == np.float64
    assert np.allclose(back, arr, atol=1e-6)


def test_hpeak_text_header_line(tmp_path: Path) -> None:
    _write(tmp_path, "hdr.txt", np.zeros((2, 2)))
    text = (tmp_path / "hdr.txt").read_text(encoding="ascii")
    assert text.splitlines()[0] == "# SYNTHETIC H_peak (m), 96 x 96, rows then columns"


def test_hpeak_text_ascii_only(tmp_path: Path) -> None:
    p = _write(tmp_path, "ascii.txt", np.arange(4.0).reshape(2, 2))
    data = p.read_bytes()
    assert all(b <= 0x7F for b in data)


def test_read_hpeak_text_non_2d_raises(tmp_path: Path) -> None:
    p = tmp_path / "oned.txt"
    np.savetxt(p, np.array([1.0, 2.0, 3.0]), fmt="%.6f")
    with pytest.raises(ValueError, match="2D"):
        read_hpeak_text(p)


def test_read_hpeak_text_nan_raises(tmp_path: Path) -> None:
    p = tmp_path / "nan.txt"
    np.savetxt(p, np.array([[1.0, np.nan], [3.0, 4.0]]), fmt="%.6f")
    with pytest.raises(ValueError, match="non-finite"):
        read_hpeak_text(p)


# ---------------------------------------------------------------------------
# hpeak_filename
# ---------------------------------------------------------------------------


def test_hpeak_filename_format() -> None:
    assert hpeak_filename(0, "R") == "hpeak_s0_R.txt"
    assert hpeak_filename(4, "S3") == "hpeak_s4_S3.txt"
    for i in range(N_SEEDS):
        for s in SCENARIO_NAMES:
            assert hpeak_filename(i, s) == "hpeak_s%d_%s.txt" % (i, s)


def test_hpeak_filename_bad_index_raises() -> None:
    for bad in (-1, 5, 10, 100):
        with pytest.raises(ValueError):
            hpeak_filename(bad, "R")


def test_hpeak_filename_bad_scenario_raises() -> None:
    with pytest.raises(ValueError):
        hpeak_filename(0, "X")
    with pytest.raises(ValueError):
        hpeak_filename(0, "r")


# ---------------------------------------------------------------------------
# manifest round trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "manifest.txt"
    write_manifest(p, SEEDS)
    back = read_manifest(p)
    assert back == {i: s for i, s in enumerate(SEEDS)}
    lines = p.read_text(encoding="ascii").splitlines()
    assert len(lines) == 5
    assert lines[0] == "0=10406733617104774339"
    assert lines[4] == "4=6574970189797259754"


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


def test_app_data_import_isolation() -> None:
    spec = importlib.util.find_spec("urbanflood.app_data")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        tree = ast.parse(f.read())
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    allowed = {"numpy", "pathlib", "typing", "__future__"}
    for name in imported:
        assert any(
            name == a or name.startswith(a + ".") for a in allowed
        ), f"disallowed import: {name}"
    assert not any("urbanflood" in n for n in imported)
