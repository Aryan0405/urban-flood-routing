"""TC6: ASCII-only scan of all project source and documentation files.

Architecture V7 requires that no file in the repository contains non-ASCII
bytes (which also rules out em dashes, en dashes, smart quotes, and any
stray non-English characters). This test walks the repository and fails on
the first offending file.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCANNED_SUFFIXES = {".py", ".md", ".toml", ".txt", ".cfg", ".yml", ".yaml", ".gitignore"}
SKIPPED_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}

#: utf-8 encodings of the em dash and en dash; a plain ASCII check already
#: rejects them, but the explicit assertion documents the intent.
EM_DASH_UTF8 = b"\xe2\x80\x94"
EN_DASH_UTF8 = b"\xe2\x80\x93"

#: require a real body of scanned files so an empty repository cannot pass
MIN_FILES_EXPECTED = 10


def iter_scanned_files():
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIPPED_DIRS for part in path.parts):
            continue
        if path.suffix in SCANNED_SUFFIXES or path.name in {".gitignore"}:
            yield path


def test_repository_is_strictly_ascii():
    files = list(iter_scanned_files())
    assert len(files) >= MIN_FILES_EXPECTED, (
        f"TC6 scan expected at least {MIN_FILES_EXPECTED} files, found {len(files)}"
    )
    for path in files:
        data = path.read_bytes()
        non_ascii_offsets = [i for i, byte in enumerate(data) if byte > 0x7F]
        assert not non_ascii_offsets, (
            f"{path}: non-ASCII byte(s) at offsets {non_ascii_offsets[:10]}"
        )
        assert EM_DASH_UTF8 not in data, f"{path}: contains an em dash"
        assert EN_DASH_UTF8 not in data, f"{path}: contains an en dash"
