r"""Smoke test for app.py using Streamlit AppTest.

Runs the default page and each of the 5 sidebar pages, asserting no
exceptions and that the page heading appears in at.header. Requires
streamlit.testing.v1.AppTest. No silent fallback: if the sidebar radio
cannot be found, the test FAILS.
"""

from __future__ import annotations

import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

PAGES = [
    ("Overview", "Overview"),
    ("Susceptibility (Experiment A)", "Susceptibility (Experiment A)"),
    ("Flood scenarios", "Flood scenarios"),
    ("Routing explorer", "Routing explorer"),
    ("Method and limitations", "Method and limitations"),
]


def _radio_element(at):
    """Return the sidebar radio widget; raise if not found."""
    radios = at.sidebar.radio
    if len(radios) == 0:
        raise AssertionError("sidebar radio not found")
    return radios[0]


def _run_all_pages(timeout: float = 60.0):
    for page_label, page_heading in PAGES:
        at = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
        at.run()
        assert len(at.exception) == 0, "page %r raised on first run: %r" % (
            page_label, list(at.exception)
        )
        radio = _radio_element(at)
        radio.set_value(page_label)
        at.run()
        assert len(at.exception) == 0, "page %r raised after select: %r" % (
            page_label, list(at.exception)
        )
        heading_values = [str(h.value) for h in at.header]
        assert any(
            page_heading in hv for hv in heading_values
        ), "page %r heading %r not in at.header values %r" % (
            page_label, page_heading, heading_values
        )
        if page_label == "Susceptibility (Experiment A)":
            md_values = [str(m.value) for m in at.markdown]
            assert any(
                "Feature ablations" in mv for mv in md_values
            ), "page %r missing 'Feature ablations' in at.markdown values %r" % (
                page_label, md_values
            )


def test_app_smoke_no_exceptions():
    start = time.perf_counter()
    _run_all_pages()
    elapsed = time.perf_counter() - start
    print("\napp_smoke runtime: %.2f s" % elapsed)
    assert elapsed < 30.0, "app_smoke took %.2f s (> 30 s)" % elapsed
