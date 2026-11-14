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
        if page_label == "Routing explorer":
            caption_values = [str(c.value) for c in at.caption]
            assert any(
                "Reachability depends only on the closure threshold" in cv
                for cv in caption_values
            ), "page %r missing closure-threshold caption in at.caption values %r" % (
                page_label, caption_values
            )


def test_app_smoke_no_exceptions():
    start = time.perf_counter()
    _run_all_pages()
    elapsed = time.perf_counter() - start
    print("\napp_smoke runtime: %.2f s" % elapsed)
    assert elapsed < 60.0, "app_smoke took %.2f s (> 60 s)" % elapsed


# ---------------------------------------------------------------------------
# Additional per-page tests (PART 2 of STAGE 20b)
# ---------------------------------------------------------------------------


def test_page4_non_default():
    """Page 4 with seed 4, S1, threshold 0.45, lambda 5, pair 3."""
    start = time.perf_counter()
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    radio = _radio_element(at)
    radio.set_value("Routing explorer")
    at.run()
    assert len(at.exception) == 0, "page raised: %r" % list(at.exception)

    at.selectbox(key="route_seed").set_value(4)
    at.selectbox(key="route_scenario").set_value("S1")
    at.radio(key="route_threshold").set_value(0.45)
    at.radio(key="route_lambda").set_value(5)
    at.selectbox(key="route_pair").set_value(3)
    at.run()
    assert len(at.exception) == 0, "page raised: %r" % list(at.exception)

    elapsed = time.perf_counter() - start
    print("\npage4 non-default runtime: %.2f s" % elapsed)


def test_page3_non_default():
    """Page 3 with seed 4, scenario S3."""
    start = time.perf_counter()
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    radio = _radio_element(at)
    radio.set_value("Flood scenarios")
    at.run()
    assert len(at.exception) == 0, "page raised: %r" % list(at.exception)

    at.selectbox(key="flood_seed").set_value(4)
    at.selectbox(key="flood_scenario").set_value("S3")
    at.run()
    assert len(at.exception) == 0, "page raised: %r" % list(at.exception)

    elapsed = time.perf_counter() - start
    print("\npage3 non-default runtime: %.2f s" % elapsed)


def test_unreachable_message():
    """Find first unreachable pair at seed 0 / R / 0.30 / lam 0 via app_core,
    drive page 4 to it, assert 'not reachable under this closure set' appears.
    """
    start = time.perf_counter()
    from urbanflood.app_core import (
        build_d_e,
        build_env,
        compute_routes,
        load_hpeak,
    )

    env, _cfg = build_env(0)
    hpeak = load_hpeak(0, "R")
    d_e = build_d_e(env, hpeak)

    unreachable_pair = None
    for pi in range(20):
        res = compute_routes(env, d_e, 0.30, 0.0, pi)
        if not res["policy_reachable"]:
            unreachable_pair = pi
            break
    assert unreachable_pair is not None, "no unreachable pair found at seed 0 / R / 0.30 / 0"

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    radio = _radio_element(at)
    radio.set_value("Routing explorer")
    at.run()
    assert len(at.exception) == 0, "page raised: %r" % list(at.exception)

    at.selectbox(key="route_seed").set_value(0)
    at.selectbox(key="route_scenario").set_value("R")
    at.radio(key="route_threshold").set_value(0.30)
    at.radio(key="route_lambda").set_value(0)
    at.selectbox(key="route_pair").set_value(unreachable_pair)
    at.run()
    assert len(at.exception) == 0, "page raised: %r" % list(at.exception)

    md_values = [str(m.value) for m in at.markdown]
    found = any(
        "not reachable under this closure set" in mv for mv in md_values
    )
    if not found:
        # Report where it is found (at.text, at.write, etc.)
        for attr in ("text", "write"):
            vals = [str(getattr(m, "value", "")) for m in getattr(at, attr, [])]
            for i, v in enumerate(vals):
                if "not reachable under this closure set" in v:
                    print(
                        "UNREACHABLE MESSAGE found in at.%s[%d]: %s"
                        % (attr, i, v[:300])
                    )
                    found = True
                    break
            if found:
                break
        if not found:
            print("UNREACHABLE MESSAGE NOT FOUND. All markdown values:")
            for i, mv in enumerate(md_values):
                print("  markdown[%d]: %s" % (i, mv[:300]))

    assert found, "unreachable message not found in page output"

    elapsed = time.perf_counter() - start
    print("\nunreachable message runtime: %.2f s" % elapsed)
