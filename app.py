r"""Streamlit app: synthetic flood recovery explorer (data layer + pages 1-2).

Do not run with `streamlit run` here; this file is created for review and
parsed with ast.parse. Everything is SYNTHETIC; no real-world or Mumbai claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import streamlit as st

_REPO = Path(__file__).resolve().parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from urbanflood.app_core import (  # noqa: E402
    APP_DATA_DIR,
    build_d_e,
    build_env,
    count_pairs,
    load_experiment_a_rows,
    load_hpeak,
    load_manifest,
    reachable_count,
    summarize_experiment_a,
)
from urbanflood.app_data import SCENARIO_NAMES  # noqa: E402

BANNER = (
    "SYNTHETIC DATA - recovery testing only; no real-world or Mumbai claim"
)

PAGES = [
    "Overview",
    "Susceptibility (Experiment A)",
    "Flood scenarios",
    "Routing explorer",
    "Method and limitations",
]

LAM = 0.0
CLOSURE_THRESHOLD = 0.30


def _set_page_config() -> None:
    st.set_page_config(page_title="Synthetic Flood Explorer", layout="wide")


def _banner() -> None:
    st.error(BANNER)


@st.cache_data
def _cached_manifest(app_data_dir: str):
    return load_manifest(Path(app_data_dir))


@st.cache_data
def _cached_hpeak(seed_index: int, scenario: str, app_data_dir: str):
    return load_hpeak(seed_index, scenario, app_data_dir=Path(app_data_dir))


@st.cache_data
def _cached_experiment_a_rows(path: str):
    return load_experiment_a_rows(Path(path))


def _draw_map(env) -> plt.Figure:
    network = env.road_network
    fig, ax = plt.subplots(figsize=(6, 6))

    # Land-only elevation: terrain colormap sampled from 0.25 to 1.0
    terrain_colors = plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256))
    land_cmap = mcolors.LinearSegmentedColormap.from_list("land_terrain", terrain_colors)
    land_mask = ~(env.is_ocean | env.is_lake)
    land_elev = np.ma.masked_where(~land_mask, env.elevation_m)
    im = ax.imshow(land_elev, origin="upper", cmap=land_cmap, interpolation="nearest")

    # Ocean cells: flat blue; lake cells: flat teal (no colormap)
    ocean_disp = np.ma.masked_where(~env.is_ocean, np.ones_like(env.elevation_m, dtype=float))
    lake_disp = np.ma.masked_where(~env.is_lake, np.ones_like(env.elevation_m, dtype=float))
    ax.imshow(ocean_disp, origin="upper", cmap=mcolors.ListedColormap(["#1f77b4"]), alpha=0.8, interpolation="nearest")
    ax.imshow(lake_disp, origin="upper", cmap=mcolors.ListedColormap(["#008080"]), alpha=0.8, interpolation="nearest")

    # Colour bar for land elevation
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("land elevation (m)")

    # Road network: one LineCollection
    rows = network.node_rows.astype(np.int64)
    cols = network.node_cols.astype(np.int64)
    segments = [
        [(cols[int(u)], rows[int(u)]), (cols[int(v)], rows[int(v)])]
        for u, v in zip(network.edge_u.astype(np.int64), network.edge_v.astype(np.int64))
    ]
    lc = LineCollection(
        segments, colors="black", linewidths=0.6, alpha=0.7
    )
    ax.add_collection(lc)

    # Legend: ocean, lake, road
    ocean_patch = mpatches.Patch(color="#1f77b4", label="ocean")
    lake_patch = mpatches.Patch(color="#008080", label="lake")
    road_line = mlines.Line2D([0], [0], color="black", lw=1.0, label="road")
    ax.legend(handles=[ocean_patch, lake_patch, road_line], loc="lower right", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Elevation with water masks and road network")
    return fig


def _page_overview(app_data_dir: str) -> None:
    st.header("Overview")
    manifest = _cached_manifest(app_data_dir)
    indices = sorted(manifest.keys())
    sel = st.selectbox(
        "Seed index (shows seed number)",
        indices,
        format_func=lambda i: "index %d  ->  seed %d" % (i, manifest[i]),
    )
    env, cfg = build_env(sel)
    hpeak = _cached_hpeak(sel, "R", app_data_dir)
    d_e = build_d_e(env, hpeak)
    n_closed = int(np.sum(d_e >= CLOSURE_THRESHOLD))
    n_reach = reachable_count(env, d_e, CLOSURE_THRESHOLD, LAM)
    network = env.road_network
    cell_size_m = float(env.grid.cell_size_m)
    st.write(
        "Grid %d x %d, %.0f m cells. Nodes: %d, Edges: %d. "
        "Scenario R, closure %.2f, lambda %.1f: closed edges %d, "
        "reachable pairs %d/20."
        % (
            env.grid.n_rows,
            env.grid.n_cols,
            cell_size_m,
            int(network.n_nodes),
            int(network.n_edges),
            CLOSURE_THRESHOLD,
            LAM,
            n_closed,
            n_reach,
        )
    )
    st.pyplot(_draw_map(env))


HEADLINE_METHODS = ("index", "elevation_only", "random_permutation")


def _page_susceptibility(app_data_dir: str) -> None:
    st.header("Susceptibility (Experiment A)")
    rows = _cached_experiment_a_rows(
        str(_REPO / "docs" / "experiment_a" / "results.csv")
    )
    summary = summarize_experiment_a(rows)
    counts = count_pairs(rows)
    by_key = {(d["scenario"], d["method"]): d for d in summary}

    # Headline table: index, elevation_only, random_permutation per scenario
    md_lines = [
        "| scenario | method | n_seeds | mean AUC 0.30 | min | max | mean lift 0.30 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sname in ("R", "S1", "S2", "S3"):
        for method in HEADLINE_METHODS:
            d = by_key.get((sname, method))
            if d is None:
                continue
            md_lines.append(
                "| %s | %s | %d | %.4f | %.4f | %.4f | %.4f |"
                % (
                    sname,
                    method,
                    d["n_seeds"],
                    d["mean_auc_030"],
                    d["min_auc_030"],
                    d["max_auc_030"],
                    d["mean_lift_030"],
                )
            )
    st.write("\n".join(md_lines))

    # Grouped bar chart with matplotlib (no pandas): index vs elevation_only mean AUC 0.30
    scenarios = ("R", "S1", "S2", "S3")
    x = np.arange(len(scenarios))
    width = 0.35
    index_vals = [
        by_key.get((s, "index"), {}).get("mean_auc_030", float("nan")) for s in scenarios
    ]
    elev_vals = [
        by_key.get((s, "elevation_only"), {}).get("mean_auc_030", float("nan"))
        for s in scenarios
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, index_vals, width, label="index", color="#1f77b4")
    ax.bar(x + width / 2, elev_vals, width, label="elevation_only", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Mean AUC 0.30")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Mean AUC at 0.30 m (5 seeds)")
    ax.legend()
    for i, v in enumerate(index_vals):
        if np.isfinite(v):
            ax.text(i - width / 2, v, "%.3f" % v, ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(elev_vals):
        if np.isfinite(v):
            ax.text(i + width / 2, v, "%.3f" % v, ha="center", va="bottom", fontsize=8)
    st.pyplot(fig)

    # Feature ablations table: remaining methods not in headline set
    remaining_methods = sorted(
        set(d["method"] for d in summary) - set(HEADLINE_METHODS)
    )
    st.write("### Feature ablations (declared weights, not re-fitted)")
    ablation_lines = [
        "| scenario | method | n_seeds | mean AUC 0.30 | min | max | mean lift 0.30 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sname in ("R", "S1", "S2", "S3"):
        for method in remaining_methods:
            d = by_key.get((sname, method))
            if d is None:
                continue
            ablation_lines.append(
                "| %s | %s | %d | %.4f | %.4f | %.4f | %.4f |"
                % (
                    sname,
                    method,
                    d["n_seeds"],
                    d["mean_auc_030"],
                    d["min_auc_030"],
                    d["max_auc_030"],
                    d["mean_lift_030"],
                )
            )
    if len(ablation_lines) > 2:
        st.write("\n".join(ablation_lines))

    st.caption(
        "AUC at 0.30 m by scenario, index vs elevation-only, from results.csv. "
        "Caveat: elevation-only often has the higher top-decile lift, so the "
        "index does not dominate on every metric. "
        "index AUC wins %d of %d; elevation-only lift wins %d of %d."
        % (
            counts["index_auc_wins_030"],
            counts["n_pairs"],
            counts["elevation_lift_wins_030"],
            counts["n_pairs"],
        )
    )


def _page_coming(title: str) -> None:
    st.header(title)
    st.write("Coming in the next stage.")


def main() -> None:
    _set_page_config()
    _banner()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES)
    app_data_dir = str(APP_DATA_DIR)
    if page == "Overview":
        _page_overview(app_data_dir)
    elif page == "Susceptibility (Experiment A)":
        _page_susceptibility(app_data_dir)
    elif page == "Flood scenarios":
        _page_coming("Flood scenarios")
    elif page == "Routing explorer":
        _page_coming("Routing explorer")
    elif page == "Method and limitations":
        _page_coming("Method and limitations")
    else:
        _page_coming(page)


if __name__ == "__main__":
    main()
