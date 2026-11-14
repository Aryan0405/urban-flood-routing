r"""Streamlit app: synthetic flood recovery explorer (data layer + pages 1-4).

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
    compute_routes,
    count_pairs,
    depth_stats,
    load_experiment_a_rows,
    load_hpeak,
    load_manifest,
    reachable_count,
    route_summary,
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
THRESHOLD_OPTIONS = (0.20, 0.30, 0.45)
LAMBDA_OPTIONS = (0, 0.5, 1, 2, 5, 10)


def _set_page_config() -> None:
    st.set_page_config(page_title="Synthetic Flood Explorer", layout="wide")


def _banner() -> None:
    st.error(BANNER)


# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _cached_build_env(seed_index: int):
    return build_env(seed_index)


@st.cache_data
def _cached_manifest(app_data_dir: str):
    return load_manifest(Path(app_data_dir))


@st.cache_data
def _cached_hpeak(seed_index: int, scenario: str, app_data_dir: str):
    return load_hpeak(seed_index, scenario, app_data_dir=Path(app_data_dir))


@st.cache_data
def _cached_d_e(seed_index: int, scenario: str, app_data_dir: str):
    env, _cfg = _cached_build_env(seed_index)
    hpeak = _cached_hpeak(seed_index, scenario, app_data_dir)
    return build_d_e(env, hpeak)


@st.cache_data
def _cached_compute_routes(
    seed_index: int, scenario: str, threshold_m: float, lam: float,
    pair_index: int, app_data_dir: str,
):
    env, _cfg = _cached_build_env(seed_index)
    d_e = _cached_d_e(seed_index, scenario, app_data_dir)
    return compute_routes(env, d_e, threshold_m, lam, pair_index)


@st.cache_data
def _cached_reachable_count(
    seed_index: int, scenario: str, threshold_m: float, lam: float,
    app_data_dir: str,
):
    env, _cfg = _cached_build_env(seed_index)
    d_e = _cached_d_e(seed_index, scenario, app_data_dir)
    return reachable_count(env, d_e, threshold_m, lam)


@st.cache_data
def _cached_experiment_a_rows(path: str):
    return load_experiment_a_rows(Path(path))


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _base_map(ax, env, draw_roads: bool = True):
    """Draw land elevation, ocean, lake, and (optionally) roads on ax.

    Returns the elevation mappable (for a colour bar). No legend, no title.
    """
    network = env.road_network

    # Land-only elevation: terrain colormap sampled from 0.25 to 1.0
    terrain_colors = plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256))
    land_cmap = mcolors.LinearSegmentedColormap.from_list(
        "land_terrain", terrain_colors
    )
    land_mask = ~(env.is_ocean | env.is_lake)
    land_elev = np.ma.masked_where(~land_mask, env.elevation_m)
    im = ax.imshow(
        land_elev, origin="upper", cmap=land_cmap, interpolation="nearest"
    )

    # Ocean cells: flat blue; lake cells: flat teal (no colormap)
    ocean_disp = np.ma.masked_where(
        ~env.is_ocean, np.ones_like(env.elevation_m, dtype=float)
    )
    lake_disp = np.ma.masked_where(
        ~env.is_lake, np.ones_like(env.elevation_m, dtype=float)
    )
    ax.imshow(
        ocean_disp,
        origin="upper",
        cmap=mcolors.ListedColormap(["#1f77b4"]),
        alpha=0.8,
        interpolation="nearest",
    )
    ax.imshow(
        lake_disp,
        origin="upper",
        cmap=mcolors.ListedColormap(["#008080"]),
        alpha=0.8,
        interpolation="nearest",
    )

    if draw_roads:
        # Road network: one LineCollection
        rows = network.node_rows.astype(np.int64)
        cols = network.node_cols.astype(np.int64)
        segments = [
            [(cols[int(u)], rows[int(u)]), (cols[int(v)], rows[int(v)])]
            for u, v in zip(
                network.edge_u.astype(np.int64),
                network.edge_v.astype(np.int64),
            )
        ]
        lc = LineCollection(
            segments, colors="black", linewidths=0.6, alpha=0.7
        )
        ax.add_collection(lc)

    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _draw_closed_edges(ax, network, closed_edge_ids):
    """Draw closed edges as one red LineCollection."""
    if not closed_edge_ids:
        return
    rows = network.node_rows.astype(np.int64)
    cols = network.node_cols.astype(np.int64)
    segments = []
    for eid in closed_edge_ids:
        u = int(network.edge_u[eid])
        v = int(network.edge_v[eid])
        segments.append([(cols[u], rows[u]), (cols[v], rows[v])])
    if segments:
        lc = LineCollection(segments, colors="red", linewidths=1.2)
        ax.add_collection(lc)


def _draw_route(ax, network, edge_ids, color, linewidth=2.5):
    """Draw one route as a LineCollection. None/empty edge_ids draws nothing."""
    if edge_ids is None or len(edge_ids) == 0:
        return
    rows = network.node_rows.astype(np.int64)
    cols = network.node_cols.astype(np.int64)
    segments = []
    for eid in edge_ids:
        u = int(network.edge_u[eid])
        v = int(network.edge_v[eid])
        segments.append([(cols[u], rows[u]), (cols[v], rows[v])])
    if segments:
        lc = LineCollection(segments, colors=color, linewidths=linewidth)
        ax.add_collection(lc)


def _draw_map(env):
    """Page 1 map: unchanged output (base map + colour bar + legend + title)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = _base_map(ax, env, draw_roads=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("land elevation (m)")
    ocean_patch = mpatches.Patch(color="#1f77b4", label="ocean")
    lake_patch = mpatches.Patch(color="#008080", label="lake")
    road_line = mlines.Line2D([0], [0], color="black", lw=1.0, label="road")
    ax.legend(
        handles=[ocean_patch, lake_patch, road_line],
        loc="lower right",
        fontsize=8,
    )
    ax.set_title("Elevation with water masks and road network")
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _page_overview(app_data_dir: str) -> None:
    st.header("Overview")
    manifest = _cached_manifest(app_data_dir)
    indices = sorted(manifest.keys())
    sel = st.selectbox(
        "Seed index (shows seed number)",
        indices,
        key="overview_seed",
        format_func=lambda i: "index %d  ->  seed %d" % (i, manifest[i]),
    )
    env, cfg = _cached_build_env(sel)
    hpeak = _cached_hpeak(sel, "R", app_data_dir)
    d_e = _cached_d_e(sel, "R", app_data_dir)
    n_closed = int(np.sum(d_e >= CLOSURE_THRESHOLD))
    n_reach = _cached_reachable_count(sel, "R", CLOSURE_THRESHOLD, LAM, app_data_dir)
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
        by_key.get((s, "index"), {}).get("mean_auc_030", float("nan"))
        for s in scenarios
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
            ax.text(
                i - width / 2,
                v,
                "%.3f" % v,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for i, v in enumerate(elev_vals):
        if np.isfinite(v):
            ax.text(
                i + width / 2,
                v,
                "%.3f" % v,
                ha="center",
                va="bottom",
                fontsize=8,
            )
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


def _page_flood_scenarios(app_data_dir: str) -> None:
    st.header("Flood scenarios")
    manifest = _cached_manifest(app_data_dir)
    indices = sorted(manifest.keys())
    sel = st.selectbox(
        "Seed index",
        indices,
        key="flood_seed",
        format_func=lambda i: "index %d  ->  seed %d" % (i, manifest[i]),
    )
    scenario = st.selectbox(
        "Scenario", list(SCENARIO_NAMES), key="flood_scenario"
    )
    env, _cfg = _cached_build_env(sel)
    hpeak = _cached_hpeak(sel, scenario, app_data_dir)
    d_e = _cached_d_e(sel, scenario, app_data_dir)
    stats = depth_stats(env, hpeak, d_e)

    fig, ax = plt.subplots(figsize=(6, 6))
    _base_map(ax, env, draw_roads=False)
    # Draw hpeak over land with sampled Blues colormap, masked where depth < 0.01 m
    land_mask = ~(env.is_ocean | env.is_lake)
    peak_land = np.ma.masked_where(
        ~land_mask | (hpeak < 0.01), hpeak
    )
    blues_colors = plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256))
    blues_cmap = mcolors.LinearSegmentedColormap.from_list("blues_sampled", blues_colors)
    im = ax.imshow(
        peak_land, origin="upper", cmap=blues_cmap,
        interpolation="nearest", vmin=0.01, vmax=1.0
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cbar.set_label("peak depth (m), colour saturates at 1.0 m")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Peak depth H_peak, scenario %s" % scenario)
    st.pyplot(fig)
    st.caption(
        "Peak depth is each cell's maximum over the run, not one moment in "
        "time. Colours saturate at 1.0 m; the maximum is in the table "
        "below. Isolated deep cells, including on high ground, appear to be "
        "local terrain depressions that collect water in the simulation."
    )

    # Markdown table from depth_stats
    md_lines = ["| stat | value |", "|---|---|"]
    stat_rows = [
        ("max_peak_m", "max peak, land cells (m)", "%.4f"),
        ("max_peak_lake_m", "max peak, lake cells (m)", "%.4f"),
        ("share_land_ge_010", "share land >= 0.10", "%.4f"),
        ("share_land_ge_030", "share land >= 0.30", "%.4f"),
        ("share_land_ge_045", "share land >= 0.45", "%.4f"),
        ("closed_020", "closed edges at 0.20", "%d"),
        ("closed_030", "closed edges at 0.30", "%d"),
        ("closed_045", "closed edges at 0.45", "%d"),
    ]
    for key, label, fmt in stat_rows:
        md_lines.append("| %s | %s |" % (label, fmt % stats[key]))
    st.write("\n".join(md_lines))


def _page_routing_explorer(app_data_dir: str) -> None:
    st.header("Routing explorer")
    manifest = _cached_manifest(app_data_dir)
    indices = sorted(manifest.keys())
    sel = st.selectbox(
        "Seed index",
        indices,
        key="route_seed",
        format_func=lambda i: "index %d  ->  seed %d" % (i, manifest[i]),
    )
    scenario = st.selectbox(
        "Scenario", list(SCENARIO_NAMES), key="route_scenario"
    )
    threshold = st.radio(
        "Closure threshold", THRESHOLD_OPTIONS, key="route_threshold",
        format_func=lambda t: "%.2f" % t, horizontal=True
    )
    lam = st.radio(
        "Lambda", LAMBDA_OPTIONS, key="route_lambda", horizontal=True
    )
    pair_index = st.selectbox(
        "Pair index", list(range(20)), key="route_pair"
    )

    env, _cfg = _cached_build_env(sel)
    d_e = _cached_d_e(sel, scenario, app_data_dir)
    routes = _cached_compute_routes(
        sel, scenario, threshold, lam, pair_index, app_data_dir
    )
    n_reach = _cached_reachable_count(
        sel, scenario, threshold, lam, app_data_dir
    )

    st.write("Reachable pairs: %d/20" % n_reach)

    # Draw map with routes
    fig, ax = plt.subplots(figsize=(6, 6))
    _base_map(ax, env, draw_roads=True)
    network = env.road_network

    # Closed edges in red
    _draw_closed_edges(ax, network, routes["closed_edge_ids"])

    # P0 route first (thick), policy route on top (thinner) so overlap shows both
    _draw_route(ax, network, routes["p0_edges"], "#2ca02c", linewidth=5.0)
    _draw_route(ax, network, routes["policy_edges"], "#ff7f0e", linewidth=2.0)

    # Origin and destination markers
    u, v = routes["od"]
    rows = network.node_rows.astype(np.int64)
    cols = network.node_cols.astype(np.int64)
    ax.scatter([cols[u]], [rows[u]], color="green", s=60, zorder=6, marker="o")
    ax.scatter([cols[v]], [rows[v]], color="red", s=60, zorder=6, marker="X")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Routes, scenario %s, threshold %.2f, lambda %.1f"
        % (scenario, threshold, lam)
    )

    handles = [
        mpatches.Patch(color="#1f77b4", label="ocean"),
        mpatches.Patch(color="#008080", label="lake"),
        mlines.Line2D([0], [0], color="black", lw=1.0, label="road"),
        mlines.Line2D([0], [0], color="red", lw=1.5, label="closed edges"),
        mlines.Line2D([0], [0], color="#2ca02c", lw=2.5, label="P0 route"),
        mlines.Line2D([0], [0], color="#ff7f0e", lw=2.5, label="policy route"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    st.pyplot(fig)

    # Route summary table
    p0_sum = route_summary(env, d_e, routes["p0_edges"])
    pol_sum = route_summary(env, d_e, routes["policy_edges"])
    md_lines = [
        "| route | length_m | free_flow_time_s | mean_d_e | max_d_e | n_edges |",
        "|---|---|---|---|---|---|",
    ]
    for name, summ in (("P0", p0_sum), ("policy", pol_sum)):
        if summ is None:
            md_lines.append(
                "| %s | not reachable under this closure set | | | | |" % name
            )
        else:
            md_lines.append(
                "| %s | %.2f | %.2f | %.4f | %.4f | %d |"
                % (
                    name,
                    summ["length_m"],
                    summ["free_flow_time_s"],
                    summ["mean_d_e"],
                    summ["max_d_e"],
                    summ["n_edges"],
                )
            )
    st.write("\n".join(md_lines))

    st.caption(
        "Reachability depends only on the closure threshold; lambda changes "
        "route choice among reachable pairs."
    )


def _page_method(app_data_dir: str) -> None:
    """Method and limitations page (page 5)."""
    st.header("Method and limitations")

    st.markdown(
        "### What this is\n"
        "A synthetic urban flood simulation and risk-aware routing project. "
        "Synthetic recovery testing only; no real-world or Mumbai claim."
    )

    st.markdown(
        "### Three separated layers\n"
        "1. Susceptibility index: built from terrain, land cover and water "
        "masks only.\n"
        "2. Flood simulator: water balance, terrain-blind drainage, tide, "
        "rainfall, per-face sea exchange.\n"
        "3. Routing: Dijkstra, closures, penalised impedance.\n"
        "The index never sees drainage, outfalls, lake specs, depths or the "
        "simulator."
    )

    st.markdown(
        "### Scenarios\n"
        "R (90 mm/h peak, 120 mm), S1 (150 mm/h, 200 mm), S2 (tide amplitude "
        "1.8 m), S3 (40 percent blocked drains). 96 x 96 grid, 50 m cells, "
        "five environment seeds."
    )

    st.markdown(
        "### Experiment A\n"
        "Index beats elevation-only on AUC at 0.30 m in 20 of 20 "
        "seed-scenario pairs; elevation-only has the higher top-decile lift "
        "in many cells; weights were declared before the experiment and not "
        "re-fitted."
    )

    st.markdown(
        "### Experiment C (closure 0.30, lambda 0, pooled over 100 pairs)\n"
        "Reachable fraction: R 0.50, S1 0.26, S2 0.48, S3 0.50. "
        "Per-seed ranges: R 0.25 to 0.65, S1 0.05 to 0.55, S2 0.25 to 0.60, "
        "S3 0.25 to 0.65."
    )

    st.markdown(
        "### Limitations\n"
        "- Reachable fraction measures network connectivity under closure, "
        "not routing quality; about 54 percent of unreachable pairs are "
        "splits into non-trivial components.\n"
        "- Detour and exposure are means over pairs where both routes exist.\n"
        "- P0 ignores depth, so exposure reduction against P0 is largely by "
        "construction and the lambda effect is small.\n"
        "- Roads are not terrain-aware; the deepest road edges appear to sit "
        "in terrain pits.\n"
        "- Peak depth maps show each cell's maximum over the run.\n"
        "- Lake cells can reach several metres in the simulation; this was "
        "not investigated.\n"
        "- No calibration against real data.\n"
        "- pandas is blocked by an application policy on the development "
        "machine, so the app uses matplotlib and markdown tables.\n"
        "- This is not a forecasting or emergency tool."
    )


def _page_coming(title: str) -> None:
    st.header(title)
    st.write("Coming in the next stage.")


def main() -> None:
    _set_page_config()
    _banner()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES, key="nav_radio")
    app_data_dir = str(APP_DATA_DIR)
    if page == "Overview":
        _page_overview(app_data_dir)
    elif page == "Susceptibility (Experiment A)":
        _page_susceptibility(app_data_dir)
    elif page == "Flood scenarios":
        _page_flood_scenarios(app_data_dir)
    elif page == "Routing explorer":
        _page_routing_explorer(app_data_dir)
    elif page == "Method and limitations":
        _page_method(app_data_dir)
    else:
        _page_coming(page)


if __name__ == "__main__":
    main()
