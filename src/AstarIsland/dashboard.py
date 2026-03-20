#!/usr/bin/env python3
"""
Astar Island Dashboard — Interactive visualization for the Norse World Prediction challenge.

Run with:
    export ASTAR_TOKEN="your_jwt_token"
    streamlit run src/AstarIsland/dashboard.py
"""

import glob as glob_mod
import json
import os
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE = "https://api.ainm.no"

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
NUM_CLASSES = 6

# Terrain raw values → class index
TERRAIN_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Display names for raw terrain values
TERRAIN_NAMES = {
    0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
    4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains",
}

# Color palette for terrain classes
CLASS_COLORS = {
    0: "#2196F3",  # Empty/Ocean/Plains — blue
    1: "#FF9800",  # Settlement — orange
    2: "#9C27B0",  # Port — purple
    3: "#795548",  # Ruin — brown
    4: "#4CAF50",  # Forest — green
    5: "#607D8B",  # Mountain — grey
}

# More detailed colors for initial terrain (distinguishes ocean/plains/empty)
INITIAL_TERRAIN_COLORS = {
    0: "#E0E0E0",   # Empty — light grey
    1: "#FF9800",   # Settlement — orange
    2: "#9C27B0",   # Port — purple
    3: "#795548",   # Ruin — brown
    4: "#4CAF50",   # Forest — green
    5: "#607D8B",   # Mountain — grey
    10: "#1565C0",  # Ocean — deep blue
    11: "#C8E6C9",  # Plains — light green
}

LEARNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "learned_priors.npz")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


# ---------------------------------------------------------------------------
# Cache helpers — load saved observations from solve.py runs
# ---------------------------------------------------------------------------

def load_cached_observations(round_id, seed_index, map_h, map_w):
    """Load cached observation arrays for a seed. Returns (counts, obs_count, n_queries) or None."""
    d = os.path.join(CACHE_DIR, round_id)
    path = os.path.join(d, f"obs_seed{seed_index}.npz")
    if os.path.exists(path):
        data = np.load(path)
        counts = data["counts"]
        obs_count = data["obs_count"]
        if counts.shape == (map_h, map_w, NUM_CLASSES) and obs_count.shape == (map_h, map_w):
            n = len(glob_mod.glob(os.path.join(d, f"query_seed{seed_index}_*.json")))
            return counts, obs_count, n
    return None


def load_cached_raw_queries(round_id, seed_index):
    """Load all raw query results for a seed. Returns list of dicts."""
    d = os.path.join(CACHE_DIR, round_id)
    if not os.path.isdir(d):
        return []
    files = sorted(glob_mod.glob(os.path.join(d, f"query_seed{seed_index}_*.json")))
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_token():
    """Get auth token from env or .env file."""
    token = os.environ.get("ASTAR_TOKEN", "")
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line.startswith("ASTAR_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
    return token


def get_session():
    """Create authenticated session."""
    token = get_token()
    if not token:
        return None
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


@st.cache_data(ttl=30)
def fetch_rounds(_token):
    """Fetch all rounds."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_token}"
    resp = session.get(f"{BASE}/astar-island/rounds")
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30)
def fetch_round_detail(_token, round_id):
    """Fetch round detail with initial states."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_token}"
    resp = session.get(f"{BASE}/astar-island/rounds/{round_id}")
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30)
def fetch_my_rounds(_token):
    """Fetch rounds with team scores."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_token}"
    resp = session.get(f"{BASE}/astar-island/my-rounds")
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30)
def fetch_my_predictions(_token, round_id):
    """Fetch team predictions for a round."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_token}"
    resp = session.get(f"{BASE}/astar-island/my-predictions/{round_id}")
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def fetch_analysis(_token, round_id, seed_index):
    """Fetch ground truth analysis for a completed round seed."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_token}"
    resp = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed_index}")
    if resp.status_code != 200:
        return None
    return resp.json()


@st.cache_data(ttl=30)
def fetch_leaderboard():
    """Fetch public leaderboard."""
    resp = requests.get(f"{BASE}/astar-island/leaderboard")
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30)
def fetch_budget(_token):
    """Fetch query budget."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_token}"
    resp = session.get(f"{BASE}/astar-island/budget")
    if resp.status_code != 200:
        return None
    return resp.json()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_terrain_heatmap(grid, title="Terrain Map", show_legend=True):
    """Create a plotly heatmap of the terrain grid with proper discrete colors."""
    grid = np.array(grid)
    h, w = grid.shape

    # Map each cell to its color
    unique_vals = sorted(set(grid.flatten()))
    # Build a numeric mapping for plotly
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    mapped = np.vectorize(val_to_idx.get)(grid)

    colorscale = []
    n = len(unique_vals)
    for i, v in enumerate(unique_vals):
        color = INITIAL_TERRAIN_COLORS.get(v, "#FFFFFF")
        lo = i / n
        hi = (i + 1) / n
        colorscale.append([lo, color])
        colorscale.append([hi, color])

    # Build hover text
    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            hover[y, x] = f"({x}, {y}): {TERRAIN_NAMES.get(grid[y, x], '?')} [{grid[y, x]}]"

    fig = go.Figure(data=go.Heatmap(
        z=mapped,
        hovertext=hover,
        hoverinfo="text",
        colorscale=colorscale,
        showscale=False,
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    if show_legend:
        for v in unique_vals:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=INITIAL_TERRAIN_COLORS.get(v, "#FFF")),
                name=TERRAIN_NAMES.get(v, f"Type {v}"),
                showlegend=True,
            ))

    return fig


def make_class_heatmap(class_grid, title="Predicted Classes", confidence=None):
    """Heatmap for class-index grid (0-5) with optional confidence overlay."""
    grid = np.array(class_grid)
    h, w = grid.shape

    colorscale = []
    for i in range(NUM_CLASSES):
        lo = i / NUM_CLASSES
        hi = (i + 1) / NUM_CLASSES
        colorscale.append([lo, CLASS_COLORS[i]])
        colorscale.append([hi, CLASS_COLORS[i]])

    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            cls = int(grid[y, x])
            text = f"({x}, {y}): {CLASS_NAMES[cls]}"
            if confidence is not None:
                text += f"<br>Confidence: {confidence[y][x]:.3f}"
            hover[y, x] = text

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        hovertext=hover,
        hoverinfo="text",
        colorscale=colorscale,
        showscale=False,
        zmin=0, zmax=NUM_CLASSES - 1,
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    for i in range(NUM_CLASSES):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=CLASS_COLORS[i]),
            name=CLASS_NAMES[i], showlegend=True,
        ))

    return fig


def make_confidence_heatmap(confidence_grid, title="Prediction Confidence"):
    """Heatmap of confidence values (0-1)."""
    grid = np.array(confidence_grid)
    h, w = grid.shape

    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            hover[y, x] = f"({x}, {y}): {grid[y, x]:.3f}"

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        hovertext=hover,
        hoverinfo="text",
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="Confidence"),
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def make_probability_heatmap(prob_grid, class_idx, title=None):
    """Heatmap of probability for a specific class across the grid."""
    probs = np.array(prob_grid)[:, :, class_idx]
    h, w = probs.shape
    if title is None:
        title = f"P({CLASS_NAMES[class_idx]})"

    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            hover[y, x] = f"({x}, {y}): {probs[y, x]:.4f}"

    fig = go.Figure(data=go.Heatmap(
        z=probs,
        hovertext=hover,
        hoverinfo="text",
        colorscale="Viridis",
        zmin=0, zmax=1,
        colorbar=dict(title="Probability"),
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def make_entropy_heatmap(prob_grid, title="Entropy (Uncertainty)"):
    """Heatmap of per-cell entropy."""
    probs = np.array(prob_grid, dtype=np.float64)
    probs_safe = np.clip(probs, 1e-15, None)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=-1)

    h, w = entropy.shape
    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            hover[y, x] = f"({x}, {y}): {entropy[y, x]:.4f}"

    fig = go.Figure(data=go.Heatmap(
        z=entropy,
        hovertext=hover,
        hoverinfo="text",
        colorscale="Inferno",
        colorbar=dict(title="Entropy"),
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def make_kl_heatmap(prediction, ground_truth, title="KL Divergence (Error Map)"):
    """Per-cell KL divergence heatmap."""
    p = np.array(ground_truth, dtype=np.float64)
    q = np.array(prediction, dtype=np.float64)
    p_safe = np.clip(p, 1e-15, None)
    q_safe = np.clip(q, 1e-15, None)
    kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)

    h, w = kl.shape
    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            hover[y, x] = f"({x}, {y}): KL={kl[y, x]:.4f}"

    fig = go.Figure(data=go.Heatmap(
        z=kl,
        hovertext=hover,
        hoverinfo="text",
        colorscale="Hot",
        colorbar=dict(title="KL Divergence"),
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def make_diff_heatmap(pred_argmax, gt_argmax, title="Prediction vs Ground Truth Diff"):
    """Binary diff: green=correct, red=wrong."""
    pred = np.array(pred_argmax)
    gt = np.array(gt_argmax)
    correct = (pred == gt).astype(int)

    h, w = correct.shape
    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            p_cls = CLASS_NAMES[int(pred[y, x])]
            g_cls = CLASS_NAMES[int(gt[y, x])]
            status = "Correct" if correct[y, x] else "Wrong"
            hover[y, x] = f"({x},{y}): {status}<br>Pred: {p_cls}<br>GT: {g_cls}"

    fig = go.Figure(data=go.Heatmap(
        z=correct,
        hovertext=hover,
        hoverinfo="text",
        colorscale=[[0, "#EF5350"], [1, "#66BB6A"]],
        showscale=False,
        zmin=0, zmax=1,
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color="#66BB6A"), name="Correct"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color="#EF5350"), name="Wrong"))
    return fig


def compute_settlement_distance(settlements, map_h, map_w):
    """Manhattan distance to nearest settlement."""
    dist = np.full((map_h, map_w), 999.0)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for y in range(map_h):
            for x in range(map_w):
                d = abs(x - sx) + abs(y - sy)
                if d < dist[y, x]:
                    dist[y, x] = d
    return dist


def make_settlement_distance_heatmap(settlements, map_h, map_w):
    """Heatmap of Manhattan distance to nearest settlement."""
    dist = compute_settlement_distance(settlements, map_h, map_w)
    dist_display = np.where(dist >= 999, np.nan, dist)

    h, w = dist.shape
    hover = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            hover[y, x] = f"({x}, {y}): dist={dist[y, x]:.0f}"

    fig = go.Figure(data=go.Heatmap(
        z=dist_display,
        hovertext=hover,
        hoverinfo="text",
        colorscale="YlOrRd_r",
        colorbar=dict(title="Distance"),
    ))

    # Mark settlement positions
    sx = [s["x"] for s in settlements]
    sy = [s["y"] for s in settlements]
    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="markers",
        marker=dict(size=10, color="black", symbol="star"),
        name="Settlements",
    ))

    fig.update_layout(
        title="Distance to Nearest Settlement",
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Learned priors visualization
# ---------------------------------------------------------------------------

def load_learned_priors():
    """Load learned priors if available."""
    if os.path.exists(LEARNED_MODEL_PATH):
        data = np.load(LEARNED_MODEL_PATH)
        return data["learned"], data["counts"]
    return None, None


def show_learned_priors_section():
    """Display learned priors visualization."""
    learned, counts = load_learned_priors()
    if learned is None:
        st.warning("No learned priors found. Run `python solve.py --learn` first.")
        return

    st.subheader("Transition Probabilities")
    st.caption("P(final class | initial terrain, distance bucket, coastal, forest-adjacent)")

    dist_labels = ["On-site (0)", "Near (1-2)", "Medium (3-5)", "Far (6+)"]

    selected_terrain = st.selectbox(
        "Initial terrain type",
        range(NUM_CLASSES),
        format_func=lambda i: CLASS_NAMES[i],
        key="prior_terrain",
    )

    # Show heatmaps: rows = distance buckets, columns = final classes
    # Aggregate over coastal/forest for a summary view
    for coastal in range(2):
        for forest in range(2):
            label = f"{'Coastal' if coastal else 'Inland'}, {'Forest-adj' if forest else 'No forest'}"
            n = counts[selected_terrain, :, coastal, forest].sum()
            if n < 10:
                continue

            probs = learned[selected_terrain, :, coastal, forest]  # (4, 6)
            fig = go.Figure(data=go.Heatmap(
                z=probs,
                x=CLASS_NAMES,
                y=dist_labels,
                colorscale="Blues",
                zmin=0, zmax=1,
                text=np.round(probs, 3).astype(str),
                texttemplate="%{text}",
                colorbar=dict(title="Prob"),
            ))
            fig.update_layout(
                title=f"{label} (n={int(n)})",
                height=250,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def compute_score(prediction, ground_truth):
    """Entropy-weighted KL divergence score."""
    p = np.array(ground_truth, dtype=np.float64)
    q = np.array(prediction, dtype=np.float64)
    p_safe = np.clip(p, 1e-15, None)
    q_safe = np.clip(q, 1e-15, None)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=-1)
    kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)
    total_entropy = entropy.sum()
    if total_entropy < 1e-10:
        return 100.0
    weighted_kl = (entropy * kl).sum() / total_entropy
    return max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Astar Island Dashboard", layout="wide", page_icon="🏝️")
    st.title("Astar Island Dashboard")

    token = get_token()
    if not token:
        st.error("No ASTAR_TOKEN found. Set the `ASTAR_TOKEN` environment variable or create a `.env` file in the AstarIsland directory.")
        st.stop()

    # Sidebar: round and seed selection
    st.sidebar.header("Round Selection")

    try:
        my_rounds = fetch_my_rounds(token)
    except Exception as e:
        st.error(f"Failed to fetch rounds: {e}")
        st.stop()

    if not my_rounds:
        st.warning("No rounds available.")
        st.stop()

    # Sort by round number descending
    my_rounds_sorted = sorted(my_rounds, key=lambda r: r["round_number"], reverse=True)

    round_options = {
        f"Round {r['round_number']} ({r['status']}) — Score: {r.get('round_score', 'N/A')}": r
        for r in my_rounds_sorted
    }

    selected_label = st.sidebar.selectbox("Round", list(round_options.keys()))
    selected_round = round_options[selected_label]
    round_id = selected_round["id"]

    # Fetch full detail
    try:
        detail = fetch_round_detail(token, round_id)
    except Exception as e:
        st.error(f"Failed to fetch round detail: {e}")
        st.stop()

    map_w = detail["map_width"]
    map_h = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    seed_idx = st.sidebar.selectbox("Seed", list(range(seeds_count)), format_func=lambda i: f"Seed {i + 1}")

    initial_state = initial_states[seed_idx]
    grid = np.array(initial_state["grid"])
    settlements = initial_state["settlements"]

    # Round info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Map size:** {map_w} x {map_h}")
    st.sidebar.markdown(f"**Seeds:** {seeds_count}")
    st.sidebar.markdown(f"**Status:** {selected_round['status']}")
    st.sidebar.markdown(f"**Queries used:** {selected_round.get('queries_used', '?')}/{selected_round.get('queries_max', '?')}")
    if selected_round.get("round_score") is not None:
        st.sidebar.markdown(f"**Round score:** {selected_round['round_score']:.2f}")
        st.sidebar.markdown(f"**Rank:** {selected_round.get('rank', '?')}/{selected_round.get('total_teams', '?')}")
    if selected_round.get("seed_scores"):
        scores = selected_round["seed_scores"]
        st.sidebar.markdown(f"**Seed {seed_idx + 1} score:** {scores[seed_idx]:.2f}" if seed_idx < len(scores) and scores[seed_idx] is not None else "")

    # Budget
    try:
        budget = fetch_budget(token)
        if budget and budget.get("active"):
            st.sidebar.markdown(f"**Active budget:** {budget['queries_used']}/{budget['queries_max']}")
    except Exception:
        pass

    # -------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------
    tabs = st.tabs([
        "Initial Map",
        "Observations (Cached)",
        "Predictions",
        "Ground Truth & Analysis",
        "Per-Class Probabilities",
        "Learned Priors",
        "Leaderboard",
    ])

    # === TAB 1: Initial Map ===
    with tabs[0]:
        st.subheader(f"Initial Terrain — Seed {seed_idx + 1}")

        col1, col2 = st.columns(2)
        with col1:
            fig_terrain = make_terrain_heatmap(grid, title=f"Terrain Grid (Seed {seed_idx + 1})")
            st.plotly_chart(fig_terrain, use_container_width=True)

        with col2:
            fig_dist = make_settlement_distance_heatmap(settlements, map_h, map_w)
            st.plotly_chart(fig_dist, use_container_width=True)

        # Settlement info
        st.subheader("Settlements")
        if settlements:
            cols_data = []
            for i, s in enumerate(settlements):
                cols_data.append({
                    "Index": i,
                    "X": s["x"],
                    "Y": s["y"],
                    "Has Port": s.get("has_port", False),
                    "Alive": s.get("alive", True),
                })
            st.dataframe(cols_data, use_container_width=True)
        else:
            st.info("No settlements in this seed.")

        # Terrain distribution
        st.subheader("Terrain Distribution")
        unique, cnts = np.unique(grid, return_counts=True)
        terrain_dist = {TERRAIN_NAMES.get(int(v), f"Type {v}"): int(c) for v, c in zip(unique, cnts)}
        fig_bar = go.Figure(data=go.Bar(
            x=list(terrain_dist.keys()),
            y=list(terrain_dist.values()),
            marker_color=[INITIAL_TERRAIN_COLORS.get(int(v), "#999") for v in unique],
        ))
        fig_bar.update_layout(title="Cell Count by Terrain Type", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # === TAB 2: Cached Observations ===
    with tabs[1]:
        st.subheader(f"Cached Observations — Seed {seed_idx + 1}")

        cached = load_cached_observations(round_id, seed_idx, map_h, map_w)
        if cached is None:
            st.info("No cached observations for this seed. Run `python solve.py` to generate queries.")
        else:
            obs_counts, obs_obs, n_queries = cached
            st.metric("Cached Queries", n_queries)

            col1, col2 = st.columns(2)
            with col1:
                # Observation count heatmap — how many times each cell was observed
                hover_obs = np.empty((map_h, map_w), dtype=object)
                for y in range(map_h):
                    for x in range(map_w):
                        hover_obs[y, x] = f"({x}, {y}): {obs_obs[y, x]} observations"

                fig_obs_count = go.Figure(data=go.Heatmap(
                    z=obs_obs,
                    hovertext=hover_obs,
                    hoverinfo="text",
                    colorscale="Blues",
                    colorbar=dict(title="# Obs"),
                ))
                fig_obs_count.update_layout(
                    title="Observation Coverage (query count per cell)",
                    yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
                    xaxis=dict(constrain="domain"),
                    height=500,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig_obs_count, use_container_width=True)

            with col2:
                # Empirical argmax from observations
                observed_mask = obs_obs > 0
                empirical_argmax = np.zeros((map_h, map_w), dtype=int)
                empirical_conf = np.zeros((map_h, map_w), dtype=float)
                for y in range(map_h):
                    for x in range(map_w):
                        if obs_obs[y, x] > 0:
                            probs = obs_counts[y, x] / obs_obs[y, x]
                            empirical_argmax[y, x] = probs.argmax()
                            empirical_conf[y, x] = probs.max()

                fig_emp = make_class_heatmap(
                    empirical_argmax, title="Empirical Class (from observations)",
                    confidence=empirical_conf,
                )
                st.plotly_chart(fig_emp, use_container_width=True)

            # Coverage stats
            total_cells = map_h * map_w
            observed_cells = (obs_obs > 0).sum()
            st.markdown(f"**Coverage:** {observed_cells}/{total_cells} cells observed ({100*observed_cells/total_cells:.1f}%)")
            st.markdown(f"**Avg observations per observed cell:** {obs_obs[obs_obs > 0].mean():.1f}" if observed_cells > 0 else "")

            # Show raw query viewports on the map
            raw_queries = load_cached_raw_queries(round_id, seed_idx)
            if raw_queries:
                st.subheader("Query Viewports")
                fig_vp = make_terrain_heatmap(grid, title="Viewports on Terrain", show_legend=False)

                colors = px.colors.qualitative.Set2
                for i, q in enumerate(raw_queries):
                    vp = q["viewport"]
                    vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
                    color = colors[i % len(colors)]
                    # Draw rectangle
                    fig_vp.add_shape(
                        type="rect",
                        x0=vx - 0.5, y0=vy - 0.5,
                        x1=vx + vw - 0.5, y1=vy + vh - 0.5,
                        line=dict(color=color, width=2),
                    )
                    fig_vp.add_annotation(
                        x=vx + vw / 2, y=vy - 1,
                        text=f"Q{i+1}", showarrow=False,
                        font=dict(size=10, color=color),
                    )

                fig_vp.update_layout(height=550)
                st.plotly_chart(fig_vp, use_container_width=True)

    # === TAB 3: Predictions ===
    with tabs[2]:
        st.subheader(f"Predictions — Seed {seed_idx + 1}")

        try:
            predictions = fetch_my_predictions(token, round_id)
        except Exception as e:
            st.error(f"Failed to fetch predictions: {e}")
            predictions = []

        seed_pred = next((p for p in predictions if p["seed_index"] == seed_idx), None)

        if seed_pred is None:
            st.info("No prediction submitted for this seed yet.")
        else:
            if seed_pred.get("score") is not None:
                st.metric("Seed Score", f"{seed_pred['score']:.2f}")
            if seed_pred.get("submitted_at"):
                st.caption(f"Submitted at: {seed_pred['submitted_at']}")

            argmax_grid = np.array(seed_pred["argmax_grid"])
            confidence_grid = np.array(seed_pred["confidence_grid"])

            col1, col2 = st.columns(2)
            with col1:
                fig_pred = make_class_heatmap(
                    argmax_grid, title="Predicted Classes (argmax)",
                    confidence=confidence_grid,
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with col2:
                fig_conf = make_confidence_heatmap(
                    confidence_grid, title="Prediction Confidence",
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # Prediction class distribution
            st.subheader("Predicted Class Distribution")
            unique_p, cnts_p = np.unique(argmax_grid, return_counts=True)
            pred_dist = {CLASS_NAMES[int(v)]: int(c) for v, c in zip(unique_p, cnts_p)}
            fig_bar_pred = go.Figure(data=go.Bar(
                x=list(pred_dist.keys()),
                y=list(pred_dist.values()),
                marker_color=[CLASS_COLORS[int(v)] for v in unique_p],
            ))
            fig_bar_pred.update_layout(title="Predicted Cell Count by Class", height=350)
            st.plotly_chart(fig_bar_pred, use_container_width=True)

            # Confidence distribution
            st.subheader("Confidence Distribution")
            fig_hist = go.Figure(data=go.Histogram(
                x=confidence_grid.flatten(),
                nbinsx=50,
                marker_color="#2196F3",
            ))
            fig_hist.update_layout(
                title="Confidence Value Distribution",
                xaxis_title="Confidence", yaxis_title="Count",
                height=300,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # === TAB 4: Ground Truth & Analysis ===
    with tabs[3]:
        st.subheader(f"Ground Truth & Analysis — Seed {seed_idx + 1}")

        if selected_round["status"] not in ("completed", "scoring"):
            st.info("Ground truth is only available after the round is completed.")
        else:
            analysis = fetch_analysis(token, round_id, seed_idx)
            if analysis is None:
                st.warning("Could not fetch analysis. Make sure you submitted a prediction for this seed.")
            else:
                gt = np.array(analysis["ground_truth"])
                gt_argmax = gt.argmax(axis=-1)
                gt_confidence = gt.max(axis=-1)

                if analysis.get("score") is not None:
                    st.metric("Score", f"{analysis['score']:.2f}")

                col1, col2 = st.columns(2)
                with col1:
                    fig_gt = make_class_heatmap(gt_argmax, title="Ground Truth (argmax)")
                    st.plotly_chart(fig_gt, use_container_width=True)

                with col2:
                    fig_gt_ent = make_entropy_heatmap(gt, title="Ground Truth Entropy")
                    st.plotly_chart(fig_gt_ent, use_container_width=True)

                # If we have prediction too, show comparison
                pred_data = analysis.get("prediction")
                if pred_data is not None:
                    pred = np.array(pred_data)
                    pred_argmax = pred.argmax(axis=-1)

                    st.subheader("Prediction vs Ground Truth")

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_diff = make_diff_heatmap(pred_argmax, gt_argmax)
                        st.plotly_chart(fig_diff, use_container_width=True)

                    with col2:
                        fig_kl = make_kl_heatmap(pred, gt)
                        st.plotly_chart(fig_kl, use_container_width=True)

                    # Accuracy summary
                    total = gt_argmax.size
                    correct = (pred_argmax == gt_argmax).sum()
                    accuracy = correct / total

                    st.subheader("Accuracy Breakdown")
                    st.metric("Overall Argmax Accuracy", f"{accuracy:.1%} ({correct}/{total})")

                    # Per-class accuracy
                    class_data = []
                    for cls_idx in range(NUM_CLASSES):
                        mask = gt_argmax == cls_idx
                        n = mask.sum()
                        if n > 0:
                            cls_correct = (pred_argmax[mask] == cls_idx).sum()
                            class_data.append({
                                "Class": CLASS_NAMES[cls_idx],
                                "GT Count": int(n),
                                "Correct": int(cls_correct),
                                "Accuracy": f"{cls_correct / n:.1%}",
                            })
                        else:
                            class_data.append({
                                "Class": CLASS_NAMES[cls_idx],
                                "GT Count": 0,
                                "Correct": 0,
                                "Accuracy": "N/A",
                            })
                    st.dataframe(class_data, use_container_width=True)

                    # Confusion-style: what did we predict for GT class X?
                    st.subheader("Confusion Analysis")
                    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
                    for y in range(map_h):
                        for x in range(map_w):
                            confusion[gt_argmax[y, x], pred_argmax[y, x]] += 1

                    fig_conf_mat = go.Figure(data=go.Heatmap(
                        z=confusion,
                        x=CLASS_NAMES,
                        y=CLASS_NAMES,
                        text=confusion.astype(str),
                        texttemplate="%{text}",
                        colorscale="Blues",
                        colorbar=dict(title="Count"),
                    ))
                    fig_conf_mat.update_layout(
                        title="Confusion Matrix (rows=GT, cols=Predicted)",
                        xaxis_title="Predicted",
                        yaxis_title="Ground Truth",
                        height=450,
                    )
                    st.plotly_chart(fig_conf_mat, use_container_width=True)

                    # Side-by-side entropy
                    st.subheader("Entropy Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pred_ent = make_entropy_heatmap(pred, title="Prediction Entropy")
                        st.plotly_chart(fig_pred_ent, use_container_width=True)
                    with col2:
                        fig_gt_ent2 = make_entropy_heatmap(gt, title="Ground Truth Entropy")
                        st.plotly_chart(fig_gt_ent2, use_container_width=True)

    # === TAB 5: Per-Class Probabilities ===
    with tabs[4]:
        st.subheader(f"Per-Class Probability Maps — Seed {seed_idx + 1}")

        # Try to get prediction or ground truth probabilities
        source = st.radio("Source", ["Prediction", "Ground Truth"], horizontal=True, key="prob_source")

        prob_data = None
        if source == "Prediction":
            try:
                analysis = fetch_analysis(token, round_id, seed_idx)
                if analysis and analysis.get("prediction"):
                    prob_data = np.array(analysis["prediction"])
            except Exception:
                pass
        else:
            try:
                analysis = fetch_analysis(token, round_id, seed_idx)
                if analysis and analysis.get("ground_truth"):
                    prob_data = np.array(analysis["ground_truth"])
            except Exception:
                pass

        if prob_data is None:
            st.info("Probability data not available. This requires a completed round with a submitted prediction.")
        else:
            # Show all 6 class probability maps in a 2x3 grid
            for row_start in range(0, NUM_CLASSES, 3):
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    cls_idx = row_start + i
                    if cls_idx < NUM_CLASSES:
                        with col:
                            fig = make_probability_heatmap(
                                prob_data, cls_idx,
                                title=f"P({CLASS_NAMES[cls_idx]})",
                            )
                            st.plotly_chart(fig, use_container_width=True)

            # Cell inspector
            st.subheader("Cell Inspector")
            col1, col2 = st.columns(2)
            with col1:
                cx = st.number_input("X coordinate", min_value=0, max_value=map_w - 1, value=0, key="cell_x")
            with col2:
                cy = st.number_input("Y coordinate", min_value=0, max_value=map_h - 1, value=0, key="cell_y")

            cell_probs = prob_data[cy, cx]
            fig_cell = go.Figure(data=go.Bar(
                x=CLASS_NAMES,
                y=cell_probs,
                marker_color=[CLASS_COLORS[i] for i in range(NUM_CLASSES)],
            ))
            fig_cell.update_layout(
                title=f"Class Probabilities at ({cx}, {cy}) — Initial terrain: {TERRAIN_NAMES.get(int(grid[cy, cx]), '?')}",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=350,
            )
            st.plotly_chart(fig_cell, use_container_width=True)

    # === TAB 6: Learned Priors ===
    with tabs[5]:
        st.subheader("Learned Priors (from historical rounds)")
        show_learned_priors_section()

    # === TAB 7: Leaderboard ===
    with tabs[6]:
        st.subheader("Leaderboard")
        try:
            lb = fetch_leaderboard()
            if lb:
                lb_data = []
                for entry in lb:
                    lb_data.append({
                        "Rank": entry.get("rank", "?"),
                        "Team": entry.get("team_name", "?"),
                        "Score": f"{entry.get('weighted_score', 0):.2f}",
                        "Hot Streak": f"{entry.get('hot_streak_score', 0):.2f}",
                        "Rounds": entry.get("rounds_participated", 0),
                        "Verified": entry.get("is_verified", False),
                    })
                st.dataframe(lb_data, use_container_width=True)

                # Bar chart
                names = [e.get("team_name", "?") for e in lb]
                scores = [e.get("weighted_score", 0) for e in lb]
                fig_lb = go.Figure(data=go.Bar(
                    x=names, y=scores,
                    marker_color="#FF9800",
                ))
                fig_lb.update_layout(
                    title="Leaderboard Scores",
                    xaxis_title="Team", yaxis_title="Score",
                    height=400,
                )
                st.plotly_chart(fig_lb, use_container_width=True)
            else:
                st.info("No leaderboard data available.")
        except Exception as e:
            st.error(f"Failed to fetch leaderboard: {e}")

    # === All-seeds overview (sidebar) ===
    if selected_round.get("seed_scores"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("All Seed Scores")
        scores = selected_round["seed_scores"]
        for i, s in enumerate(scores):
            if s is not None:
                emoji = "🟢" if s >= 70 else "🟡" if s >= 50 else "🔴"
                st.sidebar.markdown(f"{emoji} Seed {i + 1}: **{s:.2f}**")
            else:
                st.sidebar.markdown(f"⚪ Seed {i + 1}: *not scored*")


if __name__ == "__main__":
    main()
