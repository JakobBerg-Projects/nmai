#!/usr/bin/env python3
"""
Astar Island — Optimized Viking Civilisation Prediction Solver

Strategy:
- Learn transition probabilities from completed rounds' ground truth
- Static terrain (ocean, mountains) is known — no queries needed
- Focus queries on settlement-dense areas using greedy viewport selection
- Repeated sampling per viewport for better probability estimates
- Distance-aware priors conditioned on (terrain, distance, coastal, forest_adj)
- Cross-seed learning of terrain transition statistics

Usage:
    export ASTAR_TOKEN="your_jwt_token"
    python solve.py              # run against active round
    python solve.py --backtest   # test against most recent completed round
    python solve.py --backtest 1 # test against specific round number
    python solve.py --learn      # learn priors from all completed rounds
"""

import argparse
import json
import os
import time
import numpy as np
import requests

BASE = "https://api.ainm.no"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

TERRAIN_TO_CLASS = {
    0: 0,   # Empty → Empty
    10: 0,  # Ocean → Empty
    11: 0,  # Plains → Empty
    1: 1,   # Settlement
    2: 2,   # Port
    3: 3,   # Ruin
    4: 4,   # Forest
    5: 5,   # Mountain
}

NUM_CLASSES = 6
PROB_FLOOR = 0.01
PRIOR_STRENGTH = 2.0  # Prior worth this many pseudo-observations

NUM_DIST_BUCKETS = 4
LEARNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "learned_priors.npz")


# ---------------------------------------------------------------------------
# Query result cache — persists observations to disk after every query
# ---------------------------------------------------------------------------

def _round_cache_dir(round_id):
    """Return cache directory for a specific round, creating it if needed."""
    d = os.path.join(CACHE_DIR, round_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_query_result(round_id, seed_index, query_index, result):
    """Save a single raw query result as JSON."""
    d = _round_cache_dir(round_id)
    path = os.path.join(d, f"query_seed{seed_index}_{query_index:03d}.json")
    with open(path, "w") as f:
        json.dump(result, f)


def save_observations(round_id, seed_index, counts, obs_count):
    """Save accumulated observation arrays for a seed."""
    d = _round_cache_dir(round_id)
    path = os.path.join(d, f"obs_seed{seed_index}.npz")
    np.savez(path, counts=counts, obs_count=obs_count)
    print(f"  [cache] Saved observations for seed {seed_index}")


def load_observations(round_id, seed_index, map_h, map_w):
    """Load cached observations for a seed. Returns (counts, obs_count, n_queries) or None."""
    d = os.path.join(CACHE_DIR, round_id)
    path = os.path.join(d, f"obs_seed{seed_index}.npz")
    if os.path.exists(path):
        data = np.load(path)
        counts = data["counts"]
        obs_count = data["obs_count"]
        if counts.shape == (map_h, map_w, NUM_CLASSES) and obs_count.shape == (map_h, map_w):
            # Count how many raw query files exist for this seed
            n = len([f for f in os.listdir(d)
                     if f.startswith(f"query_seed{seed_index}_") and f.endswith(".json")])
            return counts, obs_count, n
    return None


def count_cached_queries(round_id):
    """Count total cached queries across all seeds for a round."""
    d = os.path.join(CACHE_DIR, round_id)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.startswith("query_seed") and f.endswith(".json")])


def distance_bucket(d):
    """Bucket distance: 0=on-site, 1=near(1-2), 2=medium(3-5), 3=far(6+)."""
    if d <= 0: return 0
    if d <= 2: return 1
    if d <= 5: return 2
    return 3


# ---------------------------------------------------------------------------
# Auth & round fetching
# ---------------------------------------------------------------------------

def get_session():
    """Create authenticated session."""
    token = os.environ.get("ASTAR_TOKEN", "")
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line.startswith("ASTAR_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not token:
        raise ValueError("Set ASTAR_TOKEN env var or create .env file")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def get_round(session, backtest_round=None):
    """Fetch round details."""
    rounds = session.get(f"{BASE}/astar-island/rounds").json()

    if backtest_round is not None:
        if backtest_round > 0:
            target = next((r for r in rounds if r["round_number"] == backtest_round), None)
        else:
            completed = sorted(
                [r for r in rounds if r["status"] == "completed"],
                key=lambda r: r["round_number"], reverse=True,
            )
            target = completed[0] if completed else None

        if not target:
            print("Available rounds:")
            for r in rounds:
                print(f"  Round {r['round_number']}: {r['status']}")
            raise RuntimeError("No matching completed round found")

        detail = session.get(f"{BASE}/astar-island/rounds/{target['id']}").json()
        print(f"Backtest round {target['round_number']} ({target['status']}): "
              f"{detail['map_width']}x{detail['map_height']}, {detail['seeds_count']} seeds")
        return detail, True
    else:
        active = next((r for r in rounds if r["status"] == "active"), None)
        if not active:
            print("No active round. Available rounds:")
            for r in rounds:
                print(f"  Round {r['round_number']}: {r['status']}")
            raise RuntimeError("No active round. Use --backtest for completed rounds.")

        detail = session.get(f"{BASE}/astar-island/rounds/{active['id']}").json()
        print(f"Active round {active['round_number']}: "
              f"{detail['map_width']}x{detail['map_height']}, {detail['seeds_count']} seeds")
        return detail, False


# ---------------------------------------------------------------------------
# Query execution with robust error handling
# ---------------------------------------------------------------------------

def simulate_query(session, round_id, seed_index, vx, vy, vw, vh, max_retries=10):
    """Run one simulation query. Returns None if budget exhausted."""
    for attempt in range(max_retries):
        resp = session.post(f"{BASE}/astar-island/simulate", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": vx,
            "viewport_y": vy,
            "viewport_w": vw,
            "viewport_h": vh,
        })
        if resp.status_code != 429:
            break
        text = resp.text.lower()
        if "budget" in text or "exhausted" in text or "limit" in text:
            try:
                data = resp.json()
                used = data.get("queries_used", 0)
                mx = data.get("queries_max", 50)
                if used >= mx:
                    print(f"  Budget exhausted ({used}/{mx})")
                    return None
            except Exception:
                pass
        wait = min(2 * (attempt + 1), 10)
        print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(wait)
    else:
        print("  Max retries exceeded on simulate query")
        return None
    resp.raise_for_status()
    return resp.json()


def check_budget(session):
    """Return (used, max) query budget."""
    try:
        data = session.get(f"{BASE}/astar-island/budget").json()
        return data.get("queries_used", 0), data.get("queries_max", 50)
    except Exception:
        return 0, 50


# ---------------------------------------------------------------------------
# Focused viewport selection — greedy set cover on settlements
# ---------------------------------------------------------------------------

def find_settlement_viewports(initial_state, map_w, map_h, max_viewports=3):
    """Greedy: find viewports covering the most settlements + neighborhoods."""
    settlements = initial_state["settlements"]
    if not settlements:
        cx, cy = map_w // 2, map_h // 2
        return [(max(0, cx - 7), max(0, cy - 7), min(15, map_w), min(15, map_h))]

    uncovered = set(range(len(settlements)))
    viewports = []

    while uncovered and len(viewports) < max_viewports:
        best_vp, best_count = None, 0

        for vy in range(max(1, map_h - 14)):
            for vx in range(max(1, map_w - 14)):
                vw = min(15, map_w - vx)
                vh = min(15, map_h - vy)
                count = sum(
                    1 for i in uncovered
                    if vx <= settlements[i]["x"] < vx + vw
                    and vy <= settlements[i]["y"] < vy + vh
                )
                if count > best_count:
                    best_count = count
                    best_vp = (vx, vy, vw, vh)

        if best_count == 0:
            break

        viewports.append(best_vp)
        vx, vy, vw, vh = best_vp
        uncovered -= {
            i for i in uncovered
            if vx <= settlements[i]["x"] < vx + vw
            and vy <= settlements[i]["y"] < vy + vh
        }

    return viewports


# ---------------------------------------------------------------------------
# Observation — focused viewports with repeated sampling
# ---------------------------------------------------------------------------

def observe_seed(session, round_id, seed_index, initial_state, map_w, map_h, query_budget):
    """Focus queries on settlement areas, repeat for better statistics.
    Saves every query result and accumulated observations to disk."""
    # Try to resume from cache
    cached = load_observations(round_id, seed_index, map_h, map_w)
    if cached is not None:
        counts, obs_count, n_cached = cached
        print(f"  Seed {seed_index}: loaded {n_cached} cached queries from disk")
        if n_cached >= query_budget:
            print(f"  Seed {seed_index}: already have {n_cached} queries cached, skipping")
            return counts, obs_count
        # Continue from where we left off
        queries_done = n_cached
    else:
        counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
        obs_count = np.zeros((map_h, map_w), dtype=np.int32)
        queries_done = 0

    max_vp = min(3, max(1, query_budget // 3))
    viewports = find_settlement_viewports(initial_state, map_w, map_h, max_viewports=max_vp)

    if not viewports:
        print(f"  Seed {seed_index}: no settlements, skipping queries")
        return counts, obs_count

    remaining = query_budget - queries_done
    print(f"  Seed {seed_index}: {len(viewports)} viewports, {remaining} new queries (of {query_budget})")

    if remaining <= 0:
        return counts, obs_count

    # Distribute remaining queries evenly across viewports (round-robin)
    vp_reps = [0] * len(viewports)
    for i in range(remaining):
        vp_reps[i % len(viewports)] += 1

    q = 0
    result = None
    for vp_idx, (vx, vy, vw, vh) in enumerate(viewports):
        reps = vp_reps[vp_idx]
        for rep in range(reps):
            q += 1
            print(f"    [{queries_done + q}/{query_budget}] ({vx},{vy}) {vw}x{vh} rep {rep+1}/{reps}")

            result = simulate_query(session, round_id, seed_index, vx, vy, vw, vh)
            if result is None:
                print("  Budget exhausted, stopping observations")
                save_observations(round_id, seed_index, counts, obs_count)
                return counts, obs_count

            # Save raw query result immediately
            save_query_result(round_id, seed_index, queries_done + q, result)

            vp = result["viewport"]
            for ri, row in enumerate(result["grid"]):
                for ci, val in enumerate(row):
                    gy, gx = vp["y"] + ri, vp["x"] + ci
                    if 0 <= gy < map_h and 0 <= gx < map_w:
                        counts[gy, gx, TERRAIN_TO_CLASS.get(val, 0)] += 1
                        obs_count[gy, gx] += 1

            # Save accumulated observations after every query
            save_observations(round_id, seed_index, counts, obs_count)

            time.sleep(0.25)

    if result is not None:
        used = result.get("queries_used", "?")
        mx = result.get("queries_max", "?")
        print(f"  Budget: {used}/{mx}")

    return counts, obs_count


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def compute_settlement_distance(settlements, map_h, map_w):
    """Manhattan distance to nearest settlement for each cell."""
    dist = np.full((map_h, map_w), 999.0)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for y in range(map_h):
            for x in range(map_w):
                d = abs(x - sx) + abs(y - sy)
                if d < dist[y, x]:
                    dist[y, x] = d
    return dist


def check_adjacent_to_ocean(grid, y, x, map_h, map_w):
    """Check if a cell is adjacent to ocean."""
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < map_h and 0 <= nx < map_w:
            if grid[ny][nx] == 10:
                return True
    return False


def check_adjacent_to_forest(grid, y, x, map_h, map_w):
    """Check if a cell is adjacent to forest."""
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < map_h and 0 <= nx < map_w:
            if grid[ny][nx] == 4:
                return True
    return False


# ---------------------------------------------------------------------------
# Historical learning — learn priors from completed rounds
# ---------------------------------------------------------------------------

def learn_from_history(session):
    """Fetch all completed rounds and learn conditional transition priors.

    Learns P(final_class | initial_terrain, distance_bucket, coastal, forest_adj)
    from ground truth of all completed rounds.
    """
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    completed = sorted(
        [r for r in rounds if r["status"] == "completed"],
        key=lambda r: r["round_number"],
    )

    if not completed:
        print("No completed rounds to learn from")
        return

    print(f"Learning from {len(completed)} completed rounds...")

    # Accumulator: (initial_class, dist_bucket, coastal, forest_adj, final_class)
    accum = np.zeros((NUM_CLASSES, NUM_DIST_BUCKETS, 2, 2, NUM_CLASSES), dtype=np.float64)
    counts = np.zeros((NUM_CLASSES, NUM_DIST_BUCKETS, 2, 2), dtype=np.float64)

    for rnd in completed:
        detail = session.get(f"{BASE}/astar-island/rounds/{rnd['id']}").json()
        map_w = detail["map_width"]
        map_h = detail["map_height"]
        seeds_count = detail["seeds_count"]
        initial_states = detail["initial_states"]

        print(f"  Round {rnd['round_number']}: {map_w}x{map_h}, {seeds_count} seeds")

        for seed_idx in range(seeds_count):
            resp = session.get(f"{BASE}/astar-island/analysis/{rnd['id']}/{seed_idx}")
            if resp.status_code != 200:
                print(f"    Seed {seed_idx}: no ground truth ({resp.status_code})")
                continue

            gt = np.array(resp.json()["ground_truth"], dtype=np.float64)
            initial = initial_states[seed_idx]
            grid = initial["grid"]
            settlements = initial["settlements"]
            dist = compute_settlement_distance(settlements, map_h, map_w)

            for y in range(map_h):
                for x in range(map_w):
                    icls = TERRAIN_TO_CLASS.get(grid[y][x], 0)
                    db = distance_bucket(dist[y, x])
                    coastal = int(check_adjacent_to_ocean(grid, y, x, map_h, map_w))
                    forest = int(check_adjacent_to_forest(grid, y, x, map_h, map_w))

                    accum[icls, db, coastal, forest] += gt[y, x]
                    counts[icls, db, coastal, forest] += 1.0

            time.sleep(0.2)

    # Average and normalize
    learned = np.full((NUM_CLASSES, NUM_DIST_BUCKETS, 2, 2, NUM_CLASSES), 1.0 / NUM_CLASSES)
    valid = counts > 5
    for idx in np.argwhere(valid):
        i, d, c, f = tuple(idx)
        avg = accum[i, d, c, f] / counts[i, d, c, f]
        avg = np.maximum(avg, PROB_FLOOR)
        avg /= avg.sum()
        learned[i, d, c, f] = avg

    np.savez(LEARNED_MODEL_PATH, learned=learned, counts=counts)

    total_cells = int(counts.sum())
    valid_bins = int(valid.sum())
    total_bins = counts.size
    print(f"\nLearned model saved to {LEARNED_MODEL_PATH}")
    print(f"  {total_cells} cells across all rounds/seeds")
    print(f"  {valid_bins}/{total_bins} feature bins with sufficient data")

    # Print summary of learned transitions
    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    dist_labels = ["on-site", "near(1-2)", "medium(3-5)", "far(6+)"]
    print(f"\nLearned transition summary:")
    for icls in range(NUM_CLASSES):
        for db in range(NUM_DIST_BUCKETS):
            c = counts[icls, db].sum()
            if c > 50:
                avg = accum[icls, db].sum(axis=(0, 1)) / max(1, counts[icls, db].sum())
                avg = np.maximum(avg, PROB_FLOOR)
                avg /= avg.sum()
                top3 = np.argsort(avg)[::-1][:3]
                desc = ", ".join(f"{class_names[t]}={avg[t]:.2f}" for t in top3)
                print(f"  {class_names[icls]} dist={dist_labels[db]}: {desc} (n={int(c)})")


def load_learned_priors():
    """Load learned prior model if available. Returns (learned, counts) or (None, None)."""
    if os.path.exists(LEARNED_MODEL_PATH):
        data = np.load(LEARNED_MODEL_PATH)
        print(f"Loaded learned priors from {LEARNED_MODEL_PATH}")
        return data["learned"], data["counts"]
    return None, None


# ---------------------------------------------------------------------------
# Prior model — learned or hand-coded fallback
# ---------------------------------------------------------------------------

def build_initial_prior(initial_state, map_h, map_w, learned_model=None):
    """Build prior from learned model (if available) or hand-coded fallback."""
    grid = initial_state["grid"]
    settlements = initial_state["settlements"]
    prior = np.full((map_h, map_w, NUM_CLASSES), PROB_FLOOR, dtype=np.float32)
    is_static = np.zeros((map_h, map_w), dtype=bool)

    dist = compute_settlement_distance(settlements, map_h, map_w)

    learned, learned_counts = None, None
    if learned_model is not None:
        learned, learned_counts = learned_model

    for y in range(map_h):
        for x in range(map_w):
            cell = grid[y][x]
            icls = TERRAIN_TO_CLASS.get(cell, 0)
            d = dist[y, x]
            db = distance_bucket(d)
            coastal = int(check_adjacent_to_ocean(grid, y, x, map_h, map_w))
            forest = int(check_adjacent_to_forest(grid, y, x, map_h, map_w))

            if cell == 10:  # Ocean — never changes
                prior[y, x, 0] = 0.98
                is_static[y, x] = True

            elif cell == 5:  # Mountain — never changes
                prior[y, x, 5] = 0.98
                is_static[y, x] = True

            elif learned is not None and learned_counts[icls, db, coastal, forest] > 5:
                # Use data-driven prior from historical rounds
                prior[y, x] = learned[icls, db, coastal, forest]

            else:
                # Hand-coded fallback
                if cell == 4:  # Forest
                    if d <= 2:
                        prior[y, x] = [0.05, 0.25, 0.02, 0.01, 0.60, PROB_FLOOR]
                    elif d <= 4:
                        prior[y, x] = [0.05, 0.16, 0.01, 0.01, 0.70, PROB_FLOOR]
                    else:
                        prior[y, x] = [0.04, 0.02, PROB_FLOOR, PROB_FLOOR, 0.88, PROB_FLOOR]

                elif cell in (0, 11):  # Empty/Plains
                    if d <= 2:
                        prior[y, x] = [0.55, 0.30, 0.02, 0.02, 0.03, PROB_FLOOR]
                    elif d <= 5:
                        prior[y, x] = [0.75, 0.15, 0.01, 0.01, 0.02, PROB_FLOOR]
                    else:
                        prior[y, x] = [0.92, 0.03, PROB_FLOOR, PROB_FLOOR, 0.01, PROB_FLOOR]

                elif cell == 1:  # Settlement
                    has_coast = check_adjacent_to_ocean(grid, y, x, map_h, map_w)
                    has_food = check_adjacent_to_forest(grid, y, x, map_h, map_w)
                    if has_coast and has_food:
                        prior[y, x] = [0.25, 0.45, 0.05, 0.04, 0.14, PROB_FLOOR]
                    elif has_coast:
                        prior[y, x] = [0.30, 0.40, 0.05, 0.04, 0.14, PROB_FLOOR]
                    elif has_food:
                        prior[y, x] = [0.30, 0.48, 0.02, 0.04, 0.10, PROB_FLOOR]
                    else:
                        prior[y, x] = [0.38, 0.40, 0.02, 0.04, 0.10, PROB_FLOOR]

                elif cell == 2:  # Port
                    prior[y, x] = [0.34, 0.21, 0.28, 0.06, 0.13, PROB_FLOOR]

                elif cell == 3:  # Ruin
                    if d <= 3:
                        prior[y, x] = [0.30, 0.15, 0.02, 0.15, 0.30, PROB_FLOOR]
                    else:
                        prior[y, x] = [0.35, 0.05, PROB_FLOOR, 0.10, 0.42, PROB_FLOOR]

            # Floor and normalize
            prior[y, x] = np.maximum(prior[y, x], PROB_FLOOR)
            prior[y, x] /= prior[y, x].sum()

    return prior, is_static


# ---------------------------------------------------------------------------
# Cross-seed transition learning (within current round)
# ---------------------------------------------------------------------------

def learn_transitions(all_counts, all_obs, initial_states, map_h, map_w):
    """Learn per-initial-terrain-type transition probabilities across all seeds."""
    trans_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    trans_total = np.zeros(NUM_CLASSES, dtype=np.float64)

    for si in range(len(initial_states)):
        grid = initial_states[si]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                if all_obs[si][y, x] > 0:
                    src = TERRAIN_TO_CLASS.get(grid[y][x], 0)
                    trans_counts[src] += all_counts[si][y, x]
                    trans_total[src] += all_obs[si][y, x]

    transitions = {}
    for cls in range(NUM_CLASSES):
        if trans_total[cls] > 10:
            t = trans_counts[cls] / trans_total[cls]
            t = np.maximum(t, PROB_FLOOR)
            t /= t.sum()
            transitions[cls] = t

    return transitions


# ---------------------------------------------------------------------------
# Prediction: Bayesian posterior combining prior + observations + cross-seed
# ---------------------------------------------------------------------------

def build_prediction(counts, obs_count, initial_state, map_h, map_w,
                     transitions=None, learned_model=None):
    """Bayesian posterior with optional cross-seed-learned transitions."""
    prior, is_static = build_initial_prior(initial_state, map_h, map_w, learned_model)
    grid = initial_state["grid"]
    prediction = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)

    for y in range(map_h):
        for x in range(map_w):
            if is_static[y, x]:
                prediction[y, x] = prior[y, x]
            elif obs_count[y, x] > 0:
                # Dirichlet posterior: prior pseudo-counts + observed counts
                n = obs_count[y, x]
                alpha = prior[y, x] * PRIOR_STRENGTH
                prediction[y, x] = (counts[y, x] + alpha) / (n + PRIOR_STRENGTH)
            elif transitions is not None:
                # Use cross-seed learned transitions blended with prior
                src = TERRAIN_TO_CLASS.get(grid[y][x], 0)
                if src in transitions:
                    prediction[y, x] = 0.6 * transitions[src] + 0.4 * prior[y, x]
                else:
                    prediction[y, x] = prior[y, x]
            else:
                prediction[y, x] = prior[y, x]

    # Floor and renormalize
    for _ in range(3):
        prediction = np.maximum(prediction, PROB_FLOOR)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)

    return prediction


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_prediction(session, round_id, seed_index, prediction, max_retries=10):
    """Submit prediction tensor for one seed."""
    for attempt in range(max_retries):
        resp = session.post(f"{BASE}/astar-island/submit", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction.tolist(),
        })
        if resp.status_code != 429:
            break
        wait = min(2 * (attempt + 1), 10)
        print(f"  Rate limited on submit, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(wait)
    else:
        raise RuntimeError(f"Submit failed after {max_retries} retries for seed {seed_index}")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Scoring (local, for backtest)
# ---------------------------------------------------------------------------

def score_prediction(prediction, ground_truth):
    """Compute entropy-weighted KL divergence score (same as server)."""
    p = np.array(ground_truth, dtype=np.float64)
    q = np.array(prediction, dtype=np.float64)

    p_safe = np.clip(p, 1e-15, None)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=-1)

    q_safe = np.clip(q, 1e-15, None)
    kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)

    total_entropy = entropy.sum()
    if total_entropy < 1e-10:
        return 100.0
    weighted_kl = (entropy * kl).sum() / total_entropy

    return max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def backtest(session, detail):
    """Test prediction quality against a completed round using ground truth."""
    round_id = detail["id"]
    map_w = detail["map_width"]
    map_h = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    learned_model = load_learned_priors()
    has_model = learned_model[0] is not None

    scores_learned = []
    scores_handcoded = []
    for seed_idx in range(seeds_count):
        print(f"\n--- Backtesting seed {seed_idx} ---")

        counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
        obs_count = np.zeros((map_h, map_w), dtype=np.int32)

        # Prediction with learned model
        pred_learned = build_prediction(
            counts, obs_count, initial_states[seed_idx], map_h, map_w,
            learned_model=learned_model,
        )
        pred_learned = np.maximum(pred_learned, PROB_FLOOR)
        pred_learned = pred_learned / pred_learned.sum(axis=-1, keepdims=True)

        # Prediction with hand-coded fallback only
        pred_handcoded = build_prediction(
            counts, obs_count, initial_states[seed_idx], map_h, map_w,
        )
        pred_handcoded = np.maximum(pred_handcoded, PROB_FLOOR)
        pred_handcoded = pred_handcoded / pred_handcoded.sum(axis=-1, keepdims=True)

        # Fetch ground truth
        resp = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}")
        if resp.status_code != 200:
            print(f"  Could not fetch ground truth: {resp.status_code} {resp.text[:200]}")
            continue

        analysis = resp.json()
        ground_truth = analysis["ground_truth"]

        score_l = score_prediction(pred_learned, ground_truth)
        score_h = score_prediction(pred_handcoded, ground_truth)
        server_score = analysis.get("score")
        scores_learned.append(score_l)
        scores_handcoded.append(score_h)

        if has_model:
            print(f"  Learned prior score:    {score_l:.2f}")
            print(f"  Hand-coded prior score: {score_h:.2f}")
            print(f"  Improvement: {score_l - score_h:+.2f}")
        else:
            print(f"  Prior-only score: {score_h:.2f}")

        if server_score is not None:
            print(f"  Server score (last submission): {server_score:.2f}")

        # Per-class breakdown
        pred = np.array(pred_learned if has_model else pred_handcoded)
        gt = np.array(ground_truth)
        gt_argmax = gt.argmax(axis=-1)
        pred_argmax = pred.argmax(axis=-1)
        class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
        for cls_idx, cls_name in enumerate(class_names):
            mask = gt_argmax == cls_idx
            if mask.sum() > 0:
                correct = (pred_argmax[mask] == cls_idx).sum()
                print(f"    {cls_name}: {correct}/{mask.sum()} ({100*correct/mask.sum():.0f}%)")

    if scores_learned:
        print(f"\n--- Backtest Summary ---")
        if has_model:
            print(f"Learned prior avg:    {np.mean(scores_learned):.2f}  "
                  f"{[f'{s:.2f}' for s in scores_learned]}")
            print(f"Hand-coded prior avg: {np.mean(scores_handcoded):.2f}  "
                  f"{[f'{s:.2f}' for s in scores_handcoded]}")
        else:
            print(f"Hand-coded prior avg: {np.mean(scores_handcoded):.2f}  "
                  f"{[f'{s:.2f}' for s in scores_handcoded]}")
            print("Run --learn first to compare against learned priors")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Astar Island solver")
    parser.add_argument("--backtest", nargs="?", const=0, type=int, default=None,
                        help="Test against completed round (default: most recent)")
    parser.add_argument("--learn", action="store_true",
                        help="Learn priors from all completed rounds")
    parser.add_argument("--quick", action="store_true",
                        help="Submit prior-only predictions immediately (0 queries), then exit")
    args = parser.parse_args()

    session = get_session()

    if args.learn:
        learn_from_history(session)
        return

    detail, is_backtest = get_round(session, backtest_round=args.backtest)

    round_id = detail["id"]
    map_w = detail["map_width"]
    map_h = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    if is_backtest:
        backtest(session, detail)
        return

    # --- Live mode ---

    # Load learned priors if available
    learned_model = load_learned_priors()

    if args.quick:
        # Quick mode: submit prior-only predictions (0 queries) and exit
        print("\n=== Quick prior-only submission (0 queries) ===")
        for seed_idx in range(seeds_count):
            counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
            obs_count = np.zeros((map_h, map_w), dtype=np.int32)
            prediction = build_prediction(
                counts, obs_count, initial_states[seed_idx], map_h, map_w,
                learned_model=learned_model,
            )
            prediction = np.maximum(prediction, PROB_FLOOR)
            prediction = prediction / prediction.sum(axis=-1, keepdims=True)
            result = submit_prediction(session, round_id, seed_idx, prediction)
            print(f"  Seed {seed_idx}: {result.get('status', 'submitted')} (prior-only)")
        print("\nDone. Run without --quick to use queries for better scores.")
        return

    # Check cache status
    cached_total = count_cached_queries(round_id)
    if cached_total > 0:
        print(f"\n[cache] Found {cached_total} cached queries for this round")

    # Allocate queries evenly
    total_budget = 50
    per_seed = total_budget // seeds_count
    remainder = total_budget - per_seed * seeds_count
    allocations = [per_seed] * seeds_count
    for i in range(remainder):
        allocations[i] += 1

    print(f"\nQuery allocation: {allocations} (total {sum(allocations)})")

    # Phase 1: Observe all seeds with focused viewports
    all_counts = []
    all_obs = []
    for seed_idx in range(seeds_count):
        print(f"\n--- Observing seed {seed_idx} ({allocations[seed_idx]} queries) ---")

        used, mx = check_budget(session)
        remaining = mx - used
        budget = min(allocations[seed_idx], remaining)

        # If no API budget left, still try to load from cache
        if budget <= 0:
            cached = load_observations(round_id, seed_idx, map_h, map_w)
            if cached is not None:
                counts, obs_count, n = cached
                print(f"  No API budget ({used}/{mx}), but loaded {n} cached queries")
                all_counts.append(counts)
                all_obs.append(obs_count)
            else:
                print(f"  No budget remaining ({used}/{mx}), using prior only")
                all_counts.append(np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32))
                all_obs.append(np.zeros((map_h, map_w), dtype=np.int32))
            continue

        if budget < allocations[seed_idx]:
            print(f"  Budget adjusted: {budget} queries remaining (was {allocations[seed_idx]})")

        counts, obs_count = observe_seed(
            session, round_id, seed_idx, initial_states[seed_idx],
            map_w, map_h, budget
        )
        all_counts.append(counts)
        all_obs.append(obs_count)

    # Phase 2: Learn cross-seed transitions (within this round)
    transitions = learn_transitions(all_counts, all_obs, initial_states, map_h, map_w)
    print(f"\nLearned in-round transitions for {len(transitions)} terrain types")

    # Phase 3: Build and submit predictions
    for seed_idx in range(seeds_count):
        print(f"\n--- Submitting seed {seed_idx} ---")
        prediction = build_prediction(
            all_counts[seed_idx], all_obs[seed_idx],
            initial_states[seed_idx], map_h, map_w,
            transitions, learned_model,
        )

        # Final safety: ensure valid probabilities
        prediction = np.maximum(prediction, PROB_FLOOR)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)

        min_val = prediction.min()
        sums = prediction.sum(axis=-1)
        print(f"  Min prob: {min_val:.4f}, sum range: {sums.min():.4f}-{sums.max():.4f}")

        result = submit_prediction(session, round_id, seed_idx, prediction)
        print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")

    print("\nAll predictions submitted!")
    budget = session.get(f"{BASE}/astar-island/budget").json()
    print(f"Final budget: {budget.get('queries_used', '?')}/{budget.get('queries_max', '?')}")


if __name__ == "__main__":
    main()
