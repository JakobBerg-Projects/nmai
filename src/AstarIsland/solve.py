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
PROB_FLOOR = 0.02  # Higher floor — protects against KL blowups on rare classes
PRIOR_STRENGTH = 50.0  # Very high — trust aggregate transition stats over noisy single observations
MAX_CONFIDENCE = 0.88  # Cap max probability for any dynamic cell

NUM_DIST_BUCKETS = 6
NUM_SETTLE_DENSITY_BUCKETS = 3  # 0=none, 1=few, 2=many nearby settlements
LEARNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "learned_priors_v5.npz")


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


def save_settlement_fates(round_id, seed_index, settlement_fates):
    """Save settlement fate data as JSON. Keys are (x,y) tuples serialized as 'x,y'."""
    d = _round_cache_dir(round_id)
    path = os.path.join(d, f"fates_seed{seed_index}.json")
    serializable = {}
    for (x, y), fate in settlement_fates.items():
        serializable[f"{x},{y}"] = fate
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_settlement_fates(round_id, seed_index):
    """Load settlement fate data from cache. Returns dict with (x,y) tuple keys."""
    d = os.path.join(CACHE_DIR, round_id)
    path = os.path.join(d, f"fates_seed{seed_index}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        fates = {}
        for key, fate in data.items():
            x, y = key.split(",")
            fates[(int(x), int(y))] = fate
        return fates
    return {}


def distance_bucket(d):
    """Bucket distance: 0=on-site, 1=adj, 2=near, 3=medium(3-4), 4=far(5-7), 5=remote(8+)."""
    if d <= 0: return 0
    if d <= 1: return 1
    if d <= 2: return 2
    if d <= 4: return 3
    if d <= 7: return 4
    return 5


def settlement_density_bucket(settlements, y, x, radius=3):
    """Count settlements within Manhattan distance `radius`, bucket into 3 levels."""
    count = 0
    for s in settlements:
        if abs(s["x"] - x) + abs(s["y"] - y) <= radius:
            count += 1
    if count == 0:
        return 0
    if count <= 2:
        return 1
    return 2


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
# Viewport selection strategies
# ---------------------------------------------------------------------------

def generate_tiling_viewports(map_w, map_h, max_vp_size=15):
    """Generate non-overlapping viewports that tile the entire map."""
    viewports = []
    y = 0
    while y < map_h:
        vh = min(max_vp_size, map_h - y)
        x = 0
        while x < map_w:
            vw = min(max_vp_size, map_w - x)
            viewports.append((x, y, vw, vh))
            x += max_vp_size
        y += max_vp_size
    return viewports


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
# Smart viewport placement — entropy-aware, avoids static cells
# ---------------------------------------------------------------------------

def compute_cell_value_map(initial_state, map_h, map_w, learned_model=None):
    """Compute per-cell expected entropy (importance) for viewport targeting.

    Static cells (ocean, mountain) get value 0. Dynamic cells get value based on
    the learned prior's entropy — higher entropy = more important to observe.
    """
    grid = initial_state["grid"]
    settlements = initial_state["settlements"]
    dist = compute_settlement_distance(settlements, map_h, map_w)
    value_map = np.zeros((map_h, map_w), dtype=np.float32)

    learned, learned_counts = None, None
    if learned_model is not None:
        learned, learned_counts = learned_model

    for y in range(map_h):
        for x in range(map_w):
            cell = grid[y][x]
            if cell == 10 or cell == 5:  # Ocean or Mountain — static, zero value
                continue

            if learned is not None:
                icls = TERRAIN_TO_CLASS.get(cell, 0)
                db = distance_bucket(dist[y, x])
                coastal = int(check_adjacent_to_ocean(grid, y, x, map_h, map_w))
                forest = int(check_adjacent_to_forest(grid, y, x, map_h, map_w))
                sd = settlement_density_bucket(settlements, y, x)

                if learned_counts[icls, db, coastal, forest, sd] > 5:
                    probs = learned[icls, db, coastal, forest, sd]
                elif learned_counts[icls, db, coastal, forest, :].sum() > 5:
                    w = learned_counts[icls, db, coastal, forest, :]
                    probs = np.zeros(NUM_CLASSES, dtype=np.float64)
                    for sdi in range(NUM_SETTLE_DENSITY_BUCKETS):
                        if w[sdi] > 0:
                            probs += learned[icls, db, coastal, forest, sdi] * w[sdi]
                    probs /= w.sum()
                else:
                    probs = None

                if probs is not None:
                    p = np.clip(probs, 1e-10, None)
                    entropy = -np.sum(p * np.log(p))
                    value_map[y, x] = entropy
                    continue

            # Fallback: distance-based heuristic
            d = dist[y, x]
            value_map[y, x] = max(0.0, 1.3 - 0.15 * d)

    return value_map


def select_smart_viewports(cell_value_map, map_w, map_h, num_viewports, vp_size=15):
    """Greedy viewport selection maximizing total uncovered cell value.

    For each slot, scans all candidate positions and picks the one with the
    highest sum of uncovered cell values.
    """
    covered = np.zeros((map_h, map_w), dtype=bool)
    viewports = []
    vp_values = []

    for _ in range(num_viewports):
        best_vp, best_score = None, -1.0

        for vy in range(max(1, map_h - vp_size + 1)):
            for vx in range(max(1, map_w - vp_size + 1)):
                vh = min(vp_size, map_h - vy)
                vw = min(vp_size, map_w - vx)
                patch = cell_value_map[vy:vy + vh, vx:vx + vw]
                mask = ~covered[vy:vy + vh, vx:vx + vw]
                score = patch[mask].sum() if mask.any() else 0.0
                if score > best_score:
                    best_score = score
                    best_vp = (vx, vy, vw, vh)

        if best_vp is None or best_score <= 0:
            break

        viewports.append(best_vp)
        vp_values.append(best_score)
        vx, vy, vw, vh = best_vp
        covered[vy:vy + vh, vx:vx + vw] = True

    return viewports, vp_values


# ---------------------------------------------------------------------------
# Observation — smart viewports with repeated sampling
# ---------------------------------------------------------------------------

def observe_seed(session, round_id, seed_index, initial_state, map_w, map_h,
                 query_budget, cell_value_map=None):
    """Smart viewport selection with repeat sampling for empirical distributions.
    Tracks settlement metadata (alive/dead/port/population) across queries.
    Saves every query result and accumulated observations to disk."""
    settlement_fates = load_settlement_fates(round_id, seed_index)

    # Try to resume from cache
    cached = load_observations(round_id, seed_index, map_h, map_w)
    if cached is not None:
        counts, obs_count, n_cached = cached
        print(f"  Seed {seed_index}: loaded {n_cached} cached queries from disk")
        if n_cached >= query_budget:
            print(f"  Seed {seed_index}: already have {n_cached} queries cached, skipping")
            return counts, obs_count, settlement_fates
        queries_done = n_cached
    else:
        counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
        obs_count = np.zeros((map_h, map_w), dtype=np.int32)
        queries_done = 0

    remaining = query_budget - queries_done
    if remaining <= 0:
        return counts, obs_count, settlement_fates

    # COVERAGE is king: each query should see NEW cells, not repeat old ones.
    # With stochastic simulation, a single observation is very noisy — but seeing
    # a cell once is far better than not seeing it at all. Maximize unique coverage.
    if cell_value_map is not None:
        # Use ALL queries for distinct viewports — no repeats
        num_distinct = remaining
        smart_vps, vp_vals = select_smart_viewports(
            cell_value_map, map_w, map_h, num_distinct)
    else:
        # Fallback to tiling viewports for maximum coverage
        smart_vps = generate_tiling_viewports(map_w, map_h)
        vp_vals = [1.0] * len(smart_vps)

    if not smart_vps:
        # Last resort: center viewport
        cx, cy = map_w // 2, map_h // 2
        smart_vps = [(max(0, cx - 7), max(0, cy - 7), min(15, map_w), min(15, map_h))]
        vp_vals = [1.0]

    # Each viewport gets exactly one query — maximize unique cell coverage
    query_plan = list(smart_vps[:remaining])

    # Log plan
    vp_counts = {}
    for vp in query_plan:
        vp_counts[vp] = vp_counts.get(vp, 0) + 1
    for vp, cnt in vp_counts.items():
        vx, vy, vw, vh = vp
        print(f"  Seed {seed_index}: viewport ({vx},{vy}) {vw}x{vh} × {cnt} queries")

    q = 0
    result = None
    for vx, vy, vw, vh in query_plan:
        q += 1
        print(f"    [{queries_done + q}/{query_budget}] ({vx},{vy}) {vw}x{vh}")

        result = simulate_query(session, round_id, seed_index, vx, vy, vw, vh)
        if result is None:
            print("  Budget exhausted, stopping observations")
            save_observations(round_id, seed_index, counts, obs_count)
            save_settlement_fates(round_id, seed_index, settlement_fates)
            return counts, obs_count, settlement_fates

        save_query_result(round_id, seed_index, queries_done + q, result)

        # Accumulate grid observations
        vp = result["viewport"]
        for ri, row in enumerate(result["grid"]):
            for ci, val in enumerate(row):
                gy, gx = vp["y"] + ri, vp["x"] + ci
                if 0 <= gy < map_h and 0 <= gx < map_w:
                    counts[gy, gx, TERRAIN_TO_CLASS.get(val, 0)] += 1
                    obs_count[gy, gx] += 1

        # Accumulate settlement metadata
        for s in result.get("settlements", []):
            sx, sy = s["x"], s["y"]
            key = (sx, sy)
            if key not in settlement_fates:
                settlement_fates[key] = {
                    "alive": 0, "dead": 0, "port": 0, "total": 0,
                    "pops": [], "foods": [], "defenses": [],
                }
            fate = settlement_fates[key]
            fate["total"] += 1
            if s.get("alive", True):
                fate["alive"] += 1
                if s.get("has_port", False):
                    fate["port"] += 1
            else:
                fate["dead"] += 1
            fate["pops"].append(s.get("population", 0))
            fate["foods"].append(s.get("food", 0))
            fate["defenses"].append(s.get("defense", 0))

        save_observations(round_id, seed_index, counts, obs_count)
        time.sleep(0.25)

    save_settlement_fates(round_id, seed_index, settlement_fates)

    if result is not None:
        used = result.get("queries_used", "?")
        mx = result.get("queries_max", "?")
        print(f"  Budget: {used}/{mx}")

    n_fates = len(settlement_fates)
    print(f"  Settlement fates tracked: {n_fates} positions")
    return counts, obs_count, settlement_fates


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

def _feature_shape():
    """Return the shape of the feature accumulator (excl. final class dim)."""
    return (NUM_CLASSES, NUM_DIST_BUCKETS, 2, 2, NUM_SETTLE_DENSITY_BUCKETS)


def compute_initial_features(initial_states, map_h, map_w):
    """Compute features from initial states for regime matching (no simulation needed).

    Returns a feature vector: [n_settlements, n_ports, frac_forest, frac_ocean,
    frac_mountain, avg_settle_density, frac_coastal_settlements, avg_min_settle_dist]
    """
    features = []
    for state in initial_states:
        grid = state["grid"]
        settlements = state["settlements"]
        total = map_h * map_w

        n_settle = len(settlements)
        n_ports = sum(1 for s in settlements if s.get("has_port", False))
        n_forest = sum(1 for row in grid for c in row if c == 4)
        n_ocean = sum(1 for row in grid for c in row if c == 10)
        n_mountain = sum(1 for row in grid for c in row if c == 5)

        # Coastal settlements (adjacent to ocean)
        n_coastal = 0
        for s in settlements:
            if check_adjacent_to_ocean(grid, s["y"], s["x"], map_h, map_w):
                n_coastal += 1

        # Average nearest-settlement distance for settlements
        if len(settlements) > 1:
            dists = []
            for i, s1 in enumerate(settlements):
                min_d = min(abs(s1["x"] - s2["x"]) + abs(s1["y"] - s2["y"])
                            for j, s2 in enumerate(settlements) if j != i)
                dists.append(min_d)
            avg_settle_dist = np.mean(dists)
        else:
            avg_settle_dist = 20.0  # default if 0-1 settlements

        features.append([
            n_settle / total, n_ports / max(n_settle, 1),
            n_forest / total, n_ocean / total, n_mountain / total,
            n_coastal / max(n_settle, 1), avg_settle_dist / map_w,
        ])

    return np.mean(features, axis=0)  # Average across seeds


def learn_from_history(session):
    """Fetch all completed rounds and learn conditional transition priors.

    Learns P(final_class | initial_terrain, distance_bucket, coastal, forest_adj, settle_density)
    both globally (averaged) and per-round (for regime detection).
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

    feat_shape = _feature_shape()
    # Global accumulator: feat_shape + (NUM_CLASSES,) for class probs
    accum = np.zeros(feat_shape + (NUM_CLASSES,), dtype=np.float64)
    counts = np.zeros(feat_shape, dtype=np.float64)

    # Per-round storage
    per_round_learned = []
    per_round_counts = []
    per_round_class_freq = []
    per_round_init_features = []
    round_numbers = []

    for rnd in completed:
        for _retry in range(5):
            resp = session.get(f"{BASE}/astar-island/rounds/{rnd['id']}")
            if resp.status_code == 429:
                print(f"    Rate limited fetching round detail, waiting {2*(_retry+1)}s...")
                time.sleep(2 * (_retry + 1))
                continue
            resp.raise_for_status()
            detail = resp.json()
            break
        else:
            print(f"  Skipping round {rnd['round_number']} (rate limited)")
            continue
        map_w = detail["map_width"]
        map_h = detail["map_height"]
        seeds_count = detail["seeds_count"]
        initial_states = detail["initial_states"]

        print(f"  Round {rnd['round_number']}: {map_w}x{map_h}, {seeds_count} seeds")

        rnd_accum = np.zeros(feat_shape + (NUM_CLASSES,), dtype=np.float64)
        rnd_counts = np.zeros(feat_shape, dtype=np.float64)
        rnd_class_freq = np.zeros(NUM_CLASSES, dtype=np.float64)
        rnd_total_cells = 0

        for seed_idx in range(seeds_count):
            resp = None
            for _retry in range(5):
                resp = session.get(f"{BASE}/astar-island/analysis/{rnd['id']}/{seed_idx}")
                if resp.status_code == 429:
                    time.sleep(2 * (_retry + 1))
                    continue
                break
            if resp is None or resp.status_code != 200:
                print(f"    Seed {seed_idx}: no ground truth ({resp.status_code if resp else 'timeout'})")
                continue

            gt = np.array(resp.json()["ground_truth"], dtype=np.float64)
            initial = initial_states[seed_idx]
            grid = initial["grid"]
            settlements = initial["settlements"]
            dist = compute_settlement_distance(settlements, map_h, map_w)

            rnd_class_freq += gt.sum(axis=(0, 1))
            rnd_total_cells += map_h * map_w

            for y in range(map_h):
                for x in range(map_w):
                    icls = TERRAIN_TO_CLASS.get(grid[y][x], 0)
                    db = distance_bucket(dist[y, x])
                    coastal = int(check_adjacent_to_ocean(grid, y, x, map_h, map_w))
                    forest = int(check_adjacent_to_forest(grid, y, x, map_h, map_w))
                    sd = settlement_density_bucket(settlements, y, x)

                    rnd_accum[icls, db, coastal, forest, sd] += gt[y, x]
                    rnd_counts[icls, db, coastal, forest, sd] += 1.0

            time.sleep(0.2)

        # Accumulate into global
        accum += rnd_accum
        counts += rnd_counts

        # Build per-round learned priors
        rnd_learned = np.full(feat_shape + (NUM_CLASSES,), 1.0 / NUM_CLASSES)
        valid_rnd = rnd_counts > 3
        for idx in np.argwhere(valid_rnd):
            i, d, c, f, sd = tuple(idx)
            avg = rnd_accum[i, d, c, f, sd] / rnd_counts[i, d, c, f, sd]
            avg = np.maximum(avg, PROB_FLOOR)
            avg /= avg.sum()
            rnd_learned[i, d, c, f, sd] = avg

        if rnd_total_cells > 0:
            rnd_class_freq /= rnd_total_cells

        per_round_learned.append(rnd_learned)
        per_round_counts.append(rnd_counts)
        per_round_class_freq.append(rnd_class_freq)
        per_round_init_features.append(
            compute_initial_features(initial_states, map_h, map_w))
        round_numbers.append(rnd["round_number"])

        print(f"    P(class): " + ", ".join(
            f"{['E','S','P','R','F','M'][i]}={rnd_class_freq[i]:.3f}"
            for i in range(NUM_CLASSES) if rnd_class_freq[i] > 0.005
        ))

    # Global average and normalize
    learned = np.full(feat_shape + (NUM_CLASSES,), 1.0 / NUM_CLASSES)
    valid = counts > 5
    for idx in np.argwhere(valid):
        i, d, c, f, sd = tuple(idx)
        avg = accum[i, d, c, f, sd] / counts[i, d, c, f, sd]
        avg = np.maximum(avg, PROB_FLOOR)
        avg /= avg.sum()
        learned[i, d, c, f, sd] = avg

    # Save everything: global + per-round data
    np.savez(
        LEARNED_MODEL_PATH,
        learned=learned,
        counts=counts,
        per_round_learned=np.array(per_round_learned),
        per_round_counts=np.array(per_round_counts),
        per_round_class_freq=np.array(per_round_class_freq),
        per_round_init_features=np.array(per_round_init_features),
        round_numbers=np.array(round_numbers),
    )

    total_cells = int(counts.sum())
    valid_bins = int(valid.sum())
    total_bins = counts.size
    print(f"\nLearned model saved to {LEARNED_MODEL_PATH}")
    print(f"  {total_cells} cells across all rounds/seeds")
    print(f"  {valid_bins}/{total_bins} feature bins with sufficient data")
    print(f"  {len(round_numbers)} per-round models saved for regime detection")

    # Print summary of learned transitions
    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    dist_labels = ["on-site(0)", "adj(1)", "near(2)", "medium(3-4)", "far(5-7)", "remote(8+)"]
    print(f"\nLearned transition summary:")
    for icls in range(NUM_CLASSES):
        for db in range(NUM_DIST_BUCKETS):
            c = counts[icls, db].sum()
            if c > 50:
                avg = accum[icls, db].sum(axis=(0, 1, 2)) / max(1, c)
                avg = np.maximum(avg, PROB_FLOOR)
                avg /= avg.sum()
                top3 = np.argsort(avg)[::-1][:3]
                desc = ", ".join(f"{class_names[t]}={avg[t]:.2f}" for t in top3)
                print(f"  {class_names[icls]} dist={dist_labels[db]}: {desc} (n={int(c)})")


def load_learned_priors():
    """Load learned prior model if available.

    Returns dict with keys: 'learned', 'counts', and optionally per-round data.
    Returns (None, None) if no model exists.
    """
    if os.path.exists(LEARNED_MODEL_PATH):
        data = np.load(LEARNED_MODEL_PATH, allow_pickle=True)
        print(f"Loaded learned priors from {LEARNED_MODEL_PATH}")

        model = (data["learned"], data["counts"])

        # Load per-round data if available
        if "per_round_learned" in data:
            per_round = {
                "learned": data["per_round_learned"],
                "counts": data["per_round_counts"],
                "class_freq": data["per_round_class_freq"],
                "round_numbers": data["round_numbers"],
                "init_features": data["per_round_init_features"] if "per_round_init_features" in data else None,
            }
            print(f"  {len(per_round['round_numbers'])} per-round models available")
            return model, per_round

        return model, None
    return (None, None), None


def adapt_prior_to_regime(observed_class_freq, per_round_data, global_learned, global_counts):
    """Build a regime-adapted prior by weighting historical rounds by similarity.

    Args:
        observed_class_freq: array of shape (NUM_CLASSES,) — estimated class frequencies
            from observations of the current round.
        per_round_data: dict with per-round learned priors and class frequencies.
        global_learned: global averaged learned priors (fallback).
        global_counts: global counts array.

    Returns:
        adapted learned prior array of same shape as global_learned.
    """
    obs_freq = np.clip(observed_class_freq, 1e-10, None)
    n_rounds = len(per_round_data["round_numbers"])
    rnd_nums = per_round_data["round_numbers"]
    max_rnd = max(rnd_nums) if len(rnd_nums) > 0 else 1

    # Compute similarity weight for each historical round using symmetric KL + recency
    weights = np.zeros(n_rounds)
    for i in range(n_rounds):
        hist_freq = np.clip(per_round_data["class_freq"][i], 1e-10, None)
        sym_kl = (np.sum(obs_freq * np.log(obs_freq / hist_freq))
                  + np.sum(hist_freq * np.log(hist_freq / obs_freq)))
        age = max_rnd - rnd_nums[i]
        recency = np.exp(-0.05 * age)
        weights[i] = recency / (sym_kl + 0.01)

    weights /= weights.sum()

    # Weighted average of per-round priors
    adapted = np.zeros_like(global_learned)
    for i in range(n_rounds):
        adapted += per_round_data["learned"][i] * weights[i]

    # Normalize bins that have enough global data
    valid = global_counts > 5
    for idx in np.argwhere(valid):
        idx_tuple = tuple(idx)
        adapted[idx_tuple] = np.maximum(adapted[idx_tuple], PROB_FLOOR)
        adapted[idx_tuple] /= adapted[idx_tuple].sum()

    # Report which rounds matched best
    top_idx = np.argsort(weights)[::-1][:3]
    rnd_nums = per_round_data["round_numbers"]
    desc = ", ".join(f"R{rnd_nums[i]}={weights[i]:.2f}" for i in top_idx)
    print(f"  Regime detection: weights [{desc}]")

    return adapted


def adapt_prior_from_initial_state(init_features, per_round_data, global_learned, global_counts):
    """Regime detection using initial state features (no observations needed).

    This avoids the bias from viewport-targeted observations.
    Falls back to uniform weighting if init_features are not available.
    """
    if per_round_data.get("init_features") is None:
        print("  No initial features in model, using uniform weights")
        n_rounds = len(per_round_data["round_numbers"])
        weights = np.ones(n_rounds) / n_rounds
    else:
        n_rounds = len(per_round_data["round_numbers"])
        rnd_nums = per_round_data["round_numbers"]
        max_rnd = max(rnd_nums) if len(rnd_nums) > 0 else 1
        weights = np.zeros(n_rounds)
        for i in range(n_rounds):
            hist_feat = per_round_data["init_features"][i]
            # Euclidean distance in feature space
            dist = np.sqrt(np.sum((init_features - hist_feat) ** 2))
            # Recency boost: recent rounds get higher weight
            age = max_rnd - rnd_nums[i]
            recency = np.exp(-0.05 * age)  # Gentle decay for older rounds
            weights[i] = recency / (dist + 0.001)
        weights /= weights.sum()

    # Weighted average of per-round priors
    adapted = np.zeros_like(global_learned)
    for i in range(n_rounds):
        adapted += per_round_data["learned"][i] * weights[i]

    # Normalize bins that have enough global data
    valid = global_counts > 5
    for idx in np.argwhere(valid):
        idx_tuple = tuple(idx)
        adapted[idx_tuple] = np.maximum(adapted[idx_tuple], PROB_FLOOR)
        adapted[idx_tuple] /= adapted[idx_tuple].sum()

    top_idx = np.argsort(weights)[::-1][:3]
    rnd_nums = per_round_data["round_numbers"]
    desc = ", ".join(f"R{rnd_nums[i]}={weights[i]:.2f}" for i in top_idx)
    print(f"  Regime detection (init features): weights [{desc}]")

    return adapted


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
                prior[y, x, 0] = 0.96
                is_static[y, x] = True

            elif cell == 5:  # Mountain — never changes
                prior[y, x, 5] = 0.96
                is_static[y, x] = True

            elif learned is not None:
                sd = settlement_density_bucket(settlements, y, x)
                # Try full feature lookup first, then fall back to aggregated
                if learned_counts[icls, db, coastal, forest, sd] > 5:
                    prior[y, x] = learned[icls, db, coastal, forest, sd]
                elif learned_counts[icls, db, coastal, forest, :].sum() > 5:
                    # Aggregate over settlement density
                    w = learned_counts[icls, db, coastal, forest, :]
                    total_w = w.sum()
                    agg = np.zeros(NUM_CLASSES, dtype=np.float64)
                    for sdi in range(NUM_SETTLE_DENSITY_BUCKETS):
                        if w[sdi] > 0:
                            agg += learned[icls, db, coastal, forest, sdi] * w[sdi]
                    agg /= total_w
                    agg = np.maximum(agg, PROB_FLOOR)
                    agg /= agg.sum()
                    prior[y, x] = agg

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
    """Learn feature-conditioned transition probabilities across all seeds.

    Returns a dict keyed by (terrain, dist_bucket, coastal) with transition probs,
    plus a coarse dict keyed by terrain only as fallback.
    """
    # Fine-grained: (terrain, dist_bucket, coastal) → class probs
    feat_shape = (NUM_CLASSES, NUM_DIST_BUCKETS, 2)
    trans_counts = np.zeros(feat_shape + (NUM_CLASSES,), dtype=np.float64)
    trans_total = np.zeros(feat_shape, dtype=np.float64)

    # Coarse: terrain → class probs
    coarse_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    coarse_total = np.zeros(NUM_CLASSES, dtype=np.float64)

    for si in range(len(initial_states)):
        grid = initial_states[si]["grid"]
        settlements = initial_states[si]["settlements"]
        dist = compute_settlement_distance(settlements, map_h, map_w)
        for y in range(map_h):
            for x in range(map_w):
                if all_obs[si][y, x] > 0:
                    src = TERRAIN_TO_CLASS.get(grid[y][x], 0)
                    db = distance_bucket(dist[y, x])
                    coastal = int(check_adjacent_to_ocean(grid, y, x, map_h, map_w))
                    trans_counts[src, db, coastal] += all_counts[si][y, x]
                    trans_total[src, db, coastal] += all_obs[si][y, x]
                    coarse_counts[src] += all_counts[si][y, x]
                    coarse_total[src] += all_obs[si][y, x]

    transitions_fine = {}
    for idx in np.argwhere(trans_total > 5):
        src, db, coastal = tuple(idx)
        t = trans_counts[src, db, coastal] / trans_total[src, db, coastal]
        t = np.maximum(t, PROB_FLOOR)
        t /= t.sum()
        transitions_fine[(src, db, coastal)] = t

    transitions_coarse = {}
    for cls in range(NUM_CLASSES):
        if coarse_total[cls] > 10:
            t = coarse_counts[cls] / coarse_total[cls]
            t = np.maximum(t, PROB_FLOOR)
            t /= t.sum()
            transitions_coarse[cls] = t

    return transitions_fine, transitions_coarse


# ---------------------------------------------------------------------------
# Settlement fate adjustments
# ---------------------------------------------------------------------------

def build_settlement_adjustments(settlement_fates, map_h, map_w):
    """Build per-cell probability adjustments from settlement metadata.

    Returns (adjustments, confidence) arrays. adjustments has shape (map_h, map_w, NUM_CLASSES),
    confidence has shape (map_h, map_w) with number of settlement observations per cell.
    """
    adjustments = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    confidence = np.zeros((map_h, map_w), dtype=np.float64)

    for (sx, sy), fate in settlement_fates.items():
        if sy >= map_h or sx >= map_w or fate["total"] == 0:
            continue
        n = fate["total"]
        confidence[sy, sx] = n

        # Direct empirical settlement fate
        p_alive = fate["alive"] / n
        p_port = fate["port"] / n
        p_dead = fate["dead"] / n

        # Adjust based on population/defense stability
        if fate["pops"]:
            avg_pop = np.mean(fate["pops"])
            avg_def = np.mean(fate["defenses"])
            avg_food = np.mean(fate["foods"])
            # Strong settlements: high confidence in survival
            if avg_pop > 2.0 and avg_def > 0.8:
                p_alive = min(1.0, p_alive * 1.15)
            # Weak settlements: higher ruin risk
            elif avg_pop < 0.5 or avg_food < 0.1:
                ruin_boost = min(0.15, (1.0 - p_dead) * 0.2)
                p_dead += ruin_boost
                p_alive = max(0.0, p_alive - ruin_boost)

        # Build distribution
        p_sett = max(0, p_alive - p_port)  # Settlement (alive, no port)
        adjustments[sy, sx, 1] = p_sett
        adjustments[sy, sx, 2] = p_port
        adjustments[sy, sx, 3] = p_dead
        # Remaining probability for empty/forest
        p_remaining = max(0, 1.0 - p_sett - p_port - p_dead)
        adjustments[sy, sx, 0] = p_remaining * 0.6  # Most remaining goes to empty
        adjustments[sy, sx, 4] = p_remaining * 0.4  # Some to forest

        # Floor and normalize
        adjustments[sy, sx] = np.maximum(adjustments[sy, sx], PROB_FLOOR)
        adjustments[sy, sx] /= adjustments[sy, sx].sum()

    return adjustments, confidence


# ---------------------------------------------------------------------------
# Prediction: Bayesian posterior combining prior + observations + settlements
# ---------------------------------------------------------------------------

def estimate_expansion_from_observations(all_counts, all_obs, initial_states, map_h, map_w):
    """Estimate expansion rate from observations across all seeds.

    Returns a dict with:
      - expansion_rate: fraction of non-static observed cells that became settlement/port/ruin
      - settle_by_dist: array of shape (NUM_DIST_BUCKETS,) with P(settlement|distance)
      - port_rate: fraction of observed coastal cells that became ports
      - ruin_rate: fraction of observed cells that became ruins
      - total_obs: total observations used
    """
    settle_counts_by_dist = np.zeros(NUM_DIST_BUCKETS, dtype=np.float64)
    total_by_dist = np.zeros(NUM_DIST_BUCKETS, dtype=np.float64)
    port_count = 0.0
    coastal_count = 0.0
    ruin_count = 0.0
    total_dynamic = 0.0
    settle_total = 0.0

    for si in range(len(initial_states)):
        grid = initial_states[si]["grid"]
        settlements = initial_states[si]["settlements"]
        dist = compute_settlement_distance(settlements, map_h, map_w)

        for y in range(map_h):
            for x in range(map_w):
                if all_obs[si][y, x] <= 0:
                    continue
                cell = grid[y][x]
                if cell == 10 or cell == 5:  # Static
                    continue

                n = all_obs[si][y, x]
                total_dynamic += n
                db = distance_bucket(dist[y, x])

                # Count settlements (class 1) and ports (class 2)
                s_count = all_counts[si][y, x, 1]
                p_count = all_counts[si][y, x, 2]
                r_count = all_counts[si][y, x, 3]

                settle_total += s_count + p_count
                settle_counts_by_dist[db] += s_count + p_count
                total_by_dist[db] += n
                ruin_count += r_count

                if check_adjacent_to_ocean(grid, y, x, map_h, map_w):
                    port_count += p_count
                    coastal_count += n

    expansion_rate = settle_total / max(total_dynamic, 1.0)
    settle_by_dist = np.zeros(NUM_DIST_BUCKETS, dtype=np.float64)
    for db in range(NUM_DIST_BUCKETS):
        if total_by_dist[db] > 0:
            settle_by_dist[db] = settle_counts_by_dist[db] / total_by_dist[db]
    port_rate = port_count / max(coastal_count, 1.0)
    ruin_rate = ruin_count / max(total_dynamic, 1.0)

    print(f"  Expansion estimate: settle_rate={expansion_rate:.3f}, port_rate={port_rate:.3f}, ruin_rate={ruin_rate:.3f}")
    print(f"    By distance: {', '.join(f'd{i}={settle_by_dist[i]:.3f}' for i in range(NUM_DIST_BUCKETS))}")

    return {
        "expansion_rate": expansion_rate,
        "settle_by_dist": settle_by_dist,
        "port_rate": port_rate,
        "ruin_rate": ruin_rate,
        "total_obs": total_dynamic,
    }


def build_prediction(counts, obs_count, initial_state, map_h, map_w,
                     transitions=None, learned_model=None, settlement_fates=None,
                     expansion_info=None):
    """Build prediction using in-round transitions as PRIMARY prior.

    Key insight: aggregate transition statistics from all observations across all
    seeds accurately capture the current round's hidden parameters — even from
    noisy single-sample observations. These are far more reliable than per-cell
    observations or historical learned priors.

    Strategy:
    - In-round transitions (from cross-seed learning) are the best prior
    - Global learned prior is only a fallback for unseen feature combinations
    - Per-cell observations barely influence the prediction (PS=50)
    - Confidence is capped to prevent catastrophic KL from over-confidence
    """
    fallback_prior, is_static = build_initial_prior(initial_state, map_h, map_w, learned_model)
    grid = initial_state["grid"]
    settlements = initial_state["settlements"]
    dist = compute_settlement_distance(settlements, map_h, map_w)

    prediction = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)

    transitions_fine, transitions_coarse = (transitions if transitions is not None
                                            else ({}, {}))

    for y in range(map_h):
        for x in range(map_w):
            if is_static[y, x]:
                prediction[y, x] = fallback_prior[y, x]
                continue

            # Build the best available prior for this cell:
            # 1. In-round transitions (this round's actual dynamics) — best
            # 2. Coarse in-round transitions — good
            # 3. Global learned prior — fallback
            src = TERRAIN_TO_CLASS.get(grid[y][x], 0)
            db = distance_bucket(dist[y, x])
            coastal = int(check_adjacent_to_ocean(grid, y, x, map_h, map_w))
            key = (src, db, coastal)

            if key in transitions_fine:
                cell_prior = transitions_fine[key]
            elif src in transitions_coarse:
                cell_prior = transitions_coarse[src]
            else:
                cell_prior = fallback_prior[y, x]

            if obs_count[y, x] > 0:
                # Bayesian update: strong prior from transitions, barely moved by
                # noisy single-sample observations
                n = obs_count[y, x]
                alpha = cell_prior * PRIOR_STRENGTH
                prediction[y, x] = (counts[y, x] + alpha) / (n + PRIOR_STRENGTH)
            else:
                prediction[y, x] = cell_prior

    # Floor, cap confidence, and renormalize
    prediction = np.maximum(prediction, PROB_FLOOR)
    for y in range(map_h):
        for x in range(map_w):
            if not is_static[y, x]:
                prediction[y, x] = np.minimum(prediction[y, x], MAX_CONFIDENCE)
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
    """Test prediction quality against a completed round using ground truth.

    Simulates the full pipeline: global prior (fallback), in-round transitions
    (primary), and prediction. No regime detection — matches live mode.
    """
    round_id = detail["id"]
    map_w = detail["map_width"]
    map_h = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    global_model, _per_round = load_learned_priors()

    # Fetch all ground truths
    all_gt = []
    for seed_idx in range(seeds_count):
        for _retry in range(3):
            resp = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}")
            if resp.status_code == 429:
                time.sleep(2)
                continue
            break
        if resp.status_code != 200:
            print(f"  Seed {seed_idx}: no ground truth ({resp.status_code})")
            all_gt.append(None)
            continue
        all_gt.append(resp.json())
        time.sleep(0.2)

    # Build observations: oracle (full GT) and realistic (smart viewports)
    gt_counts = []
    gt_obs = []
    vp_counts = []
    vp_obs = []
    for seed_idx in range(seeds_count):
        if all_gt[seed_idx] is None:
            gt_counts.append(np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32))
            gt_obs.append(np.zeros((map_h, map_w), dtype=np.int32))
            vp_counts.append(np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32))
            vp_obs.append(np.zeros((map_h, map_w), dtype=np.int32))
            continue
        gt = np.array(all_gt[seed_idx]["ground_truth"], dtype=np.float64)
        # Oracle: full map, soft observations (gt probs as counts)
        gt_counts.append((gt * 10).astype(np.float32))
        gt_obs.append(np.full((map_h, map_w), 10, dtype=np.int32))

        # Realistic: 10 smart viewports, 1 soft observation per cell
        cell_value_map = compute_cell_value_map(
            initial_states[seed_idx], map_h, map_w, global_model)
        vps_wide, _ = select_smart_viewports(cell_value_map, map_w, map_h, num_viewports=10)
        vp_mask_wide = np.zeros((map_h, map_w), dtype=bool)
        for vx, vy, vw, vh in vps_wide:
            vp_mask_wide[vy:vy+vh, vx:vx+vw] = True
        vc = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
        vo = np.zeros((map_h, map_w), dtype=np.int32)
        vc[vp_mask_wide] = gt[vp_mask_wide].astype(np.float32)
        vo[vp_mask_wide] = 1
        vp_counts.append(vc)
        vp_obs.append(vo)

    # Build in-round transitions (PRIMARY prior) from both observation sets
    transitions_gt = learn_transitions(gt_counts, gt_obs, initial_states, map_h, map_w)
    transitions_vp = learn_transitions(vp_counts, vp_obs, initial_states, map_h, map_w)

    print(f"\nIn-round transitions: oracle={len(transitions_gt[0])} fine, realistic={len(transitions_vp[0])} fine")

    scores_prior_only = []
    scores_oracle = []
    scores_realistic = []

    for seed_idx in range(seeds_count):
        if all_gt[seed_idx] is None:
            continue

        print(f"\n--- Backtesting seed {seed_idx} ---")
        ground_truth = all_gt[seed_idx]["ground_truth"]
        server_score = all_gt[seed_idx].get("score")

        counts_zero = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
        obs_zero = np.zeros((map_h, map_w), dtype=np.int32)

        # 1. Prior-only: global model, no transitions, no observations
        pred_prior = build_prediction(
            counts_zero, obs_zero, initial_states[seed_idx], map_h, map_w,
            learned_model=global_model,
        )

        # 2. Oracle: global fallback + oracle transitions
        pred_oracle = build_prediction(
            counts_zero, obs_zero, initial_states[seed_idx], map_h, map_w,
            transitions=transitions_gt, learned_model=global_model,
        )

        # 3. Realistic: global fallback + viewport transitions + viewport obs
        pred_real = build_prediction(
            vp_counts[seed_idx], vp_obs[seed_idx], initial_states[seed_idx],
            map_h, map_w,
            transitions=transitions_vp, learned_model=global_model,
        )

        s_prior = score_prediction(pred_prior, ground_truth)
        s_oracle = score_prediction(pred_oracle, ground_truth)
        s_real = score_prediction(pred_real, ground_truth)

        scores_prior_only.append(s_prior)
        scores_oracle.append(s_oracle)
        scores_realistic.append(s_real)

        print(f"  Prior only:      {s_prior:.2f}")
        print(f"  Realistic (VP):  {s_real:.2f}  ({s_real - s_prior:+.2f})")
        print(f"  Oracle (full):   {s_oracle:.2f}  ({s_oracle - s_prior:+.2f})")
        if server_score is not None:
            print(f"  Server (prev):   {server_score:.2f}")

    if scores_prior_only:
        print(f"\n--- Backtest Summary ---")
        print(f"Prior only avg:      {np.mean(scores_prior_only):.2f}  "
              f"{[f'{s:.1f}' for s in scores_prior_only]}")
        print(f"Realistic (VP) avg:  {np.mean(scores_realistic):.2f}  "
              f"{[f'{s:.1f}' for s in scores_realistic]}")
        print(f"Oracle (full) avg:   {np.mean(scores_oracle):.2f}  "
              f"{[f'{s:.1f}' for s in scores_oracle]}")


def backtest_all_rounds(session):
    """Backtest against all completed rounds to compare strategies."""
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    completed = sorted(
        [r for r in rounds if r["status"] == "completed"],
        key=lambda r: r["round_number"],
    )

    for rnd in completed:
        rn = rnd["round_number"]
        print(f"\n{'='*60}")
        print(f"  ROUND {rn}")
        print(f"{'='*60}")

        for _retry in range(3):
            resp = session.get(f"{BASE}/astar-island/rounds/{rnd['id']}")
            if resp.status_code == 429:
                time.sleep(2)
                continue
            break
        if resp.status_code != 200:
            print(f"  Skipping (status {resp.status_code})")
            continue
        detail = resp.json()
        try:
            backtest(session, detail)
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Astar Island solver")
    parser.add_argument("--backtest", nargs="?", const=0, type=int, default=None,
                        help="Test against completed round (default: most recent)")
    parser.add_argument("--backtest-all", action="store_true",
                        help="Backtest against all completed rounds we participated in")
    parser.add_argument("--learn", action="store_true",
                        help="Learn priors from all completed rounds")
    parser.add_argument("--quick", action="store_true",
                        help="Submit prior-only predictions immediately (0 queries), then exit")
    parser.add_argument("--safe", action="store_true",
                        help="Submit prior-only first, then observe and resubmit with better predictions")
    args = parser.parse_args()

    session = get_session()

    if args.learn:
        learn_from_history(session)
        return

    if args.backtest_all:
        backtest_all_rounds(session)
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

    # Load learned priors, auto-learn if new completed rounds exist
    global_model, per_round = load_learned_priors()
    if global_model[0] is not None and per_round is not None:
        # Check if new completed rounds exist since last learn
        rounds = session.get(f"{BASE}/astar-island/rounds").json()
        n_completed = len([r for r in rounds if r["status"] == "completed"])
        n_learned = len(per_round["round_numbers"])
        if n_completed > n_learned:
            print(f"\n{n_completed - n_learned} new completed round(s) found, re-learning priors...")
            learn_from_history(session)
            global_model, per_round = load_learned_priors()
    elif global_model[0] is None:
        # No learned model at all, try to learn
        print("\nNo learned priors found, attempting to learn from history...")
        learn_from_history(session)
        global_model, per_round = load_learned_priors()

    if args.quick or args.safe:
        # Quick mode: submit prior-only predictions immediately
        print("\n=== Quick prior-only submission (0 queries) ===")
        for seed_idx in range(seeds_count):
            counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
            obs_count = np.zeros((map_h, map_w), dtype=np.int32)
            prediction = build_prediction(
                counts, obs_count, initial_states[seed_idx], map_h, map_w,
                learned_model=global_model,
            )
            prediction = np.maximum(prediction, PROB_FLOOR)
            prediction = prediction / prediction.sum(axis=-1, keepdims=True)
            result = submit_prediction(session, round_id, seed_idx, prediction)
            print(f"  Seed {seed_idx}: {result.get('status', 'submitted')} (prior-only)")
        if not args.safe:
            print("\nDone. Run without --quick to use queries for better scores.")
            return
        print("\n=== Prior-only submitted. Now observing for better predictions... ===\n")

    # Check cache status
    cached_total = count_cached_queries(round_id)
    if cached_total > 0:
        print(f"\n[cache] Found {cached_total} cached queries for this round")

    # Allocate queries evenly across seeds — each seed needs coverage equally.
    # Cross-seed learning transfers information, so even coverage is better
    # than concentrating on high-settlement seeds.
    total_budget = 50
    base = total_budget // seeds_count
    allocations = [base] * seeds_count
    leftover = total_budget - sum(allocations)
    # Give extra queries to seeds with more settlements (slight preference)
    settle_counts = [len(initial_states[i]["settlements"]) for i in range(seeds_count)]
    order = sorted(range(seeds_count), key=lambda i: settle_counts[i], reverse=True)
    for i in range(leftover):
        allocations[order[i % seeds_count]] += 1

    print(f"\nQuery allocation: {allocations} (total {sum(allocations)})")

    # Phase 1: Compute cell value maps and observe with smart viewports
    all_counts = []
    all_obs = []
    all_fates = []

    for seed_idx in range(seeds_count):
        print(f"\n--- Observing seed {seed_idx} ({allocations[seed_idx]} queries) ---")

        # Compute cell value map for smart viewport placement
        cell_value_map = compute_cell_value_map(
            initial_states[seed_idx], map_h, map_w, global_model)

        used, mx = check_budget(session)
        remaining = mx - used
        budget = min(allocations[seed_idx], remaining)

        # If no API budget left, still try to load from cache
        if budget <= 0:
            cached = load_observations(round_id, seed_idx, map_h, map_w)
            fates = load_settlement_fates(round_id, seed_idx)
            if cached is not None:
                counts, obs_count, n = cached
                print(f"  No API budget ({used}/{mx}), but loaded {n} cached queries")
                all_counts.append(counts)
                all_obs.append(obs_count)
            else:
                print(f"  No budget remaining ({used}/{mx}), using prior only")
                all_counts.append(np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32))
                all_obs.append(np.zeros((map_h, map_w), dtype=np.int32))
            all_fates.append(fates)
            continue

        if budget < allocations[seed_idx]:
            print(f"  Budget adjusted: {budget} queries remaining (was {allocations[seed_idx]})")

        counts, obs_count, fates = observe_seed(
            session, round_id, seed_idx, initial_states[seed_idx],
            map_w, map_h, budget, cell_value_map
        )
        all_counts.append(counts)
        all_obs.append(obs_count)
        all_fates.append(fates)

    # Phase 2: Skip regime detection — it's unreliable across different hidden params.
    # Instead, use global learned model only as FALLBACK for in-round transitions.
    learned_model = global_model
    print(f"\nUsing global learned model as fallback (no regime detection)")

    # Phase 3: Learn cross-seed transitions (within this round) — PRIMARY prior.
    # Aggregate transition statistics from all observations across all seeds capture
    # the current round's hidden parameters far better than any historical prior.
    transitions = learn_transitions(all_counts, all_obs, initial_states, map_h, map_w)
    transitions_fine, transitions_coarse = transitions
    print(f"\nLearned in-round transitions: {len(transitions_fine)} fine, {len(transitions_coarse)} coarse")

    # Phase 4: Build and submit predictions — transitions as primary, global as fallback
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
