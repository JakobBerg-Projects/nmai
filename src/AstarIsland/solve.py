#!/usr/bin/env python3
"""
Astar Island — Viking Civilisation Prediction Solver

Observes a stochastic Norse civilization simulation through limited viewport
queries and predicts probability distributions over 6 terrain classes for
every cell on the map.

Usage:
    export ASTAR_TOKEN="your_jwt_token"
    python solve.py              # run against active round
    python solve.py --backtest   # test against most recent completed round
    python solve.py --backtest 1 # test against specific round number
"""

import argparse
import os
import time
import numpy as np
import requests

BASE = "https://api.ainm.no"

# Terrain code → prediction class index mapping
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
DIRICHLET_ALPHA = 0.5


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
    """Fetch round details. If backtest_round is given, pick a completed round."""
    rounds = session.get(f"{BASE}/astar-island/rounds").json()

    if backtest_round is not None:
        # Backtest mode: pick a completed round
        if backtest_round > 0:
            target = next((r for r in rounds if r["round_number"] == backtest_round), None)
        else:
            # Pick the most recent completed round
            completed = [r for r in rounds if r["status"] == "completed"]
            completed.sort(key=lambda r: r["round_number"], reverse=True)
            target = completed[0] if completed else None

        if not target:
            print("Available rounds:")
            for r in rounds:
                print(f"  Round {r['round_number']}: {r['status']}")
            raise RuntimeError(f"No matching completed round found")

        round_id = target["id"]
        detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
        print(f"Backtest round {target['round_number']} ({target['status']}): {detail['map_width']}x{detail['map_height']}, {detail['seeds_count']} seeds")
        return detail, True
    else:
        # Live mode: find active round
        active = next((r for r in rounds if r["status"] == "active"), None)
        if not active:
            print("No active round. Available rounds:")
            for r in rounds:
                print(f"  Round {r['round_number']}: {r['status']}")
            raise RuntimeError("No active round found. Use --backtest to test against a completed round.")

        round_id = active["id"]
        detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
        print(f"Active round {active['round_number']}: {detail['map_width']}x{detail['map_height']}, {detail['seeds_count']} seeds")
        return detail, False


def compute_tile_offsets(map_w, map_h, vp_size=15):
    """Compute viewport offsets to tile the full map with minimal queries."""
    offsets = []
    y = 0
    while y < map_h:
        x = 0
        while x < map_w:
            vw = min(vp_size, map_w - x)
            vh = min(vp_size, map_h - y)
            offsets.append((x, y, vw, vh))
            x += vp_size - 1  # 1-cell overlap
            if x >= map_w:
                break
        y += vp_size - 1
        if y >= map_h:
            break
    return offsets


def find_settlement_dense_region(initial_state, map_w, map_h):
    """Find the 15x15 region with the most initial settlements."""
    settlements = initial_state["settlements"]
    if not settlements:
        return (0, 0, 15, 15)

    best_count = -1
    best_xy = (0, 0)
    for vy in range(0, max(1, map_h - 14)):
        for vx in range(0, max(1, map_w - 14)):
            count = sum(
                1 for s in settlements
                if vx <= s["x"] < vx + 15 and vy <= s["y"] < vy + 15
            )
            if count > best_count:
                best_count = count
                best_xy = (vx, vy)

    return (best_xy[0], best_xy[1], min(15, map_w - best_xy[0]), min(15, map_h - best_xy[1]))


def simulate_query(session, round_id, seed_index, vx, vy, vw, vh):
    """Run one simulation query and return the result."""
    resp = session.post(f"{BASE}/astar-island/simulate", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx,
        "viewport_y": vy,
        "viewport_w": vw,
        "viewport_h": vh,
    })
    if resp.status_code == 429:
        print("  Rate limited, waiting 1s...")
        time.sleep(1)
        return simulate_query(session, round_id, seed_index, vx, vy, vw, vh)
    resp.raise_for_status()
    return resp.json()


def observe_seed(session, round_id, seed_index, initial_state, map_w, map_h, tile_offsets):
    """Run tiled queries for one seed, return observation counts array."""
    counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
    obs_count = np.zeros((map_h, map_w), dtype=np.int32)

    # Tiled coverage queries
    for i, (vx, vy, vw, vh) in enumerate(tile_offsets):
        print(f"  Seed {seed_index} query {i+1}/{len(tile_offsets)+1}: viewport ({vx},{vy}) {vw}x{vh}")
        result = simulate_query(session, round_id, seed_index, vx, vy, vw, vh)
        grid = result["grid"]
        vp = result["viewport"]

        for row_idx, row in enumerate(grid):
            for col_idx, cell_val in enumerate(row):
                gy = vp["y"] + row_idx
                gx = vp["x"] + col_idx
                if 0 <= gy < map_h and 0 <= gx < map_w:
                    cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                    counts[gy, gx, cls] += 1
                    obs_count[gy, gx] += 1

        time.sleep(0.25)  # Stay under rate limit

    # Bonus query on settlement-dense region
    bx, by, bw, bh = find_settlement_dense_region(initial_state, map_w, map_h)
    print(f"  Seed {seed_index} bonus query: viewport ({bx},{by}) {bw}x{bh}")
    result = simulate_query(session, round_id, seed_index, bx, by, bw, bh)
    grid = result["grid"]
    vp = result["viewport"]

    for row_idx, row in enumerate(grid):
        for col_idx, cell_val in enumerate(row):
            gy = vp["y"] + row_idx
            gx = vp["x"] + col_idx
            if 0 <= gy < map_h and 0 <= gx < map_w:
                cls = TERRAIN_TO_CLASS.get(cell_val, 0)
                counts[gy, gx, cls] += 1
                obs_count[gy, gx] += 1

    budget = result.get("queries_used", "?")
    budget_max = result.get("queries_max", "?")
    print(f"  Budget: {budget}/{budget_max}")

    return counts, obs_count


def build_initial_prior(initial_state, map_h, map_w):
    """Build prior probability tensor from the initial grid."""
    grid = initial_state["grid"]
    prior = np.full((map_h, map_w, NUM_CLASSES), PROB_FLOOR, dtype=np.float32)
    is_static = np.zeros((map_h, map_w), dtype=bool)

    for y in range(map_h):
        for x in range(map_w):
            cell = grid[y][x]
            cls = TERRAIN_TO_CLASS.get(cell, 0)

            if cell == 10:  # Ocean — never changes
                prior[y, x] = PROB_FLOOR
                prior[y, x, 0] = 0.98
                is_static[y, x] = True
            elif cell == 5:  # Mountain — never changes
                prior[y, x] = PROB_FLOOR
                prior[y, x, 5] = 0.98
                is_static[y, x] = True
            elif cell == 4:  # Forest — mostly static, can be reclaimed
                prior[y, x] = PROB_FLOOR
                prior[y, x, 4] = 0.85
                prior[y, x, 0] = 0.05
                prior[y, x, 1] = 0.03
                prior[y, x, 3] = 0.03
            elif cell in (0, 11):  # Empty/Plains — could become anything
                prior[y, x, 0] = 0.70
                prior[y, x, 4] = 0.10
                prior[y, x, 1] = 0.05
                prior[y, x, 2] = 0.03
                prior[y, x, 3] = 0.05
            elif cell == 1:  # Settlement — dynamic
                prior[y, x, 1] = 0.40
                prior[y, x, 3] = 0.25
                prior[y, x, 0] = 0.15
                prior[y, x, 2] = 0.10
            elif cell == 2:  # Port — dynamic
                prior[y, x, 2] = 0.40
                prior[y, x, 1] = 0.15
                prior[y, x, 3] = 0.25
                prior[y, x, 0] = 0.10

        # Renormalize prior rows
        for x in range(map_w):
            prior[y, x] = np.maximum(prior[y, x], PROB_FLOOR)
            prior[y, x] /= prior[y, x].sum()

    return prior, is_static


def check_adjacent_to_ocean(grid, y, x, map_h, map_w):
    """Check if a cell is adjacent to ocean."""
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < map_h and 0 <= nx < map_w:
            if grid[ny][nx] == 10:
                return True
    return False


def build_prediction(counts, obs_count, initial_state, map_h, map_w):
    """Combine observations with prior to build final prediction tensor."""
    prior, is_static = build_initial_prior(initial_state, map_h, map_w)
    prediction = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    grid = initial_state["grid"]

    for y in range(map_h):
        for x in range(map_w):
            if is_static[y, x]:
                # Static cells: use prior directly (ocean/mountain)
                prediction[y, x] = prior[y, x]
            elif obs_count[y, x] > 0:
                # Observed cells: Dirichlet-smoothed empirical distribution
                # Blend prior with observations
                n = obs_count[y, x]
                alpha = DIRICHLET_ALPHA
                # Use prior as pseudo-counts scaled by alpha
                prior_counts = prior[y, x] * alpha * NUM_CLASSES
                posterior = (counts[y, x] + prior_counts) / (n + alpha * NUM_CLASSES)
                prediction[y, x] = posterior
            else:
                # Unobserved cells: use prior
                prediction[y, x] = prior[y, x]

    # Apply probability floor and renormalize
    prediction = np.maximum(prediction, PROB_FLOOR)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True)

    return prediction


def submit_prediction(session, round_id, seed_index, prediction):
    """Submit prediction tensor for one seed."""
    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    })
    if resp.status_code == 429:
        print("  Rate limited on submit, waiting 1s...")
        time.sleep(1)
        return submit_prediction(session, round_id, seed_index, prediction)
    resp.raise_for_status()
    return resp.json()


def score_prediction(prediction, ground_truth):
    """Compute entropy-weighted KL divergence score (same as server)."""
    p = np.array(ground_truth, dtype=np.float64)
    q = np.array(prediction, dtype=np.float64)

    # Entropy per cell
    p_safe = np.clip(p, 1e-15, None)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=-1)

    # KL divergence per cell
    q_safe = np.clip(q, 1e-15, None)
    kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)

    # Entropy-weighted average KL
    total_entropy = entropy.sum()
    if total_entropy < 1e-10:
        return 100.0
    weighted_kl = (entropy * kl).sum() / total_entropy

    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
    return score


def backtest(session, detail):
    """Test prediction quality against a completed round using ground truth."""
    round_id = detail["id"]
    map_w = detail["map_width"]
    map_h = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    scores = []
    for seed_idx in range(seeds_count):
        print(f"\n--- Backtesting seed {seed_idx} ---")

        # Build prediction from initial state + prior only (no queries available for completed rounds)
        counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
        obs_count = np.zeros((map_h, map_w), dtype=np.int32)
        prediction = build_prediction(counts, obs_count, initial_states[seed_idx], map_h, map_w)

        # Sanity checks
        assert prediction.shape == (map_h, map_w, NUM_CLASSES)
        prediction = np.maximum(prediction, PROB_FLOOR)
        prediction = prediction / prediction.sum(axis=-1, keepdims=True)

        # Fetch ground truth
        resp = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}")
        if resp.status_code != 200:
            print(f"  Could not fetch ground truth: {resp.status_code} {resp.text[:200]}")
            continue

        analysis = resp.json()
        ground_truth = analysis["ground_truth"]

        # Score locally
        local_score = score_prediction(prediction, ground_truth)
        server_score = analysis.get("score")
        scores.append(local_score)

        print(f"  Prior-only score: {local_score:.2f}")
        if server_score is not None:
            print(f"  Server score (last submission): {server_score:.2f}")

        # Show per-class accuracy breakdown
        gt = np.array(ground_truth)
        pred = np.array(prediction)
        gt_argmax = gt.argmax(axis=-1)
        pred_argmax = pred.argmax(axis=-1)
        class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
        for cls_idx, cls_name in enumerate(class_names):
            mask = gt_argmax == cls_idx
            if mask.sum() > 0:
                correct = (pred_argmax[mask] == cls_idx).sum()
                print(f"    {cls_name}: {correct}/{mask.sum()} cells correct ({100*correct/mask.sum():.0f}%)")

    if scores:
        print(f"\n--- Backtest Summary ---")
        print(f"Average score (prior only): {np.mean(scores):.2f}")
        print(f"Per-seed scores: {[f'{s:.2f}' for s in scores]}")
        print(f"\nThis is the baseline using only initial terrain priors (no queries).")
        print(f"With live queries during an active round, scores should be significantly higher.")


def main():
    parser = argparse.ArgumentParser(description="Astar Island solver")
    parser.add_argument("--backtest", nargs="?", const=0, type=int, default=None,
                        help="Test against a completed round (default: most recent, or specify round number)")
    args = parser.parse_args()

    session = get_session()
    detail, is_backtest = get_round(session, backtest_round=args.backtest)

    round_id = detail["id"]
    map_w = detail["map_width"]
    map_h = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    if is_backtest:
        backtest(session, detail)
        return

    # Live mode: observe and submit
    tile_offsets = compute_tile_offsets(map_w, map_h, vp_size=15)
    print(f"Tiling strategy: {len(tile_offsets)} queries per seed")
    print(f"Total planned queries: {(len(tile_offsets) + 1) * seeds_count}")

    # Observe all seeds
    all_counts = []
    all_obs = []
    for seed_idx in range(seeds_count):
        print(f"\n--- Observing seed {seed_idx} ---")
        counts, obs_count = observe_seed(
            session, round_id, seed_idx, initial_states[seed_idx],
            map_w, map_h, tile_offsets
        )
        all_counts.append(counts)
        all_obs.append(obs_count)

    # Build and submit predictions
    for seed_idx in range(seeds_count):
        print(f"\n--- Building prediction for seed {seed_idx} ---")
        prediction = build_prediction(
            all_counts[seed_idx], all_obs[seed_idx],
            initial_states[seed_idx], map_h, map_w
        )

        # Sanity checks
        assert prediction.shape == (map_h, map_w, NUM_CLASSES)
        assert np.all(prediction >= PROB_FLOOR - 1e-6)
        sums = prediction.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=0.01), f"Sum range: {sums.min():.4f} - {sums.max():.4f}"

        result = submit_prediction(session, round_id, seed_idx, prediction)
        print(f"  Seed {seed_idx} submitted: {result.get('status', 'unknown')}")

    print("\nAll predictions submitted!")

    # Check budget
    budget = session.get(f"{BASE}/astar-island/budget").json()
    print(f"Final budget: {budget.get('queries_used', '?')}/{budget.get('queries_max', '?')}")


if __name__ == "__main__":
    main()
