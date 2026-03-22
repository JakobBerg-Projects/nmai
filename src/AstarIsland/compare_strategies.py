#!/usr/bin/env python3
"""
Compare --quick (prior-only) vs --safe (prior + observations) using cached data.
Tests both strategies against ground truth with the CURRENT code settings.
"""
import numpy as np
import time
from solve import (
    get_session, load_learned_priors, build_prediction, score_prediction,
    compute_initial_features, adapt_prior_from_initial_state, adapt_prior_to_regime,
    learn_transitions, load_observations, load_settlement_fates,
    compute_settlement_distance, compute_cell_value_map,
    TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR, BASE, CACHE_DIR,
)
import os


def test_strategies(session):
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    completed = sorted(
        [r for r in rounds if r["status"] == "completed"],
        key=lambda r: r["round_number"],
    )

    global_model, per_round = load_learned_priors()
    if global_model[0] is None:
        print("No learned priors!")
        return

    results = []

    for rnd in completed:
        rnd_num = rnd["round_number"]
        rnd_id = rnd["id"]

        # Check if we have cached observations for this round
        cache_dir = os.path.join(CACHE_DIR, rnd_id)
        if not os.path.isdir(cache_dir):
            continue

        for _retry in range(5):
            resp = session.get(f"{BASE}/astar-island/rounds/{rnd_id}")
            if resp.status_code == 429:
                time.sleep(2 * (_retry + 1))
                continue
            break
        if resp.status_code != 200:
            continue
        detail = resp.json()

        map_w = detail["map_width"]
        map_h = detail["map_height"]
        seeds_count = detail["seeds_count"]
        initial_states = detail["initial_states"]

        # Check if we have observations for at least one seed
        has_obs = False
        for si in range(seeds_count):
            cached = load_observations(rnd_id, si, map_h, map_w)
            if cached is not None and cached[2] > 0:
                has_obs = True
                break
        if not has_obs:
            continue

        print(f"\n{'='*60}")
        print(f"Round {rnd_num} (id={rnd_id[:8]}...)")

        # Build regime-adapted model
        init_features = compute_initial_features(initial_states, map_h, map_w)
        adapted_init = adapt_prior_from_initial_state(
            init_features, per_round, global_model[0], global_model[1],
        )

        # Load all observations
        all_counts = []
        all_obs = []
        all_fates = []
        total_queries = 0
        for si in range(seeds_count):
            cached = load_observations(rnd_id, si, map_h, map_w)
            fates = load_settlement_fates(rnd_id, si)
            if cached is not None:
                counts, obs_count, n_queries = cached
                total_queries += n_queries
            else:
                counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
                obs_count = np.zeros((map_h, map_w), dtype=np.int32)
            all_counts.append(counts)
            all_obs.append(obs_count)
            all_fates.append(fates)

        print(f"  Cached queries: {total_queries}")

        # Simulate regime detection from observations (as live mode does)
        total_class_obs = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_obs_sum = 0
        for si in range(seeds_count):
            mask = all_obs[si] > 0
            if mask.any():
                settlements_si = initial_states[si]["settlements"]
                dist_si = compute_settlement_distance(settlements_si, map_h, map_w)
                debias_weight = np.clip(dist_si * 0.3 + 0.5, 0.5, 3.0)
                for cls in range(NUM_CLASSES):
                    total_class_obs[cls] += (all_counts[si][:, :, cls][mask] * debias_weight[mask]).sum()
                total_obs_sum += (all_obs[si][mask] * debias_weight[mask]).sum()

        # Build adapted model with observations (as live mode does)
        if total_obs_sum > 200:
            observed_freq = total_class_obs / total_obs_sum
            adapted_obs = adapt_prior_to_regime(
                observed_freq, per_round, global_model[0], global_model[1],
            )
            adapted_learned = 0.85 * adapted_init + 0.15 * adapted_obs
            valid = global_model[1] > 5
            for idx in np.argwhere(valid):
                idx_tuple = tuple(idx)
                adapted_learned[idx_tuple] = np.maximum(adapted_learned[idx_tuple], PROB_FLOOR)
                adapted_learned[idx_tuple] /= adapted_learned[idx_tuple].sum()
            obs_model = (adapted_learned, global_model[1])
        else:
            obs_model = (adapted_init, global_model[1])

        # Quick model (prior-only with init-features regime detection)
        quick_model = (adapted_init, global_model[1])

        # Learn in-round transitions
        transitions = learn_transitions(all_counts, all_obs, initial_states, map_h, map_w)

        # Score both strategies against ground truth
        scores_quick = []
        scores_safe = []
        scores_server = []

        for si in range(seeds_count):
            for _retry in range(5):
                resp = session.get(f"{BASE}/astar-island/analysis/{rnd_id}/{si}")
                if resp.status_code == 429:
                    time.sleep(2 * (_retry + 1))
                    continue
                break
            if resp.status_code != 200:
                continue

            analysis = resp.json()
            gt = analysis["ground_truth"]
            server_score = analysis.get("score")

            # Quick: prior-only, no observations
            empty_counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
            empty_obs = np.zeros((map_h, map_w), dtype=np.int32)
            pred_quick = build_prediction(
                empty_counts, empty_obs, initial_states[si], map_h, map_w,
                learned_model=quick_model,
            )
            pred_quick = np.maximum(pred_quick, PROB_FLOOR)
            pred_quick /= pred_quick.sum(axis=-1, keepdims=True)

            # Safe: with observations, transitions, settlement fates
            pred_safe = build_prediction(
                all_counts[si], all_obs[si], initial_states[si], map_h, map_w,
                transitions=transitions, learned_model=obs_model,
                settlement_fates=all_fates[si],
            )
            pred_safe = np.maximum(pred_safe, PROB_FLOOR)
            pred_safe /= pred_safe.sum(axis=-1, keepdims=True)

            sq = score_prediction(pred_quick, gt)
            ss = score_prediction(pred_safe, gt)
            scores_quick.append(sq)
            scores_safe.append(ss)
            if server_score is not None:
                scores_server.append(server_score)

            obs_cells = int((all_obs[si] > 0).sum())
            print(f"  Seed {si}: quick={sq:.2f}  safe={ss:.2f}  "
                  f"delta={ss-sq:+.2f}  obs_cells={obs_cells}"
                  + (f"  server={server_score:.2f}" if server_score else ""))

            time.sleep(0.3)

        if scores_quick:
            avg_q = np.mean(scores_quick)
            avg_s = np.mean(scores_safe)
            delta = avg_s - avg_q
            print(f"\n  AVG: quick={avg_q:.2f}  safe={avg_s:.2f}  delta={delta:+.2f}")
            if scores_server:
                avg_srv = np.mean(scores_server)
                print(f"  Server (old code): {avg_srv:.2f}")
            results.append({
                "round": rnd_num,
                "quick": avg_q,
                "safe": avg_s,
                "delta": delta,
                "server": np.mean(scores_server) if scores_server else None,
                "queries": total_queries,
            })

    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY: --quick vs --safe (new code)")
        print(f"{'='*60}")
        print(f"{'Round':>6s} {'Quick':>8s} {'Safe':>8s} {'Delta':>8s} {'Server':>8s} {'Queries':>8s}")
        for r in results:
            srv = f"{r['server']:.2f}" if r['server'] else "N/A"
            print(f"{r['round']:6d} {r['quick']:8.2f} {r['safe']:8.2f} "
                  f"{r['delta']:+8.2f} {srv:>8s} {r['queries']:8d}")

        avg_q = np.mean([r['quick'] for r in results])
        avg_s = np.mean([r['safe'] for r in results])
        print(f"\nOverall: quick={avg_q:.2f}  safe={avg_s:.2f}  delta={avg_s-avg_q:+.2f}")
        wins = sum(1 for r in results if r['delta'] > 0)
        losses = sum(1 for r in results if r['delta'] < 0)
        print(f"Safe wins: {wins}/{len(results)}, Safe losses: {losses}/{len(results)}")


if __name__ == "__main__":
    session = get_session()
    test_strategies(session)
