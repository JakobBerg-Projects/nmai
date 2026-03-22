#!/usr/bin/env python3
"""
Detailed per-round analysis: backtest each completed round individually,
compute per-class KL contribution, and identify where the model is weakest.
"""
import numpy as np
import time
from solve import (
    get_session, get_round, load_learned_priors, build_prediction,
    build_initial_prior, score_prediction, compute_initial_features,
    adapt_prior_from_initial_state, adapt_prior_to_regime,
    TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR, BASE,
    compute_settlement_distance, distance_bucket,
    check_adjacent_to_ocean, check_adjacent_to_forest,
    settlement_density_bucket,
)


def per_class_kl_analysis(prediction, ground_truth, initial_state, map_h, map_w):
    """Compute detailed KL divergence breakdown by class and terrain type."""
    p = np.array(ground_truth, dtype=np.float64)
    q = np.array(prediction, dtype=np.float64)

    p_safe = np.clip(p, 1e-15, None)
    q_safe = np.clip(q, 1e-15, None)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=-1)
    kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)

    total_entropy = entropy.sum()
    if total_entropy < 1e-10:
        return {}

    gt_argmax = p.argmax(axis=-1)
    grid = initial_state["grid"]

    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    results = {}

    # Per ground-truth class
    for cls in range(NUM_CLASSES):
        mask = gt_argmax == cls
        if mask.sum() == 0:
            continue

        weighted_kl_cls = (entropy[mask] * kl[mask]).sum()
        entropy_cls = entropy[mask].sum()
        n_cells = int(mask.sum())
        avg_kl = kl[mask].mean()
        contribution = weighted_kl_cls / total_entropy  # Fraction of total weighted KL

        # What did we predict for these cells?
        pred_argmax = q[mask].argmax(axis=-1)
        pred_correct = (pred_argmax == cls).sum()

        # Average predicted probability for the correct class
        avg_pred_prob = q[mask, cls].mean()

        results[class_names[cls]] = {
            "n_cells": n_cells,
            "avg_kl": float(avg_kl),
            "weighted_contribution": float(contribution),
            "accuracy": float(pred_correct / n_cells),
            "avg_pred_prob": float(avg_pred_prob),
            "entropy_share": float(entropy_cls / total_entropy),
        }

    # Per initial terrain type → ground truth class (transition analysis)
    terrain_names = {0: "Empty", 10: "Ocean", 11: "Plains", 1: "Settlement",
                     2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}
    transitions = {}
    for y in range(map_h):
        for x in range(map_w):
            initial_terrain = grid[y][x]
            gt_cls = gt_argmax[y, x]
            terrain_name = terrain_names.get(initial_terrain, f"T{initial_terrain}")
            gt_name = class_names[gt_cls]
            key = f"{terrain_name}→{gt_name}"
            if key not in transitions:
                transitions[key] = {"count": 0, "kl_sum": 0.0, "entropy_kl_sum": 0.0}
            transitions[key]["count"] += 1
            transitions[key]["kl_sum"] += kl[y, x]
            transitions[key]["entropy_kl_sum"] += entropy[y, x] * kl[y, x]

    # Sort by weighted KL contribution
    for key in transitions:
        transitions[key]["weighted_contribution"] = transitions[key]["entropy_kl_sum"] / total_entropy

    results["_transitions"] = transitions
    results["_total_weighted_kl"] = float((entropy * kl).sum() / total_entropy)

    return results


def analyze_all_rounds(session):
    """Run detailed analysis on each completed round."""
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    completed = sorted(
        [r for r in rounds if r["status"] == "completed"],
        key=lambda r: r["round_number"],
    )

    global_model, per_round = load_learned_priors()
    has_model = global_model[0] is not None

    if not has_model:
        print("No learned priors! Run --learn first.")
        return

    all_round_results = []

    for rnd in completed:
        rnd_num = rnd["round_number"]
        for _retry in range(5):
            resp = session.get(f"{BASE}/astar-island/rounds/{rnd['id']}")
            if resp.status_code == 429:
                time.sleep(2 * (_retry + 1))
                continue
            break
        if resp.status_code != 200:
            print(f"Skipping round {rnd_num}")
            continue
        detail = resp.json()

        map_w = detail["map_width"]
        map_h = detail["map_height"]
        seeds_count = detail["seeds_count"]
        initial_states = detail["initial_states"]

        # Build regime-adapted model (leave-one-out would be ideal but complex)
        adapted_model = global_model
        if per_round is not None:
            init_features = compute_initial_features(initial_states, map_h, map_w)
            adapted_learned = adapt_prior_from_initial_state(
                init_features, per_round, global_model[0], global_model[1],
            )
            adapted_model = (adapted_learned, global_model[1])

        round_scores = []
        round_kl_details = []

        for seed_idx in range(seeds_count):
            for _retry in range(5):
                resp = session.get(f"{BASE}/astar-island/analysis/{rnd['id']}/{seed_idx}")
                if resp.status_code == 429:
                    time.sleep(2 * (_retry + 1))
                    continue
                break
            if resp.status_code != 200:
                continue

            analysis = resp.json()
            ground_truth = analysis["ground_truth"]
            server_score = analysis.get("score")

            # Build prediction with global model (no observations)
            counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
            obs_count = np.zeros((map_h, map_w), dtype=np.int32)

            pred_global = build_prediction(
                counts, obs_count, initial_states[seed_idx], map_h, map_w,
                learned_model=global_model,
            )
            pred_global = np.maximum(pred_global, PROB_FLOOR)
            pred_global /= pred_global.sum(axis=-1, keepdims=True)

            pred_adapted = build_prediction(
                counts, obs_count, initial_states[seed_idx], map_h, map_w,
                learned_model=adapted_model,
            )
            pred_adapted = np.maximum(pred_adapted, PROB_FLOOR)
            pred_adapted /= pred_adapted.sum(axis=-1, keepdims=True)

            score_g = score_prediction(pred_global, ground_truth)
            score_a = score_prediction(pred_adapted, ground_truth)

            kl_details = per_class_kl_analysis(
                pred_adapted, ground_truth, initial_states[seed_idx], map_h, map_w)

            round_scores.append({
                "seed": seed_idx,
                "global": score_g,
                "adapted": score_a,
                "server": server_score,
                "weighted_kl": kl_details.get("_total_weighted_kl", 0),
            })
            round_kl_details.append(kl_details)

            time.sleep(0.3)

        if not round_scores:
            continue

        avg_global = np.mean([s["global"] for s in round_scores])
        avg_adapted = np.mean([s["adapted"] for s in round_scores])
        server_scores = [s["server"] for s in round_scores if s["server"] is not None]
        avg_server = np.mean(server_scores) if server_scores else None

        print(f"\n{'='*60}")
        print(f"Round {rnd_num}: {map_w}x{map_h}, {seeds_count} seeds")
        print(f"  Global prior avg:  {avg_global:.2f}")
        print(f"  Adapted prior avg: {avg_adapted:.2f}")
        if avg_server is not None:
            print(f"  Server avg (submitted): {avg_server:.2f}")

        # Aggregate class-level analysis across seeds
        class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
        print(f"\n  Per-class breakdown (averaged across seeds):")
        for cls_name in class_names:
            cls_data = [d[cls_name] for d in round_kl_details if cls_name in d]
            if not cls_data:
                continue
            avg_n = np.mean([c["n_cells"] for c in cls_data])
            avg_kl = np.mean([c["avg_kl"] for c in cls_data])
            avg_contrib = np.mean([c["weighted_contribution"] for c in cls_data])
            avg_acc = np.mean([c["accuracy"] for c in cls_data])
            avg_pred = np.mean([c["avg_pred_prob"] for c in cls_data])
            avg_entropy = np.mean([c["entropy_share"] for c in cls_data])
            print(f"    {cls_name:12s}: cells={avg_n:5.0f}  kl={avg_kl:.4f}  "
                  f"contrib={avg_contrib:.3f}  acc={avg_acc:.2%}  "
                  f"pred_p={avg_pred:.3f}  entropy_share={avg_entropy:.3f}")

        # Top 10 worst transitions by weighted KL contribution
        all_trans = {}
        for d in round_kl_details:
            trans = d.get("_transitions", {})
            for key, val in trans.items():
                if key not in all_trans:
                    all_trans[key] = {"count": 0, "weighted_contribution": 0.0}
                all_trans[key]["count"] += val["count"]
                all_trans[key]["weighted_contribution"] += val["weighted_contribution"]

        # Average over seeds
        n_seeds = len(round_kl_details)
        for key in all_trans:
            all_trans[key]["weighted_contribution"] /= n_seeds

        sorted_trans = sorted(all_trans.items(), key=lambda x: -x[1]["weighted_contribution"])
        print(f"\n  Top 10 worst transitions (by weighted KL contribution):")
        for key, val in sorted_trans[:10]:
            print(f"    {key:25s}: cells={val['count']:5d}  contrib={val['weighted_contribution']:.4f}")

        all_round_results.append({
            "round": rnd_num,
            "avg_global": avg_global,
            "avg_adapted": avg_adapted,
            "avg_server": avg_server,
            "kl_details": round_kl_details,
        })

    # Summary across all rounds
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS ALL {len(all_round_results)} ROUNDS")
    print(f"{'='*60}")
    print(f"{'Round':>6s} {'Global':>8s} {'Adapted':>8s} {'Server':>8s}")
    for r in all_round_results:
        server_str = f"{r['avg_server']:.2f}" if r['avg_server'] is not None else "N/A"
        print(f"{r['round']:6d} {r['avg_global']:8.2f} {r['avg_adapted']:8.2f} {server_str:>8s}")

    print(f"\nOverall global avg:  {np.mean([r['avg_global'] for r in all_round_results]):.2f}")
    print(f"Overall adapted avg: {np.mean([r['avg_adapted'] for r in all_round_results]):.2f}")

    # Aggregate class contributions across ALL rounds
    print(f"\nAGGREGATE CLASS CONTRIBUTIONS (all rounds):")
    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    for cls_name in class_names:
        all_cls = []
        for r in all_round_results:
            for d in r["kl_details"]:
                if cls_name in d:
                    all_cls.append(d[cls_name])
        if not all_cls:
            continue
        avg_kl = np.mean([c["avg_kl"] for c in all_cls])
        avg_contrib = np.mean([c["weighted_contribution"] for c in all_cls])
        avg_acc = np.mean([c["accuracy"] for c in all_cls])
        avg_pred = np.mean([c["avg_pred_prob"] for c in all_cls])
        print(f"  {cls_name:12s}: avg_kl={avg_kl:.4f}  contrib={avg_contrib:.3f}  "
              f"acc={avg_acc:.2%}  avg_pred_p={avg_pred:.3f}")


if __name__ == "__main__":
    session = get_session()
    analyze_all_rounds(session)
