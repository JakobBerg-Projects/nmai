"""Performance tracker — persists streak, trend, and per-task metrics across deploys.

Stores data in performance_history.json. Each deploy/analysis cycle appends a snapshot.
Used by analyze.py to determine if recent changes improved or worsened performance.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger("tracker")

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_history.json")


def _load_history() -> dict:
    """Load history from disk, or return empty structure."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "snapshots": [],
        "current_streak": 0,           # positive = consecutive improvements, negative = consecutive regressions
        "improved": False,
        "worsened": False,
        "total_snapshots": 0,
    }


def _save_history(history: dict) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _task_score(reports: list[dict]) -> dict[str, dict]:
    """Compute per-task aggregate metrics from TASK_REPORT entries."""
    from collections import defaultdict
    by_type: dict[str, list] = defaultdict(list)
    for r in reports:
        by_type[r["task_type"]].append(r)

    scores = {}
    for task_type, runs in by_type.items():
        avg_iters = sum(r["iterations"] for r in runs) / len(runs)
        avg_writes = sum(r["write_calls"] for r in runs) / len(runs)
        avg_errors = sum(r["errors"] for r in runs) / len(runs)
        avg_time = sum(r["elapsed"] for r in runs) / len(runs)
        # Efficiency score: lower is better (fewer writes, fewer errors, fewer iterations)
        efficiency = avg_writes + avg_errors * 2 + avg_iters * 0.5
        scores[task_type] = {
            "runs": len(runs),
            "avg_iterations": round(avg_iters, 1),
            "avg_write_calls": round(avg_writes, 1),
            "avg_errors": round(avg_errors, 1),
            "avg_elapsed": round(avg_time, 1),
            "efficiency_score": round(efficiency, 2),
        }
    return scores


def _overall_efficiency(task_scores: dict[str, dict]) -> float:
    """Weighted average efficiency across all tasks (lower = better)."""
    if not task_scores:
        return 999.0
    total_runs = sum(s["runs"] for s in task_scores.values())
    if total_runs == 0:
        return 999.0
    weighted = sum(s["efficiency_score"] * s["runs"] for s in task_scores.values())
    return round(weighted / total_runs, 2)


def record_snapshot(reports: list[dict], deploy_label: str = "") -> dict:
    """Record a new performance snapshot and compute trends.

    Returns a summary dict with streak, improved, worsened, and per-task deltas.
    """
    history = _load_history()
    task_scores = _task_score(reports)
    overall = _overall_efficiency(task_scores)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "label": deploy_label,
        "task_scores": task_scores,
        "overall_efficiency": overall,
        "total_runs": sum(s["runs"] for s in task_scores.values()),
    }

    # Compare to previous snapshot
    prev = history["snapshots"][-1] if history["snapshots"] else None
    improved = False
    worsened = False
    deltas: dict[str, dict] = {}

    if prev:
        prev_overall = prev.get("overall_efficiency", 999.0)
        # Lower efficiency score = better
        if overall < prev_overall - 0.1:
            improved = True
        elif overall > prev_overall + 0.1:
            worsened = True

        # Per-task deltas
        prev_scores = prev.get("task_scores", {})
        all_tasks = set(list(task_scores.keys()) + list(prev_scores.keys()))
        for task in all_tasks:
            cur = task_scores.get(task, {})
            prv = prev_scores.get(task, {})
            if cur and prv:
                delta_eff = round(cur["efficiency_score"] - prv["efficiency_score"], 2)
                delta_errors = round(cur["avg_errors"] - prv["avg_errors"], 1)
                delta_writes = round(cur["avg_write_calls"] - prv["avg_write_calls"], 1)
                deltas[task] = {
                    "efficiency_delta": delta_eff,
                    "error_delta": delta_errors,
                    "write_delta": delta_writes,
                    "direction": "improved" if delta_eff < -0.1 else ("worsened" if delta_eff > 0.1 else "stable"),
                }
            elif cur and not prv:
                deltas[task] = {"direction": "new_task"}
            else:
                deltas[task] = {"direction": "no_data_this_run"}

    # Update streak
    streak = history.get("current_streak", 0)
    if improved:
        streak = streak + 1 if streak > 0 else 1
    elif worsened:
        streak = streak - 1 if streak < 0 else -1
    else:
        streak = 0  # Reset on no change

    snapshot["improved"] = improved
    snapshot["worsened"] = worsened
    snapshot["streak"] = streak
    snapshot["deltas"] = deltas

    # Keep last 50 snapshots
    history["snapshots"].append(snapshot)
    if len(history["snapshots"]) > 50:
        history["snapshots"] = history["snapshots"][-50:]

    history["current_streak"] = streak
    history["improved"] = improved
    history["worsened"] = worsened
    history["total_snapshots"] = history.get("total_snapshots", 0) + 1

    _save_history(history)

    return {
        "overall_efficiency": overall,
        "improved": improved,
        "worsened": worsened,
        "streak": streak,
        "total_snapshots": history["total_snapshots"],
        "deltas": deltas,
        "task_scores": task_scores,
    }


def get_trend_summary() -> str:
    """Return a human-readable summary of performance trends."""
    history = _load_history()
    if not history["snapshots"]:
        return "Ingen historiske data enda. Kjør improve.sh etter noen submissions."

    latest = history["snapshots"][-1]
    streak = history.get("current_streak", 0)
    total = history.get("total_snapshots", 0)

    lines = [f"## Performance Trend (snapshot #{total})"]

    # Streak indicator
    if streak > 0:
        lines.append(f"  Streak: {streak} forbedringer på rad")
    elif streak < 0:
        lines.append(f"  Streak: {abs(streak)} forverringer på rad — REVERTER eller fiks!")
    else:
        lines.append("  Streak: Stabil (ingen signifikant endring)")

    # Overall
    lines.append(f"  Overall efficiency: {latest.get('overall_efficiency', '?')} (lavere = bedre)")

    if latest.get("improved"):
        lines.append("  Status: FORBEDRET siden forrige snapshot")
    elif latest.get("worsened"):
        lines.append("  Status: FORVERRET siden forrige snapshot")
    else:
        lines.append("  Status: Stabil")

    # Per-task deltas
    deltas = latest.get("deltas", {})
    if deltas:
        lines.append("\n### Per-task endringer:")
        for task, d in sorted(deltas.items()):
            direction = d.get("direction", "?")
            if direction == "improved":
                lines.append(f"  + {task}: forbedret (eff: {d.get('efficiency_delta', '?')}, errors: {d.get('error_delta', '?')}, writes: {d.get('write_delta', '?')})")
            elif direction == "worsened":
                lines.append(f"  - {task}: FORVERRET (eff: {d.get('efficiency_delta', '?')}, errors: {d.get('error_delta', '?')}, writes: {d.get('write_delta', '?')})")
            elif direction == "new_task":
                lines.append(f"  * {task}: ny oppgavetype (ingen sammenligning)")
            elif direction == "no_data_this_run":
                lines.append(f"  ? {task}: ingen data denne kjøringen")
            else:
                lines.append(f"  = {task}: stabil")

    # Historical trend (last 5)
    if len(history["snapshots"]) >= 2:
        lines.append("\n### Siste snapshots:")
        for snap in history["snapshots"][-5:]:
            ts = snap.get("timestamp", "?")[:16]
            eff = snap.get("overall_efficiency", "?")
            status = "UP" if snap.get("improved") else ("DOWN" if snap.get("worsened") else "STABLE")
            runs = snap.get("total_runs", 0)
            lines.append(f"  {ts} | eff={eff} | {status} | {runs} runs")

    return "\n".join(lines)


def should_revert() -> bool:
    """Returns True if the last 2+ snapshots show worsening — signal to revert."""
    history = _load_history()
    return history.get("current_streak", 0) <= -2
