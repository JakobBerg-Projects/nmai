#!/usr/bin/env python3
"""Analyze Cloud Run logs to identify agent failures and suggest improvements.

Usage:
    # Fetch logs and analyze
    gcloud run services logs read nmai-agent --region=europe-north1 --limit=300 > /tmp/logs.txt
    python3 analyze.py /tmp/logs.txt

    # Or pipe directly
    gcloud run services logs read nmai-agent --region=europe-north1 --limit=300 | python3 analyze.py
"""

import json
import os
import re
import sys
from collections import defaultdict

import anthropic

from performance_tracker import record_snapshot, get_trend_summary, should_revert


def parse_logs(text: str) -> list[dict]:
    """Extract TASK_REPORT entries from Cloud Run logs."""
    reports = []
    pattern = re.compile(
        r"TASK_REPORT: task=(\S+) iterations=(\d+) elapsed=([\d.]+)s write_calls=(\d+) errors=(\d+)"
        r"(?:\s+path=(\S+))?(?:\s+outcome=(\S+))?"
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            reports.append({
                "task_type": m.group(1),
                "iterations": int(m.group(2)),
                "elapsed": float(m.group(3)),
                "write_calls": int(m.group(4)),
                "errors": int(m.group(5)),
                "path": m.group(6) or "unknown",
                "outcome": m.group(7) or "unknown",
            })

    # Also look for error patterns
    error_pattern = re.compile(r"Tool (\S+) → .*\"success\":\s*false.*\"error\":\s*\"([^\"]+)\"")
    errors = []
    for line in text.splitlines():
        m = error_pattern.search(line)
        if m:
            errors.append({"tool": m.group(1), "error": m.group(2)[:200]})

    return reports, errors


def summarize(reports: list[dict], errors: list[dict]) -> str:
    """Create a summary of task performance."""
    if not reports:
        return "No TASK_REPORT entries found in logs. Make sure the agent has been deployed with the latest code."

    by_type = defaultdict(list)
    for r in reports:
        by_type[r["task_type"]].append(r)

    lines = ["## Task Performance Summary\n"]
    for task_type, runs in sorted(by_type.items()):
        avg_iters = sum(r["iterations"] for r in runs) / len(runs)
        avg_writes = sum(r["write_calls"] for r in runs) / len(runs)
        avg_errors = sum(r["errors"] for r in runs) / len(runs)
        avg_time = sum(r["elapsed"] for r in runs) / len(runs)
        lines.append(
            f"- **{task_type}**: {len(runs)} runs, "
            f"avg {avg_iters:.0f} iters, {avg_writes:.0f} writes, "
            f"{avg_errors:.0f} errors, {avg_time:.0f}s"
        )

    if errors:
        lines.append("\n## Common Tool Errors\n")
        error_counts = defaultdict(int)
        for e in errors:
            key = f"{e['tool']}: {e['error'][:100]}"
            error_counts[key] += 1
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"- ({count}x) {err}")

    return "\n".join(lines)


def analyze_with_claude(summary: str, tools_code: str, prompts_code: str, trend_info: str) -> str:
    """Send summary + code + trend data to Claude for analysis and suggestions."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Du er en AI-agent-optimaliserer for NM i AI Tripletex-konkurransen.

Her er resultatene fra siste submissions:

{summary}

{trend_info}

Her er gjeldende tools.py (forkortet til viktige deler):
```python
{tools_code[:8000]}
```

Her er gjeldende prompts.py:
```python
{prompts_code[:4000]}
```

Analyser:
1. Hvilke oppgavetyper feiler mest? Hvorfor?
2. Hvilke oppgavetyper har flest errors/write calls? Hvordan redusere?
3. Foreslå konkrete kodeendringer (med filnavn og linje) for å fikse de 3 viktigste problemene.
4. Prioriter endringene etter forventet poengeffekt.
5. VIKTIG: Se på trend-dataen. Har siste endringer forbedret eller forverret agenten?
   - Hvis FORVERRET: foreslå å rulle tilbake siste endring og prøv noe annet.
   - Hvis FORBEDRET: bygg videre på det som fungerte.
   - Hvis streak er negativ (flere forverringer på rad): ADVAR tydelig.

Svar på norsk. Vær konkret og handlingsrettet.""",
        }],
    )

    return response.content[0].text


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            log_text = f.read()
    else:
        log_text = sys.stdin.read()

    reports, errors = parse_logs(log_text)
    summary = summarize(reports, errors)

    print(summary)
    print("\n---\n")

    # Record performance snapshot and compute trends
    if reports:
        label = sys.argv[2] if len(sys.argv) > 2 else ""
        tracker_result = record_snapshot(reports, deploy_label=label)
        trend_info = get_trend_summary()

        print(trend_info)
        print()

        # Warn if on a losing streak
        if should_revert():
            print("!! ADVARSEL: 2+ forverringer på rad. Vurder å rulle tilbake siste endring! !!")
            print()

        if tracker_result["improved"]:
            print(">> Siste endring FORBEDRET agenten. Streak:", tracker_result["streak"])
        elif tracker_result["worsened"]:
            print(">> Siste endring FORVERRET agenten. Streak:", tracker_result["streak"])
        else:
            print(">> Ingen signifikant endring.")
        print()
    else:
        trend_info = "Ingen TASK_REPORT data funnet — kan ikke beregne trend."

    # Read current code for context
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base_dir, "tools.py")) as f:
            tools_code = f.read()
        with open(os.path.join(base_dir, "prompts.py")) as f:
            prompts_code = f.read()
    except FileNotFoundError:
        tools_code = "(not found)"
        prompts_code = "(not found)"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to enable Claude analysis.")
        print("Writing summary to suggestions.md...")
        with open(os.path.join(base_dir, "suggestions.md"), "w") as f:
            f.write(f"# Agent Analysis\n\n{summary}\n\n{trend_info}\n\n(Set ANTHROPIC_API_KEY for AI-powered suggestions)\n")
        return

    print("Analyzing with Claude...\n")
    suggestions = analyze_with_claude(summary, tools_code, prompts_code, trend_info)

    output_path = os.path.join(base_dir, "suggestions.md")
    with open(output_path, "w") as f:
        f.write(
            f"# Agent Analysis & Suggestions\n\n{summary}\n\n---\n\n"
            f"{trend_info}\n\n---\n\n"
            f"## Claude's Analysis\n\n{suggestions}\n"
        )

    print(suggestions)
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
