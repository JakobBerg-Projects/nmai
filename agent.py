"""Core agent loop with task classification, high-level tools, and fast paths."""

import asyncio
import base64
import json
import logging
import os
import time
from datetime import date
from typing import Any

import anthropic

from classifier import TaskType, classify_task, FAST_PATH_ELIGIBLE
from prompts import build_prompt
from tools import TOOL_DEFINITIONS, ToolHandler
from tripletex import TripletexClient

logger = logging.getLogger("agent")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
MAX_AGENT_TIME_SECONDS = 240
MAX_ITERATIONS = 25
MAX_CONSECUTIVE_ERRORS = 3

PREFETCH_LOOKUPS: dict[str, tuple[str, dict[str, Any]]] = {
    "departments": ("/department", {"fields": "id,name", "count": 10}),
    "invoice_payment_types": ("/invoice/paymentType", {"fields": "id,description"}),
    "outgoing_payment_types": ("/ledger/paymentTypeOut", {"fields": "id,description"}),
    "activities": ("/activity", {"fields": "id,name", "count": 10}),
    "project_categories": ("/project/category", {"fields": "id,name"}),
    "vat_types": ("/ledger/vatType", {"fields": "id,name,number", "count": 10}),
    "travel_payment_types": ("/travelExpense/paymentType", {"fields": "id,description"}),
    "travel_cost_categories": ("/travelExpense/costCategory", {"fields": "id,description", "count": 15}),
    "travel_rate_categories": ("/travelExpense/rateCategory", {"fields": "id,name", "count": 10}),
}


# ── prefetch ────────────────────────────────────────────────────────────

async def _prefetch(tripletex: TripletexClient) -> tuple[dict[str, list], str]:
    """Fetch common reference data. Returns (structured_dict, compact_string)."""

    async def _fetch(key: str, path: str, params: dict) -> tuple[str, list]:
        r = await tripletex.request("GET", path, params=params)
        sc = r.get("status_code", 500)
        if isinstance(sc, int) and sc < 400:
            d = r.get("data", {})
            return key, d.get("values", []) if isinstance(d, dict) else []
        return key, []

    tasks = [_fetch(k, p, q) for k, (p, q) in PREFETCH_LOOKUPS.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ref: dict[str, list] = {}
    lines = ["PRE-FETCHED REFERENCE DATA (use these IDs directly — do NOT re-fetch):"]
    for item in results:
        if isinstance(item, Exception):
            continue
        key, vals = item
        ref[key] = vals
        if vals:
            lines.append(f"  {key}: {json.dumps(vals, ensure_ascii=False)[:600]}")

    return ref, "\n".join(lines) if len(lines) > 1 else ""


# ── user content ────────────────────────────────────────────────────────

def _build_user_content(prompt: str, files: list[dict], ref_str: str = "") -> list[dict]:
    blocks: list[dict] = []

    for f in files:
        mime = f.get("mime_type", "application/octet-stream")
        b64 = f.get("content_base64", "")
        fname = f.get("filename", "unknown")
        if not b64:
            continue
        if mime.startswith("image/"):
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
            blocks.append({"type": "text", "text": f"[Image: {fname}]"})
        elif mime == "application/pdf":
            blocks.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}})
            blocks.append({"type": "text", "text": f"[PDF: {fname}]"})
        else:
            try:
                text = base64.b64decode(b64).decode("utf-8", errors="replace")[:8000]
                blocks.append({"type": "text", "text": f"[File: {fname}]\n{text}"})
            except Exception:
                blocks.append({"type": "text", "text": f"[Binary file: {fname}]"})

    today = date.today().isoformat()
    ref_part = f"\n\n{ref_str}" if ref_str else ""
    blocks.append({
        "type": "text",
        "text": f"Today's date: {today}\n\nTASK:\n{prompt}\n\nComplete this task using the available tools.{ref_part}",
    })
    return blocks


# ── fast path ───────────────────────────────────────────────────────────

FAST_PATH_TOOL_MAP: dict[TaskType, str] = {
    TaskType.CREATE_EMPLOYEE: "create_employee",
    TaskType.CREATE_CUSTOMER: "create_customer",
}


async def _try_fast_path(
    client: anthropic.Anthropic,
    handler: ToolHandler,
    task_type: TaskType,
    user_content: list[dict],
) -> bool:
    """Single-shot tool call for simple Tier 1 tasks. Returns True if successful."""
    tool_name = FAST_PATH_TOOL_MAP.get(task_type)
    if not tool_name:
        return False

    tool_def = next((t for t in TOOL_DEFINITIONS if t["name"] == tool_name), None)
    if not tool_def:
        return False

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=(
                "Extract ALL values from the task and call the tool. "
                "Use values EXACTLY as written. Do NOT modify names, emails, or amounts."
            ),
            tools=[tool_def],
            messages=[{"role": "user", "content": user_content}],
        )
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return False

        result_str = await handler.execute(tool_uses[0].name, tool_uses[0].input)
        result = json.loads(result_str)
        logger.info("Fast path result: %s", result_str[:300])
        return result.get("success", False)
    except Exception as e:
        logger.warning("Fast path failed, falling back: %s", e)
        return False


# ── main agent loop ─────────────────────────────────────────────────────

async def run_agent(
    prompt: str,
    files: list[dict],
    base_url: str,
    session_token: str,
) -> None:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tripletex = TripletexClient(base_url, session_token)

    ref_dict, ref_str = await _prefetch(tripletex)
    logger.info("Pre-fetched %d reference categories", sum(1 for v in ref_dict.values() if v))

    task_type = classify_task(prompt)
    logger.info("Classified task as: %s", task_type.value)

    handler = ToolHandler(tripletex, ref_dict)
    user_content = _build_user_content(prompt, files, ref_str)

    if task_type in FAST_PATH_ELIGIBLE and not files:
        logger.info("Trying fast path for %s", task_type.value)
        if await _try_fast_path(client, handler, task_type, user_content):
            logger.info("Fast path succeeded")
            await tripletex.close()
            return
        logger.info("Fast path failed, continuing with full agent loop")

    system_prompt = build_prompt(task_type)
    logger.info("System prompt: %d chars (task=%s)", len(system_prompt), task_type.value)

    messages: list[dict] = [{"role": "user", "content": user_content}]
    system_with_cache = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
    ]

    start = time.monotonic()
    iterations = 0
    consecutive_errors = 0

    try:
        while iterations < MAX_ITERATIONS:
            elapsed = time.monotonic() - start
            if elapsed > MAX_AGENT_TIME_SECONDS:
                logger.warning("Timeout after %.1fs", elapsed)
                break

            iterations += 1
            logger.info("Iteration %d (%.1fs)", iterations, elapsed)

            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=system_with_cache,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except anthropic.APIStatusError as e:
                logger.error("Anthropic error: %s", e)
                if e.status_code == 429:
                    wait = min(20, 5 + iterations)
                    logger.info("Rate limited, waiting %ds", wait)
                    await asyncio.sleep(wait)
                    continue
                raise

            logger.info(
                "Response: stop=%s blocks=%d cache_read=%s",
                response.stop_reason, len(response.content),
                getattr(getattr(response, "usage", None), "cache_read_input_tokens", "?"),
            )

            if response.stop_reason == "end_turn":
                for b in response.content:
                    if hasattr(b, "text"):
                        logger.info("Final: %s", b.text[:300])
                break

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            assistant_content = []
            for b in response.content:
                if b.type == "text":
                    assistant_content.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use", "id": b.id,
                        "name": b.name, "input": b.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            if len(tool_uses) > 1:
                raw = await asyncio.gather(
                    *[handler.execute(tu.name, tu.input) for tu in tool_uses],
                    return_exceptions=True,
                )
            else:
                try:
                    raw = [await handler.execute(tool_uses[0].name, tool_uses[0].input)]
                except Exception as exc:
                    raw = [exc]

            tool_results = []
            batch_had_error = False
            for i, item in enumerate(raw):
                tid = tool_uses[i].id
                if isinstance(item, Exception):
                    rstr = json.dumps({"error": str(item)[:300], "success": False})
                    batch_had_error = True
                else:
                    rstr = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                    logger.info("Tool %s → %s", tool_uses[i].name, rstr[:300])

                    try:
                        rd = json.loads(rstr)
                        sc = rd.get("status_code")
                        is_err = (isinstance(sc, int) and sc >= 400) or rd.get("success") is False
                        if is_err:
                            batch_had_error = True
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                rd["_hint"] = (
                                    f"WARNING: {consecutive_errors} consecutive errors. "
                                    "Stop retrying. Try a DIFFERENT approach or finish now."
                                )
                                rstr = json.dumps(rd, ensure_ascii=False)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                tool_results.append({"type": "tool_result", "tool_use_id": tid, "content": rstr})

            if batch_had_error:
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            messages.append({"role": "user", "content": tool_results})

    finally:
        await tripletex.close()

    logger.info("Done: %d iterations, %.1fs", iterations, time.monotonic() - start)
