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
MAX_AGENT_TIME_SECONDS = 200
MAX_ITERATIONS = 8
MAX_CONSECUTIVE_ERRORS = 3

PREFETCH_LOOKUPS: dict[str, tuple[str, dict[str, Any]]] = {
    "departments": ("/department", {"fields": "id,name,departmentNumber", "count": 20}),
    "invoice_payment_types": ("/invoice/paymentType", {"fields": "id,description"}),
    "outgoing_payment_types": ("/ledger/paymentTypeOut", {"fields": "id,description"}),
    "activities": ("/activity", {"fields": "id,name,number", "count": 20}),
    "project_categories": ("/project/category", {"fields": "id,name,number"}),
    "vat_types": ("/ledger/vatType", {"fields": "id,name,number", "count": 20}),
    "travel_payment_types": ("/travelExpense/paymentType", {"fields": "id,description"}),
    "travel_cost_categories": ("/travelExpense/costCategory", {"fields": "id,description", "count": 30}),
    "travel_rate_categories": ("/travelExpense/rateCategory", {"fields": "id,name,type", "count": 20}),
    "employees": ("/employee", {"fields": "id,firstName,lastName,email,userType", "count": 50}),
    "customers": ("/customer", {"fields": "id,name,email,organizationNumber", "count": 50}),
    "suppliers": ("/supplier", {"fields": "id,name,email,organizationNumber", "count": 50}),
    "accounts": ("/ledger/account", {"fields": "id,number,name", "count": 1000}),
    # bank_accounts prefetch removed — /bank endpoint returns 400 via proxy
    "currencies": ("/currency", {"fields": "id,code", "count": 5}),
}


# ── prefetch ────────────────────────────────────────────────────────────

async def _prefetch(tripletex: TripletexClient) -> tuple[dict[str, list], str, bool]:
    """Fetch common reference data. Returns (structured_dict, compact_string, auth_ok)."""

    async def _fetch(key: str, path: str, params: dict) -> tuple[str, list, int]:
        r = await tripletex.request("GET", path, params=params)
        sc = r.get("status_code", 500)
        if isinstance(sc, int) and sc < 400:
            d = r.get("data", {})
            return key, d.get("values", []) if isinstance(d, dict) else [], sc
        return key, [], sc

    tasks = [_fetch(k, p, q) for k, (p, q) in PREFETCH_LOOKUPS.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ref: dict[str, list] = {}
    lines = ["PRE-FETCHED REFERENCE DATA (use these IDs directly — do NOT re-fetch):"]
    auth_failures = 0
    total = 0
    for item in results:
        if isinstance(item, Exception):
            continue
        key, vals, sc = item
        total += 1
        if sc == 401:
            auth_failures += 1
        ref[key] = vals
        if vals:
            max_chars = 1200 if key in ("accounts", "employees", "customers", "suppliers") else 800
            lines.append(f"  {key}: {json.dumps(vals, ensure_ascii=False)[:max_chars]}")

    auth_ok = auth_failures < total * 0.5  # If >50% return 401, auth is broken
    if not auth_ok:
        logger.error("AUTH FAILURE: %d/%d prefetch calls returned 401. Session token may be invalid.", auth_failures, total)

    return ref, "\n".join(lines) if len(lines) > 1 else "", auth_ok


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
    TaskType.CREATE_SUPPLIER_INVOICE: "create_supplier_invoice",
    TaskType.REGISTER_PAYMENT: "register_payment",
    TaskType.CREATE_PRODUCT: "create_product",
    TaskType.CREATE_DEPARTMENT: "create_department",
    TaskType.CREATE_CONTACT: "create_contact",
    TaskType.CREATE_VOUCHER: "create_voucher",
    TaskType.OPENING_BALANCE: "create_voucher",
    TaskType.YEAR_END_CLOSING: "create_voucher",
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
            temperature=0,
            system=(
                "Extract ALL values from the task and call the tool. "
                "If the task asks for MULTIPLE entities (e.g. 'create three departments'), "
                "make ONE tool call PER entity. Call the tool multiple times.\n"
                "Use values EXACTLY as written. Do NOT modify names, emails, or amounts.\n"
                "Roles: kontoadministrator/admin/administrator/Systemadministrator/account administrator→ALL_PRIVILEGES, "
                "prosjektleder/avdelingsleder/Projektleiter/chef de projet→DEPARTMENT_LEADER, "
                "regnskapsfører/Buchhalter/comptable/accountant→ACCOUNTANT, "
                "fakturaansvarlig/invoicing manager→INVOICING_MANAGER, "
                "personalansvarlig/HR manager→PERSONELL_MANAGER, revisor/auditor→AUDITOR.\n"
                "Voucher: positive=debit, negative=credit, MUST sum to 0. DOUBLE-CHECK sum before calling!\n"
                "Norwegian accounts: 1920=bank, 2400=leverandørgjeld, 3000=salg, 4000=varekjøp, 5000=lønn, "
                "6300=kontor, 6400=leie, 7700=bankgebyr, 2700=utg.mva, 8050=resultatdisponering, "
                "2050=egenkapital, 2090=opptjent egenkapital. NEVER use 8700 (doesn't exist)!\n"
                "For opening balance use useOpeningBalance:true.\n"
                "Year-end closing: ONE voucher with 2 postings — debit 8050 + credit 2050 (profit) or reverse (loss). "
                "Check pre-fetched accounts to verify which accounts exist.\n"
                "Invoice: inkl.mva→unitPriceIncludingVat, eksl.mva→unitPriceExcludingVat. vatPercent: 25 default.\n"
                "Supplier invoice: only send supplierName, invoiceNumber, invoiceDate. No lines or amounts.\n"
                "Product: use price for priceExcludingVat. vatPercent default 25. number=varenummer/produktnummer.\n"
                "Department: name + departmentNumber (string). Reminder: type SOFT_REMINDER=purring, NOTICE_OF_DEBT_COLLECTION=inkassovarsel.\n"
                "Contact: firstName + lastName + email + customerName to find customer.\n"
                "Timesheet: employeeName, projectName, activityName, entries:[{date,hours}]. "
                "hours as decimal. If '20 Stunden' on one date, entries:[{date:'YYYY-MM-DD',hours:20}].\n"
                "Payment: If 'paid in full'/'betalt fullt', do NOT set paidAmount (tool auto-detects). "
                "If specific NOK amount, set paidAmount. If foreign currency + exchange rate, multiply to NOK. "
                "customerName helps find the invoice.\n"
                "Project: name is required. Use projectManagerName, customerName, startDate, endDate if given.\n"
                "Norwegian number format: 1.000=1000 (period=thousands), 1,5=1.5 (comma=decimal)."
            ),
            tools=[tool_def],
            messages=[{"role": "user", "content": user_content}],
        )
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return False

        # Execute ALL tool calls (handles multi-entity tasks like "create 3 departments")
        any_success = False
        for tu in tool_uses:
            logger.info("Fast path tool: %s input: %s", tu.name, json.dumps(tu.input, ensure_ascii=False)[:500])
            result_str = await handler.execute(tu.name, tu.input)
            result = json.loads(result_str)
            logger.info("Fast path result: %s", result_str[:300])
            if result.get("success", False):
                any_success = True
        return any_success
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

    ref_dict, ref_str, auth_ok = await _prefetch(tripletex)
    logger.info("Pre-fetched %d reference categories (auth_ok=%s)", sum(1 for v in ref_dict.values() if v), auth_ok)

    if not auth_ok:
        # Auth is broken — try a simple GET to confirm
        test = await tripletex.request("GET", "/employee", params={"count": 1, "fields": "id"})
        if test.get("status_code") == 401:
            logger.error("FATAL: Authentication failed. Cannot proceed. Aborting.")
            await tripletex.close()
            return

    task_type = classify_task(prompt)
    logger.info("Classified task as: %s", task_type.value)

    handler = ToolHandler(tripletex, ref_dict)
    user_content = _build_user_content(prompt, files, ref_str)

    start = time.monotonic()

    # Allow fast path even with files for simple task types
    simple_fast = {TaskType.CREATE_EMPLOYEE, TaskType.CREATE_CUSTOMER, TaskType.CREATE_PRODUCT,
                   TaskType.CREATE_DEPARTMENT, TaskType.CREATE_CONTACT, TaskType.CREATE_SUPPLIER_INVOICE,
                   TaskType.REGISTER_PAYMENT, TaskType.CREATE_VOUCHER, TaskType.OPENING_BALANCE,
                   TaskType.YEAR_END_CLOSING}
    can_fast = task_type in FAST_PATH_ELIGIBLE and (not files or task_type in simple_fast)
    if can_fast:
        logger.info("Trying fast path for %s", task_type.value)
        if await _try_fast_path(client, handler, task_type, user_content):
            elapsed = time.monotonic() - start
            logger.info("Fast path succeeded")
            logger.info("TASK_REPORT: task=%s iterations=1 elapsed=%.1fs write_calls=1 errors=0 path=fast",
                        task_type.value, elapsed)
            await tripletex.close()
            return
        logger.info("Fast path failed, continuing with full agent loop")

    system_prompt = build_prompt(task_type)
    logger.info("System prompt: %d chars (task=%s)", len(system_prompt), task_type.value)

    messages: list[dict] = [{"role": "user", "content": user_content}]
    system_with_cache = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
    ]

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
                    temperature=0,
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
            auth_failed = False
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
                        # Detect 401 auth failure — abort loop
                        if isinstance(sc, int) and sc == 401:
                            auth_failed = True
                        is_err = (isinstance(sc, int) and sc >= 400) or rd.get("success") is False
                        if is_err:
                            batch_had_error = True
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                rd["_hint"] = (
                                    f"WARNING: {consecutive_errors} consecutive errors. "
                                    "Stop retrying the same approach. Try tripletex_api as fallback, "
                                    "try a DIFFERENT approach, or create prerequisite entities for partial credit."
                                )
                                rstr = json.dumps(rd, ensure_ascii=False)
                            elif consecutive_errors >= 2:
                                rd["_hint"] = (
                                    "Multiple errors. Read the error message carefully. "
                                    "Check required fields, correct IDs, and try again with fixes."
                                )
                                rstr = json.dumps(rd, ensure_ascii=False)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                tool_results.append({"type": "tool_result", "tool_use_id": tid, "content": rstr})

            # Abort immediately on auth failure — no point continuing
            if auth_failed:
                logger.error("401 Unauthorized in tool call — aborting agent loop")
                break

            if batch_had_error:
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            messages.append({"role": "user", "content": tool_results})

    finally:
        elapsed = time.monotonic() - start
        write_calls = sum(
            1 for m in messages if m["role"] == "user"
            for tc in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(tc, dict) and tc.get("type") == "tool_result"
        )
        error_count = sum(
            1 for m in messages if m["role"] == "user"
            for tc in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(tc, dict) and tc.get("type") == "tool_result"
            and '"success": false' in tc.get("content", "").lower()
        )
        # Determine outcome: success if any tool returned success, else partial/fail
        any_success = any(
            '"success": true' in tc.get("content", "").lower()
            for m in messages if m["role"] == "user"
            for tc in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(tc, dict) and tc.get("type") == "tool_result"
        )
        outcome = "success" if any_success else ("partial" if write_calls > 0 else "fail")
        logger.info(
            "TASK_REPORT: task=%s iterations=%d elapsed=%.1fs write_calls=%d errors=%d path=full outcome=%s",
            task_type.value, iterations, elapsed, write_calls, error_count, outcome,
        )
        await tripletex.close()
