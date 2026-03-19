"""Core agent loop: LLM tool-calling cycle against the Tripletex API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import date
from typing import Any

from config import AGENT_MAX_ITERATIONS, AGENT_TIMEOUT_SECONDS
from file_handler import ExtractedFile, build_file_context
from llm import LLMClient, LLMResponse
from prompts import SYSTEM_PROMPT
from tools import TOOLS
from tripletex import TripletexClient

logger = logging.getLogger(__name__)

_KEYWORDS_VAT = re.compile(
    r"(?<![.\w@])\b(mva|vat|merverdiavgift|faktura|invoice|produkt|product|ordre|order"
    r"|Rechnung|impuesto|steuer|tax)\b(?![.\w@])",
    re.IGNORECASE,
)
_KEYWORDS_PAYMENT = re.compile(
    r"\b(betaling|payment|pago|Zahlung|pagamento|paiement|innbetaling)\b",
    re.IGNORECASE,
)
_KEYWORDS_TRAVEL = re.compile(
    r"\b(reiseregning|travel.?expense|reisekostenabrechnung|gastos.?de.?viaje"
    r"|despesas.?de.?viagem|note.?de.?frais)\b",
    re.IGNORECASE,
)


async def run_agent(
    prompt: str,
    extracted_files: list[ExtractedFile],
    tripletex: TripletexClient,
    llm: LLMClient,
) -> dict[str, Any]:
    """Execute the agent loop until the LLM decides the task is complete."""
    start = time.monotonic()
    deadline = start + AGENT_TIMEOUT_SECONDS

    api_log: list[dict[str, Any]] = []
    error_count = 0
    call_count = 0
    # Per-session GET cache: cache_key -> (serialized_result, call_info)
    get_cache: dict[str, tuple[str, dict[str, Any]]] = {}

    ref_data = await _prefetch_reference_data(prompt, tripletex, api_log)
    call_count += len(ref_data.get("_calls", []))

    messages = _build_initial_messages(prompt, extracted_files, llm, ref_data)

    for iteration in range(1, AGENT_MAX_ITERATIONS + 1):
        if time.monotonic() > deadline:
            logger.warning("Agent timeout after %d iterations", iteration)
            break

        remaining = deadline - time.monotonic()
        try:
            response: LLMResponse = await asyncio.wait_for(
                llm.chat(messages, TOOLS),
                timeout=max(remaining, 5),
            )
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out at iteration %d", iteration)
            break
        except Exception:
            logger.exception("LLM call failed at iteration %d", iteration)
            break

        if not response.tool_calls:
            logger.info(
                "Agent done after %d iterations, %d API calls, %d errors. LLM: %s",
                iteration, call_count, error_count,
                (response.text or "")[:200],
            )
            return _build_summary(prompt, call_count, error_count, start, api_log, "completed")

        messages.append(
            llm.format_assistant_tool_calls(response.text, response.tool_calls)
        )

        if time.monotonic() > deadline:
            return _build_summary(prompt, call_count, error_count, start, api_log, "timeout")

        # Execute all tool calls in parallel — independent calls within one LLM
        # response can safely run concurrently.
        tool_results = await asyncio.gather(*[
            _execute_tool(tc.name, tc.arguments, tripletex, get_cache)
            for tc in response.tool_calls
        ])

        for tc, (result_str, call_info) in zip(response.tool_calls, tool_results):
            call_count += 1
            api_log.append(call_info)

            if call_info.get("is_error"):
                error_count += 1
                friendly = _format_error_for_llm(call_info)
                if friendly:
                    result_str = friendly

            messages.append(llm.format_tool_result(tc.id, result_str))

    elapsed = time.monotonic() - start
    logger.info(
        "Agent finished: %d API calls, %d errors, %.1fs elapsed",
        call_count, error_count, elapsed,
    )
    return _build_summary(prompt, call_count, error_count, start, api_log, "max_iterations")


async def _fetch_one_ref(
    key: str,
    path: str,
    params: dict | None,
    tripletex: TripletexClient,
) -> tuple[str, list[Any], dict[str, Any] | None]:
    """Fetch a single reference endpoint. Returns (key, values, call_info)."""
    try:
        resp = await tripletex.request("GET", path, params=params)
        is_error = not resp.ok
        call_info: dict[str, Any] = {
            "tool": "tripletex_request",
            "method": "GET",
            "path": path,
            "status_code": resp.status_code,
            "is_error": is_error,
            "timestamp": time.time(),
        }
        if is_error:
            call_info["error_type"] = _classify_error(resp.status_code, resp.body)
            call_info["validation_messages"] = _extract_validation_messages(resp.body)

        values: list[Any] = []
        if resp.ok and isinstance(resp.body, dict):
            values = resp.body.get("values", [])
            logger.info("Pre-fetched %s: %d items", key, len(values))

        return key, values, call_info
    except Exception:
        logger.warning("Failed to pre-fetch %s from %s", key, path)
        return key, [], None


async def _prefetch_reference_data(
    prompt: str,
    tripletex: TripletexClient,
    api_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pre-fetch reference data in parallel based on prompt keywords."""
    ref_data: dict[str, Any] = {"_calls": []}
    tasks: list[tuple[str, str, dict | None]] = []

    needs_vat = bool(_KEYWORDS_VAT.search(prompt))
    needs_payment = bool(_KEYWORDS_PAYMENT.search(prompt))
    needs_travel = bool(_KEYWORDS_TRAVEL.search(prompt))

    if needs_vat:
        tasks.append(("vat_types", "/ledger/vatType", {
            "count": 100, "fields": "id,number,name,percentage",
            "typeOfVat": "OUTGOING",
        }))
    if needs_payment:
        tasks.append(("payment_types", "/invoice/paymentType", {"count": 100, "fields": "id,description"}))
    if needs_travel:
        tasks.append(("cost_categories", "/travelExpense/costCategory", {"count": 100, "fields": "id,name,number"}))
        tasks.append(("travel_payment_types", "/travelExpense/paymentType", {"count": 100, "fields": "id,description"}))

    if not tasks:
        return ref_data

    # Fetch all reference data in parallel
    results = await asyncio.gather(*[
        _fetch_one_ref(key, path, params, tripletex)
        for key, path, params in tasks
    ])

    for key, values, call_info in results:
        if call_info is not None:
            api_log.append(call_info)
            ref_data["_calls"].append(call_info)
        if values:
            ref_data[key] = values

    return ref_data


def _format_ref_data_context(ref_data: dict[str, Any]) -> str:
    """Format pre-fetched reference data as context for the LLM."""
    parts: list[str] = []

    if "vat_types" in ref_data:
        parts.append("## Pre-fetched OUTGOING VAT Types (from GET /ledger/vatType)")
        parts.append("Use these exact IDs for order lines and products — do NOT guess VAT type IDs.")
        parts.append("IMPORTANT: For standard 25% Norwegian MVA, pick the one with percentage=25.0")
        common = []
        other = []
        for vt in ref_data["vat_types"]:
            pct = vt.get("percentage")
            entry = (f"  - id={vt.get('id')}, number=\"{vt.get('number')}\", "
                     f"name=\"{vt.get('name')}\", percentage={pct}")
            if pct in (25.0, 15.0, 12.0, 0.0):
                common.append(entry)
            else:
                other.append(entry)
        if common:
            parts.append("### Common rates:")
            parts.extend(common)
        if other:
            parts.append("### Other rates:")
            parts.extend(other[:10])
        parts.append("")

    if "payment_types" in ref_data:
        parts.append("## Pre-fetched Payment Types (from GET /invoice/paymentType)")
        parts.append("Use these exact IDs for :payment calls — do NOT guess.")
        for pt in ref_data["payment_types"][:20]:
            parts.append(f"  - id={pt.get('id')}, description=\"{pt.get('description')}\"")
        parts.append("")

    if "cost_categories" in ref_data:
        parts.append("## Pre-fetched Travel Cost Categories (from GET /travelExpense/costCategory)")
        for cc in ref_data["cost_categories"][:20]:
            parts.append(f"  - id={cc.get('id')}, name=\"{cc.get('name')}\", number=\"{cc.get('number')}\"")
        parts.append("")

    if "travel_payment_types" in ref_data:
        parts.append("## Pre-fetched Travel Payment Types (from GET /travelExpense/paymentType)")
        for tp in ref_data["travel_payment_types"][:20]:
            parts.append(f"  - id={tp.get('id')}, description=\"{tp.get('description')}\"")
        parts.append("")

    return "\n".join(parts)


def _build_summary(
    prompt: str,
    call_count: int,
    error_count: int,
    start: float,
    api_log: list[dict[str, Any]],
    exit_reason: str,
) -> dict[str, Any]:
    return {
        "prompt_preview": prompt[:200],
        "total_api_calls": call_count,
        "error_count": error_count,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "exit_reason": exit_reason,
        "api_calls": api_log,
    }


def _build_initial_messages(
    prompt: str,
    extracted_files: list[ExtractedFile],
    llm: LLMClient,
    ref_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Construct the initial message list with system prompt, file context, and task."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    user_parts: list[Any] = []
    file_context = build_file_context(extracted_files)

    today = date.today().isoformat()
    text_content = f"# Task\n\nToday's date: {today}\n\n{prompt}"
    if file_context:
        text_content += f"\n\n{file_context}"

    if ref_data:
        ref_context = _format_ref_data_context(ref_data)
        if ref_context:
            text_content += (
                "\n\n# Pre-fetched Reference Data\n"
                "The following reference data was already fetched from the API. "
                "Use these IDs directly — do NOT call these endpoints again.\n\n"
                + ref_context
            )

    user_parts.append({"type": "text", "text": text_content})

    for ef in extracted_files:
        if ef.image_base64 and ef.mime_type:
            user_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{ef.mime_type};base64,{ef.image_base64}",
                },
            })

    if len(user_parts) == 1:
        messages.append({"role": "user", "content": text_content})
    else:
        messages.append({"role": "user", "content": user_parts})

    return messages


async def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    tripletex: TripletexClient,
    get_cache: dict[str, tuple[str, dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Dispatch a tool call and return (result_string, call_info_dict)."""
    call_info: dict[str, Any] = {
        "tool": name,
        "arguments": arguments,
        "timestamp": time.time(),
    }

    if name != "tripletex_request":
        error_result = json.dumps({"error": f"Unknown tool: {name}"})
        call_info.update(status_code=0, is_error=True, error_type="unknown_tool")
        return error_result, call_info

    method = arguments.get("method", "GET")
    path = arguments.get("path", "/")
    params = arguments.get("params")
    json_body = arguments.get("json_body")

    if not path.startswith("/"):
        path = "/" + path

    call_info["method"] = method
    call_info["path"] = path

    # Return cached GET responses to avoid duplicate lookups
    if method == "GET" and get_cache is not None:
        cache_key = f"{path}?{json.dumps(params or {}, sort_keys=True)}"
        if cache_key in get_cache:
            logger.info("Cache hit for GET %s", path)
            cached_str, cached_info = get_cache[cache_key]
            # Return a copy with updated timestamp
            return cached_str, {**cached_info, "timestamp": time.time(), "from_cache": True}

    try:
        resp = await tripletex.request(method, path, params=params, json_body=json_body)

        # Retry once on rate limit
        if resp.status_code == 429:
            logger.warning("Rate limited on %s %s, retrying after 3s", method, path)
            await asyncio.sleep(3)
            resp = await tripletex.request(method, path, params=params, json_body=json_body)

    except Exception as exc:
        error_result = json.dumps({"error": f"Request failed: {exc}"})
        call_info.update(status_code=0, is_error=True, error_type="request_exception")
        return error_result, call_info

    call_info["status_code"] = resp.status_code
    # Treat both 4xx and 5xx as errors
    call_info["is_error"] = not resp.ok

    if call_info["is_error"]:
        call_info["error_type"] = _classify_error(resp.status_code, resp.body)
        call_info["validation_messages"] = _extract_validation_messages(resp.body)

    result = {"status_code": resp.status_code, "body": resp.body}
    serialized = _smart_serialize(result)

    # Cache successful GET responses for the session
    if method == "GET" and resp.ok and get_cache is not None:
        cache_key = f"{path}?{json.dumps(params or {}, sort_keys=True)}"
        get_cache[cache_key] = (serialized, dict(call_info))

    return serialized, call_info


def _smart_serialize(result: dict[str, Any], max_len: int = 12000) -> str:
    """Serialize a result, truncating large list responses intelligently."""
    body = result.get("body", {})

    # For large list responses: trim the values array progressively
    if isinstance(body, dict) and "values" in body:
        values = body.get("values", [])
        if len(values) > 0:
            for keep in [len(values), 200, 100, 50, 20, 10]:
                if keep > len(values):
                    continue
                candidate_body = {**body, "values": values[:keep]}
                if keep < len(values):
                    candidate_body["_truncated"] = True
                    candidate_body["_original_count"] = len(values)
                candidate = json.dumps(
                    {"status_code": result["status_code"], "body": candidate_body},
                    ensure_ascii=False, default=str,
                )
                if len(candidate) <= max_len:
                    return candidate

    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if len(serialized) > max_len:
        serialized = serialized[:max_len] + "... [truncated]"
    return serialized


def _classify_error(status_code: int, body: Any) -> str:
    """Classify a Tripletex error into a category for logging."""
    if status_code == 401:
        return "auth_error"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 422:
        code = body.get("code", 0) if isinstance(body, dict) else 0
        if code == 15000:
            return "validation_error"
        if code == 16000:
            return "mapping_error"
        return "bad_request_422"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return f"server_error_{status_code}"
    return f"client_error_{status_code}"


def _extract_validation_messages(body: Any) -> list[dict[str, str]]:
    """Pull out the validationMessages array from a Tripletex error response."""
    if not isinstance(body, dict):
        return []
    msgs = body.get("validationMessages")
    if isinstance(msgs, list):
        return [{"field": m.get("field", ""), "message": m.get("message", "")} for m in msgs if isinstance(m, dict)]
    return []


_ERROR_RECOVERY_HINTS: dict[str, dict[str, str]] = {
    "validation_error": {
        "userType": "Include userType: \"STANDARD\" in the employee body.",
        "department": "Create a department first with POST /department, then reference its ID.",
        "dateOfBirth": "dateOfBirth is required for PUT /employee. If not in the task, skip the PUT.",
        "deliveryDate": "deliveryDate is required for POST /order. Use today's date if not specified.",
        "invoiceDate": "invoiceDate is required. Use YYYY-MM-DD format.",
        "invoiceDueDate": "invoiceDueDate is required. Use YYYY-MM-DD format.",
        "customer": "customer is required. Create a customer first with POST /customer.",
        "orders": "orders is required for POST /invoice. Create an order first.",
        "firstName": "firstName is required for POST /employee or POST /contact.",
        "lastName": "lastName is required for POST /employee or POST /contact.",
        "name": "name is required. Check the task prompt for the correct name.",
        "employee": "employee is required. Create an employee first with POST /employee.",
        "title": "title is required for POST /travelExpense.",
        "projectManager": "projectManager is required for POST /project. Create an employee first.",
        "amount": "amount is required for voucher postings and must be a number.",
        "account": "account is required for voucher postings. Look up via GET /ledger/account.",
        "postings": "Voucher postings must balance (sum of amounts = 0).",
    },
    "not_found": {
        ":grantEntitlementsByTemplate": (
            "Valid templates: NONE_PRIVILEGES, ALL_PRIVILEGES, INVOICING_MANAGER, "
            "PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER. "
            "Do NOT use ADMINISTRATOR or ACCOUNTANT_ADMINISTRATOR."
        ),
        ":payment": "Use QUERY params: paymentDate, paymentTypeId, paidAmount. No body needed.",
        ":createCreditNote": "Use QUERY params: date (YYYY-MM-DD). No body needed.",
        ":invoice": "Use PUT /order/{orderId}/:invoice with query param invoiceDate.",
        ":deliver": "Use PUT /travelExpense/:deliver with query param id={expenseId}.",
    },
}


def _format_error_for_llm(call_info: dict[str, Any]) -> str | None:
    """Build a concise, structured error message with recovery hints."""
    status = call_info.get("status_code", 0)
    error_type = call_info.get("error_type", "")
    validation = call_info.get("validation_messages", [])
    method = call_info.get("method", "")
    path = call_info.get("path", "")

    parts = [f'{{"status_code": {status}']

    if error_type:
        parts.append(f', "error_type": "{error_type}"')

    if validation:
        field_errors = "; ".join(f"{v['field']}: {v['message']}" for v in validation if v.get("field"))
        if field_errors:
            parts.append(f', "field_errors": "{field_errors}"')

    hints: list[str] = []

    if error_type == "validation_error" and validation:
        field_hints = _ERROR_RECOVERY_HINTS.get("validation_error", {})
        for v in validation:
            field = v.get("field", "")
            for key, hint in field_hints.items():
                if key in field:
                    hints.append(hint)
                    break

    if error_type == "not_found":
        path_hints = _ERROR_RECOVERY_HINTS.get("not_found", {})
        for key, hint in path_hints.items():
            if key in path:
                hints.append(hint)
                break
        if not hints:
            hints.append(f"Check that {method} {path} is a valid endpoint path and all IDs exist.")

    if error_type == "auth_error":
        hints.append("Authentication failed — credentials may be invalid.")
    elif error_type == "rate_limited":
        hints.append("Rate limited — a retry was already attempted. Wait before trying again.")
    elif error_type == "conflict":
        hints.append("Conflict — the entity may already exist or version mismatch. "
                      "Include id and version from the previous response for PUT updates.")
    elif error_type and error_type.startswith("server_error"):
        hints.append("Server error — try the request again. If it persists, skip and continue.")

    if hints:
        combined = " ".join(hints)
        parts.append(f', "recovery_hint": "{combined}"')

    if not validation and not hints:
        return None

    parts.append("}")
    return "".join(parts)
