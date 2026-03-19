"""Core agent loop: LLM tool-calling cycle against the Tripletex API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from config import AGENT_MAX_ITERATIONS, AGENT_TIMEOUT_SECONDS
from file_handler import ExtractedFile, build_file_context
from llm import LLMClient, LLMResponse
from prompts import SYSTEM_PROMPT
from tools import TOOLS
from tripletex import TripletexClient

logger = logging.getLogger(__name__)


async def run_agent(
    prompt: str,
    extracted_files: list[ExtractedFile],
    tripletex: TripletexClient,
    llm: LLMClient,
) -> dict[str, Any]:
    """Execute the agent loop until the LLM decides the task is complete.

    Returns a summary dict for structured logging.
    """
    start = time.monotonic()
    deadline = start + AGENT_TIMEOUT_SECONDS

    messages = _build_initial_messages(prompt, extracted_files, llm)
    api_log: list[dict[str, Any]] = []
    error_count = 0
    call_count = 0

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

        for tc in response.tool_calls:
            if time.monotonic() > deadline:
                logger.warning("Deadline reached during tool execution")
                return _build_summary(prompt, call_count, error_count, start, api_log, "timeout")

            result_str, call_info = await _execute_tool(tc.name, tc.arguments, tripletex)
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
) -> list[dict[str, Any]]:
    """Construct the initial message list with system prompt, file context, and task."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    user_parts: list[Any] = []
    file_context = build_file_context(extracted_files)

    text_content = f"# Task\n\n{prompt}"
    if file_context:
        text_content += f"\n\n{file_context}"
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
    name: str, arguments: dict[str, Any], tripletex: TripletexClient
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

    try:
        resp = await tripletex.request(method, path, params=params, json_body=json_body)
    except Exception as exc:
        error_result = json.dumps({"error": f"Request failed: {exc}"})
        call_info.update(status_code=0, is_error=True, error_type="request_exception")
        return error_result, call_info

    call_info["status_code"] = resp.status_code
    call_info["is_error"] = 400 <= resp.status_code < 500

    if call_info["is_error"]:
        call_info["error_type"] = _classify_error(resp.status_code, resp.body)
        call_info["validation_messages"] = _extract_validation_messages(resp.body)

    result = {"status_code": resp.status_code, "body": resp.body}
    serialized = json.dumps(result, ensure_ascii=False, default=str)

    max_len = 12000
    if len(serialized) > max_len:
        serialized = serialized[:max_len] + "... [truncated]"

    return serialized, call_info


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
    return f"client_error_{status_code}"


def _extract_validation_messages(body: Any) -> list[dict[str, str]]:
    """Pull out the validationMessages array from a Tripletex error response."""
    if not isinstance(body, dict):
        return []
    msgs = body.get("validationMessages")
    if isinstance(msgs, list):
        return [{"field": m.get("field", ""), "message": m.get("message", "")} for m in msgs if isinstance(m, dict)]
    return []


def _format_error_for_llm(call_info: dict[str, Any]) -> str | None:
    """Build a concise, structured error message for the LLM to reason about.

    Returns None if no special formatting is needed (use raw result).
    """
    status = call_info.get("status_code", 0)
    error_type = call_info.get("error_type", "")
    validation = call_info.get("validation_messages", [])

    if not validation and error_type not in ("not_found", "auth_error", "rate_limited"):
        return None

    parts = [f'{{"status_code": {status}']

    if error_type:
        parts.append(f', "error_type": "{error_type}"')

    if validation:
        field_errors = "; ".join(f"{v['field']}: {v['message']}" for v in validation if v.get("field"))
        if field_errors:
            parts.append(f', "field_errors": "{field_errors}"')

    if error_type == "not_found":
        method = call_info.get("method", "")
        path = call_info.get("path", "")
        parts.append(f', "hint": "Check that {method} {path} is a valid endpoint path"')
    elif error_type == "auth_error":
        parts.append(', "hint": "Authentication failed - verify credentials are correct"')
    elif error_type == "rate_limited":
        parts.append(', "hint": "Rate limited - wait before retrying"')

    parts.append("}")
    return "".join(parts)
