"""FastAPI application exposing the /solve endpoint for the Tripletex competition."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from agent import run_agent
from config import API_KEY, LOG_LEVEL
from file_handler import process_files
from llm import LLMClient
from tripletex import TripletexClient

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Accounting Agent")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve")
async def solve(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    if API_KEY:
        expected = f"Bearer {API_KEY}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    body: dict[str, Any] = await request.json()
    prompt: str = body.get("prompt", "")
    files: list[dict] = body.get("files", [])
    creds: dict = body.get("tripletex_credentials", {})
    task_id: str = body.get("task_id", body.get("taskId", body.get("task_type", "")))

    base_url: str = creds.get("base_url", "")
    session_token: str = creds.get("session_token", "")

    if not prompt or not base_url or not session_token:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: prompt, tripletex_credentials.base_url, tripletex_credentials.session_token",
        )

    # Log the FULL prompt and all body keys to file for debugging
    import os
    log_dir = "/tmp/task_logs"
    os.makedirs(log_dir, exist_ok=True)
    body_keys = [k for k in body.keys() if k != "tripletex_credentials"]
    with open(f"{log_dir}/tasks.jsonl", "a") as fh:
        fh.write(json.dumps({
            "task_id": task_id,
            "body_keys": body_keys,
            "files": [f.get("filename") for f in files],
            "prompt": prompt,
        }, ensure_ascii=False) + "\n")

    logger.info("Received task_id=%r keys=%s prompt=%.400s", task_id, body_keys, prompt)
    request_start = time.time()

    extracted_files = process_files(files)
    file_names = [f.get("filename", "?") for f in files] if files else []

    tripletex = TripletexClient(base_url, session_token)
    llm = LLMClient()
    summary: dict[str, Any] = {}

    try:
        summary = await run_agent(prompt, extracted_files, tripletex, llm)
    except Exception:
        logger.exception("Agent execution failed")
        summary = {"exit_reason": "exception", "total_api_calls": 0, "error_count": 0}
    finally:
        await tripletex.close()

    _log_submission(prompt, file_names, summary, request_start)

    return JSONResponse({"status": "completed"})


def _log_submission(
    prompt: str,
    file_names: list[str],
    summary: dict[str, Any],
    request_start: float,
) -> None:
    """Emit a structured JSON log line with the full submission summary."""
    api_calls = summary.get("api_calls", [])
    call_summary = [
        {
            "method": c.get("method", "?"),
            "path": c.get("path", "?"),
            "status": c.get("status_code", 0),
            **({"error_type": c["error_type"]} if c.get("is_error") else {}),
            **({"validation": c["validation_messages"]} if c.get("validation_messages") else {}),
        }
        for c in api_calls
    ]

    log_entry = {
        "event": "submission_complete",
        "prompt_preview": prompt[:300],
        "files": file_names,
        "total_api_calls": summary.get("total_api_calls", 0),
        "error_count": summary.get("error_count", 0),
        "exit_reason": summary.get("exit_reason", "unknown"),
        "elapsed_seconds": round(time.time() - request_start, 2),
        "api_calls": call_summary,
    }

    logger.info("SUBMISSION_SUMMARY %s", json.dumps(log_entry, ensure_ascii=False, default=str))
