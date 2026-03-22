import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("tripletex-api")

MAX_RESPONSE_CHARS = 8000

IMPORTANT_KEYS = {"id", "version", "name", "amount", "amountOutstanding", "amountCurrency",
                  "invoiceNumber", "number", "firstName", "lastName", "email",
                  "value", "fullResultSize", "status", "description", "error", "message",
                  "userType", "isCustomer", "isSupplier", "organizationNumber",
                  "accountNumber", "title", "date", "invoiceDate", "dueDate"}


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
        )

    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        method = method.upper()

        if isinstance(json_body, str):
            try:
                json_body = json.loads(json_body)
            except (json.JSONDecodeError, TypeError):
                pass

        logger.info("%s %s params=%s", method, path, params)
        if json_body:
            logger.info("Body: %s", json.dumps(json_body, ensure_ascii=False)[:1000])

        resp = await self._do_request(method, url, params, json_body)
        if resp is None:
            return {"error": "Request failed after retry", "status_code": 500}

        logger.info("Response: %d (%d bytes)", resp.status_code, len(resp.content))

        body_text = resp.text.strip()
        if not body_text:
            data = {"message": "OK (empty response)"}
        else:
            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": body_text[:2000]}

        result = {"status_code": resp.status_code, "data": data}

        serialized = json.dumps(result, ensure_ascii=False)
        if len(serialized) > MAX_RESPONSE_CHARS:
            result["data"] = _truncate(data, MAX_RESPONSE_CHARS)

        return result

    async def _do_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        retries: int = 1,
    ) -> httpx.Response | None:
        # Skip retries entirely for endpoints known to always fail
        if method != "GET" and any(ep in url for ep in ("/supplierInvoice", "/bank")):
            retries = 0
        for attempt in range(retries + 1):
            try:
                resp = await self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body if json_body else None,
                    auth=self.auth,
                )
                if resp.status_code >= 500 and attempt < retries:
                    wait = 1 + attempt
                    logger.warning("Server error %d, retrying in %ds...", resp.status_code, wait)
                    await asyncio.sleep(wait)
                    continue
                return resp
            except httpx.TimeoutException:
                if attempt < retries:
                    wait = 2 + attempt
                    logger.warning("Timeout, retrying in %ds...", wait)
                    await asyncio.sleep(wait)
                    continue
                return _error_response(408, "Request timed out")
            except Exception as e:
                if attempt < retries:
                    wait = 2 + attempt
                    logger.warning("Error %s, retrying in %ds...", e, wait)
                    await asyncio.sleep(wait)
                    continue
                return _error_response(500, str(e)[:200])
        return None

    async def close(self):
        await self._client.aclose()


class _FakeResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self.content = json.dumps(data).encode()
        self.text = json.dumps(data)

    def json(self):
        return json.loads(self.text)


def _error_response(status: int, msg: str) -> _FakeResponse:
    return _FakeResponse(status, {"error": msg})


def _truncate(data: Any, max_chars: int) -> Any:
    if isinstance(data, dict):
        if "values" in data and isinstance(data["values"], list):
            values = data["values"]
            for limit in [20, 10, 5, 2]:
                truncated = values[:limit]
                candidate = {**data, "values": truncated}
                if len(values) > limit:
                    candidate["_note"] = f"Showing {limit} of {len(values)}"
                s = json.dumps(candidate, ensure_ascii=False)
                if len(s) <= max_chars:
                    return candidate
            return {
                "fullResultSize": data.get("fullResultSize", len(values)),
                "values": values[:1],
                "_note": f"Heavily truncated, {len(values)} total",
            }

        if "value" in data and isinstance(data["value"], dict):
            inner = data["value"]
            slim = {k: v for k, v in inner.items() if k in IMPORTANT_KEYS or not isinstance(v, (dict, list))}
            candidate = {**data, "value": slim}
            s = json.dumps(candidate, ensure_ascii=False)
            if len(s) <= max_chars:
                return candidate

        s = json.dumps(data, ensure_ascii=False)
        if len(s) <= max_chars:
            return data

        slim = {}
        for k, v in data.items():
            if k in IMPORTANT_KEYS:
                slim[k] = v
            elif not isinstance(v, (dict, list)):
                slim[k] = v
        slim["_truncated"] = True
        s = json.dumps(slim, ensure_ascii=False)
        if len(s) <= max_chars:
            return slim

        return {"_truncated": True, "keys": list(data.keys())[:15]}
    return data
