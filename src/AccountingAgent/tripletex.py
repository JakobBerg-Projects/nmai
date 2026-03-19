from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from config import TRIPLETEX_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripletexResponse:
    status_code: int
    body: Any
    ok: bool


class TripletexClient:
    """Async wrapper around the Tripletex v2 REST API accessed via proxy."""

    def __init__(self, base_url: str, session_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = ("0", session_token)
        self._client = httpx.AsyncClient(
            auth=self._auth,
            timeout=httpx.Timeout(TRIPLETEX_REQUEST_TIMEOUT),
        )

    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> TripletexResponse:
        url = f"{self._base_url}{path}"
        method = method.upper()

        logger.info("Tripletex %s %s params=%s body=%s", method, path, params,
                    json.dumps(json_body, ensure_ascii=False) if json_body else None)

        response = await self._client.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
        )

        try:
            body = response.json()
        except Exception:
            body = response.text

        ok = 200 <= response.status_code < 300

        if not ok:
            logger.warning(
                "Tripletex %s %s -> %d: %s",
                method, path, response.status_code, body,
            )
        else:
            logger.info("Tripletex %s %s -> %d", method, path, response.status_code)

        return TripletexResponse(
            status_code=response.status_code,
            body=body,
            ok=ok,
        )

    async def close(self) -> None:
        await self._client.aclose()
