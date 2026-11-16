"""Synchronous httpx client for context-runtime's narrow ``/memory/*`` control-plane routes.

Deliberately NOT a Python import of the ``context_runtime`` package -- this backend reaches the
bi-temporal fact store purely over HTTP, mirroring how the existing ``mem0`` backend avoids a
hard dependency on the ``mem0ai`` package. The MemoryManager contract is synchronous, so this is
a plain ``httpx.Client`` (constructed with an optional ``transport`` so tests can inject
``httpx.MockTransport``).
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class ContextRuntimeAPIError(RuntimeError):
    """Any context-runtime request failure (transport, 4xx/5xx)."""


class ContextRuntimeAuthError(ContextRuntimeAPIError):
    """401 -- missing or invalid API key."""


class ContextRuntimeClient:
    """Thin wrapper over the three ``/memory/*`` routes DeerFlow uses."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise ContextRuntimeAPIError(f"context-runtime request failed: {e}") from e
        if resp.status_code == 401:
            raise ContextRuntimeAuthError("context-runtime authentication failed (check the API key)")
        if resp.status_code >= 400:
            raise ContextRuntimeAPIError(f"context-runtime {method} {path} -> {resp.status_code}: {resp.text[:200]}")
        if not resp.content:
            return None
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise ContextRuntimeAPIError(f"context-runtime {method} {path} returned malformed JSON: {e}") from e

    def write_fact(
        self,
        *,
        namespace: str,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = "",
        valid_to: str | None = None,
        recorded_at: str = "",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "namespace": namespace,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "recorded_at": recorded_at,
            "meta": meta or {},
        }
        return self._request("POST", "/memory/fact", json=body)

    def read_facts(
        self,
        *,
        namespace: str,
        query: str = "",
        k: int = 5,
        at: str | None = None,
        known_at: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"namespace": namespace, "query": query, "k": k}
        if at is not None:
            params["at"] = at
        if known_at is not None:
            params["known_at"] = known_at
        return self._request("GET", "/memory/fact", params=params) or []

    def changes(
        self,
        *,
        namespace: str,
        query: str = "",
        since: str = "",
        until: str = "",
        k: int = 20,
    ) -> list[dict[str, Any]]:
        params = {"namespace": namespace, "query": query, "since": since, "until": until, "k": k}
        return self._request("GET", "/memory/changes", params=params) or []
