"""context-runtime memory backend -- a stateless HTTP MemoryManager over the control plane's
narrow ``/memory/*`` routes (bi-temporal fact storage, ``TemporalStore``).

All state lives server-side (context-runtime owns persistence); this backend keeps no queue,
watermark, or cache beyond its own HTTP client, so it is safe for multi-worker Gateway
deployments. Deliberately does NOT import the ``context_runtime`` Python package -- see
``client.py``'s docstring for why.

Scope, honestly: the underlying ``TemporalStore`` is purely additive/bi-temporal (it never
deletes a fact, by design -- that's the whole point of "what did we believe at time X"). This
backend therefore does not implement real deletion: ``clear_memory``/``delete_memory`` are not
overridden and inherit the base class's ``NotImplementedError`` (same "dead contract, zero
callers" status the base class documents for every backend). A bucket maps to one namespace,
keyed from whichever of (user_id, agent_name, thread_id) is available -- narrower than mem0's
combinable filter predicates, by design (this is a thin pass-through, not a redesign of
TemporalStore's own query model).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import PrivateAttr

# ABC contract -- the ONE allowed `from deerflow` import in this backend folder.
from deerflow.agents.memory.manager import MemoryManager, MemoryManagerError, MemoryReadError

from .client import ContextRuntimeAPIError, ContextRuntimeClient
from .config import ContextRuntimeConfig
from .message_filtering import extract_message_text, filter_messages_for_memory

logger = logging.getLogger(__name__)

_ROLE_MAP = {"human": "user", "ai": "assistant"}


def _namespace(prefix: str, *, user_id: str | None, agent_name: str | None) -> str:
    """One namespace per (user_id, agent_name) bucket -- deliberately NOT keyed on thread_id:
    add() always has a thread_id but get_context()/search() often don't (thread_id is optional
    on both and absent from search()'s signature entirely), so folding it into the namespace key
    would put writes and reads in different namespaces. thread_id instead becomes each fact's
    *subject* (see add()) -- a real bug found and fixed via the e2e test, not a theoretical one:
    the first version of this function kept writes and reads from ever meeting."""
    parts = [p for p in (user_id, agent_name) if p]
    return f"{prefix}:{'|'.join(parts)}" if parts else prefix


def _to_fact(hit: dict[str, Any]) -> dict[str, Any]:
    """Map a /memory/fact Hit (meta carries subject/predicate/object/valid_from/.../recorded_at)
    to the backend-neutral fact shape consumed by the host (agents/memory/tools.py)."""
    meta = hit.get("meta") or {}
    return {
        "id": str(hit.get("chunk_id", "")),
        "content": str(hit.get("text", "")),
        "category": str(meta.get("role", "context")),
        "confidence": float(hit.get("score") or 0.0),
        "createdAt": str(meta.get("recorded_at") or hit.get("created_at") or ""),
        "source": str(meta.get("source", "deerflow")),
    }


class ContextRuntimeManager(MemoryManager):
    """MemoryManager backed by context-runtime's control-plane ``/memory/*`` HTTP routes."""

    supports_search: ClassVar[bool] = True
    requires_passive_writes_in_tool_mode: ClassVar[bool] = True

    _config: ContextRuntimeConfig = PrivateAttr()
    _client: Any = PrivateAttr(default=None)  # ContextRuntimeClient; tests inject a fake

    def model_post_init(self, __context: Any) -> None:
        self._config = ContextRuntimeConfig.from_backend_config(self.backend_config)
        self._client = ContextRuntimeClient(
            base_url=self._config.base_url,
            api_key=self._config.resolve_api_key(),
            timeout_seconds=self._config.timeout_seconds,
        )

    @classmethod
    def from_config(
        cls,
        backend_config: dict[str, Any] | None = None,
        *,
        mode: Literal["middleware", "tool"] = "middleware",
        **host_hooks: Any,
    ) -> ContextRuntimeManager:
        mgr = cls(backend_config=backend_config, mode=mode)
        return mgr

    def close(self) -> None:
        self._client.close()

    @classmethod
    def read_failures_are_fatal_for_config(cls, backend_config: dict[str, Any] | None) -> bool:
        return ContextRuntimeConfig.from_backend_config(backend_config).read_policy == "fail_closed"

    # ── Error policies ───────────────────────────────────────────────────
    def _read_or_fallback(self, fallback: Any, fn: Any) -> Any:
        try:
            return fn()
        except ContextRuntimeAPIError as e:
            if self._config.read_policy == "fail_open":
                logger.warning("context-runtime read failed (%s); continuing without memory", e)
                return fallback
            raise MemoryReadError(f"context-runtime read failed: {e}") from e

    def _write_or_drop(self, fn: Any) -> None:
        try:
            fn()
        except ContextRuntimeAPIError as e:
            if self._config.write_policy == "log_and_drop":
                logger.warning("context-runtime write failed (%s); dropping update", e)
                return
            raise MemoryManagerError(f"context-runtime write failed: {e}") from e

    # ── Tier 1: write ────────────────────────────────────────────────────
    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Record each kept message as one bi-temporal fact: subject=thread_id,
        predicate=role, object=message text, recorded_at=now, provenance in meta. No
        server-side extraction happens (unlike mem0) -- this is a raw, provenance-tracked
        journal of the conversation, not a fact-extraction pipeline."""
        kept = filter_messages_for_memory(messages)
        ns = _namespace(self._config.namespace_prefix, user_id=user_id, agent_name=agent_name)
        now = datetime.now(UTC).isoformat()
        for m in kept:
            role = _ROLE_MAP.get(getattr(m, "type", ""))
            if role is None:
                continue
            text = extract_message_text(m).strip()
            if not text:
                continue
            self._write_or_drop(
                lambda role=role, text=text: self._client.write_fact(
                    namespace=ns,
                    subject=thread_id,
                    predicate=role,
                    obj=text,
                    valid_from=now,
                    recorded_at=now,
                    meta={"role": role, "agent_name": agent_name, "user_id": user_id, "trace_id": trace_id, "source": "deerflow"},
                )
            )

    async def aadd(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.add,
            thread_id,
            messages,
            agent_name=agent_name,
            user_id=user_id,
            trace_id=trace_id,
        )

    # ── Tier 1: read-inject ──────────────────────────────────────────────
    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        """Query-less recall: an empty query ranks ties by most-recently-valid first (see
        TemporalStore._ranked), so this returns the bucket's most recent facts, up to top_k."""
        ns = _namespace(self._config.namespace_prefix, user_id=user_id, agent_name=agent_name)
        hits = self._read_or_fallback([], lambda: self._client.read_facts(namespace=ns, query="", k=self._config.top_k))
        lines = [f"- {h.get('text', '')}" for h in hits if h.get("text")]
        return "\n".join(lines)

    async def aget_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        return await asyncio.to_thread(self.get_context, user_id, agent_name=agent_name, thread_id=thread_id)

    # ── Tier 2: search ───────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        ns = _namespace(self._config.namespace_prefix, user_id=user_id, agent_name=agent_name)
        hits = self._read_or_fallback([], lambda: self._client.read_facts(namespace=ns, query=query, k=top_k))
        facts = [_to_fact(h) for h in hits]
        if category:
            facts = [f for f in facts if f["category"] == category]
        return facts[:top_k]

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.search,
            query,
            top_k,
            user_id=user_id,
            agent_name=agent_name,
            category=category,
        )

    # ── Tier 2: management (read-only; see module docstring re: deletion) ──
    def get_memory(
        self,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        ns = _namespace(self._config.namespace_prefix, user_id=user_id, agent_name=agent_name)
        hits = self._read_or_fallback([], lambda: self._client.read_facts(namespace=ns, query="", k=1000))
        return {"facts": [_to_fact(h) for h in hits]}
