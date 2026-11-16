"""Markdown-memory backend -- a git-diffable, human-readable :class:`MemoryManager`.

Motivation: a plain, git-backed Markdown store is auditable in a way an opaque
vector/graph store is not -- a human (or another process) can open the file
directly, read it, ``git diff`` a single changed fact, and ``git blame`` who/when
it was written, without going through this backend at all. This does not claim
to reproduce any specific external benchmark; it makes that auditability
property available as a selectable backend, following the pluggable-backend
contract documented in ``manager.py`` and demonstrated by ``noop``.

Portability golden rule (see ``noop/config.py`` for the full version): the ONLY
``from deerflow`` import in this folder is the ABC contract line below.
Everything else (storage root, caps) arrives via ``backend_config``.

Storage layout (one fact per line -> minimal, reviewable diffs on every write):

    {storage_path}/users/{user_id or "_global"}/agents/{agent_name or "_global"}/
        memory.md     -- one bullet line per fact: `- fact-<id> [category] (confidence=0.NN) <iso8601> :: content`
        threads/{thread_id}.md  -- append-only raw conversation log (one dated
                                    block per add() call; never rewritten)

``add()``/``add_nowait()`` record the raw conversation (no LLM extraction inside
this backend -- see the module-level note below); ``create_fact`` / `update_fact`
/ `delete_fact` are the structured, atomic, one-line-per-fact store, which is
where the auditability property is sharpest.
"""

from __future__ import annotations

import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import PrivateAttr

# ABC contract -- the ONE allowed `from deerflow` in this backend folder.
from deerflow.agents.memory.manager import MemoryManager

from .config import MarkdownMemoryConfig

_FACT_LINE_RE = re.compile(
    r"^- fact-(?P<id>[0-9a-f]{12}) \[(?P<category>[^\]]*)\] \(confidence=(?P<confidence>[0-9.]+)\) (?P<created>\S+) :: (?P<content>.*)$"
)

_FACTS_HEADER = (
    "# Memory facts\n\n"
    "<!-- deerflow markdown-memory backend v1. One fact per line, so a `git diff`\n"
    "     on this file shows exactly which facts changed. Safe to hand-edit: keep\n"
    "     the `- fact-<id> [category] (confidence=0.NN) <iso8601> :: content` shape. -->\n\n"
)


def _slug(value: str | None, *, default: str = "_global") -> str:
    """Filesystem-safe bucket name; never empty, never path-traversing."""
    if not value:
        return default
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._") or default
    return cleaned


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class MarkdownMemoryManager(MemoryManager):
    """Markdown/git-backed :class:`MemoryManager` -- facts as diffable lines.

    ``model_post_init`` parses ``backend_config`` into a
    :class:`MarkdownMemoryConfig`; ``self._lock`` guards read-modify-write of a
    bucket's facts file against concurrent calls **within this process** (this
    backend does not do cross-process file locking -- acceptable for a
    single-process research/dev deployment, a real limitation for a multi-worker
    one; see the backend's PR description).
    """

    _config: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default=None)

    # Overrides search() below -> required to be True (checked by the base
    # class's _check_invariants).
    supports_search: ClassVar[bool] = True

    def model_post_init(self, __context: Any) -> None:
        self._config = MarkdownMemoryConfig.from_backend_config(self.backend_config)
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        backend_config: dict[str, Any] | None = None,
        *,
        mode: Literal["middleware", "tool"] = "middleware",
        **host_hooks: Any,
    ) -> MarkdownMemoryManager:
        """No external deps to wire; host_hooks (tracing, llm factory, ...) are
        unused -- this backend never calls an LLM itself."""
        return cls(backend_config=backend_config, mode=mode)

    # ── Paths ────────────────────────────────────────────────────────────
    def _bucket_dir(self, *, user_id: str | None, agent_name: str | None) -> Path:
        root = Path(self._config.storage_path or ".") / "users" / _slug(user_id) / "agents" / _slug(agent_name)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _facts_path(self, *, user_id: str | None, agent_name: str | None) -> Path:
        return self._bucket_dir(user_id=user_id, agent_name=agent_name) / "memory.md"

    def _thread_log_path(self, *, user_id: str | None, agent_name: str | None, thread_id: str) -> Path:
        d = self._bucket_dir(user_id=user_id, agent_name=agent_name) / "threads"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{_slug(thread_id, default='_unknown')}.md"

    # ── Fact-file parse/render (one line per fact -> minimal diffs) ────────
    def _read_facts(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        facts: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _FACT_LINE_RE.match(line)
            if m:
                facts.append(
                    {
                        "id": f"fact-{m.group('id')}",
                        "category": m.group("category"),
                        "confidence": float(m.group("confidence")),
                        "created": m.group("created"),
                        "content": m.group("content"),
                    }
                )
        return facts

    def _write_facts(self, path: Path, facts: list[dict[str, Any]]) -> None:
        lines = [_FACTS_HEADER.rstrip("\n")]
        for f in facts:
            fid = f["id"].removeprefix("fact-")
            lines.append(f"- fact-{fid} [{f['category']}] (confidence={float(f['confidence']):.2f}) {f['created']} :: {f['content']}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _memory_doc(self, facts: list[dict[str, Any]], *, user_id: str | None) -> dict[str, Any]:
        return {"version": 1, "lastUpdated": _now_iso(), "user": user_id, "facts": facts}

    # ── Write (tier 1) ───────────────────────────────────────────────────
    def add(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Append the turn's user+assistant text to the thread's log file.

        No LLM-based fact extraction happens here (this backend has no LLM
        dependency at all -- see the module docstring); callers that want
        structured facts call ``create_fact`` (directly, or via the
        ``mode='tool'`` memory_add tool) with already-decided content. This
        keeps the raw transcript too, since even an un-extracted record is
        real, auditable memory.
        """
        turns: list[str] = []
        for m in messages:
            kind = getattr(m, "type", None)
            content = getattr(m, "content", None)
            if kind not in ("human", "ai") or not content:
                continue
            additional_kwargs = getattr(m, "additional_kwargs", None) or {}
            if additional_kwargs.get("hide_from_ui"):
                keep = self._config.should_keep_hidden_message
                if not (keep and keep(additional_kwargs)):
                    continue
            text = content if isinstance(content, str) else str(content)
            turns.append(f"**{'User' if kind == 'human' else 'Assistant'}:** {text}")
        if not turns:
            return
        path = self._thread_log_path(user_id=user_id, agent_name=agent_name, thread_id=thread_id)
        block = f"\n## Turn {_now_iso()}" + (f" (trace {trace_id})" if trace_id else "") + "\n\n" + "\n\n".join(turns) + "\n"
        with self._lock:
            is_new = not path.exists()
            with path.open("a", encoding="utf-8") as fh:
                if is_new:
                    fh.write(f"# Conversation log: thread {thread_id}\n")
                fh.write(block)

    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> None:
        # Synchronous writes, no debounce queue -> "immediate" is identical to "add".
        self.add(thread_id, messages, agent_name=agent_name, user_id=user_id)

    # ── Read (tier 1) ────────────────────────────────────────────────────
    def get_context(
        self,
        user_id: str | None,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        with self._lock:
            facts = self._read_facts(path)
        if not facts:
            return ""
        lines = [f"- [{f['category']}] {f['content']}" for f in facts]
        return "\n".join(lines)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Case-insensitive substring match over fact content, newest first.

        No embeddings/ranking model -- this backend has no vector index by
        design; a caller that needs semantic search should pick a different
        backend. ``category`` filters before the ``top_k`` slice (matches the
        contract in ``manager.py``).
        """
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        with self._lock:
            facts = self._read_facts(path)
        if category:
            facts = [f for f in facts if f["category"] == category]
        needle = query.strip().lower()
        if needle:
            facts = [f for f in facts if needle in f["content"].lower()]
        facts.sort(key=lambda f: f["created"], reverse=True)
        return facts[: max(0, top_k)]

    # ── Manage (tier 2) ──────────────────────────────────────────────────
    def get_memory(self, *, user_id: str | None = None, agent_name: str | None = None) -> dict[str, Any]:
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        with self._lock:
            facts = self._read_facts(path)
        return self._memory_doc(facts, user_id=user_id)

    def clear_memory(self, *, user_id: str | None = None, agent_name: str | None = None) -> dict[str, Any]:
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        with self._lock:
            self._write_facts(path, [])
        return self._memory_doc([], user_id=user_id)

    def import_memory(
        self,
        memory_data: dict[str, Any],
        *,
        user_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        incoming = memory_data.get("facts") or []
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        with self._lock:
            facts = self._read_facts(path)
            existing_ids = {f["id"] for f in facts}
            for raw in incoming:
                fid = raw.get("id") or f"fact-{uuid.uuid4().hex[:12]}"
                if fid in existing_ids:
                    continue
                facts.append(
                    {
                        "id": fid,
                        "category": raw.get("category", "context"),
                        "confidence": float(raw.get("confidence", 0.5)),
                        "created": raw.get("created") or _now_iso(),
                        "content": str(raw.get("content", "")),
                    }
                )
                existing_ids.add(fid)
            facts = self._apply_cap(facts)
            self._write_facts(path, facts)
        return self._memory_doc(facts, user_id=user_id)

    # ── Lifecycle ───────────────────────────────────────────────────────
    def shutdown_flush(self, timeout: float) -> bool:
        # Every write above is synchronous (no debounce buffer) -> nothing to drain.
        return True

    # ── Tier 3: fact CRUD (the sharpest expression of the diffability goal) ─
    def _apply_cap(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cap = self._config.max_facts
        if cap and len(facts) > cap:
            facts = sorted(facts, key=lambda f: f["created"])[-cap:]
        return facts

    def create_fact(
        self,
        content: str,
        category: str = "context",
        confidence: float = 0.5,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        fact_id = f"fact-{uuid.uuid4().hex[:12]}"
        new_fact = {"id": fact_id, "category": category, "confidence": float(confidence), "created": _now_iso(), "content": content}
        with self._lock:
            facts = self._read_facts(path)
            facts.append(new_fact)
            before_cap = {f["id"] for f in facts}
            facts = self._apply_cap(facts)
            evicted = fact_id not in {f["id"] for f in facts} if fact_id in before_cap else False
            self._write_facts(path, facts)
        return self._memory_doc(facts, user_id=user_id), (None if evicted else fact_id)

    def delete_fact(self, fact_id: str, *, agent_name: str | None = None, user_id: str | None = None) -> dict[str, Any]:
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        with self._lock:
            facts = [f for f in self._read_facts(path) if f["id"] != fact_id]
            self._write_facts(path, facts)
        return self._memory_doc(facts, user_id=user_id)

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        path = self._facts_path(user_id=user_id, agent_name=agent_name)
        with self._lock:
            facts = self._read_facts(path)
            for f in facts:
                if f["id"] == fact_id:
                    if content is not None:
                        f["content"] = content
                    if category is not None:
                        f["category"] = category
                    if confidence is not None:
                        f["confidence"] = float(confidence)
                    break
            self._write_facts(path, facts)
        return self._memory_doc(facts, user_id=user_id)
