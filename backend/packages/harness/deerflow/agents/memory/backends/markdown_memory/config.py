"""Markdown-memory backend config -- parses ``backend_config``.

See ``noop/config.py`` for the portability golden rule this mirrors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class MarkdownMemoryConfig:
    """Parsed config for the markdown-memory backend.

    Attributes:
        storage_path: Writable state dir, host-injected (or user-set). Facts
            and thread logs are written under here as plain ``.md`` files, one
            fact per line, so ``git diff`` / ``git log`` / ``git blame`` on
            this directory are directly meaningful.
        max_facts: Soft cap per (user, agent) bucket; oldest fact is evicted
            (by created-time) when exceeded. ``0`` = unbounded.
        should_keep_hidden_message: Host-injected hook; ``None`` = skip all
            ``hide_from_ui`` messages when logging a thread's conversation.
    """

    storage_path: str = ""
    max_facts: int = 500
    should_keep_hidden_message: Callable[[Any], bool] | None = None

    @classmethod
    def from_backend_config(cls, backend_config: dict[str, Any] | None) -> MarkdownMemoryConfig:
        cfg = dict(backend_config or {})
        return cls(
            storage_path=str(cfg.get("storage_path") or ""),
            max_facts=int(cfg.get("max_facts", 500) or 0),
            should_keep_hidden_message=cfg.get("should_keep_hidden_message"),
        )
