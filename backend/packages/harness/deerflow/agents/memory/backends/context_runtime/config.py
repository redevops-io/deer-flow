"""context-runtime backend config -- parses and validates ``backend_config``.

Follows the mem0/noop-template pattern: a plain frozen dataclass +
``from_backend_config``. The host injects ``storage_path`` (and optionally
``should_keep_hidden_message``) into every backend's config dict; those keys
are accepted and ignored. Any OTHER unknown key is rejected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from typing import Any
from urllib.parse import urlsplit

#: Keys the host factory injects into backend_config; accepted and ignored.
_HOST_INJECTED_KEYS = frozenset({"storage_path", "should_keep_hidden_message"})

_READ_POLICIES = frozenset({"fail_open", "fail_closed"})
_WRITE_POLICIES = frozenset({"log_and_drop", "raise"})


@dataclass(frozen=True)
class ContextRuntimeConfig:
    """Validated knobs for the context-runtime HTTP memory backend."""

    #: Name of the environment variable holding the control-plane API key.
    #: The key itself never appears in config.yaml.
    api_key_env: str = "CONTEXT_RUNTIME_API_KEY"
    #: context-runtime control-plane root (``uvicorn ...control_plane.app:app``).
    base_url: str = "http://127.0.0.1:8080"
    #: Permit sending the API key over plaintext HTTP. Intended only for
    #: trusted local development networks (e.g. same-host or same-LAN).
    allow_insecure_http: bool = True
    #: Which namespace (one isolated TemporalStore) this bucket's facts live
    #: in; see context_runtime_manager.py for how a bucket maps to one.
    namespace_prefix: str = "deerflow"
    #: Max facts injected by get_context / default search breadth.
    top_k: int = 8
    #: Per-request HTTP timeout in seconds.
    timeout_seconds: float = 10.0
    #: "fail_open" = recall errors inject nothing and continue;
    #: "fail_closed" = recall errors raise MemoryManagerError.
    read_policy: str = "fail_open"
    #: "log_and_drop" = write errors are logged and dropped (at-most-once);
    #: "raise" = write errors raise MemoryManagerError.
    write_policy: str = "log_and_drop"

    @classmethod
    def from_backend_config(cls, backend_config: dict[str, Any] | None) -> ContextRuntimeConfig:
        cfg = dict(backend_config or {})
        failure_policy = cfg.pop("failure_policy", {}) or {}
        unknown = (
            set(cfg)
            - {
                "api_key_env",
                "base_url",
                "allow_insecure_http",
                "namespace_prefix",
                "top_k",
                "timeout_seconds",
            }
            - _HOST_INJECTED_KEYS
        )
        if unknown:
            raise ValueError(f"context_runtime backend_config has unknown keys: {sorted(unknown)}")
        if not isinstance(failure_policy, dict):
            raise ValueError("context_runtime failure_policy must be a mapping {read, write}")
        unknown_fp = set(failure_policy) - {"read", "write"}
        if unknown_fp:
            raise ValueError(f"context_runtime failure_policy has unknown keys: {sorted(unknown_fp)}")
        allow_insecure_http = cfg.get("allow_insecure_http", True)
        if not isinstance(allow_insecure_http, bool):
            raise ValueError("context_runtime allow_insecure_http must be a boolean")

        config = cls(
            api_key_env=str(cfg.get("api_key_env", "CONTEXT_RUNTIME_API_KEY")),
            base_url=str(cfg.get("base_url", "http://127.0.0.1:8080")).rstrip("/"),
            allow_insecure_http=allow_insecure_http,
            namespace_prefix=str(cfg.get("namespace_prefix", "deerflow")),
            top_k=int(cfg.get("top_k", 8)),
            timeout_seconds=float(cfg.get("timeout_seconds", 10.0)),
            read_policy=str(failure_policy.get("read", "fail_open")),
            write_policy=str(failure_policy.get("write", "log_and_drop")),
        )
        if config.read_policy not in _READ_POLICIES:
            raise ValueError(f"context_runtime failure_policy.read must be one of {sorted(_READ_POLICIES)}")
        if config.write_policy not in _WRITE_POLICIES:
            raise ValueError(f"context_runtime failure_policy.write must be one of {sorted(_WRITE_POLICIES)}")
        if not 1 <= config.top_k <= 1000:
            raise ValueError("context_runtime top_k must be in [1, 1000]")
        if not isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
            raise ValueError("context_runtime timeout_seconds must be a finite value > 0")
        if not config.api_key_env.strip():
            raise ValueError("context_runtime api_key_env must be a non-empty env var name")
        if not config.namespace_prefix.strip():
            raise ValueError("context_runtime namespace_prefix must be non-empty")
        parsed = urlsplit(config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("context_runtime base_url must be an absolute http:// or https:// URL")
        if parsed.scheme == "http" and not config.allow_insecure_http:
            raise ValueError("context_runtime base_url must use https:// because it carries the API key; set allow_insecure_http: true only for trusted local/LAN deployments")
        return config

    def resolve_api_key(self) -> str | None:
        """Read the API key from the configured environment variable; None when unset (the
        control plane itself fails closed with 503 when no key is configured server-side, so an
        unset key here is only valid against a server explicitly deployed without auth)."""
        key = os.environ.get(self.api_key_env, "").strip()
        return key or None
