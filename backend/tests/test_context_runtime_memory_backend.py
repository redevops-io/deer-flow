"""Unit tests for the context-runtime HTTP memory backend (backends/context_runtime/)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.memory.backends.context_runtime.client import (
    ContextRuntimeAPIError,
    ContextRuntimeAuthError,
    ContextRuntimeClient,
)
from deerflow.agents.memory.backends.context_runtime.config import ContextRuntimeConfig
from deerflow.agents.memory.backends.context_runtime.context_runtime_manager import (
    ContextRuntimeManager,
)
from deerflow.agents.memory.manager import MemoryManagerError


class TestContextRuntimeConfig:
    def test_defaults(self) -> None:
        cfg = ContextRuntimeConfig.from_backend_config({})
        assert cfg.api_key_env == "CONTEXT_RUNTIME_API_KEY"
        assert cfg.base_url == "http://127.0.0.1:8080"
        assert cfg.allow_insecure_http is True
        assert cfg.namespace_prefix == "deerflow"
        assert cfg.top_k == 8
        assert cfg.read_policy == "fail_open"
        assert cfg.write_policy == "log_and_drop"

    def test_unknown_keys_rejected_except_host_injected(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            ContextRuntimeConfig.from_backend_config({"typo_knob": 1})
        cfg = ContextRuntimeConfig.from_backend_config({"storage_path": "/tmp/x"})
        assert cfg.base_url == "http://127.0.0.1:8080"

    def test_https_required_without_opt_in(self) -> None:
        with pytest.raises(ValueError, match="allow_insecure_http"):
            ContextRuntimeConfig.from_backend_config({"base_url": "http://x:1", "allow_insecure_http": False})

    @pytest.mark.parametrize("base_url", ["cr.local:8080", "ftp://cr.local"])
    def test_invalid_base_url_rejected(self, base_url: str) -> None:
        with pytest.raises(ValueError, match="base_url"):
            ContextRuntimeConfig.from_backend_config({"base_url": base_url})

    @pytest.mark.parametrize(("key", "value"), [("top_k", 0), ("timeout_seconds", 0), ("namespace_prefix", "")])
    def test_invalid_values_rejected(self, key: str, value: object) -> None:
        with pytest.raises(ValueError):
            ContextRuntimeConfig.from_backend_config({key: value})


class TestContextRuntimeClient:
    def _client(self, handler: Any) -> ContextRuntimeClient:
        return ContextRuntimeClient(base_url="http://cr.local", api_key="k", transport=httpx.MockTransport(handler))

    def test_write_fact_posts_body(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["json"] = httpx.Request.read(request) and request.content
            return httpx.Response(200, json={"subject": "t1", "predicate": "user", "object": "hi"})

        c = self._client(handler)
        out = c.write_fact(namespace="ns", subject="t1", predicate="user", obj="hi")
        assert out["subject"] == "t1"

    def test_read_facts_returns_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            return httpx.Response(200, json=[{"text": "hi", "meta": {}}])

        c = self._client(handler)
        assert c.read_facts(namespace="ns", query="hi") == [{"text": "hi", "meta": {}}]

    def test_401_raises_auth_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "nope"})

        c = self._client(handler)
        with pytest.raises(ContextRuntimeAuthError):
            c.read_facts(namespace="ns")

    def test_5xx_raises_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        c = self._client(handler)
        with pytest.raises(ContextRuntimeAPIError):
            c.read_facts(namespace="ns")


class FakeContextRuntimeClient:
    """Test double injected as manager._client (records calls, returns fixtures)."""

    def __init__(self) -> None:
        self.written: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []
        self.read_results: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.closed = False

    def _maybe_raise(self) -> None:
        if self.error is not None:
            raise self.error

    def write_fact(self, **kwargs: Any) -> dict[str, Any]:
        self._maybe_raise()
        self.written.append(kwargs)
        return kwargs

    def read_facts(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._maybe_raise()
        self.read_calls.append(kwargs)
        return self.read_results

    def close(self) -> None:
        self.closed = True


def _manager(backend_config: dict | None = None, *, mode: str = "middleware") -> tuple[ContextRuntimeManager, FakeContextRuntimeClient]:
    mgr = ContextRuntimeManager(backend_config=backend_config or {}, mode=mode)
    fake = FakeContextRuntimeClient()
    mgr._client = fake
    return mgr, fake


class TestContextRuntimeManagerAdd:
    def test_add_writes_one_fact_per_kept_message(self) -> None:
        mgr, fake = _manager()
        tool_ai = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
        mgr.add(
            "thread-1",
            [HumanMessage(content="I prefer dark mode"), tool_ai, AIMessage(content="Noted.")],
            agent_name="lead_agent",
            user_id="u1",
        )
        assert len(fake.written) == 2
        first = fake.written[0]
        assert first["subject"] == "thread-1"
        assert first["predicate"] == "user"
        assert first["obj"] == "I prefer dark mode"
        assert first["meta"]["agent_name"] == "lead_agent"
        assert first["meta"]["user_id"] == "u1"
        assert first["meta"]["source"] == "deerflow"
        assert fake.written[1]["predicate"] == "assistant"

    def test_add_empty_after_filter_is_noop(self) -> None:
        mgr, fake = _manager()
        hidden = HumanMessage(content="internal", additional_kwargs={"hide_from_ui": True})
        mgr.add("thread-1", [hidden], user_id="u1")
        assert fake.written == []

    def test_add_write_error_log_and_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        mgr, fake = _manager()
        fake.error = ContextRuntimeAPIError("server down")
        mgr.add("thread-1", [HumanMessage(content="hi")], user_id="u1")  # must not raise
        assert any("context-runtime" in r.message for r in caplog.records)

    def test_add_write_error_raise_policy(self) -> None:
        mgr, fake = _manager({"failure_policy": {"write": "raise"}})
        fake.error = ContextRuntimeAPIError("server down")
        with pytest.raises(MemoryManagerError):
            mgr.add("thread-1", [HumanMessage(content="hi")], user_id="u1")


class TestContextRuntimeManagerGetContext:
    def test_get_context_joins_fact_texts(self) -> None:
        mgr, fake = _manager()
        fake.read_results = [{"text": "owner is alice", "meta": {}}, {"text": "status is ok", "meta": {}}]
        ctx = mgr.get_context("u1", agent_name="lead_agent", thread_id="t1")
        assert ctx == "- owner is alice\n- status is ok"
        assert fake.read_calls[0]["namespace"] == "deerflow:u1|lead_agent"

    def test_get_context_read_error_fail_open_returns_empty(self) -> None:
        mgr, fake = _manager()
        fake.error = ContextRuntimeAPIError("down")
        assert mgr.get_context("u1") == ""

    def test_get_context_read_error_fail_closed_raises(self) -> None:
        from deerflow.agents.memory.manager import MemoryReadError

        mgr, fake = _manager({"failure_policy": {"read": "fail_closed"}})
        fake.error = ContextRuntimeAPIError("down")
        with pytest.raises(MemoryReadError):
            mgr.get_context("u1")


class TestContextRuntimeManagerSearch:
    def test_search_maps_hits_to_facts(self) -> None:
        mgr, fake = _manager(mode="tool")
        fake.read_results = [{"chunk_id": "a:b:-", "text": "x", "score": 0.5, "meta": {"role": "user", "recorded_at": "t"}}]
        out = mgr.search("x", top_k=3, user_id="u1")
        assert out == [{"id": "a:b:-", "content": "x", "category": "user", "confidence": 0.5, "createdAt": "t", "source": "deerflow"}]

    def test_supports_search_flag_true(self) -> None:
        mgr, _ = _manager(mode="tool")
        assert mgr.supports_search is True


class TestContextRuntimeManagerConstruction:
    def test_close_releases_http_client(self) -> None:
        mgr, fake = _manager()
        mgr.close()
        assert fake.closed is True

    def test_from_config_does_not_require_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CONTEXT_RUNTIME_API_KEY", raising=False)
        mgr = ContextRuntimeManager.from_config({})  # must not raise -- server may run keyless locally
        assert mgr is not None
