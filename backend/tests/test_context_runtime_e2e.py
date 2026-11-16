"""Real end-to-end test: starts the actual context-runtime control-plane service as a
subprocess (its own venv, its own process -- not imported into this process) and has the new
ContextRuntimeManager backend talk to it over a real TCP port via plain httpx. Not mocked on
either side.

Requires CONTEXT_RUNTIME_SERVICE_PYTHON to point at a Python executable with context-runtime's
control-plane extras installed (``pip install -e ".[control-plane]"`` in that checkout); skips
otherwise, since most CI runs won't have a sibling context-runtime checkout available.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.memory.backends.context_runtime.context_runtime_manager import (
    ContextRuntimeManager,
)

SERVICE_PYTHON = os.environ.get("CONTEXT_RUNTIME_SERVICE_PYTHON", "")

pytestmark = pytest.mark.skipif(not SERVICE_PYTHON, reason="set CONTEXT_RUNTIME_SERVICE_PYTHON to a context-runtime venv's python to run this")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 20.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as e:
            last_exc = e
        time.sleep(0.25)
    raise RuntimeError(f"context-runtime service did not become healthy in time: {last_exc}")


@pytest.fixture
def live_service(tmp_path):
    port = _free_port()
    env = dict(os.environ)
    env["CONTEXT_RUNTIME_API_KEY"] = "e2e-test-key"
    env["CONTEXT_RUNTIME_HOME"] = str(tmp_path)
    env["CR_CORS_ORIGINS"] = ""
    proc = subprocess.Popen(
        [SERVICE_PYTHON, "-m", "uvicorn", "context_runtime.control_plane.app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_real_round_trip_write_then_search_then_get_context(live_service, monkeypatch):
    """The core, not-mocked assertion: a real HTTP write through the new backend lands in the
    real TemporalStore-backed service, and a real HTTP read gets it back with provenance."""
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", "e2e-test-key")
    mgr = ContextRuntimeManager.from_config({"base_url": live_service, "namespace_prefix": "e2e-test"})
    try:
        mgr.add(
            "thread-e2e-1",
            [HumanMessage(content="our team is expanding, we need more seats"), AIMessage(content="noted: upsell signal")],
            agent_name="lead_agent",
            user_id="user-e2e",
        )

        ctx = mgr.get_context("user-e2e", agent_name="lead_agent", thread_id="thread-e2e-1")
        assert "our team is expanding" in ctx
        assert "noted: upsell signal" in ctx

        hits = mgr.search("expanding seats", top_k=5, user_id="user-e2e", agent_name="lead_agent")
        assert len(hits) >= 1
        assert any("expanding" in h["content"] for h in hits)
        assert all(h["source"] == "deerflow" for h in hits)
    finally:
        mgr.close()


def test_real_as_of_point_in_time_via_raw_client(live_service, monkeypatch):
    """Exercise the point-in-time query path directly through the client (not just get_context's
    query-less recall), against the real running service."""
    from deerflow.agents.memory.backends.context_runtime.client import ContextRuntimeClient

    client = ContextRuntimeClient(base_url=live_service, api_key="e2e-test-key")
    try:
        client.write_fact(namespace="e2e-asof", subject="svc", predicate="owner", obj="alice", valid_from="2026-01-01", valid_to="2026-06-01")
        client.write_fact(namespace="e2e-asof", subject="svc", predicate="owner", obj="bob", valid_from="2026-06-01")
        before = client.read_facts(namespace="e2e-asof", query="svc owner", at="2026-03-01")
        after = client.read_facts(namespace="e2e-asof", query="svc owner", at="2026-07-01")
        assert before[0]["meta"]["object"] == "alice"
        assert after[0]["meta"]["object"] == "bob"
    finally:
        client.close()


def test_real_namespace_isolation_across_two_managers(live_service, monkeypatch):
    """Two different buckets (different user_id) must not see each other's facts, for real."""
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", "e2e-test-key")
    mgr_a = ContextRuntimeManager.from_config({"base_url": live_service, "namespace_prefix": "e2e-iso"})
    mgr_b = ContextRuntimeManager.from_config({"base_url": live_service, "namespace_prefix": "e2e-iso"})
    try:
        mgr_a.add("t1", [HumanMessage(content="alice's secret preference")], user_id="alice")
        ctx_b = mgr_b.get_context("bob")
        assert "secret" not in ctx_b
        ctx_a = mgr_a.get_context("alice")
        assert "secret" in ctx_a
    finally:
        mgr_a.close()
        mgr_b.close()
