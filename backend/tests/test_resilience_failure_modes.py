"""Resilience benchmark suite for three of the four DeerFlow failure modes
named in the S5 (DeerFlow Integration and Long-Running Agent Resilience)
next-release plan:

1. Empty final responses               -> TerminalResponseMiddleware
2. Subagent recursion exhaustion        -> SubagentExecutor / GraphRecursionError
3. Interrupted watchers (orphaned runs) -> RunManager.reconcile_orphaned_inflight_runs
4. Top-level (lead-agent) recursion     -> FIXED, proven empirically below

Tests 1-3 exercise mechanisms that were already built and covered by
narrower unit tests elsewhere in this suite (``test_terminal_response_middleware.py``,
``test_subagent_turn_budget.py`` / ``test_subagent_executor.py``,
``test_multi_worker_run_ownership.py``). They are consolidated here, under
the failure-mode names, as a single resilience regression surface — they
are expected to PASS, proving recovery works.

Test 4 is this suite's own fix: the top-level graph run (driven by
``deerflow.runtime.runs.worker.run_agent``) previously let ``GraphRecursionError``
propagate as an undifferentiated, unrecovered run failure. It now catches it
specifically and recovers a partial result the same way ``SubagentExecutor``
already did one layer down.

The fifth named failure mode (model switching mid-thread) has its own
dedicated test module: ``test_model_switch_reasoning_continuity_warning.py``.
It is a separate fix (``deerflow.models.factory``) landing as a separate
change from this one, so it is kept in its own file rather than folded in
here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

# Reuse the real fixtures from test_subagent_executor.py rather than
# re-implementing its module-mocking dance (it breaks a genuine import cycle
# between deerflow.subagents.__init__ and deerflow.agents via a conftest.py
# sys.modules substitution that this fixture chain temporarily undoes).
# These names are pytest fixtures: importing the already-decorated functions
# makes them usable here exactly as in their home module.
from test_subagent_executor import (  # noqa: F401
    _setup_executor_classes,  # noqa: F401
    base_config,
    classes,
    mock_agent,
    msg,
)

# ---------------------------------------------------------------------------
# 1. Empty final responses — TerminalResponseMiddleware (already handled)
# ---------------------------------------------------------------------------
from deerflow.agents.middlewares.terminal_response_middleware import TerminalResponseMiddleware

# ---------------------------------------------------------------------------
# 3. Interrupted watchers / orphaned runs — RunManager (already handled)
# ---------------------------------------------------------------------------
# 4. Top-level recursion — worker.run_agent (fixed by this suite)
# ---------------------------------------------------------------------------
from deerflow.runtime import TOP_LEVEL_RECURSION_STOP_REASON, RunContext, RunManager, RunStatus
from deerflow.runtime.runs.store.memory import MemoryRunStore
from deerflow.runtime.runs.worker import run_agent


@tool
def lookup_status() -> str:
    """Return a deterministic tool result."""
    return "tool completed"


# ---------------------------------------------------------------------------
# 1. Empty final responses: retries once, then falls back to a visible message
# ---------------------------------------------------------------------------


class _PostToolEmptyThenEmptyModel(BaseChatModel):
    """Calls a tool, then returns an empty terminal response twice in a row."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "post-tool-empty-twice"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            message = AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "lookup_status", "args": {}}],
                response_metadata={"finish_reason": "tool_calls"},
            )
        else:
            # Both the first post-tool answer and the retried answer are empty.
            message = AIMessage(content="", response_metadata={"finish_reason": "stop"})
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_empty_final_response_retries_once_then_falls_back_to_visible_message():
    """Benchmarks failure mode #1: empty final responses.

    Confirms the already-built recovery path: TerminalResponseMiddleware
    retries exactly once on an empty post-tool-call assistant response, and
    when the retry is *also* empty, persists the visible fallback message
    instead of silently returning nothing to the user.
    """
    model = _PostToolEmptyThenEmptyModel()
    agent = create_agent(model=model, tools=[lookup_status], middleware=[TerminalResponseMiddleware()])

    result = agent.invoke(
        {"messages": [HumanMessage(content="Check the status")]},
        context={"thread_id": "resilience-thread-1", "run_id": "resilience-run-1"},
    )

    # tool-call turn + empty turn + one retry turn = 3 model calls.
    assert model.call_count == 3
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "returned no final response" in str(final.content)
    assert final.additional_kwargs["deerflow_error_fallback"] is True
    # No empty AIMessage is left dangling in the persisted transcript.
    assert not any(isinstance(m, AIMessage) and not m.tool_calls and not str(m.content).strip() for m in result["messages"])


# ---------------------------------------------------------------------------
# 2. Subagent recursion exhaustion: GraphRecursionError recovers a partial result
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_subagent_recursion_exhaustion_recovers_partial_result(
    classes,  # noqa: F811 - pytest fixture parameter, shadows the module-level import by design
    base_config,  # noqa: F811 - pytest fixture parameter, shadows the module-level import by design
    mock_agent,  # noqa: F811 - pytest fixture parameter, shadows the module-level import by design
    msg,  # noqa: F811 - pytest fixture parameter, shadows the module-level import by design
):
    """Benchmarks failure mode #2: subagent recursion exhaustion.

    Confirms the already-built recovery path in
    ``deerflow.subagents.executor.SubagentExecutor``: when a subagent's
    LangGraph loop exhausts its ``max_turns`` budget, LangGraph raises
    ``GraphRecursionError`` mid-stream. The executor catches it and, when the
    stream already produced usable partial work (a non-empty AIMessage),
    recovers that partial result as a COMPLETED run with
    ``stop_reason="turn_capped"`` instead of raising or reporting a bare
    failure.
    """
    SubagentExecutor = classes["SubagentExecutor"]
    SubagentStatus = classes["SubagentStatus"]

    partial_ai = msg.ai("Found 3 of 5 sources; still working", "msg-resilience-1")
    partial_state = {"messages": [msg.human("Task"), partial_ai]}

    async def mock_astream(*args, **kwargs):
        yield partial_state
        raise GraphRecursionError(f"Recursion limit of {base_config.max_turns} reached")

    mock_agent.astream = mock_astream

    executor = SubagentExecutor(config=base_config, tools=[], thread_id="resilience-thread-2")

    with patch.object(executor, "_create_agent", return_value=mock_agent):
        result = await executor._aexecute("Task")

    assert result.status == SubagentStatus.COMPLETED
    assert result.result == "Found 3 of 5 sources; still working"
    assert result.stop_reason == "turn_capped"
    assert result.error is None


# ---------------------------------------------------------------------------
# 3. Interrupted watchers: orphaned runs with an expired lease are recovered
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orphaned_run_with_expired_lease_is_recovered():
    """Benchmarks failure mode #3: interrupted watchers / crashed workers.

    Confirms the already-built recovery path:
    ``RunManager.reconcile_orphaned_inflight_runs`` detects a persisted
    in-flight run whose lease expired (its owning worker crashed mid-run
    without renewing the heartbeat) and reclaims it as a terminal error with
    ``ORPHAN_RECOVERY_STOP_REASON`` instead of leaving it stuck "running"
    forever.
    """
    from datetime import UTC, datetime, timedelta

    from deerflow.config.run_ownership_config import RunOwnershipConfig
    from deerflow.runtime import ORPHAN_RECOVERY_STOP_REASON

    store = MemoryRunStore()
    manager = RunManager(
        store=store,
        run_ownership_config=RunOwnershipConfig(lease_seconds=30, grace_seconds=10, heartbeat_enabled=False),
    )

    expired_lease = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
    await store.put(
        "resilience-orphan-run",
        thread_id="resilience-thread-3",
        status="running",
        owner_worker_id="worker-that-crashed",
        lease_expires_at=expired_lease,
        created_at=(datetime.now(UTC) - timedelta(seconds=120)).isoformat(),
    )

    recovered = await manager.reconcile_orphaned_inflight_runs(
        error="Worker crashed mid-run (lease expired).",
        stop_reason=ORPHAN_RECOVERY_STOP_REASON,
    )

    assert len(recovered) == 1
    assert recovered[0].run_id == "resilience-orphan-run"
    assert recovered[0].status == RunStatus.error
    assert recovered[0].stop_reason == ORPHAN_RECOVERY_STOP_REASON

    stored = await store.get("resilience-orphan-run")
    assert stored["status"] == "error"


# ---------------------------------------------------------------------------
# 4. FIXED — top-level (lead-agent) recursion exhaustion now recovers
# ---------------------------------------------------------------------------


class _AlwaysToolCallModel(BaseChatModel):
    """Never produces a final answer; always re-invokes the same tool."""

    @property
    def _llm_type(self) -> str:
        return "always-tool-call"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="", tool_calls=[{"id": "call-loop", "name": "lookup_status", "args": {}}])
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_bare_graph_recursion_exhaustion_still_raises_without_a_recovery_layer():
    """Sanity check that the underlying mechanism is unchanged.

    A bare ``create_agent`` graph (no worker-level recovery wrapped around
    it — this mirrors the lead agent's own graph, not the fix) still raises
    ``GraphRecursionError`` straight out of ``agent.invoke()`` when it
    exhausts ``recursion_limit``. The fix lives in
    ``deerflow.runtime.runs.worker.run_agent`` (which drives the real
    top-level graph via ``agent.astream(...)``), proven by the two tests
    below — not in the graph itself.
    """
    model = _AlwaysToolCallModel()
    agent = create_agent(model=model, tools=[lookup_status], checkpointer=False)

    with pytest.raises(GraphRecursionError):
        agent.invoke({"messages": [HumanMessage(content="go")]}, {"recursion_limit": 3})


@pytest.mark.asyncio
async def test_top_level_recursion_exhaustion_recovers_partial_result():
    """Benchmarks the fixed failure mode #4: top-level recursion exhaustion.

    Confirms the new recovery path in ``deerflow.runtime.runs.worker.run_agent``:
    when the top-level graph's ``agent.astream(...)`` loop exhausts the
    Gateway's ``recursion_limit`` and raises ``GraphRecursionError`` mid-stream
    (the same error ``SubagentExecutor`` recovers one layer down), the worker
    now catches it specifically and, when the stream already produced a
    usable partial result (a non-empty assistant message in the last
    ``stream_mode="values"`` snapshot), recovers that partial result as a
    ``RunStatus.success`` run with ``stop_reason=TOP_LEVEL_RECURSION_STOP_REASON``
    (the top-level analog of the subagent's own ``"turn_capped"``) instead of
    an undifferentiated ``RunStatus.error``.
    """

    class DummyAgent:
        metadata: dict[str, object] = {"model_name": "test-model"}

        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            # Default stream_mode (no stream_modes requested below) is "values":
            # a full-state snapshot, exactly what the real Gateway stream
            # produces per super-step.
            yield {"messages": [HumanMessage(content="go"), AIMessage(content="Found 3 of 5 sources; still working")]}
            raise GraphRecursionError("Recursion limit of 25 reached")

    run_manager = RunManager()
    record = await run_manager.create("resilience-thread-4a")
    bridge = AsyncMock()
    bridge.publish = AsyncMock()
    bridge.publish_end = AsyncMock()
    bridge.cleanup = AsyncMock()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={"messages": []},
        config={},
    )

    fetched = await run_manager.get(record.run_id)
    assert fetched is not None
    assert fetched.status == RunStatus.success
    assert fetched.stop_reason == TOP_LEVEL_RECURSION_STOP_REASON
    assert fetched.error is None


@pytest.mark.asyncio
async def test_top_level_recursion_exhaustion_without_partial_result_marks_error_with_stop_reason():
    """When the recursion limit trips with no usable partial content, the run
    still identifies as a recursion-limit exhaustion (``stop_reason`` set) —
    not a bare, undifferentiated generic failure."""

    class DummyAgent:
        metadata: dict[str, object] = {"model_name": "test-model"}

        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            # The only AIMessage in the snapshot is empty — no usable partial work.
            yield {"messages": [HumanMessage(content="go"), AIMessage(content="")]}
            raise GraphRecursionError("Recursion limit of 3 reached")

    run_manager = RunManager()
    record = await run_manager.create("resilience-thread-4b")
    bridge = AsyncMock()
    bridge.publish = AsyncMock()
    bridge.publish_end = AsyncMock()
    bridge.cleanup = AsyncMock()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={"messages": []},
        config={},
    )

    fetched = await run_manager.get(record.run_id)
    assert fetched is not None
    assert fetched.status == RunStatus.error
    assert fetched.stop_reason == TOP_LEVEL_RECURSION_STOP_REASON
    assert fetched.error is not None
    assert "Recursion limit" in fetched.error
