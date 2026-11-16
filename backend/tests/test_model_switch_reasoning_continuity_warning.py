"""The fifth named failure mode from the S5 resilience plan: model switching
mid-thread silently dropping reasoning continuity.

CONFIRMED GAP (now partially fixed) — the drop itself is unchanged, but it
no longer happens silently.

The audit found per-call provider/model selection (``deerflow.models.factory``)
including vLLM's Qwen ``enable_thinking`` reasoning field
(``deerflow.models.vllm_provider.VllmChatModel``), but no mechanism for
swapping the underlying model mid-session (between two turns of the same
checkpointed thread) while preserving reasoning/tool-call continuity.
Building that migration is a separate, larger "model succession" effort
(shared with Stage D of the combined release plan) and is explicitly out of
scope here. This module only proves that the silent part of the gap is
closed.

Kept as its own module (rather than folded into ``test_resilience_failure_modes.py``)
because this is a separate fix (``deerflow.models.factory``) landing as a
separate change from the top-level-recursion fix in that file.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from deerflow.config.app_config import AppConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.models.factory import warn_if_reasoning_continuity_at_risk
from deerflow.models.vllm_provider import VllmChatModel


def test_model_switch_mid_checkpointed_thread_drops_reasoning_continuity_but_now_warns(caplog):
    """CONFIRMED GAP from the audit — the drop itself is unchanged, but it no
    longer happens silently.

    Mechanism: ``VllmChatModel`` (a ``ChatOpenAI`` subclass) overrides
    ``_get_request_payload`` to re-inject a prior assistant turn's
    ``reasoning`` / ``reasoning_content`` onto the outgoing request so vLLM's
    Qwen reasoning parser can see the assistant's prior chain-of-thought on
    the next turn. That override is private to ``VllmChatModel``; the base
    ``ChatOpenAI`` (or any other provider class an operator might switch to
    mid-session) has no equivalent and simply omits the field, because
    "reasoning" is not a standard OpenAI chat-completion message field.

    A checkpointed thread's message history (what LangGraph persists and
    replays across turns) is identical in both cases below — only the model
    class bound to turn 2 differs, exactly as it would if an operator
    reconfigured the thread's model between turns.

    OBSERVED BEHAVIOR: staying on the same provider (``VllmChatModel`` on
    both turns) preserves the prior turn's reasoning on the outgoing request,
    and (now) raises no warning, since nothing is at risk. Switching the
    configured model to a different provider class (``ChatOpenAI``) for turn
    2 of the *same checkpointed thread* still silently drops that reasoning
    field from the outgoing request — ``_get_request_payload`` itself still
    raises nothing and logs nothing, because that is third-party
    ``langchain_openai`` code this fix does not touch. What closes the gap is
    ``deerflow.models.factory.warn_if_reasoning_continuity_at_risk`` — the
    real per-run check ``deerflow.runtime.runs.worker.run_agent`` now runs
    against the thread's checkpointed history before each turn — which now
    logs an operator-visible warning for exactly this switch.
    """
    # The checkpointed thread history after turn 1: a user message, a
    # vLLM/Qwen assistant turn carrying reasoning + a tool call, the tool
    # result, and a new turn-2 user message. This is exactly what LangGraph's
    # checkpointer would replay into turn 2 on the same thread_id, regardless
    # of which model class turn 2 is configured to use.
    checkpointed_history = [
        HumanMessage(content="Investigate the outage"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning": "The error logs suggest a DB connection leak; checking status first."},
            tool_calls=[{"id": "call-1", "name": "lookup_status", "args": {}}],
        ),
        ToolMessage(content="tool completed", tool_call_id="call-1"),
        HumanMessage(content="What did you find?"),
    ]

    # Scenario A: turn 2 stays on the same provider (no model switch).
    same_provider_turn2 = VllmChatModel(model="qwen-test", api_key="test-key", base_url="http://localhost:0/v1")
    same_provider_payload = same_provider_turn2._get_request_payload(checkpointed_history, stop=None)
    same_provider_assistant_msg = next(m for m in same_provider_payload["messages"] if m.get("role") == "assistant")
    assert same_provider_assistant_msg.get("reasoning") == "The error logs suggest a DB connection leak; checking status first."

    # Scenario B: turn 2 is switched to a different model/provider class
    # mid-session, on the SAME checkpointed thread history.
    switched_provider_turn2 = ChatOpenAI(model="gpt-test", api_key="test-key", base_url="http://localhost:0/v1")
    switched_provider_payload = switched_provider_turn2._get_request_payload(checkpointed_history, stop=None)
    switched_provider_assistant_msg = next(m for m in switched_provider_payload["messages"] if m.get("role") == "assistant")

    # The drop itself is unchanged: the prior turn's reasoning is still
    # silently absent from the raw request payload. No error is raised, no
    # warning is logged by either langchain_openai class — building a durable
    # cross-model-version handoff remains out of scope for this fix.
    assert "reasoning" not in switched_provider_assistant_msg

    # What IS new: the real per-run detection fires an operator-visible
    # warning for exactly this scenario (same checkpointed_history, model
    # switched to a non-reasoning-preserving provider) ...
    app_config = AppConfig(
        models=[
            ModelConfig(name="vllm-test", display_name="vllm-test", description=None, use="deerflow.models.vllm_provider:VllmChatModel", model="vllm-test"),
            ModelConfig(name="gpt-test", display_name="gpt-test", description=None, use="langchain_openai:ChatOpenAI", model="gpt-test"),
        ],
        sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
    )

    with caplog.at_level(logging.WARNING, logger="deerflow.models.factory"):
        fired = warn_if_reasoning_continuity_at_risk(
            "gpt-test",
            checkpointed_history,
            app_config=app_config,
            thread_id="resilience-thread-5",
        )
    assert fired is True
    assert any("reasoning continuity will be lost" in record.message for record in caplog.records)

    # ... and does NOT fire a false-positive warning when the thread stays on
    # a reasoning-preserving provider (nothing is actually at risk).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="deerflow.models.factory"):
        fired_same_provider = warn_if_reasoning_continuity_at_risk(
            "vllm-test",
            checkpointed_history,
            app_config=app_config,
            thread_id="resilience-thread-5",
        )
    assert fired_same_provider is False
    assert not caplog.records
