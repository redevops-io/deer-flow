"""Withhold write-capable tools once untrusted remote content has entered context.

The "lethal trifecta": an agent that can (a) read untrusted external text
(web fetch/search/browser results), (b) see private/internal data, and (c)
call write/shell tools is an unsolvable prompt-injection risk — content an
attacker controls can instruct the model to act, and the model cannot
reliably tell an instruction embedded in fetched content from a real one.

This middleware removes that combination structurally rather than warning
about it: once any tool result from an untrusted-source tool group (default
``web``, ``browser``) appears anywhere in the thread's message history,
write-capable tool groups (default ``bash``, ``file:write``) are removed from
``request.tools`` before every subsequent model call in that run, and any
tool call that still names one is rejected before execution. A prompt
embedded in fetched content cannot make a withheld tool schema reappear —
unlike a warning, which such a prompt could simply instruct the model to
ignore.

Scope and known limitations, confirmed by adversarial testing (see
``test_untrusted_tool_gate_middleware.py`` and
``test_untrusted_tool_gate_wiring.py``):

- Detection scans ``state["messages"]`` for a qualifying ``ToolMessage``, so
  the gate is thread-durable only as long as that message stays in the
  visible window. If a future summarization pass compacts it out of history
  without leaving an equivalent durable marker, a later turn in the same
  thread could regain write-tool access. No such marker exists yet; adding
  one (mirroring ``DurableContextMiddleware``'s receipt-survives-
  summarization pattern) is a reasonable follow-up, not implemented here to
  keep this fix scoped to the gating mechanism itself.
- Coverage is a group-label allowlist, not a tool-capability inspection: a
  tool is only withheld if its configured ``group`` (in config.yaml) appears
  in ``write_tool_groups``. A new side-effecting tool (sends email, writes to
  an external system, executes code under a different name) that is not
  tagged into one of those groups is invisible to this gate and will NOT be
  withheld, even after untrusted content has been seen. There is no
  automatic way to detect "this tool has a side effect" from configuration
  alone; operators must deliberately put every write/side-effecting tool
  into a group listed in ``write_tool_groups`` for it to be covered.
  ``_build_runtime_middlewares`` logs a warning if the configured groups
  resolve to zero tools while tools exist, so a renamed/typo'd group is at
  least visible, but a *wrongly* -tagged-but-still-resolving tool is not
  detectable from config alone.
- The check is scoped to the current thread's own message history. A fact
  written to long-lived memory (DeerMem) while untrusted content was active
  in one thread carries no provenance tag; if it resurfaces in a brand-new
  thread that never itself called an untrusted-source tool, this gate would
  not block write tools there even if that resurfaced fact originated from a
  prompt-injection attempt. Closing this needs provenance tracking through
  the memory write/read path, which is out of scope for this middleware.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class UntrustedContentToolGateMiddleware(AgentMiddleware[AgentState]):
    """Remove *write_tool_names* from model binding once a result from
    *untrusted_source_names* has appeared in the thread's history."""

    def __init__(self, untrusted_source_names: frozenset[str], write_tool_names: frozenset[str]) -> None:
        super().__init__()
        self._untrusted_source_names = untrusted_source_names
        self._write_tool_names = write_tool_names

    def release_policy_parameters(self) -> dict[str, object]:
        return {
            "untrusted_source_names": sorted(self._untrusted_source_names),
            "write_tool_names": sorted(self._write_tool_names),
        }

    def _is_untrusted_result(self, message: object) -> bool:
        if not isinstance(message, ToolMessage):
            return False
        return str(getattr(message, "name", "") or "") in self._untrusted_source_names

    def _untrusted_content_seen(self, state: object) -> bool:
        messages = (state or {}).get("messages") or []
        return any(self._is_untrusted_result(m) for m in messages)

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        if not self._write_tool_names or not self._untrusted_content_seen(request.state):
            return request
        active = [t for t in request.tools if getattr(t, "name", None) not in self._write_tool_names]
        if len(active) < len(request.tools):
            logger.warning(
                "Untrusted remote content present in this thread; withholding %d write-capable tool schema(s) from model binding for the rest of this run",
                len(request.tools) - len(active),
            )
        return request.override(tools=active)

    def _blocked_tool_message(self, request: ToolCallRequest) -> ToolMessage | None:
        if not self._write_tool_names:
            return None
        name = str(request.tool_call.get("name") or "")
        if not name or name not in self._write_tool_names:
            return None
        if not self._untrusted_content_seen(request.state):
            return None
        tool_call_id = str(request.tool_call.get("id") or "missing_tool_call_id")
        return ToolMessage(
            content=(
                f"Error: Tool '{name}' is unavailable because untrusted external content "
                "(a web fetch/search or browser result) has entered this conversation. "
                "Write and shell tools are withheld for the rest of this run so that a "
                "prompt-injection attempt embedded in that content cannot trigger a side effect."
            ),
            tool_call_id=tool_call_id,
            name=name,
            status="error",
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._filter_tools(request))

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        blocked = self._blocked_tool_message(request)
        if blocked is not None:
            return blocked
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._filter_tools(request))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        blocked = self._blocked_tool_message(request)
        if blocked is not None:
            return blocked
        return await handler(request)
