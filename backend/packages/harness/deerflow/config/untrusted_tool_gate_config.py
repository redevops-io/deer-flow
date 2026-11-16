"""Configuration for the untrusted-content tool-gating middleware.

Mitigates the "lethal trifecta" (untrusted external text + private data +
write/shell tools, together, in one agent context): once a tool result from
an untrusted-source group (default: ``web``, ``browser``) appears in a
thread's history, write-capable tools (default: ``bash``, ``file:write``)
are withheld from model binding for the rest of that run.
"""

from pydantic import BaseModel, Field


class UntrustedToolGateConfig(BaseModel):
    """Configuration for ``UntrustedContentToolGateMiddleware``."""

    enabled: bool = Field(
        default=True,
        description=(
            "Whether to withhold write-capable tools once untrusted remote content (web fetch/search/browser "
            "results) has entered the thread's history. Default on, since this closes a real "
            "prompt-injection-to-side-effect gap. Disable only when write tools and untrusted input are already "
            "mutually exclusive by some other infrastructure control (e.g. a fully isolated single-purpose sandbox)."
        ),
    )
    untrusted_source_groups: frozenset[str] = Field(
        default=frozenset({"web", "browser"}),
        description="Tool groups (from config.yaml's tool_groups) whose results are treated as untrusted external content.",
    )
    write_tool_groups: frozenset[str] = Field(
        default=frozenset({"bash", "file:write"}),
        description="Tool groups withheld from model binding once untrusted content has been seen in the thread.",
    )
