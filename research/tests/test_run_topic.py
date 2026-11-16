"""Unit tests for run_topic.py's independent judge-acceptance gate.

Added 2026-11-15 per REDEVOPS_PRODUCTION_AGENT_GAPS_AUDIT_2026-11-15.md #1:
the previous version's only acceptance check was a character-count
threshold (self-grading, no real verification). ``judge_accepts`` replaces
that with a fresh, isolated model call that shares no conversation history
with the agent that produced the candidate answer.

Run with: PYTHONPATH=../backend:. python -m pytest research/tests/test_run_topic.py -v
(from the repo root), or see the module docstring's invocation examples.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # research/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))  # backend/

import run_topic  # noqa: E402


def test_parse_args_defaults():
    positional, model, judge_model = run_topic._parse_args(["topics.json", "out"])
    assert positional == ["topics.json", "out"]
    assert model == "deepseek-lan"
    assert judge_model == "flash-next"


def test_parse_args_overrides_and_no_judge():
    positional, model, judge_model = run_topic._parse_args(
        ["topics.json", "out", "--model", "qwen-dual-gpu", "--judge-model", "kimi"]
    )
    assert positional == ["topics.json", "out"]
    assert model == "qwen-dual-gpu"
    assert judge_model == "kimi"

    _, _, judge_model_disabled = run_topic._parse_args(["topics.json", "out", "--no-judge"])
    assert judge_model_disabled is None


@pytest.mark.asyncio
async def test_judge_accepts_returns_true_without_a_configured_judge():
    """No judge configured -> no independent check runs; caller keeps the length-only gate."""
    accepted, text = await run_topic.judge_accepts("query", "answer", None)
    assert accepted is True
    assert "no judge configured" in text


@pytest.mark.asyncio
async def test_judge_accepts_calls_a_fresh_isolated_model_with_no_shared_history():
    """The judge call must carry only the query + candidate answer -- never the
    writer's own conversation -- and must drive the accept/reject decision."""
    fake_response = MagicMock(content="ACCEPT")
    fake_judge = MagicMock()
    fake_judge.ainvoke = AsyncMock(return_value=fake_response)

    with patch("deerflow.models.factory.create_chat_model", return_value=fake_judge) as create_mock:
        accepted, text = await run_topic.judge_accepts(
            "What is the capital of France?", "Paris is the capital of France.", "flash-next"
        )

    assert accepted is True
    assert text == "ACCEPT"
    create_mock.assert_called_once_with("flash-next")
    # Exactly one message was sent to the judge: no prior conversation turns.
    (sent_messages,), _ = fake_judge.ainvoke.call_args
    assert len(sent_messages) == 1
    prompt_text = sent_messages[0].content
    assert "What is the capital of France?" in prompt_text
    assert "Paris is the capital of France." in prompt_text


@pytest.mark.asyncio
async def test_judge_accepts_rejects_a_confidently_wrong_answer():
    """This is the exact failure mode the audit flagged: a long, confident,
    off-topic answer must not pass just because it clears a length floor."""
    fake_response = MagicMock(content="REJECT")
    fake_judge = MagicMock()
    fake_judge.ainvoke = AsyncMock(return_value=fake_response)

    with patch("deerflow.models.factory.create_chat_model", return_value=fake_judge):
        accepted, text = await run_topic.judge_accepts(
            "Summarize the attached quarterly earnings report.",
            "Here is a long essay about the history of jazz music in New Orleans " * 40,
            "flash-next",
        )

    assert accepted is False
    assert text == "REJECT"


@pytest.mark.asyncio
async def test_judge_accepts_treats_non_accept_verdicts_as_rejection():
    """A verdict that doesn't start with ACCEPT (garbled, refusal, etc.) must
    fail closed -- never default to accepting an unclear judge response."""
    fake_response = MagicMock(content="I'm not sure, this seems incomplete.")
    fake_judge = MagicMock()
    fake_judge.ainvoke = AsyncMock(return_value=fake_response)

    with patch("deerflow.models.factory.create_chat_model", return_value=fake_judge):
        accepted, _ = await run_topic.judge_accepts("q", "a", "flash-next")

    assert accepted is False
