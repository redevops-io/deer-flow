#!/usr/bin/env python
"""Headless DeerFlow deep-research runner (in-process lead agent, no gateway/UI).

    cd ~/projects/deer-flow-v2/backend && PYTHONPATH=. .venv/bin/python ../research/run_topic.py <topics.json> <out_dir> [slug ...] [--model deepseek-lan] [--judge-model flash-next] [--no-judge]

topics.json = {"slug": {"title": "...", "query": "..."}, ...}. Resumable: a slug whose out_dir/<slug>.md exists and is substantial is skipped.
The final chat message AND any files the agent wrote to its outputs folder are collected, so the full report is kept.

Acceptance gate: a candidate answer must both (a) clear a length floor and (b)
be affirmed by an independent judge model call -- a fresh prompt with just the
original query and the candidate answer, no shared conversation history with
the agent that produced it. This closes the self-grading gap audited
2026-11-15 (REDEVOPS_PRODUCTION_AGENT_GAPS_AUDIT_2026-11-15.md #1): the
previous version's only check was a character-count threshold, so a
confidently-wrong answer over that length passed with no real verification.
Pass --no-judge to fall back to the length-only gate (useful for fast
iteration without model calls); --judge-model overrides which model judges.
"""
import asyncio, json, shutil, sys, time
from pathlib import Path

CONC = 3

_JUDGE_PROMPT = """You are an independent reviewer. You did not write the candidate answer below, you have no stake in it being accepted, and you have no memory of how it was produced -- judge only what is in front of you.

Research question (the full brief the writer was given):
{query}

Candidate answer:
{answer}

Does the candidate answer substantively address the research question's stated objective and deliverables -- real, specific content engaging with what was actually asked, not a refusal, a restatement of the question, filler, or a confident-sounding non-answer? Reply with exactly one word, nothing else: ACCEPT or REJECT."""


async def judge_accepts(query: str, answer: str, judge_model_name: str | None) -> tuple[bool, str]:
    """Fresh, isolated accept/reject call. Returns (accepted, raw_verdict_text).

    No judge configured (``judge_model_name`` is None) means no independent
    check runs -- the caller falls back to the length-only gate. This is a
    deliberately separate call from the agent that produced ``answer``: it
    shares no conversation history, so it cannot simply replay the writer's
    own confidence back as a verdict.
    """
    if not judge_model_name:
        return True, "(no judge configured)"
    from langchain_core.messages import HumanMessage

    from deerflow.models.factory import create_chat_model

    judge = create_chat_model(judge_model_name)
    prompt = _JUDGE_PROMPT.format(query=query[:4000], answer=answer[:12000])
    resp = await judge.ainvoke([HumanMessage(content=prompt)])
    text = (resp.content or "") if isinstance(resp.content, str) else ""
    verdict = text.strip().upper()
    return verdict.startswith("ACCEPT"), text.strip()


async def one(slug, spec, sem, *, model: str, judge_model: str | None, out: Path):
    from langchain_core.messages import HumanMessage
    from langgraph.runtime import Runtime
    from deerflow.agents import make_lead_agent
    f = out / f"{slug}.md"
    if f.exists() and f.stat().st_size > 3000:
        print(f"[skip] {slug}", flush=True); return
    async with sem:
        tid = f"redevops-app-{slug}-{int(time.time())}"
        cfg = {"configurable": {"thread_id": tid, "thinking_enabled": False, "is_plan_mode": True, "model_name": model},
               "recursion_limit": 400}
        cfg["configurable"]["__pregel_runtime"] = Runtime(context={"thread_id": tid})
        for attempt in (1, 2, 3):
            tid = f"redevops-app-{slug}-{int(time.time())}"
            cfg = {"configurable": {"thread_id": tid, "thinking_enabled": False, "is_plan_mode": True, "model_name": model},
                   "recursion_limit": 400}
            cfg["configurable"]["__pregel_runtime"] = Runtime(context={"thread_id": tid})
            t0 = time.time(); print(f"[run ] {slug} ({model}) attempt {attempt}", flush=True)
            q = spec["query"] + ("" if attempt == 1 else " IMPORTANT: end with a NON-EMPTY written final message that summarises the findings, and make sure the markdown report file is saved.")
            try:
                agent = make_lead_agent(cfg)
                res = await agent.ainvoke({"messages": [HumanMessage(content=q)]}, config=cfg)
                msgs = res.get("messages") or []
                final = (msgs[-1].content if msgs else "") or ""
                if isinstance(final, list):
                    final = "\n".join(p.get("text", "") for p in final if isinstance(p, dict))
                extra = []
                for d in Path(".").rglob("outputs"):
                    if tid in str(d):
                        for g in d.rglob("*.md"):
                            extra.append((g.name, g.read_text(errors="replace")))
                substantive = len(final.strip()) + sum(len(t) for _, t in extra)
                if substantive < 1500:
                    print(f"[retry] {slug}: empty/thin answer ({substantive} chars) after {time.time()-t0:.0f}s", flush=True); continue
                combined_answer = final.strip() + "\n\n" + "\n\n".join(t for _, t in extra)
                accepted, verdict_text = await judge_accepts(spec["query"], combined_answer, judge_model)
                if not accepted:
                    print(f"[retry] {slug}: judge rejected ({judge_model}) after {time.time()-t0:.0f}s: {verdict_text[:200]!r}", flush=True); continue
                body = f"# {spec.get('title', slug)}\n\n> Research query: {spec['query'][:600]}\n> Engine: DeerFlow 2.1.0 lead agent, model `{model}`, run {tid}, attempt {attempt}, {time.time()-t0:.0f}s\n> Judge: `{judge_model}` -> {verdict_text[:120] if judge_model else 'not run (--no-judge)'}\n\n## Agent summary\n\n{final}\n"
                for name, txt in extra:
                    body += f"\n\n---\n\n## Report file: {name}\n\n{txt}\n"
                out.mkdir(parents=True, exist_ok=True); f.write_text(body)
                print(f"[done] {slug}: {len(body)} chars, {len(extra)} report file(s), {time.time()-t0:.0f}s, attempt {attempt}, judge={judge_model or 'skipped'}", flush=True); return
            except Exception as e:  # noqa: BLE001
                print(f"[retry] {slug}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        print(f"[FAIL] {slug}: gave up after 3 attempts", flush=True)

def _parse_args(argv: list[str]) -> tuple[list[str], str, str | None]:
    """Parse CLI flags, returning (positional_args, model, judge_model)."""
    positional = [a for a in argv if not a.startswith("--")]
    model = "deepseek-lan"
    judge_model: str | None = "flash-next"
    if "--model" in argv:
        model = argv[argv.index("--model") + 1]
        positional = [a for a in positional if a != model]
    if "--judge-model" in argv:
        judge_model = argv[argv.index("--judge-model") + 1]
        positional = [a for a in positional if a != judge_model]
    if "--no-judge" in argv:
        judge_model = None
    return positional, model, judge_model


async def main():
    from deerflow.config import get_app_config
    from deerflow.config.app_config import apply_logging_level
    apply_logging_level(get_app_config().log_level)
    from deerflow.mcp import initialize_mcp_tools
    try: await initialize_mcp_tools()
    except Exception as e: print("warn: MCP init:", e)

    args, model, judge_model = _parse_args(sys.argv[1:])
    topics = json.load(open(args[0])); out = Path(args[1]).expanduser(); only = set(args[2:])
    sem = asyncio.Semaphore(CONC)
    await asyncio.gather(*[
        one(s, sp, sem, model=model, judge_model=judge_model, out=out)
        for s, sp in topics.items() if not only or s in only
    ])


if __name__ == "__main__":
    asyncio.run(main())
