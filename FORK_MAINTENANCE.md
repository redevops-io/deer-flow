# Fork maintenance policy

`redevops-io/deer-flow` is a fork of [`bytedance/deer-flow`](https://github.com/bytedance/deer-flow) (MIT). ReDevOps tracks two branches, not one:

- `main` — upstream's default branch.
- `2.1.x-dev` — **the branch ReDevOps actually runs in production**, on a Proxmox server against local Qwen3.8-27B models. Most of ReDevOps's own fork-local work (resilience fixes, etc.) targets this branch, not `main`.

This document covers five things. Nothing else.

## 1. Upstream tracking

`.github/workflows/sync-upstream.yml` runs weekly (Monday 06:00 UTC — upstream is active but ReDevOps's own commits land occasionally, so daily would mostly be noise) and on manual dispatch. For each tracked branch (`main`, `2.1.x-dev`):

- If the fork's branch has no commits upstream lacks (true for both today), it fast-forwards directly — no PR, no review needed.
- Once the fork has its own commits on a branch (true for `2.1.x-dev` once PRs #1/#2 merge), the workflow stops fast-forwarding and instead opens a `sync/<branch>-<date>` PR merging upstream's new tip in, for human review. It never force-pushes or rewrites fork history.

Check `upstream`'s security advisories (Settings → Advisories, or watch the repo) independently of this schedule — see §4.

## 2. Dependency pinning

Anything in the ReDevOps stack that depends on DeerFlow (deployment manifests, container builds, scripts) must pin to **a specific fork commit SHA or a container image digest built from that SHA** — never to a floating `redevops-io/deer-flow@main` or `@2.1.x-dev` branch ref, and never to `bytedance/deer-flow` directly.

The pin is bumped deliberately, as one visible commit per stack repo, only after: the fork commit exists on the tracked branch, §3's CI is green on it, and (if it carries a ReDevOps-local change) its own PR has been reviewed and merged into the fork. Automation may propose the bump; a human merges it.

## 3. CI regression tests on the fork

Before any pin elsewhere in the stack moves to a new fork commit, that commit must pass:

- Upstream's own existing CI (`backend-unit-tests.yml`, `frontend-unit-tests.yml`, `e2e-tests.yml`, etc.) — unchanged, still required. This fork does not replace upstream's test suite.
- The fork-local resilience suite: `backend/tests/test_resilience_failure_modes.py` and `backend/tests/test_model_switch_reasoning_continuity_warning.py` (from PRs #1 and #2) — these exist specifically because they test behavior upstream doesn't, and must keep passing on every commit ReDevOps intends to pin to.

Run both via the existing `uv run pytest backend/tests/...` convention already used in this repo; no new test framework.

## 4. Security-update review

- **Confirmed CVE affecting a tracked branch:** review and, if it applies to how ReDevOps runs DeerFlow, merge the fix into the fork's tracked branch within the same business day — don't wait for the weekly sync.
- **Routine upstream patch (no CVE):** let the weekly sync (§1) pick it up; review the PR it opens (once the branch has diverged) in the following week's normal review.
- Either way: the sync workflow or a manual `git fetch upstream` is how you *see* the fix; a human always decides whether and when it merges. No automatic merge of security content.

This is sized for a small team — one reviewer, no formal CAB. If that changes, revise this section, not the rest of the document.

## 5. Release promotion

The one path from "a commit exists on the fork's tracked branch" to "the rest of the ReDevOps stack points at it":

1. Commit lands on `2.1.x-dev` (or `main`) on the fork, via a reviewed PR (ReDevOps-local change) or the sync workflow's fast-forward/PR (upstream change).
2. §3's CI runs on that commit and is green.
3. A container image is built from that exact commit and tagged with its SHA (or a digest derived from it) — not `latest`, not the branch name.
4. Each dependent repo's pin (§2) is bumped to that SHA/digest in its own PR, reviewed and merged there.

Step 4 is the only place a pin changes. Nothing upstream of it should ever write to a dependent repo directly.
