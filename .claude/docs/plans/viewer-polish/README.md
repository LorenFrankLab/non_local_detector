<!-- markdownlint-disable MD024 MD004 MD050 MD031 -->

# Interactive Decoder Viewer — Post-v1 Polish Plan

Branch-local follow-up plan for the `interactive-decoder-viewer`
branch. Complements [TASKS.md](../../../../TASKS.md) (v1
implementation tasks) with the polish work identified by the
multi-agent review on 2026-05-09.

**This directory is temporary** — delete the whole
`.claude/docs/plans/viewer-polish/` tree before merging the branch.

**Source review:** the review pass that produced this plan covered
the branch vs `origin/main` (74 commits, ~22k LOC), parity vs
`~/Documents/GitHub/statespacecheck-paper-viewer/`, and a
Heer-style UX audit. Key findings the user explicitly approved are
encoded as tasks in the per-phase files; deferred items are listed
at the end of this README with rationale.

---

## Phase index

| Phase | File | Effort | Skill |
| --- | --- | --- | --- |
| 1 — Pre-merge blockers | [phase-1-blockers.md](phase-1-blockers.md) | ~1 h | none (tactical fixes) |
| 2 — UX correctness + discoverability | [phase-2-ux-correctness.md](phase-2-ux-correctness.md) | ~5 h | `scientific-tdd` |
| 3 — Interaction feel + parity | [phase-3-interaction-feel.md](phase-3-interaction-feel.md) | ~4 h | `scientific-tdd` |
| 4 — Architecture hygiene | [phase-4-architecture-hygiene.md](phase-4-architecture-hygiene.md) | ~4 h | `safe-refactoring` |
| 5 — Direct zarr data source | [phase-5-direct-zarr.md](phase-5-direct-zarr.md) | ~12 h / 1.5 d | `safe-refactoring` + `scientific-tdd` |
| 6 — Library-native API + sidecar docs | [phase-6-library-api.md](phase-6-library-api.md) | ~4 h | `scientific-tdd` |

**Total:** ~30 hours / 3–4 focused days.

**Phase ordering** is sequential on a single branch:
1 → 2 → 3 → 4 → 5. Hard dependencies in the graph:

- **Phase 4 must precede Phase 5** (Phase 5 relies on the cleaner
  Protocol from Phase 4).
- **Phase 6 splits**: tasks **6.1–6.4** require only Phase 1 (they
  touch the public `launch_qt` API, docstrings, the notebook
  quickstart, and a new devtool subcommand — surfaces that don't
  conflict with the rest of the polish work). Task **6.5** edits
  `_build_controls_bar` and must come after Phase 2, which also
  edits that function heavily. So 6.5 is `≥ Phase 2`.

The rest of the order (2 → 3 → 4) is the recommended review
sequence, not a hard dependency. See
[phase-6-library-api.md](phase-6-library-api.md)
§Dependencies for the per-task split.

---

## Working with this plan in Claude Code

A fresh Claude session picking up this plan should:

1. **Use `uv run` for every Python invocation** (the project's hooks
   enforce this; bare `python` will be rejected).
2. **Never commit, push, or update snapshots without explicit user
   approval.** Per CLAUDE.md's Guided Autonomy Boundaries: code
   changes / test runs / quality checks are auto-allowed; commits,
   pushes, snapshot updates, and tolerance changes require
   permission. If the full sweep surfaces a snapshot diff,
   **stop and ask** — none of the tasks in this plan should
   produce one.
3. **Announce skill usage.** When a phase lists a skill (e.g.
   `scientific-tdd`), say "I'm using the scientific-tdd skill to
   write the failing test first, then implement to pass" before
   starting.
4. **Use `TodoWrite` within a session** to track which checkboxes
   are in flight; mark each `[ ]` → `[x]` in the relevant phase
   file *only* when the task's verification has actually passed.
5. **Stop at phase boundaries — wait for user review.** Do not
   start Phase N+1 until the user has reviewed Phase N and
   explicitly given the go-ahead. This applies *even in auto-mode*:
   the phase boundary is a hard checkpoint, not a hint. When a
   phase's "Done when" gate is satisfied:

   1. Run the full sweep one more time and post the result.
   2. Summarize what landed (file paths + LOC + test count delta).
   3. Surface every `[manual]` item from the phase to the user.
   4. **Stop and wait.** Do not begin the next phase, do not
      commit (commits require explicit approval per item 2), do
      not push, until the user says "go" / "proceed" /
      "next phase".

   Phase ordering is a dependency graph, not a schedule — the
   user controls the pace.

6. **All work lands on `interactive-decoder-viewer`.** No feature
   branches, no per-phase PRs. Each phase is a commit cluster on
   the existing branch; the whole branch merges to `main` once at
   the end via a single PR.
7. **Resumability.** If a session ends mid-phase, leave the
   in-progress state notes in
   [SCRATCHPAD.md](../../../../SCRATCHPAD.md) (used the same way
   Milestone-3 work used it). The next session reads SCRATCHPAD →
   this README → the active phase file → the relevant code, in
   that order.
8. **Manual-smoke items** are tagged `[manual]` inline. These
   require a human at a real GUI (touchpad gestures, tooltip hover,
   live session inspection) — Claude Code with offscreen Qt cannot
   verify them. Mark the task implementation done, then **explicitly
   surface the manual smoke to the user** before considering the
   task closed.
9. **Phase ordering.** Sequential on a single branch (see Phase
   index above). Don't run phases "in parallel" — there's only
   one branch.
10. **Commit-message convention.** Recent commits on this branch
    end with `Co-Authored-By: Claude Opus 4.7 (1M context)
    <noreply@anthropic.com>`. Match that footer for consistency
    when (with user approval) committing.
11. **File paths and line numbers.** Paths are relative to the
    repo root; line numbers are advisory and may drift as work
    lands. **If a cited line number is wrong:** grep for the
    function name or the exact comment text quoted in the task.
    The task body is the source of truth.
12. **Auto-mode awareness.** Most tasks are safe for autonomous
    execution. Tasks that aren't are flagged `[manual]` or
    `[needs design review]`. Do not skip those gates.

---

## Per-phase verification

Run before handing back to the user for review (full sweep is
~3 min — surface the wait time to the user, do not silently block):

```bash
uv run ruff check src/ && \
  uv run ruff format --check src/ && \
  uv run pytest src/non_local_detector/tests/interactive/
```

Baseline at phase start (count the passing tests) ≥ baseline at
phase end. No regressions.

---

## Pre-merge checklist (final)

Before opening the merge PR for the whole
`interactive-decoder-viewer` branch (this checklist runs *once* at
the end, not per phase):

- [ ] All 6 phases complete (or explicitly deferred and the phase
  file marked accordingly)
- [ ] `uv run ruff check src/` — clean
- [ ] `uv run ruff format --check src/` — clean
- [ ] `uv run mypy src/non_local_detector/` — no new errors vs `main`
- [ ] `uv run pytest src/non_local_detector/tests/interactive/` —
  all pass (target: ≥260 tests after this work; was 252 at start)
- [ ] **[manual]** Real-session smoke (1+ hour recording):
  resize/scroll feels fluid, no churn, autoscroll works on first
  press, help dialog opens, slice legend tracks overlay mode.
  *User-driven; Claude must surface this to the user and wait
  for confirmation rather than auto-checking.*
- [ ] **[manual]** Touchpad smoke: wheel-resize is fluid (not
  steppy). User-driven.
- [ ] Help dialog screenshot in the PR description (UX evidence)
- [ ] Latency benchmark numbers in the PR description (perf
  evidence for Phase 5; recorded from
  `scripts/bench_data_source.py` per Task 5.5)
- [ ] `.claude/docs/plans/viewer-polish/` directory deleted in
  the merge commit (matches the v1
  [TASKS.md](../../../../TASKS.md) "delete before merging"
  convention)

---

## Deferred — agreed but out of this plan's scope

The following review findings are real but the user explicitly
chose not to include them. Listed here so a future reviewer doesn't
re-flag them:

- **`AUTOSCROLL_DEFAULT_SPEED = 1.0`** — current 0.05× supports
  frame-by-frame inspection which is the typical neuroscience use
  case. Revisit if feedback says playback "doesn't seem to work".
- **Heatmap colorbars / vmax scale labels** — the silent
  posterior/likelihood vmax mismatch (0.25 vs 1.0) is real but
  adding pyqtgraph `ColorBarItem` is meaningful layout work; defer
  until users actually misread scales.
- **Slice y-axis hard-coded `[-0.02, 1.05]`** — peak-normalized
  rendering keeps visual height consistent across overlay modes.
  Revisit if a user reports specific misperception.
- **Pinning gesture affordance / visibility** — pin vocabulary
  (yellow header + gold line + ★) is internally consistent but
  hidden until the user clicks. The `?` help dialog from Phase 2
  will cover discoverability; visual affordances are a follow-up
  if still confusing post-help.
- **`_validate_time_grid_alignment` perf memoization** — the
  original draft suggested an `id()`-keyed cache; deferred because
  unsafe given mutable RunBundle internals. See Phase 4's
  task 4.5 deferred-with-rationale block.

---

## References

- [TASKS.md](../../../../TASKS.md) — v1 implementation plan
  (predecessor to this plan)
- [SCRATCHPAD.md](../../../../SCRATCHPAD.md) — inline working
  notes during implementation
- [CLAUDE.md](../../../../CLAUDE.md) — project conventions, skills,
  numerical-validation standards
- [docs/plans/2026-05-06-interactive-decoder-viewer.md](../../../../docs/plans/2026-05-06-interactive-decoder-viewer.md) —
  original viewer design plan
- `~/Documents/GitHub/statespacecheck-paper-viewer/` — upstream
  reference viewer (parity baseline)
- Review pass that produced this plan: 2026-05-09 conversation
  thread (code-reviewer + Explore + ux-reviewer agents,
  synthesized)
