# Detector interface: data sources, inference options, parameter lifetimes

**Status:** Not started. Awaiting the user's approval of this plan; the
[open questions](overview.md#open-questions) list the choices still theirs.

The detectors' methods take spike data as separate positional arrays, with one
copy of every method for sorted spikes and another for clusterless spikes.
`predict` and `estimate_parameters` expose how inference runs as flags that
constrain each other, and only seven of the constructor parameters a fit depends
on are checked for changes afterwards. This plan gives the detectors one data
argument and one set of data methods, one argument for how inference runs, and
one declaration of when each constructor parameter is read. The first two change
the public call form, so they belong before 0.7.0 is released.

## How to work with this plan

- Each phase file is a brief for one pull request: the goal, what "done" means,
  the constraints that are fixed and why, the code to read, a suggested approach
  and the validation. The approach is a suggestion. Deviate when the code calls
  for it and say why in the pull request. The fixed constraints and the *hard*
  thresholds are not suggestions: if one cannot be met, stop and report.
- Thresholds are marked *hard* (invariants, regressions, exact equality) or
  *target* (a miss is reported with its numbers, not loosened and not hidden).
- Line numbers were checked at commit `0b88f984` (main, 2026-11-06). They drift;
  find the code by the symbol name given with each one.
- Code blocks are marked *interface sketch*: they show the shape of an
  interface, not text to paste.
- The repository's `CLAUDE.md` rules apply and are not repeated here (skills to
  use, `uv run`, asking before committing to main or pushing, golden data).
  Run targeted test selections while working; the full suite takes about
  20 minutes and CI covers it.
- Code, tests and docstrings written from this plan do not mention the plan,
  its phases or these files.
- When a phase merges, update **Status** above.

## Reading order

Load only what the task needs:

1. Working a phase: open its phase file. It links what it depends on.
2. The data classes, the family contract and the stored names that must not
   change: [shared-contracts.md](shared-contracts.md).
3. Scope, risks, rollout and open questions: [overview.md](overview.md).
4. Spyglass call sites, call-site counts and what 0.6.9 released:
   [appendix.md](appendix.md).

## Files

- [overview.md](overview.md): goals, non-goals, integration points, risks,
  rollout, open questions.
- [shared-contracts.md](shared-contracts.md): data classes, family contract,
  rules for later sources, stored artifacts.
- Phases, each one pull request, in order:
  - [phase-1-family-seam.md](phase-1-family-seam.md): route every
    family-specific step through one family object. No public change.
  - [phase-2-data-argument.md](phase-2-data-argument.md): one data argument,
    data methods written once. Public change, with the Spyglass update.
  - [phase-3-inference-argument.md](phase-3-inference-argument.md): one
    `inference` argument in place of nine of `predict`'s execution flags and
    five of `estimate_parameters`'. Public change, with the Spyglass update.
  - [phase-4-parameter-lifetimes.md](phase-4-parameter-lifetimes.md): declare
    when each constructor parameter is read; detect fit-time changes for all of
    them. Independent of phases 1 to 3.
- [appendix.md](appendix.md): Spyglass references at a fixed commit, call-site
  counts, released options.

## Decision log

Decisions are the user's unless marked "planner's call", which the user has not
confirmed.

| Date | Decision |
| --- | --- |
| 2026-11-05 | Possible future data: calcium traces and LFP. LFP could enter as its own likelihood, and later extensions might combine several data sources. None of this is built now; the design must not block it. |
| 2026-11-05 | Changing the clusterless call form in 0.7.0 is acceptable provided Spyglass PR #1618 can follow. Checked at Spyglass `412c4fc61`: it can (six call sites, see appendix). |
| 2026-11-05 | Tracking gaps at decode time warn only; they are not decoded as missing. So no `valid_position_intervals` argument is added to `predict`. Superseded on 2026-11-06, below. |
| 2026-11-05 | Saved-model format: no new stored fields for change detection, no format-version bump, no migration. Additions carry defaults. |
| 2026-11-05 | `no_spike_rate` stays a decode-time setting for now. |
| 2026-11-05 | No tests or rules that force an edit on routine future changes (a new parameter, field or algorithm). |
| 2026-11-03 | Models, results and parameter rows that Spyglass users stored with earlier versions must keep loading. A waiver of backward compatibility covers call signatures only. |
| 2026-11-05 | Planner's call: the data argument is named `neural_data`. See open question 1. |
| 2026-11-05 | Planner's call: a plain sequence of spike-time arrays passed to a sorted-spikes detector means sorted spikes. |
| 2026-11-05 | Planner's call: `inference` takes `InMemory` or `Checkpointed`; `estimate_parameters` drops `transition_representation`; `cache_likelihood` stays a keyword. See open questions 2 and 3. |
| 2026-11-05 | Planner's call: `is_missing` and `row_slice` become keyword-only in `compute_log_likelihood`, so an old clusterless positional call fails instead of binding the features to `is_missing`. |
| 2026-11-05 | Planner's call: `frozen_discrete_transition_rows` is fit-time, because `fit` snapshots the frozen rows and `save_model` stores the snapshot. |
| 2026-11-05 | An independent reader checked the first draft against the code. Its findings are folded in: phase 1's completion list, detectors without a family, the EM mapping for `Checkpointed()` with no field set, the Spyglass translation for `n_chunks=1`, and the count of tests that pass `transition_representation` to `estimate_parameters`. |
| 2026-11-05 | Planner's call: decode-time parameter values are written to the result attributes. See open question 4. |
| 2026-11-05 | Planner's call: two items from the design discussion are left out, with reasons in overview.md ("Deliberately not in this plan"): removing the engines' repeated input checks, and giving every HMM driver one transition input. |
| 2026-11-06 | `valid_position_intervals` is an argument of `predict`, `most_likely_sequence` and `compute_log_likelihood` (#208), and a bin outside the intervals decodes as missing. This supersedes the 2026-11-05 decision on tracking gaps. |
| 2026-11-06 | The plan was checked against the code again at `0b88f984`, after #208, #209, #211 and #212. Its findings are folded in: phase 1's move of `_validated_inputs`, the rules `predict` keeps, `fit_encoding_model` in phase 4, NaN-aware comparison, the API reference, rule 4 of the later sources, and the removal from phase 4 of the EM report clearing that main already does. |
| 2026-11-06 | Planner's call: Spyglass rows choose a strategy with a plain `inference` dict in `decoding_kwargs`, because DataJoint cannot pack a dataclass. Spyglass's adapter builds the object and refuses the flat keys it would otherwise drop. See phase 3. |
| 2026-11-06 | Planner's call: a replaced `compute_log_likelihood` receives the family's data object, whichever form the caller passed. See phase 2. |
