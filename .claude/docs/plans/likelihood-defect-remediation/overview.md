# Overview

## Goals

1. Stop the four defects that change a decoded posterior with no test failing.
2. Remove numerical failure modes producing NaN or impossible values (negative
   squared distances, mixture weights summing to 24, means collapsing to 0 under
   a global weight rescale).
3. Establish one time vocabulary ([C3](shared-contracts.md#c3--time-vocabulary))
   and calibrate Poisson intensities to real bin durations.
4. Take the measured performance wins that require no algorithmic change.

## Non-goals

- Retiring the duplicate `clusterless_kde.py` / `clusterless_kde_log.py`
  implementations — blocked on the known ~3e-2 log-vs-prob golden parity gap.
- A typed/versioned encoding-model schema replacing the `dict` +
  `**encoding_model` splat. Phase 6d adds a single unit marker, not a schema.
- Clearing the 190 mypy errors. Phases must not add errors; a sweep is separate.
- A backend-owned EM damping rebuild protocol. Phase 5 rejects damping where it
  cannot be supported.

## Architecture map

```
detector.fit()
  └─ fit_encoding_model()          models/base.py:2900 (clusterless), :3890 (sorted)
     ├─ is_group = is_training & is_encoding & is_environment
     ├─ group spikes               _get_group_spike_data / _get_group_spikes
     └─ registry fit fn            likelihoods/__init__.py:41 (sorted), :59 (clusterless)

detector.predict()
  └─ compute_log_likelihood()      models/base.py:3047 (clusterless), :4011 (sorted)
     └─ core chunked HMM
        ├─ chunked_filter_smoother                      core.py:490
        └─ chunked_filter_smoother_covariate_dependent   core.py:1081
```

**The two core paths are stationary vs. covariate-dependent transitions — not
sorted vs. clusterless.** The first draft of this plan mislabelled them, which
would have let a test suite exercise one path twice and miss the other. Any phase
touching `core.py` must cover both, and both detector types route through
whichever path their transition model selects.

## Cross-cutting risks

| Risk | Mitigation |
|---|---|
| Phase 6a/6b re-baseline every golden, masking a regression from an earlier phase. | Phases 1–5 ship first, each with its own parity test. |
| Phase 1 exposes all-zero-exposure groups whose guards used to live in phase 4. | Zero-exposure handling moved **into** phase 1, so it is independently shippable. |
| Phases 2 and 4 both move GMM outputs; a combined diff is unattributable. | Separate PRs. Phase 2's diff must be confined to bins where the pre-fix code clamped. |
| Fixing float32 cancellation changes well-conditioned results too, via summation order. | Assert invariants (Mahalanobis ≥ 0) plus `rtol=1e-5` parity on well-conditioned fixtures, not bit-exactness. |
| A phase's acceptance criteria are never run against its own proposed fix. | Every phase carries a **Falsification** line to execute before implementing. |

## Deferred with triggers

Recorded so they are not silently dropped.

| Item | Trigger to revisit |
|---|---|
| Duration-calibrated continuous/discrete transitions, so nonuniform detector bins become valid. Currently `core.py:643` applies one transition per row regardless of duration; phase 6c restricts detectors to uniform bins instead. | When a user needs nonuniform decoding, or when variable-`dt` event-based decoding is scoped. |
| Streaming the `(n_encoding_spikes, n_position_bins)` kernel in the linear `clusterless_kde` path. `clusterless_kde_log` already tiles; the right fix is to retire the duplicate. | When the log-vs-prob golden parity gap closes. |
| `GaussianMixture.score_samples` computes discarded responsibilities (`gmm.py:1068`). | Next GMM cleanup. |
| Typed/versioned encoding-model schema. | When a third consumer of the encoding dict appears, or after phase 6d's unit marker proves insufficient. |

## Environment

All commands run through `uv run`. Baseline at planning time:

```
uv run pytest src/non_local_detector/tests/likelihoods -q
# 376 passed, 2 skipped, 23 warnings in 173s
uvx ruff check src/non_local_detector/likelihoods/     # All checks passed
uv run mypy src/non_local_detector/likelihoods/        # 190 errors in 11 files
```

The first two must not regress. The mypy count is a ceiling, not a target.

## Open questions

**Blocking.**

- **C1's degeneracy policy is withdrawn.** "Floor only `-inf`" is non-monotonic:
  an impossible observation scores −34.54 while a 1e-44 one scores −101.31.
  Monotonicity requires flooring both or neither. See
  [shared-contracts.md](shared-contracts.md#c1--degeneracy-policy-for-log-intensities)
  for the three options. Phase 2 cannot proceed until this is chosen.
- **C3's `sample_cell_durations` undercounts exposure** by `(N-1)/N` and gives a
  single sample zero exposure. Endpoint half-cells must be extrapolated,
  acquisition bounds supplied, or short input rejected. A policy for long
  timestamp gaps (exposure vs. missing data) is also undefined. Phases 6a and 6b
  cannot proceed until this is settled.

**Delegated judgement calls.**

- Phase 4: if a fixture starts warning about collapsed components after the
  centred-difference rewrite, raise the fixture's `reg_covar` rather than
  lowering the floor.
- Phase 5: whether *any* backend supports EM damping is an audit result, not an
  assumption — the phase may legitimately conclude the supported set is empty.

## Found during review, not yet scheduled

- **`save_model` is broken for every GLM detector.** Reproduced end-to-end:
  `sorted_spikes_kde` saves fine, `sorted_spikes_glm` raises
  `NotImplementedError: Sorry, pickling not yet supported` because the encoding
  dict holds a Patsy `DesignInfo` (`sorted_spikes_glm.py:316`) and `save_model`
  uses stdlib `pickle` (`models/base.py:2337`). This is a live user-facing bug
  independent of this plan and should be fixed on its own, not folded into
  phase 6d. Remedies: reconstruct `DesignInfo` from the formula and knots at load
  time, or store the spline basis numerically instead of the Patsy object.
