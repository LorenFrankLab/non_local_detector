> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 5 — Window ownership and validation gaps

## Contracts referenced

- [C2 — Exposure ownership and spike weights](shared-contracts.md#c2--exposure-ownership-and-spike-weights)

---

## Defect 1 — Group windows overlap and contradict their own comment

Both helpers widen each contiguous run by a whole `time_delta` while the comment
says half (`models/base.py:3812-3815` sorted, `:2851-2854` clusterless), and
`time_delta` is a single global first difference (`:3804`, `:2838`), wrong for
non-uniform timestamps.

Reproduced — mask `[T,T,F,T,T]` at times 0…4, spike at t=2:

```
run 1: window [-1.0, 2.0] contains spike@2.0 -> True
run 2: window [ 2.0, 5.0] contains spike@2.0 -> True
```

### Ownership rule

Per [C2](shared-contracts.md#c2--exposure-ownership-and-spike-weights), the
interpolated weight is canonical and the window is a coarse pre-filter that must
(a) never discard a spike carrying non-zero weight and (b) never let two runs
claim the same spike.

For the example above the correct outcome is **zero runs** own the spike at
`t=2`: sample 2 is excluded, contributes no exposure, and a spike attributed to
it would have no denominator. An earlier draft's acceptance test demanded
"exactly one", which contradicts the contract — the test was wrong, not the code.

### Fix

Half a *local* interval on each side, from the run's own boundary, with the start
inclusive and the stop exclusive so abutting runs cannot both claim a boundary:

```python
                run_inds = np.flatnonzero(group_labels == group)
                start_ind, stop_ind = run_inds[0], run_inds[-1]
                start_time = position_time[start_ind]
                stop_time = position_time[stop_ind]
                # Half a sample interval on each side, measured at this run's own
                # boundary so non-uniform timestamps stay correct. Half (not a
                # full delta) keeps adjacent runs from overlapping; the window is
                # only a pre-filter, the interpolated weight does the real work.
                if start_ind > 0:
                    start_time -= 0.5 * (start_time - position_time[start_ind - 1])
                if stop_ind < position_time.size - 1:
                    stop_time += 0.5 * (position_time[stop_ind + 1] - stop_time)
                is_valid_spike_time = (
                    (neuron_spike_times >= start_time)
                    & (neuron_spike_times < stop_time)
                )
```

The final run of the array uses `<=` for its `stop_time` so the last spike is not
lost — mirror the right-closed final cell in
[C3](shared-contracts.md#c3--time-vocabulary). A single-sample run at an array end
gets a zero-width window, which is correct: there is no interval to attribute.

Exact-midpoint ties now resolve to the earlier run only, because `stop` is
exclusive.

### Also — `strict=False` hides population mismatches

`models/base.py:2842-2843` zips spike times against waveform features with
`strict=False`, truncating silently *before* the backend validators run. Do not
merely flip it to `strict=True` — that raises a bare `ValueError` and still misses
per-electrode spike-count/feature-row mismatches. Call the package validators
first:

```python
        validate_population_lengths(
            "electrode",
            spike_times=spike_times,
            spike_waveform_features=spike_waveform_features,
        )
        for electrode, (times, feats) in enumerate(
            zip(spike_times, spike_waveform_features, strict=True)
        ):
            _validate_spike_feature_pair(times, feats, electrode)
```

`_validate_spike_feature_pair` lives in `clusterless_gmm.py:196-223`; move it to
`common.py` rather than importing across sibling modules.

---

## Defect 2 — EM damping silently no-ops for most backends

`_apply_encoding_damping` (`models/base.py:1787-1810`) blends only `place_fields`
and recomputes `no_spike_part_log_likelihood`. So:

- clusterless dicts have no `place_fields` — damping does nothing;
- diffusion and MRF keep a stale `interior_log_place_fields`, which the predictor
  uses directly as the matmul operand
  (`sorted_spikes_diffusion.py:682-689`), so the damped fields never reach decode;
- GLM `coefficients` are not blended, so its local path (recomputed from
  `coefficients`, `sorted_spikes_glm.py:461`) and non-local path (from
  `place_fields`) come from different fits.

### Fix — reject where unsupported, and reject *before* refitting

Validation currently sits at `models/base.py:1938-1941`, but `estimate_parameters`
has already refit and mutated the detector by the time damping is applied. Move
the check ahead of the first `fit` call in the EM loop.

**Audit result (decided):** no registered backend can safely use the current
place-field-only blend. `sorted_spikes_kde`'s local branch reads `marginal_models`
and `mean_rates`, diffusion/MRF read `interior_log_place_fields`, GLM reads
`coefficients`, and clusterless dicts have no `place_fields` at all. Reject
unconditionally, before the concrete wrappers' first `fit` call
(`base.py:3581`, `base.py:4470`) rather than mid-EM. Original reasoning: `sorted_spikes_kde`'s local branch uses
`marginal_models` and `mean_rates` (`sorted_spikes_kde.py:337-377`), not
`place_fields` — so a place-field-only blend leaves its local path inconsistent
too. The default observation model includes a local state, so the honest
conclusion may be that **no backend supports damping**, in which case reject it
unconditionally and say so:

```python
        if encoding_update_damping > 0.0:
            raise ValidationError(
                "encoding_update_damping is not supported",
                expected="encoding_update_damping=0.0",
                got=f"{encoding_update_damping}",
                hint=(
                    "Encoding models carry state derived from the fit "
                    "(KDE marginal models, diffusion/MRF interior log fields, GLM "
                    "coefficients) that a place-field blend would leave "
                    "inconsistent with the damped fields."
                ),
            )
```

Do not assume the set is non-empty to keep a feature alive. Record the audit
result in the PR description.

CHANGELOG `Changed`: `encoding_update_damping` now raises for backends that
cannot support it, rather than silently having no effect (clusterless) or
producing an inconsistent model (diffusion, MRF, GLM).

---

## Defect 3 — `position=None` crashes the clusterless GMM predictor

`models/base.py:3110-3129` permits `position=None`; the GMM predictor
dereferences it unconditionally at `clusterless_gmm.py:639`. Reproduced:
`AttributeError: 'NoneType' object has no attribute 'ndim'`.

```python
    # position is only needed for the local path; the detector permits None when
    # no observation model is local.
    if position is not None:
        position = _as_jnp(position if position.ndim > 1 else position[:, None])
    elif is_local:
        raise ValidationError(
            "position is required for local clusterless GMM decoding",
            expected="array with shape (n_time_position, n_position_dims)",
            got="None",
        )
```

`clusterless_diffusion` already handles this correctly — it touches `position`
only under `is_local` and validates there (`clusterless_diffusion.py:542-547`).
Use it as the reference. Audit `clusterless_kde` and `clusterless_kde_log` for
the same pattern; add the guard only where the non-local path actually
dereferences `position`.

---

## Defect 4 — sorted diffusion and MRF skip population validation

Neither module imports `validate_population_lengths` (confirmed by grep), yet the
predictor builds a count matrix from `spike_times`
(`sorted_spikes_diffusion.py:559-567`) and matmuls it against
`interior_log_place_fields` (`:689`). Add at the top of
`predict_sorted_spikes_diffusion_log_likelihood` (`:649`, before the `is_local`
branch):

```python
    validate_population_lengths(
        "neuron",
        spike_times=spike_times,
        place_fields=place_fields,
    )
```

including `interior_log_place_fields` when not `None`.
`predict_sorted_spikes_mrf_log_likelihood` *is* this same function (see
`sorted_spikes_diffusion.py:591-598`), so one edit covers both.

---

## Validation

| Test | Asserts | File |
|---|---|---|
| `test_excluded_sample_spike_owned_by_no_run` | Mask `[T,T,F,T,T]`, spike at t=2 → claimed by **zero** runs, per C2. | `tests/models/test_group_spikes.py` |
| `test_adjacent_runs_do_not_overlap` | For any mask, the windows are pairwise disjoint. | same |
| `test_window_covers_nonzero_weight_spikes` | Every spike whose interpolated weight is > 0 falls inside some window — the pre-filter never discards real exposure. | same |
| `test_group_windows_nonuniform_timestamps` | With jittered `position_time`, each run extends by half its own boundary interval. | same |
| `test_group_population_mismatch_raises_validation_error` | 3 spike lists vs 2 feature lists raises `ValidationError` (not bare `ValueError`), before any indexing. | same |
| `test_per_electrode_row_mismatch_raises` | Matching list lengths but mismatched rows within one electrode still raises. | same |
| `test_damping_rejected` | `encoding_update_damping=0.5` raises `ValidationError` naming the reason. | `tests/models/test_em.py` |
| `test_damping_rejection_leaves_state_unchanged` | After the raise, the detector's `encoding_model_` is identical to before — the check runs before any refit. | same |
| `test_gmm_predict_without_position` | Non-local GMM with `position=None` returns a finite `(n_time, n_bins)` array. | `tests/likelihoods/test_gmm.py` |
| `test_gmm_local_without_position_raises` | `is_local=True, position=None` raises `ValidationError`. | same |
| `test_diffusion_population_mismatch_raises` | Fewer spike lists than fitted neurons raises rather than broadcasting. | `test_sorted_spikes_diffusion.py` |
| `test_mrf_population_mismatch_raises` | Same via the MRF entry point. | `test_sorted_spikes_mrf.py` |

```bash
uv run pytest src/non_local_detector/tests -q
uvx ruff check src/non_local_detector/
```

Goldens should be unchanged — defect 1 changes which spikes a *fragmented* mask
claims, and golden fixtures use contiguous full coverage. If one moves, that
fixture has a fragmented mask; investigate before assuming the new value is right.

## Review

Dispatch `code-reviewer`. Ask for an independent judgement on the damping support
set — specifically whether `sorted_spikes_kde`'s local decode path is consistent
after a place-field-only blend, since that determines whether the set is empty.
