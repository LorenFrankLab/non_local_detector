> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 6d — Detect saved models with the old rate units

Ships with or immediately after 6b.

## Problem

Encoding models are persisted by pickling the detector
(`models/base.py:2327` `save_model`). A model fitted before phase 6b stores rates
in spikes-per-position-sample; a post-6b predictor interprets them as Hz and
scales by `dt`. Nothing detects this — the decode runs and produces a plausible,
silently miscalibrated posterior.

For a 500 Hz grid the error is a factor of 500.

A second, noisier failure: phase 6b adds `encoding_exposure_duration` to every
encoding dict, and predictors are called as `predict_fn(**encoding_model)`
(`models/base.py:3174`). A model pickled *after* 6b, loaded by code whose
predictor lacks the parameter, raises `TypeError` — and several predictors are
strict about their signature (e.g. `sorted_spikes_kde.py:266-281` enumerates its
parameters with no `**kwargs`).

## Decision

No backwards-compatibility window is required (user-confirmed), so: **detect and
reject with a clear error.** Do not migrate old models — the conversion factor
depends on the `position_time` grid used at fit, which is not stored.

## Tasks

### 1. Stamp the units

Every encoding fit adds a unit marker alongside `encoding_exposure_duration`:

```python
        "encoding_rate_units": "hz",
```

Define the constant once in `likelihoods/common.py`:

```python
# Bump when the meaning of a stored rate or field changes in a way that a
# previously-pickled encoding model cannot be reinterpreted from. Predictors
# reject any model whose marker does not match.
ENCODING_RATE_UNITS = "hz"
```

### 2. Reject at predict, once

Check in `compute_log_likelihood` (both implementations) rather than in each of
the 13 predictors:

```python
            stored_units = self.encoding_model_[likelihood_name[:2]].get(
                "encoding_rate_units"
            )
            if stored_units != ENCODING_RATE_UNITS:
                raise ValidationError(
                    "this fitted model predates the Hz rate convention and "
                    "cannot be decoded by this version",
                    expected=f"encoding_rate_units={ENCODING_RATE_UNITS!r}",
                    got=repr(stored_units),
                    hint=(
                        "Refit the encoding model. Old models stored rates per "
                        "position sample; converting them needs the original "
                        "position sampling grid, which is not saved."
                    ),
                )
```

A model saved before this phase has no such key, so `.get` returns `None` and the
check fires — which is the desired behaviour, and is why the check reads the key
rather than a version integer.

### 3. Make predictors tolerant of added keys

`predict_fn(**encoding_model)` means every new dict key is a potential
`TypeError`. Add `**_encoding_extras: object` to the predictors that enumerate
their parameters — `sorted_spikes_diffusion.py:587` already does exactly this and
documents why; copy that pattern and its docstring wording to
`sorted_spikes_kde.py`, `sorted_spikes_glm.py`, `clusterless_kde.py`, and
`clusterless_kde_log.py`.

This also removes a class of future breakage whenever a fit gains a field.

### 4. Documentation

CHANGELOG `Changed` (breaking): models fitted with earlier versions must be
refit; loading one and calling `predict` raises with an explanatory message
rather than silently miscalibrating. Add a short "Upgrading" note to the README
covering refit.

Check whether `save_model` / `load_model` docstrings promise cross-version
compatibility; if so, correct them.

## Validation

| Test | Asserts |
|---|---|
| `test_old_encoding_model_rejected` | An encoding dict with the units key removed raises `ValidationError` naming refit as the remedy. |
| `test_current_model_roundtrips` | Fit → `save_model` → `load_model` → `predict` works and gives identical results. |
| `test_predictors_tolerate_extra_keys` | Every registry predictor accepts an encoding dict with an unknown extra key. Parametrized over both registries — this is what catches a predictor that was missed in task 3. |
| `test_units_marker_present` | Every registry fit returns a dict containing `encoding_rate_units == "hz"`. Parametrized over both registries. |

```bash
uv run pytest src/non_local_detector/tests/models src/non_local_detector/tests/likelihoods -q
```

No golden impact — goldens are refit, not loaded from pickles. Confirm that
assumption by grepping the golden fixtures for `load_model`; if any loads a
pickle, it must be regenerated and that needs approval.

## Review

Dispatch `code-reviewer`. Ask whether any path reaches a predictor without
passing the units check — in particular `estimate_parameters`, which refits
mid-loop and may bypass `compute_log_likelihood`'s entry.
