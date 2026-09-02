> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 6c — Detector requires uniform bins

Small phase, but it must ship with 6a/6b: those make nonuniform edges
*expressible*, and this records that the HMM cannot honour them.

## Contracts referenced

- [C3 — Time vocabulary](shared-contracts.md#c3--time-vocabulary), uniform-bin
  restriction.

## Problem

`core.py` applies one transition matrix per row regardless of that row's
duration — the `_filter_internal` call at `core.py:643` and its
covariate-dependent twin advance the state by one transition per observation. A
row spanning 1 s therefore gets the same continuous-position diffusion and the
same discrete-state transition probability as a row spanning 1 ms.

Phase 6b makes the *likelihood* correct for nonuniform bins. That is not enough:
the posterior is a product of likelihood and transition terms, so scaling only
the likelihood leaves the result time-miscalibrated in a way that looks
plausible.

An earlier draft advertised a `test_nonuniform_bins_supported` acceptance test.
That over-promised — the likelihood would pass it while the posterior remained
wrong.

## Decision

Detector-level `predict` **requires uniform edges** and raises otherwise.
Nonuniform edges remain valid on the direct `predict_*_log_likelihood` API, where
no transition model is involved and the likelihood alone is the answer.

Duration-calibrated transitions are recorded as a deferred item with a trigger in
[overview.md](overview.md#deferred-with-triggers).

## Tasks

### 1. Validate at the detector boundary

In both `predict` implementations (`models/base.py:3239`, `:4201`), after edge
validation and before computing anything:

```python
        durations = np.diff(time)
        if not np.allclose(durations, durations[0], rtol=1e-6, atol=0.0):
            raise ValidationError(
                "detector decoding requires uniform time bins",
                expected="evenly spaced time edges",
                got=(
                    f"bin durations spanning [{durations.min():.6g}, "
                    f"{durations.max():.6g}]"
                ),
                hint=(
                    "The HMM applies one transition per bin regardless of its "
                    "duration, so a nonuniform grid gives a time-miscalibrated "
                    "posterior even though the likelihood is correct. Use "
                    "detector.calculate_time_bins(), or call "
                    "predict_*_log_likelihood directly if you only need the "
                    "likelihood."
                ),
            )
```

`rtol=1e-6` tolerates float accumulation in `arange`-built grids while rejecting
genuinely irregular ones. Confirm `calculate_time_bins` output passes.

### 2. Document the split

Add a note to both `predict` docstrings and to the module docstring of
`likelihoods/__init__.py`: nonuniform edges are supported by the likelihood API
and rejected by the detector, with the reason.

### 3. CHANGELOG

`Changed`: `predict` now raises for nonuniform time bins. The likelihood
functions accept them.

## Validation

| Test | Asserts |
|---|---|
| `test_uniform_bins_accepted` | `calculate_time_bins()` output passes for several `sampling_frequency` values and durations. |
| `test_nonuniform_bins_rejected` | A jittered edge array raises `ValidationError` naming the transition-model reason. |
| `test_float_accumulation_tolerated` | A long `arange`-built grid (1e6 bins) is accepted despite accumulated float error. |
| `test_likelihood_api_accepts_nonuniform` | `predict_*_log_likelihood` with nonuniform edges returns N finite rows with per-bin intensities proportional to each bin's width. Parametrized over both registries. |

```bash
uv run pytest src/non_local_detector/tests/models src/non_local_detector/tests/likelihoods -q
```

No golden impact.

## Review

Dispatch `code-reviewer`. Ask whether any detector entry point reaches the HMM
without passing through the new check — `estimate_parameters`, Viterbi, and the
covariate-dependent path each need it.
