> **Snippets unverified.** The code below has not been executed. Per the process
> rule in [PLAN.md](PLAN.md), treat it as intent to be re-derived by prototyping,
> not as code to paste. The problem statements and measurements are verified.

# Phase 8 — Remaining audit findings

Three confirmed findings that were neither scheduled nor deferred in the first
draft. Each is small; grouped because none warrants its own PR.

---

## Finding 1 — `to_density` normalizes counts, not mass per volume

`diffusion.to_density` (`diffusion.py:549-571`) divides each column by
`bin_sizes @ smoothed`. Its docstring calls the input "count fields", so a density
should be `count_i / volume_i / total`, not `count_i / total`.

**Impact is smaller than it looks — verified.** `sorted_spikes_diffusion` divides
two outputs of the *same* `to_density` call (`sorted_spikes_diffusion.py:510-527`),
and the volume factor cancels:

```
per-bin ratio (current) : [0.4035 0.9282 1.8107 3.4473 0.4218 0.0468]
per-bin ratio (proposed): [0.7490 1.7229 3.3608 6.3987 0.7828 0.0868]
ratios proportional (shape identical)? True
global scale factor difference: 0.5387
equal volumes -> identical: True
```

So the error is a **per-neuron global scale factor** on the rate, not a per-bin
shape distortion, and it is exactly 1.0 on equal-volume grids (i.e. every regular
grid). It bites only on linearized track graphs with variable edge spacing.

Because the scale multiplies the rate, it does affect the posterior through the
`−λ` term of `xlogy(k, λ) − λ`, which is bin-dependent. Real, second-order.

### Fix

Divide by each cell's own volume, and rename so the semantics are unambiguous:

```python
def to_density(mass: np.ndarray, bin_sizes: np.ndarray) -> np.ndarray:
    """Convert per-bin mass to a density integrating to one.

    ``density_i = (mass_i / bin_sizes_i) / sum_j mass_j``, so
    ``bin_sizes @ density == 1``. On equal-volume grids this equals the previous
    ``mass / (bin_sizes @ mass)`` up to a global constant; on variable-volume
    grids (linearized track graphs with uneven edge spacing) it differs by a
    per-column scale factor.
    """
    total = mass.sum(axis=0)
    safe = np.where(total > 0, total, 1.0)
    return np.where(total > 0, (mass / bin_sizes[:, None]) / safe, 0.0)
```

Verify the caller still gets the units it expects — `occupancy` and `marginals`
both come from this call, and phase 6b may have changed what "mass" means for the
diffusion exposure field. Sequence this **after** 6b.

---

## Finding 2 — MRF invents rates on unoccupied disconnected components

`fit_sorted_spikes_mrf_encoding_model` checks only *total* exposure
(`sorted_spikes_mrf.py:825`, `if not occupancy_field.sum() > 0.0`). A graph with
several connected components can have zero occupancy on one of them while the
total is positive. That component's unpenalized null mode then retains the global
warm-start rate (`sorted_spikes_mrf.py:266`), producing a finite, confident rate
in a region the animal never visited.

`diffusion.connected_component_labels` (`diffusion.py:350-372`) already computes
the labels needed.

### Fix

Check exposure per component and floor unoccupied components to the EPS rate:

```python
    labels = connected_component_labels(graph)
    component_exposure = np.zeros(labels.max() + 1)
    np.add.at(component_exposure, labels, occupancy_field)
    unoccupied = component_exposure[labels] <= 0.0
    # A component the animal never entered has no exposure, so its rate is
    # unidentified: the penalized fit's null mode would otherwise keep the global
    # warm-start rate and assert a confident rate where there is no data.
    rate_interior = np.where(unoccupied[:, None], EPS, rate_interior)
```

Warn once naming the component count, so a disconnected environment is not a
silent surprise.

---

## Finding 3 — `indices_are_sorted=True` on unvalidated spike times

Nine production `segment_sum` call sites in `likelihoods/` pass `indices_are_sorted=True`
(e.g. `clusterless_kde.py:476`), which is only valid if each unit's spike times
are sorted. Nothing validates that: `ensure_monotonic_increasing` is applied to
`time` only (`models/base.py:1908`).

Verified: CPU/XLA currently tolerates unsorted indices — identical results with
and without the hint. **GPU behaviour unverified**, and the hint is a contract, so
this is a latent portability defect rather than a live bug.

`tests/likelihoods/test_clusterless_kde.py:152` tests unsorted input to
`get_spike_time_bin_ind` but never reaches the `segment_sum`.

### Fix

Validate once per unit at the predict entry, where the spike arrays are already
being filtered:

```python
def validate_sorted_spike_times(spike_times: list[np.ndarray], unit_name: str) -> None:
    """Require each unit's spike times to be non-decreasing.

    The segment reductions pass ``indices_are_sorted=True`` to XLA, which is a
    contract rather than a hint: unsorted indices are undefined behaviour even
    where a given backend currently tolerates them.
    """
    for unit, times in enumerate(spike_times):
        times = np.asarray(times)
        if times.size > 1 and np.any(np.diff(times) < 0):
            raise ValidationError(
                f"{unit_name} {unit} has unsorted spike times",
                expected="non-decreasing spike times",
                got=f"{int(np.sum(np.diff(times) < 0))} out-of-order entries",
                hint="Sort each unit's spike times before fitting or predicting.",
            )
```

in `common.py`, called from each clusterless predictor.

**Consider the cheaper alternative first:** simply drop `indices_are_sorted=True`.
It is an optimization hint, and removing it costs nothing measurable unless
profiling shows otherwise — whereas validating turns previously-accepted unsorted
input into an error, which is a breaking public-API change that needs its own
CHANGELOG entry. Measure before choosing the breaking option.

---

## Validation

| Test | Asserts | File |
|---|---|---|
| `test_to_density_integrates_to_one` | `bin_sizes @ density == 1` per column, for equal and variable volumes. | `test_diffusion.py` |
| `test_to_density_equal_volumes_unchanged` | On an equal-volume grid the place-field output is unchanged from pre-fix, `rtol=1e-6`. | `test_sorted_spikes_diffusion.py` |
| `test_to_density_variable_volume_rate_scale` | On a variable-volume grid the rate changes by the predicted scale factor and the per-bin shape is unchanged. | same |
| `test_mrf_unoccupied_component_floors` | Two-component graph, occupancy on one only: rates on the unoccupied component are EPS, and a warning names the component count. | `test_sorted_spikes_mrf.py` |
| `test_mrf_single_component_unchanged` | A connected graph gives identical results to pre-fix. | same |
| `test_unsorted_spike_times_raise` | Unsorted times raise `ValidationError` naming the unit. Parametrized over the clusterless predictors. | `test_clusterless_kde.py` |
| `test_sorted_spike_times_accepted` | Sorted input still runs. | same |

```bash
uv run pytest src/non_local_detector/tests/likelihoods -q
uv run pytest src/non_local_detector/tests/test_golden_regression.py -q
```

Goldens: findings 2 and 3 should not move them (single-component fixtures, sorted
inputs). Finding 1 moves them **only** if a fixture uses a variable-volume grid —
check before assuming, and if one does, the diff must be a per-neuron constant.

## Review

Dispatch `code-reviewer`. Ask whether finding 1's rename left any caller reading
`to_density` output as counts, and whether the sorted-spike-times validator is
reached by every `segment_sum` call site or only some.
