> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 6a — Time vocabulary migration

Shape migration only. **No rate-unit change** — that is phase 6b. Keeping them
separate means a golden diff in 6a is explicable purely by "one more real row",
and a diff in 6b purely by unit conversion.

## Contracts referenced

- [C3 — Time vocabulary](shared-contracts.md#c3--time-vocabulary)

## Problem

`get_spike_time_bin_ind` (`common.py:242-255`) digitizes against `time[1:-1]`,
returning indices `0 … len(time)-2`, while `get_spikecount_per_time_bin`
(`common.py:569-589`) allocates `len(time)` rows and every predictor returns
`len(time)` rows. The final row can never hold a spike:

```
time = [0..5], spikes = [0.5,1.5,2.5,3.5,4.5]
counts = [1,1,1,1,1,0]      # 5 spikes, 6 rows
```

`no_spike.py:35` documents N+1 edges → N bins while `no_spike.py:80,92` returns
`len(time)` rows, and its docstring example raises `TypeError` (annotated
`list[list[float]]`, implementation needs arrays).

## Falsification

Rewriting `get_spikecount_per_time_bin` in place **immediately breaks GLM
fitting**. `sorted_spikes_glm.py:299` sets `time = np.asarray(position_time)` —
sample centers — and `:339` feeds it to the helper. Under the new contract N
samples yield N−1 counts against an N-row design matrix and N-row weights.

Before implementing, run the sorted-GLM encoding tests against a scratch branch
containing only the helper rewrite. They must fail. If they pass, the encoding
path is not exercised and the phase's task 2 is untested.

## Tasks

### 1. Decode-side helpers take edges

```python
def get_spike_time_bin_ind(spike_times: np.ndarray, time_edges: np.ndarray) -> np.ndarray:
    """Index of the bin containing each spike time.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Assumed already restricted to ``[time_edges[0], time_edges[-1]]``.
    time_edges : np.ndarray, shape (n_bins + 1,)
        Monotonically increasing bin edges.

    Returns
    -------
    ind : np.ndarray, shape (n_spikes,)
        Values in ``[0, n_bins - 1]``. Bin ``i`` covers ``[edges[i], edges[i+1])``
        except the last, which is right-closed.
    """
    n_bins = time_edges.shape[0] - 1
    return np.clip(np.digitize(spike_times, time_edges[1:-1]), 0, n_bins - 1)


def get_spikecount_per_time_bin(spike_times: np.ndarray, time_edges: np.ndarray) -> np.ndarray:
    """Number of spikes per bin, shape (n_bins,)."""
    n_bins = time_edges.shape[0] - 1
    spike_times = np.asarray(spike_times)
    in_range = (spike_times >= time_edges[0]) & (spike_times <= time_edges[-1])
    return np.bincount(
        get_spike_time_bin_ind(spike_times[in_range], time_edges), minlength=n_bins
    )
```

Add `bin_durations(time_edges)` and `bin_centers(time_edges)` to `common.py` as
the single sources of those quantities.

### 2. Give encoding its own sample-cell helper

Add `sample_cell_durations(position_time)` and `sample_cell_edges(position_time)`
to `common.py`, per [C3](shared-contracts.md#c3--time-vocabulary). Then in
`sorted_spikes_glm.py`, replace `time = np.asarray(position_time)` (`:299`) and
the `:339` call:

```python
    # Encoding bins are the position samples' own midpoint cells, not decode
    # edges: the design matrix and weights have one row per position sample.
    encoding_edges = sample_cell_edges(np.asarray(position_time))
    ...
        spike_count_per_time_bin = get_spikecount_per_time_bin(
            neuron_spike_times, encoding_edges
        )
```

`sample_cell_edges` returns `n_samples + 1` edges, so the counts are
`n_samples`-long and align with the design matrix. Audit every other
`get_spikecount_per_time_bin` call for which of the two roles it is in — grep
rather than assuming this is the only one.

### 3. Propagate N+1 edges through every caller

This is the bulk of the phase. Each must be changed, not just the predictors:

| Site | Change |
|---|---|
| `models/base.py:2308-2325` `calculate_time_bins` | Emit `n_bins + 1` edges: `time_range[0] + np.arange(n_bins + 1) / self.sampling_frequency`. |
| `models/base.py:3133` and sorted counterpart | `n_time = len(time_edges) - 1`. |
| `core.py:568`, `core.py:1164` | `n_time = len(time) - 1` — both chunked entry points. Cached likelihoods have N rows; edge indices would over-index by one. |
| `models/base.py:3379` | `is_missing` must have N values, not `len(time)`. |
| `models/base.py:174` `_validate_covariate_time_length` | Transition covariates must have N rows. |
| `sorted_spikes_kde.py:333` and every other local path | Position is interpolated at **centers**, not edges — `get_position_at_time(position_time, position, bin_centers(time_edges), env)`. Currently produces N+1 values. |
| `_compute_local_position_kernel`, `_compute_non_local_position_penalty` | Same: evaluate at centers, return N rows. |
| xarray assembly / `_convert_seq_to_df` | `time` coordinate is `bin_centers(time_edges)`. |

Grep for `len(time)`, `time.shape[0]`, and `n_time` across `likelihoods/`,
`models/`, and `core.py` and classify each occurrence as edges-count or
rows-count. Leaving one unconverted produces an off-by-one that tests may not
catch if fixtures are uniform.

### 4. Fix `no_spike`

`no_spike.py:22-92`: annotate `spike_times: list[np.ndarray]`, describe `time` as
N+1 edges, return `(n_bins, 1)`, and correct the example so it runs. Replace the
median-interval scaling at `:79` with per-bin durations — but keep the rate in
its current units for this phase; 6b converts it.

### 5. Documentation

CHANGELOG `Changed` (breaking): `time` is N+1 bin edges; all predictors return N
rows; the xarray `time` coordinate is bin centers. Include a before/after snippet
for `predict(time=...)`. Update README/getting-started examples and both
`predict` docstrings.

Callers previously passing N sample centers must now pass N+1 edges. State
explicitly that re-running old code unchanged will decode a *different interval*
as well as a different length.

## Validation

| Test | Asserts | File |
|---|---|---|
| `test_all_spikes_counted` | For random edges and spikes, `sum(counts)` equals spikes in `[edges[0], edges[-1]]`. | `test_kde_common.py` |
| `test_last_bin_reachable` | A spike in `[edges[-2], edges[-1]]` lands in bin `N-1`. | same |
| `test_spike_at_final_edge_counted` | A spike exactly at `edges[-1]` lands in bin `N-1`. | same |
| `test_sample_cell_edges_roundtrip` | `sample_cell_edges(t)` has `len(t)+1` entries; durations sum to the covered span; jittered timestamps give unequal cells. | same |
| `test_predictors_return_n_bins_rows` | Every registry entry returns `len(edges)-1` rows. Parametrized over both registries. | `test_likelihood_cross_model.py` |
| `test_glm_encoding_counts_align_with_design` | GLM spike counts have exactly `n_position_samples` entries. | `test_sorted_spikes_glm.py` |
| `test_local_interpolation_returns_n_rows` | Local paths return N rows, and the interpolated position equals interpolation at centers. | `test_likelihood_properties.py` |
| `test_is_missing_length_validated` | `is_missing` of length N+1 raises. | `tests/models/` |
| `test_covariate_length_validated` | Covariates of length N+1 raise. | same |
| `test_no_spike_example_runs` | The corrected docstring example executes and returns its claimed shape. | `test_likelihood_edge_cases.py` |

Coverage must include EM, Viterbi, cached, chunked, and covariate-dependent
paths — each indexes time independently.

```bash
uv run pytest src/non_local_detector/tests -q
uv run pytest -m snapshot -q
```

**Approval gate.** Every golden and snapshot moves. Beyond the four-part
analysis, show:

1. For a fixture where the previous last row held no spikes, the first N−1 rows
   are unchanged within `rtol=1e-6`. This is the strongest evidence the migration
   is purely a shape change.
2. The new final row is non-degenerate — it holds the spikes it should.

If (1) fails, something numerical changed in a shape-only phase.

## Review

Dispatch `code-reviewer`. Ask for an independent sweep of every remaining
`len(time)` / `time.shape[0]` in `likelihoods/`, `models/`, and `core.py`,
classified as edges or rows.
