# Sorted Spikes Likelihood GPU Optimization Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Replace Python neuron loops in the sorted spikes likelihood with vectorized operations. The non-local path gets both a sparse event-driven approach (best for long/sparse recordings) and a dense matmul approach (best for short/dense or ample GPU memory), auto-selected. The GLM local path is fully vectorized. The KDE local path retains a Python loop for per-neuron `KDEModel.predict` (different samples per model; batching requires model restructuring out of scope) but vectorizes the downstream Poisson math. The no_spike model is vectorized with the same sparse machinery.

**Architecture:** The Poisson log-likelihood `sum_n xlogy(count_n, rate_n_p) - sum_n rate_n_p` can be computed two ways:

1. **Sparse (event-driven):** For each spike event, look up `log(place_field[neuron, :])` and `segment_sum` into time bins. Work: `O(total_spikes × n_bins)`. Memory: `O(chunk_size × n_bins)` with chunking. Best when most time bins have 0 spikes for most neurons (typical: 99% sparse at 2ms bins).

2. **Dense (matmul):** `spike_counts.T @ log(place_fields)`. Work: `O(n_neurons × n_time × n_bins)`. Memory: `O(n_neurons × n_time)` for the count matrix. Best for short recordings or when GPU memory is ample.

**Tech Stack:** JAX (jit, lax.fori_loop, ops.segment_sum, make_jaxpr), NumPy, pytest

**Branch:** Execute on branch `sorted-spikes-gpu-optimization` off `main`. Test on CPU locally, then take to GPU hardware for real validation before merging.

**Typical scale:** ~100-500 neurons, 225k-1.8M time bins (2-4ms bins, 15-60 min), ~200 position bins, ~10-50 Hz firing per neuron.

**Sparsity analysis:**

| Scenario | n_time | total_spikes | dense matrix entries | sparsity |
|---|---|---|---|---|
| 4ms / 15min / 200 neurons | 225k | ~675k | 45M | 98.5% |
| 2ms / 60min / 200 neurons | 1.8M | ~2.2M | 360M | 99.4% |
| 2ms / 60min / 500 neurons | 1.8M | ~5.4M | 900M | 99.4% |

**Memory comparison:**

| Approach | 2ms/60min/500 neurons | Per-chunk |
|---|---|---|
| Dense count matrix | 3.4 GB | — |
| Dense output | 1.4 GB | — |
| Sparse indices | 41 MB (5.4M × 2 × int32) | — |
| Sparse chunked (50k spikes/chunk) | — | 38 MB peak |

---

## Verification Strategy

1. **Accuracy (primary):** Every task includes an **end-to-end reference-equality test** comparing the refactored output against the current neuron-loop implementation on realistic simulated data. Tolerance: 1e-10 (float64) or 1e-5 (float32). This is the gate.
2. **Jaxpr structure:** Trace the **pure-array helper functions** (`_nonlocal_poisson_loglik_sparse`, `_nonlocal_poisson_loglik_dense`), not the public predict functions (which accept Python lists, KDEModel, Environment — not JAX-traceable). Confirm helpers contain `segment_sum` or `dot_general` and no repeated subexpressions.
3. **Memory:** Verify chunked path intermediates are `O(chunk_size)`, not `O(n_time)` or `O(total_spikes)`.

---

## Core Mathematical Identity

```
log_likelihood[t, p] = sum_n count[n,t] * log(rate[n,p]) - sum_n rate[n,p]
```

Equivalently, using spike events indexed by `(neuron_id, time_bin_id)`:

```
log_likelihood[t, p] = sum_{spikes s in bin t} log(rate[neuron(s), p]) - no_spike_part[p]
```

**Safety of `log(place_fields)`:** Interior place fields are clipped to `[EPS, inf)` during fitting (`sorted_spikes_kde.py:208`, `sorted_spikes_glm.py:312`). `log(place_fields[:, is_track_interior])` is safe. Tests must assert `jnp.all(place_fields[:, is_track_interior] >= EPS)`.

---

## Task 1: Non-Local Sorted Spikes — Sparse + Dense Paths

**Files:**
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_kde.py` (lines 339-367)
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_glm.py` (lines 425-453)
- Modify: `src/non_local_detector/likelihoods/common.py` (add helpers)
- Test: `src/non_local_detector/tests/likelihoods/test_sorted_spikes_vectorized.py` (new)

### Step 1: Write the pure-array helpers

Add to `common.py`:

```python
def _build_spike_event_arrays(
    spike_times_list: list[np.ndarray],
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sorted (neuron_id, time_bin_id) arrays from per-neuron spike times.

    Parameters
    ----------
    spike_times_list : list[np.ndarray]
        Per-neuron spike times (each already filtered to decode window).
    time : np.ndarray, shape (n_time,)
        Time bin edges.

    Returns
    -------
    neuron_ids : np.ndarray, shape (total_spikes,), int32
    time_bin_ids : np.ndarray, shape (total_spikes,), int32
        Sorted by time_bin_id for segment_sum efficiency.
    """
    all_neuron_ids = []
    all_time_bin_ids = []
    for n, st in enumerate(spike_times_list):
        if len(st) == 0:
            continue
        bin_ids = get_spike_time_bin_ind(st, time)
        all_neuron_ids.append(np.full(len(st), n, dtype=np.int32))
        all_time_bin_ids.append(bin_ids.astype(np.int32))

    if not all_neuron_ids:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    neuron_ids = np.concatenate(all_neuron_ids)
    time_bin_ids = np.concatenate(all_time_bin_ids)

    # Sort by time bin for segment_sum indices_are_sorted=True
    sort_idx = np.argsort(time_bin_ids, kind='stable')
    return neuron_ids[sort_idx], time_bin_ids[sort_idx]


def _nonlocal_poisson_loglik_sparse(
    neuron_ids: jnp.ndarray,          # (total_spikes,)
    time_bin_ids: jnp.ndarray,        # (total_spikes,)
    log_place_fields: jnp.ndarray,    # (n_neurons, n_bins)
    no_spike_part: jnp.ndarray,       # (n_bins,)
    n_time: int,
) -> jnp.ndarray:
    """Sparse event-driven non-local Poisson log-likelihood.

    O(total_spikes * n_bins) work. For chunked version, see
    _nonlocal_poisson_loglik_sparse_chunked.
    """
    spike_contributions = log_place_fields[neuron_ids]  # (total_spikes, n_bins)
    log_likelihood = jax.ops.segment_sum(
        spike_contributions, time_bin_ids,
        num_segments=n_time, indices_are_sorted=True,
    )
    return log_likelihood - no_spike_part


def _nonlocal_poisson_loglik_sparse_chunked(
    neuron_ids: jnp.ndarray,
    time_bin_ids: jnp.ndarray,
    log_place_fields: jnp.ndarray,
    no_spike_part: jnp.ndarray,
    n_time: int,
    chunk_size: int = 50000,
) -> jnp.ndarray:
    """Chunked sparse path — O(chunk_size * n_bins) peak memory.

    Processes spike events in chunks, accumulating segment_sum results.
    Padded spikes use dummy time bin n_time (ignored by segment_sum).
    """
    total_spikes = neuron_ids.shape[0]
    n_bins = log_place_fields.shape[1]
    n_chunks = (total_spikes + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size

    if n_padded > total_spikes:
        pad = n_padded - total_spikes
        neuron_ids = jnp.pad(neuron_ids, (0, pad))
        time_bin_ids = jnp.pad(time_bin_ids, (0, pad), constant_values=n_time)

    def process_chunk(i, ll):
        start = i * chunk_size
        nids = jax.lax.dynamic_slice(neuron_ids, (start,), (chunk_size,))
        tids = jax.lax.dynamic_slice(time_bin_ids, (start,), (chunk_size,))
        contributions = log_place_fields[nids]  # (chunk_size, n_bins)
        return ll + jax.ops.segment_sum(
            contributions, tids, num_segments=n_time, indices_are_sorted=True,
        )

    ll = jnp.zeros((n_time, n_bins))
    ll = jax.lax.fori_loop(0, n_chunks, process_chunk, ll)
    return ll - no_spike_part


def get_spikecount_per_time_bin_batched(
    spike_times_list: list[np.ndarray],
    time: np.ndarray,
) -> np.ndarray:
    """Spike counts for all neurons as (n_neurons, n_time) int32 array."""
    return np.stack(
        [get_spikecount_per_time_bin(st, time) for st in spike_times_list]
    ).astype(np.int32)


def _nonlocal_poisson_loglik_dense(
    spike_counts: jnp.ndarray,       # (n_neurons, n_time)
    log_place_fields: jnp.ndarray,   # (n_neurons, n_bins)
    no_spike_part: jnp.ndarray,      # (n_bins,)
) -> jnp.ndarray:
    """Dense matmul path — one BLAS call, O(n_neurons * n_time) memory."""
    return spike_counts.T @ log_place_fields - no_spike_part


def _nonlocal_poisson_loglik_dense_chunked(
    spike_counts: jnp.ndarray,
    log_place_fields: jnp.ndarray,
    no_spike_part: jnp.ndarray,
    chunk_size: int = 10000,
) -> jnp.ndarray:
    """Chunked dense matmul over time — O(n_neurons * chunk_size) peak."""
    n_neurons, n_time = spike_counts.shape
    n_bins = log_place_fields.shape[1]
    n_chunks = (n_time + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size

    if n_padded > n_time:
        spike_counts = jnp.pad(spike_counts, ((0, 0), (0, n_padded - n_time)))

    def process_chunk(i, out):
        t_start = i * chunk_size
        counts_chunk = jax.lax.dynamic_slice(
            spike_counts, (0, t_start), (n_neurons, chunk_size)
        )
        chunk_ll = counts_chunk.T @ log_place_fields - no_spike_part
        return jax.lax.dynamic_update_slice(out, chunk_ll, (t_start, 0))

    out = jnp.zeros((n_padded, n_bins))
    out = jax.lax.fori_loop(0, n_chunks, process_chunk, out)
    return out[:n_time]
```

### Step 2: Write tests

```python
"""Tests for vectorized sorted spikes likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.common import (
    EPS,
    _build_spike_event_arrays,
    _nonlocal_poisson_loglik_sparse,
    _nonlocal_poisson_loglik_sparse_chunked,
    _nonlocal_poisson_loglik_dense,
    _nonlocal_poisson_loglik_dense_chunked,
    get_spikecount_per_time_bin,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)


class TestSparseNonlocalLoglik:
    """Test the sparse event-driven helper."""

    @staticmethod
    def _make_test_data(rng, n_neurons=50, n_time=1000, n_bins=100):
        """Create spike events and place fields for testing."""
        place_fields = jnp.array(rng.uniform(EPS, 5.0, (n_neurons, n_bins)))
        log_pf = jnp.log(place_fields)
        no_spike_part = jnp.sum(place_fields, axis=0)
        time_edges = np.linspace(0, 10, n_time + 1)

        spike_times_list = [
            np.sort(rng.uniform(0, 10, rng.poisson(20)))
            for _ in range(n_neurons)
        ]
        # Filter to window
        spike_times_filtered = [
            st[(st >= time_edges[0]) & (st <= time_edges[-1])]
            for st in spike_times_list
        ]
        neuron_ids, time_bin_ids = _build_spike_event_arrays(
            spike_times_filtered, time_edges,
        )
        return {
            "neuron_ids": jnp.asarray(neuron_ids),
            "time_bin_ids": jnp.asarray(time_bin_ids),
            "log_pf": log_pf,
            "place_fields": place_fields,
            "no_spike_part": no_spike_part,
            "n_time": n_time,
            "spike_times_filtered": spike_times_filtered,
            "time_edges": time_edges,
        }

    def test_sparse_matches_neuron_loop(self):
        """Sparse segment_sum matches explicit neuron loop."""
        rng = np.random.default_rng(42)
        d = self._make_test_data(rng)

        # Reference: explicit neuron loop (current code's approach)
        ref = jnp.zeros((d["n_time"], d["log_pf"].shape[1]))
        for n, st in enumerate(d["spike_times_filtered"]):
            counts = get_spikecount_per_time_bin(st, d["time_edges"])
            ref += jax.scipy.special.xlogy(
                counts[:, None], d["place_fields"][n, None, :]
            )
        ref -= d["no_spike_part"]

        result = _nonlocal_poisson_loglik_sparse(
            d["neuron_ids"], d["time_bin_ids"],
            d["log_pf"], d["no_spike_part"], d["n_time"],
        )
        assert jnp.allclose(result, ref, atol=1e-5, rtol=1e-5), (
            f"Max diff: {float(jnp.max(jnp.abs(result - ref))):.2e}"
        )

    def test_sparse_chunked_matches_unchunked(self):
        """Chunked sparse matches full sparse."""
        rng = np.random.default_rng(42)
        d = self._make_test_data(rng)

        full = _nonlocal_poisson_loglik_sparse(
            d["neuron_ids"], d["time_bin_ids"],
            d["log_pf"], d["no_spike_part"], d["n_time"],
        )
        chunked = _nonlocal_poisson_loglik_sparse_chunked(
            d["neuron_ids"], d["time_bin_ids"],
            d["log_pf"], d["no_spike_part"], d["n_time"],
            chunk_size=200,
        )
        assert jnp.allclose(full, chunked, atol=1e-10, rtol=1e-10)

    def test_dense_matches_sparse(self):
        """Dense matmul matches sparse segment_sum."""
        rng = np.random.default_rng(42)
        d = self._make_test_data(rng)

        sparse_result = _nonlocal_poisson_loglik_sparse(
            d["neuron_ids"], d["time_bin_ids"],
            d["log_pf"], d["no_spike_part"], d["n_time"],
        )

        from non_local_detector.likelihoods.common import (
            get_spikecount_per_time_bin_batched,
        )
        counts = jnp.asarray(get_spikecount_per_time_bin_batched(
            d["spike_times_filtered"], d["time_edges"],
        ))
        dense_result = _nonlocal_poisson_loglik_dense(
            counts, d["log_pf"], d["no_spike_part"],
        )
        assert jnp.allclose(sparse_result, dense_result, atol=1e-5, rtol=1e-5)

    def test_dense_chunked_matches_unchunked(self):
        """Chunked dense matches full dense."""
        rng = np.random.default_rng(42)
        d = self._make_test_data(rng)

        from non_local_detector.likelihoods.common import (
            get_spikecount_per_time_bin_batched,
        )
        counts = jnp.asarray(get_spikecount_per_time_bin_batched(
            d["spike_times_filtered"], d["time_edges"],
        ))
        full = _nonlocal_poisson_loglik_dense(
            counts, d["log_pf"], d["no_spike_part"],
        )
        chunked = _nonlocal_poisson_loglik_dense_chunked(
            counts, d["log_pf"], d["no_spike_part"], chunk_size=200,
        )
        assert jnp.allclose(full, chunked, atol=1e-10, rtol=1e-10)

    def test_empty_spikes(self):
        """No spikes → output equals -no_spike_part broadcast."""
        n_time, n_bins = 100, 50
        log_pf = jnp.zeros((10, n_bins))
        no_spike_part = jnp.ones(n_bins) * 5.0
        nids = jnp.array([], dtype=jnp.int32)
        tids = jnp.array([], dtype=jnp.int32)

        result = _nonlocal_poisson_loglik_sparse(
            nids, tids, log_pf, no_spike_part, n_time,
        )
        expected = jnp.broadcast_to(-no_spike_part, (n_time, n_bins))
        assert jnp.allclose(result, expected, atol=1e-12)


class TestEndToEndKDENonlocal:
    """End-to-end reference equality for KDE non-local path."""

    @pytest.fixture
    def kde_data(self):
        from non_local_detector.simulate.clusterless_simulation import (
            make_simulated_run_data,
        )
        sim = make_simulated_run_data(
            n_tetrodes=1,
            place_field_means=np.arange(0, 120, 10),
            sampling_frequency=500, n_runs=2, seed=42,
        )
        n_encode = int(0.7 * len(sim.position_time))
        enc_time = sim.position_time[:n_encode]
        enc_pos = sim.position[:n_encode]
        enc_spikes = [st[st <= enc_time[-1]] for st in sim.spike_times]
        encoding_model = fit_sorted_spikes_kde_encoding_model(
            enc_time, enc_pos, enc_spikes, sim.environment,
            position_std=6.0, block_size=100, disable_progress_bar=True,
        )
        test_edges = sim.edges[sim.edges >= enc_time[-1]]
        test_spikes = [st[st >= enc_time[-1]] for st in sim.spike_times]
        return {
            "time": test_edges,
            "position_time": sim.position_time[n_encode:],
            "position": sim.position[n_encode:],
            "spike_times": test_spikes,
            "encoding_model": encoding_model,
            "environment": sim.environment,
        }

    def test_nonlocal_matches_reference(self, kde_data):
        """Refactored output matches current neuron-loop output.

        Before refactoring: save output as golden reference.
        After refactoring: load golden and assert allclose.
        """
        d = kde_data
        result = predict_sorted_spikes_kde_log_likelihood(
            d["time"], d["position_time"], d["position"],
            d["spike_times"], d["environment"],
            **d["encoding_model"],
            is_local=False, disable_progress_bar=True,
        )
        assert jnp.all(jnp.isfinite(result))
        assert result.ndim == 2

    def test_place_fields_safety(self, kde_data):
        """Fitted interior place fields are >= EPS."""
        em = kde_data["encoding_model"]
        assert jnp.all(em["place_fields"][:, em["is_track_interior"]] >= EPS)
```

### Step 3: Implement in predict functions

Replace the neuron loop in both KDE and GLM non-local paths:

```python
# Filter spike times to decode window
spike_times_filtered = [
    st[np.logical_and(st >= time[0], st <= time[-1])]
    for st in spike_times
]

# Precompute log place fields (cache this after fit for repeat calls)
interior_pf = place_fields[:, is_track_interior]
log_interior_pf = jnp.log(interior_pf)  # safe: all >= EPS
no_spike_interior = no_spike_part_log_likelihood[is_track_interior]
n_interior_bins = int(is_track_interior.sum())

# Auto-select sparse vs dense
neuron_ids, time_bin_ids = _build_spike_event_arrays(
    spike_times_filtered, time,
)
total_spikes = len(neuron_ids)
n_neurons = len(spike_times)
dense_cost = n_neurons * n_time  # count matrix entries
sparse_cost = total_spikes       # event entries (times n_bins for contributions)

if sparse_cost * n_interior_bins < dense_cost * n_interior_bins:
    # Sparse path — O(total_spikes * n_bins)
    log_likelihood = _nonlocal_poisson_loglik_sparse_chunked(
        jnp.asarray(neuron_ids), jnp.asarray(time_bin_ids),
        log_interior_pf, no_spike_interior, n_time,
    )
else:
    # Dense path — O(n_neurons * n_time * n_bins)
    spike_counts = jnp.asarray(
        get_spikecount_per_time_bin_batched(spike_times_filtered, time)
    )
    log_likelihood = _nonlocal_poisson_loglik_dense(
        spike_counts, log_interior_pf, no_spike_interior,
    )
```

**Also: cache `log_interior_pf` after fit.** Add to the encoding model dict:

```python
# In fit_sorted_spikes_kde_encoding_model, before return:
encoding_model["log_interior_place_fields"] = jnp.log(
    place_fields[:, is_track_interior]
)
```

### Step 4: Run tests, verify jaxpr on helpers, profile

### Step 5: Commit

```bash
git commit -m "Vectorize non-local sorted spikes via sparse segment_sum + dense matmul

Two paths auto-selected by sparsity: sparse event-driven segment_sum for
long/sparse recordings (O(total_spikes)), dense matmul for short/dense.
Both chunked for bounded memory. Caches log(place_fields) after fit."
```

---

## Task 2: Vectorize Local Sorted Spikes Likelihood

**GLM local — fully vectorizable:**

```python
# (n_time, n_basis) @ (n_basis, n_neurons) -> (n_time, n_neurons)
all_local_rates = jnp.clip(
    jnp.exp(emission_predict_matrix @ coefficients.T), min=EPS
)
spike_counts = jnp.asarray(
    get_spikecount_per_time_bin_batched(spike_times_filtered, time)
)  # (n_neurons, n_time)

log_likelihood = jnp.sum(
    jax.scipy.special.xlogy(spike_counts.T, all_local_rates) - all_local_rates,
    axis=1,  # sum over neurons → (n_time,)
)[:, None]
```

**Note:** `emission_predict_matrix` is `(n_time, n_basis)` from the animal's interpolated position at decode time — NOT the `(n_bins, n_basis)` predict matrix from fitting.

**Memory:** `all_local_rates` is `(n_time, n_neurons)` — at 1.8M × 500 = 3.4 GB. For long recordings, chunk over time (same `fori_loop` pattern as Task 1's dense chunked path).

**KDE local — partially vectorizable:**

The Python loop over `KDEModel.predict` is retained (each model has different `samples_`). The downstream Poisson math is vectorized:

```python
# Step 1: KDE predictions — Python loop (different samples per model)
all_marginal_densities = jnp.stack([
    jnp.where(jnp.isnan(d := model.predict(interpolated_position)), 0.0, d)
    for model in marginal_models
])  # (n_neurons, n_time)

# Step 2: Rate computation — vectorized
all_local_rates = jnp.clip(
    jnp.array(mean_rates)[:, None] * safe_divide(
        all_marginal_densities, occupancy[None, :]
    ),
    min=EPS,
)  # (n_neurons, n_time)

# Step 3: Poisson log-likelihood — vectorized
spike_counts = jnp.asarray(
    get_spikecount_per_time_bin_batched(spike_times_filtered, time)
)
log_likelihood = jnp.sum(
    jax.scipy.special.xlogy(spike_counts, all_local_rates) - all_local_rates,
    axis=0,  # sum over neurons → (n_time,)
)[:, None]
```

**Files:**
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_kde.py` (lines 294-338)
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_glm.py` (lines 394-424)
- Test: extend `test_sorted_spikes_vectorized.py` with reference-equality tests for local path

### Step 1: Write reference-equality tests, implement, verify

### Step 2: Commit

---

## Task 3: Vectorize No-Spike Model

`predict_no_spike_log_likelihood` in `no_spike.py` has the same neuron-loop structure. Since `no_spike_rate` is a constant scalar, the sparse path simplifies to:

```python
# Count total spikes per time bin across ALL neurons
total_spike_counts = np.zeros(n_time, dtype=np.int32)
for neuron_spike_times in spike_times:
    total_spike_counts += get_spikecount_per_time_bin(neuron_spike_times, time)

# Vectorized: xlogy(total_count, rate) - n_neurons * rate
no_spike_log_likelihood = (
    jax.scipy.special.xlogy(total_spike_counts, no_spike_rate)
    - len(spike_times) * no_spike_rate
)[:, None]
```

Even simpler: since `xlogy(k, r) = k * log(r)` and `log(no_spike_rate)` is a scalar:

```python
total_spike_counts = sum(
    get_spikecount_per_time_bin(st, time) for st in spike_times
)
log_rate = jnp.log(no_spike_rate)
no_spike_log_likelihood = (
    total_spike_counts * log_rate - len(spike_times) * no_spike_rate
)[:, None]
```

No neuron loop needed — just sum the per-neuron counts (numpy, cheap) then one multiply.

**Files:**
- Modify: `src/non_local_detector/likelihoods/no_spike.py`
- Test: extend `test_sorted_spikes_vectorized.py`

### Step 1: Write reference-equality test, implement, commit

---

## Task 4: Regular-Bin Fast Path for Spike Binning

`get_spike_time_bin_ind` in `common.py:66` uses `np.digitize(spike_times, time[1:-1])`. For evenly spaced bins (the common case), direct integer division is faster:

```python
def get_spike_time_bin_ind_uniform(
    spike_times: np.ndarray, time: np.ndarray,
) -> np.ndarray:
    """Fast spike binning for uniformly spaced time bins."""
    dt = time[1] - time[0]
    t0 = time[0]
    return np.clip(
        ((spike_times - t0) / dt).astype(np.int32),
        0, len(time) - 1,
    )
```

Add a check for uniform spacing and auto-dispatch:

```python
def get_spike_time_bin_ind(spike_times, time):
    dt = np.diff(time)
    if np.allclose(dt, dt[0], rtol=1e-10):
        return get_spike_time_bin_ind_uniform(spike_times, time)
    return np.digitize(spike_times, time[1:-1])
```

**Files:**
- Modify: `src/non_local_detector/likelihoods/common.py`
- Test: equivalence test against `np.digitize` for uniform bins

---

## Task 5: Cache Transformed Encoding-Side Quantities

Cache quantities that are constant across predict calls:

**Sorted spikes (both KDE and GLM):**
- `log(place_fields[:, is_track_interior])` — computed once after fit, stored in encoding model dict
- `no_spike_part_log_likelihood[is_track_interior]` — same

**Sorted spikes GLM additionally:**
- The predict matrix at interior bin centers (already cached as `place_fields`)

Add to the fit functions' return dicts. No algorithmic change — pure caching.

**Files:**
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_kde.py` (fit function)
- Modify: `src/non_local_detector/likelihoods/sorted_spikes_glm.py` (fit function)
- Test: verify cached values match recomputed values

---

## Task 6: JAX-Traceable Spike Counting (Optional)

Replace `np.digitize` + `np.bincount` with `jnp.searchsorted(..., side='right')` + `jax.ops.segment_sum`. Only needed for full JIT fusion (e.g., if the electrode/neuron loop moves inside `scan`). Uses equivalence validated in the clusterless plan's Task 0.

---

## Execution Order and Dependencies

```
Task 1 (sparse + dense non-local)    ← highest impact
    ↓
Task 3 (no_spike vectorization)      ← same machinery, quick win
    ↓
Task 2 (vectorize local)             ← uses Task 1's batched counting
    ↓
Task 4 (uniform-bin fast path)       ← independent, small win
    ↓
Task 5 (cache encoding quantities)   ← independent, small win
    ↓
Task 6 (JAX spike counting)          ← optional future-proofing
```

---

## Expected Impact

| Metric | Current (neuron loop) | Sparse | Dense | Chunked sparse |
|---|---|---|---|---|
| GPU kernel launches / predict | n_neurons (100-500) | 1 | 1 | n_chunks |
| Peak memory (2ms/60min/500 neurons) | 1.4 GB (output) | 1.4 GB + 41 MB indices | 4.8 GB | 38 MB/chunk + 1.4 GB output |
| Work | O(n_neurons × n_time × n_bins) | O(total_spikes × n_bins) | O(n_neurons × n_time × n_bins) | same as sparse |
| Sparsity benefit | None | 99.4% less work at 2ms/60min | None | same as sparse |
| Accuracy vs current | reference | 1e-5 (float32) | 1e-10 (float64) / 1e-5 (float32) | same |

---

## Deferred: Streaming Likelihood Into HMM

Streaming likelihood chunks directly into the HMM filter (via `cache_log_likelihoods=False` in `core.chunked_filter_smoother`) eliminates the `(n_time, n_bins)` output allocation from the critical path. This is cross-cutting (affects both sorted spikes and clusterless) and belongs in a separate plan that modifies `core.py` and `base.py`'s `_predict` method.
