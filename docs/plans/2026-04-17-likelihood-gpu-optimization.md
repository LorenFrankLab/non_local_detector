# Clusterless KDE Likelihood GPU Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Move the three Python loops in the clusterless KDE likelihood pipeline inside JIT so the entire computation compiles to a single GPU program, achieving near-peak tensor core utilization at million-scale spike counts.

**Architecture:** The current pipeline has three nested Python loops (electrodes, decoding spike blocks, encoding spike chunks) that cause thousands of sequential GPU kernel launches. We restructure bottom-up: first move the block loop inside JIT via `fori_loop`, then precompute reusable GEMM quantities to eliminate redundant work, then move the electrode loop inside JIT via `scan`. Each step is independently valuable and testable. We verify speed, accuracy, and memory at every step using CPU profiling and jaxpr inspection (GPU behavior is predictable from these).

**Tech Stack:** JAX (jit, lax.fori_loop, lax.scan, make_jaxpr), NumPy, pytest

**Branch:** Execute on branch `clusterless-gpu-optimization` off `main`. Test on CPU locally, then take to GPU hardware for real validation before merging.

**Typical scale:** ~64 electrodes, ~10k-20k encoding spikes per electrode (1-2M total), ~10k-20k decoding spikes per electrode (1-2M total), 200 position bins, 4 waveform features (but variable per electrode — a bad tetrode wire means fewer features).

---

## Verification Strategy

We cannot test on GPU in this environment. Every task verifies three properties on CPU:

1. **Accuracy (primary):** Every task must include an **end-to-end numerical equivalence test** on the non-local likelihood path. Save the output of the current (pre-refactor) implementation on realistic simulated data, then compare the refactored implementation against it. Tolerance: 1e-10 atol for structural refactors (Tasks 1-2), 1e-4 for algorithmic changes. This is the gate — if the numbers don't match, the refactor is wrong regardless of what jaxpr shows.
2. **Jaxpr structure (secondary):** `jax.make_jaxpr` on the **private `_impl` function** (not the JIT-wrapped public API) confirms Python loops are replaced by JAX primitives. Important: tracing the JIT-wrapped function shows `jit` at the top level, hiding the inner structure. Always trace the unwrapped `_impl` to inspect loop primitives.
3. **Memory:** Intermediate tensor sizes from jaxpr analysis confirm bounds are maintained.

Speed on CPU is a tertiary signal — BLAS dominates and doesn't benefit from fusion. The real GPU wins come from eliminating kernel launch boundaries, which we verify structurally via jaxpr on `_impl`.

---

## Code Pattern: Private-Impl for JIT-wrapped Functions

Several functions in `clusterless_kde_log.py` are decorated with `@jax.jit` or wrapped via `jax.jit(fn, ...)`. When these need to be called inside `fori_loop` or `scan` bodies, JAX must trace through the un-JITted implementation. The current codebase uses `fn.__wrapped__` for this (see `core.py:184`), but this relies on `functools.wraps` internals.

**Adopt the private-impl pattern** used in `core.py` for `_filter_internal`/`_smoother_internal`:

```python
# Define the implementation as a private function
def _estimate_log_joint_mark_intensity_impl(
    decoding_spike_waveform_features, encoding_spike_waveform_features,
    waveform_stds, occupancy, mean_rate, log_position_distance,
    use_gemm, pos_tile_size, enc_tile_size, use_streaming,
    encoding_positions, position_eval_points, position_std,
):
    ...  # existing body

# Public API: JIT-wrapped for direct calls
estimate_log_joint_mark_intensity = jax.jit(
    _estimate_log_joint_mark_intensity_impl,
    static_argnames=("use_gemm", "pos_tile_size", "enc_tile_size", "use_streaming"),
)
```

Apply the same pattern to `log_kde_distance`, `log_kde_distance_streaming`, and `block_estimate_log_joint_mark_intensity`. The `fori_loop`/`scan` bodies call the `_impl` functions directly.

This refactor should be done as a prerequisite step in Task 1 before converting the loop.

---

## Task 0 (Prerequisite): `searchsorted` vs `digitize` Equivalence

`get_spike_time_bin_ind` in `common.py:66` uses `np.digitize(spike_times, time[1:-1])` which is not JAX-traceable. Task 3 needs a JAX equivalent inside `scan`. Before writing any Task 3 code, establish the correct replacement and its boundary behavior.

**Files:**
- Test: `src/non_local_detector/tests/likelihoods/test_jit_block_estimate.py` (add to new file)

### Step 1: Write the equivalence test

```python
class TestSearchsortedDigitizeEquivalence:
    """Verify jnp.searchsorted matches np.digitize for spike binning."""

    def test_random_spike_times(self):
        """Equivalence on 1000 random spike times."""
        rng = np.random.default_rng(42)
        time_edges = np.linspace(0, 10, 501)  # 500 time bins
        spike_times = np.sort(rng.uniform(-0.1, 10.1, 1000))  # some out of bounds

        reference = np.digitize(spike_times, time_edges[1:-1])
        # np.digitize(x, bins) with default args is equivalent to
        # searchsorted(bins, x, side='right')
        result = jnp.searchsorted(time_edges[1:-1], spike_times, side='right')

        np.testing.assert_array_equal(np.asarray(result), reference)

    def test_spikes_on_edges(self):
        """Spikes exactly on bin edges are handled correctly."""
        time_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        spike_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.5])

        reference = np.digitize(spike_times, time_edges[1:-1])
        result = jnp.searchsorted(time_edges[1:-1], spike_times, side='right')

        np.testing.assert_array_equal(np.asarray(result), reference)

    def test_empty_spikes(self):
        """Empty spike array returns empty result."""
        time_edges = np.linspace(0, 10, 101)
        spike_times = np.array([])

        reference = np.digitize(spike_times, time_edges[1:-1])
        result = jnp.searchsorted(time_edges[1:-1], spike_times, side='right')

        np.testing.assert_array_equal(np.asarray(result), reference)
```

### Step 2: Run and verify the tests pass

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_jit_block_estimate.py::TestSearchsortedDigitizeEquivalence -v`

Expected: ALL PASS — confirming `jnp.searchsorted(..., side='right')` is the correct replacement.

### Step 3: Commit

```bash
git commit -m "Add equivalence test for jnp.searchsorted vs np.digitize

Confirms that jnp.searchsorted(bins, x, side='right') produces identical
results to np.digitize(x, bins) for spike time binning, including edge
cases (spikes on bin boundaries, out-of-bounds, empty arrays). This JAX-
traceable replacement is needed for Task 3 (electrode loop inside scan)."
```

---

## Task 1: Move `block_estimate` Loop Inside JIT

The Python for-loop in `block_estimate_log_joint_mark_intensity` (lines 1152-1168 of `clusterless_kde_log.py`) iterates over decoding spike blocks, calling `estimate_log_joint_mark_intensity` per block. Each call is a separate JIT dispatch = separate GPU kernel launch. At 20k decoding spikes with block_size=100, that's 200 launches per electrode.

**Files:**
- Modify: `src/non_local_detector/likelihoods/clusterless_kde_log.py`
  - Refactor `estimate_log_joint_mark_intensity` to private-impl pattern
  - Rewrite `block_estimate_log_joint_mark_intensity` body with `fori_loop`
  - JIT-compile `block_estimate_log_joint_mark_intensity`
- Test: `src/non_local_detector/tests/likelihoods/test_jit_block_estimate.py` (new)

### Step 1: Refactor to private-impl pattern

Rename `estimate_log_joint_mark_intensity` → `_estimate_log_joint_mark_intensity_impl`, then:

```python
estimate_log_joint_mark_intensity = jax.jit(
    _estimate_log_joint_mark_intensity_impl,
    static_argnames=("use_gemm", "pos_tile_size", "enc_tile_size", "use_streaming"),
)
```

Same for `log_kde_distance` → `_log_kde_distance_impl` and `log_kde_distance_streaming` → `_log_kde_distance_streaming_impl`.

### Step 2: Write the accuracy and jaxpr tests

Create `src/non_local_detector/tests/likelihoods/test_jit_block_estimate.py`:

```python
"""Tests for JIT-fused block_estimate_log_joint_mark_intensity."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
    log_kde_distance,
)


class TestJitBlockEstimateAccuracy:
    """Block estimate inside JIT matches Python-loop version."""

    @staticmethod
    def _make_electrode_data(rng, n_enc, n_dec, n_pos, n_wf=4):
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)
        enc_pos = jnp.array(rng.uniform(0, 200, (n_enc, 1)))
        eval_pos = jnp.linspace(0, 200, n_pos)[:, None]
        pos_std = jnp.array([3.5])
        log_pos_dist = log_kde_distance(eval_pos, enc_pos, pos_std)
        occupancy = jnp.ones(n_pos) * 0.01
        return dec_wf, enc_wf, wf_std, occupancy, log_pos_dist

    @pytest.mark.parametrize("block_size", [50, 100, 200, 499, 500])
    def test_matches_single_call(self, block_size):
        """JIT block_estimate matches single-call (all dec spikes at once)."""
        rng = np.random.default_rng(42)
        dec_wf, enc_wf, wf_std, occ, log_pos = self._make_electrode_data(
            rng, n_enc=1000, n_dec=500, n_pos=100
        )
        result = block_estimate_log_joint_mark_intensity(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, block_size=block_size,
        )
        reference = estimate_log_joint_mark_intensity(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos,
        )
        from non_local_detector.likelihoods.common import LOG_EPS
        reference = jnp.clip(reference, a_min=LOG_EPS)
        # Structural refactor: should be near-exact
        assert jnp.allclose(result, reference, atol=1e-10, rtol=1e-10), (
            f"block_size={block_size}: max diff="
            f"{float(jnp.max(jnp.abs(result - reference))):.2e}"
        )

    def test_large_scale(self):
        """Accuracy at scale: 10k enc, 5k dec."""
        rng = np.random.default_rng(123)
        dec_wf, enc_wf, wf_std, occ, log_pos = self._make_electrode_data(
            rng, n_enc=10000, n_dec=5000, n_pos=200
        )
        result = block_estimate_log_joint_mark_intensity(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, block_size=500,
        )
        assert result.shape == (5000, 200)
        assert jnp.all(jnp.isfinite(result))

    def test_empty_decoding_spikes(self):
        """Handles zero decoding spikes without error.

        Note: the empty case is handled BEFORE the JIT boundary
        (n_decoding_spikes is a shape, known at trace time). The JIT-compiled
        fori_loop is never called for empty input.
        """
        rng = np.random.default_rng(42)
        _, enc_wf, wf_std, occ, log_pos = self._make_electrode_data(
            rng, n_enc=100, n_dec=10, n_pos=50
        )
        dec_wf_empty = jnp.zeros((0, 4))
        result = block_estimate_log_joint_mark_intensity(
            dec_wf_empty, enc_wf, wf_std, occ, 5.0, log_pos, block_size=100,
        )
        from non_local_detector.likelihoods.common import LOG_EPS
        assert result.shape == (0, 50)


class TestJitBlockEstimateJaxpr:
    """Verify the block loop compiles to JAX primitives (no Python loop).

    IMPORTANT: Trace the private _impl function, NOT the JIT-wrapped
    public API. Tracing the JIT-wrapped version shows 'jit' at the top
    level, hiding the inner loop structure.
    """

    def test_no_python_loop_in_jaxpr(self):
        """The _impl jaxpr should contain a scan/while primitive, not
        repeated subexpressions from a Python for-loop.
        """
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _block_estimate_log_joint_mark_intensity_impl,
        )

        n_enc, n_dec, n_pos, n_wf = 100, 200, 50, 4
        dec_wf = jnp.zeros((n_dec, n_wf))
        enc_wf = jnp.zeros((n_enc, n_wf))
        wf_std = jnp.full(n_wf, 24.0)
        occ = jnp.ones(n_pos) * 0.01
        log_pos = jnp.zeros((n_enc, n_pos))

        # Trace the PRIVATE _impl (not the JIT-wrapped public API)
        fn = functools.partial(
            _block_estimate_log_joint_mark_intensity_impl,
            block_size=50, use_gemm=True, pos_tile_size=None,
            enc_tile_size=None, use_streaming=False,
        )
        jaxpr = jax.make_jaxpr(fn)(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos,
        )

        primitives = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]
        has_loop_primitive = any(
            p in primitives for p in ("scan", "while", "fori_loop")
        )
        n_dot_general = primitives.count("dot_general")

        assert has_loop_primitive, (
            f"Expected scan/while primitive in jaxpr, got: {set(primitives)}"
        )
        # fori_loop body has 1-2 dot_general; Python loop of 4 iters has 4+
        assert n_dot_general <= 3, (
            f"Expected <=3 dot_general (fori_loop body), got {n_dot_general}"
        )
```

### Step 3: Implement JIT-fused block_estimate

Replace the body of `block_estimate_log_joint_mark_intensity` with `fori_loop`.

**Key design decisions:**
- The `n_decoding_spikes == 0` early return stays OUTSIDE the JIT boundary as a shape-level check (shapes are static at trace time). The JIT-compiled `fori_loop` path only runs for n_dec > 0.
- Pad decoding features to a multiple of `block_size` for static shapes.
- Call `_estimate_log_joint_mark_intensity_impl` (the private unwrapped function) inside the loop body.

```python
def _block_estimate_log_joint_mark_intensity_impl(
    decoding_spike_waveform_features, encoding_spike_waveform_features,
    waveform_stds, occupancy, mean_rate, log_position_distance,
    block_size, use_gemm, pos_tile_size, enc_tile_size, use_streaming,
    encoding_positions, position_eval_points, position_std,
):
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]

    # Pad to multiple of block_size
    n_blocks = (n_decoding_spikes + block_size - 1) // block_size
    n_padded = n_blocks * block_size
    pad_amount = n_padded - n_decoding_spikes

    if pad_amount > 0:
        decoding_spike_waveform_features = jnp.pad(
            decoding_spike_waveform_features,
            ((0, pad_amount), (0, 0)), constant_values=0.0,
        )

    def process_block(i, out):
        start = i * block_size
        block_features = jax.lax.dynamic_slice(
            decoding_spike_waveform_features,
            (start, 0),
            (block_size, decoding_spike_waveform_features.shape[1]),
        )
        block_result = _estimate_log_joint_mark_intensity_impl(
            block_features, encoding_spike_waveform_features,
            waveform_stds, occupancy, mean_rate, log_position_distance,
            use_gemm=use_gemm, pos_tile_size=pos_tile_size,
            enc_tile_size=enc_tile_size, use_streaming=use_streaming,
            encoding_positions=encoding_positions,
            position_eval_points=position_eval_points,
            position_std=position_std,
        )
        return jax.lax.dynamic_update_slice(out, block_result, (start, 0))

    out = jnp.zeros((n_padded, n_position_bins))
    out = jax.lax.fori_loop(0, n_blocks, process_block, out)
    return jnp.clip(out[:n_decoding_spikes], a_min=LOG_EPS)


def block_estimate_log_joint_mark_intensity(
    decoding_spike_waveform_features, ...
):
    """Public wrapper — plain Python, NOT JIT-wrapped.

    Handles the empty-array edge case (shape-level check) then dispatches
    to the JIT-compiled _impl. The _impl is what gets compiled and fused;
    the wrapper stays in Python so the `if n_decoding_spikes == 0` branch
    does not cause a ConcretizationTypeError.
    """
    n_decoding_spikes = decoding_spike_waveform_features.shape[0]
    n_position_bins = occupancy.shape[0]
    if n_decoding_spikes == 0:
        return jnp.full((0, n_position_bins), LOG_EPS)
    return _block_estimate_jit(...)  # calls the JIT-compiled _impl


# Only the _impl is JIT-compiled
_block_estimate_jit = jax.jit(
    _block_estimate_log_joint_mark_intensity_impl,
    static_argnames=("block_size", "use_gemm", "pos_tile_size",
                     "enc_tile_size", "use_streaming"),
)
```

### Step 4: Run tests

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_jit_block_estimate.py -v`
Run: `uv run pytest src/non_local_detector/tests/likelihoods/ src/non_local_detector/tests/integration/ -q --tb=line`

### Step 5: Profile and inspect jaxpr

Measure timing (with warmup + `block_until_ready`), inspect jaxpr for loop primitives, extract intermediate tensor sizes.

### Step 6: Commit

---

## Task 2: Precompute Reusable GEMM Quantities

`_compute_log_mark_kernel_gemm` recomputes `inv_sigma`, `log_norm_const`, `Y = enc_features * inv_sigma`, and `y2 = sum(Y^2)` on every call. Inside the `fori_loop` from Task 1, these are recomputed per decoding-spike block even though they depend only on the encoding features (constant per electrode).

**Files:**
- Modify: `src/non_local_detector/likelihoods/clusterless_kde_log.py`
  - Add `_precompute_encoding_gemm_quantities` function
  - Add `_compute_log_mark_kernel_from_precomputed` function
  - Modify `_block_estimate_impl` to precompute once outside the fori_loop
- Test: `src/non_local_detector/tests/likelihoods/test_jit_block_estimate.py` (extend)

### Step 1: Write the accuracy test

```python
class TestGemmPrecompute:
    """Verify precomputed GEMM quantities match full computation."""

    def test_precomputed_matches_full_gemm(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _compute_log_mark_kernel_gemm,
            _precompute_encoding_gemm_quantities,
            _compute_log_mark_kernel_from_precomputed,
        )
        rng = np.random.default_rng(42)
        n_enc, n_dec, n_wf = 1000, 200, 4
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)

        reference = _compute_log_mark_kernel_gemm(dec_wf, enc_wf, wf_std)
        precomp = _precompute_encoding_gemm_quantities(enc_wf, wf_std)
        result = _compute_log_mark_kernel_from_precomputed(dec_wf, precomp)

        # Exact same computation, just split — should be near-exact
        assert jnp.allclose(result, reference, atol=1e-10, rtol=1e-10)

    def test_precomputed_reused_across_blocks(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _compute_log_mark_kernel_gemm,
            _precompute_encoding_gemm_quantities,
            _compute_log_mark_kernel_from_precomputed,
        )
        rng = np.random.default_rng(42)
        n_enc, n_wf = 500, 4
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)
        precomp = _precompute_encoding_gemm_quantities(enc_wf, wf_std)

        for block_seed in range(5):
            dec_wf = jnp.array(
                np.random.default_rng(block_seed).standard_normal((100, n_wf)) * 50
            )
            reference = _compute_log_mark_kernel_gemm(dec_wf, enc_wf, wf_std)
            result = _compute_log_mark_kernel_from_precomputed(dec_wf, precomp)
            assert jnp.allclose(result, reference, atol=1e-10, rtol=1e-10)
```

### Step 2: Implement precompute functions

```python
def _precompute_encoding_gemm_quantities(
    encoding_features: jnp.ndarray,
    waveform_stds: jnp.ndarray,
) -> tuple[jnp.ndarray, float, jnp.ndarray, jnp.ndarray]:
    """Precompute encoding-side GEMM quantities constant per electrode.

    Returns (inv_sigma, log_norm_const, Y, y2) as a tuple (JAX pytree-compatible).
    """
    n_features = waveform_stds.shape[0]
    waveform_stds = jnp.clip(waveform_stds, a_min=EPS)
    inv_sigma = 1.0 / waveform_stds
    log_norm_const = -0.5 * (
        n_features * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(waveform_stds))
    )
    Y = encoding_features * inv_sigma[None, :]
    y2 = jnp.sum(Y ** 2, axis=1)
    return (inv_sigma, log_norm_const, Y, y2)


def _compute_log_mark_kernel_from_precomputed(
    decoding_features: jnp.ndarray,
    precomp: tuple[jnp.ndarray, float, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Compute log mark kernel using precomputed encoding quantities."""
    inv_sigma, log_norm_const, Y, y2 = precomp
    X = decoding_features * inv_sigma[None, :]
    x2 = jnp.sum(X ** 2, axis=1)
    cross_term = X @ Y.T
    sq_dist = jnp.maximum(y2[:, None] + x2[None, :] - 2.0 * cross_term.T, 0.0)
    return log_norm_const - 0.5 * sq_dist
```

**Note:** Returns a tuple, not a dict, for better JAX pytree compatibility inside scan/fori_loop carries.

### Step 3: Wire into block_estimate's fori_loop

Move the precompute call outside the `fori_loop` in `_block_estimate_impl`. The loop body calls `_compute_log_mark_kernel_from_precomputed` instead of the full GEMM.

### Step 4: Run tests, profile, inspect jaxpr

Verify precomputed arrays are NOT inside the fori_loop body (check jaxpr equation count — should decrease since encoding scaling is hoisted out).

### Step 5: Commit

---

## Task 3: Move Non-Local Electrode Loop Inside JIT

The Python electrode loop in `predict_clusterless_kde_log_likelihood` (lines 1432-1502) iterates over electrodes sequentially. Moving this inside JIT via `jax.lax.scan` eliminates ~64 Python dispatch round-trips.

**Scope: non-local path only.** The local path depends on `get_position_at_time()` (backed by `scipy.interpolate.interpn`, not JAX-traceable) and per-electrode `KDEModel.predict()` (Python object methods). These cannot be placed inside a `scan`. The local path stays as a Python loop. JIT-ing the local path would require replacing SciPy interpolation with a JAX-traceable equivalent — a separate future task.

**Challenges:**

1. Variable spike counts per electrode → pad to max and use validity masks
2. **Variable `n_wf` per electrode** (e.g., bad tetrode wire → fewer features) → two strategies, controlled by `pad_waveform_features` parameter (default `False`):
   - **`pad_waveform_features=False` (default, grouping):** Group electrodes by `n_wf`, run a separate `scan` per group. Exact — no numerical shift. Costs 2-3 JIT compilations (one per group) but preserves exact marginal likelihoods for EM convergence monitoring and model comparison.
   - **`pad_waveform_features=True` (padding):** Pad features and `waveform_stds` to `max_n_wf`. Adds a per-padded-dimension shift of `-0.5*log(2π) - log(σ)` to every mark kernel entry per spike. After `segment_sum`, this becomes a **spike-count-weighted shift per time bin**: `n_spikes_in_bin * C` (NOT a single constant). However, this shift is the same across all position bins within a time bin, so it cancels in the HMM's `_condition_on` normalization (posterior unaffected, max diff <5e-8). The marginal log-likelihood shifts by the spike-count-weighted sum. **Guard:** Padding must NOT push any electrode's `n_wf` across the `_COMPENSATED_LINEAR_MAX_FEATURES` threshold (currently 8). If `max_n_wf > 8` and any electrode has `n_wf <= 8`, padding would change that electrode from the compensated-linear path to the logsumexp path, altering both performance and numerics beyond a simple shift. In this case, fall back to grouping for that electrode. Simpler code path when applicable, one JIT compilation, negligible extra FLOPs (`n_wf << n_enc`).
3. `log_kde_distance` is `@jax.jit` → use `_log_kde_distance_impl` inside scan body
4. `get_spike_time_bin_ind` uses `np.digitize` → replace with `jnp.searchsorted(..., side='right')` (validated in Task 0)
5. Padded spikes must not contribute to segment_sum → assign to dummy bin `n_time` and use `num_segments=n_time` (ignores dummy bin)

**Memory design:** `log_position_distance` is computed per-electrode INSIDE the scan body (not batched across all electrodes). This keeps peak memory at single-electrode scale: `max_n_enc × n_pos × 4 bytes` (e.g., 20k × 200 × 4 = 15 MB). The alternative — batching log_position_distance as `(n_electrodes, max_n_enc, n_pos)` — would require `64 × 20k × 200 × 4 = 1 GB`, which fits on datacenter GPUs but not consumer GPUs.

**Files:**
- Modify: `src/non_local_detector/likelihoods/clusterless_kde_log.py`
  - Add `_pad_electrode_data` helper (pads spike counts, features, and waveform_stds to uniform shapes)
  - Add `_predict_nonlocal_electrodes_jit` function (scan over padded electrode batch)
  - Modify `predict_clusterless_kde_log_likelihood` to use scan for `is_local=False`, keep Python loop for `is_local=True`
- Test: `src/non_local_detector/tests/likelihoods/test_jit_electrode_loop.py` (new)

### Step 1: Write the accuracy test

```python
"""Tests for JIT-fused electrode loop."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    predict_clusterless_kde_log_likelihood,
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data


@pytest.fixture
def simulated_data():
    """Create simulated data with multiple electrodes."""
    sim = make_simulated_run_data(
        n_tetrodes=3,
        place_field_means=np.arange(0, 120, 10),
        sampling_frequency=500,
        n_runs=2,
        seed=42,
    )
    n_encode = int(0.7 * len(sim.position_time))
    encode_time = sim.position_time[:n_encode]
    encode_pos = sim.position[:n_encode]
    encode_spikes = [st[st <= encode_time[-1]] for st in sim.spike_times]
    encode_wf = [
        wf[st <= encode_time[-1]]
        for st, wf in zip(sim.spike_times, sim.spike_waveform_features, strict=False)
    ]
    encoding_model = fit_clusterless_kde_encoding_model(
        encode_time, encode_pos, encode_spikes, encode_wf,
        sim.environment, position_std=6.0, waveform_std=24.0,
        block_size=100, disable_progress_bar=True,
    )
    test_edges = sim.edges[sim.edges >= encode_time[-1]]
    test_spikes = [st[st >= encode_time[-1]] for st in sim.spike_times]
    test_wf = [
        wf[st >= encode_time[-1]]
        for st, wf in zip(sim.spike_times, sim.spike_waveform_features, strict=False)
    ]
    return {
        "test_edges": test_edges,
        "test_time": sim.position_time[n_encode:],
        "test_position": sim.position[n_encode:],
        "test_spikes": test_spikes,
        "test_wf": test_wf,
        "encoding_model": encoding_model,
    }


class TestJitElectrodeLoop:
    def test_nonlocal_matches_golden_reference(self, simulated_data):
        """Non-local likelihood matches pre-refactor golden reference.

        IMPORTANT: The reference must be saved BEFORE the refactor.
        After the refactor, both `reference` and `result` would call the
        same (new) code, making the test vacuous.

        Workflow:
        1. Before refactoring: run this test → it creates/saves the golden file
        2. After refactoring: run this test → it loads the golden and compares

        The golden file is saved as a pickle alongside the test file.
        """
        import pickle
        from pathlib import Path

        d = simulated_data
        golden_path = Path(__file__).parent / "golden_data" / "task3_nonlocal_reference.pkl"

        result = predict_clusterless_kde_log_likelihood(
            d["test_edges"], d["test_time"], d["test_position"],
            d["test_spikes"], d["test_wf"],
            **d["encoding_model"],
            is_local=False, disable_progress_bar=True,
        )

        if not golden_path.exists():
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            with open(golden_path, "wb") as f:
                pickle.dump(np.asarray(result), f)
            pytest.skip("Golden reference created — re-run after refactoring")

        with open(golden_path, "rb") as f:
            reference = pickle.load(f)

        assert result.shape == reference.shape
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result, reference, atol=1e-4, rtol=1e-4), (
            f"Max diff: {float(jnp.max(jnp.abs(result - np.asarray(reference)))):.2e}"
        )
```

### Step 2: Implement electrode data preparation and the scan body

**Default (`pad_waveform_features=False`):** Group electrodes by `n_wf` using a `_group_electrodes_by_n_wf` helper. Within each group, pad spike counts to `(max_n_enc_in_group, n_wf)` and `(max_n_dec_in_group, n_wf)`. Run a separate `scan` per group. Recombine results by summing each group's contribution to `log_likelihood`.

**Optional (`pad_waveform_features=True`):** Pad all electrodes to `(max_n_enc, max_n_wf)` and `(max_n_dec, max_n_wf)`. Electrodes with fewer features get zero-padded features and `waveform_stds` padded with the same `σ` value. Single `scan` over all electrodes.

The scan body (same for both modes):

1. Computes `log_position_distance` per electrode (inside scan, not batched)
2. Calls `_block_estimate_impl` (from Task 1)
3. Uses `jnp.searchsorted(..., side='right')` for spike binning
4. Assigns padded spikes to dummy bin `n_time` (out of range for `segment_sum`)
5. Accumulates into `log_likelihood`

### Step 4: Test, verify jaxpr, profile

Verify jaxpr contains `scan` over electrodes. Count kernel launches in the jaxpr.

### Step 5: Run full regression suite

### Step 6: Commit

---

## Task 4: Fuse segment_sum Into Block Loop

Instead of producing the full `(n_dec, n_pos)` mark intensity matrix then calling `segment_sum`, accumulate directly into `(n_time, n_pos)` inside the block fori_loop.

**Padding convention for segment_sum:** Padded spikes (beyond valid `n_dec`) are assigned segment ID `n_time` (a dummy bin). `jax.ops.segment_sum(..., num_segments=n_time)` ignores indices >= `num_segments`, so padded spikes contribute nothing.

**Files:**
- Modify: `src/non_local_detector/likelihoods/clusterless_kde_log.py`
- Test: extend `test_jit_block_estimate.py`

### Step 1: Write the test

```python
class TestFusedSegmentSum:
    def test_fused_matches_separate(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_log_joint_mark_intensity,
            _block_estimate_with_segment_sum,
        )
        from non_local_detector.likelihoods.common import get_spike_time_bin_ind

        rng = np.random.default_rng(42)
        n_enc, n_dec, n_pos, n_time = 1000, 500, 100, 200
        dec_wf, enc_wf, wf_std, occ, log_pos = (
            TestJitBlockEstimateAccuracy._make_electrode_data(rng, n_enc, n_dec, n_pos)
        )
        spike_times = np.sort(rng.uniform(0, 10, n_dec))
        time_edges = np.linspace(0, 10, n_time + 1)
        seg_ids = jnp.array(get_spike_time_bin_ind(spike_times, time_edges))

        # Separate: block_estimate then segment_sum
        mark_intensity = block_estimate_log_joint_mark_intensity(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, block_size=100,
        )
        reference = jax.ops.segment_sum(
            mark_intensity, seg_ids, num_segments=n_time, indices_are_sorted=True,
        )

        # Fused version
        result = _block_estimate_with_segment_sum(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos,
            seg_ids, n_time, block_size=100,
        )

        assert jnp.allclose(result, reference, atol=1e-10, rtol=1e-10)
```

### Step 2: Implement

The fused fori_loop body computes each block's mark intensity and immediately calls `segment_sum` on that block's segment IDs, accumulating into `(n_time, n_pos)`:

```python
def process_block(i, time_likelihood):
    start = i * block_size
    block_features = jax.lax.dynamic_slice(dec_wf_padded, (start, 0), (block_size, n_wf))
    block_seg_ids = jax.lax.dynamic_slice(seg_ids_padded, (start,), (block_size,))

    block_mark = _estimate_log_joint_mark_intensity_impl(
        block_features, enc_wf, wf_std, occ, mean_rate, log_pos, ...
    )
    block_mark = jnp.clip(block_mark, a_min=LOG_EPS)

    # Padded spikes have seg_id = n_time (dummy bin, ignored by segment_sum)
    block_contribution = jax.ops.segment_sum(
        block_mark, block_seg_ids, num_segments=n_time, indices_are_sorted=True,
    )
    return time_likelihood + block_contribution
```

### Step 3: Test, profile, commit

---

## Task 5: Auto-Select Tiling Parameters

**Files:**
- Modify: `src/non_local_detector/likelihoods/clusterless_kde_log.py`
- Test: `src/non_local_detector/tests/likelihoods/test_auto_tiling.py` (new)

### Step 1: Implement

```python
def _auto_select_tile_sizes(
    n_enc: int, n_dec: int, n_pos: int, n_wf: int,
    memory_budget_bytes: int | None = None,
) -> dict:
    """Auto-select tiling parameters based on memory budget."""
    if memory_budget_bytes is None:
        try:
            device = jax.devices()[0]
            stats = device.memory_stats()
            memory_budget_bytes = int(stats.get("bytes_limit", 2e9) * 0.25)
        except (AttributeError, KeyError):
            memory_budget_bytes = int(2e9)

    f32 = 4
    # block_size: controls (n_enc × block_size) mark kernel size
    max_block = max(memory_budget_bytes // (3 * n_enc * f32), 1)
    block_size = min(max_block, n_dec)

    # enc_tile_size: needed when logK_pos = n_enc × n_pos doesn't fit
    if n_enc * n_pos * f32 > memory_budget_bytes:
        enc_tile_size = max(memory_budget_bytes // (n_pos * f32 * 3), 256)
    else:
        enc_tile_size = None

    return {"block_size": block_size, "enc_tile_size": enc_tile_size}
```

### Step 2: Test with various scales, commit

---

## Execution Order and Dependencies

```
Task 0 (searchsorted equivalence)     ← prerequisite for Task 3
    ↓
Task 1 (block loop → fori_loop)       ← includes private-impl refactor
    ↓
Task 2 (precompute GEMM)              ← modifies Task 1's fori_loop body
    ↓
Task 4 (fuse segment_sum)             ← modifies Task 1's fori_loop body
    ↓
Task 3 (electrode loop → scan)        ← depends on Tasks 0, 1, 2; uses Task 4 if available
    ↓
Task 5 (auto-select tiling)           ← independent, can be done anytime after Task 1
```

Tasks 0-2 are the highest priority — they give the most speedup for the least risk. Task 4 is a natural extension of Task 1. Task 3 is the largest change and biggest GPU win. Task 5 is convenience/polish.
