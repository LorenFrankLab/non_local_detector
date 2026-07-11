# clusterless_diffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in clusterless likelihood `clusterless_diffusion` that estimates the marked-point-process intensity of `clusterless_kde` but replaces the Gaussian *position* kernel with the environment's graph heat kernel `exp(-tL)` — for geometry-respecting spatial smoothing (goal A) and speed via a cached low-rank operator (goal B).

**Architecture:** New module `likelihoods/clusterless_diffusion.py` with a fit + predict pair, registered in `_CLUSTERLESS_ALGORITHMS`. The mark kernel stays `kde_distance` (Gaussian over waveform features); the spatial smoother becomes a JAX low-rank heat-kernel matmul (`heat_kernel_apply`) over the same interior-bin eigenbasis that `sorted_spikes_diffusion` uses (cached on `Environment`). Occupancy and the ground-process field are diffused once at fit; only the mark-weighted, EM-weighted per-spike histogram `D_e` is diffused at predict.

**Tech Stack:** Python, JAX (`jnp`, jit), NumPy/SciPy (host eigenbasis), the existing `likelihoods/diffusion.py` engine, `pytest`.

**Design spec:** `docs/superpowers/specs/2026-07-11-clusterless-diffusion-design.md` (commit `6147b6e`). Read it first — this plan implements it verbatim.

## Global Constraints

- **Branch:** `clusterless-diffusion` (already checked out; spec committed).
- **Run everything via `uv run`** (`uv run pytest ...`, `uv run ruff ...`). Never bare `python`/`pytest`.
- **Default unchanged:** `clusterless_kde` stays the default clusterless algorithm. This feature is purely additive/opt-in via `clusterless_algorithm="clusterless_diffusion"`. Existing golden/snapshot outputs MUST NOT change.
- **float32 regime:** the package is float32-explicit (no global x64). Device basis dtype is float32.
- **Numerical constants (verbatim from spec):** `memory_budget` default = `536_870_912` bytes (512 MiB); block-memory factor `c = 4`; `safety = 2`; `LOG_EPS`/`EPS` from `likelihoods.common`.
- **Marked-point-process contract (verbatim):** a zero-rate electrode (`weight_total_e == 0`: zero EM weights or zero encoding spikes) is floored to `LOG_EPS` per observed decode spike (NOT skipped) and adds 0 ground-process intensity — matching `clusterless_gmm`.
- **Density normalization (verbatim):** `p_e(x, m_j) = heat_kernel_apply(D_e)[x,j] / ((Σ_i w_i)·ΔV(x))`; mark normalizer lives in `kde_distance` only; **no** per-column unit-integral (must not use `to_density`); `ΔV(x)` cancels in every `p/π` likelihood ratio.
- **`heat_kernel_apply` = a JAX port of `diffusion.diffuse`:** `Q (exp(-tΛ) ⊙ (Qᵀ F))`, then clip `≥0`, then rescale each column to input mass **per graph component**. Clip-only is a bug (breaks mass conservation).
- **Commit discipline:** commit after each task's tests pass. End commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do NOT push or touch golden/snapshot data.

---

## File Structure

- **Create** `src/non_local_detector/likelihoods/clusterless_diffusion.py` — fit + predict; the algorithm.
- **Modify** `src/non_local_detector/likelihoods/diffusion.py` — add `heat_kernel_apply` (JAX) and `get_device_basis` (device cache helper).
- **Modify** `src/non_local_detector/environment.py` — invalidate the device cache in `fit_place_grid`; add `__getstate__` dropping the transient device cache.
- **Modify** `src/non_local_detector/likelihoods/__init__.py` — register `"clusterless_diffusion"` and export fit/predict.
- **Modify** `CHANGELOG.md` — Added entry.
- **Create tests** under `src/non_local_detector/tests/likelihoods/`:
  - `test_clusterless_diffusion_spike.py` (Task 0 — the de-risking spike; deleted/folded at the end)
  - `test_clusterless_diffusion.py` (density-correctness, weighted-EM, degeneracy, local, agreement, geometry)
  - `test_diffusion_device_cache.py` (device-cache helper + lifecycle)
  - `test_clusterless_diffusion_integration.py` (registry + end-to-end)

Reused helpers (do not reimplement): `common.kde_distance`? → **`kde_distance` lives in `clusterless_kde.py:25`**; import it. `common.{validate_weights, validate_finite, interpolate_weights_at_spike_times, weighted_mean_rate, get_position_at_time, get_spike_time_bin_ind, as_std_array, EPS, LOG_EPS, safe_log}`. `sorted_spikes_diffusion.{pixellate_interior_fields, _interior_bin_indices, _full_to_local}` for histogramming positions → interior bins. `diffusion.{environment_graph, cached_eigenbasis, cached_heat_kernel_eigenbasis, diffuse}`.

---

## Task 0: De-risking spike — non-local correctness + perf vs `clusterless_kde`

**Purpose:** goals A and B rest on two hypotheses — (1) the heat-kernel + `Σw·ΔV` normalization reproduces `clusterless_kde`'s posterior on simple geometry, and (2) the cached low-rank matmul is faster than KDE's pairwise position kernel on a large grid. Validate both on a *minimal* non-local path before building the production module. **If either fails, STOP and revisit the spec** (do not proceed to Task 1).

This task is intentionally throwaway-quality (a single scratch module + script). No blocking, no device cache, no local path, no validators.

**Files:**
- Create: `src/non_local_detector/likelihoods/tests/likelihoods/test_clusterless_diffusion_spike.py` → actually `src/non_local_detector/tests/likelihoods/test_clusterless_diffusion_spike.py`

**Interfaces produced:** confirmed values of `position_std → σ`, the exact normalization, and a measured speedup — feeds Tasks 3–4.

- [ ] **Step 1: Write the spike test (agreement + timing)**

```python
# src/non_local_detector/tests/likelihoods/test_clusterless_diffusion_spike.py
"""SPIKE (Task 0): validate clusterless_diffusion non-local math + perf vs KDE.
Throwaway; folded into the production suite by the final task."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.clusterless_kde import kde_distance
from non_local_detector.likelihoods.common import LOG_EPS, get_spike_time_bin_ind
from non_local_detector.likelihoods.diffusion import cached_eigenbasis, environment_graph


def _sim(seed=0, n_bins_side=10):
    rng = np.random.default_rng(seed)
    dt = 0.02
    n_time = 60
    time = np.arange(n_time) * dt
    t_end = float(time[-1])
    pt = np.linspace(0.0, t_end, 200)
    pos = np.column_stack([np.linspace(0, 10, pt.size),
                           np.sin(np.linspace(0, 2 * np.pi, pt.size)) * 2])
    n_features = 2
    enc_t = [np.sort(rng.uniform(0, t_end, 60)) for _ in range(3)]
    enc_f = [rng.standard_normal((t.size, n_features)).astype(np.float32) for t in enc_t]
    dec_t = [np.sort(rng.uniform(0, t_end, 20)) for _ in range(3)]
    dec_f = [rng.standard_normal((t.size, n_features)).astype(np.float32) for t in dec_t]
    env = Environment(position_range=[(0, 10), (-3, 3)],
                      place_bin_size=20.0 / n_bins_side)
    env = env.fit_place_grid(position=pos, infer_track_interior=True)
    return dict(time=time, pt=pt, pos=pos, enc_t=enc_t, enc_f=enc_f,
                dec_t=dec_t, dec_f=dec_f, env=env)


def _diffusion_nonlocal_spike(s, position_std=6.0, waveform_std=24.0, low_rank=False):
    """Minimal non-local clusterless_diffusion, split into fit (run once) + a
    ``predict`` closure (the part goal B times).

    ``low_rank=True`` uses the **bandwidth-aware truncated** basis
    (``cached_heat_kernel_eigenbasis``) that goal B actually depends on;
    ``low_rank=False`` uses the full-rank basis (correctness reference).
    Returns the ``predict`` closure (call it to get a jnp ``(n_time, n_bins)``).
    """
    from non_local_detector.likelihoods.diffusion import cached_heat_kernel_eigenbasis
    from non_local_detector.likelihoods.sorted_spikes_diffusion import (
        _full_to_local, _interior_bin_indices)
    from non_local_detector.likelihoods.common import get_position_at_time, weighted_mean_rate
    env = s["env"]
    graph, node_order, _ = environment_graph(env)
    if low_rank:
        eigvals, eigvecs = cached_heat_kernel_eigenbasis(env, position_std)  # bandwidth-aware low rank
    else:
        eigvals, eigvecs = cached_eigenbasis(env, rank=None)                 # full rank
    eigvals = jnp.asarray(eigvals); Q = jnp.asarray(eigvecs)
    interior = env.is_track_interior_.ravel()
    n_bins = Q.shape[0]
    dV = 1.0  # uniform grid in the spike; exact ΔV = bin_sizes in production
    coeff = jnp.exp(-(position_std ** 2 / 2.0) * eigvals)

    def diffuse_cols(F):  # single-component clip+rescale-to-input-mass
        sm = Q @ (coeff[:, None] * (Q.T @ F))
        cl = jnp.clip(sm, 0.0, None)
        scale = jnp.where(cl.sum(0) > 0, F.sum(0) / jnp.where(cl.sum(0) > 0, cl.sum(0), 1.0), 0.0)
        return cl * scale

    f2l = _full_to_local(node_order, interior.shape[0])
    def bins_of(p):
        return _interior_bin_indices(env, p, f2l)

    # --- fit (run once; not timed) ---
    occ_pos = get_position_at_time(s["pt"], s["pos"], s["pt"], env)
    O = jnp.zeros(n_bins).at[jnp.asarray(bins_of(occ_pos))].add(1.0)
    w_pos = float(len(occ_pos))
    pi = jnp.clip(diffuse_cols(O[:, None])[:, 0] / (w_pos * dV), 1e-15, None)
    per_e, summed_gpi = [], jnp.zeros(n_bins)
    for e in range(len(s["enc_t"])):
        enc_bins = jnp.asarray(bins_of(get_position_at_time(s["pt"], s["pos"], s["enc_t"][e], env)))
        enc_marks = jnp.asarray(s["enc_f"][e]); n_enc = enc_marks.shape[0]
        mean_rate = weighted_mean_rate(np.ones(n_enc), w_pos)
        S = jnp.zeros(n_bins).at[enc_bins].add(1.0)
        summed_gpi = summed_gpi + mean_rate * diffuse_cols(S[:, None])[:, 0] / (n_enc * dV) / pi
        per_e.append((enc_bins, enc_marks, n_enc, mean_rate,
                      jnp.asarray(s["dec_f"][e]),
                      jnp.asarray(get_spike_time_bin_ind(s["dec_t"][e], s["time"]))))
    n_dec_time = s["time"].shape[0]

    def predict():  # decode-only — this is what goal B times
        ll = jnp.zeros((n_dec_time, n_bins)) - summed_gpi[None, :]
        for (enc_bins, enc_marks, n_enc, mean_rate, dm, seg) in per_e:
            K = kde_distance(dm, enc_marks, jnp.full(2, waveform_std))  # already exp'd — no jnp.exp
            D = jnp.zeros((n_bins, dm.shape[0])).at[enc_bins].add(K)    # scatter (uniform w=1)
            p_e = diffuse_cols(D) / (n_enc * dV)
            lc = jnp.log(jnp.clip(mean_rate * p_e / pi[:, None], jnp.exp(LOG_EPS), None))
            ll = ll + jnp.zeros((n_dec_time, n_bins)).at[seg].add(lc.T)
        return ll  # jnp device array

    return predict


@pytest.mark.slow
def test_spike_agreement_and_speed():
    s = _sim(seed=0)
    ll_diff = np.asarray(_diffusion_nonlocal_spike(s)())  # full-rank correctness reference; call the closure
    # KDE reference
    enc = clusterless_kde.fit_clusterless_kde_encoding_model(
        jnp.asarray(s["pt"]), jnp.asarray(s["pos"]),
        [jnp.asarray(t) for t in s["enc_t"]], [jnp.asarray(f) for f in s["enc_f"]],
        s["env"], sampling_frequency=50, position_std=6.0, waveform_std=24.0,
        block_size=100, disable_progress_bar=True)
    ll_kde = np.asarray(clusterless_kde.predict_clusterless_kde_log_likelihood(
        jnp.asarray(s["time"]), jnp.asarray(s["pt"]), jnp.asarray(s["pos"]),
        [jnp.asarray(t) for t in s["dec_t"]], [jnp.asarray(f) for f in s["dec_f"]],
        **enc, is_local=False))
    assert np.all(np.isfinite(ll_diff))
    # posterior agreement per time bin (rank correlation of the two log-likelihood rows)
    rhos = [stats.spearmanr(ll_diff[t], ll_kde[t]).statistic
            for t in range(ll_diff.shape[0]) if np.ptp(ll_kde[t]) > 0]
    median_rho = float(np.nanmedian(rhos))
    print(f"[SPIKE] median per-timebin Spearman(diffusion, kde) = {median_rho:.3f}")
    assert median_rho > 0.6, "diffusion posterior does not track KDE on simple geometry"
```

- [ ] **Step 2: Run it, read the printed agreement**

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_clusterless_diffusion_spike.py -v -s -m slow`
Expected: PASS with a printed median Spearman > 0.6. If it fails badly (≈0 or negative), the normalization/kernel mapping is wrong — **stop and revisit §1/§4 of the spec** before continuing.

- [ ] **Step 3: Add a rough timing comparison (large grid)**

Append to the test file:

```python
def _time(fn, iters=5):
    jax.block_until_ready(fn())            # warm compile, not timed
    t0 = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn())        # block on the DEVICE output each iter
    return (time.perf_counter() - t0) / iters


@pytest.mark.slow
def test_spike_speed_large_grid():
    # Genuinely large matched grid: place_bin_size = 20/160 = 0.125 -> ~80x48 grid.
    s = _sim(seed=1, n_bins_side=160)
    n_bins = s["env"].place_bin_centers_.shape[0]
    assert n_bins > 2000, f"grid too small for a meaningful perf gate: {n_bins} bins"

    dec_t = [jnp.asarray(t) for t in s["dec_t"]]
    dec_f = [jnp.asarray(f) for f in s["dec_f"]]
    tm, pt, po = jnp.asarray(s["time"]), jnp.asarray(s["pt"]), jnp.asarray(s["pos"])

    # KDE: fit ONCE, then time PREDICT-ONLY (returns a jnp device array).
    enc = clusterless_kde.fit_clusterless_kde_encoding_model(
        pt, po, [jnp.asarray(t) for t in s["enc_t"]], [jnp.asarray(f) for f in s["enc_f"]],
        s["env"], sampling_frequency=50, position_std=6.0, waveform_std=24.0,
        block_size=10_000, disable_progress_bar=True)
    kde_predict = lambda: clusterless_kde.predict_clusterless_kde_log_likelihood(
        tm, pt, po, dec_t, dec_f, **enc, is_local=False)
    dt_kde = _time(kde_predict)

    # Diffusion: fit ONCE with the BANDWIDTH-AWARE LOW-RANK basis (goal B), then time
    # the PREDICT-ONLY closure — apples-to-apples with KDE predict.
    diff_predict = _diffusion_nonlocal_spike(s, low_rank=True)  # returns predict()
    dt_diff = _time(diff_predict)

    print(f"[SPIKE] {n_bins} bins | KDE predict: {dt_kde*1e3:.1f} ms | "
          f"diffusion predict (low-rank): {dt_diff*1e3:.1f} ms | speedup: {dt_kde/dt_diff:.2f}x")
    # Informational gate: record both numbers; decide goal B in review.
```

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_clusterless_diffusion_spike.py::test_spike_speed_large_grid -v -s -m slow`
Expected: PASS; both predict-only timings printed at a >2000-bin grid. **Goal-B decision (review gate):** the low-rank diffusion predict should be clearly faster than KDE predict at large `n_bins`. If not, discuss before the full build — the low-rank operator is the whole basis for goal B.

- [ ] **Step 4: Commit the spike**

```bash
git add src/non_local_detector/tests/likelihoods/test_clusterless_diffusion_spike.py
git commit -m "spike: validate clusterless_diffusion non-local math + perf vs KDE

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

> **GATE:** Do not start Task 1 until Step 2 passes and the Step 3 timing has been reviewed against KDE. The spike encodes the two hypotheses the whole feature depends on.

---

## Task 1: `heat_kernel_apply` — JAX port of `diffusion.diffuse`

**Files:**
- Modify: `src/non_local_detector/likelihoods/diffusion.py` (add function near `diffuse`)
- Test: `src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py`

**Interfaces:**
- Produces: `heat_kernel_apply(eigvals: jnp.ndarray, eigvecs: jnp.ndarray, sigma: float, fields: jnp.ndarray, component_labels: jnp.ndarray | None) -> jnp.ndarray` — same shapes/semantics as `diffuse` but pure JAX (clip + per-component rescale-to-input-mass). Consumed by Tasks 3–5.

- [ ] **Step 1: Write the failing test (parity with `diffuse` incl. truncation lobes)**

```python
# src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py
import jax.numpy as jnp
import numpy as np
import networkx as nx
from numpy.testing import assert_allclose

from non_local_detector.likelihoods.diffusion import (
    build_laplacian, diffusion_eigenbasis, diffuse, heat_kernel_apply)


def _grid_basis(n=6, rank=None):
    g = nx.grid_2d_graph(n, n)
    g = nx.convert_node_labels_to_integers(g)
    for u, v in g.edges():
        g[u][v]["distance"] = 1.0
    L = build_laplacian(g)
    vals, vecs = diffusion_eigenbasis(L, rank=rank)
    return vals, vecs


def test_heat_kernel_apply_matches_diffuse_full_and_truncated():
    for rank in (None, 8):  # full-rank (no lobes) and truncated (negative lobes)
        vals, vecs = _grid_basis(n=6, rank=rank)
        rng = np.random.default_rng(0)
        fields = np.abs(rng.standard_normal((vecs.shape[0], 4)))  # point-ish sources
        ref = diffuse(vals, vecs, sigma=2.0, fields=fields, component_labels=None)
        got = np.asarray(heat_kernel_apply(
            jnp.asarray(vals), jnp.asarray(vecs), 2.0, jnp.asarray(fields), None))
        assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
        # mass conservation regardless of rank
        assert_allclose(got.sum(0), np.asarray(fields).sum(0), rtol=1e-5, atol=1e-6)


def test_heat_kernel_apply_disconnected_components_preserve_mass_independently():
    """Two disjoint grids with UNEQUAL masses: each component's mass is preserved
    separately (a single global rescale would leak mass between them)."""
    import scipy.sparse.csgraph
    g = nx.disjoint_union(nx.grid_2d_graph(4, 4), nx.grid_2d_graph(3, 3))
    g = nx.convert_node_labels_to_integers(g)
    for u, v in g.edges():
        g[u][v]["distance"] = 1.0
    L = build_laplacian(g)
    n_comp, labels = scipy.sparse.csgraph.connected_components(L, directed=False)
    assert n_comp == 2
    vals, vecs = diffusion_eigenbasis(L, rank=6)  # truncated -> point sources produce negative lobes
    # Point sources (one hot bin per component) with very different masses -> lobes
    # and component-specific rescaling both guaranteed to matter.
    comp0 = np.flatnonzero(labels == 0)
    comp1 = np.flatnonzero(labels == 1)
    fields = np.zeros((vecs.shape[0], 2), dtype=float)
    fields[comp0[0], 0] = 10.0   # heavy point source in component 0
    fields[comp1[0], 1] = 1.0    # light point source in component 1
    ref = diffuse(vals, vecs, sigma=1.5, fields=fields, component_labels=labels)
    got = np.asarray(heat_kernel_apply(
        jnp.asarray(vals), jnp.asarray(vecs), 1.5, jnp.asarray(fields), labels))
    assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    # per-component mass preserved independently
    for comp in (0, 1):
        m = labels == comp
        assert_allclose(got[m].sum(0), fields[m].sum(0), rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py::test_heat_kernel_apply_matches_diffuse_full_and_truncated -v`
Expected: FAIL — `heat_kernel_apply` not defined.

- [ ] **Step 3: Implement `heat_kernel_apply`** (add to `diffusion.py`; import `jax.numpy as jnp` is already present via the engine? if not, add `import jax.numpy as jnp`)

```python
from functools import partial
import jax


@partial(jax.jit, static_argnames=("sigma", "n_components"))
def _heat_kernel_apply_jit(eigvals, eigvecs, sigma, fields, labels, n_components):
    t = sigma ** 2 / 2.0
    coeff = jnp.exp(-t * eigvals)                                   # (m,)
    smoothed = eigvecs @ (coeff[:, None] * (eigvecs.T @ fields))    # (n_bins, n_fields)
    clipped = jnp.clip(smoothed, 0.0, None)
    # Vectorized per-component mass rescale (jittable; labels are segment ids).
    in_mass = jax.ops.segment_sum(fields, labels, num_segments=n_components)   # (n_components, n_fields)
    cl_mass = jax.ops.segment_sum(clipped, labels, num_segments=n_components)
    scale = jnp.where(cl_mass > 0, in_mass / jnp.where(cl_mass > 0, cl_mass, 1.0), 0.0)
    return clipped * scale[labels]                                  # gather back to (n_bins, n_fields)


def heat_kernel_apply(eigvals, eigvecs, sigma, fields, component_labels=None,
                      n_components=None):
    """JAX port of ``diffuse``: exp(-tL) F, clipped and per-component mass-rescaled.

    Mirrors :func:`diffuse` exactly (clip to >=0, then rescale each column to its
    input mass, per connected component) so likelihood magnitude is truncation-rank
    stable. Fully vectorized/jittable. ``component_labels`` are 0..K-1 segment ids
    (a device/host int array; a single connected graph passes all zeros or None).

    ``n_components`` is the **static** component count — pass it explicitly (the
    production caller gets it from :func:`get_device_basis`) to avoid a per-call
    device->host sync of the labels. If ``None``, it is derived host-side once
    (convenience for tests / one-off calls only — NOT the hot predict path).
    """
    n_bins = eigvecs.shape[0]
    if component_labels is None:
        labels = jnp.zeros(n_bins, dtype=jnp.int32)
        n_components = 1
    else:
        labels = jnp.asarray(component_labels, dtype=jnp.int32)
        if n_components is None:
            n_components = int(np.asarray(component_labels).max()) + 1  # host sync — non-hot path only
    return _heat_kernel_apply_jit(eigvals, eigvecs, float(sigma), fields, labels, int(n_components))
```

> `n_components` is a **static** arg, so `segment_sum(num_segments=n_components)` has a fixed shape under `jit`; `labels` are device ints used as segment ids + gather (no Python loop, no `np.unique`). The predict path passes `n_components` from `get_device_basis`, so no labels ever leave the device per block.

- [ ] **Step 4: Run to verify pass (BOTH tests, incl. the disconnected-component one)**

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py -k heat_kernel -v`
Expected: PASS — both `test_heat_kernel_apply_matches_diffuse_full_and_truncated` **and** `test_heat_kernel_apply_disconnected_components_preserve_mass_independently` (the latter is what actually exercises per-component rescaling).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check --fix src/non_local_detector/likelihoods/diffusion.py
uv run ruff format src/non_local_detector/likelihoods/diffusion.py
git add src/non_local_detector/likelihoods/diffusion.py src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py
git commit -m "feat(diffusion): add heat_kernel_apply (JAX port of diffuse)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Device-basis cache + lifecycle (`get_device_basis`, invalidation, `__getstate__`)

**Files:**
- Modify: `src/non_local_detector/likelihoods/diffusion.py` (`get_device_basis`)
- Modify: `src/non_local_detector/environment.py` (`fit_place_grid` invalidation + `__getstate__`)
- Test: `src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py`

**Interfaces:**
- Produces: `get_device_basis(environment, resolved_rank) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]` returning device `(eigvals, eigvecs, component_labels, n_components)` — labels are a **device `jnp.ndarray`**, `n_components` a Python `int` (static, computed once). Cached on `environment._diffusion_device_basis_` keyed by `(resolved_rank, device, dtype)`; miss-only host retrieval via `cached_eigenbasis`. Consumed by Tasks 4–5, which pass `n_components` straight into `heat_kernel_apply` (no per-block label sync).

- [ ] **Step 1: Write failing tests (cache hit/miss, refit invalidation, pickle exclusion)**

```python
# append to test_diffusion_device_cache.py
import pickle
import jax
import numpy as np
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.diffusion import get_device_basis, environment_graph


def _fitted_env():
    env = Environment(position_range=[(0, 10)], place_bin_size=1.0)
    return env.fit_place_grid(position=np.linspace(0, 10, 50)[:, None],
                              infer_track_interior=True)


def test_get_device_basis_hit_and_device_arrays():
    env = _fitted_env()
    _, vecs = __import__("non_local_detector.likelihoods.diffusion",
                         fromlist=["cached_eigenbasis"]).cached_eigenbasis(env, None)
    rank = vecs.shape[1]
    b1 = get_device_basis(env, rank)
    b2 = get_device_basis(env, rank)
    assert isinstance(b1[1], jax.Array)  # device eigvecs
    assert b1[1] is b2[1]  # cache hit returns same object


def test_device_cache_invalidated_on_refit():
    env = _fitted_env()
    rank = get_device_basis(env, __import__(
        "non_local_detector.likelihoods.diffusion",
        fromlist=["cached_eigenbasis"]).cached_eigenbasis(env, None)[1].shape[1])[1].shape[1]
    assert hasattr(env, "_diffusion_device_basis_")
    env.fit_place_grid(position=np.linspace(0, 20, 80)[:, None], infer_track_interior=True)
    assert not hasattr(env, "_diffusion_device_basis_")


def test_device_cache_excluded_from_pickle():
    env = _fitted_env()
    get_device_basis(env, __import__(
        "non_local_detector.likelihoods.diffusion",
        fromlist=["cached_eigenbasis"]).cached_eigenbasis(env, None)[1].shape[1])
    blob = pickle.dumps(env)  # must not raise on the JAX Device key
    env2 = pickle.loads(blob)
    assert not hasattr(env2, "_diffusion_device_basis_")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py -k "device" -v`
Expected: FAIL (`get_device_basis` undefined; no invalidation; pickle may currently succeed but attribute persists).

- [ ] **Step 3: Implement `get_device_basis` in `diffusion.py`**

```python
def get_device_basis(environment, resolved_rank):
    """Device (jnp) eigenbasis for the clusterless_diffusion predict matmul.

    Transient cache on ``environment._diffusion_device_basis_`` keyed by
    ``(resolved_rank, device, dtype)``. On a miss, retrieve the host basis via
    ``cached_eigenbasis`` and convert once. Excluded from pickling and invalidated
    on grid refit (see Environment).
    """
    import jax
    import jax.numpy as jnp
    # The device the arrays will actually live on. jnp.asarray follows the active
    # default-device context, so select it explicitly and device_put onto it — the
    # cache key must name the SAME device the cached arrays are placed on.
    device = jax.local_devices()[0] if jax.config.jax_default_device is None \
        else jax.config.jax_default_device
    dtype = jnp.float32
    key = (int(resolved_rank), device, dtype)
    cache = getattr(environment, "_diffusion_device_basis_", None)
    if cache is None:
        cache = {}
        environment._diffusion_device_basis_ = cache
    if key not in cache:
        eigvals, eigvecs = cached_eigenbasis(environment, resolved_rank)
        # Component labels from the existing helper (indexed by interior-bin/node
        # order) — do NOT rebuild the Laplacian from the exposed frozen graph.
        graph, _, _ = environment_graph(environment)
        labels_host = connected_component_labels(graph)          # np.ndarray, 0..K-1
        n_components = int(labels_host.max()) + 1                 # static, computed once
        cache[key] = (
            jax.device_put(np.asarray(eigvals, dtype=np.float32), device),
            jax.device_put(np.asarray(eigvecs, dtype=np.float32), device),
            jax.device_put(np.asarray(labels_host, dtype=np.int32), device),
            n_components,
        )
    return cache[key]
```

- [ ] **Step 4: Add invalidation + `__getstate__` in `environment.py`**

In `fit_place_grid`, inside the existing block that `del`s `_diffusion_graph_` / `_diffusion_laplacian_` / `_diffusion_eigenbasis_` / `_diffusion_heat_kernel_rank_`, append:

```python
        if hasattr(self, "_diffusion_device_basis_"):
            del self._diffusion_device_basis_
```

Add to the `Environment` class (transient device cache excluded from pickling):

```python
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_diffusion_device_basis_", None)  # transient JAX device cache
        return state
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py -k "device" -v`
Expected: PASS (all three).

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check --fix src/non_local_detector/likelihoods/diffusion.py src/non_local_detector/environment.py
uv run ruff format src/non_local_detector/likelihoods/diffusion.py src/non_local_detector/environment.py
git add src/non_local_detector/likelihoods/diffusion.py src/non_local_detector/environment.py src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py
git commit -m "feat(diffusion): device-basis cache with refit invalidation + pickle exclusion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `fit_clusterless_diffusion_encoding_model`

**Files:**
- Create: `src/non_local_detector/likelihoods/clusterless_diffusion.py`
- Test: `src/non_local_detector/tests/likelihoods/test_clusterless_diffusion.py`

**Interfaces:**
- Produces `fit_clusterless_diffusion_encoding_model(position_time, position, spike_times, spike_waveform_features, environment, *, sampling_frequency=500, position_std=..., waveform_std=..., weights=None, heat_kernel_rank=None, block_size=10_000, memory_budget=536_870_912, disable_progress_bar=False, **kwargs) -> dict`. Returned dict keys (consumed by Task 4/5 predict): `environment, occupancy` (π, `(n_interior,)`), `summed_ground_process_intensity` `(n_interior,)`, `encoding_bin_indices` (list per electrode), `encoding_marks` (list), `encoding_weights` (list), `weight_total` (list of float), `mean_rates` (list), `resolved_rank` (int), `node_order`, `bin_sizes` (ΔV per interior bin), `position_std`, `waveform_std`, `block_size`, `memory_budget`, `disable_progress_bar`.

This task uses the **spec §2 contract verbatim** (weighted histograms; `π = H·O/(Σw_pos·ΔV)`; per-electrode zero-rate marking with warning; `resolved_rank = eigvecs.shape[1]`). Reuse `sorted_spikes_diffusion.pixellate_interior_fields` semantics but keep marks + per-spike bin indices. Full implementation code, the density-correctness tests that pin it, and its TDD steps are specified in the spec's §2/§4 and the tests in Task 6 (write those tests first per TDD; this task delivers the code that makes them pass). Break Task 3 into: (3a) validators + eigenbasis resolution + occupancy π; (3b) per-electrode marks/bins/weights/mean_rate + zero-rate marking; (3c) `summed_ground_process_intensity`. Commit after each sub-step's targeted test passes.

> **TDD ordering (important):** the density-correctness contracts listed in Task 6 (`test_mark_marginal_recovery`, `test_absolute_log_likelihood_small_fixture`, `test_mass_invariance_across_ranks`, `test_binary_weights_match_hard_subset`, `test_zero_rate_fit_and_predict_finite_and_warn`) are the **RED tests for Tasks 3–4** — write them *before* implementing the fit/predict bodies (they fail with "module not found"), then implement 3a–3c and Task 4 until they pass, committing per sub-step. Task 6 is where they are consolidated/confirmed as a suite; they are authored here, not deferred. This keeps the numerical body validated by executable contracts, not prose.

- [ ] **Step 1:** Write `test_fit_returns_finite_densities_and_keys` (fit on `_sim`-style data; assert dict has every key above; `occupancy`/`summed_ground_process_intensity` finite and non-negative; `resolved_rank` int; per-electrode list lengths equal n_electrodes).
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement 3a–3c per spec §2. Exact contracts:
  - **ΔV:** `graph, node_order, bin_sizes = environment_graph(environment)`. Use `bin_sizes` (shape `(n_interior,)`, already aligned with `node_order`, correct for uniform grids **and** linearized tracks) as `ΔV`. Do **not** hand-roll `np.diff(env.edges_)`.
  - **Occupancy (floored):** `O = weighted histogram of occupancy positions onto interior bins`; `Ohat = heat_kernel_apply(Λ, Q, position_std, O[:,None], labels)[:,0]`; then
    `pi = clip(Ohat / (where(Σw_pos>0, Σw_pos, 1.0) * bin_sizes), EPS, None)` — the `EPS` floor is applied to `pi` before it is ever a denominator. If `Σw_pos == 0`, emit the diffusion-specific `UserWarning`.
  - **Validators:** `common.validate_weights`, `common.validate_finite` on in-window features; positive `position_std`/`waveform_std`; `heat_kernel_rank`/`memory_budget`/`block_size` per Task 7 (Task 3 may raise-forward or Task 7 adds them — keep the checks in one `_validate_*` helper).
  - **Per electrode:** in-window clip; `encoding_bin_indices` via `_interior_bin_indices`; `encoding_weights` via `interpolate_weights_at_spike_times`; `weight_total = float(w.sum())`; if `weight_total == 0` → store `weight_total=0`, `mean_rate=0.0`, `warn`, contribute **0** to `summed_ground_process_intensity`; else `mean_rate = weighted_mean_rate(w, weight_sum)` and add `mean_rate * (heat_kernel_apply(S_e) / (weight_total*bin_sizes)) / pi`.
  - **Rank:** resolve the basis once (`cached_heat_kernel_eigenbasis(env, position_std)` if `heat_kernel_rank is None` else `cached_eigenbasis(env, heat_kernel_rank)`); `resolved_rank = eigvecs.shape[1]`; store it. Predict retrieves the device basis via `get_device_basis(env, resolved_rank)`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(clusterless_diffusion): fit encoding model`.

---

## Task 4: `predict_clusterless_diffusion_log_likelihood` — non-local path + memory policy

**Files:**
- Modify: `src/non_local_detector/likelihoods/clusterless_diffusion.py`
- Test: `src/non_local_detector/tests/likelihoods/test_clusterless_diffusion.py`

**Interfaces:**
- Produces `predict_clusterless_diffusion_log_likelihood(time, position_time, position, spike_times, spike_waveform_features, *, is_local=False, **encoding_model) -> jnp.ndarray` shape `(n_time, n_bins)` (non-local). Consumes Task 1 `heat_kernel_apply`, Task 2 `get_device_basis`, Task 3 dict.

Implements spec §3 non-local pseudocode **verbatim**: per electrode, zero-rate guard first (`weight_total_e == 0` → `LOG_EPS` per in-window decode spike via scatter-add, `continue`); else weighted `D_e` (`scatter (w_i·K)`), `heat_kernel_apply`, `p_e = P_e/(weight_total_e·ΔV)`, `safe_log(μ_e·p_e/π)`, `segment_sum`. Memory policy: `effective_block = clip(floor(memory_budget/(itemsize·(n_enc+rank+4·n_bins)·2)), 1, block_size)`, block over decode spikes, log the effective block.

- [ ] **Step 1:** Write `test_nonlocal_finite_and_zero_rate` (finite output; a `Σ w_e==0` electrode contributes `LOG_EPS` per decode spike, no NaN) + `test_block_parity` (`effective_block` small vs one large block → identical to `rtol=1e-5`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement per §3.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(clusterless_diffusion): non-local predict + memory-budget block size`.

---

## Task 5: local predict path (`is_local=True`)

**Files:** Modify module + tests.
**Interfaces:** same predict function with `is_local=True` → `(n_time, 1)`; per spec §3 Local (nearest-bin; same `heat_kernel_apply`; index `bin(x_a(t_j))`; ground-process from fit-time field at `x_a(t)`).

- [ ] **Step 1:** Write `test_local_equals_nonlocal_per_spike` (compare the internal per-spike `lc` at the animal's bin to the non-local per-spike value on a **one-spike stationary** fixture; subtract the ground-process term) + `test_local_finite_and_zero_rate`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** Commit `feat(clusterless_diffusion): local predict path`.

---

## Task 6: Density-correctness + weighted-EM + degeneracy tests (the numerical contract)

**Files:** `test_clusterless_diffusion.py`. These are the executable contracts that validate Tasks 3–5; per the note in Task 3, write them early.

- [ ] `test_mark_marginal_recovery` — `Σ_x p_e(x,m_j)·ΔV(x) ≈ Σ_i w_i K(m_j,m_i)/Σ_i w_i` (rtol 1e-3).
- [ ] `test_absolute_log_likelihood_small_fixture` — hand-computed `log λ_e` on a tiny 3-bin, 2-spike fixture (exact value, not shape).
- [ ] `test_mass_invariance_across_ranks` — integrated mass `Σ_x heat_kernel_apply(D_e)` equal across `heat_kernel_rank ∈ {full, half}`; do NOT assert pointwise equality.
- [ ] `test_binary_weights_match_hard_subset` — binary EM weights == dropping zero-weight encoding spikes (mirror `test_clusterless_weights`).
- [ ] `test_zero_rate_fit_and_predict_finite_and_warn` — `Σ w_pos==0` and `Σ w_e==0` produce finite output, `LOG_EPS` per decode spike, and the two `UserWarning`s fire.
- [ ] Commit `test(clusterless_diffusion): density-correctness, weighted-EM, degeneracy`.

---

## Task 7: Tier-1 validation

**Files:** module + tests. Cover the **full** Tier-1 contract at **both** fit and predict (bad inputs otherwise propagate NaNs through the prob-space path, and `safe_log` will not repair them).

- [ ] `test_validation_fit` — at fit, each of these raises `ValidationError`:
  - `heat_kernel_rank` float / bool / `0` / negative / `< n_components`;
  - `memory_budget` non-finite / `0` / negative / non-int; `block_size` non-positive;
  - non-positive `position_std` / `waveform_std`;
  - invalid `weights` (wrong length / non-finite / negative — via `common.validate_weights`);
  - non-finite `position` (`common.validate_finite`);
  - non-finite **in-window** encoding `spike_waveform_features` (validate after the `is_in_bounds` clip, as `clusterless_kde`/`clusterless_kde_log` do — a non-finite feature on an out-of-window spike must NOT raise).
- [ ] `test_validation_predict` — at predict, non-finite **in-window** decoding `spike_waveform_features`, non-finite decoding `position`, and non-finite `time` each raise `ValidationError` (again validated post-`is_in_bounds`).
- [ ] Implement the shared `_validate_*` helper called from both fit and predict entry (reuse `common.validate_finite` / `common.validate_weights`; surface `_require_rank_covers_components` for the rank/component constraint).
- [ ] Commit `feat(clusterless_diffusion): tier-1 input validation (fit + predict)`.

---

## Task 8: Registry integration + end-to-end smoke

**MUST precede Task 9** — Task 9's detector-level tests use `clusterless_algorithm="clusterless_diffusion"`, which only resolves once the registry entry exists here.

**Files:** Modify `likelihoods/__init__.py`; test.
- [ ] **Step 1:** `test_registry_and_end_to_end` — `"clusterless_diffusion"` in `_CLUSTERLESS_ALGORITHMS`; `ClusterlessDecoder`/`NonLocalClusterlessDetector` with it does fit+predict and yields a finite, normalized posterior.
- [ ] **Step 2:** Run → FAIL (key missing).
- [ ] **Step 3:** Add imports + registry entry `"clusterless_diffusion": (fit_..., predict_...)` and export the two functions.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(clusterless_diffusion): register algorithm + end-to-end smoke`.

---

## Task 9: Device-cache lifecycle at the algorithm level (refit + save/load parity)

**Depends on Task 8** (constructs a detector with the registered algorithm).

**Files:** `test_clusterless_diffusion_integration.py`.
- [ ] `test_refit_requires_encoding_refit` — build cache via a predict; `env.fit_place_grid(new)`; assert device cache dropped; **re-fit encoding model**; predict works (fresh basis, right shape).
- [ ] `test_save_load_predict_parity` — build a `NonLocalClusterlessDetector(clusterless_algorithm="clusterless_diffusion")`, fit, predict, `save_model`→`load_model` (no `Device` pickling error), predict matches within `rtol=1e-5`.
- [ ] `test_memory_budget_override_survives_handoff` — a non-default `memory_budget` at fit is stored and changes predict's effective block vs default.
- [ ] Commit `test(clusterless_diffusion): device-cache lifecycle + save/load parity`.

---

## Task 10: Agreement, geometry, and grid-independence

**Files:** `test_clusterless_diffusion.py`.
- [ ] `test_agreement_with_kde_simple_geometry` — 1D + 2D open field: decoded-posterior rank-correlation/argmax agreement with `clusterless_kde` (per §Testing; not bit-identical). Use the Task-0 spike's measured median rho as the threshold floor.
- [ ] `test_geometry_no_barrier_leak` — 2D barrier/multi-arm: quantify leaked mass on the wrong side; assert `clusterless_diffusion` leaks materially less than `clusterless_kde`.
- [ ] `test_grid_independence` — posterior stable across two `place_bin_size`s (position_std physical), mirroring `sorted_spikes_diffusion`.
- [ ] Commit `test(clusterless_diffusion): agreement, geometry, grid-independence`.

---

## Task 11: Benchmark (non-gating), CHANGELOG, docstrings, spike cleanup

**Files:** `test_clusterless_diffusion.py` (benchmark, `@pytest.mark.slow`, non-gating), `CHANGELOG.md`, module docstring; delete the Task-0 spike file.
- [ ] Add a non-gating benchmark printing predict time vs `clusterless_kde` at a large grid (no memory assertion — JAX preallocation).
- [ ] Module docstring: describe the algorithm, goals A/B, bandwidth semantics, and that the default remains `clusterless_kde`.
- [ ] `CHANGELOG.md` "Added": new opt-in `clusterless_diffusion` — one paragraph mirroring the `sorted_spikes_diffusion` entry's style (geometry-respecting, cached low-rank operator, prob-space, default unchanged).
- [ ] Delete `test_clusterless_diffusion_spike.py` (its assertions now live in Tasks 6/10).
- [ ] Full check: `uv run ruff check src/ && uv run ruff format --check src/`, then the **shared-surface regression suites** (this feature modified the shared `diffusion.py` engine and `Environment`, so run more than the new tests):
  ```bash
  uv run pytest \
    src/non_local_detector/tests/likelihoods/test_clusterless_diffusion.py \
    src/non_local_detector/tests/likelihoods/test_diffusion_device_cache.py \
    src/non_local_detector/tests/likelihoods/test_clusterless_diffusion_integration.py \
    -q
  # shared engine / environment / serialization that these changes could regress:
  uv run pytest -k "diffusion or sorted_spikes_diffusion or environment or save or load or pickle" -q
  uv run pytest src/non_local_detector/tests/test_golden_regression.py -q      # must be unchanged
  uv run pytest -m snapshot -q                                                  # must be unchanged
  ```
  Golden + snapshot MUST pass with no diffs (default algorithm unchanged). If either moves, stop — the additive feature leaked into an existing path.
- [ ] Commit `feat(clusterless_diffusion): benchmark, CHANGELOG, docs; remove spike`.

---

## Self-Review notes (author)

- **Spec coverage:** §1 core math → Tasks 3/4/6; §2 fit (incl. device basis, safe branches, memory_budget storage) → Tasks 2/3/7/9; §3 predict (non-local, local, memory policy) → Tasks 4/5; §4 bandwidths/normalization/degeneracy/validation → Tasks 1/3/4/7; §Testing → Tasks 6/9/10/11; sequencing (spike first) → Task 0; **registry → Task 8 (before the Task 9 detector-lifecycle tests that need it)**.
- **TDD honesty:** the heavy numerical bodies (Tasks 3–5) are validated by the executable contracts in Task 6 (marginal recovery, absolute LL, mass-invariance, binary-weights) and Task 10 (agreement/geometry) — write those tests before/with the code.
- **Types:** predict/fit signatures and dict keys are declared once in Tasks 3–4 Interfaces and reused verbatim in 5/8/9. `heat_kernel_apply` and `get_device_basis` signatures fixed in Tasks 1–2.
