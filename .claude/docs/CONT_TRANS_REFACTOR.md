Here’s a tight, “Claude-code–ready” refactor plan that you can hand to an LLM to implement in small, verifiable steps. It’s organized into milestones with explicit tasks, file targets, signatures, acceptance checks, and no ambiguous language.

---

# Milestone 1 — Introduce Transition **Operators** (no behavior change)

**Goal:** Replace dense `N×N` continuous transitions with lightweight operator objects while keeping the old API working.

**Files:** `base.py`, `continuous_state_transitions.py`

**Tasks**

1. **Create protocol + dataclasses**

   * Add `transitions/operators.py`:

     * `class TransitionOp(Protocol):  # apply(p: Array) -> Array`
     * `@dataclass class CSRRandomWalkOp(TransitionOp): indptr: Array; indices: Array; weights: Array`
     * `@dataclass class UniformOp(TransitionOp): q: Array   # row-stochastic vector`
     * `@dataclass class IdentityOp(TransitionOp): pass`
     * (optional) `@dataclass class Conv1DOp(TransitionOp): kernel: Array; padding: tuple[int,int]`
2. **Factory functions** (pure, NumPy build → JAX storage)

   * `make_csr_random_walk(centers, dist_fn, mean, var, radius) -> CSRRandomWalkOp`
   * `make_uniform(mask) -> UniformOp`
   * `make_identity(n) -> IdentityOp`
3. **Operator application (JAX)**

   * `apply(op: TransitionOp, p: Array) -> Array` using pattern-matching:

     * CSR: `segment_sum(weights * p[indices], row_ids)`
     * Uniform: return `op.q` (no matmul)
     * Identity: return `p`
     * Conv1D: `lax.conv_general_dilated(...)`
4. **Wire into `base.py`**

   * Replace dense continuous transition matrices storage with a registry:

     * `self.continuous_ops_: dict[tuple[int,int], TransitionOp]`
   * Keep the **existing public getters** returning dense matrices for now by **materializing on demand** (compat layer).

**Acceptance**

* All tests pass with zero numerical deltas (±1e-6).
* Memory profile shows no `N×N` allocation during filtering for random-walk/uniform blocks.

---

# Milestone 2 — Core Predict Step uses Operators (chunk-safe, donated)

**Goal:** Replace `filtered @ A` with operator application inside `core.py`.

**Files:** `core.py`

**Tasks**

1. **Predict helper**

   * `def predict_step(p: Array, ops: Sequence[TransitionOp]) -> Array:`

     * Accumulate: `p_next = sum(apply(op, p) for op in ops_in_env)`
     * Normalize once (safe denom).
     * Wrap heavy math in `with default_matmul_precision('high')`.
     * Add `@jax.jit(donate_argnums=(0,))`.
2. **Filter/Smoother kernels**

   * Route continuous part via `predict_step` instead of dense multiply.
   * Keep discrete mixing as now (state×state small blocks) by **scaling op weights** when covariate-dependent:

     * Add `scale(op, row_scale: Array, col_scale: Array) -> TransitionOp` for CSR (multiply `weights` in-place form or return a view).
3. **Chunk drivers**

   * Ensure per-chunk covariate discrete factors are applied **without** forming `T×N×N`.
   * Donate: prior and per-chunk discrete factors.

**Acceptance**

* Same posterior vs baseline (≤1e-6).
* Peak memory drops when `N` large and ops are CSR/Uniform.
* No dense `@` on `N×N` appears in HLO (compile dump spot-check).

---

# Milestone 3 — Viterbi: Per-Chunk + Operator Path

**Goal:** Eliminate full `T×N×N` pre-indexing; support operators.

**Files:** `core.py`

**Tasks**

1. **Chunked Viterbi wrapper**

   * Mirror filter driver: slice `logL` and **per-chunk** discrete factors.
2. **Backward score step (log-space)**

   * Build `logA_t` **on the fly**:

     * For CSR op: compute `scores_i = logsumexp_over_j( log(w_ij) + best_next[j] + logL[t+1,j] )`
     * For Uniform: `scores = log(q) + best_next + logL[t+1]` (rank-1 simplification).
   * Avoid materializing dense; iterate over edges (CSR) or use small conv window.
3. **Path reconstruction**

   * Keep current argmax pass, storing backpointers per chunk.

**Acceptance**

* Paths identical to dense baseline on small tests.
* Memory flat with `T, N` scaling (no `T×N×N`).

---

# Milestone 4 — API Cleanups & Dtype/Precision Controls

**Goal:** Make knobs explicit and harmonize returns.

**Files:** `core.py`, `base.py`

**Tasks**

1. **Config object**

   * `@dataclass class FilterConfig: dtype: str='float32'; enable_x64: bool=False; precision: Literal['default','high']='high'`
2. **Thread config**

   * Cast inputs once per entrypoint.
   * Wrap hot paths under precision context; keep `donate_argnums` on kernels.
3. **Return types**

   * Ensure *all* public functions return `np.ndarray` (use `np.asarray` at boundaries).
4. **Finite checks (debug only)**

   * Host-side `_assert_finite(name, x)` behind `if debug:` to avoid JAX version pitfalls.

**Acceptance**

* Consistent dtype across modules.
* Users can flip `precision='high'` without code edits.
* Return types consistent.

---

# Milestone 5 — Performance Pass (optional but recommended)

**Goal:** Remove residual hotspots, add specialized ops, and document patterns.

**Files:** `transitions/operators.py`, `continuous_state_transitions.py`, `core.py`, `docs/`

**Tasks**

1. **Specialized ops**

   * `BoxcarOp` (uniform window) with O(N) prefix-sum implementation.
   * `Separable2DOp` (if you have grid states): two 1D convs instead of 2D.
2. **Boundary policies**

   * Parameterize CSR/Conv ops with `padding=('reflect'|'wrap'|'zero')` and ensure row-stochasticity.
3. **Profiling hooks**

   * Add `profile=True` option that runs a short trace via `jax.profiler.trace` and prints peak memory & key kernels.
4. **Docs**

   * `docs/TRANSITIONS.md`: operator catalog, complexity table, when to choose CSR vs Conv vs Uniform.

**Acceptance**

* Benchmarks: show `O(kN)` scaling for random-walk, O(N) for uniform.
* Clear docs with examples.

---

## Implementation Sketches (Claude-friendly)

**Operator protocol & apply**

```python
# transitions/operators.py
from dataclasses import dataclass
from typing import Protocol, Literal
import jax, jax.numpy as jnp
from jax import lax, default_matmul_precision as dmp

class TransitionOp(Protocol):
    def apply(self, p: jnp.ndarray) -> jnp.ndarray: ...

@dataclass
class CSRRandomWalkOp:
    indptr: jnp.ndarray  # [N+1]
    indices: jnp.ndarray # [nnz]
    weights: jnp.ndarray # [nnz]
    def apply(self, p: jnp.ndarray) -> jnp.ndarray:
        rows = jnp.repeat(jnp.arange(self.indptr.size - 1), jnp.diff(self.indptr))
        contrib = self.weights * p[self.indices]
        with dmp('high'):
            return jax.ops.segment_sum(contrib, rows, num_segments=self.indptr.size - 1)

@dataclass
class UniformOp:
    q: jnp.ndarray
    def apply(self, p: jnp.ndarray) -> jnp.ndarray:
        return self.q  # rank-1: p @ (1·qᵀ) == q

@dataclass
class IdentityOp:
    n: int
    def apply(self, p: jnp.ndarray) -> jnp.ndarray:
        return p
```

**Predict step**

```python
def predict_step(p: jnp.ndarray, ops: tuple[TransitionOp, ...]) -> jnp.ndarray:
    with dmp('high'):
        acc = sum(op.apply(p) for op in ops)
    # safe normalize
    s = jnp.sum(acc)
    acc = jnp.where(s > 0, acc / s, acc)
    return acc
```

**Discrete × continuous mixing (per-time)**

```python
def scale_csr(op: CSRRandomWalkOp, row_scale: jnp.ndarray, col_scale: jnp.ndarray) -> CSRRandomWalkOp:
    # scale each edge i<-j by row_scale[i] * col_scale[j]
    rows = jnp.repeat(jnp.arange(op.indptr.size - 1), jnp.diff(op.indptr))
    w = op.weights * row_scale[rows] * col_scale[op.indices]
    return CSRRandomWalkOp(op.indptr, op.indices, w)
```

---

## Testing Checklist (copy/paste into tasks)

* [ ] Unit: `apply(CSRRandomWalkOp)` equals dense multiply within 1e-6 on random small graphs.
* [ ] Unit: `apply(UniformOp)` equals masked uniform vector.
* [ ] Property: row-stochasticity preserved after mixing with discrete factors.
* [ ] Integration: filter/smoother posteriors match dense baseline on toy problems.
* [ ] Memory: no `N×N` HLO allocations for random-walk/uniform.
* [ ] Perf: time scales ~O(kN) vs O(N²) for increasing N with fixed bandwidth k.

---

## Folder/Module Layout

```
transitions/
  operators.py        # TransitionOp, CSRRandomWalkOp, UniformOp, IdentityOp, helpers
  __init__.py

continuous_state_transitions.py  # now emits ops via factory funcs
discrete_state_transitions.py    # unchanged API, provides small state×state factors
base.py                          # stores continuous_ops_ registry
core.py                          # uses predict_step(apply ops), chunked viterbi
docs/TRANSITIONS.md              # rationale, complexity, examples
```

---

This plan keeps changes incremental, testable after each milestone, and minimizes risk: you introduce operators behind a compatibility layer, route the core compute through them, then optimize Viterbi and polish the API.
