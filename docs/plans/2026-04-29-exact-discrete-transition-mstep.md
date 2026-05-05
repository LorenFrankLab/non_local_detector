# Plan: Exact Discrete Transition M-Step for Expanded State HMM

## Motivation

The current discrete transition M-step estimates transitions between aggregated
discrete states after the E-step has marginalized over position bins. This is a
reasonable lumped-state approximation, but it is not exact EM for the model that
is actually filtered.

The fitted HMM state is not only the discrete label:

```
S_t in {Local, No-Spike, Non-Local, ...}
```

For spatial states, the latent state is expanded:

```
X_t = (S_t, B_t)
```

where `B_t` is a position bin. The filtering transition matrix is assembled as:

```
P(X_{t+1}=l | X_t=k)
    = P(S_{t+1}=j | S_t=i) * P(B_{t+1}=c | B_t=b, S_t=i, S_{t+1}=j)
    = A[i, j] * C[k, l]
```

with `k = (i, b)` and `l = (j, c)`.

Current code multiplies the continuous transition matrix by the discrete
transition matrix before filtering in `_predict()`:

```python
self.continuous_state_transitions_[cross_is_track_interior]
* discrete_transitions[np.ix_(state_ind, state_ind)]
```

The exact M-step for `A[i, j]` should therefore use expected transition counts
between expanded bins, then aggregate those counts back to discrete states. The
current implementation instead uses only aggregated discrete-state filtered,
predictive, and smoothed probabilities in `estimate_joint_distribution()`.

This can misattribute transition credit when different source states predict
different target-bin distributions within the same target discrete state.

## Current Behavior

The EM loop calls `_estimate_discrete_transition()` with:

- `causal_state_probabilities`
- `predictive_state_probabilities`
- `acausal_state_probabilities`

These arrays have shape `(n_time, n_discrete_states)`.

`estimate_joint_distribution()` then computes:

```
xi_t(i, j)
    = A[i, j]
      * P(S_t=i | y_1:t)
      * P(S_{t+1}=j | y_1:T) / P(S_{t+1}=j | y_1:t)
```

This is exact for a pure discrete HMM where each state has one bin. It is an
approximation for the expanded spatial HMM because the ratio

```
P(S_{t+1}=j | y_1:T) / P(S_{t+1}=j | y_1:t)
```

has already summed over target bins.

The exact expanded-state update needs the bin-specific correction:

```
P(X_{t+1}=l | y_1:T) / P(X_{t+1}=l | y_1:t)
```

before summing over target bins.

## Failure Mode

Suppose two source states both transition into the same spatial target state:

- State A predicts target bins near position 10.
- State B predicts target bins near position 80.
- The next observation strongly supports position 80.

Exact EM gives more transition credit to `B -> target`, because the bin-level
transition path from B better explains the next observation.

The current aggregated update can give too much credit to `A -> target` if A had
high aggregate filtered mass at the previous time, because source-specific
target-bin predictions are marginalized away before the transition count is
computed.

This matters most for transitions involving spatial states whose continuous
dynamics are informative:

- multi-bin Local with `local_position_std is not None`
- Non-Local continuous random-walk states
- track-graph movement models
- any source-state pair with different target-bin predictions

It matters least when:

- every state is singleton
- target emissions are flat within each target discrete state
- continuous transitions into the target state are effectively uniform and
  source-independent

## Goal

Estimate discrete transition parameters from exact expanded-state expected
transition counts:

```
N[i, j] =
    sum_t sum_{k: state(k)=i} sum_{l: state(l)=j}
        P(X_t=k, X_{t+1}=l | y_1:T)
```

Then update:

```
A[i, j] = N[i, j] / sum_j N[i, j]
```

with the existing prior and row-normalization behavior.

## Design

### Core Pair Posterior

For a stationary full transition matrix `T`, compute:

```
xi_t(k, l)
    = filtered_t(k)
      * T[k, l]
      * smoothed_{t+1}(l) / predictive_{t+1}(l)
```

where:

- `filtered_t(k) = P(X_t=k | y_1:t)`
- `predictive_{t+1}(l) = P(X_{t+1}=l | y_1:t)`
- `smoothed_{t+1}(l) = P(X_{t+1}=l | y_1:T)`
- `T[k, l] = P(X_{t+1}=l | X_t=k)`

Use safe division: if `predictive_{t+1}(l) == 0`, the correction for that target
bin is zero.

Aggregate immediately:

```
N[state(k), state(l)] += xi_t(k, l)
```

Do not materialize `(n_time, n_state_bins, n_state_bins)` unless the problem is
small and explicitly requested for debugging.

### Stationary Discrete Transitions

For stationary discrete transitions:

1. Build the full transition matrix already used by filtering:

   ```
   full_transition =
       continuous_state_transitions[interior, interior]
       * discrete_transition[state_ind, state_ind]
   ```

2. Stream over `t = 0 .. n_time - 2`.
3. Compute `xi_t(k, l)` at the expanded-bin level.
4. Aggregate into `joint_sum[i, j]`.
5. Pass `joint_sum` into a stationary transition update that applies the
   existing prior behavior and row normalization.

The stationary estimator should be split into two layers:

- `estimate_stationary_state_transition_from_posteriors(...)`
- `estimate_stationary_state_transition_from_counts(joint_sum, ...)`

This keeps the prior logic in one place and makes the exact-count path testable.

### Nonstationary Discrete Transitions

For covariate-dependent transitions, the response matrix for time `t` should be:

```
response_t[i, j] =
    sum_{k: state(k)=i} sum_{l: state(l)=j} xi_t(k, l)
```

Then for each source state `i`, fit the multinomial/logistic row model using:

```
design_matrix[:-1]
response[:, i, :]
```

This matches the current API shape, but replaces the approximate aggregated
`joint_distribution[:, i, :]` with exact expanded-bin counts.

## Implementation

### Phase 1: Add Exact Count Helper

Add a helper in `src/non_local_detector/discrete_state_transitions.py`:

```python
def estimate_discrete_transition_counts_from_expanded_posteriors(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Return exact expected discrete transition counts.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
    predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
    transition_matrix : np.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : np.ndarray, shape (n_state_bins,)

    Returns
    -------
    joint_sum : np.ndarray, shape (n_discrete_states, n_discrete_states)
    """
```

Implementation sketch:

```python
n_states = int(state_ind.max()) + 1
joint_sum = np.zeros((n_states, n_states))

for t in range(causal_posterior.shape[0] - 1):
    ratio = np.divide(
        acausal_posterior[t + 1],
        predictive_posterior[t + 1],
        out=np.zeros_like(acausal_posterior[t + 1]),
        where=predictive_posterior[t + 1] > 0,
    )
    xi = causal_posterior[t, :, None] * transition_matrix * ratio[None, :]
    np.add.at(joint_sum, (state_ind[:, None], state_ind[None, :]), xi)

return joint_sum
```

The `np.add.at` sketch is simple but may be slow. A faster implementation can
use a precomputed one-hot aggregation matrix:

```python
G = one_hot(state_ind)  # shape (n_state_bins, n_discrete_states)
joint_sum += G.T @ xi @ G
```

or block sums by state index masks. Start with the clearest implementation and
benchmark if needed.

The implemented helpers use a parameterized JAX `lax.scan` kernel with
`segment_sum` to stream over time without materializing all pair posteriors.
Public helper outputs are converted back to NumPy arrays so the surrounding
SciPy-based M-step APIs remain unchanged. The factorized JAX path also avoids
materializing
`continuous_transition * discrete_transition[state_ind, state_ind]` for both
stationary counts and stationary or time-varying response series; it first
aggregates the continuous transition into target discrete states, then applies
source-state aggregation and the small discrete transition matrix. This still
assumes the current dense continuous transition representation, so a future
sparse/banded transition representation would be needed to reduce the memory
model from `O(n_state_bins^2)` to `O(nnz(C))`.

With the default JAX configuration in the development environment
(`jax_enable_x64=False`), parity checks use float32-level tolerance
(`atol=1e-5`). Strict NumPy parity was also checked with `JAX_ENABLE_X64=1`,
where the observed max absolute differences were:

- expanded response helper: `5.551e-17`
- stationary factorized counts helper: `8.882e-16`
- expanded counts helper: `8.882e-16`

The stationary factorized count helper is intentionally stationary-only; it
rejects time-varying discrete transition arrays instead of silently indexing
them as if they were stationary.

### Phase 2: Wire Stationary EM Path

Update `_DetectorBase.estimate_parameters()` so the discrete transition M-step
uses expanded posteriors when available:

- `causal_posterior`
- `predictive_posterior`
- `acausal_posterior`
- `self.state_ind_[self.is_track_interior_state_bins_]`
- the full transition matrix used in `_predict()`

For the stationary case, call:

```python
joint_sum = estimate_discrete_transition_counts_from_expanded_posteriors(...)
discrete_transition = estimate_stationary_state_transition_from_counts(
    joint_sum,
    concentration=...,
    stickiness=...,
    prior_weight=...,
)
```

Keep the current aggregate-probability path behind an internal fallback only if
needed for compatibility.

### Phase 3: Wire Nonstationary EM Path

Add a nonstationary helper:

```python
def estimate_discrete_transition_responses_from_expanded_posteriors(
    ...
) -> np.ndarray:
    """Return response, shape (n_time - 1, n_states, n_states)."""
```

For each `t`, use the time-specific full transition matrix:

```
full_transition_t =
    continuous_transition
    * discrete_transition_t[state_ind, state_ind]
```

Aggregate `xi_t` into `response[t]`.

Then update `estimate_non_stationary_state_transition()` so it can accept either:

- posterior inputs and compute approximate responses, preserving current behavior
- precomputed exact responses, used by the new EM path

Prefer a new explicit function if that keeps the API clearer:

```python
estimate_non_stationary_state_transition_from_responses(
    transition_coefficients,
    design_matrix,
    response,
    ...
)
```

### Phase 4: Prior Cleanup

While changing the transition M-step, fix or explicitly defer two prior issues:

1. `discrete_transition_concentration < 1` can create negative stationary
   pseudo-counts because the current code adds `alpha - 1` to expected counts.
   The implemented MAP update rejects `concentration < 1` during detector
   validation and in the low-level transition estimators. A future sparse-prior
   implementation should use a proper constrained MAP treatment instead of
   adding negative pseudo-counts.

2. The nonstationary prior currently adds `alpha - 1` to every time sample in
   the objective. That does not match the stationary path's fixed-count prior.
   Decide whether the nonstationary prior should be:
   - per-time regularization, documented as such
   - fixed total pseudo-counts distributed across time
   - replaced by the existing L2 penalty plus optional row stickiness

Do not mix exact-count changes with silent prior semantic changes. If prior
cleanup is too large, make it a separate PR and add tests documenting current
behavior.

## Verification Strategy

### Unit Tests

1. **Pure discrete HMM parity.**
   With one bin per state and identity continuous transitions, exact expanded
   counts must match the current aggregated `estimate_joint_distribution()`.

2. **Uniform continuous transitions parity.**
   When all source states induce the same target-bin distribution within each
   target state, exact and aggregated counts should match within tolerance.

3. **Source-specific spatial prediction test.**
   Construct two source states and one spatial target state:
   - source A predicts target bin 0
   - source B predicts target bin 1
   - next smoothed posterior is concentrated on target bin 1

   Exact counts should credit `B -> target` more than `A -> target`. The current
   aggregated method should fail or show weaker discrimination. This is the
   regression test that justifies the change.

4. **Zero predictive bins.**
   If `predictive_posterior[t + 1, l] == 0`, the helper should return finite
   counts and no NaNs.

5. **Stationary prior application from counts.**
   `estimate_stationary_state_transition_from_counts()` should match the legacy
   stationary estimator when passed the legacy `joint_sum`.

6. **Concentration guard.**
   Add tests asserting a clear validation error for `concentration < 1` in both
   detector validation and low-level transition estimation.

7. **Nonstationary recovery evidence.**
   Simulate covariate-dependent transition responses from known nonstationary
   coefficients, fit `estimate_non_stationary_state_transition_from_responses()`,
   and verify the recovered transition curve matches the true curve within
   sampling tolerance.

### Integration Tests

1. Run a sorted-spikes detector with `local_position_std=None`; verify results
   are unchanged or within floating-point tolerance for a case with singleton
   Local and simple transitions.

2. Run sorted spikes with `local_position_std > 0`; verify:
   - transition rows remain stochastic
   - log likelihood does not decrease after a discrete-transition-only M-step,
     or document if the prior/regularized objective makes this a generalized EM
     check instead of exact monotonicity

3. Run a non-local continuous state on a small track graph where movement
   dynamics are source-specific; verify the exact update shifts transition mass
   toward the source state whose predicted bins match the next observations.

### Performance Checks

Measure runtime and memory for:

- current aggregated path
- exact streaming path
- exact materialized debug path, if implemented

Use at least:

- small synthetic sorted-spikes case for correctness
- moderate synthetic case with realistic `n_time` and `n_state_bins`

Do not claim performance acceptability without recording these numbers.

Current smoke evidence on the simulated clusterless run
(`make_simulated_run_data(n_tetrodes=2, place_field_means=np.arange(0, 80, 20),
n_runs=3, seed=42)`, `max_iter=3`, `estimate_encoding_model=False`):

- branch smoke: `elapsed_s=3.510`, `n_ll=2`, `final_ll=-30780.550781`,
  `maxrss=958545920`
- row stochasticity smoke: one-state transition sum `1.0`

Earlier same-setup comparison against `main` showed identical final likelihood
and a first-call compile/runtime cost for the JAX exact-count branch on this
small case. A warmed kernel-only microbenchmark favored the JAX aggregation
kernel, so PR notes should distinguish first-call compile overhead from warmed
kernel throughput.

## Rollout

Recommended rollout order:

1. Add exact-count helper and unit tests.
2. Split stationary update into `from_counts` and posterior-wrapper functions.
3. Wire stationary transition M-step to exact counts.
4. Add integration tests for sorted spikes.
5. Add nonstationary response helper and tests.
6. Wire nonstationary path.
7. Address or explicitly defer prior cleanup.

If risk needs to be minimized, add an internal flag first:

```python
discrete_transition_mstep: Literal["expanded", "aggregated"] = "expanded"
```

But the preferred end state is for exact expanded-state counts to be the
default, because that matches the HMM actually used by filtering.

## Out of Scope

- Changing the Local encoding-model M-step.
- Changing `local_position_std` semantics.
- Refitting continuous transition matrices.
- Optimizing the likelihood computation.
- Renaming states or changing default state definitions.

## Open Questions

1. Should the approximate aggregated path remain public, private, or be removed
   after exact-count parity tests pass?

2. Should exact transition counts be computed during the smoother pass to avoid
   storing full `causal_posterior`, `predictive_posterior`, and
   `acausal_posterior`?

3. Should row freezing apply before or after exact-count prior normalization?
   Current behavior restores frozen stationary rows after the M-step; preserving
   that behavior is simplest.

4. Should sparse Dirichlet priors with `concentration < 1` be supported later?
   The current implementation intentionally rejects them because the present
   pseudo-count update is only valid for nonnegative `alpha - 1`.
