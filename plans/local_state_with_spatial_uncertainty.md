# Plan: Local State with Spatial Uncertainty

## Motivation

The current Local state uses a **single position bin** — a delta function at the animal's
observed position. The likelihood is a scalar per time step, and the `Discrete()` continuous
transition is a 1×1 identity matrix. This means the Local state carries no spatial uncertainty.

In reality, CA1 place cells coding for the animal's current position have inherent uncertainty
from the population code, position measurement noise, and the animal's spatial extent. The
Local state should express "the animal is **near** position X" rather than "the animal **is at**
position X."

## Design

### Core Concept

Make the Local state structurally similar to the non-local states: it uses **all position bins**
with the same place fields. The difference is a **position uncertainty kernel** — a normalized
Gaussian centered on the animal's observed position — added to the log-likelihood at each time
step. This kernel anchors the Local state to the animal while allowing spatial uncertainty.

The local log-likelihood at time *t* becomes:

```
log P(spikes_t | all bins) + log P(bin | animal_pos_t, σ)
```

where the first term is the standard full-spatial likelihood (same encoding model as non-local)
and the second term is the position kernel.

### Key Design Decisions

1. **Likelihood anchoring, not transition anchoring.** The position kernel lives in the
   likelihood, not in the transition matrix. This avoids time-varying transitions and keeps
   the existing static transition machinery.

2. **Nearly memoryless Local → Local content transition.** Use `Uniform()` for the
   Local → Local continuous transition. The position kernel does all the spatial anchoring.
   This is correct because "local" means "tied to the animal now," not "tied to the previous
   latent estimate." A random walk would let the local state drift away from the animal.

3. **Sticky at the event level, memoryless at the content level.** The discrete transition
   `P(event_t = Local | event_{t-1} = Local)` remains high (unchanged). But
   `P(content_t | content_{t-1}, event_t = Local)` is nearly memoryless — the current
   animal position determines the content, not the previous decoded bin.

4. **Model-level parameter.** `local_position_std` is a single parameter on the detector
   class, not per-ObservationModel. There's no use case for multiple local states with
   different kernel widths.

5. **Track graph distances.** The position kernel should use track graph distances (not
   Euclidean) when a track graph is available, to handle junctions correctly.

6. **Backward compatibility.** `local_position_std=None` (default) preserves legacy 1-bin
   behavior exactly. The new behavior is opt-in.

---

## Implementation

### Phase 1: ObservationModel and State Index

#### 1a. Add `local_position_std` to detector classes

**File:** `src/non_local_detector/models/non_local_model.py`

Add `local_position_std: float | None = None` as a constructor parameter to
`NonLocalClusterlessDetector` and `NonLocalSortedSpikesDetector`. Store it as `self.local_position_std`.

This is a model-level parameter, not on `ObservationModel`. The observation model's `is_local`
flag still controls which state is the local state; `local_position_std` controls how it's
represented spatially.

#### 1b. Modify `initialize_state_index()` to allocate full bins for multi-bin local

**File:** `src/non_local_detector/models/base.py` (line 793)

Current:
```python
if obs.is_local or obs.is_no_spike:
    bin_sizes.append(1)
```

Change to check whether multi-bin local is active:
```python
if obs.is_no_spike or (obs.is_local and self.local_position_std is None):
    bin_sizes.append(1)
```

When `local_position_std` is set, the local state falls through to the `else` branch and gets
all `environment.place_bin_centers_.shape[0]` bins, same as non-local states.

The `local_position_std` attribute needs to exist on the base class (or be checked via
`getattr`). Use `getattr(self, 'local_position_std', None)` for safety, since not all
subclasses (decoders, classifiers) will have this parameter.

#### 1c. Modify initial conditions for multi-bin local

**File:** `src/non_local_detector/initial_conditions.py` (line 58)

Current:
```python
if observation_model.is_local or observation_model.is_no_spike:
    initial_conditions = np.ones((1,), dtype=np.float32)
```

Problem: `UniformInitialConditions.make_initial_conditions()` doesn't have access to
`local_position_std` since it only receives the `ObservationModel`.

Solution: Add an `is_multi_bin_local` flag to the call site. In `initialize_initial_conditions()`
(base.py line 816), when `local_position_std is not None` and `obs.is_local`, temporarily
override the observation model's behavior by creating a non-local-like observation model for
initial conditions, OR pass the flag through.

Simplest approach: modify `make_initial_conditions` to accept an optional `force_spatial=False`
parameter. When `True`, skip the `is_local` check and return uniform over all position bins.
The call site in `base.py` passes `force_spatial=True` when `local_position_std is not None`
and `obs.is_local`.

### Phase 2: Continuous Transitions

#### 2a. Handle multi-bin → 1-bin transitions in assembly code

**File:** `src/non_local_detector/models/base.py` (line ~920)

The existing code handles 1-bin → n-bin transitions (Discrete to spatial: uniform).
Add the symmetric case for n-bin → 1-bin transitions (spatial to Discrete):

```python
elif n_row_bins > 1 and n_col_bins == 1:
    # Spatial to non-spatial: each source bin transitions to the
    # single target bin with probability 1.
    self.continuous_state_transitions_[inds] = np.ones((n_row_bins, 1))
```

This handles Local (multi-bin) → No-Spike (1-bin) transitions.

#### 2b. Update default continuous transitions for multi-bin local

**File:** `src/non_local_detector/models/_defaults.py`

The defaults don't need to change — they still use `Discrete()` for Local transitions.
When multi-bin local is active, the assembly code in `initialize_continuous_state_transitions`
needs to auto-upgrade transitions involving the local state:

- **Local → Local:** If `local_position_std is not None` and both states are multi-bin and
  transition is `Discrete()`, replace with `Uniform()`. This makes local content nearly
  memoryless — the position kernel handles spatial anchoring.

- **Local → Non-Local:** `Uniform()` (already the default, and now sizes match n→n).

- **Non-Local → Local:** `Discrete()` transitions from 1-bin to n-bin are already handled as
  uniform by the assembly code. With multi-bin local, this becomes n→n and `Uniform()` is
  already the default.

Implementation: In `initialize_continuous_state_transitions()`, add logic after the transition
type is determined:

```python
# Auto-upgrade Discrete() for multi-bin local states
local_position_std = getattr(self, 'local_position_std', None)
if (isinstance(transition, Discrete) and local_position_std is not None
    and n_row_bins > 1 and n_col_bins > 1):
    # Multi-bin local: use Uniform instead of Discrete for content transitions
    transition = Uniform()
```

### Phase 3: Position Uncertainty Kernel

#### 3a. Add `compute_local_position_kernel()` utility

**File:** `src/non_local_detector/likelihoods/common.py`

New function that computes the log of a normalized Gaussian kernel centered on the animal's
observed position, using track graph distances when available:

```python
def compute_local_position_kernel(
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    environment: Environment,
    local_position_std: float,
) -> jnp.ndarray:
    """Compute log position uncertainty kernel for local state.

    Returns a normalized (in log-space) Gaussian kernel centered on the
    animal's interpolated position at each time step.

    Parameters
    ----------
    time : jnp.ndarray, shape (n_time,)
        Decoding time bins.
    position_time : jnp.ndarray, shape (n_time_position,)
        Position sampling times.
    position : jnp.ndarray, shape (n_time_position, n_position_dims)
        Position samples.
    environment : Environment
        The spatial environment.
    local_position_std : float
        Standard deviation of the position uncertainty kernel.

    Returns
    -------
    log_kernel : jnp.ndarray, shape (n_time, n_interior_bins)
        Log of the normalized position uncertainty kernel.
    """
    animal_pos = get_position_at_time(position_time, position, time, environment)
    is_interior = environment.is_track_interior_.ravel()
    bin_centers = environment.place_bin_centers_[is_interior]

    # Compute distances — use track graph distances if available
    if environment.track_graph is not None:
        # Use graph distances for each (animal_pos, bin_center) pair
        # Need to compute pairwise distances for each time step
        # animal_pos: (n_time, n_dims), bin_centers: (n_interior_bins, n_dims)
        # This requires calling environment methods that use the graph
        # Implementation: vectorize over time steps
        n_time = time.shape[0]
        n_bins = bin_centers.shape[0]
        sq_dist = np.zeros((n_time, n_bins))
        for t in range(n_time):
            pos_t = animal_pos[t:t+1]  # (1, n_dims)
            pos_repeated = np.broadcast_to(pos_t, (n_bins, pos_t.shape[-1]))
            distances = environment.pairwise_distance(pos_repeated, bin_centers)
            sq_dist[t] = distances ** 2
        sq_dist = jnp.array(sq_dist)
    else:
        # Euclidean distances
        if animal_pos.ndim == 1:
            animal_pos = animal_pos[:, jnp.newaxis]
        if bin_centers.ndim == 1:
            bin_centers = bin_centers[:, jnp.newaxis]
        sq_dist = jnp.sum(
            (animal_pos[:, jnp.newaxis, :] - bin_centers[jnp.newaxis, :, :]) ** 2,
            axis=-1,
        )

    log_kernel = -0.5 * sq_dist / (local_position_std ** 2)
    # Normalize per time step so kernel is a proper log-probability
    log_kernel -= logsumexp(log_kernel, axis=1, keepdims=True)

    return log_kernel
```

Note: The track graph distance path loops over time steps, which may be slow for long
recordings. This can be optimized later (vectorize via bin index lookup, or precompute
distance matrix and index into it). For v1, correctness over speed.

#### 3b. Alternative: precompute distance matrix for track graph case

Instead of looping over time steps, precompute a distance matrix between all bin centers
and all bin centers (already available as `environment.distance_between_nodes_`), then for
each time step find the nearest bin to the animal position and look up distances from that
bin to all other bins. This would be much faster.

```python
# Faster track graph approach:
animal_bin_inds = environment.get_bin_ind(animal_pos)  # (n_time,)
# distance_between_nodes_ is (n_bins, n_bins) or dict
# Index: distances[animal_bin_ind, :] gives distances to all bins
```

This avoids the per-time-step loop entirely. Use this approach for v1.

### Phase 4: Likelihood Backends

#### 4a. Modify `compute_log_likelihood` in base classes

**Files:**
- `src/non_local_detector/models/base.py` lines 2536-2671 (clusterless)
- `src/non_local_detector/models/base.py` lines 3409-3540 (sorted spikes)

For multi-bin local states, the likelihood should be computed as non-local (full spatial)
and then the position kernel added. The key changes:

1. **Likelihood computation:** When `local_position_std is not None` and `obs.is_local`,
   call the likelihood function with `is_local=False` to get the full spatial likelihood,
   then add the position kernel.

2. **Caching key:** The `likelihood_name` tuple must distinguish multi-bin local from
   non-local. Two states sharing the same environment/encoding_group but one being
   multi-bin local and the other non-local compute the **same base likelihood** (both use
   `is_local=False`). The difference is only the position kernel added afterward.

   This means we can **share** the base likelihood computation between multi-bin local
   and non-local states. Compute it once with `is_local=False`, cache it, and add the
   position kernel only for the local state.

3. **Assembly:** After computing likelihoods, add the position kernel to the multi-bin
   local state's bins:

```python
# After assembling log_likelihood array
local_position_std = getattr(self, 'local_position_std', None)
if local_position_std is not None:
    env_kernels = {}
    for state_id, obs in enumerate(self.observation_models):
        if obs.is_local:
            env_name = obs.environment_name
            if env_name not in env_kernels:
                env = self._get_environment_by_name(env_name)
                env_kernels[env_name] = compute_local_position_kernel(
                    time, position_time, position, env, local_position_std
                )
            is_state_bin = state_bin_masks[state_id]
            log_likelihood = log_likelihood.at[:, is_state_bin].add(
                env_kernels[env_name]
            )
```

This mirrors the pattern used by `non_local_position_penalty` (lines 2651-2668).

4. **Likelihood name for caching:** When `local_position_std is not None`, the local state
   should use `is_local=False` in its `likelihood_name` so it shares the base likelihood
   with non-local states:

```python
likelihood_name = (
    obs.environment_name,
    obs.encoding_group,
    obs.is_local and getattr(self, 'local_position_std', None) is None,  # True only for legacy local
    obs.is_no_spike,
)
```

And pass `is_local=False` when multi-bin local:
```python
effective_is_local = obs.is_local and getattr(self, 'local_position_std', None) is None
likelihood_results[state_id] = likelihood_func(
    ...,
    is_local=effective_is_local,
)
```

#### 4b. No changes to individual likelihood backends

Since multi-bin local uses `is_local=False` to compute the base likelihood and adds the
position kernel afterward, the individual likelihood functions (`sorted_spikes_kde`,
`clusterless_kde`, `sorted_spikes_glm`, `clusterless_gmm`, `clusterless_kde_log`) do not
need any modifications. The kernel is added at the assembly level in `compute_log_likelihood`.

### Phase 5: Results Assembly

#### 5a. Fix position coordinate assignment for multi-bin local

**File:** `src/non_local_detector/models/base.py` (lines 2036-2049)

Current code assigns NaN positions for local states:
```python
if obs.is_local or obs.is_no_spike:
    position.append(np.full((1, n_position_dims), np.nan))
```

Change to:
```python
if obs.is_no_spike or (obs.is_local and getattr(self, 'local_position_std', None) is None):
    position.append(np.full((1, n_position_dims), np.nan))
```

When multi-bin local is active, the local state falls through and gets real position
coordinates from `environment.place_bin_centers_`, same as non-local states.

#### 5b. Fix Viterbi/MAP sequence conversion

**File:** `src/non_local_detector/models/base.py` (lines 2174-2178)

Same pattern — change the condition so multi-bin local gets real positions:
```python
if obs.is_no_spike or (obs.is_local and getattr(self, 'local_position_std', None) is None):
    position.append(np.full((1, n_position_dims), np.nan))
```

### Phase 6: Testing

#### 6a. Unit tests for position kernel

- Kernel is a valid log-probability distribution (sums to 1 in probability space per time step)
- Kernel is centered on the animal's position (peak at nearest bin)
- Kernel narrows as `local_position_std` decreases
- Track graph distances are used when track graph is available
- Handles edge cases: animal at track boundary, NaN positions

#### 6b. Integration tests

- Multi-bin local model produces valid posteriors (probabilities sum to 1)
- Local state posterior is concentrated near the animal's position
- Legacy behavior (`local_position_std=None`) produces identical results to current code
- Multi-bin local + non-local states produce valid combined posteriors
- Transition matrices are properly constructed (stochastic, correct shape)

#### 6c. Backward compatibility tests

- All existing tests pass unchanged when `local_position_std=None`
- Snapshot tests produce identical results (no regression)
- Golden regression tests pass

---

## Files Changed (Summary)

| File | Change | Phase |
|------|--------|-------|
| `models/non_local_model.py` | Add `local_position_std` constructor param | 1a |
| `models/base.py` `initialize_state_index` | Multi-bin local gets full position bins | 1b |
| `initial_conditions.py` | `force_spatial` param for multi-bin local | 1c |
| `models/base.py` `initialize_continuous_state_transitions` | n→1 case; auto-upgrade Discrete for multi-bin | 2a, 2b |
| `likelihoods/common.py` | New `compute_local_position_kernel()` | 3a |
| `models/base.py` `compute_log_likelihood` (×2) | Effective is_local, add kernel post-assembly | 4a |
| `models/base.py` results assembly (×2) | Multi-bin local gets real positions | 5a, 5b |

**Files NOT changed:**
- `observation_models.py` — `is_local` semantics unchanged
- `models/_defaults.py` — defaults unchanged (backward compatible)
- All likelihood backends — no changes (kernel added at assembly level)
- `continuous_state_transitions.py` — no new classes needed

## Open Questions for Later

1. **Non-local position penalty interaction:** With multi-bin local, the local state may absorb
   some of what the penalty was doing. Test empirically whether the penalty is still needed.

2. **Performance of track graph kernel:** The precomputed distance matrix approach should be
   fast, but profile on real data to confirm.

3. **Optimal `local_position_std`:** What value makes scientific sense? Likely depends on
   position tracking precision and place field width. Could be estimated from data.

4. **EM parameter re-estimation:** The multi-bin local posterior is now spatial. Verify that
   the existing EM code correctly re-estimates encoding models when the local state contributes
   spatial posterior mass.
