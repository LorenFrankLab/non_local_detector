# Plan: Local State with Spatial Uncertainty

## Motivation

The current Local state uses a **single position bin** — a delta function at the animal's
observed position. The likelihood is a scalar per time step, and the `Discrete()` continuous
transition is a 1x1 identity matrix. This means the Local state carries no spatial uncertainty.

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
log P(spikes_t | all bins) + log P(bin | animal_pos_t, sigma)
```

where the first term is the standard full-spatial likelihood (same encoding model as non-local)
and the second term is the position kernel.

### Architectural Rationale: Kernel in Likelihood vs Transition

The position kernel lives in the **likelihood**, not in the transition matrix. This is a
deliberate choice:

- Time-varying transition matrices would break `jax.lax.scan` and the static transition
  assumption used throughout `core.py`.
- The existing `compute_log_likelihood` already has a post-assembly injection pattern
  (see `_compute_non_local_position_penalty`, base.py lines 2651-2668) that we can mirror.
- The kernel is conceptually part of the observation model ("how likely is this neural
  activity given the animal is near position X?"), not the dynamics.

### Key Design Decisions

1. **Likelihood anchoring, not transition anchoring.** See architectural rationale above.

2. **Nearly memoryless Local -> Local content transition.** Use `Uniform()` for the
   Local -> Local continuous transition. The position kernel does all the spatial anchoring.
   This is correct because "local" means "tied to the animal now," not "tied to the previous
   latent estimate." A random walk would let the local state drift away from the animal
   during prolonged local periods — exactly the wrong failure mode. The important distinction:
   `P(event_t = Local | event_{t-1} = Local)` should be high (sticky at event level), but
   `P(content_t | content_{t-1}, event_t = Local)` does not need spatial memory.

3. **Model-level parameter.** `local_position_std` is a single parameter on the detector
   class, not per-ObservationModel. There's no use case for multiple local states with
   different kernel widths.

4. **Track graph distances.** The position kernel uses track graph distances (not Euclidean)
   when a track graph is available, to handle junctions correctly. Uses precomputed
   `distance_between_nodes_` matrix indexed by bin, avoiding per-time-step loops.

5. **Backward compatibility.** `local_position_std=None` (default) preserves legacy 1-bin
   behavior exactly. The new behavior is opt-in.

### Encoding Model Sharing (Important Detail)

The encoding model (place fields) is fit once per unique `(environment_name, encoding_group)`
via `fit_encoding_model` (base.py line 2454). `ObservationModel.__eq__` ignores `is_local`
(line 74 of observation_models.py), so the Local and Non-Local states sharing the same
environment are deduplicated by `np.unique()`. The encoding model is **always the full spatial
model** — `is_local` only affects the predict function, not the fit function.

This means multi-bin local can safely call the predict function with `is_local=False` and
get full spatial likelihoods using the already-fitted encoding model. No changes to
`fit_encoding_model` are needed.

---

## Implementation

### Phase 1: Parameter and State Index

#### 1a. Add `local_position_std` to detector classes and base

**File:** `src/non_local_detector/models/base.py`

Add `local_position_std: float | None = None` as a parameter to the base `_SortedSpikesBaseDetector`
and `_ClusterlessBaseDetector` constructors (or whichever common base makes sense). Store as
`self.local_position_std`. This eliminates the need for defensive `getattr()` calls throughout
the codebase — the attribute is always present with a `None` default.

**File:** `src/non_local_detector/models/non_local_model.py`

Add `local_position_std: float | None = None` as a constructor parameter to
`NonLocalClusterlessDetector` and `NonLocalSortedSpikesDetector`, passing through to super.

**Validation:** Add input validation matching the style of `_validate_penalty_params`.
Three valid modes: `None` (legacy single-bin), `0.0` (delta kernel — multi-bin with
one-hot at the animal's bin), `> 0` (Gaussian kernel). Negative values are rejected:
```python
if local_position_std is not None and local_position_std < 0:
    raise ValidationError(
        "local_position_std must be non-negative",
        expected="float >= 0 or None",
        got=str(local_position_std),
        hint=(
            "Set to None for legacy single-bin local behavior, "
            "0.0 for a delta kernel at the animal's bin, or "
            "a positive value for a Gaussian kernel of that width"
        ),
    )
```

#### 1b. Modify `initialize_state_index()` to allocate full bins for multi-bin local

**File:** `src/non_local_detector/models/base.py` (line 793)

Current:
```python
if obs.is_local or obs.is_no_spike:
    bin_sizes.append(1)
    state_ind.append(np.full(1, ind, dtype=int))
    is_track_interior.append(np.ones(1, dtype=bool))
```

Change to:
```python
if obs.is_no_spike or (obs.is_local and self.local_position_std is None):
    bin_sizes.append(1)
    state_ind.append(np.full(1, ind, dtype=int))
    is_track_interior.append(np.ones(1, dtype=bool))
```

When `local_position_std` is set, the local state falls through to the `else` branch and gets
all `environment.place_bin_centers_.shape[0]` bins, same as non-local states.

Note: The `else` branch requires `obs.environment_name` to find the correct environment.
The default `ObservationModel(is_local=True)` has `environment_name=""`, which matches the
default environment. This already works for the non-local defaults. Verify this with a test.

#### 1c. Modify initial conditions for multi-bin local

**File:** `src/non_local_detector/models/base.py` — `initialize_initial_conditions()` (line 816)

At the call site where `make_initial_conditions(obs, environments)` is called, when
`self.local_position_std is not None` and `obs.is_local`, pass a modified copy of the
observation model using `dataclasses.replace(obs, is_local=False)`. This causes
`UniformInitialConditions.make_initial_conditions` to take the spatial (non-local) code path
and return uniform initial conditions over all position bins.

```python
from dataclasses import replace

# In initialize_initial_conditions():
for obs, ic_type in zip(self.observation_models, self.continuous_initial_conditions_types):
    effective_obs = obs
    if obs.is_local and self.local_position_std is not None:
        effective_obs = replace(obs, is_local=False)
    ic = ic_type.make_initial_conditions(effective_obs, self.environments)
    ...
```

No changes to `initial_conditions.py` needed. The `UniformInitialConditions` API stays clean.

### Phase 2: Continuous Transitions

#### 2a. Handle multi-bin -> 1-bin transitions in assembly code

**File:** `src/non_local_detector/models/base.py` (line ~920, in the `else` branch at line 919)

The existing code at line 923 handles 1-bin -> n-bin transitions (`n_row_bins == 1, n_col_bins > 1`).
Add the symmetric case for n-bin -> 1-bin transitions BEFORE the final `else` at line 948:

```python
elif n_row_bins > 1 and n_col_bins == 1:
    # Spatial to non-spatial: each source bin transitions to the
    # single target bin with probability 1.
    self.continuous_state_transitions_[inds] = np.ones((n_row_bins, 1))
```

This handles Local (multi-bin) -> No-Spike (1-bin) transitions correctly.

#### 2b. Auto-upgrade `Discrete()` for multi-bin local states

**File:** `src/non_local_detector/models/base.py` (in `initialize_continuous_state_transitions`)

**Critical: this must happen BEFORE `make_state_transition` is called.** If `Discrete()` is
left in place when `inds` spans `(n_bins, n_bins)`, `make_state_transition()` returns a
`(1, 1)` matrix that NumPy silently broadcasts into the `(n_bins, n_bins)` slice, filling
the entire block with 1.0. This is a data-corruption bug.

Add the auto-upgrade as an early check in the `else` branch (line 919), BEFORE the
size checks at line 920:

```python
else:
    # Auto-upgrade Discrete() for multi-bin local states.
    # Must happen BEFORE make_state_transition() to avoid (1,1) -> (n,n) broadcast.
    if (isinstance(transition, Discrete) and self.local_position_std is not None):
        from_obs = self.observation_models[from_state]
        to_obs = self.observation_models[to_state]
        if from_obs.is_local or to_obs.is_local:
            # Determine correct environment name from the local state's obs model
            local_obs = from_obs if from_obs.is_local else to_obs
            transition = Uniform(environment_name=local_obs.environment_name)

    n_row_bins = np.max(inds[0].shape)
    n_col_bins = np.max(inds[1].shape)
    ...
```

The `Uniform` must be constructed with the correct `environment_name` from the local state's
`ObservationModel`, not a bare `Uniform()` which defaults to `environment_name=""`.

**Scope limitation:** This auto-upgrade only applies to same-environment transitions
(`from_obs.environment_name == to_obs.environment_name`). For cross-environment blocks,
`Uniform` requires `environment2_name` (see `continuous_state_transitions.py` line 285).
If `Discrete()` is used for a cross-environment transition involving a multi-bin local state,
raise an error rather than silently building the wrong block. This scopes the feature to
single-environment detectors for v1; multi-environment support can be added later if needed.

### Phase 3: Position Uncertainty Kernel

#### 3a. Add `_compute_local_position_kernel()` as a base class method

**File:** `src/non_local_detector/models/base.py`

Add as a method on the base class, mirroring `_compute_non_local_position_penalty` (lines
382-427). This keeps it discoverable alongside the penalty and gives direct access to
`self.local_position_std`.

```python
def _compute_local_position_kernel(
    self,
    time: jnp.ndarray,
    position_time: jnp.ndarray,
    position: jnp.ndarray,
    environment: Environment,
) -> jnp.ndarray:
    """Compute log position uncertainty kernel for local state.

    Returns a normalized (in log-space) Gaussian kernel centered on the
    animal's interpolated position at each time step. Uses track graph
    distances when a track graph is available.

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

    Returns
    -------
    log_kernel : jnp.ndarray, shape (n_time, n_interior_bins)
        Log of the normalized position uncertainty kernel.
        n_interior_bins = environment.is_track_interior_.sum(), matching
        the interior-only convention used by state_bin_masks in
        compute_log_likelihood.
    """
    from non_local_detector.likelihoods.common import get_position_at_time

    animal_pos = get_position_at_time(position_time, position, time, environment)

    # Guard against NaN positions before any bin lookup.
    # get_position_at_time() can return NaN rows (e.g., position gaps),
    # and environment.get_bin_ind() has no NaN guard — it will crash or
    # return garbage indices. For NaN time steps, assign a uniform (flat)
    # kernel so the local state is uninformative rather than broken.
    if animal_pos.ndim == 1:
        nan_mask = jnp.isnan(animal_pos)
    else:
        nan_mask = jnp.any(jnp.isnan(animal_pos), axis=-1)  # (n_time,)

    is_interior = environment.is_track_interior_.ravel()
    bin_centers = environment.place_bin_centers_[is_interior]

    # Compute squared distances using precomputed distance matrix
    if (environment.track_graph is not None
            and environment.distance_between_nodes_ is not None):
        # Track graph case: use bin-index lookup into precomputed distance matrix
        animal_bin_inds = environment.get_bin_ind(np.asarray(animal_pos))
        # distance_between_nodes_ is dict-of-dicts for 1D track graph
        # Convert animal bin indices to node IDs
        bin_to_node = np.asarray(environment.place_bin_centers_nodes_df_.node_id)
        interior_bin_indices = np.where(is_interior)[0]
        interior_node_ids = bin_to_node[interior_bin_indices]
        animal_node_ids = bin_to_node[animal_bin_inds]

        n_time = len(time)
        n_bins = int(is_interior.sum())
        sq_dist = np.full((n_time, n_bins), np.inf)
        for t in range(n_time):
            a_node = animal_node_ids[t]
            if a_node == -1:
                continue
            for b, b_node in enumerate(interior_node_ids):
                if b_node == -1:
                    continue
                try:
                    d = environment.distance_between_nodes_[a_node][b_node]
                    sq_dist[t, b] = d ** 2
                except KeyError:
                    pass
        sq_dist = jnp.array(sq_dist)
    elif (environment.distance_between_nodes_ is not None
            and not isinstance(environment.distance_between_nodes_, dict)):
        # 2D grid case: distance_between_nodes_ is an (n_bins, n_bins) array
        animal_bin_inds = environment.get_bin_ind(np.asarray(animal_pos))
        interior_bin_indices = np.where(is_interior)[0]
        # Index: distance_between_nodes_[animal_bin, interior_bins]
        dist_matrix = environment.distance_between_nodes_
        sq_dist = jnp.array(
            dist_matrix[np.ix_(animal_bin_inds, interior_bin_indices)] ** 2
        )
    else:
        # Fallback: Euclidean distances
        if animal_pos.ndim == 1:
            animal_pos = animal_pos[:, jnp.newaxis]
        if bin_centers.ndim == 1:
            bin_centers = bin_centers[:, jnp.newaxis]
        sq_dist = jnp.sum(
            (animal_pos[:, jnp.newaxis, :] - bin_centers[jnp.newaxis, :, :]) ** 2,
            axis=-1,
        )

    log_kernel = -0.5 * sq_dist / (self.local_position_std ** 2)
    # Normalize per time step so kernel is a proper log-probability
    log_kernel -= jax.scipy.special.logsumexp(log_kernel, axis=1, keepdims=True)

    # NaN positions get a uniform (flat) kernel — uninformative rather than broken.
    n_bins = int(is_interior.sum())
    uniform_log_kernel = -jnp.log(n_bins)
    log_kernel = jnp.where(nan_mask[:, jnp.newaxis], uniform_log_kernel, log_kernel)

    return log_kernel
```

**Performance note on track graph path:** The per-time-step loop through the dict-of-dicts
is O(n_time * n_interior_bins). For a typical session (n_time ~ 100k at 500 Hz, n_bins ~ 50),
this is ~5M dict lookups. If profiling shows this is too slow, convert the dict-of-dicts to
a dense matrix once during `fit()` and use array indexing. For v1, correctness over speed.

**Shape convention:** The returned kernel has shape `(n_time, n_interior_bins)` which matches
`state_bin_masks[state_id]` used in `compute_log_likelihood`. This is the same convention
used by `_compute_non_local_position_penalty`.

### Phase 4: Likelihood Assembly

#### 4a. Modify `compute_log_likelihood` in both base classes

**Files:**
- `src/non_local_detector/models/base.py` lines 2536-2671 (clusterless)
- `src/non_local_detector/models/base.py` lines 3409-3540 (sorted spikes)

Both methods need identical changes. Four modifications in each:

**Change 1: Update `needs_position` guard** (lines 2569-2572 and 3441-3444)

Current:
```python
needs_position = (
    np.any([obs.is_local for obs in self.observation_models])
    or non_local_penalty > 0
)
```

Add multi-bin local kernel case:
```python
needs_position = (
    np.any([obs.is_local for obs in self.observation_models])
    or non_local_penalty > 0
    or self.local_position_std is not None
)
```

(Note: when `local_position_std is not None`, `obs.is_local` is likely True anyway, but
being explicit prevents bugs if the conditions are ever refactored independently.)

**Change 2: Compute effective `is_local` for caching and likelihood call** (lines 2606-2627 and 3477-3497)

```python
for state_id, obs in enumerate(self.observation_models):
    # Multi-bin local computes full spatial likelihood (is_local=False)
    # then adds position kernel afterward
    effective_is_local = obs.is_local and self.local_position_std is None

    likelihood_name = (
        obs.environment_name,
        obs.encoding_group,
        effective_is_local,
        obs.is_no_spike,
    )

    if obs.is_no_spike:
        likelihood_results[state_id] = predict_no_spike_log_likelihood(...)
    elif likelihood_name not in computed_likelihoods:
        likelihood_results[state_id] = likelihood_func(
            ...,
            is_local=effective_is_local,
        )
        computed_likelihoods[likelihood_name] = state_id
    else:
        likelihood_results[state_id] = computed_likelihoods[likelihood_name]
```

This means multi-bin local and non-local states sharing `(env_name, encoding_group)` will
have the same `likelihood_name` tuple (both with `effective_is_local=False`). The base
spatial likelihood is computed once and shared. The position kernel is added only to the
local state's bins in the next step.

**Change 3: Add position kernel post-assembly** (after existing penalty application)

```python
# Apply local position kernel if configured
if self.local_position_std is not None:
    env_kernels = {}
    for state_id, obs in enumerate(self.observation_models):
        if obs.is_local:
            env_name = obs.environment_name
            if env_name not in env_kernels:
                env = self._get_environment_by_name(env_name)
                env_kernels[env_name] = self._compute_local_position_kernel(
                    time, position_time, position, env
                )
            is_state_bin = state_bin_masks[state_id]
            log_likelihood = log_likelihood.at[:, is_state_bin].add(
                env_kernels[env_name]
            )
```

This mirrors the `non_local_position_penalty` pattern (lines 2651-2668).

**Change 4: No changes needed to the assembly loop** (lines 2637-2649 and 3507-3519)

The assembly loop uses `state_bin_masks[state_id]` which already accounts for the correct
number of bins per state (set up by `initialize_state_index` in Phase 1b). When multi-bin
local is active, `state_bin_masks` for the local state will have `n_interior_bins` True
entries instead of 1, and the likelihood result will have matching shape since it was
computed with `is_local=False`.

#### 4b. No changes to individual likelihood backends

Since multi-bin local uses `is_local=False` to compute the base likelihood and adds the
position kernel afterward, the individual likelihood functions (`sorted_spikes_kde`,
`clusterless_kde`, `sorted_spikes_glm`, `clusterless_gmm`, `clusterless_kde_log`) do not
need any modifications. The kernel is added at the assembly level in `compute_log_likelihood`.

### Phase 5: Results Assembly

There is one `_convert_results_to_xarray` (line 1976) and one `_convert_seq_to_df` (line 2155)
on the shared base class. Both need the same condition change.

#### 5a. Fix position coordinate assignment in `_convert_results_to_xarray`

**File:** `src/non_local_detector/models/base.py` (line 2037)

Current:
```python
if obs.is_local or obs.is_no_spike:
    position.append(np.full((1, n_position_dims), np.nan))
```

Change to:
```python
if obs.is_no_spike or (obs.is_local and self.local_position_std is None):
    position.append(np.full((1, n_position_dims), np.nan))
```

When multi-bin local is active, the local state falls through and gets real position
coordinates from `environment.place_bin_centers_`, same as non-local states.

#### 5b. Fix Viterbi/MAP sequence conversion in `_convert_seq_to_df`

**File:** `src/non_local_detector/models/base.py` (line 2175)

Same pattern:
```python
if obs.is_no_spike or (obs.is_local and self.local_position_std is None):
    position.append(np.full((1, n_position_dims), np.nan))
```

### Phase 6: Testing

#### 6a. Unit tests for position kernel

- Kernel is a valid log-probability distribution (sums to 1 in probability space per time step)
- Kernel is centered on the animal's position (peak at nearest bin)
- Kernel narrows as `local_position_std` decreases
- Track graph distances are used when track graph is available
- Handles edge cases: animal at track boundary, NaN positions
- Shape assertion: output shape is `(n_time, n_interior_bins)`

#### 6b. Integration tests

- Multi-bin local model produces valid posteriors (probabilities sum to 1)
- Local state posterior is concentrated near the animal's position
- Legacy behavior (`local_position_std=None`) produces identical results to current code
- Multi-bin local + non-local states produce valid combined posteriors
- Transition matrices are properly constructed (stochastic, correct shape)
- Validation rejects `local_position_std < 0`; accepts `None`, `0.0` (delta kernel), and `> 0` (Gaussian kernel)

#### 6c. Backward compatibility tests

- All existing tests pass unchanged when `local_position_std=None`
- Snapshot tests produce identical results (no regression)
- Golden regression tests pass
- Add a `local_position_std=None` snapshot alongside the new `local_position_std=X` snapshot
  for ongoing regression protection

#### 6d. EM re-estimation validation

**Correction:** The current EM M-step uses only the discrete local-state probability per time
point (`acausal_state_probabilities[:, local_state_index]`) as weights to `fit_encoding_model`
(base.py line ~1595), NOT the spatial posterior over bins. The "blur place fields across bins"
concern was based on a misreading of the M-step. Verify:

- Confirm the M-step still uses discrete state probabilities (not spatial posterior) with
  multi-bin local — i.e., multi-bin local does not change the EM encoding update path
- If the M-step mechanism has changed to consume spatial posteriors, test for place field
  distortion and add masking if needed
- Document the finding either way

#### 6e. Interaction tests

- Test that `non_local_position_penalty > 0` and `local_position_std` enabled simultaneously
  produces valid results without double-counting
- Document the intended usage: the penalty suppresses non-local near the animal; the kernel
  anchors local near the animal. They are complementary but users should understand both.

---

## Files Changed (Summary)

| File | Change | Phase |
|------|--------|-------|
| `models/base.py` constructor | Add `local_position_std` param with validation | 1a |
| `models/non_local_model.py` | Add `local_position_std` param, pass to super | 1a |
| `models/base.py` `initialize_state_index` | Multi-bin local gets full position bins | 1b |
| `models/base.py` `initialize_initial_conditions` | Use `dataclasses.replace` for multi-bin local obs | 1c |
| `models/base.py` `initialize_continuous_state_transitions` | n->1 case; auto-upgrade `Discrete` BEFORE `make_state_transition` | 2a, 2b |
| `models/base.py` `_compute_local_position_kernel` | New method mirroring penalty pattern | 3a |
| `models/base.py` `compute_log_likelihood` (x2) | `needs_position` guard; effective `is_local`; kernel post-assembly | 4a |
| `models/base.py` `_convert_results_to_xarray` | Multi-bin local gets real positions | 5a |
| `models/base.py` `_convert_seq_to_df` | Multi-bin local gets real positions | 5b |

**Files NOT changed:**
- `observation_models.py` — `is_local` semantics unchanged
- `initial_conditions.py` — no API changes (handled via `dataclasses.replace` at call site)
- `models/_defaults.py` — defaults unchanged (backward compatible)
- All likelihood backends — no changes (kernel added at assembly level)
- `continuous_state_transitions.py` — no new classes needed

## Open Questions for Later

1. **Non-local position penalty interaction:** With multi-bin local, the local state may absorb
   some of what the penalty was doing. Test empirically whether the penalty is still needed.
   (Phase 6e adds a test for correctness when both are enabled.)

2. **Performance of track graph kernel:** The dict-of-dicts loop is O(n_time * n_interior_bins).
   If too slow, convert to a dense matrix during `fit()`. Profile on real data.

3. **Optimal `local_position_std`:** What value makes scientific sense? Likely depends on
   position tracking precision and place field width. Could be estimated from data.
