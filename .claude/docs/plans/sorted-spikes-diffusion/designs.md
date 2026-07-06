# Designs

[← back to PLAN.md](PLAN.md)

Per-component algorithms. Phases reference these by anchor.

- [Finite-difference Laplacian](#laplacian)
- [Eigendecomposition + diffuse](#eig)
- [Density normalization + place field](#density)
- [Adapter: N-D / 1D-grid branch](#adapter-nd)
- [Adapter: linearized track_graph branch](#adapter-1d)
- [MRF-GAM population fit (Phase 3)](#mrf)

---

## <a id="laplacian"></a>Finite-difference Laplacian

Rationale: neurospatial weights edges `exp(-d²/2σ²)`, which makes the effective bandwidth
≈ σ·bin_size (grid-dependent — the B1 blocker; numbers in [appendix.md](appendix.md)). The
finite-difference weight `1/d²` on a **face-adjacent** grid gives `L ≈ -∂²`, so `exp(-tL)`,
`t=σ²/2`, is a Gaussian of std σ **independent of bin size** (verified: std/σ = 1.000 at
bin sizes 0.5/1/2/4 in 1D, and = 1.000 on a 2D face-adjacent grid). `L` is σ-independent →
built once.

**Face-adjacency only.** `build_laplacian` must receive a face-adjacent graph. Do **not**
include Moore/diagonal edges: with `1/d²` weighting the diagonals inflate the small-wavenumber
diffusion coefficient, oversmoothing by ≈√2 in 2D (verified: 8-connected std/σ = 1.413 vs
face-only 1.000). The N-D adapter drops diagonal edges (see [adapter-nd](#adapter-nd)); the
linearized branch is already a chain+junction graph with no grid diagonals. (A calibrated
isotropic 9-point stencil is a possible future refinement; face-only is the first-cut choice.)

```python
def build_laplacian(graph):
    n = graph.number_of_nodes()
    rows, cols, vals = [], [], []
    for u, v, data in graph.edges(data=True):
        w = 1.0 / (data["distance"] ** 2)      # finite-difference weight
        rows += [u, v]; cols += [v, u]; vals += [w, w]
    W = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(W.sum(axis=1)).ravel()
    return (scipy.sparse.diags(deg) - W).tocsr()
```

Symmetric, `1ᵀL = 0`. On disconnected components each component has its own null mode;
`exp(-tL)` never mixes across them (reflecting boundaries / no arm leakage).

## <a id="eig"></a>Eigendecomposition + diffuse

`L = QΛQᵀ`. Dense `scipy.linalg.eigh(L.toarray())` for `n_bins ≤ threshold`; truncated
smallest-`rank` modes for large `n_bins` via
`scipy.sparse.linalg.eigsh(L, k=rank, sigma=-1e-8, which="LM")` — **negative shift**, because
`sigma=0` factorizes the singular Laplacian and is unreliable (raises "Factor is exactly
singular" in some environments, silently returns garbage in others; verified). `which="SM"`
without shift-invert is the fallback. **Keep all zero modes** — one per connected component
(multiplicity = number of components), so require `rank ≥ n_components`; omitting any breaks
component-wise mass conservation and can corrupt an isolated arm. Truncating from a cached
full-rank `eigh` is equivalent to slicing its first `rank` columns. Clip eigenvalues at 0.

```python
def diffuse(eigvals, eigvecs, sigma, fields):
    t = sigma ** 2 / 2.0
    coeff = np.exp(-t * eigvals)                    # (m,)
    proj = eigvecs.T @ fields                       # (m, n_fields)
    out = eigvecs @ (coeff[:, None] * proj)         # (n_bins, n_fields)
    return np.clip(out, 0.0, None)
```

`fields` stacks occupancy + all neuron count-fields → all neurons diffused in one matmul.
Mode reconstruction `(eigvecs * coeff) @ eigvecs.T == expm(-tL)` at full rank
(test 2; mgcv_mrf's `test_basis_are_diffusion_modes`).

## <a id="density"></a>Density normalization + place field

`exp(-tL)` conserves field *sums*, not the area-integral. Convert a smoothed count field to
an ∫=1 density with per-bin volumes:

```python
def to_density(smoothed, bin_sizes):          # smoothed (n_bins, n_fields)
    mass = bin_sizes @ smoothed               # (n_fields,)
    safe = np.where(mass > 0, mass, 1.0)
    return np.where(mass > 0, smoothed / safe, 0.0)
```

Place field mirrors `sorted_spikes_kde.py:204-219` exactly (so units match; pinned by the
KDE drop-in test). `occupancy` is the `to_density` of the position field weighted by
`weights`; `marginal_k` is the `to_density` of neuron `k`'s spike field **weighted by
`weights_at_spike_times`** (the `weights` interpolated to each spike time,
`sorted_spikes_kde.py:178`) — not unweighted spike counts. `mean_rate_k =
weights_at_spike_times.sum() / weight_sum` (KDE convention, default `weights` = ones). EM
passes non-uniform posterior `weights` (`base.py:1991`), so this weighting is load-bearing.
The rate is computed on interior bins then **scattered into a full-grid array** exactly like
KDE (non-interior bins stay 0), so `place_fields` is `(n_neurons, n_total_bins)`:

```python
rate_interior = mean_rate_k * np.where(occupancy > 0.0,
                                       marginal_k / np.where(occupancy > 0.0, occupancy, 1.0),
                                       EPS)
place_field_k = (jnp.zeros((n_total_bins,))
                 .at[is_track_interior].set(jnp.clip(rate_interior, EPS, None)))
```

`no_spike_part_log_likelihood = place_fields.sum(0)` is likewise full-grid; predict slices
`[is_track_interior]` (non-local) and indexes by `get_bin_ind` (local, full-grid). On a
uniform grid the `bin_sizes` factor cancels in the ratio; on uneven bins it is required
(test 6).

## <a id="adapter-nd"></a>Adapter — N-D / 1D-grid branch (`track_graph is None`)

`track_graphDD` node ids = flat bin index over **all** bins
(`make_nD_track_graph_from_environment`, `environment.py:1599`); interior nodes carry the
Moore-neighborhood edges with `distance` (`environment.py:1659`); non-interior nodes are
isolated.

```python
interior = np.where(environment.is_track_interior_.ravel())[0]   # node_order
sub = environment.track_graphDD.subgraph(interior).copy()
# Drop Moore/diagonal edges — keep only face-adjacent pairs (centers differ in exactly one
# dimension). Diagonals with 1/d² weights oversmooth ≈√2 in 2D (see #laplacian).
pos = nx.get_node_attributes(sub, "pos")
face_edges = [(u, v, d) for u, v, d in sub.edges(data=True)
              if np.count_nonzero(np.asarray(pos[u]) - np.asarray(pos[v])) == 1]
face = nx.Graph(); face.add_nodes_from(sub.nodes(data=True)); face.add_edges_from(face_edges)
relabel = {old: new for new, old in enumerate(interior)}          # → 0..n_interior-1
graph = nx.relabel_nodes(face, relabel, copy=True)                # edges keep 'distance'
```

`bin_sizes`: product of per-dim `np.diff(edges_[d])` widths, meshed `'ij'`, raveled, then
`[interior]`. Exclude padding bins — they are non-interior, so the `[interior]` selection
drops them (`get_grid` padding, `environment.py:1068`).

## <a id="adapter-1d"></a>Adapter — linearized `track_graph` branch (highest risk)

Substrate is the environment's `track_graph_with_bin_centers_edges_` +
`place_bin_centers_nodes_df_` (from `get_track_grid`). Bin-center nodes have
`is_bin_edge=False`; `place_bin_centers_nodes_df_.node_id` maps each place bin (in
`place_bin_centers_` order) to its node id, with `-1` for gap bins
(`edge_spacing>0`, `is_track_interior=False`).

**Primary construction — contract to a bin-center graph.** Interior bin-center node ids are
`node_id[is_track_interior]` (excludes `-1` gaps → arms auto-disconnect at gaps). For each
interior bin-center node, BFS over `track_graph_with_bin_centers_edges_` through *non-center*
nodes (bin-edge and original junction nodes) until reaching other bin-center nodes; add an
edge to each such neighbor with `distance` = summed path length. Junctions connect arms
through shared original nodes; gap bins are absent so arms stay separate. Relabel interior
centers to `0..n_interior-1` in `place_bin_centers_[is_track_interior]` order.

**Fallback (if contraction proves brittle):** run the diffusion on the *full*
`track_graph_with_bin_centers_edges_` (all node types as conduction points), then read
values only at interior bin-center rows. Document whichever is chosen; both must pass the
same round-trip + junction + gap tests (Phase 1).

`bin_sizes`: `np.diff(place_bin_edges_)` then `[is_track_interior]` (excludes the wide gap
"bins", `environment.py:1481`).

## <a id="mrf"></a>MRF-GAM population fit (Phase 3)

`B` = the `rank` smoothest eigenmodes (from the shared engine); penalty weights = their
eigenvalues `d`. Model `n_ik ~ Poisson(o_i · exp((Bγ_k)_i))`, occupancy `o` a **shared
log-offset**. Penalty `λ·γ_kᵀ diag(d) γ_k` — a generalized ridge in the eigenbasis.

Population fit (NeMoS trick): `B (n_bins, rank)` and `log(o)` shared; response
`N (n_bins, n_neurons)`; coefficient matrix `Γ (rank, n_neurons)`. Penalized IRLS vectorized
over neurons:

```
η = B @ Γ                          # (n_bins, n_neurons)
μ = o[:, None] * exp(η)            # (n_bins, n_neurons)
# per-neuron Newton step, vmapped over the neuron axis:
#   (Bᵀ diag(μ_k) B + λ diag(d)) Δγ_k = Bᵀ(N_k - μ_k) - λ diag(d) γ_k
```

`λ` by REML (Wood 2011); default a single shared `λ` (Open Question 3). Emit `η` (log-rate)
→ `place_fields = exp(η)`; same encoding-dict + Poisson-predict contract as Phase 2. Fit in
JAX (`vmap` over neurons); depend-on-NeMoS vs implement-directly is Open Question 2.
Structurally mirrors `sorted_spikes_glm.py:223,359`.
