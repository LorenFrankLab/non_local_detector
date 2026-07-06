# Shared contracts

[← back to PLAN.md](PLAN.md)

Contracts referenced by ≥2 phases. Each appears once here; phases link in by anchor.

- [Engine API (`diffusion.py`)](#engine-api) — Phase 1 defines, Phases 2 & 3 consume.
- [Environment eig cache](#eig-cache) — Phase 1 defines, Phases 2 & 3 rely on.
- [Sorted-spikes encoding-dict + predict contract](#encoding-dict) — Phase 2 & 3.

---

## <a id="engine-api"></a>Engine API (`likelihoods/diffusion.py`)

Pure NumPy/SciPy (host-side; runs at fit time, like the KDE fit's `scipy.interpolate`).
Array shapes documented NumPy-style. Algorithms in [designs.md](designs.md).

```python
def build_laplacian(graph: nx.Graph) -> scipy.sparse.csr_matrix:
    """Finite-difference symmetric graph Laplacian L = D - W, edge weight w = 1/distance**2.
    graph nodes are 0..n_interior-1 (contiguous, interior-bin order); each edge has a
    'distance' attribute. σ-independent."""

def diffusion_eigenbasis(
    L: scipy.sparse.spmatrix, rank: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of L. Returns (eigvals (m,), eigvecs (n_bins, m)) ascending.
    rank None -> dense scipy.linalg.eigh (all n modes). rank < n -> truncated smallest-`rank`
    modes via scipy.sparse.linalg.eigsh **with sigma=-1e-8, which='LM'** (shift-invert at a
    small NEGATIVE shift — sigma=0 factorizes the singular L and is unreliable: it raises
    'Factor is exactly singular' in some environments and silently returns garbage in others;
    `which='SM'` is the no-shift-invert fallback). Truncation MUST include **all** zero modes
    (one per connected component, multiplicity = number of components), so require
    rank >= n_components; omitting any breaks component-wise mass conservation."""

def diffuse(
    eigvals: np.ndarray, eigvecs: np.ndarray, sigma: float, fields: np.ndarray
) -> np.ndarray:
    """Apply the heat kernel exp(-t L), t = sigma**2/2, to fields (n_bins, n_fields):
    eigvecs @ (exp(-t*eigvals)[:, None] * (eigvecs.T @ fields)). Clip tiny negatives to 0.
    Returns (n_bins, n_fields)."""

def to_density(smoothed: np.ndarray, bin_sizes: np.ndarray) -> np.ndarray:
    """Normalize each column of `smoothed` (n_bins, n_fields) to an ∫=1 density:
    smoothed / (bin_sizes @ smoothed). Zero-mass columns -> zeros."""

def environment_graph(
    environment: "Environment",
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Interior-bin graph (nodes 0..n_interior-1 in place_bin_centers_[is_track_interior]
    order, edges carry 'distance'), node_order (the interior flat-bin indices, shape
    (n_interior,)), and bin_sizes (per-interior-bin volume, shape (n_interior,))."""
```

**Invariants (do not weaken):**
- `L` is symmetric with `1ᵀL = 0` (so `exp(-tL)` is column-stochastic → mass-conserving).
- Node/row order is exactly `place_bin_centers_[is_track_interior]` order, i.e.
  `np.where(is_track_interior_.ravel())[0]`. Every consumer indexes place fields on interior
  bins in this order.
- `diffuse` never materializes a dense `(n_bins, n_bins)` kernel.

## <a id="eig-cache"></a>Environment eig cache

The eigenbasis depends only on the graph (not σ, not weights), so it is computed once and
cached on the `Environment`, mirroring the `_bin_distance_matrix_` pattern
(`environment.py:480`).

- Attribute: `environment._diffusion_eigenbasis_` is a **dict keyed by `rank`**
  (`None` = full) → `(eigvals, eigvecs)`, lazily populated per rank. A single tuple is wrong:
  `diffusion_eigenbasis` takes `rank`, so first-caller-wins would let a truncated Phase-2 fit
  poison a later full-rank MRF fit (or a dense first call defeat truncation). A cached
  full-rank (`None`) entry may serve any `rank` request by slicing its first `rank` columns
  (valid — `eigh` returns modes ascending); a truncated entry serves only that rank.
- **Invalidation:** clear `_diffusion_eigenbasis_` alongside the existing
  `_bin_distance_matrix_` invalidation in `fit_place_grid` (`environment.py:480-481`).
- Keyed by environment identity + rank (the graph is fixed once fitted); reused across all
  neurons and all EM refits.

## <a id="encoding-dict"></a>Sorted-spikes encoding-dict + predict contract

`base.py:4043-4074` **splats the entire encoding dict as kwargs** into the predict
function: `likelihood_func(time, position_time, position, spike_times,
**encoding_model_[name], is_local=…)`. Therefore:

- **Every key in the fit's returned dict must be a parameter of the predict function**
  (extra keys → `TypeError: unexpected keyword argument`).
- `base.py:3886` filters the fit's *inputs* by `inspect.signature`, so the fit's parameter
  names must match `_encoding_model_data` keys it needs.

**`sorted_spikes_diffusion` encoding dict** = the `sorted_spikes_kde` keys —
`environment`, `occupancy`, `mean_rates`, `place_fields`, `no_spike_part_log_likelihood`,
`is_track_interior`, `disable_progress_bar` — plus the diffusion additions
`node_order`, `bin_sizes` (and any engine handle needed by the local predict path). The
dedicated `predict_sorted_spikes_diffusion_log_likelihood` must list **all** of these as
parameters. It does **not** reuse `predict_sorted_spikes_kde_log_likelihood` (whose
signature requires `marginal_models`/`occupancy_model`, which the diffusion dict omits).

**`place_fields` and `no_spike_part_log_likelihood` are FULL-GRID, exactly like KDE**
(shapes `(n_neurons, n_total_bins)` and `(n_total_bins,)`), **not** interior-only. The
engine returns interior-bin densities; the fit scatters them into full-grid zeros via
`jnp.zeros((n_total_bins,)).at[is_track_interior].set(...)` (mirroring
`sorted_spikes_kde.py:204-219`). This is load-bearing: the non-local predict slices
`place_field[is_track_interior]` (`sorted_spikes_kde.py:361`), and the local predict indexes
by `environment.get_bin_ind(...)`, which returns **full-grid flat indices**
(`environment.py:707`). Interior-only storage silently breaks any environment with
non-interior (gap/barrier) bins.
