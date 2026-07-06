# Appendix — upstream references and verification

[← back to PLAN.md](PLAN.md)

## Upstream source (read-only references)

- **neurospatial** `/Users/edeno/Documents/GitHub/neurospatial` @ `648ffda`:
  - `src/neurospatial/ops/smoothing.py:72-234` `compute_diffusion_kernels` — the ported
    idea (graph-Laplacian heat kernel), and `:55-69` `_assign_gaussian_weights_from_distance`
    — the `exp(-d²/2σ²)` weighting we **do not** port (B1). `:237-439` `apply_kernel`
    (forward/adjoint) — reference for the mass-weighted adjoint if ever needed.
  - `src/neurospatial/environment/fields.py:44-148` `compute_kernel` (dense `expm`),
    `:150-299` `smooth`; `src/neurospatial/encoding/_smoothing.py:340-342` — the
    `rate = smooth(spikes)/smooth(occupancy)` form (parity with `place_field.py`); `:495`
    passes bandwidth straight to `compute_kernel` (no bin-size rescaling → inherits B1).
- **mgcv MRF prototype** `/Users/edeno/Downloads/mgcv_mrf.py` (Phase 3 reference):
  penalty `S = D - A = L` (`build_grid_graph`), reduced-rank basis = smallest-eigenvalue
  eigenvectors of `L` (`mrf_basis`), penalized-Poisson IRLS + REML (`_fit_*`, `_reml_*`),
  occupancy as log-offset (`mrf_rate_map`). `test_mgcv_mrf.py:66-86`
  (`test_basis_are_diffusion_modes`) is the source of Phase-1 test 2 (mode reconstruction).
- **NeMoS** https://github.com/flatironinstitute/nemos — JAX; `PopulationGLM().fit(X,
  spike_counts)` with `spike_counts` shape `(n_samples, n_neurons)` → shared design matrix,
  coefficient matrix `(n_features, n_neurons)`; Ridge/Lasso/GroupLasso/ElasticNet. The
  Phase-3 population-fit pattern.
- **spatstat port** `/Users/edeno/Downloads/density_heat.py`, `test_density_heat.py` —
  optional dense oracle; source of the mass-conservation and analytic-Gaussian invariant
  tests. Its `Nstep = max(16, ⌈σ²/(2·pmax·minstep²)⌉)` calibration is the same physical
  bandwidth calibration the `1/d²` weighting achieves.

## Verification numbers (reproduced during design review)

**B1 — effective smoothing std vs bin size** (σ=5, 1-D path graph, `expm_multiply`):

| bin_size h | Gaussian weight `exp(-d²/2σ²)` | finite-diff `1/d²` |
| --- | --- | --- |
| 0.5 | 0.499·σ | 1.000·σ |
| 1.0 | 0.990·σ | 1.000·σ |
| 2.0 | 1.922·σ | 1.000·σ |
| 4.0 | 3.409·σ | 1.000·σ |

Finite-difference weighting → bandwidth = `position_std`, grid-independent. Gaussian
weighting → ≈ σ·bin_size. This is the B1 regression the Phase-1 bandwidth-invariance test
guards. (neurospatial's `diffusion_kde` uses the Gaussian weighting with no bin-size
rescaling, so it carries the same grid-dependence — worth an upstream report.)

**B2 — density normalization.** With `K_raw = exp(-t·M⁻¹L)`, neurospatial's column-
normalized density kernel satisfies `K_density @ field = K_raw @ (field / bin_sizes)`
(recoverable only by pre-dividing inputs by `bin_sizes`, not post-scaling). Our design
avoids `M⁻¹L` entirely: symmetric `L` + explicit `to_density` normalization
([designs.md](designs.md#density)); on uniform grids the factor cancels in the ratio.
