# Sorted-spikes diffusion / MRF likelihood — Implementation Plan

**Status:** Not started.

Adds opt-in sorted-spike place-field likelihoods that smooth on the environment's
manifold graph instead of with Gaussian KDE, so place fields respect track geometry
(walls, holes, junctions) in 1D and N-D. A shared spectral engine (the eigendecomposition
of a finite-difference graph Laplacian, cached on the `Environment`) powers a linear
diffusion smoother (`sorted_spikes_diffusion`, shipped first) and — as a fast-follow — a
penalized-Poisson MRF-GAM (`sorted_spikes_mrf`) fit for the whole population at once. Users
opt in via `sorted_spikes_algorithm="sorted_spikes_diffusion"`; nothing changes for
existing algorithms.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design / algorithm?** [designs.md](designs.md).
4. **Need broader scope / risks / integration points?** [overview.md](overview.md).
5. **Need upstream-repo line refs / verification numbers?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, risks, rollout.
- [shared-contracts.md](shared-contracts.md) — `diffusion.py` engine API, Environment eig
  cache, sorted-spikes encoding-dict + predict contract.
- [designs.md](designs.md) — finite-difference `L`, eig + diffuse, density normalization,
  the two adapter branches, MRF-GAM population fit.
- Phases (each ships as a separable PR):
  - [phase-1-spectral-engine.md](phase-1-spectral-engine.md) — `diffusion.py`: `L`,
    `eig(L)`, diffuse-via-modes, both adapter branches, `bin_sizes`, Environment caching +
    numerical tests.
  - [phase-2-diffusion-likelihood.md](phase-2-diffusion-likelihood.md) —
    `sorted_spikes_diffusion` fit/predict, registry, KDE-equivalence / end-to-end /
    snapshot, docs.
  - [phase-3-mrf-gam.md](phase-3-mrf-gam.md) — fast-follow: `sorted_spikes_mrf` population
    Poisson GAM on the shared engine.
- [appendix.md](appendix.md) — neurospatial / mgcv_mrf / NeMoS / spatstat refs, B1/B2
  verification numbers.
