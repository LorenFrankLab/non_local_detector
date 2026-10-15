# Consistent marked and ground processes for clusterless GMMs

New `clusterless_gmm` fits use the position marginal of each electrode's joint
position/mark GMM for its ground-process intensity. This is the GMM-only scope
of PR #76; diffusion kernels and their normalization are unchanged.

## Mathematical contract

Let `r` be the electrode's mean firing rate in Hz, `p_occ(x)` its encoding
occupancy density, and `p_joint(x, m)` its fitted joint position/mark density.
The marked intensity is

```text
lambda(x, m) = r * p_joint(x, m) / p_occ(x).
```

Integrating over marks gives the ground intensity used in the expected-count
term:

```text
lambda_ground(x) = r * p_joint_position(x) / p_occ(x)
                = integral lambda(x, m) dm.
```

For a Gaussian mixture, this marginal retains the component weights and the
position coordinates of each mean, and uses the position covariance blocks.
Its precision Cholesky factors must be recomputed from those covariances;
slicing a joint precision matrix would describe a conditional Gaussian instead.
Full, tied, diagonal, and spherical covariance types are supported.

The previous independent position GMM could have different mixture weights,
means, covariances, and component count. Its ground intensity need not equal the
integral of the marked intensity, even when both models fit successfully.

The existing numerical floors and empty-electrode policy remain in effect.
The identity above describes electrodes with fitted joint models, before the
summed ground intensity's `EPS` floor. It does not introduce a background-spike
model for electrodes with no effective encoding spikes.

## Configuration and saved models

- Set `gmm_components_joint` and `gmm_covariance_type_joint` to configure both
  the joint distribution and its marginal. Sparse electrodes reduce both to
  the same effective component count.
- `gmm_components_gpi` and `gmm_covariance_type_gpi` remain accepted for source
  compatibility, but explicit non-`None` values produce a `FutureWarning` and
  have no effect. Their defaults are now `None`.
- The encoding dictionary retains `gpi_models`, containing the derived
  marginals. Its requested/effective GPI component counts describe those
  marginals. Their convergence metadata refers to the joint fit; their
  `lower_bound_` is `None` because no separate marginal optimization occurred.
- Predictions from previously saved fits continue to use their stored GPI
  models. Refit to obtain the consistent ground process. Other model-version
  and time-contract compatibility checks still apply.

The event-intensity prediction kernels and occupancy fit are unchanged.
Refitting changes the ground-rate term, so both local and non-local likelihoods
and downstream decoded results can change materially.

## Validation

Deterministic simulated tests compare the marginal to an independent SciPy
Gaussian-mixture calculation for every covariance type, including correlated
two-dimensional positions. Numerical integration of the public marked
likelihood over marks agrees with the public no-spike compensator for both
local and non-local prediction. Tests also cover nonuniform encoding weights,
pickle round trips, and fitting only the occupancy and joint models.

A CPU comparison against `origin/main` at `aff157c3` used two slices of the
local `Jaq_03_16` recording. Position was downsampled to 50 Hz; positions were
linearized, spatial bins were 5 cm, and decode bins were 20 ms. Fits used seed
0, full covariance, 8 occupancy components, 16 joint components, and the
default covariance regularization of 1.0. The baseline used 8 independent GPI
components. The first slice trained for 60 s on one electrode and evaluated
the next 5 s; the larger slice trained for 180 s on three electrodes and
evaluated the next 10 s.

| Measurement | 60 s / one electrode | 180 s / three electrodes |
|---|---:|---:|
| Training spikes | 3,575 | 9,774 / 18,053 / 5,757 |
| Held-out spikes | 282 | 482 / 1,078 / 299 |
| Maximum relative ground discrepancy before | 2.279 | 1.171 |
| Maximum relative ground discrepancy after | 2.80e-6 | 5.01e-6 |
| Maximum absolute non-local log-likelihood change | 6.11 | 9.21 |
| Maximum absolute local log-likelihood change | 18.79 | 9.21 |
| Bins with changed spatial likelihood argmax | 244 / 250 | 335 / 500 |

The ground discrepancy uses an independent float64 SciPy marginal and is
measured where that reference exceeds 1% of its peak. The fitted joint and
occupancy weights, means, and covariances were bitwise identical before and
after. Non-local log-likelihood changes matched the change in the
expected-count term within 3.7e-5 absolute error. All evaluated likelihoods
were finite. Loading the saved baseline models under the changed code
reproduced both local and non-local baseline likelihoods bitwise on both slices.

These short-training examples show that this correction is not merely
rounding error. The argmax comparison is over spatial likelihoods, not an HMM
posterior; it does not establish improved decoding accuracy. This validation
does not measure GPU performance or claim a speedup.
