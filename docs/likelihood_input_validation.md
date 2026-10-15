# Likelihood input and fitted-state validation

Encoding fits reject numerical configurations that would otherwise produce a
plausible-looking but unusable model, and they keep a few edge cases (short
spatial axes, empty populations, integer features) working.

## Kernel bandwidths

`validate_bandwidth` checks `position_std` (sorted and clusterless KDE) and
`waveform_std` (clusterless KDE) at fit time. A bandwidth may be a Python,
NumPy or JAX scalar, broadcast to every dimension, or one value per dimension.
Every value must be finite and positive and stay so in JAX's default floating
dtype. A value that overflows to inf or flushes to zero in float32 is rejected
unless float64 is enabled. A zero, negative or non-finite bandwidth would
otherwise fit a degenerate kernel: on main, sorted KDE fit with `position_std=0`
gave a constant 5e-13 floor rate at every bin instead of an error, and NumPy
scalars such as `np.float32(1.0)` raised a bare `AssertionError`. Invalid
values raise `ValidationError` naming the parameter.

## Gaussian mixtures

A GMM's lower bound is evaluated before its final M-step, so a finite lower
bound does not show that the returned weights, means, covariances and
precision factors are usable. Each restart's returned parameters must be
finite before the restart can replace the best one. One unregularized EM step
on identical samples previously returned a zero covariance, a finite lower
bound of -1.419 and NaN scores; it now raises the existing singular-covariance
fitting error. Integer and boolean features are converted to floats before
fitting, instead of raising.

## Spline design matrices

When a varying spatial axis is shorter than the knot spacing, no interior knot
falls inside the data and Patsy raised an error (for example, a 0-10
environment at the default spacing). Such an axis now gets one natural-cubic
knot at the midpoint of its observed positions. Each axis keeps its own knot
vector, and Patsy's `te` forms the tensor product; the previous `np.meshgrid`
of the knot vectors emptied every axis when one was empty. Axes with interior
knots, and constant axes, are unchanged.

## Other boundary contracts

- `predict_no_spike_log_likelihood` requires a finite, nonnegative scalar
  rate. A rate of -1 gave a log likelihood of +1. The event term is guarded so
  a zero rate keeps its exposure gradient of -1 in empty bins (it was NaN).
- Sorted KDE and GLM fits with no neurons return place fields of shape
  `(0, n_bins)` and a zero summed rate instead of failing to stack.
- Sorted diffusion and MRF prediction rebuilds `interior_log_place_fields` from
  `place_fields` when it is omitted, as its signature documents.
- Clusterless GMM fits keep their all-interior fallback for grids without an
  interior mask; an unconditional `is_track_interior_.ravel()` overwrote it.

Valid inputs give the same outputs as before.
