# Likelihood input and fitted-state repairs

The public fits reject invalid numerical configuration before producing a
plausible but unusable encoding model. Scalar Python, NumPy and JAX bandwidths
and one-dimensional bandwidth vectors share one validation path. Bandwidths
must be finite, positive, match the spatial/mark dimensions and remain
representable in the active JAX floating-point dtype.

Short varying spatial axes receive one natural-cubic interior knot at their
observed midpoint when the requested spacing supplies none. Each axis retains
its independent knot vector: Patsy's `te` constructs the tensor product.
Constant axes retain the existing intercept behavior. This fixes ordinary
0–10 environments at the default spacing without changing optimizer settings.

GMM fits convert integer measurements to floating point before effective-sample
validation. Every restart's **returned** covariance, precision, means, weights
and lower bound must be finite before it can replace the best restart. A finite
objective evaluated before the final M-step cannot certify those parameters.
The existing lower-bound semantics and convergence criteria are unchanged.

Additional consistent contracts: standalone No-Spike rates must be finite and
nonnegative; valid rate differentiation remains supported. Sorted KDE/GLM
empty populations return empty fields and zero summed rate. Diffusion/MRF
prediction reconstructs an omitted interior-log-field cache. GMM respects its
existing all-interior fallback for precomputed grids without an interior mask.

## Numerical evidence

- Before: a one-step, unregularized, warm-started GMM on identical samples
  succeeded with lower bound −1.4189385, zero covariance and NaN scores.
  After: it rejects the singular final state with the existing actionable
  fitting error.
- Before: NumPy float32 bandwidths raised AssertionError; invalid bandwidths
  could generate floor-rate maps. After: valid values fit, invalid values fail
  with named ValidationError messages.
- The original 13 new regression cases all failed before implementation and
  pass after the repair. Existing focused GMM-hardening, GLM and sorted-KDE
  coverage also passed (142 tests in the first focused run).

Stored snapshot/golden data and test tolerances are not changed by this PR.

Independent review verified the accepted zero-rate boundary: an empty
one-second bin has likelihood `-rate` and derivative −1 even at rate zero.
Guarding xlogy's inactive event argument preserves this derivative and leaves
positive counts at zero rate impossible. The final reviewed input/Poisson
focused run passed 105 tests at unchanged tolerances. Existing tests which
passed invalid negative/nonfinite public rates now assert the new explicit
ValidationError contract.
