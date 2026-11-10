"""GMM EM convergence diagnostics: ``converged_``, ``n_iter_`` and the warning."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.gmm import GaussianMixtureModel

pytestmark = pytest.mark.unit


def test_gmm_max_iter_warning():
    """``GaussianMixtureModel.fit`` should warn and set ``converged_=False``
    when EM hits ``max_iter`` before the lower-bound delta drops below ``tol``.
    """
    rng = np.random.default_rng(0)
    # Two well-separated Gaussians: a reasonable EM target that still needs
    # more than one iteration to drive the lower-bound delta below 1e-20.
    X = jnp.asarray(
        np.vstack(
            [
                rng.normal(loc=[0.0, 0.0], size=(100, 2)),
                rng.normal(loc=[5.0, 5.0], size=(100, 2)),
            ]
        ).astype(np.float32)
    )
    key = jax.random.PRNGKey(0)

    model = GaussianMixtureModel(n_components=3, max_iter=1, tol=1e-20, random_state=0)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(X, key)

    assert model.converged_ is False
    user_warnings = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning) and "did not converge" in str(w.message)
    ]
    assert user_warnings, (
        f"Expected a UserWarning about non-convergence, got: "
        f"{[str(w.message) for w in recorded]}"
    )


def test_gmm_converged_true_on_well_separated_data():
    """A fit that reaches tolerance well before ``max_iter`` sets
    ``converged_=True`` and emits no non-convergence ``UserWarning``."""
    rng = np.random.default_rng(0)
    X = jnp.asarray(
        np.vstack(
            [
                rng.normal(loc=[0.0, 0.0], scale=0.1, size=(200, 2)),
                rng.normal(loc=[10.0, 10.0], scale=0.1, size=(200, 2)),
            ]
        ).astype(np.float32)
    )
    key = jax.random.PRNGKey(0)

    model = GaussianMixtureModel(
        n_components=2, max_iter=200, tol=1e-3, n_init=1, random_state=0
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(X, key)

    assert model.converged_ is True
    assert model.n_iter_ < model.max_iter
    non_conv = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning) and "did not converge" in str(w.message)
    ]
    assert non_conv == [], (
        f"Unexpected non-convergence warning on a converging fit: "
        f"{[str(w.message) for w in non_conv]}"
    )


def test_gmm_converged_true_at_max_iter_boundary():
    """``converged_`` must be True when the lower-bound delta drops below
    ``tol`` on the same iteration that hits ``max_iter``.

    The lower-bound delta is only finite from the second iteration onward
    (the first compares against an initial ``-inf``). Initializing at the
    solution makes the delta negligible by iteration 2, so with ``max_iter=2``
    the loop stops with ``final_i == max_iter`` AND ``delta <= tol``. The old
    ``n_iter < max_iter`` heuristic mislabeled this as non-converged.
    """
    rng = np.random.default_rng(0)
    c0 = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(200, 2))
    c1 = rng.normal(loc=[10.0, 10.0], scale=0.1, size=(200, 2))
    X = jnp.asarray(np.vstack([c0, c1]).astype(np.float32))
    key = jax.random.PRNGKey(0)

    # Initialize at the (well-separated) cluster solution -- means, weights AND
    # covariances -- so the first M-step is essentially a no-op and the
    # lower-bound delta is negligible by the second iteration.
    means_init = jnp.asarray(
        np.vstack([c0.mean(axis=0), c1.mean(axis=0)]).astype(np.float32)
    )
    covariances_init = jnp.asarray(
        np.stack([np.cov(c0, rowvar=False), np.cov(c1, rowvar=False)]).astype(
            np.float32
        )
    )
    model = GaussianMixtureModel(
        n_components=2,
        max_iter=2,
        tol=1e-2,
        n_init=1,
        means_init=means_init,
        weights_init=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        covariances_init=covariances_init,
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model.fit(X, key)

    assert model.n_iter_ == 2, (
        "Test premise: EM should run exactly max_iter=2 iterations to exercise "
        f"the boundary; got n_iter_={model.n_iter_}"
    )
    assert model.converged_ is True, (
        "EM reached tol on the final allowed iteration but converged_ was False "
        "(max_iter off-by-one)"
    )
    non_conv = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning) and "did not converge" in str(w.message)
    ]
    assert non_conv == [], (
        f"Unexpected non-convergence warning at the converged boundary: "
        f"{[str(w.message) for w in non_conv]}"
    )
