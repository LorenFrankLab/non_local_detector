"""Unit tests for the JAX ``GaussianMixtureModel`` EM implementation.

These cover behavioral guarantees beyond the KDE-vs-GMM convergence comparisons
in ``test_gmm_kde_convergence.py``: stale-state handling on refit, partial
user-init behavior, multi-restart initialization, input validation, and
singular-covariance error reporting.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.gmm import GaussianMixtureModel


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _three_clusters(seed: int = 42, n: int = 200, scale: float = 0.4) -> jnp.ndarray:
    """Three well-separated 2-D Gaussian clusters."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        np.vstack(
            [
                rng.normal([0.0, 0.0], scale, (n, 2)),
                rng.normal([6.0, 6.0], scale, (n, 2)),
                rng.normal([0.0, 7.0], scale, (n, 2)),
            ]
        ).astype(np.float64)
    )


def _two_clusters(seed: int = 1, n: int = 80, scale: float = 0.3) -> jnp.ndarray:
    """Two well-separated 2-D Gaussian clusters."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        np.vstack(
            [
                rng.normal([0.0, 0.0], scale, (n, 2)),
                rng.normal([6.0, 6.0], scale, (n, 2)),
            ]
        ).astype(np.float64)
    )


# ---------------------------------------------------------------------
# Finding #5: refit must not retain stale state
# ---------------------------------------------------------------------
def test_refit_after_total_failure_does_not_retain_stale_fit(key):
    """A refit whose every restart fails must raise, not silently keep the
    previous fit's parameters.

    Regression: ``fit`` reset only ``converged_``/``n_iter_`` but not
    ``weights_``/``means_``/``covariances_``. After a successful fit, a second
    fit that fully fails left the stale (now-misleading) parameters in place,
    the ``weights_ is None`` guard never fired, and the model reported
    ``converged_=False, n_iter_=0`` while still holding the old fit.
    """
    good = _two_clusters()
    # Identical points -> singular empirical covariance with reg_covar=0, so
    # every EM restart yields a NaN lower bound and no restart wins.
    degenerate = jnp.ones((60, 2), dtype=good.dtype)

    model = GaussianMixtureModel(n_components=2, reg_covar=0.0, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(good, key)
    assert model.weights_ is not None  # first fit succeeded

    with pytest.raises(RuntimeError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(degenerate, key)
    # The failed refit must not leave the previous fit's parameters in place.
    assert model.weights_ is None


# ---------------------------------------------------------------------
# Finding #2: partial user init must not collapse to a degenerate solution
# ---------------------------------------------------------------------
def test_partial_user_init_only_weights_does_not_collapse(key):
    """Providing only ``weights_init`` must not collapse every component onto
    the global mean.

    Regression: any user init sent the model down a branch that filled the
    *missing* means with ``n_components`` copies of the global mean and
    identical covariances. EM cannot break that symmetry, so it converged
    instantly to a 1-component solution. Missing pieces should instead come
    from the responsibility-based init (sklearn parity).
    """
    X = _three_clusters()
    true_centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]])

    model = GaussianMixtureModel(
        n_components=3,
        covariance_type="full",
        weights_init=jnp.asarray([0.2, 0.3, 0.5]),
        random_state=0,
        max_iter=100,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, key)
    means = np.asarray(model.means_)

    # The bug produced three identical rows (all == global mean).
    assert not np.allclose(means, means[0], atol=1e-2), (
        f"Components collapsed to identical means:\n{means}"
    )
    # Each true cluster center should be recovered by some fitted mean.
    for c in true_centers:
        assert np.min(np.linalg.norm(means - c, axis=1)) < 1.0, (
            f"No fitted mean near true center {c}; means=\n{means}"
        )


def test_partial_user_init_only_means_are_respected(key):
    """User-provided ``means_init`` must be used verbatim as the initial means
    while the missing weights/covariances come from the resp-based init."""
    X = _three_clusters()
    means_init = jnp.asarray([[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]])
    model = GaussianMixtureModel(
        n_components=3, covariance_type="full", means_init=means_init, random_state=0
    )
    params = model._initialize_parameters(X, key)
    init_weights, init_means, init_cov = params
    # Provided means used as-is at initialization.
    assert np.allclose(np.asarray(init_means), np.asarray(means_init))
    # Missing weights default to the resp-based estimate (a valid simplex).
    assert np.isclose(float(np.asarray(init_weights).sum()), 1.0, atol=1e-6)
    # Missing covariances come from the resp-based init (real cluster spread,
    # scale ~0.4**2), not the old tiled-identity default.
    assert init_cov.shape == (3, 2, 2)
    identity_stack = np.broadcast_to(np.eye(2), (3, 2, 2))
    assert not np.allclose(np.asarray(init_cov), identity_stack, atol=1e-2), (
        "Missing covariances were filled with identity instead of the "
        "responsibility-based cluster covariances"
    )
