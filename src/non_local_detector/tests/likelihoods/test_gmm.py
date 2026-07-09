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
