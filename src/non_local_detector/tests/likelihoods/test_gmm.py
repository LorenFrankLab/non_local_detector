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

from non_local_detector.exceptions import ValidationError
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


# ---------------------------------------------------------------------
# Finding #1: kmeans n_init restarts must actually differ
# ---------------------------------------------------------------------
def test_kmeans_restarts_differ_across_inits(key):
    """With ``init_params='kmeans'`` and a fixed ``random_state``, restarts
    beyond the first must use distinct KMeans seeds.

    Regression: ``_initialize_kmeans_resp`` ignored the per-restart key and
    seeded every KMeans with ``self.random_state``, so all ``n_init`` restarts
    produced byte-identical responsibilities and ``n_init`` did nothing for the
    default ``kmeans`` init.
    """
    X = _three_clusters()
    model = GaussianMixtureModel(
        n_components=3, init_params="kmeans", n_init=5, random_state=0
    )
    init_keys = jax.random.split(key, model.n_init)
    resps = [
        np.asarray(model._initialize_kmeans_resp(X, init_keys[i], init_index=i))
        for i in range(model.n_init)
    ]
    assert any(
        not np.array_equal(resps[0], resps[i]) for i in range(1, model.n_init)
    ), "All kmeans restarts produced identical responsibilities"


def test_kmeans_first_restart_preserves_random_state(key):
    """The primary restart (index 0) with a set ``random_state`` must keep
    using that seed directly, so single-init fits stay byte-identical to the
    pre-fix behavior (and to a direct ``KMeans(random_state=...)``)."""
    from sklearn.cluster import KMeans

    X = _three_clusters()
    model = GaussianMixtureModel(
        n_components=3, init_params="kmeans", n_init=5, random_state=0
    )
    init_keys = jax.random.split(key, model.n_init)
    resp0 = np.asarray(model._initialize_kmeans_resp(X, init_keys[0], init_index=0))

    km = KMeans(n_clusters=3, init="k-means++", n_init=1, random_state=0).fit(
        np.asarray(X)
    )
    ref = np.zeros((X.shape[0], 3), dtype=resp0.dtype)
    ref[np.arange(X.shape[0]), km.labels_] = 1.0
    assert np.array_equal(resp0, ref)


# ---------------------------------------------------------------------
# Finding #3: validate init arrays and fit inputs
# ---------------------------------------------------------------------
def test_weights_init_wrong_length_raises():
    """``weights_init`` whose length != n_components is rejected at
    construction with a ``ValidationError`` (not a deep broadcasting error)."""
    with pytest.raises(ValidationError):
        GaussianMixtureModel(n_components=3, weights_init=jnp.asarray([0.5, 0.5]))


def test_weights_init_not_summing_to_one_raises():
    """``weights_init`` must sum to 1."""
    with pytest.raises(ValidationError):
        GaussianMixtureModel(n_components=3, weights_init=jnp.asarray([0.2, 0.2, 0.2]))


def test_weights_init_negative_entry_raises():
    """A negative ``weights_init`` entry that still sums to 1 must be rejected
    at construction.

    Otherwise ``jnp.log(weights)`` produces NaN deep in EM, which ``fit`` would
    misreport as a singular covariance (wrong remedy for the user).
    """
    with pytest.raises(ValidationError):
        GaussianMixtureModel(n_components=2, weights_init=jnp.asarray([-0.2, 1.2]))


def test_nonfinite_X_raises_validation_error(key):
    """Non-finite ``X`` is rejected with a clear ``ValidationError`` rather than
    an opaque sklearn KMeans error or a misleading singular-covariance error."""
    X = np.array(_two_clusters())  # writable copy
    X[0, 0] = np.nan
    model = GaussianMixtureModel(n_components=2, random_state=0)
    with pytest.raises(ValidationError):
        model.fit(jnp.asarray(X), key)


def test_means_init_wrong_n_components_raises():
    """``means_init`` with the wrong leading dimension is rejected at
    construction."""
    with pytest.raises(ValidationError):
        GaussianMixtureModel(n_components=3, means_init=jnp.zeros((2, 4)))


def test_means_init_wrong_n_features_raises(key):
    """``means_init`` whose feature dimension disagrees with X is rejected at
    fit time."""
    X = _three_clusters()  # 2 features
    model = GaussianMixtureModel(n_components=3, means_init=jnp.zeros((3, 5)))
    with pytest.raises(ValidationError):
        model.fit(X, key)


def test_sample_weight_negative_raises(key):
    """Negative ``sample_weight`` entries are rejected."""
    X = _two_clusters()
    sw = np.ones(X.shape[0])
    sw[0] = -1.0
    model = GaussianMixtureModel(n_components=2, random_state=0)
    with pytest.raises(ValidationError):
        model.fit(X, key, sample_weight=jnp.asarray(sw))


def test_sample_weight_wrong_length_raises(key):
    """``sample_weight`` whose length != n_samples is rejected."""
    X = _two_clusters()
    model = GaussianMixtureModel(n_components=2, random_state=0)
    with pytest.raises(ValidationError):
        model.fit(X, key, sample_weight=jnp.ones(X.shape[0] - 1))


def test_covariances_init_wrong_shape_raises(key):
    """``covariances_init`` inconsistent with ``covariance_type`` is rejected
    at fit time (full expects (n_components, n_features, n_features))."""
    X = _three_clusters()  # 2 features
    model = GaussianMixtureModel(
        n_components=3, covariance_type="full", covariances_init=jnp.zeros((3, 5, 5))
    )
    with pytest.raises(ValidationError):
        model.fit(X, key)


# ---------------------------------------------------------------------
# Finding #4: singular-covariance failures get an actionable error
# ---------------------------------------------------------------------
def test_singular_covariance_raises_actionable_error(key):
    """When every restart collapses to a singular covariance (NaN lower
    bound), ``fit`` raises a ``RuntimeError`` that names the cause and the
    ``reg_covar`` remedy instead of a bare ``"Fitting failed."``.
    """
    degenerate = jnp.ones((60, 2), dtype=jnp.float32)
    model = GaussianMixtureModel(n_components=2, reg_covar=0.0, random_state=0)
    with pytest.raises(RuntimeError, match="reg_covar"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(degenerate, key)
