"""Independent references for Gaussian scoring, centered covariances, and weighted fits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.gmm import (
    GaussianMixtureModel,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_spherical,
    _estimate_gaussian_covariances_tied,
    _estimate_log_gaussian_prob,
)

pytestmark = pytest.mark.unit


def _data() -> tuple[jnp.ndarray, np.ndarray]:
    """Two well-separated float32 clusters; every 13th sample weight is zeroed."""
    rng = np.random.default_rng(41)
    X = np.r_[rng.normal(5, 1, (60, 2)), rng.normal(25, 1, (60, 2))].astype(np.float32)
    weights = rng.uniform(0.5, 2.0, len(X)).astype(np.float32)
    weights[::13] = 0
    return jnp.asarray(X), weights


def _fit(
    X: jnp.ndarray,
    weights: np.ndarray | jnp.ndarray,
    covariance_type: str = "full",
    **kwargs,
) -> GaussianMixtureModel:
    """Fit a two-component GMM on ``X`` with the given weights, seeded for reuse."""
    return GaussianMixtureModel(
        n_components=2, covariance_type=covariance_type, random_state=4, **kwargs
    ).fit(X, jax.random.PRNGKey(8), sample_weight=weights)


def _reference_log_gaussian_prob(
    X: np.ndarray, means: np.ndarray, precision: np.ndarray, covariance_type: str
) -> np.ndarray:
    """Float64 reference log N(x | mu, P) for diagonal and spherical precisions."""
    n_features = means.shape[1]
    diagonal = np.asarray(precision, float)
    if covariance_type != "diag":
        diagonal = np.repeat(diagonal[:, None], n_features, axis=1)
    squared = (
        (np.asarray(X, float)[:, None] - np.asarray(means, float)) * diagonal
    ) ** 2
    log_det = np.log(diagonal).sum(axis=1)
    return -0.5 * (n_features * np.log(2 * np.pi) + squared.sum(axis=2)) + log_det


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
@pytest.mark.parametrize("n_features", [1, 7, 8, 9, 17, 33])
def test_gaussian_feature_accumulation_matches_reference(covariance_type, n_features):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(11, n_features)).astype(np.float32)
    means = rng.normal(size=(5, n_features)).astype(np.float32)
    shape = means.shape if covariance_type == "diag" else (len(means),)
    precision = rng.uniform(0.5, 1.5, size=shape).astype(np.float32)
    expected = _reference_log_gaussian_prob(X, means, precision, covariance_type)
    actual = _estimate_log_gaussian_prob(X, means, precision, covariance_type)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
@pytest.mark.parametrize("n_features", [1, 3])
def test_gaussian_evaluation_rejects_mismatched_feature_dimensions(
    covariance_type, n_features
):
    precision = jnp.ones((2, 2)) if covariance_type == "diag" else jnp.ones(2)
    with pytest.raises(ValueError, match="feature"):
        _estimate_log_gaussian_prob(
            jnp.zeros((2, n_features)), jnp.ones((2, 2)), precision, covariance_type
        )


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
def test_gaussian_evaluation_rejects_mismatched_precision_shape(covariance_type):
    precision = jnp.ones((2, 1)) if covariance_type == "diag" else jnp.ones(1)
    with pytest.raises(ValueError, match="precisions"):
        _estimate_log_gaussian_prob(
            jnp.zeros((2, 2)), jnp.ones((2, 2)), precision, covariance_type
        )


@pytest.mark.parametrize("covariance_type", ["diag", "spherical", "tied"])
def test_covariance_updates_are_exact_for_tight_distant_components(covariance_type):
    """Two components at +-1e3 with standard deviation 0.1. Centering on one
    shared point (e.g. the global mean) still leaves |mu - c| / sigma = 1e4,
    which cancels every float32 digit of the variance; only centering each
    component on its own mean recovers it."""
    rng = np.random.default_rng(5)
    X = np.concatenate(
        [rng.normal(-1e3, 0.1, (200, 3)), rng.normal(1e3, 0.1, (200, 3))]
    ).astype(np.float32)
    resp = np.zeros((400, 2), np.float32)
    resp[:200, 0] = resp[200:, 1] = 1.0
    nk = resp.sum(axis=0)
    means = (resp.T @ X) / nk[:, None]
    X64, means64 = X.astype(float), means.astype(float)
    per_component = np.stack(
        [((X64[resp[:, k] > 0] - means64[k]) ** 2).mean(axis=0) for k in range(2)]
    )
    expected = {
        "diag": per_component,
        "spherical": per_component.mean(axis=1),
        "tied": sum(
            (X64[resp[:, k] > 0] - means64[k]).T @ (X64[resp[:, k] > 0] - means64[k])
            for k in range(2)
        )
        / 400,
    }[covariance_type]
    function = {
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
        "tied": _estimate_gaussian_covariances_tied,
    }[covariance_type]
    actual = function(
        jnp.asarray(resp), jnp.asarray(X), jnp.asarray(nk), jnp.asarray(means), 0.0
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("covariance_type", ["diag", "spherical", "tied"])
@pytest.mark.parametrize("offset", [0.0, 1e5])
def test_covariance_updates_match_centered_reference(covariance_type, offset):
    X = jnp.array(
        [[offset - 1, offset + 2], [offset + 1, offset - 2]], dtype=jnp.float32
    )
    means = jnp.full((1, 2), offset, dtype=jnp.float32)
    resp = jnp.ones((2, 1), dtype=jnp.float32)
    nk = jnp.array([2.0], dtype=jnp.float32)
    functions = {
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
        "tied": _estimate_gaussian_covariances_tied,
    }
    expected = {
        "diag": [[1.01, 4.01]],
        "spherical": [2.51],
        "tied": [[1.01, -2], [-2, 4.01]],
    }
    actual = functions[covariance_type](resp, X, nk, means, 0.01)
    np.testing.assert_allclose(actual, expected[covariance_type], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
@pytest.mark.parametrize("offset", [0.0, 1e2, 1e3, 1e4, 1e5])
def test_gaussian_scores_match_centered_float64_reference(covariance_type, offset):
    X = np.array([[offset + 1, offset], [offset - 1, offset + 2]], np.float32)
    means = np.array([[offset, offset], [offset + 2, offset - 1]], np.float32)
    precision = np.array([[1.0, 0.5], [0.5, 2.0]], np.float32)
    if covariance_type == "spherical":
        precision = np.array([1.0, 0.5], np.float32)
    diagonal = (
        precision
        if covariance_type == "diag"
        else np.repeat(precision[:, None], 2, axis=1)
    )
    expected = _reference_log_gaussian_prob(X, means, precision, covariance_type)
    actual = _estimate_log_gaussian_prob(
        jnp.asarray(X), jnp.asarray(means), jnp.asarray(precision), covariance_type
    )
    full = _estimate_log_gaussian_prob(
        jnp.asarray(X),
        jnp.asarray(means),
        jnp.asarray([np.diag(p) for p in diagonal]),
        "full",
    )
    tolerance = 1e-5 if offset == 0 else 1e-4
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-6)
    np.testing.assert_allclose(actual, full, rtol=tolerance, atol=1e-6)
    recovered_distance = -2 * (
        np.asarray(actual) - np.log(diagonal).sum(axis=1)
    ) - 2 * np.log(2 * np.pi)
    assert np.all(recovered_distance >= 0)


# (covariance_type, weight scale, weight dtype): float32 weights cover the
# subnormal end of the representable range, float64 weights the range that only
# survives if normalization happens on the host before the JAX conversion.
_WEIGHT_RESCALINGS = [
    ("full", 1e-40, np.float32),
    ("full", 1e-9, np.float32),
    ("full", 1e35, np.float32),
    ("full", 1e-200, np.float64),
    ("full", 1e200, np.float64),
    ("tied", 1e-37, np.float32),
    ("tied", 1e37, np.float32),
    ("diag", 1e-37, np.float32),
    ("diag", 1e37, np.float32),
    ("spherical", 1e-37, np.float32),
    ("spherical", 1e37, np.float32),
]


@pytest.mark.parametrize(("covariance_type", "scale", "dtype"), _WEIGHT_RESCALINGS)
def test_weight_rescaling_preserves_the_whole_estimator(covariance_type, scale, dtype):
    """Only the ratios between weights may matter, over the representable range."""
    X, weights = _data()
    baseline = _fit(X, weights, covariance_type)
    scaled_weights = weights.astype(dtype) * dtype(scale)
    original = scaled_weights.copy()
    scaled = _fit(X, scaled_weights, covariance_type)
    for name in ("means_", "covariances_", "weights_", "lower_bound_"):
        np.testing.assert_allclose(
            getattr(scaled, name),
            getattr(baseline, name),
            rtol=1e-4,
            atol=1e-6,
            err_msg=name,
        )
    np.testing.assert_allclose(
        scaled.score_samples(X), baseline.score_samples(X), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(np.sum(scaled.weights_), 1.0, rtol=0, atol=1e-6)
    # Normalization happens on a copy: the caller's array is never rescaled.
    np.testing.assert_array_equal(scaled_weights, original)


@pytest.mark.parametrize("init_params", ["kmeans", "random"])
def test_zero_weight_observations_cannot_influence_fit(init_params):
    X, weights = _data()
    positive = weights > 0
    # Extreme zero-weight rows must not seed centers or change random draws.
    with_zeros = jnp.concatenate([X, jnp.full((17, 2), 10000.0)])
    padded_weights = np.r_[weights, np.zeros(17, np.float32)]
    reference = _fit(X[positive], weights[positive], init_params=init_params)
    actual = _fit(with_zeros, padded_weights, init_params=init_params)
    for name in ("means_", "covariances_", "weights_", "lower_bound_"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(reference, name), rtol=1e-5, atol=1e-6
        )


def test_nonfinite_values_in_zero_weight_rows_are_accepted():
    """``fit`` requires finite features only on the rows it actually trains on.

    The zero-weight rows are not part of the fitted data, so a NaN there must
    neither reject the call nor change the fit.
    """
    X, weights = _data()
    positive = weights > 0
    contaminated = np.asarray(X).copy()
    contaminated[~positive] = np.nan
    reference = _fit(X[positive], weights[positive])
    actual = _fit(jnp.asarray(contaminated), weights)
    for name in ("means_", "covariances_", "weights_", "lower_bound_"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(reference, name), rtol=1e-6, atol=1e-7
        )


_GUARD_TEST_X = jnp.asarray(
    np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [9.0, 9.0]], np.float32)
)


def _fit_guard_test(n_components, sample_weight):
    return GaussianMixtureModel(n_components=n_components, random_state=0).fit(
        _GUARD_TEST_X, jax.random.PRNGKey(0), sample_weight=sample_weight
    )


# 1e-60 underflows float32 outright; 1e-30 and 1e-10 are representable but
# weigh less than the 10 * eps (~1.2e-6) empty-component guard.
@pytest.mark.parametrize("vanishing_weight", [1e-60, 1e-30, 1e-10])
def test_weights_below_the_count_guard_are_treated_as_zero(vanishing_weight):
    """A weight below the empty-component guard must not reserve a component.

    Counting such a row as data seeded a component whose count is dominated by
    the ``10 * eps`` guard, leaving a phantom component near the origin
    (mixture weight ~1e-6, covariance ``reg_covar * I``).
    """
    fit = _fit_guard_test
    flushed = np.array([1.0, 1.0, 1.0, vanishing_weight])
    actual = fit(3, flushed)
    dropped = fit(3, np.array([1.0, 1.0, 1.0, 0.0]))
    for name in ("means_", "covariances_", "weights_", "lower_bound_"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(dropped, name), rtol=1e-6, atol=1e-7
        )
    assert np.min(np.asarray(actual.weights_)) > 1e-3  # no guard-level component
    # The count guard sees the same rows as EM: a component reserved for the
    # flushed row is rejected rather than silently left empty.
    with pytest.raises(ValidationError, match="usable positive"):
        fit(4, flushed)


def test_small_weight_above_the_count_guard_keeps_its_component():
    """The cutoff drops only guard-level weights: a small weight well above it
    is still data and seeds a component at its own sample, not at the origin."""
    model = _fit_guard_test(4, np.array([1.0, 1.0, 1.0, 1e-4]))
    lightest = int(np.argmin(np.asarray(model.weights_)))
    np.testing.assert_allclose(model.means_[lightest], [9.0, 9.0], rtol=2e-2)


def test_full_warm_start_allows_fewer_samples_than_components():
    """Supplying every init array skips data-driven initialization, so EM may
    start from those parameters on a batch smaller than ``n_components``."""
    n_components = 5
    X = jnp.asarray(np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]], np.float32))
    model = GaussianMixtureModel(
        n_components=n_components,
        random_state=0,
        weights_init=np.full(n_components, 1.0 / n_components, np.float32),
        means_init=np.arange(2 * n_components, dtype=np.float32).reshape(-1, 2),
        covariances_init=np.broadcast_to(np.eye(2), (n_components, 2, 2)).astype(
            np.float32
        ),
    ).fit(X, jax.random.PRNGKey(0))
    assert np.isfinite(model.lower_bound_)
    np.testing.assert_allclose(np.sum(model.weights_), 1.0, atol=1e-6)


def test_kmeans_initialization_honors_relative_sample_weights():
    X = np.arange(12, dtype=np.float32)[:, None]
    weights = np.array([1.0] * 10 + [100.0] * 2, np.float32)
    model = GaussianMixtureModel(n_components=2, random_state=4)
    _, means, _ = model._initialize_parameters(
        jnp.asarray(X), jax.random.PRNGKey(8), sample_weight=jnp.asarray(weights)
    )
    labels = (
        KMeans(n_clusters=2, n_init=1, random_state=4)
        .fit(X, sample_weight=weights)
        .labels_
    )
    expected = np.array(
        [np.average(X[labels == k, 0], weights=weights[labels == k]) for k in range(2)]
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(means).ravel()), np.sort(expected), rtol=1e-5
    )


def test_weighted_objective_and_restart_selection_use_pre_m_step_parameters(
    monkeypatch,
):
    X = jnp.array([[-2.0], [-1.0], [0.0], [1.0], [3.0]])
    weights = jnp.array([0.01, 0.2, 5.0, 1.0, 20.0])
    model = GaussianMixtureModel(n_components=2, n_init=2, max_iter=1)
    initial_means = [np.array([-1.0, 0.0]), np.array([1.0, 3.0])]

    def initialize(X, key, sample_weight=None, init_index=0):
        return (
            jnp.full(2, 0.5),
            jnp.asarray(initial_means[init_index][:, None]),
            jnp.ones((2, 1, 1)),
        )

    monkeypatch.setattr(model, "_initialize_parameters", initialize)
    log_components = np.stack(
        [
            -0.5 * (np.asarray(X, float) - means) ** 2
            - 0.5 * np.log(2 * np.pi)
            + np.log(0.5)
            for means in initial_means
        ]
    )
    logp = logsumexp(log_components, axis=2)
    expected = np.average(logp, axis=1, weights=np.asarray(weights, float))
    winner = np.argmax(expected)
    assert winner != np.argmax(logp.mean(axis=1))
    with pytest.warns(UserWarning, match="did not converge"):
        model.fit(X, jax.random.PRNGKey(4), sample_weight=weights)
    np.testing.assert_allclose(model.lower_bound_, expected.max(), rtol=1e-5)
    responsibilities = np.exp(log_components[winner] - logp[winner, :, None])
    weighted = responsibilities * np.asarray(weights, float)[:, None]
    expected_means = (weighted.T @ np.asarray(X, float)) / weighted.sum(axis=0)[:, None]
    np.testing.assert_allclose(model.means_, expected_means, rtol=1e-5, atol=1e-6)
    assert model.n_iter_ == 1


@pytest.mark.parametrize("init_params", ["kmeans", "random"])
def test_fewer_positive_weights_than_components_is_rejected(init_params):
    """Zero-weight rows are dropped before fitting, so the effective sample count
    must still cover every component; otherwise KMeans raises an opaque sklearn
    error and random initialization silently fits duplicate components."""
    X = jnp.arange(10.0).reshape(5, 2)
    weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    model = GaussianMixtureModel(
        n_components=2, init_params=init_params, random_state=0
    )
    with pytest.raises(ValidationError, match="positive-weight samples"):
        model.fit(X, jax.random.PRNGKey(0), sample_weight=weights)


def test_fewer_samples_than_components_is_rejected():
    with pytest.raises(ValidationError, match="samples"):
        GaussianMixtureModel(n_components=2, random_state=0).fit(
            jnp.zeros((1, 2)), jax.random.PRNGKey(0)
        )


def test_unweighted_mixture_weights_sum_to_one_with_empty_components():
    """Empty components carry the small count guard; normalizing by the sum of the
    guarded counts keeps the total at one (dividing by n_samples left it
    1 + n_components * 10 * eps, 10 float32 ulps for this fit)."""
    rng = np.random.default_rng(3)
    X = jnp.asarray(
        np.r_[rng.normal(5, 0.5, (5, 2)), rng.normal(25, 0.5, (5, 2))].astype(
            np.float32
        )
    )
    n_components, n_empty = 10, 8
    # Two components sit on the data and the rest far outside it. Supplying all
    # three init arrays skips KMeans, so the far components start empty (zero
    # responsibility for every sample) and stay empty: their M-step mean
    # collapses to the origin, which is still remote from both clusters.
    means_init = np.concatenate(
        [np.array([[5.0, 5.0], [25.0, 25.0]]), np.full((n_empty, 2), 1e4)]
    ).astype(np.float32)
    model = GaussianMixtureModel(
        n_components=n_components,
        random_state=4,
        weights_init=np.full(n_components, 1.0 / n_components, np.float32),
        means_init=means_init,
        covariances_init=np.broadcast_to(np.eye(2), (n_components, 2, 2)).astype(
            np.float32
        ),
    ).fit(X, jax.random.PRNGKey(8))

    guard = 10 * np.finfo(np.float32).eps  # the empty-component count guard
    weights = np.asarray(model.weights_, dtype=np.float64)
    # An empty component's count is exactly the guard; the counts sum to
    # n_samples plus one guard per component.
    expected_empty = guard / (X.shape[0] + n_components * guard)
    np.testing.assert_allclose(np.sort(weights)[:n_empty], expected_empty, rtol=1e-3)
    # A few float32 ulps: the weights are rounded float32 divisions by a float32
    # count sum that cannot represent every guard exactly (measured 1.5 ulps
    # here). Dividing the counts by n_samples instead biases the total by
    # n_components * guard / n_samples = 10 ulps, which this tolerance rejects.
    np.testing.assert_allclose(
        np.sum(weights), 1.0, rtol=0, atol=4 * np.finfo(np.float32).eps
    )


@pytest.mark.parametrize(
    "weights", [np.zeros(5), np.full(5, np.nan), np.full(5, np.inf), -np.ones(5)]
)
def test_invalid_or_all_zero_weights_still_raise(weights):
    with pytest.raises(ValidationError, match="sample_weight"):
        _fit(jnp.arange(10.0).reshape(5, 2), weights)
