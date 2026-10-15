"""A fitted joint GMM and its ground process describe the same spike rate."""

import pickle

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_gmm
from non_local_detector.likelihoods.gmm import (
    GaussianMixtureModel,
    _compute_precision_cholesky,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", params=["full", "tied", "diag", "spherical"])
def encoding(request):
    return _fit_encoding(request.param)


def _fit_encoding(covariance_type, **kwargs):
    rng = np.random.default_rng(76)
    time = np.arange(200) / 20.0
    position = np.linspace(-4.0, 4.0, len(time))[:, None]
    indices = np.r_[np.arange(10, 60), np.arange(140, 190)]
    marks = 2 * position[indices] + rng.normal(0, 0.3, (len(indices), 1))
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False
    )
    model = clusterless_gmm.fit_clusterless_gmm_encoding_model(
        time,
        position,
        [time[indices]],
        [marks],
        env,
        weights=np.linspace(0.2, 1.0, len(time)),
        gmm_components_occupancy=2,
        gmm_components_joint=2,
        gmm_covariance_type_joint=covariance_type,
        gmm_reg_covar=0.1,
        disable_progress_bar=True,
        **kwargs,
    )
    return model


def _position_log_density(joint, points):
    """Independent analytic marginal, using covariance blocks, not precisions."""
    p = points.shape[1]
    covariance = np.asarray(joint.covariances_)
    k, d = joint.means_.shape
    if joint.covariance_type == "tied":
        covariance = np.repeat(covariance[None], k, axis=0)
    elif joint.covariance_type == "diag":
        covariance = np.array([np.diag(row) for row in covariance])
    elif joint.covariance_type == "spherical":
        covariance = covariance[:, None, None] * np.eye(d)
    return logsumexp(
        [
            np.log(w) + multivariate_normal.logpdf(points, mean[:p], cov[:p, :p])
            for w, mean, cov in zip(
                joint.weights_, joint.means_, covariance, strict=True
            )
        ],
        axis=0,
    )


def test_fitted_ground_uses_joint_position_marginal(encoding):
    bins = np.asarray(encoding["interior_place_bin_centers"])
    joint = encoding["joint_models"][0]
    expected_log = _position_log_density(joint, bins)
    np.testing.assert_allclose(
        encoding["gpi_models"][0].score_samples(bins),
        expected_log,
        rtol=2e-5,
        atol=2e-5,
    )
    expected_rate = float(encoding["mean_rates"][0]) * np.exp(
        expected_log - np.asarray(encoding["log_occupancy"])
    )
    np.testing.assert_allclose(
        encoding["summed_ground_process_intensity"],
        expected_rate,
        rtol=3e-5,
    )
    assert encoding["gmm_effective_components"]["gpi"] == [joint.n_components]


@pytest.mark.parametrize("is_local", [False, True])
def test_public_event_intensity_integrates_to_compensator(encoding, is_local):
    marks = np.linspace(-20.0, 20.0, 801)
    edges = np.arange(len(marks) + 1) * 0.01
    position = np.full((2, 1), 2.5)
    kwargs = dict(
        position_time=np.array([edges[0], edges[-1]]),
        position=position,
        time_edges=edges,
        is_local=is_local,
        block_size=128,
        bin_tile_size=3,
        **encoding,
    )
    empty = clusterless_gmm.predict_clusterless_gmm_log_likelihood(
        spike_times=[np.empty(0)],
        spike_waveform_features=[np.empty((0, 1))],
        **kwargs,
    )
    events = clusterless_gmm.predict_clusterless_gmm_log_likelihood(
        spike_times=[edges[:-1] + 0.005],
        spike_waveform_features=[marks[:, None]],
        **kwargs,
    )
    intensity = np.exp(np.asarray(events) - np.asarray(empty)) / np.diff(edges)[:, None]
    integrated = trapezoid(intensity, marks, axis=0)
    ground = -np.asarray(empty)[0] / np.diff(edges)[0]
    np.testing.assert_allclose(integrated, ground, rtol=1e-4, atol=1e-5)


def test_marginal_survives_pickle(encoding):
    restored = pickle.loads(pickle.dumps(encoding))
    bins = encoding["interior_place_bin_centers"]
    np.testing.assert_array_equal(
        restored["gpi_models"][0].score_samples(bins),
        encoding["gpi_models"][0].score_samples(bins),
    )


@pytest.mark.parametrize("kind", ["full", "tied", "diag", "spherical"])
def test_two_position_dimensions_use_covariance_marginal(kind):
    rng = np.random.default_rng(763)
    joint = GaussianMixtureModel(n_components=3, covariance_type=kind)
    joint.means_ = rng.normal(size=(3, 4))
    joint.weights_ = np.array([0.1, 0.3, 0.6])
    factors = rng.normal(size=(3, 4, 4))
    covariance = factors @ factors.transpose(0, 2, 1) + np.eye(4)
    joint.covariances_ = {
        "full": covariance,
        "tied": covariance[0],
        "diag": np.diagonal(covariance, axis1=1, axis2=2),
        "spherical": np.array([0.5, 1.0, 2.0]),
    }[kind]
    joint.precisions_chol_ = _compute_precision_cholesky(joint.covariances_, kind)
    points = rng.normal(size=(7, 2))
    marginal = clusterless_gmm._position_marginal(joint, 2)
    np.testing.assert_allclose(
        marginal.score_samples(points),
        _position_log_density(joint, points),
        rtol=2e-5,
        atol=2e-5,
    )


def test_fit_only_optimizes_occupancy_and_joint(monkeypatch):
    fitted_dims = []
    original = clusterless_gmm._fit_gmm_density

    def record_fit(*args, **kwargs):
        fitted_dims.append(kwargs["X"].shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(clusterless_gmm, "_fit_gmm_density", record_fit)
    _fit_encoding("full")
    assert fitted_dims == [1, 2]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("gmm_components_gpi", 1), ("gmm_covariance_type_gpi", "full")],
)
def test_removed_gpi_settings_are_rejected(parameter, value):
    with pytest.raises(TypeError, match=parameter):
        _fit_encoding("full", **{parameter: value})
