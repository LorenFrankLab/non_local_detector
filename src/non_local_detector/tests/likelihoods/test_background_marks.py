"""Regression tests for a common reference measure for clusterless marks."""

import pickle
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from non_local_detector import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.likelihoods.background_marks import (
    BackgroundMarkModel,
    fit_background_mark_model,
)
from non_local_detector.likelihoods.common import RATE_EPS_HZ
from non_local_detector.models import NonLocalClusterlessDetector


def _data():
    time = np.linspace(0, 1, 51)
    position = np.column_stack((10 * time, 3 * np.sin(2 * np.pi * time)))
    spikes = [np.array([0.12, 0.27, 0.44, 0.61, 0.78, 0.93])]
    marks = [
        np.column_stack((np.arange(6), np.array([1, 3, 2, 5, 4, 7]))).astype(float)
    ]
    return time, position, spikes, marks


def _fit(algorithm, scale=1, shift=0):
    time, position, spikes, marks = _data()
    params = {
        "position_std": 2.0,
        "waveform_std": 2.0 * scale,
        "block_size": 4,
        "disable_progress_bar": True,
    }
    model = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=2.0),
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params=params,
        infer_track_interior=False,
        no_spike_rate=0.5,
    ).fit(time, position, spikes, [scale * marks[0] + shift])
    return model


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm", ["clusterless_kde", "clusterless_kde_log", "clusterless_diffusion"]
)
def test_mark_units_preserve_odds_against_no_spike(algorithm):
    time, position, spikes, marks = _data()
    edges = np.linspace(0, 1, 11)
    first = _fit(algorithm).compute_log_likelihood(
        time, position, spikes, marks, time_edges=edges
    )
    second = _fit(algorithm, 10, 30).compute_log_likelihood(
        time, position, spikes, [10 * marks[0] + 30], time_edges=edges
    )
    # Differences to the same state/bin are log posterior odds under equal priors.
    assert_allclose(
        np.asarray(first) - np.asarray(first)[:, :1],
        np.asarray(second) - np.asarray(second)[:, :1],
        atol=1e-5,
    )


@pytest.mark.integration
def test_background_reference_survives_pickle_and_row_chunks():
    time, position, spikes, marks = _data()
    model = _fit("clusterless_kde_log")
    edges = np.linspace(0, 1, 11)
    full = model.compute_log_likelihood(time, position, spikes, marks, time_edges=edges)
    restored = pickle.loads(pickle.dumps(model))
    assert hasattr(restored, "background_mark_models_")
    chunks = [
        restored.compute_log_likelihood(
            time, position, spikes, marks, time_edges=edges, row_slice=slice(a, b)
        )
        for a, b in [(0, 3), (3, 10)]
    ]
    assert_allclose(np.concatenate(chunks), full, atol=1e-6)


@pytest.mark.unit
def test_reference_normalization_and_unit_jacobian():
    reference = BackgroundMarkModel(np.array([2.0]), np.array([3.0]))
    grid = np.linspace(-25, 29, 10001)[:, None]
    probability = np.trapezoid(
        np.exp(np.asarray(reference.log_density(grid))), grid[:, 0]
    )
    assert_allclose(probability, 1.0, atol=1e-7)
    transformed = BackgroundMarkModel(
        10 * reference.location + 30, 10 * reference.scale
    )
    assert_allclose(
        transformed.log_density(10 * grid + 30),
        reference.log_density(grid) - np.log(10),
        atol=1e-5,
    )


@pytest.mark.unit
def test_reference_excludes_zero_weight_nonfinite_marks():
    model = fit_background_mark_model(
        np.array([[1.0], [3.0], [np.nan]]), np.array([1.0, 3.0, 0.0]), 2.0
    )
    assert_allclose(model.location, [2.5])
    assert_allclose(model.scale, [np.sqrt(0.75)])
    empty = fit_background_mark_model(np.empty((0, 2)), np.empty(0), 2.0)
    assert empty is None


@pytest.mark.integration
def test_reference_training_mask_and_posterior_refit():
    time, position, spikes, marks = _data()
    model = _fit("clusterless_kde")
    training = time < 0.5
    model.fit_encoding_model(time, position, spikes, marks, is_training=training)
    reference = model.background_mark_models_[0]
    # All three permitted marks enter once, without the held-out latter half.
    assert_allclose(reference.location, marks[0][:3].mean(axis=0))
    model.fit_encoding_model(
        time,
        position,
        spikes,
        marks,
        weights=np.linspace(0.1, 1, len(time)),
        _refit_background_marks=False,
    )
    assert model.background_mark_models_[0] is reference
    for encoding in model.encoding_model_.values():
        assert encoding["background_mark_models"] is model.background_mark_models_


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", list(_CLUSTERLESS_ALGORITHMS))
@pytest.mark.parametrize("local", [False, True])
def test_zero_rate_electrode_uses_normalized_reference_and_compensator(
    algorithm, local
):
    time, position, _, _ = _data()
    environment = Environment(place_bin_size=2).fit_place_grid(
        position, infer_track_interior=False
    )
    fit, predict = _CLUSTERLESS_ALGORITHMS[algorithm]
    params = {"disable_progress_bar": True}
    if algorithm == "clusterless_gmm":
        params.update(
            gmm_components_occupancy=2, gmm_components_gpi=2, gmm_components_joint=2
        )
    else:
        params.update(position_std=2.0, waveform_std=2.0, block_size=4)
    encoding = fit(
        time, position, [np.empty(0)], [np.empty((0, 2))], environment, **params
    )
    reference = BackgroundMarkModel(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    encoding["background_mark_models"] = [reference]
    edges = np.array([0.0, 0.5, 1.0])
    marks = [np.array([[2.0, 4.0]])]
    result = predict(
        time,
        position,
        [np.array([0.25])],
        marks,
        time_edges=edges,
        is_local=local,
        **encoding,
    )
    expected = np.array(
        [
            np.log(RATE_EPS_HZ * 0.5)
            + float(reference.log_density(marks[0])[0])
            - 0.5 * RATE_EPS_HZ,
            -0.5 * RATE_EPS_HZ,
        ]
    )
    assert_allclose(
        result, np.broadcast_to(expected[:, None], np.shape(result)), atol=1e-5
    )


@pytest.mark.integration
def test_unseen_electrode_requires_training_marks():
    time, position, spikes, marks = _data()
    model = _fit("clusterless_kde")
    model.fit_encoding_model(time, position, [np.empty(0)], [np.empty((0, 2))])
    empty = model.compute_log_likelihood(
        time,
        position,
        [np.empty(0)],
        [np.empty((0, 2))],
        time_edges=np.array([0.0, 1.0]),
    )
    assert np.isfinite(empty).all()
    with pytest.raises(ValidationError, match="no permitted training marks"):
        model.compute_log_likelihood(
            time, position, spikes, marks, time_edges=np.array([0.0, 1.0])
        )


@pytest.mark.integration
def test_legacy_clusterless_pickle_requires_refit():
    time, position, spikes, marks = _data()
    model = _fit("clusterless_kde")
    del model.background_mark_contract_
    with pytest.raises(ValidationError, match="background mark contract"):
        model.compute_log_likelihood(
            time, position, spikes, marks, time_edges=np.array([0.0, 1.0])
        )


@pytest.mark.integration
@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gmm_mark_affine_units_preserve_state_odds(covariance_type):
    """Transform the same fitted physical model, including its covariances."""
    from non_local_detector.likelihoods.gmm import _compute_precision_cholesky

    time, position, spikes, marks = _data()
    model = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=2),
        infer_track_interior=False,
        no_spike_rate=0.5,
        clusterless_algorithm="clusterless_gmm",
        clusterless_algorithm_params={
            "gmm_components_occupancy": 2,
            "gmm_components_gpi": 2,
            "gmm_components_joint": 2,
            "gmm_covariance_type_joint": covariance_type,
            "disable_progress_bar": True,
        },
    ).fit(time, position, spikes, marks)
    edges = np.linspace(0, 1, 11)
    first = model.compute_log_likelihood(
        time, position, spikes, marks, time_edges=edges
    )
    transformed = deepcopy(model)
    # A spherical component cannot represent independent mark-only scaling.
    # Promote it to diagonal for the identical transformed distribution.
    scale, shift = 10.0, 30.0
    old_ref = transformed.background_mark_models_[0]
    transformed.background_mark_models_[0] = BackgroundMarkModel(
        old_ref.location * scale + shift, old_ref.scale * scale
    )
    for encoding in transformed.encoding_model_.values():
        joint = encoding["joint_models"][0]
        ndims = joint.means_.shape[1]
        factors = np.array([1.0, 1.0, scale, scale])
        means = np.asarray(joint.means_).copy()
        means[:, 2:] = means[:, 2:] * scale + shift
        joint.means_ = means
        covariance = np.asarray(joint.covariances_)
        if covariance_type in ["full", "tied"]:
            joint.covariances_ = covariance * factors[:, None] * factors[None, :]
        elif covariance_type == "diag":
            joint.covariances_ = covariance * factors**2
        else:
            joint.covariance_type = "diag"
            joint.covariances_ = (
                np.broadcast_to(covariance[:, None], (len(covariance), ndims))
                * factors**2
            )
        joint.precisions_chol_ = _compute_precision_cholesky(
            joint.covariances_, joint.covariance_type
        )
    second = transformed.compute_log_likelihood(
        time, position, spikes, [scale * marks[0] + shift], time_edges=edges
    )
    assert_allclose(
        np.asarray(first) - np.asarray(first)[:, :1],
        np.asarray(second) - np.asarray(second)[:, :1],
        atol=1e-5,
    )


@pytest.mark.unit
def test_positive_gemm_sum_recovers_many_subnormal_contributions():
    """A nonzero GEMM sum can still miss enough small terms to bias evidence."""
    from scipy.special import logsumexp

    from non_local_detector.likelihoods.background_marks import _kde_log_joint_nonlocal

    a = 90.0
    positions = np.array([[0.0]], dtype=np.float32)
    marks = np.array([[0.0]], dtype=np.float32)
    encoded_positions = np.array(
        [0.0, np.sqrt(2 * a), *([np.sqrt(a)] * 256), 0.0], dtype=np.float32
    )[:, None]
    encoded_marks = np.array(
        [np.sqrt(2 * a), 0.0, *([np.sqrt(a)] * 256), np.sqrt(2 * (a - 8))],
        dtype=np.float32,
    )[:, None]
    weights = np.ones(len(encoded_marks), dtype=np.float32)
    actual = _kde_log_joint_nonlocal(
        positions,
        marks,
        encoded_positions,
        encoded_marks,
        np.array([1.0]),
        np.array([1.0]),
        weights,
        None,
    )
    expected = (
        logsumexp(
            -0.5
            * (
                encoded_positions[:, 0].astype(float) ** 2
                + encoded_marks[:, 0].astype(float) ** 2
            )
        )
        - np.log(2 * np.pi)
        - np.log(len(weights))
    )
    assert_allclose(actual, expected, atol=1e-5)
    batch = _kde_log_joint_nonlocal(
        positions,
        np.array([[0.0], [np.sqrt(2 * a)]]),
        encoded_positions,
        encoded_marks,
        np.array([1.0]),
        np.array([1.0]),
        weights,
        64,
    )
    assert_allclose(batch[0], actual[0], atol=1e-5)


@pytest.mark.integration
def test_empty_electrode_borrows_only_matching_training_marks():
    time, position, spikes, marks = _data()
    model = _fit("clusterless_kde")
    model.fit_encoding_model(
        time,
        position,
        [*spikes, np.empty(0)],
        [*marks, np.empty((0, 2))],
        is_training=time < 0.5,
    )
    assert model.background_mark_sources_[1] == {
        "kind": "pooled_same_dimension",
        "donor_electrodes": (0,),
    }
    assert_allclose(
        model.background_mark_models_[1].location, marks[0][:3].mean(axis=0)
    )
    output = model.compute_log_likelihood(
        time,
        position,
        [np.empty(0), np.array([0.2])],
        [np.empty((0, 2)), np.array([[4.0, 2.0]])],
        time_edges=np.array([0.0, 1.0]),
    )
    assert np.isfinite(output).all()


@pytest.mark.unit
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("n_duplicates", [1, 32, 1000])
def test_mark_normalizer_survives_far_position_tail(local, n_duplicates):
    from non_local_detector.likelihoods.background_marks import _raw_kde_marks
    from non_local_detector.likelihoods.common import KDEModel

    occupancy = KDEModel(std=np.array([1.0])).fit(
        np.zeros((n_duplicates, 1)), weights=np.linspace(1.0, 3.0, n_duplicates)
    )
    model = {
        "occupancy_model": occupancy,
        "encoding_positions": [np.array([[0.0]])],
        "encoding_spike_waveform_features": [np.array([[0.0]])],
        "mean_rates": np.array([1.0]),
        "position_std": np.array([1.0]),
        "waveform_std": np.array([1.0]),
    }
    result = _raw_kde_marks(
        model, 0, np.array([[200.0]]), np.array([[0.0]]), local, None
    )
    assert_allclose(result, -0.5 * np.log(2 * np.pi), atol=1e-5)


@pytest.mark.integration
def test_zero_coverage_active_group_keeps_permitted_reference_training():
    time, position, spikes, marks = _data()
    model = _fit("clusterless_kde")
    for observation in model.observation_models:
        observation.encoding_group = 1
    training = time < 0.5
    with pytest.warns(UserWarning, match="no training samples"):
        model.fit_encoding_model(
            time,
            position,
            spikes,
            marks,
            is_training=training,
            encoding_group_labels=(~training).astype(int),
        )
    reference = model.background_mark_models_[0]
    assert_allclose(reference.location, marks[0][:3].mean(axis=0))
    assert model.background_mark_sources_[0]["kind"] == "electrode"
    assert all(
        np.asarray(encoding["mean_rates"]).sum() == 0
        for encoding in model.encoding_model_.values()
    )
    output = model.compute_log_likelihood(
        time, position, spikes, marks, time_edges=np.linspace(0, 1, 11)
    )
    assert np.isfinite(output).all()


@pytest.mark.unit
def test_position_tiles_bound_background_compensator_density_workspace(monkeypatch):
    from types import SimpleNamespace

    from non_local_detector.likelihoods import background_marks
    from non_local_detector.likelihoods.common import KDEModel

    centers = np.linspace(0, 1, 37)[:, None]
    model = {
        "environment": SimpleNamespace(
            place_bin_centers_=centers,
            is_track_interior_=np.ones(37, dtype=bool),
        ),
        "occupancy_model": KDEModel(std=1.0).fit(centers),
        "gpi_models": [KDEModel(std=1.0).fit(centers)],
        "mean_rates": np.ones(1),
        "position_block_size": 4,
    }
    compiled_density = background_marks._kde_density_parts
    query_sizes = []

    def observed_density(points, *args, **kwargs):
        query_sizes.append(len(points))
        return compiled_density(points, *args, **kwargs)

    monkeypatch.setattr(background_marks, "_kde_density_parts", observed_density)
    result = background_marks.predict_clusterless_with_background(
        "clusterless_kde",
        model,
        None,
        None,
        [np.empty(0)],
        [np.empty((0, 1))],
        np.array([0.0, 1.0]),
        background_mark_models=[BackgroundMarkModel(np.zeros(1), np.ones(1))],
    )
    assert result.shape == (1, 37)
    assert query_sizes == [4] * 9 + [1] + [4] * 9 + [1]
