"""The mark integral of an event intensity must equal its ground process."""

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from scipy.integrate import trapezoid

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.clusterless_diffusion import (
    _linear_heat_apply,
    _positive_source_columns,
    _prediction_source_operator,
    _source_cache_budget,
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_gmm import (
    _raw_gmm_log_intensity,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    connected_component_labels,
    diffusion_eigenbasis,
)
from non_local_detector.likelihoods.gmm import (
    GaussianMixtureModel,
    _compute_precision_cholesky,
)

pytestmark = pytest.mark.unit


def _environment():
    return Environment(place_bin_size=1.0, position_range=((0.0, 3.0),)).fit_place_grid(
        np.linspace(0.0, 3.0, 30)[:, None], infer_track_interior=False
    )


def test_source_kernel_is_shared_across_refits_and_invalidated_with_grid():
    env = _environment()
    kwargs = {
        "position_time": np.arange(3.0),
        "position": env.place_bin_centers_,
        "spike_times": [np.array([0.0, 2.0])],
        "spike_waveform_features": [np.array([[-5.0], [5.0]])],
        "environment": env,
        "position_std": 1.0,
        "waveform_std": 0.25,
        "heat_kernel_rank": 2,
        "disable_progress_bar": True,
    }
    first = fit_clusterless_diffusion_encoding_model(**kwargs)
    second = fit_clusterless_diffusion_encoding_model(
        **kwargs, weights=np.array([1.0, 0.5, 1.0])
    )
    from non_local_detector.likelihoods.diffusion import get_device_basis

    _, basis, _, _ = get_device_basis(env, first["resolved_rank"])
    assert (
        _prediction_source_operator(first, basis)[0]
        is _prediction_source_operator(second, basis)[0]
    )
    assert "_diffusion_source_kernel_" not in env.__getstate__()
    env.fit_place_grid(env.place_bin_centers_, infer_track_interior=False)
    assert not hasattr(env, "_diffusion_source_kernel_")


def _gaussian(means, covariance, covariance_type="full"):
    means = jnp.asarray(means)
    covariance = jnp.asarray(covariance)
    if covariance_type == "tied":
        covariance = covariance[0]
    elif covariance_type == "diag":
        covariance = jnp.diagonal(covariance, axis1=-2, axis2=-1)
    elif covariance_type == "spherical":
        covariance = jnp.diagonal(covariance, axis1=-2, axis2=-1).mean(axis=1)
    model = GaussianMixtureModel(
        n_components=means.shape[0], covariance_type=covariance_type
    )
    model.means_ = means
    model.weights_ = jnp.full(means.shape[0], 1.0 / means.shape[0])
    model.covariances_ = covariance
    model.precisions_chol_ = _compute_precision_cholesky(covariance, covariance_type)
    return model


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("bin_tile_size", [None, 2])
def test_gmm_conditional_mark_normalizer_survives_deep_position_tails(
    covariance_type, is_local, bin_tile_size
):
    env = Environment(
        place_bin_size=200.0, position_range=((0.0, 4000.0),)
    ).fit_place_grid(np.array([[0.0], [4000.0]]), infer_track_interior=False)
    occupancy = _gaussian([[0.0]], [[[1.0]]])
    joint = _gaussian([[0.0, 0.0]], [np.eye(2)], covariance_type)
    model = {
        "joint_models": [joint],
        "gpi_models": [occupancy],
        "occupancy_model": occupancy,
        "mean_rates": [1.0],
    }
    positions = jnp.array([[20.0], [200.0], [2000.0]])
    marks = jnp.full((3 if is_local else 1, 1), 0.25)
    expected = -0.5 * np.log(2 * np.pi) - 0.5 * 0.25**2
    raw = _raw_gmm_log_intensity(model, 0, positions, marks, is_local)
    np.testing.assert_allclose(raw, expected, rtol=1e-5, atol=1e-6)
    bins = jnp.asarray(env.place_bin_centers_)
    score = predict_clusterless_gmm_log_likelihood(
        position_time=np.array([0.0, 1.0]),
        position=np.full((2, 1), 2000.0),
        spike_times=[np.array([0.5])],
        spike_waveform_features=[np.array([[0.25]])],
        time_edges=np.array([0.0, 1.0]),
        environment=env,
        interior_place_bin_centers=bins,
        log_occupancy=occupancy.score_samples(bins),
        summed_ground_process_intensity=jnp.ones(bins.shape[0]),
        mark_dimensions=[1],
        is_local=is_local,
        bin_tile_size=bin_tile_size,
        disable_progress_bar=True,
        **model,
    )
    np.testing.assert_allclose(score, expected - 1.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gmm_deep_tail_conditional_mixture_retains_component_weights(covariance_type):
    occupancy = _gaussian([[0.0]], [[[1.0]]])
    joint = _gaussian(
        [[0.0, -2.0], [0.0, 2.0]], [np.eye(2), np.eye(2)], covariance_type
    )
    joint.weights_ = jnp.array([0.2, 0.8])
    model = {
        "joint_models": [joint],
        "gpi_models": [occupancy],
        "occupancy_model": occupancy,
        "mean_rates": [1.0],
    }
    raw = _raw_gmm_log_intensity(
        model, 0, jnp.array([[2000.0]]), jnp.array([[-2.0]]), True
    )
    expected = np.log((0.2 + 0.8 * np.exp(-8.0)) / np.sqrt(2 * np.pi))
    np.testing.assert_allclose(raw, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("covariance_type", ["full", "tied"])
def test_correlated_gaussian_conditional_matches_schur_reference_in_deep_tail(
    covariance_type,
):
    occupancy = _gaussian([[0.0]], [[[1.0]]])
    joint = _gaussian([[0.0, 0.0]], [[[1.0, 0.5], [0.5, 1.0]]], covariance_type)
    model = {
        "joint_models": [joint],
        "gpi_models": [occupancy],
        "occupancy_model": occupancy,
        "mean_rates": [1.0],
    }
    raw = _raw_gmm_log_intensity(
        model, 0, jnp.array([[2000.0]]), jnp.array([[1000.25]]), True
    )
    # Conditional mark mean is 0.5*x and variance is 1 - 0.5**2.
    expected = -0.5 * np.log(2 * np.pi * 0.75) - 0.5 * 0.25**2 / 0.75
    np.testing.assert_allclose(raw, expected, rtol=1e-5, atol=1e-6)


def test_diffusion_rejects_refit_with_same_indices_but_changed_physical_grid():
    env = _environment()
    encoding = fit_clusterless_diffusion_encoding_model(
        np.arange(3.0),
        env.place_bin_centers_,
        [np.array([0.0, 2.0])],
        [np.array([[-5.0], [5.0]])],
        env,
        position_std=1.0,
        waveform_std=0.25,
        disable_progress_bar=True,
    )
    env.place_bin_size = 2.0
    env.position_range = ((0.0, 6.0),)
    env.fit_place_grid(np.array([[0.0], [6.0]]), infer_track_interior=False)
    assert env.place_bin_centers_.shape[0] == encoding["occupancy"].shape[0]
    with pytest.raises(ValidationError, match="stale|refit"):
        predict_clusterless_diffusion_log_likelihood(
            np.arange(3.0),
            np.zeros((3, 1)),
            [np.array([0.5])],
            [np.array([[0.0]])],
            time_edges=np.array([0.0, 1.0]),
            **encoding,
        )


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("bin_tile_size", [None, 2])
@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gmm_mark_integral_recovers_separately_fitted_ground_process(
    is_local, bin_tile_size, covariance_type
):
    env = _environment()
    bins = jnp.asarray(env.place_bin_centers_)
    occupancy = _gaussian([[1.5]], [[[4.0]]])
    gpi = _gaussian([[0.5]], [[[2.0]]])
    joint = _gaussian(
        [[0.0, -1.0], [2.0, 1.0]],
        [[[1.0, 0.4], [0.4, 1.0]], [[0.5, -0.2], [-0.2, 2.0]]],
        covariance_type,
    )
    rate = 2.0
    ground = rate * np.exp(
        np.asarray(gpi.score_samples(bins) - occupancy.score_samples(bins))
    )
    marks = np.linspace(-12.0, 12.0, 801)
    edges = np.arange(marks.size + 1, dtype=float)
    likelihood = predict_clusterless_gmm_log_likelihood(
        position_time=np.array([0.0, float(marks.size)]),
        position=np.full((2, 1), 1.5),
        spike_times=[edges[:-1] + 0.5],
        spike_waveform_features=[marks[:, None]],
        time_edges=edges,
        environment=env,
        occupancy_model=occupancy,
        interior_place_bin_centers=bins,
        log_occupancy=occupancy.score_samples(bins),
        gpi_models=[gpi],
        joint_models=[joint],
        mean_rates=jnp.asarray([rate]),
        summed_ground_process_intensity=jnp.asarray(ground),
        mark_dimensions=[1],
        is_local=is_local,
        bin_tile_size=bin_tile_size,
        disable_progress_bar=True,
    )
    expected = ground[1:2] if is_local else ground
    recovered = trapezoid(np.exp(np.asarray(likelihood) + expected), marks, axis=0)
    np.testing.assert_allclose(recovered, expected, rtol=1e-5, atol=1e-6)
    model = {
        "joint_models": [joint],
        "gpi_models": [gpi],
        "occupancy_model": occupancy,
        "mean_rates": [rate],
    }
    positions = jnp.full((marks.size, 1), 1.5) if is_local else bins
    raw = _raw_gmm_log_intensity(
        model, 0, positions, jnp.asarray(marks[:, None]), is_local
    )
    if is_local:
        raw = raw[:, None]
    np.testing.assert_allclose(
        trapezoid(np.exp(np.asarray(raw)), marks, axis=0),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("is_local", [False, True])
def test_raw_zero_weight_gmm_has_zero_event_intensity(is_local):
    model = {"joint_models": [None], "mean_rates": [0.0]}
    raw = _raw_gmm_log_intensity(model, 0, jnp.ones((2, 1)), jnp.ones((2, 1)), is_local)
    assert raw.shape == ((2,) if is_local else (2, 2))
    assert np.all(np.isneginf(np.asarray(raw)))


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("memory_budget", [64, 536870912])
def test_diffusion_mark_integral_recovers_ground_process_after_truncation(
    rank, is_local, memory_budget
):
    env = _environment()
    encoding = fit_clusterless_diffusion_encoding_model(
        np.arange(3.0),
        env.place_bin_centers_,
        [np.array([0.0, 2.0])],
        [np.array([[-5.0], [5.0]])],
        env,
        position_std=0.1,
        waveform_std=0.25,
        heat_kernel_rank=rank,
        memory_budget=memory_budget,
        disable_progress_bar=True,
    )
    marks = np.linspace(-8.0, 8.0, 801)
    edges = np.arange(marks.size + 1, dtype=float)
    likelihood = predict_clusterless_diffusion_log_likelihood(
        np.array([0.0, float(marks.size)]),
        np.full((2, 1), 1.5),
        [edges[:-1] + 0.5],
        [marks[:, None]],
        time_edges=edges,
        is_local=is_local,
        **encoding,
    )
    ground = np.asarray(encoding["summed_ground_process_intensity"])
    expected = ground[1:2] if is_local else ground
    recovered = trapezoid(np.exp(np.asarray(likelihood) + expected), marks, axis=0)
    np.testing.assert_allclose(recovered, expected, rtol=1e-5, atol=1e-6)
    cache = getattr(encoding["environment"], "_diffusion_source_kernel_", None)
    cache_bytes = 0 if cache is None else cache[-1].nbytes + cache[-2].nbytes
    assert cache_bytes <= _source_cache_budget(memory_budget)


@pytest.mark.parametrize("rank", [2, 4, 6])
def test_corrected_source_operator_is_positive_linear_and_component_mass_preserving(
    rank,
):
    graph = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
    for left, right in graph.edges:
        graph.edges[left, right]["distance"] = 1.0
    eigvals, eigvecs = diffusion_eigenbasis(build_laplacian(graph), rank)
    eigvals, eigvecs = jnp.asarray(eigvals), jnp.asarray(eigvecs)
    labels = jnp.asarray(connected_component_labels(graph))
    kernel = _positive_source_columns(
        eigvals, eigvecs, labels, jnp.arange(6), position_std=0.1
    )
    columns = np.asarray(kernel)
    assert np.all(columns >= 0.0)
    np.testing.assert_allclose(columns.sum(axis=0), 1.0, rtol=1e-6, atol=1e-7)
    assert np.all(columns[:3, 3:] == 0.0)
    assert np.all(columns[3:, :3] == 0.0)
    fields = jnp.zeros((6, 2)).at[0, 0].set(1.0).at[5, 1].set(3.0)
    cached = np.asarray(
        _linear_heat_apply(
            eigvals,
            eigvecs,
            labels,
            fields,
            kernel,
            position_std=0.1,
            source_tile_size=1,
        )
    )
    tiled = np.asarray(
        _linear_heat_apply(
            eigvals,
            eigvecs,
            labels,
            fields,
            jnp.zeros((6, 0)),
            position_std=0.1,
            source_tile_size=2,
        )
    )
    combined = np.asarray(
        _linear_heat_apply(
            eigvals,
            eigvecs,
            labels,
            fields.sum(axis=1, keepdims=True),
            kernel,
            position_std=0.1,
            source_tile_size=1,
        )
    )
    np.testing.assert_allclose(cached, tiled, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(cached.sum(axis=1), combined[:, 0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(cached.sum(axis=0), [1.0, 3.0], rtol=1e-6, atol=1e-7)
