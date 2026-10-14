"""Occupied source columns preserve the full positive diffusion operator."""

import pickle

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_diffusion import (
    _cached_source_kernel,
    _linear_heat_apply,
    _positive_source_columns,
    _prediction_source_operator,
    _source_cache_budget,
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    connected_component_labels,
    diffusion_eigenbasis,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rank", [2, 4, 6])
@pytest.mark.parametrize("cached", [False, True])
def test_compact_sources_match_dense_fields_on_disconnected_graph(rank, cached):
    graph = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
    for u, v in graph.edges:
        graph.edges[u, v]["distance"] = 1.0
    vals, vecs = diffusion_eigenbasis(build_laplacian(graph), rank)
    vals, vecs = jnp.asarray(vals), jnp.asarray(vecs)
    labels = jnp.asarray(connected_component_labels(graph))
    sources = jnp.array([0, 2, 5])
    compact_fields = jnp.array([[1.0, 3.0], [2.0, 0.0], [0.5, 7.0]])
    dense_fields = jnp.zeros((6, 2)).at[sources].set(compact_fields)
    full_kernel = _positive_source_columns(
        vals, vecs, labels, jnp.arange(6), position_std=0.1
    )
    reference = np.asarray(full_kernel @ dense_fields)
    kernel = full_kernel[:, sources] if cached else jnp.zeros((6, 0))
    result = np.asarray(
        _linear_heat_apply(
            vals,
            vecs,
            labels,
            compact_fields,
            kernel,
            position_std=0.1,
            source_tile_size=2,
            source_bins=sources,
        )
    )
    np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result[:3].sum(axis=0), [3.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(result[3:].sum(axis=0), [0.5, 7.0], rtol=1e-6)


def _sparse_encoding(memory_budget=512 * 1024 * 1024, weights=None, environment=None):
    env = environment or Environment(
        place_bin_size=1.0, position_range=((0.0, 32.0),)
    ).fit_place_grid(np.array([[0.0], [32.0]]), infer_track_interior=False)
    times = np.arange(33.0)
    position = (times + 0.01)[:, None]
    model = fit_clusterless_diffusion_encoding_model(
        times,
        position,
        [np.array([1.0, 30.0])],
        [np.array([[-1.0], [1.0]])],
        env,
        position_std=2.0,
        waveform_std=0.5,
        heat_kernel_rank=8,
        memory_budget=memory_budget,
        disable_progress_bar=True,
        weights=weights,
    )
    return model


def _resolved_source_kernel(model):
    from non_local_detector.likelihoods.diffusion import get_device_basis

    _, vecs, _, _ = get_device_basis(model["environment"], model["resolved_rank"])
    return _prediction_source_operator(model, vecs)[0]


def test_fit_caches_only_immutable_encoding_source_support():
    weights = np.ones(33)
    weights[30] = 0.0
    first = _sparse_encoding(weights=weights)
    sources = first["source_bin_indices"]
    assert sources.shape == (2,)
    assert not sources.flags.writeable
    first_kernel = _resolved_source_kernel(first)
    assert first_kernel.shape == (32, 2)
    assert first["source_kernel"].nbytes == 0
    assert first["encoding_source_indices"][0].tolist() == [0, 1]
    assert not first["encoding_source_indices"][0].flags.writeable
    second = _sparse_encoding(weights=np.ones(33), environment=first["environment"])
    assert first_kernel is _resolved_source_kernel(second)
    np.testing.assert_array_equal(
        first["source_bin_indices"], second["source_bin_indices"]
    )
    assert float(second["encoding_weights"][0][1]) > 0.0
    np.testing.assert_allclose(
        np.asarray(_resolved_source_kernel(second)).sum(axis=0), 1.0, rtol=1e-6
    )


@pytest.mark.parametrize("is_local", [False, True])
def test_public_compact_prediction_matches_full_operator(is_local):
    model = _sparse_encoding()
    env = model["environment"]
    from non_local_detector.likelihoods.diffusion import get_device_basis

    vals, vecs, labels, _ = get_device_basis(env, model["resolved_rank"])
    full = dict(model)
    full.pop("source_bin_indices")
    full.pop("encoding_source_indices")
    full["source_kernel"] = _positive_source_columns(
        vals, vecs, labels, jnp.arange(32), position_std=2.0
    )
    kwargs = {
        "position_time": np.array([0.0, 1.0]),
        "position": np.array([[1.01], [30.01]]),
        "spike_times": [np.array([0.1, 0.2, 0.8])],
        "spike_waveform_features": [np.array([[-1.0], [0.0], [1.0]])],
        "time_edges": np.array([0.0, 0.5, 1.0]),
        "is_local": is_local,
    }
    reference = predict_clusterless_diffusion_log_likelihood(**kwargs, **full)
    result = predict_clusterless_diffusion_log_likelihood(**kwargs, **model)
    tiled = dict(model, source_kernel=jnp.zeros((32, 0)), memory_budget=64)
    fallback = predict_clusterless_diffusion_log_likelihood(**kwargs, **tiled)
    full_tiled = dict(full, source_kernel=jnp.zeros((32, 0)), memory_budget=64)
    reference_tiled = predict_clusterless_diffusion_log_likelihood(
        **kwargs, **full_tiled
    )
    np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-7)
    # Match the same column-construction tile size; changing that size already
    # changes float32 heat-column rounding in the legacy full-source path.
    np.testing.assert_allclose(fallback, reference_tiled, rtol=1e-6, atol=1e-7)


def test_cache_budget_applies_to_all_source_columns_and_replaces_old_support():
    model = _sparse_encoding()
    env = model["environment"]
    from non_local_detector.likelihoods.diffusion import get_device_basis

    vals, vecs, labels, _ = get_device_basis(env, model["resolved_rank"])
    first = _cached_source_kernel(
        vals, vecs, labels, 2.0, 2048, env, sources=np.array([0, 31])
    )
    second = _cached_source_kernel(
        vals, vecs, labels, 2.0, 2048, env, sources=np.array([1, 30])
    )
    assert first is not second
    assert second.nbytes <= _source_cache_budget(2048)
    assert env._diffusion_source_kernel_[-1] is second
    assert env._diffusion_source_kernel_[
        -2
    ].nbytes + second.nbytes <= _source_cache_budget(2048)
    changed_bandwidth = _cached_source_kernel(
        vals, vecs, labels, 3.0, 2048, env, sources=np.array([1, 30])
    )
    assert changed_bandwidth is not second
    changed_basis = _cached_source_kernel(
        vals[:4], vecs[:, :4], labels, 3.0, 2048, env, sources=np.array([1, 30])
    )
    assert changed_basis is not changed_bandwidth
    np.testing.assert_allclose(np.asarray(changed_basis).sum(axis=0), 1.0, rtol=1e-6)
    empty = _cached_source_kernel(
        vals, vecs, labels, 2.0, 128, env, sources=np.arange(32)
    )
    assert empty.shape == (32, 0)
    assert not hasattr(env, "_diffusion_source_kernel_")


def _second_encoding_group(first, memory_budget):
    return fit_clusterless_diffusion_encoding_model(
        np.arange(33.0),
        (np.arange(33.0) + 0.01)[:, None],
        [np.array([2.0, 29.0])],
        [np.array([[-1.0], [1.0]])],
        first["environment"],
        position_std=2.0,
        waveform_std=0.5,
        heat_kernel_rank=8,
        memory_budget=memory_budget,
        disable_progress_bar=True,
    )


def test_live_encoding_groups_cannot_retain_independent_cache_matrices():
    memory_budget = 1536
    first = _sparse_encoding(memory_budget=memory_budget)
    second = _second_encoding_group(first, memory_budget)
    arrays = {
        id(array): array
        for array in (
            first["source_kernel"],
            second["source_kernel"],
            first["environment"]._diffusion_source_kernel_[-1],
        )
    }
    support_bytes = first["environment"]._diffusion_source_kernel_[-2].nbytes
    assert sum(
        array.nbytes for array in arrays.values()
    ) + support_bytes <= _source_cache_budget(memory_budget)


def test_groups_share_union_of_occupied_columns_when_it_fits():
    memory_budget = 4096
    first = _sparse_encoding(memory_budget=memory_budget)
    second = _second_encoding_group(first, memory_budget)
    environment = first["environment"]
    np.testing.assert_array_equal(
        environment._diffusion_source_kernel_[-2], np.array([1, 2, 29, 30])
    )
    from non_local_detector.likelihoods.diffusion import get_device_basis

    _, vecs, _, _ = get_device_basis(environment, first["resolved_rank"])
    cached_union = environment._diffusion_source_kernel_[-1]
    for model, columns in ((first, [0, 3]), (second, [1, 2])):
        operator, _ = _prediction_source_operator(model, vecs)
        np.testing.assert_array_equal(operator, cached_union[:, np.array(columns)])
    assert environment._diffusion_source_kernel_[-1] is cached_union


@pytest.mark.parametrize("is_local", [False, True])
def test_serialized_model_reconstructs_transient_source_cache(is_local):
    model = _sparse_encoding()
    kwargs = {
        "position_time": np.array([0.0, 1.0]),
        "position": np.array([[1.01], [30.01]]),
        "spike_times": [np.array([0.1, 0.2, 0.8])],
        "spike_waveform_features": [np.array([[-1.0], [0.0], [1.0]])],
        "time_edges": np.array([0.0, 0.5, 1.0]),
        "is_local": is_local,
    }
    expected = predict_clusterless_diffusion_log_likelihood(**kwargs, **model)
    restored = pickle.loads(pickle.dumps(model))
    assert not hasattr(restored["environment"], "_diffusion_source_kernel_")
    result = predict_clusterless_diffusion_log_likelihood(**kwargs, **restored)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    assert restored["source_kernel"].nbytes == 0
    assert hasattr(restored["environment"], "_diffusion_source_kernel_")
