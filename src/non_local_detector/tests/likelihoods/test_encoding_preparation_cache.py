"""Content-checked encoding preparation survives weighted refits within a cap."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import SortedSpikesDecoder
from non_local_detector.encoding_time import EncodingSupport, cached_encoding_support
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
)

pytestmark = pytest.mark.unit


def _data(n=129):
    time = np.linspace(0.0, 4.0, n)
    position = (10.0 + 8.0 * np.sin(time))[:, None]
    spikes = [np.linspace(0.05, 3.95, 41)]
    env = Environment(place_bin_size=1.0, position_range=((0.0, 20.0),)).fit_place_grid(
        position, infer_track_interior=False
    )
    return time, position, spikes, env


def test_support_content_cache_reuses_copies_and_invalidates_mutation_and_bounds():
    class Owner:
        pass

    owner = Owner()
    time, position, _, _ = _data()
    first = cached_encoding_support(owner, time, position)
    assert cached_encoding_support(owner, time.copy(), position.copy()) is first
    assert not first.time.flags.writeable
    assert not first.exposure.flags.writeable
    position[0, 0] += 0.1
    second = cached_encoding_support(owner, time, position)
    assert second is not first
    third = cached_encoding_support(
        owner, time, position, encoding_time_range=[0.25, 3.75]
    )
    assert third is not second
    assert third.exposure.sum() == pytest.approx(3.5)
    old_time = third.time.copy()
    time += 0.1
    fourth = cached_encoding_support(
        owner, time, position, encoding_time_range=[0.25, 3.75]
    )
    assert fourth is not third
    np.testing.assert_array_equal(third.time, old_time)


def test_public_refit_keeps_support_and_drops_it_from_serialization():
    time, position, spikes, env = _data()
    detector = SortedSpikesDecoder(
        environments=env,
        sorted_spikes_algorithm_params={
            "position_std": 2.0,
            "disable_progress_bar": True,
        },
        infer_track_interior=False,
    )
    detector.fit(position_time=time, position=position, spike_times=spikes)
    first = detector._encoding_support_cache_[1]
    quadrature = first.quadrature(position, spatial_scale=2.0)
    detector.fit_encoding_model(
        time, position, spikes, weights=np.linspace(0.2, 1.0, len(time))
    )
    assert detector._encoding_support_cache_[1] is first
    assert first.quadrature(position, spatial_scale=2.0) is quadrature
    operator = next(
        value[0] for key, value in first._prepared_cache.items() if key[0] == "kde"
    )
    assert operator[0].shape == (len(time), len(env.place_bin_centers_))
    assert "_encoding_support_cache_" not in detector.__getstate__()


def test_prepared_cache_and_quadrature_share_one_aggregate_budget(monkeypatch):
    from non_local_detector import encoding_time

    time, position, _, _ = _data(17)
    support = EncodingSupport(time, position)
    monkeypatch.setattr(encoding_time, "_QUADRATURE_CACHE_BYTES", 12288)
    quadrature = support.quadrature(position, spatial_scale=2.0)
    support.remember_preparation(("test", 0), np.zeros(100, dtype=np.float32))
    support.remember_preparation(("test", 1), np.zeros(100_000, dtype=np.float32))
    assert support.preparation_nbytes <= 12288
    assert support.prepared(("test", 0)) is not None
    assert support.prepared(("test", 1)) is None
    assert support.quadrature(position, spatial_scale=2.0) is quadrature


@pytest.mark.parametrize("mostly_missing", [False, True])
def test_integrated_kde_construction_respects_its_temporary_workspace(
    monkeypatch, mostly_missing
):
    from non_local_detector import encoding_time
    from non_local_detector.likelihoods.common import KDEModel

    time, position, _, env = _data(65)
    if mostly_missing:
        position[:-7] = np.nan
    support = EncodingSupport(time, position)
    quadrature = support.quadrature(position, spatial_scale=2.0)
    budget = (
        2
        * 8
        * np.dtype(np.float64).itemsize
        * max(len(quadrature.positions), len(time))
    )
    monkeypatch.setattr(
        encoding_time, "_KDE_OPERATOR_WORKSPACE_BYTES", budget, raising=False
    )
    original = encoding_time._integrated_kde_operator_tile
    calls = []

    def tracked(centers, samples, *args, **kwargs):
        calls.append((len(centers), len(samples)))
        return original(centers, samples, *args, **kwargs)

    monkeypatch.setattr(encoding_time, "_integrated_kde_operator_tile", tracked)
    weights = np.linspace(0.2, 1.0, len(time))
    actual = support.cached_kde_log_density(
        quadrature,
        quadrature.positions,
        env.place_bin_centers_,
        np.array([2.0]),
        weights,
    )
    assert len(calls) > 1
    assert all(
        columns * max(rows, len(time)) * 8 * 8 <= budget for columns, rows in calls
    )
    expected = (
        KDEModel(std=np.array([2.0]))
        .fit(quadrature.positions, weights=quadrature.weighted_exposure(weights))
        .predict_log(env.place_bin_centers_)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_integrated_kde_skips_construction_when_one_column_exceeds_workspace(
    monkeypatch,
):
    from non_local_detector import encoding_time

    time, position, _, env = _data(65)
    support = EncodingSupport(time, position)
    quadrature = support.quadrature(position, spatial_scale=2.0)
    monkeypatch.setattr(
        encoding_time,
        "_KDE_OPERATOR_WORKSPACE_BYTES",
        8 * np.dtype(np.float64).itemsize * len(quadrature.positions) - 1,
    )
    assert (
        support.cached_kde_log_density(
            quadrature,
            quadrature.positions,
            env.place_bin_centers_,
            np.array([2.0]),
            np.ones(len(time)),
        )
        is None
    )
    assert not support._prepared_cache


def test_integrated_kde_zero_basis_mass_cannot_poison_a_distant_source():
    from non_local_detector.encoding_time import _integrated_kde_operator_tile
    from non_local_detector.likelihoods.common import _log_kernel_matrix

    centers, positions, std = np.array([[0.0]]), np.array([[0.0], [200.0]]), np.ones(1)
    offset, mass = _integrated_kde_operator_tile(
        centers,
        positions,
        std,
        np.array([0.0, 1.0]),
        np.zeros(2),
        np.zeros(2, dtype=np.intp),
        np.ones(2, dtype=np.intp),
        n_samples=2,
    )
    np.testing.assert_array_equal(
        offset[:1], _log_kernel_matrix(centers, positions[1:], std)
    )
    np.testing.assert_array_equal(mass[:1], np.zeros((1, 1)))
    assert np.isneginf(offset[1, 0])
    assert np.isneginf(mass[1, 0])


def test_prepared_kde_empty_tracking_support_has_no_mass():
    from non_local_detector.encoding_time import preparation_key

    assert preparation_key(np.empty((0, 1))) != preparation_key(np.empty((0, 2)))
    time = np.linspace(0.0, 1.0, 17)
    position = np.full((len(time), 1), np.nan)
    support = EncodingSupport(time, position)
    quadrature = support.quadrature(position, spatial_scale=2.0)
    actual = support.cached_kde_log_density(
        quadrature,
        quadrature.positions,
        np.array([[0.0], [200.0]]),
        np.ones(1),
        np.ones(len(time)),
    )
    assert np.all(np.isneginf(actual))


def test_glm_weighted_refit_reuses_design_and_invalidates_knot_and_grid_changes(
    monkeypatch,
):
    from non_local_detector.likelihoods import sorted_spikes_glm as glm

    time, position, spikes, env = _data()
    support = EncodingSupport(time, position)
    calls = []
    original = glm.make_spline_predict_matrix

    def tracked(info, points):
        calls.append(len(points))
        return original(info, points)

    monkeypatch.setattr(glm, "make_spline_predict_matrix", tracked)
    kwargs = {
        "position_time": time,
        "position": position,
        "spike_times": spikes,
        "environment": env,
        "place_bin_edges": env.place_bin_edges_,
        "edges": env.edges_,
        "is_track_interior": env.is_track_interior_,
        "is_track_boundary": env.is_track_boundary_,
        "emission_knot_spacing": 5.0,
        "l2_penalty": 0.5,
        "disable_progress_bar": True,
        "_encoding_support": support,
    }
    first = fit_sorted_spikes_glm_encoding_model(**kwargs)
    calls.clear()
    cached = fit_sorted_spikes_glm_encoding_model(
        **kwargs, weights=np.linspace(0.2, 1.0, len(time))
    )
    assert not calls, "basis geometry should be reused when only weights change"
    uncached = fit_sorted_spikes_glm_encoding_model(
        **dict(kwargs, _encoding_support=None), weights=np.linspace(0.2, 1.0, len(time))
    )
    np.testing.assert_array_equal(
        np.asarray(cached["place_fields"]), np.asarray(uncached["place_fields"])
    )
    assert np.all(np.isfinite(np.asarray(first["place_fields"])))
    calls.clear()
    fit_sorted_spikes_glm_encoding_model(**dict(kwargs, emission_knot_spacing=4.0))
    assert calls
    calls.clear()
    env.place_bin_size = 1.25
    env.fit_place_grid(position, infer_track_interior=False)
    changed_grid = dict(
        kwargs,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
    )
    fit_sorted_spikes_glm_encoding_model(**changed_grid)
    assert calls
    assert support.preparation_nbytes <= 32 * 2**20
    assert all(
        not value.flags.writeable
        for value in vars(support._quadrature_cache[1]).values()
    )


def test_glm_partial_cache_reuses_prefix_without_growing_or_changing_rates(monkeypatch):
    from non_local_detector import encoding_time
    from non_local_detector.likelihoods import sorted_spikes_glm as glm

    time, position, spikes, env = _data()
    support = EncodingSupport(time, position)
    monkeypatch.setattr(encoding_time, "_QUADRATURE_CACHE_BYTES", 8192)
    monkeypatch.setattr(glm, "_GLM_DESIGN_ROWS", 64)
    calls = []
    original = glm.make_spline_predict_matrix

    def tracked(info, points):
        calls.append(len(points))
        return original(info, points)

    monkeypatch.setattr(glm, "make_spline_predict_matrix", tracked)
    kwargs = {
        "position_time": time,
        "position": position,
        "spike_times": spikes,
        "environment": env,
        "place_bin_edges": env.place_bin_edges_,
        "edges": env.edges_,
        "is_track_interior": env.is_track_interior_,
        "is_track_boundary": env.is_track_boundary_,
        "emission_knot_spacing": 5.0,
        "disable_progress_bar": True,
        "_encoding_support": support,
    }
    first = fit_sorted_spikes_glm_encoding_model(**kwargs)
    first_calls = len(calls)
    first_bytes = support.preparation_nbytes
    calls.clear()
    second = fit_sorted_spikes_glm_encoding_model(**kwargs)
    assert 0 < len(calls) < first_calls
    assert support.preparation_nbytes == first_bytes <= 8192
    np.testing.assert_array_equal(
        np.asarray(first["place_fields"]), np.asarray(second["place_fields"])
    )


def test_support_cache_tracks_declared_intervals_without_freezing_caller_arrays():
    class Owner:
        pass

    time, position, _, _ = _data()
    intervals = np.array([[0.0, 2.0], [3.0, 4.0]])
    owner = Owner()
    first = cached_encoding_support(
        owner, time, position, valid_position_intervals=intervals
    )
    assert intervals.flags.writeable
    assert not first.contains([2.5])[0]
    intervals[0, 1] = 2.75
    second = cached_encoding_support(
        owner, time, position, valid_position_intervals=intervals
    )
    assert second is not first
    assert second.contains([2.5])[0]
    assert not first.contains([2.5])[0]


def test_fragmented_support_metadata_is_counted_before_retaining_owner_cache(
    monkeypatch,
):
    from non_local_detector import encoding_time

    class Owner:
        pass

    time = np.linspace(0.0, 1.0, 511)
    position = np.zeros((len(time), 1))
    position[1::2] = np.nan
    monkeypatch.setattr(encoding_time, "_QUADRATURE_CACHE_BYTES", 16384)
    owner = Owner()
    support = cached_encoding_support(owner, time, position)
    array_bytes = sum(
        value.nbytes
        for value in (support.time, support.exposure, support.indices, support.bounds)
    )
    array_bytes += sum(indices.nbytes for _, _, indices in support.segments)
    assert array_bytes < 16384
    assert len(support.segments) == 256
    assert support.preparation_nbytes > 16384
    assert not hasattr(owner, "_encoding_support_cache_")


def test_cached_kde_operator_invalidates_mutated_bandwidth_and_centers():
    from non_local_detector.likelihoods.common import KDEModel

    time, position, _, env = _data(33)
    support = EncodingSupport(time, position)
    quad = support.quadrature(position, spatial_scale=2.0)
    centers = env.place_bin_centers_.copy()
    std = np.array([2.0])
    weights = np.linspace(0.2, 1.0, len(time))
    first = support.cached_kde_log_density(quad, quad.positions, centers, std, weights)
    first_kernel = next(iter(support._prepared_cache.values()))[0]
    std[0] = 1.5
    centers += 0.1
    changed = support.cached_kde_log_density(
        quad, quad.positions, centers, std, weights
    )
    second_kernel = next(iter(support._prepared_cache.values()))[0]
    assert first_kernel is not second_kernel
    assert not np.array_equal(np.asarray(first), np.asarray(changed))
    expected = (
        KDEModel(std=std)
        .fit(quad.positions, weights=quad.weighted_exposure(weights))
        .predict_log(centers)
    )
    np.testing.assert_allclose(changed, expected, rtol=1e-6, atol=1e-6)


def test_zero_exposure_cached_density_has_finite_zero_gradient():
    from non_local_detector.encoding_time import _prepared_kde_log_density

    kernel = jnp.array([[-1.0, -2.0], [-3.0, -4.0]])
    result = _prepared_kde_log_density(kernel, jnp.zeros(2))
    assert np.all(np.isneginf(np.asarray(result)))
    gradient = jax.grad(
        lambda weights: jnp.exp(_prepared_kde_log_density(kernel, weights)).sum()
    )(jnp.zeros(2))
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros(2))


def test_integrated_kde_cache_uses_original_rows_and_recovers_activated_tail():
    from non_local_detector.likelihoods.common import KDEModel

    time = np.linspace(0.0, 1.0, 129)
    position = (10.0 * time)[:, None]
    support = EncodingSupport(time, position, encoding_time_range=[0.0, 1.0])
    q = support.quadrature(position, spatial_scale=0.1)
    centers = np.array([[-100.0], [10.0]])
    support.cached_kde_log_density(
        q, q.positions, centers, np.array([0.1]), np.ones(len(time))
    )
    operator = next(iter(support._prepared_cache.values()))[0]
    assert operator[0].shape == (len(time), len(centers))
    assert len(q.times) > 4 * (len(time) - 1)
    weights = np.zeros(len(time))
    weights[-1] = 1.0
    actual = support.cached_kde_log_density(
        q, q.positions, centers, np.array([0.1]), weights
    )
    expected = (
        KDEModel(std=np.array([0.1]))
        .fit(q.positions, weights=q.weighted_exposure(weights))
        .predict_log(centers)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(np.asarray(actual)))
    assert float(np.asarray(operator[2])[-1, 0]) == 0.0, (
        "tail must exercise log fallback"
    )
    single = support.cached_kde_log_density(
        q, q.positions, centers[1:], np.array([0.1]), weights
    )
    np.testing.assert_array_equal(np.asarray(actual)[1:], np.asarray(single))


def test_cached_kde_occupancy_has_stable_tails_and_weight_updates():
    time, position, _, env = _data(33)
    support = EncodingSupport(time, position)
    quad = support.quadrature(position, spatial_scale=2.0)
    centers = np.vstack((env.place_bin_centers_, [[1000.0]]))
    weights = np.linspace(0.0, 1.0, len(time))
    actual = support.cached_kde_log_density(
        quad, quad.positions, centers, np.array([2.0]), weights
    )
    from non_local_detector.likelihoods.common import KDEModel

    model = KDEModel(std=np.array([2.0])).fit(
        quad.positions, weights=quad.weighted_exposure(weights)
    )
    expected = model.predict_log(centers)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(np.asarray(actual)))
    previous = support.preparation_nbytes
    changed = support.cached_kde_log_density(
        quad, quad.positions, centers, np.array([2.0]), 1.0 - weights
    )
    assert support.preparation_nbytes == previous
    assert not np.array_equal(np.asarray(actual), np.asarray(changed))
    assert all(
        all(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(value[0]))
        for key, value in support._prepared_cache.items()
        if key[0] == "kde"
    )
