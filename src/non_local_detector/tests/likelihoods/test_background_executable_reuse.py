"""Ragged mark populations reuse padded reference-aware scoring executables."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from non_local_detector.likelihoods import background_marks
from non_local_detector.likelihoods.common import _SpikeTimeOrder
from non_local_detector.tests.likelihoods.test_background_mark_numerics import (
    _repeated_position_model,
)


@pytest.mark.unit
def test_background_kde_reuses_fused_executable_for_nearby_counts():
    model = _repeated_position_model(0.0, None)
    model["block_size"] = 100
    reference = background_marks.BackgroundMarkModel(np.zeros(1), np.ones(1))
    edges = np.linspace(0, 1, 129)
    kernel = background_marks._background_kde_rows
    kernel.clear_cache()
    background_marks._reference_log_density.clear_cache()
    for count in (64, 33, 34, 35, 36, 37, 38, 39, 40):
        times = np.linspace(0.01, 0.99, count)
        marks = np.zeros((count, 1))
        result = background_marks.predict_clusterless_with_background(
            "clusterless_kde",
            model,
            None,
            None,
            [times],
            [marks],
            edges,
            background_mark_models=[reference],
        )
        jax.block_until_ready(result)
        assert np.isfinite(np.asarray(result)).all()
        assert kernel._cache_size() == 1
        assert background_marks._reference_log_density._cache_size() == 1


@pytest.mark.unit
def test_reference_preparation_is_shared_across_states_and_keyed_by_row_range(
    monkeypatch,
):
    reference = background_marks.BackgroundMarkModel(np.zeros(1), np.ones(1))
    order = _SpikeTimeOrder()
    edges = np.linspace(0, 1, 129)
    times = [np.linspace(0.01, 0.99, 37)]
    marks = [np.zeros((37, 1))]
    standardized = []
    original = background_marks.BackgroundMarkModel.standardize

    def spy(self, points, **kwargs):
        standardized.append(len(points))
        return original(self, points, **kwargs)

    monkeypatch.setattr(background_marks.BackgroundMarkModel, "standardize", spy)
    first = background_marks.background_mark_row_terms(
        [reference],
        times,
        marks,
        edges,
        spike_time_order=order,
    )
    second = background_marks.background_mark_row_terms(
        [reference],
        times,
        marks,
        edges,
        spike_time_order=order,
    )
    assert_allclose(first, second, atol=0, rtol=0)
    assert standardized == [37]
    chunks = [
        background_marks.background_mark_row_terms(
            [reference],
            times,
            marks,
            edges,
            row_slice=slice(start, stop),
            spike_time_order=order,
        )
        for start, stop in ((0, 50), (50, 128))
    ]
    assert_allclose(np.concatenate(chunks), first, atol=1e-6)
    assert len(standardized) == 3


@pytest.mark.unit
def test_nonlocal_density_fields_are_prepared_once_for_prediction(monkeypatch):
    model = _repeated_position_model(0.0, None)
    reference = background_marks.BackgroundMarkModel(np.zeros(1), np.ones(1))
    order = _SpikeTimeOrder()
    times = [np.array([0.1, 0.5])]
    marks = [np.zeros((2, 1))]
    edges = np.linspace(0, 1, 5)
    calls = []
    original = background_marks._kde_model_density_parts

    def spy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(background_marks, "_kde_model_density_parts", spy)
    for row_slice in (slice(0, 2), slice(2, 4)):
        background_marks.predict_clusterless_with_background(
            "clusterless_kde",
            model,
            None,
            None,
            times,
            marks,
            edges,
            background_mark_models=[reference],
            spike_time_order=order,
            row_slice=row_slice,
        )
    assert calls == [model["occupancy_model"], model["gpi_models"][0]]


@pytest.mark.unit
def test_fused_gmm_bounds_cartesian_scoring_without_changing_event_rows(monkeypatch):
    from non_local_detector.likelihoods import clusterless_gmm

    original = clusterless_gmm._gmm_score
    scored_counts = []

    def score(points, *args):
        scored_counts.append(points.shape[0])
        return original(points, *args)

    monkeypatch.setattr(clusterless_gmm, "_gmm_score", score)
    background_marks._background_gmm_rows.clear_cache()
    features = np.linspace(-1.0, 1.0, 64, dtype=np.float32)[:, None]
    rows = np.arange(64) % 32
    positions = jnp.linspace(-0.5, 0.5, 513)[:, None]
    joint_parameters = (jnp.zeros((1, 2)), jnp.eye(2)[None, :, :], jnp.ones(1))
    occupancy_parameters = (jnp.zeros((1, 1)), jnp.ones((1, 1, 1)), jnp.ones(1))
    result = background_marks._background_gmm_rows(
        jnp.asarray(features),
        jnp.asarray(rows),
        jnp.full(64, -100.0),
        positions,
        joint_parameters,
        occupancy_parameters,
        jnp.asarray(0.0),
        is_local=False,
        n_rows=32,
        indices_are_sorted=False,
        joint_covariance_type="full",
        occupancy_covariance_type="full",
    )
    expected = np.zeros(32)
    np.add.at(expected, rows, -0.5 * features[:, 0] ** 2 - 0.5 * np.log(2 * np.pi))
    assert_allclose(result, np.broadcast_to(expected[:, None], result.shape), atol=1e-6)
    assert scored_counts
    assert max(scored_counts) <= len(features) * 128


@pytest.mark.unit
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("block_size", [16, 64, 100])
def test_background_padding_drops_fake_events_and_preserves_compensator(
    local, block_size
):
    model = _repeated_position_model(0.0, None)
    model["block_size"] = block_size
    reference = background_marks.BackgroundMarkModel(np.zeros(1), np.ones(1))
    edges = np.array([0.0, 0.2, 0.5, 1.0])
    times = np.linspace(0.01, 0.99, 37)[::-1]
    result = background_marks.predict_clusterless_with_background(
        "clusterless_kde",
        model,
        np.array([0.0, 1.0]),
        np.zeros((2, 1)),
        [times],
        [np.zeros((37, 1))],
        edges,
        background_mark_models=[reference],
        is_local=local,
    )
    counts = np.histogram(times, edges)[0]
    expected = counts * (-0.5 * np.log(2 * np.pi) + np.log(np.diff(edges))) - np.diff(
        edges
    )
    assert_allclose(result[:, 0], expected, atol=1e-5)
