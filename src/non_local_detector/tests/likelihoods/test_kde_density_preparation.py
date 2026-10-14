"""True KDE densities share only immutable, bounded prediction preparation."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    common,
    fit_clusterless_kde_encoding_model,
    fit_clusterless_kde_log_encoding_model,
    fit_sorted_spikes_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
    predict_clusterless_kde_log_log_likelihood,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.time_edges import _DecodeTimeGrid

pytestmark = pytest.mark.unit


def test_local_density_preparation_reuses_geometry_and_invalidates_model_leaves(
    monkeypatch,
):
    time = np.arange(5.0)
    position = np.arange(5.0)[:, None]
    environment = Environment(place_bin_size=1.0).fit_place_grid(position)
    model = common.KDEModel(std=1.0).fit(jnp.asarray(position))
    grid = _DecodeTimeGrid(np.arange(5.0))
    order = common._SpikeTimeOrder()
    interpolations, densities = [], []
    original_position = common.get_position_at_time
    original_density = common.kde_log_density

    def interpolate(*args, **kwargs):
        interpolations.append(1)
        return original_position(*args, **kwargs)

    def density(*args, **kwargs):
        densities.append(1)
        return original_density(*args, **kwargs)

    monkeypatch.setattr(common, "get_position_at_time", interpolate)
    monkeypatch.setattr(common, "kde_log_density", density)

    def prepare(active_order=order, start=0, stop=4):
        return common._prepare_kde_local_inputs(
            model, time, position, environment, grid, start, stop, active_order
        )

    first_points, first_density = prepare()
    second_points, second_density = prepare()
    assert first_points is second_points
    assert first_density is second_density
    assert len(interpolations) == len(densities) == 1
    model.weights_ = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
    _, updated_density = prepare()
    assert len(interpolations) == 1
    assert len(densities) == 2
    assert not np.allclose(updated_density, first_density)
    prepare(start=1, stop=4)
    prepare(common._SpikeTimeOrder())
    assert len(interpolations) == 3
    assert len(densities) == 4


def test_prediction_density_preparation_respects_its_byte_budget(monkeypatch):
    monkeypatch.setattr(common, "_KDE_PREPARATION_BYTES", 64)
    order = common._SpikeTimeOrder()
    sources = [object() for _ in range(20)]
    for source in sources:
        order.memo_kde("density", (source,), lambda: jnp.zeros(8, jnp.float32))
        assert order._kde_prepared_bytes <= 64
    assert len(order._kde_prepared) <= 2


def test_replaced_fitted_leaves_are_included_in_the_preparation_budget(monkeypatch):
    monkeypatch.setattr(common, "_KDE_PREPARATION_BYTES", 128)
    model = common.KDEModel(std=1.0).fit(jnp.zeros((12, 1)))
    points = jnp.zeros((1, 1))
    order = common._SpikeTimeOrder()
    for number in range(5):
        model.weights_ = jnp.full(12, number + 1.0)
        common._prepared_kde_log_density(model, points, order)
        assert len(order._kde_prepared) <= 1
        assert order._kde_prepared_bytes <= 128
        retained_sources = sum(
            int(getattr(source, "nbytes", 0))
            for sources, _, _ in order._kde_prepared.values()
            for source in sources
        )
        assert retained_sources <= order._kde_prepared_bytes


@pytest.mark.parametrize("block_size", [None, 2])
def test_true_density_with_no_evaluation_points_is_empty(block_size):
    model = common.KDEModel(std=1.0, block_size=block_size).fit(jnp.zeros((3, 1)))
    assert common.kde_log_density(model, jnp.zeros((0, 1))).shape == (0,)


def test_position_log_kernel_reuses_immutable_leaves_and_declines_large_cache(
    monkeypatch,
):
    order = common._SpikeTimeOrder()
    points = jnp.ones((3, 1))
    samples = jnp.zeros((4, 1))
    std = jnp.ones(1)
    first = common._prepared_kde_position_log_kernel(order, points, samples, std)
    assert first is common._prepared_kde_position_log_kernel(
        order, points, samples, std
    )
    changed = common._prepared_kde_position_log_kernel(order, points, samples, std * 2)
    assert not np.allclose(first, changed)
    monkeypatch.setattr(common, "_KDE_PREPARATION_BYTES", 16)
    assert common._prepared_kde_position_log_kernel(order, points, samples, std) is None


def test_kde_true_density_keeps_custom_prediction_and_unfitted_errors():
    class Custom(common.KDEModel):
        def predict_log(self, points):
            return jnp.full(points.shape[0], -17.0)

    model = Custom(std=1.0).fit(jnp.zeros((2, 1)))
    np.testing.assert_array_equal(
        common.kde_log_density(model, jnp.ones((3, 1))), -17.0
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        common.kde_log_density(common.KDEModel(std=1.0), jnp.ones((3, 1)))


@pytest.mark.parametrize(
    "fit,predict,clusterless",
    [
        (
            fit_sorted_spikes_kde_encoding_model,
            predict_sorted_spikes_kde_log_likelihood,
            False,
        ),
        (
            fit_clusterless_kde_encoding_model,
            predict_clusterless_kde_log_likelihood,
            True,
        ),
        (
            fit_clusterless_kde_log_encoding_model,
            predict_clusterless_kde_log_log_likelihood,
            True,
        ),
    ],
)
def test_public_local_kde_paths_share_prediction_geometry(
    fit, predict, clusterless, monkeypatch
):
    time = np.arange(5.0)
    position = np.arange(5.0)[:, None]
    environment = Environment(place_bin_size=1.0).fit_place_grid(position)
    args = [time, position, [time], environment]
    options = {"position_std": 1.0, "disable_progress_bar": True}
    if clusterless:
        args.insert(3, [np.zeros((5, 1))])
        options["waveform_std"] = 1.0
    model = fit(*args, **options)
    grid = _DecodeTimeGrid(np.arange(5.0))
    order = common._SpikeTimeOrder()
    interpolations = []
    original = common.get_position_at_time

    def interpolate(*args, **kwargs):
        interpolations.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(common, "get_position_at_time", interpolate)
    decode_args = [time, position, [np.empty(0)]]
    if clusterless:
        decode_args.append([np.empty((0, 1))])
    kwargs = dict(
        time_edges=grid.edges,
        _time_grid=grid,
        _spike_time_order=order,
        is_local=True,
        **model,
    )
    first = predict(*decode_args, **kwargs)
    second = predict(*decode_args, **kwargs)
    np.testing.assert_array_equal(first, second)
    assert len(interpolations) == 1
    kwargs["_spike_time_order"] = common._SpikeTimeOrder()
    predict(*decode_args, **kwargs)
    assert len(interpolations) == 2
