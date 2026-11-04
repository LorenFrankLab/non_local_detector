"""True KDE densities share only immutable, bounded prediction preparation."""

import jax.numpy as jnp
import numpy as np
import pytest

import non_local_detector.likelihoods.clusterless_kde as clusterless_kde_module
import non_local_detector.likelihoods.common.kernel_density as kde_module
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    common,
    fit_clusterless_kde_encoding_model,
    fit_sorted_spikes_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import spike_rows
from non_local_detector.likelihoods.mark_reference import fit_mark_references
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
    order = common.SpikeTimeOrder()
    interpolations, densities = [], []
    original_position = common.get_position_at_time
    original_density = common.kde_log_density

    def interpolate(*args, **kwargs):
        interpolations.append(1)
        return original_position(*args, **kwargs)

    def density(*args, **kwargs):
        densities.append(1)
        return original_density(*args, **kwargs)

    monkeypatch.setattr(kde_module, "get_position_at_time", interpolate)
    monkeypatch.setattr(kde_module, "kde_log_density", density)

    def prepare(active_order=order, start=0, stop=4):
        return common.prepare_kde_local_inputs(
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
    prepare(common.SpikeTimeOrder())
    assert len(interpolations) == 3
    assert len(densities) == 4


def test_prediction_density_preparation_respects_its_byte_budget(monkeypatch):
    monkeypatch.setattr(spike_rows, "_KDE_PREPARATION_BYTES", 64)
    order = common.SpikeTimeOrder()
    sources = [object() for _ in range(20)]
    for source in sources:
        order.memo_kde("density", (source,), lambda: jnp.zeros(8, jnp.float32))
        assert order._kde_prepared_bytes <= 64
    assert len(order._kde_prepared) <= 2


def test_replaced_fitted_leaves_are_included_in_the_preparation_budget(monkeypatch):
    monkeypatch.setattr(spike_rows, "_KDE_PREPARATION_BYTES", 128)
    model = common.KDEModel(std=1.0).fit(jnp.zeros((12, 1)))
    points = jnp.zeros((1, 1))
    order = common.SpikeTimeOrder()
    for number in range(5):
        model.weights_ = jnp.full(12, number + 1.0)
        common.prepared_kde_log_density(model, points, order)
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
    ("fit", "predict", "clusterless"),
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
    if clusterless:
        model["mark_reference"] = fit_mark_references(args[3], kernel_scale=1.0)
    grid = _DecodeTimeGrid(np.arange(5.0))
    order = common.SpikeTimeOrder()
    interpolations = []
    original = common.get_position_at_time

    def interpolate(*args, **kwargs):
        interpolations.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(kde_module, "get_position_at_time", interpolate)
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
    kwargs["_spike_time_order"] = common.SpikeTimeOrder()
    predict(*decode_args, **kwargs)
    assert len(interpolations) == 2


@pytest.mark.parametrize("is_local", [True, False])
def test_clusterless_prediction_evaluates_fitted_densities_in_its_block_size(
    is_local, monkeypatch
):
    """A block size given at predict also blocks the fitted occupancy and
    ground-process densities, without changing the stored models."""
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 20.0, 0.01)
    position = (10.0 + 9.0 * np.sin(time))[:, None]
    environment = Environment(place_bin_size=1.0).fit_place_grid(position)
    spike_times = [np.sort(rng.uniform(0.0, 20.0, 200))]
    marks = [rng.normal(size=(200, 2))]
    model = fit_clusterless_kde_encoding_model(
        time,
        position,
        spike_times,
        marks,
        environment=environment,
        position_std=1.0,
        waveform_std=1.0,
        block_size=10_000,
        disable_progress_bar=True,
    )
    model["mark_reference"] = fit_mark_references(marks, kernel_scale=1.0)
    if not is_local:
        model["log_occupancy"] = None  # computed from the occupancy model
    blocks = []
    original = kde_module.kde_log_density

    def record(density_model, points):
        blocks.append(density_model.block_size)
        return original(density_model, points)

    monkeypatch.setattr(kde_module, "kde_log_density", record)
    monkeypatch.setattr(clusterless_kde_module, "kde_log_density", record)
    edges = np.arange(0.0, 20.0 + 1e-9, 0.02)

    def predict(**settings):
        return np.asarray(
            predict_clusterless_kde_log_likelihood(
                time,
                position,
                spike_times,
                marks,
                time_edges=edges,
                is_local=is_local,
                **{**model, **settings},
            )
        )

    fitted = predict()
    assert set(blocks) == {10_000}
    blocks.clear()
    blocked = predict(block_size=7)
    assert set(blocks) == {7}
    assert model["occupancy_model"].block_size == 10_000
    assert {gpi.block_size for gpi in model["gpi_models"]} == {10_000}
    np.testing.assert_allclose(blocked, fitted, rtol=1e-5, atol=1e-4)
