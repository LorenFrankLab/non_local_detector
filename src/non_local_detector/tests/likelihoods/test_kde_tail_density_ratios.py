"""Equal KDEs have unit density ratio even when their densities underflow."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    fit_clusterless_kde_encoding_model,
    fit_clusterless_kde_log_encoding_model,
    fit_sorted_spikes_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
    predict_clusterless_kde_log_log_likelihood,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import KDEModel, log_density_intensity
from non_local_detector.likelihoods.streamed_kde import (
    _streamed_joint_mark_log_intensity,
)

pytestmark = pytest.mark.unit

MODELS = [
    (
        fit_sorted_spikes_kde_encoding_model,
        predict_sorted_spikes_kde_log_likelihood,
        False,
    ),
    (fit_clusterless_kde_encoding_model, predict_clusterless_kde_log_likelihood, True),
    (
        fit_clusterless_kde_log_encoding_model,
        predict_clusterless_kde_log_log_likelihood,
        True,
    ),
]


@pytest.mark.parametrize("fit,predict,clusterless", MODELS)
@pytest.mark.parametrize("is_local", [False, True])
def test_equal_occupancy_and_spike_densities_keep_the_mean_rate(
    fit, predict, clusterless, is_local
):
    time = np.arange(5.0)
    position = np.zeros((5, 1))
    env = Environment(
        place_bin_size=5.0, position_range=((0.0, 200.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    args = [time, position, [time], env]
    kwargs = {"position_std": 1.0, "disable_progress_bar": True}
    if clusterless:
        args.insert(3, [np.zeros((5, 1))])
        kwargs["waveform_std"] = 1.0
    model = fit(*args, **kwargs)
    decode_args = [time, np.full_like(position, 20.0), [np.empty(0)]]
    if clusterless:
        decode_args.append([np.empty((0, 1))])
    score = predict(
        *decode_args, time_edges=np.array([0.0, 1.0]), is_local=is_local, **model
    )
    np.testing.assert_allclose(score, -1.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fit,predict,clusterless", MODELS[1:])
@pytest.mark.parametrize("is_local", [False, True])
def test_mark_event_term_survives_small_position_densities(
    fit, predict, clusterless, is_local
):
    time = np.arange(5.0)
    position = np.zeros((5, 1))
    env = Environment(
        place_bin_size=5.0, position_range=((0.0, 200.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    model = fit(
        time,
        position,
        [time],
        [np.zeros((5, 1))],
        env,
        position_std=1.0,
        waveform_std=1.0,
        disable_progress_bar=True,
    )
    common = dict(time_edges=np.array([0.0, 1.0]), is_local=is_local, **model)
    with_spike = predict(
        time,
        np.full_like(position, 20.0),
        [np.array([0.5])],
        [np.zeros((1, 1))],
        **common,
    )
    baseline = predict(
        time, np.full_like(position, 20.0), [np.empty(0)], [np.empty((0, 1))], **common
    )
    expected = -0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(
        np.asarray(with_spike - baseline), expected, rtol=1e-5, atol=2e-5
    )


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("n_features", [1, 9])
@pytest.mark.parametrize("mode", ["linear_tiled", "log_chunked", "log_streaming"])
def test_tiled_mark_ratios_match_the_gaussian_reference(mode, n_features, is_local):
    time = np.arange(5.0)
    position = np.zeros((5, 1))
    env = Environment(
        place_bin_size=5.0, position_range=((0.0, 200.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    fit, predict, _ = MODELS[1 if mode == "linear_tiled" else 2]
    options = (
        {"encoding_block_size": 2, "position_block_size": 7}
        if mode == "linear_tiled"
        else {
            "enc_tile_size": 2,
            "pos_tile_size": 7,
            "use_streaming": mode == "log_streaming",
        }
    )
    model = fit(
        time,
        position,
        [time],
        [np.zeros((5, n_features))],
        env,
        position_std=1.0,
        waveform_std=1.0,
        block_size=2,
        disable_progress_bar=True,
        **options,
    )
    common = dict(time_edges=np.array([0.0, 1.0]), is_local=is_local, **model)
    baseline = predict(
        time,
        np.full_like(position, 200.0),
        [np.empty(0)],
        [np.empty((0, n_features))],
        **common,
    )
    with_spike = predict(
        time,
        np.full_like(position, 200.0),
        [np.array([0.5])],
        [np.zeros((1, n_features))],
        **common,
    )
    np.testing.assert_allclose(baseline, -1.0, rtol=1e-5, atol=1e-5)
    expected = -0.5 * n_features * np.log(2 * np.pi)
    np.testing.assert_allclose(with_spike - baseline, expected, rtol=1e-5, atol=2e-5)


def test_log_ratio_keeps_tiny_rate_and_reports_invalid_densities():
    log_rate = log_density_intensity(1e-30, jnp.asarray([95.0]), jnp.asarray([0.0]))
    np.testing.assert_allclose(np.exp(log_rate), 1e-30 * np.exp(95.0), rtol=1e-5)
    assert np.isneginf(
        log_density_intensity(0.0, jnp.asarray(-np.inf), jnp.asarray(-np.inf))
    )
    assert np.isnan(log_density_intensity(1.0, jnp.asarray(np.nan), jnp.asarray(0.0)))
    assert np.isnan(log_density_intensity(1.0, jnp.asarray(0.0), jnp.asarray(np.nan)))


def test_tiled_tail_ratio_has_finite_correct_gradient_with_zero_weight_prefix():
    def score(mark):
        return _streamed_joint_mark_log_intensity(
            mark.reshape(1, 1),
            jnp.array([[4.0], [4.0], [0.0]]),
            jnp.array([[100.0], [100.0], [0.0]]),
            jnp.array([[20.0]]),
            jnp.ones(1),
            jnp.ones(1),
            jnp.zeros(1),
            1.0,
            jnp.array([0.0, 0.0, 1.0]),
            encoding_tile_size=2,
            position_tile_size=1,
            decoding_tile_size=1,
            log_occupancy=jnp.asarray([-200.0 - 0.5 * np.log(2 * np.pi)]),
        )[0, 0]

    value, gradient = jax.value_and_grad(score)(jnp.asarray(0.2))
    np.testing.assert_allclose(
        value, -0.02 - 0.5 * np.log(2 * np.pi), rtol=1e-5, atol=2e-5
    )
    np.testing.assert_allclose(gradient, -0.2, rtol=1e-5, atol=1e-6)


def test_sorted_public_local_path_preserves_nan_position_diagnostic():
    model = KDEModel(std=1.0, block_size=2).fit(jnp.array([[0.0], [1.0]]))
    result = predict_sorted_spikes_kde_log_likelihood(
        position_time=np.array([0.0, 1.0]),
        position=np.full((2, 1), np.nan),
        spike_times=[np.empty(0)],
        environment=None,
        marginal_models=[model],
        occupancy_model=model,
        occupancy=jnp.ones(1),
        mean_rates=[1.0],
        place_fields=jnp.ones((1, 1)),
        no_spike_part_log_likelihood=jnp.ones(1),
        is_track_interior=np.ones(1, bool),
        is_local=True,
        time_edges=np.array([0.0, 1.0]),
        disable_progress_bar=True,
    )
    assert result.shape == (1, 1)
    assert np.isnan(np.asarray(result)[0, 0])
