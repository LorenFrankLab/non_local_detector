import numpy as np
import jax.numpy as jnp
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
    get_spike_time_bin_ind,
    kde_distance,
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import get_position_at_time, safe_divide


def rng(seed=0):
    return np.random.default_rng(seed)


def make_simple_env_1d():
    env = Environment(environment_name="line", place_bin_size=1.0, position_range=((0.0, 10.0),))
    dummy_pos = np.linspace(0.0, 10.0, 11)[:, None]
    env = env.fit_place_grid(position=dummy_pos, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    return env


def test_get_spike_time_bin_ind_right_edge_last_bin():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    spikes = np.array([0.0, 0.5, 2.0, 3.0])
    inds = get_spike_time_bin_ind(spikes, edges)
    # bins: [0,1), [1,2), [2,3]; right-edge 3.0 -> last bin index 2
    assert inds.tolist() == [0, 0, 2, 2]


def test_kde_distance_shapes_and_values():
    r = rng(2)
    samples = r.normal(size=(7, 2))
    eval_points = r.normal(size=(5, 2))
    std = jnp.array([1.0, 0.5])
    dist = kde_distance(jnp.asarray(eval_points), jnp.asarray(samples), std)
    assert dist.shape == (samples.shape[0], eval_points.shape[0])
    assert jnp.all(dist >= 0)


def test_block_estimate_log_joint_mark_intensity_matches_unblocked():
    r = rng(3)
    dec_feats = jnp.asarray(r.normal(size=(9, 2)))
    enc_feats = jnp.asarray(r.normal(size=(15, 2)))
    weights = jnp.asarray(r.uniform(0.5, 1.5, size=(enc_feats.shape[0],)))
    wstd = jnp.array([1.0, 1.0])
    occupancy = jnp.asarray(r.uniform(0.1, 1.0, size=(6,)))
    mean_rate = 3.0
    pos_dist = jnp.asarray(r.uniform(0.0, 1.0, size=(enc_feats.shape[0], occupancy.shape[0])))

    base = estimate_log_joint_mark_intensity(
        dec_feats, enc_feats, weights, wstd, occupancy, mean_rate, pos_dist
    )
    blk = block_estimate_log_joint_mark_intensity(
        dec_feats, enc_feats, weights, wstd, occupancy, mean_rate, pos_dist, block_size=4
    )
    assert base.shape == blk.shape
    assert jnp.allclose(base, blk, rtol=1e-5, atol=1e-7)


def test_fit_and_predict_clusterless_kde_minimal():
    env = make_simple_env_1d()
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    weights = jnp.ones_like(t_pos)

    # one electrode: few encoding spikes with 2D waveform features
    enc_spike_times = jnp.array([2.0, 5.0, 7.5])
    enc_feats = jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)

    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times],
        spike_waveform_features=[enc_feats],
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )

    for key in (
        "occupancy",
        "occupancy_model",
        "gpi_models",
        "encoding_spike_waveform_features",
        "encoding_positions",
        "encoding_spike_weights",
        "mean_rates",
        "summed_ground_process_intensity",
    ):
        assert key in encoding

    # decoding: 2 spikes within time edges with features near encoding cloud
    t_edges = jnp.linspace(0.0, 10.0, 6)
    dec_spike_times = [jnp.array([2.1, 5.2])]
    dec_feats = [jnp.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]

    ll_nonlocal = predict_clusterless_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        spike_waveform_features=dec_feats,
        occupancy=encoding["occupancy"],
        occupancy_model=encoding["occupancy_model"],
        gpi_models=encoding["gpi_models"],
        encoding_spike_waveform_features=encoding["encoding_spike_waveform_features"],
        encoding_positions=encoding["encoding_positions"],
        encoding_spike_weights=encoding["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=jnp.asarray(encoding["position_std"]),
        waveform_std=jnp.asarray(encoding["waveform_std"]),
        is_local=False,
        block_size=8,
        disable_progress_bar=True,
    )
    # shape: (n_time, n_interior_bins)
    assert ll_nonlocal.shape[0] == t_edges.shape[0]
    assert ll_nonlocal.ndim == 2 and ll_nonlocal.shape[1] > 0
    assert jnp.all(jnp.isfinite(ll_nonlocal))

    ll_local = predict_clusterless_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        spike_waveform_features=dec_feats,
        occupancy=encoding["occupancy"],
        occupancy_model=encoding["occupancy_model"],
        gpi_models=encoding["gpi_models"],
        encoding_spike_waveform_features=encoding["encoding_spike_waveform_features"],
        encoding_positions=encoding["encoding_positions"],
        encoding_spike_weights=encoding["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=jnp.asarray(encoding["position_std"]),
        waveform_std=jnp.asarray(encoding["waveform_std"]),
        is_local=True,
        block_size=8,
        disable_progress_bar=True,
    )
    assert ll_local.shape == (t_edges.shape[0], 1)
    assert jnp.all(jnp.isfinite(ll_local))


def test_clusterless_local_zero_spikes_equals_negative_gpi_sum():
    env = make_simple_env_1d()
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    weights = jnp.ones_like(t_pos)

    enc_spike_times = jnp.array([2.0, 5.0, 7.5])
    enc_feats = jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)

    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times],
        spike_waveform_features=[enc_feats],
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )

    t_edges = jnp.linspace(0.0, 10.0, 6)
    empty_spikes = [jnp.array([])]
    empty_feats = [jnp.zeros((0, 2))]
    ll_local = predict_clusterless_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=empty_spikes,
        spike_waveform_features=empty_feats,
        occupancy=encoding["occupancy"],
        occupancy_model=encoding["occupancy_model"],
        gpi_models=encoding["gpi_models"],
        encoding_spike_waveform_features=encoding["encoding_spike_waveform_features"],
        encoding_positions=encoding["encoding_positions"],
        encoding_spike_weights=encoding["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=jnp.asarray(encoding["position_std"]),
        waveform_std=jnp.asarray(encoding["waveform_std"]),
        is_local=True,
        block_size=8,
        disable_progress_bar=True,
    )

    interpolated_position = get_position_at_time(t_pos, pos, t_edges, env)
    occupancy_at_time = encoding["occupancy_model"].predict(interpolated_position)
    expected = -encoding["mean_rates"][0] * safe_divide(
        encoding["gpi_models"][0].predict(interpolated_position), occupancy_at_time
    )
    expected = jnp.expand_dims(expected, axis=1)
    assert jnp.allclose(ll_local, expected, rtol=1e-5, atol=1e-6)


def test_get_spike_time_bin_ind_unsorted_and_interior_edges():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    spikes = np.array([2.0, 0.0, 1.0, 1.0, 0.5])  # unsorted, includes interior edges 1.0
    inds = get_spike_time_bin_ind(spikes, edges)
    # bins: [0,1), [1,2), [2,3]; interior edge 1.0 -> bin 1 (right side)
    assert inds.tolist() == [2, 0, 1, 1, 0]


def test_fit_clusterless_kde_raises_without_place_grid():
    env = Environment(environment_name="nofit")
    t_pos = jnp.linspace(0.0, 10.0, 11)
    pos = jnp.linspace(0.0, 10.0, 11)[:, None]
    with pytest.raises(ValueError):
        fit_clusterless_kde_encoding_model(
            position_time=t_pos,
            position=pos,
            spike_times=[jnp.array([2.0])],
            spike_waveform_features=[jnp.array([[0.0, 1.0]])],
            environment=env,
        )


def test_encoding_spike_weights_and_mean_rates_match_interpolation():
    env = make_simple_env_1d()
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    # weights ramp from 0 to 1
    weights = (t_pos - t_pos.min()) / (t_pos.max() - t_pos.min())

    enc_spike_times = jnp.array([1.0, 3.0, 9.0])
    enc_feats = jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)

    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times],
        spike_waveform_features=[enc_feats],
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )

    # Interpolated weights at spike times should match returned encoding weights
    expected_w = np.interp(enc_spike_times, t_pos, weights)
    got_w = np.asarray(encoding["encoding_spike_weights"][0])
    assert np.allclose(got_w, expected_w, rtol=1e-7, atol=1e-9)

    # Mean rate = sum(weights_at_spikes) / sum(all weights)
    expected_mean = expected_w.sum() / float(weights.sum())
    assert np.allclose(float(encoding["mean_rates"][0]), expected_mean, rtol=1e-7, atol=1e-9)
