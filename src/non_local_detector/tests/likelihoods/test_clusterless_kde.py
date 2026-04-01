import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    get_spike_time_bin_ind,
    kde_distance,
    predict_clusterless_kde_log_likelihood,
)


def rng(seed=0):
    return np.random.default_rng(seed)


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
    wstd = jnp.array([1.0, 1.0])
    occupancy = jnp.asarray(r.uniform(0.1, 1.0, size=(6,)))
    mean_rate = 3.0
    pos_dist = jnp.asarray(
        r.uniform(0.0, 1.0, size=(enc_feats.shape[0], occupancy.shape[0]))
    )

    base = estimate_log_joint_mark_intensity(
        dec_feats, enc_feats, wstd, occupancy, mean_rate, pos_dist
    )
    blk = block_estimate_log_joint_mark_intensity(
        dec_feats,
        enc_feats,
        wstd,
        occupancy,
        mean_rate,
        pos_dist,
        block_size=4,
    )
    assert base.shape == blk.shape
    assert jnp.allclose(base, blk, rtol=1e-5, atol=1e-7)


def test_fit_and_predict_clusterless_kde_minimal(simple_1d_environment):
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]

    # one electrode: few encoding spikes with 2D waveform features
    enc_spike_times = jnp.array([2.0, 5.0, 7.5])
    enc_feats = jnp.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)

    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times],
        spike_waveform_features=[enc_feats],
        environment=env,
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


def test_get_spike_time_bin_ind_unsorted_and_interior_edges():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    spikes = np.array(
        [2.0, 0.0, 1.0, 1.0, 0.5]
    )  # unsorted, includes interior edges 1.0
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


def test_clusterless_kde_varying_electrode_feature_counts(simple_1d_environment):
    """Test that electrodes with different numbers of waveform features work correctly.

    This tests the fix for the bug where scalar waveform_std was expanded based on
    the first electrode's feature count, causing failures when other electrodes had
    different numbers of features (e.g., 3 vs 4 features per tetrode).
    """
    env = simple_1d_environment
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]

    # Electrode 0: 4 waveform features (typical tetrode)
    enc_spike_times_0 = jnp.array([2.0, 5.0, 7.5])
    enc_feats_0 = jnp.array(
        [[0.0, 0.0, 0.1, 0.2], [1.0, -1.0, 0.5, -0.5], [0.5, 0.5, 0.3, 0.1]],
        dtype=float,
    )

    # Electrode 1: 3 waveform features (e.g., one bad channel)
    enc_spike_times_1 = jnp.array([3.0, 6.0, 8.0])
    enc_feats_1 = jnp.array(
        [[0.2, -0.1, 0.3], [0.8, 0.4, -0.2], [-0.3, 0.6, 0.1]], dtype=float
    )

    # Electrode 2: 4 waveform features again
    enc_spike_times_2 = jnp.array([1.0, 4.0])
    enc_feats_2 = jnp.array([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, 0.1, 0.2]], dtype=float)

    # Fit with scalar waveform_std - this should work for all electrodes
    encoding = fit_clusterless_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=[enc_spike_times_0, enc_spike_times_1, enc_spike_times_2],
        spike_waveform_features=[enc_feats_0, enc_feats_1, enc_feats_2],
        environment=env,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=24.0,  # Scalar - should expand per-electrode
        block_size=8,
        disable_progress_bar=True,
    )

    # Verify encoding succeeded
    assert len(encoding["encoding_spike_waveform_features"]) == 3
    assert encoding["encoding_spike_waveform_features"][0].shape[1] == 4
    assert encoding["encoding_spike_waveform_features"][1].shape[1] == 3
    assert encoding["encoding_spike_waveform_features"][2].shape[1] == 4

    # Decoding spikes with matching feature counts
    t_edges = jnp.linspace(0.0, 10.0, 6)
    dec_spike_times = [
        jnp.array([2.1, 5.2]),  # electrode 0
        jnp.array([3.1]),  # electrode 1
        jnp.array([1.1, 4.1]),  # electrode 2
    ]
    dec_feats = [
        jnp.array([[0.1, 0.05, 0.15, 0.25], [1.1, -0.9, 0.4, -0.4]], dtype=float),
        jnp.array([[0.25, -0.05, 0.35]], dtype=float),
        jnp.array([[0.15, 0.25, 0.35, 0.45], [-0.05, -0.15, 0.15, 0.25]], dtype=float),
    ]

    # Non-local prediction should work
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
        environment=env,
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=jnp.asarray(encoding["position_std"]),
        waveform_std=encoding["waveform_std"],  # Still scalar from fit
        is_local=False,
        block_size=8,
        disable_progress_bar=True,
    )

    assert ll_nonlocal.shape[0] == t_edges.shape[0]
    assert ll_nonlocal.ndim == 2 and ll_nonlocal.shape[1] > 0
    assert jnp.all(jnp.isfinite(ll_nonlocal))

    # Local prediction should also work
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
        environment=env,
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=jnp.asarray(encoding["position_std"]),
        waveform_std=encoding["waveform_std"],
        is_local=True,
        block_size=8,
        disable_progress_bar=True,
    )

    assert ll_local.shape == (t_edges.shape[0], 1)
    assert jnp.all(jnp.isfinite(ll_local))
