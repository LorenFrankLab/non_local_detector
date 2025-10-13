import numpy as np
import jax.numpy as jnp

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model as fit_lin,
    predict_clusterless_kde_log_likelihood as pred_lin,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    fit_clusterless_kde_encoding_model as fit_log,
    predict_clusterless_kde_log_likelihood as pred_log,
)


def make_env_1d():
    env = Environment(environment_name="line", place_bin_size=1.0, position_range=((0.0, 10.0),))
    dummy_pos = np.linspace(0.0, 10.0, 11)[:, None]
    env = env.fit_place_grid(position=dummy_pos, infer_track_interior=False)
    return env


def test_clusterless_log_vs_linear_parity_nonlocal():
    env = make_env_1d()
    t_pos = np.linspace(0.0, 10.0, 101)
    pos = np.linspace(0.0, 10.0, 101)[:, None]
    weights = np.ones_like(t_pos)

    enc_times = [np.array([2.0, 5.0, 7.5])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]

    enc_lin = fit_lin(
        position_time=t_pos,
        position=pos,
        spike_times=enc_times,
        spike_waveform_features=enc_feats,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )
    enc_log = fit_log(
        position_time=t_pos,
        position=pos,
        spike_times=enc_times,
        spike_waveform_features=enc_feats,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )

    t_edges = np.linspace(0.0, 10.0, 6)
    dec_times = [np.array([2.1, 5.2])]
    dec_feats = [np.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]

    ll_lin = pred_lin(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_times,
        spike_waveform_features=dec_feats,
        occupancy=enc_lin["occupancy"],
        occupancy_model=enc_lin["occupancy_model"],
        gpi_models=enc_lin["gpi_models"],
        encoding_spike_waveform_features=enc_lin["encoding_spike_waveform_features"],
        encoding_positions=enc_lin["encoding_positions"],
        encoding_spike_weights=enc_lin["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.asarray(enc_lin["mean_rates"]),
        summed_ground_process_intensity=enc_lin["summed_ground_process_intensity"],
        position_std=jnp.asarray(enc_lin["position_std"]),
        waveform_std=jnp.asarray(enc_lin["waveform_std"]),
        is_local=False,
        block_size=8,
        disable_progress_bar=True,
    )

    ll_log = pred_log(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_times,
        spike_waveform_features=dec_feats,
        occupancy=enc_log["occupancy"],
        occupancy_model=enc_log["occupancy_model"],
        gpi_models=enc_log["gpi_models"],
        encoding_spike_waveform_features=enc_log["encoding_spike_waveform_features"],
        encoding_positions=enc_log["encoding_positions"],
        encoding_spike_weights=enc_log["encoding_spike_weights"],
        environment=env,
        mean_rates=jnp.asarray(enc_log["mean_rates"]),
        summed_ground_process_intensity=enc_log["summed_ground_process_intensity"],
        position_std=jnp.asarray(enc_log["position_std"]),
        waveform_std=jnp.asarray(enc_log["waveform_std"]),
        is_local=False,
        block_size=8,
        disable_progress_bar=True,
    )

    # Compare shapes and values within tolerance
    assert ll_lin.shape == ll_log.shape
    # Allow small numeric differences; focus on relative closeness
    assert np.allclose(np.asarray(ll_lin), np.asarray(ll_log), rtol=1e-4, atol=1e-5)

