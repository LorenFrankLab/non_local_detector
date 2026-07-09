import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model as fit_lin,
)
from non_local_detector.likelihoods.clusterless_kde import (
    predict_clusterless_kde_log_likelihood as pred_lin,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    fit_clusterless_kde_encoding_model as fit_log,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    predict_clusterless_kde_log_likelihood as pred_log,
)


def test_clusterless_log_vs_linear_parity_nonlocal(simple_1d_environment):
    env = simple_1d_environment
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


def test_clusterless_log_vs_linear_parity_nonlocal_weighted(simple_1d_environment):
    """Non-uniform per-sample weights must match between the log and linear paths.

    Both modules compute the same weighted KDE (occupancy, ground process, and the
    decode-time joint mark intensity), so for identical weights the non-local log
    likelihood must agree. Guards that the log-domain module actually threads
    ``weights`` (fit) and ``encoding_weights`` (predict), not just accepts them.
    """
    env = simple_1d_environment
    t_pos = np.linspace(0.0, 10.0, 101)
    pos = np.linspace(0.0, 10.0, 101)[:, None]
    # Up-weight the left half of the track; keep every weight strictly positive so
    # the two paths' occupancy/GPI KDEs stay well-defined for an exact comparison.
    weights = np.where(t_pos < 5.0, 3.0, 0.5)

    enc_times = [np.array([2.0, 5.0, 7.5])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]

    fit_kwargs = {
        "position_time": t_pos,
        "position": pos,
        "spike_times": enc_times,
        "spike_waveform_features": enc_feats,
        "environment": env,
        "weights": weights,
        "sampling_frequency": 10,
        "position_std": np.sqrt(1.0),
        "waveform_std": 1.0,
        "block_size": 8,
        "disable_progress_bar": True,
    }
    enc_lin = fit_lin(**fit_kwargs)
    enc_log = fit_log(**fit_kwargs)

    # Weights must reach the occupancy field (else they are silently ignored).
    assert not np.allclose(
        np.asarray(enc_log["occupancy"]),
        np.asarray(
            fit_log(**{**fit_kwargs, "weights": np.ones_like(t_pos)})["occupancy"]
        ),
    )

    t_edges = np.linspace(0.0, 10.0, 6)
    dec_times = [np.array([2.1, 5.2])]
    dec_feats = [np.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]

    def _predict(pred, enc):
        return np.asarray(
            pred(
                time=t_edges,
                position_time=t_pos,
                position=pos,
                spike_times=dec_times,
                spike_waveform_features=dec_feats,
                occupancy=enc["occupancy"],
                occupancy_model=enc["occupancy_model"],
                gpi_models=enc["gpi_models"],
                encoding_spike_waveform_features=enc[
                    "encoding_spike_waveform_features"
                ],
                encoding_positions=enc["encoding_positions"],
                encoding_weights=enc["encoding_weights"],
                environment=env,
                mean_rates=jnp.asarray(enc["mean_rates"]),
                summed_ground_process_intensity=enc["summed_ground_process_intensity"],
                position_std=jnp.asarray(enc["position_std"]),
                waveform_std=jnp.asarray(enc["waveform_std"]),
                is_local=False,
                block_size=8,
                disable_progress_bar=True,
            )
        )

    ll_lin = _predict(pred_lin, enc_lin)
    ll_log = _predict(pred_log, enc_log)

    assert ll_lin.shape == ll_log.shape
    assert np.allclose(ll_lin, ll_log, rtol=1e-4, atol=1e-5), (
        f"weighted log vs linear mismatch; max|diff|={np.abs(ll_lin - ll_log).max():.3e}"
    )


def test_clusterless_log_vs_linear_parity_local_weighted(simple_1d_environment):
    """Local (at-position) log likelihood must match the linear path under weights.

    Mirrors the non-local weighted parity but with ``is_local=True``, exercising the
    weighted local marginal-density KDE (``block_log_kde`` weights) rather than the
    joint-mark-intensity paths.
    """
    env = simple_1d_environment
    t_pos = np.linspace(0.0, 10.0, 101)
    pos = np.linspace(0.0, 10.0, 101)[:, None]
    weights = np.where(t_pos < 5.0, 3.0, 0.5)

    enc_times = [np.array([2.0, 5.0, 7.5])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]

    fit_kwargs = {
        "position_time": t_pos,
        "position": pos,
        "spike_times": enc_times,
        "spike_waveform_features": enc_feats,
        "environment": env,
        "weights": weights,
        "sampling_frequency": 10,
        "position_std": np.sqrt(1.0),
        "waveform_std": 1.0,
        "block_size": 8,
        "disable_progress_bar": True,
    }
    enc_lin = fit_lin(**fit_kwargs)
    enc_log = fit_log(**fit_kwargs)

    t_edges = np.linspace(0.0, 10.0, 6)
    dec_times = [np.array([2.1, 5.2])]
    dec_feats = [np.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]

    def _predict_local(pred, enc):
        return np.asarray(
            pred(
                time=t_edges,
                position_time=t_pos,
                position=pos,
                spike_times=dec_times,
                spike_waveform_features=dec_feats,
                occupancy=enc["occupancy"],
                occupancy_model=enc["occupancy_model"],
                gpi_models=enc["gpi_models"],
                encoding_spike_waveform_features=enc[
                    "encoding_spike_waveform_features"
                ],
                encoding_positions=enc["encoding_positions"],
                encoding_weights=enc["encoding_weights"],
                environment=env,
                mean_rates=jnp.asarray(enc["mean_rates"]),
                summed_ground_process_intensity=enc["summed_ground_process_intensity"],
                position_std=jnp.asarray(enc["position_std"]),
                waveform_std=jnp.asarray(enc["waveform_std"]),
                is_local=True,
                block_size=8,
                disable_progress_bar=True,
            )
        )

    ll_lin = _predict_local(pred_lin, enc_lin)
    ll_log = _predict_local(pred_log, enc_log)

    assert ll_lin.shape == ll_log.shape == (t_edges.shape[0], 1)
    assert np.allclose(ll_lin, ll_log, rtol=1e-4, atol=1e-5), (
        f"weighted local log vs linear mismatch; "
        f"max|diff|={np.abs(ll_lin - ll_log).max():.3e}"
    )


def test_clusterless_local_all_zero_weight_electrode_matches_linear(
    simple_1d_environment,
):
    """A fully de-weighted electrode's local likelihood must floor like the linear path.

    When an electrode's encoding spikes all fall in a zero-weight region its mean rate
    is 0, so the linear path's ``safe_log(mean_rate * density / occ)`` floors every
    spike contribution at ``LOG_EPS``. The log path must do the same: without flooring
    the assembled ``log_mean_rate + log_marginal_density - log_occupancy`` it emits
    values that diverge from the linear path by several nats.
    """
    env = simple_1d_environment
    t_pos = np.linspace(0.0, 10.0, 101)
    pos = np.linspace(0.0, 10.0, 101)[:, None]
    # Left half weighted, right half zero. The single electrode fires only on the
    # right (zero-weight) half, so its interpolated encoding weights are all 0.
    weights = np.where(t_pos < 5.0, 1.0, 0.0)

    enc_times = [np.array([6.0, 7.0, 8.0])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]

    fit_kwargs = {
        "position_time": t_pos,
        "position": pos,
        "spike_times": enc_times,
        "spike_waveform_features": enc_feats,
        "environment": env,
        "weights": weights,
        "sampling_frequency": 10,
        "position_std": np.sqrt(1.0),
        "waveform_std": 1.0,
        "block_size": 8,
        "disable_progress_bar": True,
    }
    enc_lin = fit_lin(**fit_kwargs)
    enc_log = fit_log(**fit_kwargs)

    t_edges = np.linspace(0.0, 10.0, 6)
    dec_times = [np.array([6.5, 7.5])]
    dec_feats = [np.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]

    def _predict_local(pred, enc):
        return np.asarray(
            pred(
                time=t_edges,
                position_time=t_pos,
                position=pos,
                spike_times=dec_times,
                spike_waveform_features=dec_feats,
                occupancy=enc["occupancy"],
                occupancy_model=enc["occupancy_model"],
                gpi_models=enc["gpi_models"],
                encoding_spike_waveform_features=enc[
                    "encoding_spike_waveform_features"
                ],
                encoding_positions=enc["encoding_positions"],
                encoding_weights=enc["encoding_weights"],
                environment=env,
                mean_rates=jnp.asarray(enc["mean_rates"]),
                summed_ground_process_intensity=enc["summed_ground_process_intensity"],
                position_std=jnp.asarray(enc["position_std"]),
                waveform_std=jnp.asarray(enc["waveform_std"]),
                is_local=True,
                block_size=8,
                disable_progress_bar=True,
            )
        )

    ll_lin = _predict_local(pred_lin, enc_lin)
    ll_log = _predict_local(pred_log, enc_log)

    assert np.all(np.isfinite(ll_log))
    assert np.allclose(ll_lin, ll_log, rtol=1e-4, atol=1e-5), (
        f"zero-weight local log vs linear mismatch; "
        f"max|diff|={np.abs(ll_lin - ll_log).max():.3e}"
    )
