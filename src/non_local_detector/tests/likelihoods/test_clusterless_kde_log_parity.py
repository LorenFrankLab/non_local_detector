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

T_POS = np.linspace(0.0, 10.0, 101)
POS = np.linspace(0.0, 10.0, 101)[:, None]
T_EDGES = np.linspace(0.0, 10.0, 6)
ENC_TIMES = [np.array([2.0, 5.0, 7.5])]
ENC_FEATS = [np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)]
DEC_TIMES = [np.array([2.1, 5.2])]
DEC_FEATS = [np.array([[0.1, 0.05], [1.1, -0.9]], dtype=float)]


def _fit_both(env, weights, enc_times=None):
    """Fit the linear and log encoding models on identical inputs."""
    kw = {
        "position_time": T_POS,
        "position": POS,
        "spike_times": ENC_TIMES if enc_times is None else enc_times,
        "spike_waveform_features": ENC_FEATS,
        "environment": env,
        "weights": weights,
        "sampling_frequency": 10,
        "position_std": np.sqrt(1.0),
        "waveform_std": 1.0,
        "block_size": 8,
        "disable_progress_bar": True,
    }
    return fit_lin(**kw), fit_log(**kw)


def _predict(pred, enc, env, *, is_local, dec_times=None, with_encoding_weights=True):
    """Run predict on a fitted encoding model, returning a numpy likelihood."""
    kwargs = {
        "time": T_EDGES,
        "position_time": T_POS,
        "position": POS,
        "spike_times": DEC_TIMES if dec_times is None else dec_times,
        "spike_waveform_features": DEC_FEATS,
        "occupancy": enc["occupancy"],
        "occupancy_model": enc["occupancy_model"],
        "gpi_models": enc["gpi_models"],
        "encoding_spike_waveform_features": enc["encoding_spike_waveform_features"],
        "encoding_positions": enc["encoding_positions"],
        "environment": env,
        "mean_rates": jnp.asarray(enc["mean_rates"]),
        "summed_ground_process_intensity": enc["summed_ground_process_intensity"],
        "position_std": jnp.asarray(enc["position_std"]),
        "waveform_std": jnp.asarray(enc["waveform_std"]),
        "is_local": is_local,
        "block_size": 8,
        "disable_progress_bar": True,
    }
    if with_encoding_weights:
        kwargs["encoding_weights"] = enc["encoding_weights"]
    return np.asarray(pred(**kwargs))


def test_clusterless_log_vs_linear_parity_nonlocal(simple_1d_environment):
    env = simple_1d_environment
    enc_lin, enc_log = _fit_both(env, np.ones_like(T_POS))

    ll_lin = _predict(
        pred_lin, enc_lin, env, is_local=False, with_encoding_weights=False
    )
    ll_log = _predict(
        pred_log, enc_log, env, is_local=False, with_encoding_weights=False
    )

    # Compare shapes and values within tolerance (small numeric differences allowed).
    assert ll_lin.shape == ll_log.shape
    assert np.allclose(ll_lin, ll_log, rtol=1e-4, atol=1e-5)


def test_clusterless_log_vs_linear_parity_nonlocal_weighted(simple_1d_environment):
    """Non-uniform per-sample weights must match between the log and linear paths.

    Both modules compute the same weighted KDE (occupancy, ground process, and the
    decode-time joint mark intensity), so for identical weights the non-local log
    likelihood must agree. Guards that the log-domain module actually threads
    ``weights`` (fit) and ``encoding_weights`` (predict), not just accepts them.
    """
    env = simple_1d_environment
    # Up-weight the left half of the track; keep every weight strictly positive so
    # the two paths' occupancy/GPI KDEs stay well-defined for an exact comparison.
    weights = np.where(T_POS < 5.0, 3.0, 0.5)
    enc_lin, enc_log = _fit_both(env, weights)

    # Weights must reach the occupancy field (else they are silently ignored).
    _, enc_log_uniform = _fit_both(env, np.ones_like(T_POS))
    assert not np.allclose(
        np.asarray(enc_log["occupancy"]), np.asarray(enc_log_uniform["occupancy"])
    )

    ll_lin = _predict(pred_lin, enc_lin, env, is_local=False)
    ll_log = _predict(pred_log, enc_log, env, is_local=False)

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
    weights = np.where(T_POS < 5.0, 3.0, 0.5)
    enc_lin, enc_log = _fit_both(env, weights)

    ll_lin = _predict(pred_lin, enc_lin, env, is_local=True)
    ll_log = _predict(pred_log, enc_log, env, is_local=True)

    assert ll_lin.shape == ll_log.shape == (T_EDGES.shape[0], 1)
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
    # Left half weighted, right half zero. The single electrode fires only on the
    # right (zero-weight) half, so its interpolated encoding weights are all 0.
    weights = np.where(T_POS < 5.0, 1.0, 0.0)
    enc_times = [np.array([6.0, 7.0, 8.0])]
    enc_lin, enc_log = _fit_both(env, weights, enc_times=enc_times)

    dec_times = [np.array([6.5, 7.5])]
    ll_lin = _predict(pred_lin, enc_lin, env, is_local=True, dec_times=dec_times)
    ll_log = _predict(pred_log, enc_log, env, is_local=True, dec_times=dec_times)

    assert np.all(np.isfinite(ll_log))
    assert np.allclose(ll_lin, ll_log, rtol=1e-4, atol=1e-5), (
        f"zero-weight local log vs linear mismatch; "
        f"max|diff|={np.abs(ll_lin - ll_log).max():.3e}"
    )
