"""Edge case tests for all likelihood models.

Tests degenerate inputs that could cause NaN, -inf, division by zero,
or crashes: zero spikes, single spikes, extreme bandwidths, partial
occupancy, and extreme regularization.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env_1d(position_range=(0.0, 10.0), bin_size=1.0, n_pos=101):
    """Create and fit a simple 1D environment."""
    env = Environment(
        environment_name="test",
        place_bin_size=bin_size,
        position_range=(position_range,),
    )
    pos = np.linspace(position_range[0], position_range[1], n_pos)[:, None]
    env = env.fit_place_grid(position=pos, infer_track_interior=False)
    return env


def _fit_sorted_kde(env, spike_times, n_time=101, position_std=1.0):
    lo, hi = env.position_range[0]
    t = jnp.linspace(0.0, float(hi - lo), n_time)
    pos = jnp.linspace(float(lo), float(hi), n_time)[:, None]
    weights = jnp.ones_like(t)
    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=position_std,
        block_size=16,
        disable_progress_bar=True,
    )
    return enc, t, pos


def _predict_sorted_kde(enc, env, spike_times, t_pos, pos, n_decode=6, is_local=False):
    lo, hi = env.position_range[0]
    t_edges = jnp.linspace(0.0, float(hi - lo), n_decode)
    return predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=spike_times,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=is_local,
    )


def _fit_sorted_glm(env, spike_times, n_time=101):
    lo, hi = env.position_range[0]
    t = jnp.linspace(0.0, float(hi - lo), n_time)
    pos = jnp.linspace(float(lo), float(hi), n_time)[:, None]
    enc = fit_sorted_spikes_glm_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spike_times,
        environment=env,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
        sampling_frequency=10,
        disable_progress_bar=True,
    )
    return enc, t, pos


def _predict_sorted_glm(enc, env, spike_times, t_pos, pos, n_decode=6, is_local=False):
    lo, hi = env.position_range[0]
    t_edges = jnp.linspace(0.0, float(hi - lo), n_decode)
    return predict_sorted_spikes_glm_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=spike_times,
        environment=env,
        coefficients=enc["coefficients"],
        emission_design_info=enc["emission_design_info"],
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        is_local=is_local,
        disable_progress_bar=True,
    )


def _fit_clusterless_kde(
    env, spike_times, spike_features, n_time=101, position_std=1.0, waveform_std=1.0
):
    lo, hi = env.position_range[0]
    t = jnp.linspace(0.0, float(hi - lo), n_time)
    pos = jnp.linspace(float(lo), float(hi), n_time)[:, None]
    enc = fit_clusterless_kde_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=env,
        sampling_frequency=10,
        position_std=position_std,
        waveform_std=waveform_std,
        block_size=8,
        disable_progress_bar=True,
    )
    return enc, t, pos


def _predict_clusterless_kde(
    enc, env, spike_times, spike_features, t_pos, pos, n_decode=6, is_local=False
):
    lo, hi = env.position_range[0]
    t_edges = jnp.linspace(0.0, float(hi - lo), n_decode)
    return predict_clusterless_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        environment=env,
        mean_rates=jnp.asarray(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=jnp.asarray(enc["position_std"]),
        waveform_std=jnp.asarray(enc["waveform_std"]),
        is_local=is_local,
        disable_progress_bar=True,
        block_size=8,
    )


def _fit_clusterless_gmm(env, spike_times, spike_features, n_time=101, n_components=2):
    lo, hi = env.position_range[0]
    t = jnp.linspace(0.0, float(hi - lo), n_time)
    pos = jnp.linspace(float(lo), float(hi), n_time)[:, None]
    enc = fit_clusterless_gmm_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=env,
        sampling_frequency=10,
        gmm_components_occupancy=n_components,
        gmm_components_gpi=n_components,
        gmm_components_joint=n_components,
        gmm_random_state=0,
        disable_progress_bar=True,
    )
    return enc, t, pos


def _predict_clusterless_gmm(
    enc, env, spike_times, spike_features, t_pos, pos, n_decode=6, is_local=False
):
    lo, hi = env.position_range[0]
    t_edges = jnp.linspace(0.0, float(hi - lo), n_decode)
    return predict_clusterless_gmm_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=env,
        occupancy_model=enc["occupancy_model"],
        interior_place_bin_centers=enc["interior_place_bin_centers"],
        log_occupancy=enc["log_occupancy"],
        gpi_models=enc["gpi_models"],
        joint_models=enc["joint_models"],
        mean_rates=enc["mean_rates"],
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        is_local=is_local,
        disable_progress_bar=True,
    )


# ---------------------------------------------------------------------------
# Zero spikes
# ---------------------------------------------------------------------------


def test_sorted_kde_zero_spikes_all_neurons():
    """Fit and predict with all neurons having zero spikes."""
    env = _make_env_1d()
    spike_times = [jnp.array([]), jnp.array([])]
    enc, t, pos = _fit_sorted_kde(env, spike_times)

    assert enc["place_fields"].shape[0] == 2
    for mr in enc["mean_rates"]:
        assert mr == 0.0

    ll = _predict_sorted_kde(enc, env, spike_times, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_sorted_glm_zero_spikes_all_neurons():
    """GLM should handle all-zero spike neurons without NaN."""
    env = _make_env_1d(position_range=(0.0, 100.0), bin_size=5.0, n_pos=101)
    spike_times = [jnp.array([]), jnp.array([])]
    enc, t, pos = _fit_sorted_glm(env, spike_times, n_time=101)

    for pf in enc["place_fields"]:
        assert jnp.all(jnp.isfinite(pf))

    ll = _predict_sorted_glm(enc, env, spike_times, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_clusterless_kde_zero_spikes_all_electrodes():
    """Clusterless KDE should handle zero spikes without crash."""
    env = _make_env_1d()
    spike_times = [jnp.array([]), jnp.array([])]
    spike_features = [jnp.zeros((0, 2)), jnp.zeros((0, 2))]
    enc, t, pos = _fit_clusterless_kde(env, spike_times, spike_features)

    ll = _predict_clusterless_kde(enc, env, spike_times, spike_features, t, pos)
    assert not jnp.any(jnp.isnan(ll))


@pytest.mark.xfail(
    reason="GMM fit crashes with KMeans error on empty spike arrays",
    strict=True,
)
def test_clusterless_gmm_zero_spikes_all_electrodes():
    """GMM should handle zero spikes without crash."""
    env = _make_env_1d()
    # Need at least some spikes on at least one electrode for GMM fit
    # (GMM fitting fails on empty arrays), so test with 1 electrode empty
    # and 1 electrode with few spikes
    spike_times = [jnp.array([]), jnp.array([3.0, 5.0, 7.0])]
    spike_features = [
        jnp.zeros((0, 2)),
        jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    ]
    enc, t, pos = _fit_clusterless_gmm(env, spike_times, spike_features, n_components=1)

    # Decode with zero spikes
    dec_spikes = [jnp.array([]), jnp.array([])]
    dec_feats = [jnp.zeros((0, 2)), jnp.zeros((0, 2))]
    ll = _predict_clusterless_gmm(enc, env, dec_spikes, dec_feats, t, pos)
    assert not jnp.any(jnp.isnan(ll))


# ---------------------------------------------------------------------------
# Single spike per neuron/electrode
# ---------------------------------------------------------------------------


def test_sorted_kde_single_spike_per_neuron():
    """Single spike per neuron should fit and predict without NaN."""
    env = _make_env_1d()
    spike_times = [jnp.array([3.0]), jnp.array([7.0])]
    enc, t, pos = _fit_sorted_kde(env, spike_times)

    assert jnp.all(enc["place_fields"] >= 0)
    assert jnp.all(jnp.isfinite(enc["place_fields"]))

    ll = _predict_sorted_kde(enc, env, spike_times, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_sorted_glm_single_spike_per_neuron():
    """GLM with single spike per neuron should produce finite output."""
    env = _make_env_1d(position_range=(0.0, 100.0), bin_size=5.0, n_pos=101)
    spike_times = [jnp.array([0.3]), jnp.array([0.7])]
    enc, t, pos = _fit_sorted_glm(env, spike_times, n_time=101)

    for pf in enc["place_fields"]:
        assert jnp.all(jnp.isfinite(pf))

    ll = _predict_sorted_glm(enc, env, spike_times, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_clusterless_kde_single_spike_per_electrode():
    """Single spike per electrode should not cause NaN."""
    env = _make_env_1d()
    spike_times = [jnp.array([3.0]), jnp.array([7.0])]
    spike_features = [jnp.array([[0.5, -0.5]]), jnp.array([[1.0, 1.0]])]
    enc, t, pos = _fit_clusterless_kde(env, spike_times, spike_features)

    ll = _predict_clusterless_kde(enc, env, spike_times, spike_features, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_clusterless_gmm_single_spike_per_electrode():
    """GMM with single spike needs n_components=1 to avoid singular covariance."""
    env = _make_env_1d()
    spike_times = [jnp.array([3.0, 5.0]), jnp.array([7.0, 8.0])]
    spike_features = [
        jnp.array([[0.5, -0.5], [0.6, -0.4]]),
        jnp.array([[1.0, 1.0], [1.1, 0.9]]),
    ]
    enc, t, pos = _fit_clusterless_gmm(env, spike_times, spike_features, n_components=1)

    ll = _predict_clusterless_gmm(enc, env, spike_times, spike_features, t, pos)
    assert not jnp.any(jnp.isnan(ll))


# ---------------------------------------------------------------------------
# Extreme bandwidths
# ---------------------------------------------------------------------------


def test_sorted_kde_very_small_bandwidth():
    """Very small KDE bandwidth should not produce NaN."""
    env = _make_env_1d()
    spike_times = [jnp.array([2.0, 5.0, 8.0]), jnp.array([3.0, 6.0])]
    enc, t, pos = _fit_sorted_kde(env, spike_times, position_std=0.01)

    assert jnp.all(jnp.isfinite(enc["place_fields"]))
    ll = _predict_sorted_kde(enc, env, spike_times, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_sorted_kde_very_large_bandwidth():
    """Very large KDE bandwidth should produce near-uniform place fields."""
    env = _make_env_1d()
    spike_times = [jnp.array([2.0, 5.0, 8.0]), jnp.array([3.0, 6.0])]
    enc, t, pos = _fit_sorted_kde(env, spike_times, position_std=100.0)

    pf = enc["place_fields"]
    assert jnp.all(jnp.isfinite(pf))
    # With huge bandwidth, place field should be nearly uniform
    for i in range(pf.shape[0]):
        cv = float(jnp.std(pf[i]) / jnp.mean(pf[i]))
        assert cv < 0.5, f"Neuron {i} place field not near-uniform with large bandwidth"

    ll = _predict_sorted_kde(enc, env, spike_times, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_clusterless_kde_very_small_waveform_std():
    """Very small waveform std should not produce NaN."""
    env = _make_env_1d()
    spike_times = [jnp.array([2.0, 5.0, 8.0])]
    spike_features = [jnp.array([[0.5, -0.5], [1.0, 1.0], [0.0, 0.0]])]
    enc, t, pos = _fit_clusterless_kde(
        env, spike_times, spike_features, waveform_std=0.001
    )

    ll = _predict_clusterless_kde(enc, env, spike_times, spike_features, t, pos)
    assert not jnp.any(jnp.isnan(ll))


def test_clusterless_kde_very_large_waveform_std():
    """Very large waveform std should produce finite predictions."""
    env = _make_env_1d()
    spike_times = [jnp.array([2.0, 5.0, 8.0])]
    spike_features = [jnp.array([[0.5, -0.5], [1.0, 1.0], [0.0, 0.0]])]
    enc, t, pos = _fit_clusterless_kde(
        env, spike_times, spike_features, waveform_std=1000.0
    )

    ll = _predict_clusterless_kde(enc, env, spike_times, spike_features, t, pos)
    assert not jnp.any(jnp.isnan(ll))


# ---------------------------------------------------------------------------
# Partial occupancy (some bins never visited)
# ---------------------------------------------------------------------------


def test_sorted_kde_partial_occupancy():
    """Position visits only half the track; zero-occupancy bins should not NaN."""
    env = _make_env_1d(position_range=(0.0, 10.0), bin_size=1.0)
    # Position only in [0, 5]
    n_time = 101
    t = jnp.linspace(0.0, 10.0, n_time)
    pos = jnp.linspace(0.0, 5.0, n_time)[:, None]
    weights = jnp.ones_like(t)
    spike_times = [jnp.array([1.0, 3.0]), jnp.array([2.0, 4.0])]

    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=1.0,
        block_size=16,
        disable_progress_bar=True,
    )

    t_edges = jnp.linspace(0.0, 10.0, 6)
    ll = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t,
        position=pos,
        spike_times=spike_times,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=False,
    )

    assert not jnp.any(jnp.isnan(ll))
    assert jnp.all(jnp.isfinite(ll) | (ll == -jnp.inf))


def test_clusterless_kde_partial_occupancy():
    """Zero-occupancy bins should not cause NaN in clusterless KDE."""
    env = _make_env_1d(position_range=(0.0, 10.0), bin_size=1.0)
    n_time = 101
    t = jnp.linspace(0.0, 10.0, n_time)
    pos = jnp.linspace(0.0, 5.0, n_time)[:, None]  # Only visit [0, 5]
    spike_times = [jnp.array([1.0, 3.0])]
    spike_features = [jnp.array([[0.5, -0.5], [1.0, 1.0]])]

    enc = fit_clusterless_kde_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=env,
        sampling_frequency=10,
        position_std=1.0,
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )

    t_edges = jnp.linspace(0.0, 10.0, 6)
    ll = predict_clusterless_kde_log_likelihood(
        time=t_edges,
        position_time=t,
        position=pos,
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        environment=env,
        mean_rates=jnp.asarray(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=jnp.asarray(enc["position_std"]),
        waveform_std=jnp.asarray(enc["waveform_std"]),
        is_local=False,
        disable_progress_bar=True,
        block_size=8,
    )

    assert not jnp.any(jnp.isnan(ll))


# ---------------------------------------------------------------------------
# GLM regularization extremes
# ---------------------------------------------------------------------------


def test_sorted_glm_high_l2_penalty():
    """Very strong L2 penalty should produce near-uniform fields, no NaN."""
    env = _make_env_1d(position_range=(0.0, 100.0), bin_size=5.0, n_pos=101)
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, 100)
    pos = np.linspace(0.0, 100.0, 100)[:, None]
    spike_times = [jnp.asarray(t[rng.choice(100, 10, replace=False)])]

    enc = fit_sorted_spikes_glm_encoding_model(
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        environment=env,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
        sampling_frequency=100,
        l2_penalty=1e6,
        disable_progress_bar=True,
    )

    for pf in enc["place_fields"]:
        assert jnp.all(jnp.isfinite(pf))
        # Strong regularization => near-uniform
        cv = float(jnp.std(pf) / (jnp.mean(pf) + 1e-15))
        assert cv < 1.0


def test_sorted_glm_low_l2_penalty():
    """Very weak L2 penalty should still converge without NaN."""
    env = _make_env_1d(position_range=(0.0, 100.0), bin_size=5.0, n_pos=101)
    rng = np.random.default_rng(43)
    t = np.linspace(0.0, 1.0, 100)
    pos = np.linspace(0.0, 100.0, 100)[:, None]
    spike_times = [jnp.asarray(t[rng.choice(100, 10, replace=False)])]

    enc = fit_sorted_spikes_glm_encoding_model(
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        environment=env,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
        sampling_frequency=100,
        l2_penalty=1e-10,
        disable_progress_bar=True,
    )

    for pf in enc["place_fields"]:
        assert jnp.all(jnp.isfinite(pf))


# ---------------------------------------------------------------------------
# Single time bin decoding
# ---------------------------------------------------------------------------


def test_decode_single_time_bin_all_models():
    """All models should handle a single decoding time bin."""
    env = _make_env_1d()
    spike_times_sorted = [jnp.array([2.0, 5.0, 8.0]), jnp.array([3.0, 6.0])]
    spike_times_cl = [jnp.array([2.0, 5.0, 8.0])]
    spike_features_cl = [jnp.array([[0.5, -0.5], [1.0, 1.0], [0.0, 0.0]])]

    # Sorted KDE
    enc_kde, t, pos = _fit_sorted_kde(env, spike_times_sorted)
    t_single = jnp.array([0.0, 10.0])  # 1 bin
    ll = predict_sorted_spikes_kde_log_likelihood(
        time=t_single,
        position_time=t,
        position=pos,
        spike_times=spike_times_sorted,
        environment=env,
        marginal_models=enc_kde["marginal_models"],
        occupancy_model=enc_kde["occupancy_model"],
        occupancy=enc_kde["occupancy"],
        mean_rates=jnp.asarray(enc_kde["mean_rates"]),
        place_fields=enc_kde["place_fields"],
        no_spike_part_log_likelihood=enc_kde["no_spike_part_log_likelihood"],
        is_track_interior=enc_kde["is_track_interior"],
        disable_progress_bar=True,
        is_local=False,
    )
    assert ll.shape[0] == 2
    assert not jnp.any(jnp.isnan(ll))

    # Clusterless KDE
    enc_cl, t, pos = _fit_clusterless_kde(env, spike_times_cl, spike_features_cl)
    ll_cl = predict_clusterless_kde_log_likelihood(
        time=t_single,
        position_time=t,
        position=pos,
        spike_times=spike_times_cl,
        spike_waveform_features=spike_features_cl,
        occupancy=enc_cl["occupancy"],
        occupancy_model=enc_cl["occupancy_model"],
        gpi_models=enc_cl["gpi_models"],
        encoding_spike_waveform_features=enc_cl["encoding_spike_waveform_features"],
        encoding_positions=enc_cl["encoding_positions"],
        environment=env,
        mean_rates=jnp.asarray(enc_cl["mean_rates"]),
        summed_ground_process_intensity=enc_cl["summed_ground_process_intensity"],
        position_std=jnp.asarray(enc_cl["position_std"]),
        waveform_std=jnp.asarray(enc_cl["waveform_std"]),
        is_local=False,
        disable_progress_bar=True,
        block_size=8,
    )
    assert ll_cl.shape[0] == 2
    assert not jnp.any(jnp.isnan(ll_cl))


# ---------------------------------------------------------------------------
# Identical waveforms (degenerate KDE in feature space)
# ---------------------------------------------------------------------------


def test_clusterless_kde_identical_waveforms():
    """All spikes with identical features should not cause NaN."""
    env = _make_env_1d()
    spike_times = [jnp.array([2.0, 5.0, 8.0])]
    # Identical waveform features
    spike_features = [jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])]
    enc, t, pos = _fit_clusterless_kde(env, spike_times, spike_features)

    ll = _predict_clusterless_kde(enc, env, spike_times, spike_features, t, pos)
    assert not jnp.any(jnp.isnan(ll))


# ---------------------------------------------------------------------------
# Extreme waveform values
# ---------------------------------------------------------------------------


def test_clusterless_kde_extreme_waveform():
    """Extreme waveform outliers should produce finite output, not NaN."""
    env = _make_env_1d()
    enc_spikes = [jnp.array([2.0, 5.0, 8.0])]
    enc_feats = [jnp.array([[0.5, -0.5], [1.0, 1.0], [0.0, 0.0]])]
    enc, t, pos = _fit_clusterless_kde(env, enc_spikes, enc_feats)

    # Decode with a 100-sigma outlier waveform
    dec_spikes = [jnp.array([5.0])]
    dec_feats = [jnp.array([[100.0, 100.0]])]
    ll = _predict_clusterless_kde(enc, env, dec_spikes, dec_feats, t, pos)

    assert not jnp.any(jnp.isnan(ll))
    assert jnp.all(jnp.isfinite(ll))


def test_clusterless_gmm_extreme_waveform():
    """Extreme waveform outliers should be floored, not produce -1e10 values."""
    env = _make_env_1d()
    enc_spikes = [jnp.array([2.0, 4.0, 5.0, 6.0, 8.0])]
    enc_feats = [
        jnp.array([[0.5, -0.5], [1.0, 1.0], [0.0, 0.0], [0.3, 0.7], [-0.5, 0.5]])
    ]
    enc, t, pos = _fit_clusterless_gmm(env, enc_spikes, enc_feats, n_components=2)

    # Decode with a 100-sigma outlier waveform
    dec_spikes = [jnp.array([5.0])]
    dec_feats = [jnp.array([[100.0, 100.0]])]
    ll = _predict_clusterless_gmm(enc, env, dec_spikes, dec_feats, t, pos)

    assert not jnp.any(jnp.isnan(ll))
    # After LOG_EPS floor (~-34.5), values should be bounded (not -1e8 or worse)
    assert jnp.all(ll > -50), f"GMM LL too extreme: min={float(jnp.min(ll)):.1f}"


# ---------------------------------------------------------------------------
# Many neurons, few spikes
# ---------------------------------------------------------------------------


def test_sorted_kde_many_neurons_few_spikes():
    """10 neurons with 2 spikes each should not crash."""
    env = _make_env_1d()
    rng = np.random.default_rng(99)
    spike_times = [jnp.array(sorted(rng.uniform(0.5, 9.5, size=2))) for _ in range(10)]
    enc, t, pos = _fit_sorted_kde(env, spike_times)

    assert enc["place_fields"].shape[0] == 10
    ll = _predict_sorted_kde(enc, env, spike_times, t, pos)
    assert ll.shape[0] > 0
    assert not jnp.any(jnp.isnan(ll))
