"""Cross-model consistency tests for likelihood models.

Verifies that different likelihood estimators (KDE, GLM, GMM) produce
consistent results on the same data: correlated place fields, correlated
log-likelihoods, and agreement on basic directional properties.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

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

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cross_model_env():
    """1D environment for cross-model comparisons."""
    env = Environment(
        environment_name="test",
        place_bin_size=5.0,
        position_range=((0.0, 100.0),),
    )
    pos = np.linspace(0, 100, 201)[:, None]
    return env.fit_place_grid(position=pos, infer_track_interior=False)


@pytest.fixture(scope="module")
def cross_model_data():
    """Generate spike data with clear place fields for cross-model tests.

    3 neurons with place fields centered at positions 25, 50, 75.
    Position sweeps linearly from 0 to 100 over 200 time steps.
    """
    rng = np.random.default_rng(777)
    n_time = 200
    t = np.linspace(0.0, 1.0, n_time)
    pos = np.linspace(0.0, 100.0, n_time)[:, None]

    spike_times = []
    spike_waveform_features = []
    for center in [25.0, 50.0, 75.0]:
        # Generate spikes near the preferred position
        near = np.abs(pos.squeeze() - center) < 20
        candidate_idx = np.where(near)[0]
        n_spikes = min(20, len(candidate_idx))
        selected = np.sort(rng.choice(candidate_idx, n_spikes, replace=False))
        spike_times.append(jnp.asarray(t[selected]))
        # Waveform features: 2D, tightly clustered for identifiability
        feats = np.column_stack(
            [
                rng.standard_normal(n_spikes) * 0.3 + center / 50.0,
                rng.standard_normal(n_spikes) * 0.3 - center / 50.0,
            ]
        ).astype(np.float32)
        spike_waveform_features.append(jnp.asarray(feats))

    return {
        "position_time": t,
        "position": pos,
        "spike_times": [jnp.asarray(st) for st in spike_times],
        "spike_waveform_features": spike_waveform_features,
        "sampling_frequency": float(n_time),
        "n_neurons": 3,
    }


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------


def _fit_sorted_kde(env, data):
    weights = jnp.ones(len(data["position_time"]))
    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        environment=env,
        weights=weights,
        sampling_frequency=data["sampling_frequency"],
        position_std=np.sqrt(12.5),
        block_size=16,
        disable_progress_bar=True,
    )
    return enc


def _fit_sorted_glm(env, data):
    enc = fit_sorted_spikes_glm_encoding_model(
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        environment=env,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
        sampling_frequency=data["sampling_frequency"],
        disable_progress_bar=True,
    )
    return enc


def _predict_sorted_kde(enc, env, data, is_local=False):
    t_edges = jnp.linspace(0.0, 1.0, 11)
    return predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
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


def _predict_sorted_glm(enc, env, data, is_local=False):
    t_edges = jnp.linspace(0.0, 1.0, 11)
    return predict_sorted_spikes_glm_log_likelihood(
        time=t_edges,
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        environment=env,
        coefficients=enc["coefficients"],
        emission_design_info=enc["emission_design_info"],
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        is_local=is_local,
        disable_progress_bar=True,
    )


def _fit_clusterless_kde(env, data):
    enc = fit_clusterless_kde_encoding_model(
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        environment=env,
        sampling_frequency=data["sampling_frequency"],
        position_std=np.sqrt(12.5),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )
    return enc


def _predict_clusterless_kde(enc, env, data, is_local=False):
    t_edges = jnp.linspace(0.0, 1.0, 11)
    return predict_clusterless_kde_log_likelihood(
        time=t_edges,
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
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


def _fit_clusterless_gmm(env, data):
    enc = fit_clusterless_gmm_encoding_model(
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
        environment=env,
        sampling_frequency=data["sampling_frequency"],
        gmm_components_occupancy=10,
        gmm_components_gpi=5,
        gmm_components_joint=10,
        gmm_random_state=0,
        disable_progress_bar=True,
    )
    return enc


def _predict_clusterless_gmm(enc, env, data, is_local=False):
    t_edges = jnp.linspace(0.0, 1.0, 11)
    return predict_clusterless_gmm_log_likelihood(
        time=t_edges,
        position_time=jnp.asarray(data["position_time"]),
        position=jnp.asarray(data["position"]),
        spike_times=data["spike_times"],
        spike_waveform_features=data["spike_waveform_features"],
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
# Tests: Sorted spikes KDE vs GLM
# ---------------------------------------------------------------------------


def test_sorted_kde_vs_glm_place_field_peak_agreement(
    cross_model_env, cross_model_data
):
    """KDE and GLM should agree on where each neuron's place field peaks."""
    env = cross_model_env
    data = cross_model_data

    enc_kde = _fit_sorted_kde(env, data)
    enc_glm = _fit_sorted_glm(env, data)

    pf_kde = np.asarray(enc_kde["place_fields"])
    pf_glm = np.asarray(enc_glm["place_fields"])

    # GLM returns list of arrays, KDE returns 2D array
    if isinstance(pf_glm, list):
        pf_glm = np.stack(pf_glm)

    for i in range(data["n_neurons"]):
        peak_kde = np.argmax(pf_kde[i])
        peak_glm = np.argmax(pf_glm[i])
        assert (
            abs(int(peak_kde) - int(peak_glm)) <= 2
        ), f"Neuron {i}: KDE peak at bin {peak_kde}, GLM peak at bin {peak_glm}"


def test_sorted_kde_vs_glm_place_field_correlation(cross_model_env, cross_model_data):
    """KDE and GLM place fields should be positively correlated per neuron."""
    env = cross_model_env
    data = cross_model_data

    enc_kde = _fit_sorted_kde(env, data)
    enc_glm = _fit_sorted_glm(env, data)

    pf_kde = np.asarray(enc_kde["place_fields"])
    pf_glm = np.asarray(enc_glm["place_fields"])
    if isinstance(pf_glm, list):
        pf_glm = np.stack(pf_glm)

    for i in range(data["n_neurons"]):
        corr = np.corrcoef(pf_kde[i], pf_glm[i])[0, 1]
        assert corr > 0.5, f"Neuron {i}: KDE-GLM place field correlation = {corr:.3f}"


def test_sorted_kde_vs_glm_log_likelihood_correlation(
    cross_model_env, cross_model_data
):
    """KDE and GLM log-likelihoods should be positively correlated across positions."""
    env = cross_model_env
    data = cross_model_data

    enc_kde = _fit_sorted_kde(env, data)
    enc_glm = _fit_sorted_glm(env, data)

    ll_kde = np.asarray(_predict_sorted_kde(enc_kde, env, data))
    ll_glm = np.asarray(_predict_sorted_glm(enc_glm, env, data))

    # Check correlation across position bins for each time step
    correlations = []
    for t_idx in range(ll_kde.shape[0]):
        if np.std(ll_kde[t_idx]) > 1e-10 and np.std(ll_glm[t_idx]) > 1e-10:
            corr = np.corrcoef(ll_kde[t_idx], ll_glm[t_idx])[0, 1]
            correlations.append(corr)

    mean_corr = np.mean(correlations) if correlations else 0.0
    assert mean_corr > 0.3, f"Mean KDE-GLM LL correlation = {mean_corr:.3f}"


# ---------------------------------------------------------------------------
# Tests: Clusterless KDE vs GMM
# ---------------------------------------------------------------------------


def test_clusterless_kde_vs_gmm_log_likelihood_correlation(
    cross_model_env, cross_model_data
):
    """Clusterless KDE and GMM log-likelihoods should be positively correlated."""
    env = cross_model_env
    data = cross_model_data

    enc_kde = _fit_clusterless_kde(env, data)
    enc_gmm = _fit_clusterless_gmm(env, data)

    ll_kde = np.asarray(_predict_clusterless_kde(enc_kde, env, data))
    ll_gmm = np.asarray(_predict_clusterless_gmm(enc_gmm, env, data))

    # Use rank correlation on flattened arrays (more robust than per-time-bin
    # Pearson for small data with fundamentally different density estimators)
    finite_mask = np.isfinite(ll_kde.ravel()) & np.isfinite(ll_gmm.ravel())
    corr, _ = stats.spearmanr(ll_kde.ravel()[finite_mask], ll_gmm.ravel()[finite_mask])
    assert corr > 0.0, f"KDE-GMM Spearman LL correlation = {corr:.3f}"


def test_clusterless_kde_vs_gmm_ground_process_correlation(
    cross_model_env, cross_model_data
):
    """Ground process intensity from KDE and GMM should be correlated."""
    env = cross_model_env
    data = cross_model_data

    enc_kde = _fit_clusterless_kde(env, data)
    enc_gmm = _fit_clusterless_gmm(env, data)

    gpi_kde = np.asarray(enc_kde["summed_ground_process_intensity"])
    gpi_gmm = np.asarray(enc_gmm["summed_ground_process_intensity"])

    # Both should have same shape
    assert gpi_kde.shape == gpi_gmm.shape

    # Check rank correlation (more robust to nonlinear scaling)
    corr, _ = stats.spearmanr(gpi_kde, gpi_gmm)
    assert corr > 0.3, f"KDE-GMM ground process Spearman correlation = {corr:.3f}"


# ---------------------------------------------------------------------------
# Tests: Universal properties across all models
# ---------------------------------------------------------------------------


def test_all_models_no_spike_produces_negative_likelihood(
    cross_model_env, cross_model_data
):
    """With zero decoding spikes, all models should produce LL <= 0 everywhere.

    This follows from the Poisson no-spike term: -lambda, where lambda > 0.
    """
    env = cross_model_env
    data = cross_model_data

    # Create zero-spike versions for decoding
    n = data["n_neurons"]
    empty_sorted = {**data, "spike_times": [jnp.array([]) for _ in range(n)]}
    empty_clusterless = {
        **data,
        "spike_times": [jnp.array([]) for _ in range(n)],
        "spike_waveform_features": [jnp.zeros((0, 2)) for _ in range(n)],
    }

    # Sorted KDE
    enc_kde = _fit_sorted_kde(env, data)
    ll_kde = np.asarray(_predict_sorted_kde(enc_kde, env, empty_sorted))
    finite_kde = ll_kde[np.isfinite(ll_kde)]
    assert np.all(finite_kde <= 1e-7), "Sorted KDE: no-spike LL should be <= 0"

    # Sorted GLM
    enc_glm = _fit_sorted_glm(env, data)
    ll_glm = np.asarray(_predict_sorted_glm(enc_glm, env, empty_sorted))
    finite_glm = ll_glm[np.isfinite(ll_glm)]
    assert np.all(finite_glm <= 1e-7), "Sorted GLM: no-spike LL should be <= 0"

    # Clusterless KDE
    enc_cl_kde = _fit_clusterless_kde(env, data)
    ll_cl_kde = np.asarray(_predict_clusterless_kde(enc_cl_kde, env, empty_clusterless))
    finite_cl_kde = ll_cl_kde[np.isfinite(ll_cl_kde)]
    assert np.all(finite_cl_kde <= 1e-7), "Clusterless KDE: no-spike LL should be <= 0"

    # Clusterless GMM
    enc_cl_gmm = _fit_clusterless_gmm(env, data)
    ll_cl_gmm = np.asarray(_predict_clusterless_gmm(enc_cl_gmm, env, empty_clusterless))
    finite_cl_gmm = ll_cl_gmm[np.isfinite(ll_cl_gmm)]
    assert np.all(finite_cl_gmm <= 1e-7), "Clusterless GMM: no-spike LL should be <= 0"


def test_sorted_models_kde_glm_agree_on_high_vs_low_ll_time_bins(
    cross_model_env, cross_model_data
):
    """KDE and GLM should agree on which time bins have relatively high vs low LL.

    Uses rank correlation over mean LL per time bin (averaged across positions).
    Both models use Poisson statistics, so their relative ordering of time bins
    should be consistent even though absolute magnitudes differ.
    """
    env = cross_model_env
    data = cross_model_data

    enc_kde = _fit_sorted_kde(env, data)
    enc_glm = _fit_sorted_glm(env, data)

    ll_kde = np.asarray(_predict_sorted_kde(enc_kde, env, data))
    ll_glm = np.asarray(_predict_sorted_glm(enc_glm, env, data))

    # Mean LL per time bin (across position bins)
    mean_kde = ll_kde.mean(axis=1)
    mean_glm = ll_glm.mean(axis=1)

    corr, _ = stats.spearmanr(mean_kde, mean_glm)
    assert corr > 0.3, f"KDE-GLM rank correlation of mean LL per time bin = {corr:.3f}"


def test_local_likelihood_finite_all_models(cross_model_env, cross_model_data):
    """All models should produce finite local log-likelihoods."""
    env = cross_model_env
    data = cross_model_data

    # Sorted KDE
    enc_kde = _fit_sorted_kde(env, data)
    ll_kde = np.asarray(_predict_sorted_kde(enc_kde, env, data, is_local=True))
    assert ll_kde.shape[1] == 1
    assert np.all(np.isfinite(ll_kde))

    # Sorted GLM
    enc_glm = _fit_sorted_glm(env, data)
    ll_glm = np.asarray(_predict_sorted_glm(enc_glm, env, data, is_local=True))
    assert ll_glm.shape[1] == 1
    assert np.all(np.isfinite(ll_glm))

    # Clusterless KDE
    enc_cl_kde = _fit_clusterless_kde(env, data)
    ll_cl_kde = np.asarray(
        _predict_clusterless_kde(enc_cl_kde, env, data, is_local=True)
    )
    assert ll_cl_kde.shape[1] == 1
    assert np.all(np.isfinite(ll_cl_kde))

    # Clusterless GMM
    enc_cl_gmm = _fit_clusterless_gmm(env, data)
    ll_cl_gmm = np.asarray(
        _predict_clusterless_gmm(enc_cl_gmm, env, data, is_local=True)
    )
    assert ll_cl_gmm.shape[1] == 1
    assert np.all(np.isfinite(ll_cl_gmm))
