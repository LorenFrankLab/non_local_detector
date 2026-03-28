"""Test KDE-GMM equivalence when GMM parameters match KDE bandwidths.

A KDE with N training samples and bandwidth σ is mathematically equivalent to
a GMM with N components, equal weights (1/N), means at the training sample
positions, and diagonal covariance diag(σ²). This test constructs such a
bandwidth-matched GMM and verifies the likelihoods match.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_gmm import (
    _gmm_logp,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.common import get_position_at_time
from non_local_detector.likelihoods.gmm import (
    GaussianMixtureModel,
    _compute_precision_cholesky,
)


def _make_kde_matched_gmm(
    means: np.ndarray,
    bandwidths: np.ndarray,
) -> GaussianMixtureModel:
    """Construct a GMM whose density equals a KDE with given bandwidths.

    Parameters
    ----------
    means : np.ndarray, shape (n_components, n_features)
        Component means (= training sample positions).
    bandwidths : np.ndarray, shape (n_features,)
        KDE bandwidth per feature dimension.

    Returns
    -------
    GaussianMixtureModel
        A GMM with equal weights, means at the sample positions,
        and diagonal covariance matching the KDE bandwidth.
    """
    n_components, n_features = means.shape
    means = jnp.asarray(means)
    bandwidths = jnp.asarray(bandwidths)

    weights = jnp.ones(n_components) / n_components
    # Full covariance: (K, D, D) diagonal matrices
    covariances = jnp.zeros((n_components, n_features, n_features))
    for i in range(n_components):
        covariances = covariances.at[i].set(jnp.diag(bandwidths**2))

    precisions_chol = _compute_precision_cholesky(covariances, "full")

    gmm = GaussianMixtureModel(n_components=n_components, covariance_type="full")
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_chol_ = precisions_chol

    return gmm


@pytest.fixture
def equivalence_data():
    """Create test data for KDE-GMM equivalence testing."""
    rng = np.random.default_rng(42)

    n_time = 30
    dt = 0.02
    time = np.arange(n_time + 1) * dt

    position_time = np.linspace(0, time[-1], 200)
    position = np.linspace(0, 10, len(position_time))[:, None]

    n_spikes = 30
    spike_times = [np.sort(rng.uniform(time[0], time[-1], n_spikes))]
    spike_features = [rng.standard_normal((n_spikes, 2)).astype(np.float32)]

    environment = Environment(position_range=[(0, 10)])
    environment = environment.fit_place_grid(
        position=position, infer_track_interior=True
    )

    return {
        "time": time,
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
        "spike_features": spike_features,
        "environment": environment,
    }


def test_kde_gmm_density_equivalence(equivalence_data):
    """Verify that a bandwidth-matched GMM produces the same density as KDE.

    Constructs a GMM with one component per training spike, centered at the
    spike's (position, waveform) with covariance matching the KDE bandwidth.
    The GMM joint density should equal the KDE marginal density.
    """
    data = equivalence_data
    position_std = 1.5
    waveform_std = 1.0

    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    env = data["environment"]
    spike_times_jnp = [jnp.asarray(st) for st in data["spike_times"]]
    spike_features_jnp = [jnp.asarray(sf) for sf in data["spike_features"]]

    # Fit KDE encoding model
    kde_enc = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times_jnp,
        spike_waveform_features=spike_features_jnp,
        environment=env,
        position_std=position_std,
        waveform_std=waveform_std,
        disable_progress_bar=True,
    )

    is_track_interior = env.is_track_interior_.ravel()
    interior_bins = env.place_bin_centers_[is_track_interior]

    # Get the encoding positions and features for the single electrode
    enc_positions = kde_enc["encoding_positions"][0]  # (n_spikes, 1)
    enc_features = kde_enc["encoding_spike_waveform_features"][0]  # (n_spikes, 2)

    # Construct joint samples: [position, waveform_features]
    joint_means = jnp.concatenate([enc_positions, enc_features], axis=1)  # (n_spikes, 3)
    n_pos_dims = enc_positions.shape[1]
    n_mark_dims = enc_features.shape[1]

    # Bandwidth vector: [position_std, waveform_std, waveform_std]
    bandwidths = jnp.concatenate([
        jnp.asarray(kde_enc["position_std"]),
        jnp.full(n_mark_dims, waveform_std),
    ])

    # Build bandwidth-matched joint GMM (one component per spike)
    joint_gmm = _make_kde_matched_gmm(np.asarray(joint_means), np.asarray(bandwidths))

    # Build occupancy GMM from ALL position samples (not just spike positions),
    # matching how KDE fits occupancy on the full position trajectory
    occ_gmm = _make_kde_matched_gmm(
        np.asarray(position), np.asarray(kde_enc["position_std"])
    )

    # Evaluate occupancy: GMM vs KDE
    log_occ_gmm = _gmm_logp(occ_gmm, interior_bins)
    occ_kde = kde_enc["occupancy"]

    occ_gmm_linear = jnp.exp(log_occ_gmm)
    np.testing.assert_allclose(
        np.asarray(occ_gmm_linear),
        np.asarray(occ_kde),
        rtol=1e-4,
        atol=1e-6,
        err_msg="GMM occupancy density does not match KDE occupancy",
    )

    # Evaluate joint density at a few test points: pick first 5 decoding spikes
    test_features = spike_features_jnp[0][:5]  # (5, 2)

    # For each test spike, evaluate joint density at all position bins
    for i in range(test_features.shape[0]):
        mark = jnp.tile(test_features[i], (interior_bins.shape[0], 1))  # (n_bins, 2)
        eval_points = jnp.concatenate([interior_bins, mark], axis=1)  # (n_bins, 3)

        # GMM joint density
        log_joint_gmm = _gmm_logp(joint_gmm, eval_points)

        # KDE: marginal_density = (1/N) * Σ_i K_mark(m, m_i) * K_pos(x, x_i)
        # This is what estimate_log_joint_mark_intensity computes before the
        # mean_rate/occupancy division
        from non_local_detector.likelihoods.clusterless_kde import kde_distance

        pos_distance = kde_distance(
            interior_bins, enc_positions, kde_enc["position_std"]
        )  # (n_spikes, n_bins)
        mark_distance = kde_distance(
            test_features[i : i + 1], enc_features, jnp.full(n_mark_dims, waveform_std)
        )  # (n_spikes, 1)

        n_enc = enc_positions.shape[0]
        marginal_density = (mark_distance.T @ pos_distance / n_enc).squeeze()
        log_kde_density = jnp.log(marginal_density)

        np.testing.assert_allclose(
            np.asarray(log_joint_gmm),
            np.asarray(log_kde_density),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Joint density mismatch for test spike {i}",
        )


def test_kde_gmm_full_likelihood_equivalence(equivalence_data):
    """Verify end-to-end likelihood equivalence between KDE and bandwidth-matched GMM.

    Uses the full predict pipeline for both KDE and a manually constructed GMM
    encoding model, and verifies the log-likelihoods match.
    """
    data = equivalence_data
    position_std = 1.5
    waveform_std = 1.0

    position_time = jnp.asarray(data["position_time"])
    position = jnp.asarray(data["position"])
    time = jnp.asarray(data["time"])
    env = data["environment"]
    spike_times_jnp = [jnp.asarray(st) for st in data["spike_times"]]
    spike_features_jnp = [jnp.asarray(sf) for sf in data["spike_features"]]

    # Fit KDE and predict
    kde_enc = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times_jnp,
        spike_waveform_features=spike_features_jnp,
        environment=env,
        position_std=position_std,
        waveform_std=waveform_std,
        disable_progress_bar=True,
    )

    ll_kde = predict_clusterless_kde_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times_jnp,
        spike_waveform_features=spike_features_jnp,
        occupancy=kde_enc["occupancy"],
        occupancy_model=kde_enc["occupancy_model"],
        gpi_models=kde_enc["gpi_models"],
        encoding_spike_waveform_features=kde_enc["encoding_spike_waveform_features"],
        encoding_positions=kde_enc["encoding_positions"],
        environment=env,
        mean_rates=kde_enc["mean_rates"],
        summed_ground_process_intensity=kde_enc["summed_ground_process_intensity"],
        position_std=kde_enc["position_std"],
        waveform_std=kde_enc["waveform_std"],
        is_local=False,
        disable_progress_bar=True,
    )

    # Build bandwidth-matched GMM encoding model
    is_track_interior = env.is_track_interior_.ravel()
    interior_bins = env.place_bin_centers_[is_track_interior]
    enc_positions = kde_enc["encoding_positions"][0]
    enc_features = kde_enc["encoding_spike_waveform_features"][0]
    n_mark_dims = enc_features.shape[1]

    joint_means = jnp.concatenate([enc_positions, enc_features], axis=1)
    bandwidths = jnp.concatenate([
        jnp.asarray(kde_enc["position_std"]),
        jnp.full(n_mark_dims, waveform_std),
    ])

    joint_gmm = _make_kde_matched_gmm(np.asarray(joint_means), np.asarray(bandwidths))
    # Occupancy GMM: one component per position sample (matches KDE occupancy)
    occ_gmm = _make_kde_matched_gmm(
        np.asarray(position),
        np.asarray(kde_enc["position_std"]),
    )
    # GPI GMM: one component per spike position (matches KDE GPI)
    gpi_gmm = _make_kde_matched_gmm(
        np.asarray(enc_positions),
        np.asarray(kde_enc["position_std"]),
    )

    log_occupancy = _gmm_logp(occ_gmm, interior_bins)
    occupancy = jnp.exp(log_occupancy)

    # Compute GPI: mean_rate * gpi_density / occupancy
    from non_local_detector.likelihoods.clusterless_gmm import _gmm_density

    gpi_density = _gmm_density(gpi_gmm, interior_bins)
    mean_rate = kde_enc["mean_rates"][0]
    from non_local_detector.likelihoods.common import EPS

    summed_gpi = jnp.clip(
        mean_rate * jnp.where(occupancy > 0.0, gpi_density / occupancy, EPS),
        a_min=EPS,
    )

    gmm_encoding = {
        "environment": env,
        "occupancy_model": occ_gmm,
        "interior_place_bin_centers": interior_bins,
        "log_occupancy": log_occupancy,
        "gpi_models": [gpi_gmm],
        "joint_models": [joint_gmm],
        "mean_rates": jnp.asarray([mean_rate]),
        "summed_ground_process_intensity": summed_gpi,
        "disable_progress_bar": True,
    }

    ll_gmm = predict_clusterless_gmm_log_likelihood(
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times_jnp,
        spike_waveform_features=spike_features_jnp,
        **gmm_encoding,
        is_local=False,
    )

    # The likelihoods should match closely
    np.testing.assert_allclose(
        np.asarray(ll_gmm),
        np.asarray(ll_kde),
        rtol=1e-3,
        atol=1e-2,
        err_msg="End-to-end KDE and bandwidth-matched GMM likelihoods diverge",
    )
