"""Cross-model consistency tests for likelihood models.

Verifies that different likelihood estimators (KDE, GLM, GMM) produce
consistent results on the same data: correlated place fields, correlated
log-likelihoods, and agreement on basic directional properties.
"""

import numpy as np
import pytest
from scipy import stats

from non_local_detector.environment import Environment
from non_local_detector.tests.likelihoods.conftest import (
    ALGORITHMS,
    fit_registered_backends,
)

pytestmark = pytest.mark.integration

POSITION_STD = np.sqrt(12.5)
# The KDE and GMM fits compared below; the others mostly use their defaults.
FIT_PARAMS = {
    "sorted_spikes_kde": {"position_std": POSITION_STD, "block_size": 16},
    "clusterless_kde": {
        "position_std": POSITION_STD,
        "waveform_std": 1.0,
        "block_size": 8,
    },
    "clusterless_gmm": {
        "gmm_components_occupancy": 10,
        "gmm_components_joint": 10,
        "gmm_random_state": 0,
    },
    # Smooth across the 5 cm bins (the default position_std is below one bin).
    "sorted_spikes_diffusion": {"position_std": 10.0},
    "clusterless_diffusion": {"position_std": 10.0},
}
DECODE_EDGES = np.linspace(0.0, 1.0, 11)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cross_model_data():
    """Spike data with clear place fields, and every backend fit to it once.

    3 neurons with place fields centered at positions 25, 50, 75.
    Position sweeps linearly from 0 to 100 over 200 time steps; each spike
    is at a position sample's time. Tests must not mutate the fitted models.
    """
    environment = Environment(
        environment_name="test",
        place_bin_size=5.0,
        position_range=((0.0, 100.0),),
    ).fit_place_grid(
        position=np.linspace(0, 100, 201)[:, None], infer_track_interior=False
    )
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
        spike_times.append(t[selected])
        # Waveform features: 2D, tightly clustered for identifiability
        feats = np.column_stack(
            [
                rng.standard_normal(n_spikes) * 0.3 + center / 50.0,
                rng.standard_normal(n_spikes) * 0.3 - center / 50.0,
            ]
        ).astype(np.float32)
        spike_waveform_features.append(feats)

    # Float32 inputs, the working precision; with float64 inputs one cell's
    # GLM fit stops just short of its gradient tolerance and warns.
    data = {
        "position_time": t.astype(np.float32),
        "position": pos.astype(np.float32),
        "environment": environment,
        "encoding_spike_times": [times.astype(np.float32) for times in spike_times],
        "encoding_features": spike_waveform_features,
    }
    data["fitted"] = {
        name: (predict, model, is_clusterless)
        for name, predict, model, is_clusterless in fit_registered_backends(
            data, FIT_PARAMS
        )
    }
    return data


def _predict(data, name, *, time_edges=DECODE_EDGES, empty=False, is_local=False):
    """Decode the encoding spikes (or none) with backend ``name``."""
    predict, model, is_clusterless = data["fitted"][name]
    spike_times = data["encoding_spike_times"]
    features = data["encoding_features"]
    if empty:
        spike_times = [np.array([]) for _ in spike_times]
        features = [np.zeros((0, 2)) for _ in features]
    args = [data["position_time"], data["position"], spike_times]
    if is_clusterless:
        args.append(features)
    return np.asarray(predict(*args, time_edges=time_edges, is_local=is_local, **model))


def _place_fields(data, name):
    """Interior place fields, shape (n_neurons, n_interior_bins)."""
    _, model, _ = data["fitted"][name]
    interior = data["environment"].is_track_interior_.ravel()
    return np.asarray(model["place_fields"])[:, interior]


# ---------------------------------------------------------------------------
# Tests: Sorted spikes KDE vs GLM
# ---------------------------------------------------------------------------


def test_sorted_kde_vs_glm_place_field_peak_agreement(cross_model_data):
    """KDE and GLM should agree on where each neuron's place field peaks."""
    pf_kde = _place_fields(cross_model_data, "sorted_spikes_kde")
    pf_glm = _place_fields(cross_model_data, "sorted_spikes_glm")

    for i in range(pf_kde.shape[0]):
        peak_kde = np.argmax(pf_kde[i])
        peak_glm = np.argmax(pf_glm[i])
        assert abs(int(peak_kde) - int(peak_glm)) <= 2, (
            f"Neuron {i}: KDE peak at bin {peak_kde}, GLM peak at bin {peak_glm}"
        )


def test_sorted_kde_vs_glm_place_field_correlation(cross_model_data):
    """KDE and GLM place fields should be positively correlated per neuron."""
    pf_kde = _place_fields(cross_model_data, "sorted_spikes_kde")
    pf_glm = _place_fields(cross_model_data, "sorted_spikes_glm")

    for i in range(pf_kde.shape[0]):
        corr = np.corrcoef(pf_kde[i], pf_glm[i])[0, 1]
        assert corr > 0.5, f"Neuron {i}: KDE-GLM place field correlation = {corr:.3f}"


def test_sorted_kde_vs_glm_log_likelihood_correlation(cross_model_data):
    """KDE and GLM log-likelihoods should be positively correlated across positions."""
    ll_kde = _predict(cross_model_data, "sorted_spikes_kde")
    ll_glm = _predict(cross_model_data, "sorted_spikes_glm")

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


def test_clusterless_kde_vs_gmm_ground_process_correlation(cross_model_data):
    """Ground process intensity from KDE and GMM should be correlated."""
    fitted = cross_model_data["fitted"]
    gpi_kde = np.asarray(
        fitted["clusterless_kde"][1]["summed_ground_process_intensity"]
    )
    gpi_gmm = np.asarray(
        fitted["clusterless_gmm"][1]["summed_ground_process_intensity"]
    )

    # Both should have same shape
    assert gpi_kde.shape == gpi_gmm.shape

    # Check rank correlation (more robust to nonlinear scaling)
    corr, _ = stats.spearmanr(gpi_kde, gpi_gmm)
    assert corr > 0.3, f"KDE-GMM ground process Spearman correlation = {corr:.3f}"


# ---------------------------------------------------------------------------
# Tests: Universal properties across all models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["sorted_spikes_kde", "sorted_spikes_glm", "clusterless_kde", "clusterless_gmm"],
)
@pytest.mark.parametrize("bin_width", [0.02, 0.1])
def test_no_spike_likelihood_matches_expected_counts(cross_model_data, name, bin_width):
    """No-spike log likelihood is minus the fitted Hz rate times bin duration."""
    edges = np.linspace(0.0, 1.0, round(1.0 / bin_width) + 1)
    _, model, is_clusterless = cross_model_data["fitted"][name]
    if is_clusterless:
        rates = np.asarray(model["summed_ground_process_intensity"])
    else:
        interior = cross_model_data["environment"].is_track_interior_.ravel()
        rates = np.asarray(model["no_spike_part_log_likelihood"])[interior]
    assert np.all(np.isfinite(rates))
    assert np.all(rates > 0), "Fixture must have positive rates to detect zero output"

    likelihood = _predict(cross_model_data, name, time_edges=edges, empty=True)
    expected = -np.diff(edges)[:, None] * rates[None, :]
    assert likelihood.shape == expected.shape
    assert np.all(np.isfinite(likelihood)), "No-spike likelihood must be finite"
    np.testing.assert_allclose(likelihood, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("name", ALGORITHMS)
def test_local_likelihood_finite_all_models(cross_model_data, name):
    """Every registered model produces finite local log-likelihoods."""
    ll = _predict(cross_model_data, name, is_local=True)
    assert ll.shape == (len(DECODE_EDGES) - 1, 1)
    assert np.all(np.isfinite(ll))
