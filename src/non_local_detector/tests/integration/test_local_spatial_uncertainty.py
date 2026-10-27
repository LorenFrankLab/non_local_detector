"""Integration tests for local state with spatial uncertainty.

Tests the full fit/predict pipeline with local_position_std enabled,
verifying posteriors, position coordinates, and mathematical invariants.

Each detector configuration is fitted on the whole simulated recording once
per module and decodes the same 16 s window, which contains a replay event;
tests only read the shared results.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.models import NonLocalClusterlessDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

pytestmark = pytest.mark.integration

# 6.0-22.0 s at 1500 Hz: covers the replay events at 6.5 and 19.5 s.
DECODE_WINDOW = slice(9_000, 33_000)


@pytest.fixture(scope="module")
def simulated_data():
    """Generate simulated data for testing."""
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        sampling_frequency,
        is_event,
        place_fields,
    ) = make_simulated_data(n_neurons=30)

    return {
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "is_event": is_event,
    }


@pytest.fixture(scope="module")
def fit_predict(simulated_data):
    """Fit and decode once per detector configuration; return a cached getter.

    Returns
    -------
    callable
        ``fit_predict(**detector_kwargs) -> (detector, results)``. Callers must
        not modify the returned objects.
    """
    cache = {}

    def get(**detector_kwargs):
        key = tuple(sorted(detector_kwargs.items()))
        if key not in cache:
            detector = NonLocalSortedSpikesDetector(
                sorted_spikes_algorithm="sorted_spikes_kde",
                sorted_spikes_algorithm_params={
                    "position_std": 6.0,
                    "block_size": int(2**12),
                },
                **detector_kwargs,
            ).fit(
                simulated_data["time"],
                simulated_data["position"],
                simulated_data["spike_times"],
                is_training=~simulated_data["is_event"],
            )
            time = simulated_data["time"][DECODE_WINDOW]
            results = detector.predict(
                spike_times=simulated_data["spike_times"],
                time_edges=time_edges_from_centers(time),
                position=simulated_data["position"][DECODE_WINDOW],
                position_time=time,
            )
            cache[key] = (detector, results)
        return cache[key]

    return get


def assert_valid_posterior(results, label):
    """Finite, non-negative posterior and state probabilities summing to 1,
    and finite marginal log likelihoods."""
    posterior = results.acausal_posterior.values
    state_probs = results.acausal_state_probabilities.values
    assert np.all(np.isfinite(posterior)), f"{label}: posterior has NaN/Inf"
    assert np.all(np.isfinite(state_probs)), f"{label}: state probs have NaN/Inf"
    assert np.all(state_probs >= 0), f"{label}: negative state probabilities"
    np.testing.assert_allclose(
        posterior.sum(axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"{label}: posterior sums != 1",
    )
    np.testing.assert_allclose(
        state_probs.sum(axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"{label}: state prob sums != 1",
    )
    marginal_log_likelihoods = results.attrs.get("marginal_log_likelihoods")
    assert marginal_log_likelihoods is not None, (
        f"{label}: missing marginal_log_likelihoods"
    )
    assert np.all(np.isfinite(marginal_log_likelihoods)), (
        f"{label}: marginal log likelihoods are not finite"
    )


def test_multibin_local_fit_predict(fit_predict):
    """Multi-bin local model produces valid posteriors on synthetic data."""
    detector, results = fit_predict(local_position_std=5.0)

    assert_valid_posterior(results, "multi-bin local")
    assert detector.bin_sizes_[0] > 1, "Local state should have multiple bins"


def test_multibin_local_has_position_coordinates(fit_predict):
    """Multi-bin local posterior has real position coordinates, not NaN.

    The Local bins carry the same interior position grid as the non-local
    continuous state.
    """
    _, results = fit_predict(local_position_std=5.0)

    state_bins = results.acausal_posterior.indexes["state_bins"]
    states = state_bins.get_level_values("state")
    positions = np.asarray(state_bins.get_level_values("position"))
    local_positions = positions[states == "Local"]

    assert len(local_positions) > 1, (
        f"Expected multiple local bins with local_position_std, "
        f"got {len(local_positions)}"
    )
    assert np.all(np.isfinite(local_positions)), (
        "Local position coordinates contain NaN"
    )
    np.testing.assert_array_equal(
        local_positions, positions[states == "Non-Local Continuous"]
    )


def test_penalty_and_kernel_simultaneous(fit_predict):
    """Both non_local_position_penalty and local_position_std produce valid results."""
    _, results = fit_predict(local_position_std=5.0, non_local_position_penalty=1.0)

    assert_valid_posterior(results, "penalty and kernel")


@pytest.fixture(scope="module")
def clusterless_simulated_data():
    """Clusterless simulated data for testing: three runs (70 s)."""
    sim = make_simulated_run_data(n_tetrodes=5, n_runs=3, seed=42)
    return sim


def test_clusterless_multibin_local_fit_predict(clusterless_simulated_data):
    """NonLocalClusterlessDetector with local_position_std produces valid posteriors."""
    sim = clusterless_simulated_data

    detector = NonLocalClusterlessDetector(
        local_position_std=5.0,
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        sim.position_time,
        sim.position,
        sim.spike_times,
        sim.spike_waveform_features,
    )

    # Decode the first 20 s (20,000 bins) of the 70 s recording.
    results = detector.predict(
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
        time_edges=sim.edges[:20_001],
        position=sim.position,
        position_time=sim.position_time,
    )

    assert detector.bin_sizes_[0] > 1, "Local state should have multiple bins"
    assert_valid_posterior(results, "clusterless multi-bin local")


def test_sharp_sigma_matches_legacy(fit_predict):
    """In the sharp-sigma limit, multi-bin local should match legacy.

    Validates the mass-balance scaling (kernel * n_bins): when the kernel
    is concentrated at a single bin, the multi-bin local posterior mass
    should equal the legacy single-bin local mass up to float32 precision.
    The legacy model (``local_position_std=None``, the default) keeps a single
    Local bin.
    """
    legacy_detector, legacy_results = fit_predict()
    # Very sharp kernel — effectively a delta at the animal's bin
    sharp_detector, sharp_results = fit_predict(local_position_std=0.01)

    assert legacy_detector.bin_sizes_[0] == 1, "Legacy Local state has one bin"
    assert sharp_detector.bin_sizes_[0] > 1
    assert_valid_posterior(legacy_results, "legacy local")
    assert_valid_posterior(sharp_results, "sharp sigma")

    # Mean occupancy per discrete state should match within 1% relative
    legacy_mean = legacy_results.acausal_state_probabilities.values.mean(axis=0)
    sharp_mean = sharp_results.acausal_state_probabilities.values.mean(axis=0)
    # The animal is mostly where it is tracked, so Local dominates.
    assert legacy_mean[0] > 0.7, f"Legacy mean Local probability = {legacy_mean[0]}"
    np.testing.assert_allclose(
        sharp_mean,
        legacy_mean,
        rtol=1e-2,
        atol=1e-3,
        err_msg=(
            "Sharp-sigma multi-bin local should match legacy state "
            f"occupancies. Legacy={legacy_mean}, Sharp={sharp_mean}"
        ),
    )


def test_large_sigma_spreads_local_posterior(fit_predict):
    """With very large sigma, local state posterior spreads across bins.

    With ``local_position_std`` larger than the track, the kernel is
    essentially uniform so the local state has no spatial preference.
    Under these conditions, the local posterior over bins (conditional
    on being in the Local state) should be approximately uniform across
    interior bins.
    """
    # much larger than the simulated track
    detector, results = fit_predict(local_position_std=1000.0)

    # Extract the local-state slice of the full posterior.
    # Hard-assert the state-layout assumption so that a refactor which
    # reorders states doesn't silently measure the wrong state.
    n_local_bins = int(detector.bin_sizes_[0])
    assert n_local_bins > 1, "Local state should have multiple bins"
    assert np.all(detector.state_ind_[:n_local_bins] == 0), (
        "Local state must be state 0 and its bins must come first"
    )
    local_posterior = results.acausal_posterior.values[:, :n_local_bins]

    # When the local state is meaningfully occupied, the conditional
    # posterior over local bins should be approximately uniform.
    # Normalize per time step; skip rows where local mass is negligible.
    local_mass = local_posterior.sum(axis=1, keepdims=True)
    active = local_mass.ravel() > 1e-3
    assert active.sum() > 0, "No time steps with meaningful local mass"

    conditional = local_posterior[active] / local_mass[active]
    # The coefficient of variation across bins should be small when the
    # kernel is uniform (no spatial preference).
    cv = conditional.std(axis=1) / (conditional.mean(axis=1) + 1e-12)
    assert cv.mean() < 0.5, (
        f"Large-sigma local posterior should be near-uniform across bins "
        f"(mean CV={cv.mean():.3f})"
    )


def test_delta_kernel_fit_predict(fit_predict):
    """local_position_std=0.0 fits + predicts without NaN/Inf, Local dominant."""
    _, results = fit_predict(local_position_std=0.0)

    assert_valid_posterior(results, "delta kernel")
    mean_state_probs = results.acausal_state_probabilities.values.mean(axis=0)
    assert np.argmax(mean_state_probs) == 0, (
        f"Delta-kernel Local state should dominate; mean probs = {mean_state_probs}"
    )


def test_delta_kernel_matches_narrow_gaussian(fit_predict):
    """Posteriors from σ=0 (delta) and σ=0.01 (very narrow Gaussian) match closely.

    The delta kernel is the σ→0 limit of the Gaussian kernel. A very
    narrow Gaussian (e.g., σ=0.01 cm, much smaller than a bin) should
    produce nearly-identical state occupancies.
    """
    _, delta_results = fit_predict(local_position_std=0.0)
    _, narrow_results = fit_predict(local_position_std=0.01)

    # State occupancies should match within 1% rtol, same as the
    # sharp-sigma-vs-legacy test.
    delta_mean = delta_results.acausal_state_probabilities.values.mean(axis=0)
    narrow_mean = narrow_results.acausal_state_probabilities.values.mean(axis=0)
    np.testing.assert_allclose(
        delta_mean,
        narrow_mean,
        rtol=1e-2,
        atol=1e-3,
        err_msg=(
            f"σ=0 delta kernel should match σ=0.01 Gaussian. "
            f"delta={delta_mean}, narrow={narrow_mean}"
        ),
    )
