"""Test that EM log-likelihood is monotonically non-decreasing.

This is a fundamental guarantee of the EM algorithm: the marginal
log-likelihood must never decrease across iterations. Violation
indicates a bug in the E-step or M-step.

Tests cover:
- Clusterless decoder with discrete-only updates
- Sorted spikes (KDE) encoding update guard integration
- Sorted spikes (GLM) end-to-end smoke test
"""

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder, NonLocalSortedSpikesDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


def _assert_monotonic(lls, label=""):
    """Assert that log-likelihoods are non-decreasing (with float tolerance)."""
    assert len(lls) >= 2, f"{label}: Expected at least 2 EM iterations, got {len(lls)}"
    for i in range(1, len(lls)):
        assert lls[i] >= lls[i - 1] - 1e-6, (
            f"{label}: EM log-likelihood decreased at iteration {i}: "
            f"{lls[i]:.6f} < {lls[i - 1]:.6f} "
            f"(diff={lls[i] - lls[i - 1]:.2e})"
        )


@pytest.mark.slow
@pytest.mark.integration
class TestEMMonotonicity:
    """EM marginal log-likelihood must be non-decreasing across iterations."""

    def test_clusterless_decoder_em_monotonicity(self):
        """Fit a ClusterlessDecoder with multiple EM iterations and verify
        that marginal_log_likelihoods is non-decreasing."""
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            n_runs=3,
            seed=42,
        )

        decoder = ClusterlessDecoder()
        results = decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            max_iter=5,
            estimate_encoding_model=False,
        )

        lls = results.attrs["marginal_log_likelihoods"]
        _assert_monotonic(lls, "ClusterlessDecoder (discrete only)")


@pytest.mark.slow
@pytest.mark.integration
class TestEncodingUpdateGuards:
    """Encoding update guards (ESS/mass, rollback, damping) work correctly."""

    def test_sorted_spikes_kde_encoding_with_rollback(self):
        """NonLocalSortedSpikesDetector with KDE encoding updates should
        run to completion with rollback guard active."""
        speed, position, spike_times, time, event_times, sampling_freq, is_event, _ = (
            make_simulated_data(seed=42, n_neurons=5)
        )

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_kde",
        )
        results = detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time=time,
            is_training=~is_event,
            max_iter=5,
            estimate_encoding_model=True,
        )

        lls = results.attrs["marginal_log_likelihoods"]
        assert len(lls) >= 2, f"Expected at least 2 EM iterations, got {len(lls)}"
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results

    def test_sorted_spikes_glm_encoding_runs_end_to_end(self):
        """NonLocalSortedSpikesDetector with GLM encoding updates should
        run to completion without errors."""
        speed, position, spike_times, time, event_times, sampling_freq, is_event, _ = (
            make_simulated_data(seed=42, n_neurons=5)
        )

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_glm",
            sorted_spikes_algorithm_params={},
        )
        results = detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time=time,
            is_training=~is_event,
            max_iter=3,
            estimate_encoding_model=True,
        )

        lls = results.attrs["marginal_log_likelihoods"]
        assert len(lls) >= 2, f"Expected at least 2 EM iterations, got {len(lls)}"
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results
