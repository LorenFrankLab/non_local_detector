"""Test that EM log-likelihood is monotonically non-decreasing.

This is a fundamental guarantee of the EM algorithm: the marginal
log-likelihood must never decrease across iterations. Violation
indicates a bug in the E-step or M-step.
"""

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder
from non_local_detector.simulate.clusterless_simulation import (
    make_simulated_run_data,
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
        assert len(lls) >= 2, f"Expected at least 2 EM iterations, got {len(lls)}"

        # EM guarantee: log-likelihood never decreases.
        # Small epsilon for floating-point noise.
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i - 1] - 1e-6, (
                f"EM log-likelihood decreased at iteration {i}: "
                f"{lls[i]:.6f} < {lls[i - 1]:.6f} "
                f"(diff={lls[i] - lls[i - 1]:.2e})"
            )
