"""Test that non-local position penalty actually affects the posterior.

The parameter validation tests in test_non_local_penalty.py check that
invalid parameters are rejected. This test verifies the penalty has a
measurable effect on decoding output — catching a regression where the
penalty is accepted but silently ignored.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.mark.slow
@pytest.mark.integration
def test_penalty_changes_posterior():
    """Posterior with penalty != 0 should differ from penalty == 0."""
    speed, position, spike_times, time, event_times, sampling_freq, is_event, _ = (
        make_simulated_data(seed=42, n_neurons=5)
    )

    kwargs = {
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "is_training": ~is_event,
    }

    detector_off = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        non_local_position_penalty=0.0,
    )
    results_off = detector_off.estimate_parameters(**kwargs)

    detector_on = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        non_local_position_penalty=5.0,
        non_local_penalty_sigma=5.0,
    )
    results_on = detector_on.estimate_parameters(**kwargs)

    # Posteriors should differ when penalty is active
    assert not np.allclose(
        results_off.acausal_posterior.values,
        results_on.acausal_posterior.values,
        atol=1e-8,
    ), "Penalty had no effect on posterior — may be silently ignored"
