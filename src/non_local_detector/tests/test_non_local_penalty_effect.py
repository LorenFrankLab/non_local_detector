"""Test that the non-local position penalty suppresses non-local mass near the animal.

The parameter validation tests in test_non_local_penalty.py check that
invalid parameters are rejected. This test verifies the penalty has the
intended effect on decoding output: it lowers the non-local posterior near
the animal's current position, catching a penalty that is ignored, has the
wrong sign, or is centered in the wrong place.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

PENALTY_STD = 5.0


def _non_local_mass_near_animal(results, animal_position, radius):
    """Fraction of non-local posterior mass within ``radius`` of the animal.

    Parameters
    ----------
    results : xr.Dataset
        Detector output with ``acausal_posterior`` of shape (n_time, n_state_bins).
    animal_position : np.ndarray, shape (n_time,)
    radius : float

    Returns
    -------
    fraction : float
    """
    posterior = results.acausal_posterior
    is_non_local = np.isin(
        posterior.state.values, ["Non-Local Continuous", "Non-Local Fragmented"]
    )
    is_near = (
        np.abs(posterior.position.values[None, :] - animal_position[:, None]) <= radius
    )
    non_local_mass = posterior.values * is_non_local[None, :]
    return float((non_local_mass * is_near).sum() / non_local_mass.sum())


@pytest.mark.integration
def test_penalty_suppresses_non_local_mass_near_animal():
    """A positive penalty removes most non-local mass within one std of the animal.

    The penalty is applied to the decode likelihood, so one fitted detector is
    reused and only the penalty parameters change between the two predictions.
    """
    _, position, spike_times, time, _, _, is_event, _ = make_simulated_data(
        seed=42, n_neurons=5
    )

    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        non_local_position_penalty=0.0,
    ).fit(time, position, spike_times, is_training=~is_event)

    n_decode = 20_000
    decode_time = time[:n_decode]
    decode_position = position[:n_decode]
    predict_kwargs = {
        "spike_times": [st[st <= decode_time[-1]] for st in spike_times],
        "time_edges": time_edges_from_centers(decode_time),
        "position_time": decode_time,
        "position": decode_position,
    }
    animal_position = np.asarray(decode_position).reshape(n_decode, -1)[:, 0]

    results_off = detector.predict(**predict_kwargs)
    detector.non_local_position_penalty = 5.0
    detector.non_local_penalty_std = PENALTY_STD
    results_on = detector.predict(**predict_kwargs)

    near_off = _non_local_mass_near_animal(results_off, animal_position, PENALTY_STD)
    near_on = _non_local_mass_near_animal(results_on, animal_position, PENALTY_STD)

    assert near_off > 0.01, f"too little non-local mass near the animal: {near_off}"
    assert near_on < 0.1 * near_off, (
        f"penalty did not suppress non-local mass near the animal: "
        f"{near_off:.4f} without penalty, {near_on:.4f} with"
    )
