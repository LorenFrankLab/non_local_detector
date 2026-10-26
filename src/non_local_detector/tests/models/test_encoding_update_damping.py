"""Nonzero ``encoding_update_damping`` is rejected before any state changes.

Damping blended only ``place_fields`` and the no-spike term. No backend keeps
every fitted quantity consistent under that blend: GLM/KDE local paths read
other fitted state, diffusion/MRF read ``interior_log_place_fields`` for the
spike term, and every clusterless backend has no ``place_fields`` at all, so
the blend was a silent no-op there. Any nonzero value is therefore rejected,
and it must be rejected before the wrapper's initial ``fit`` replaces the
encoding model.
"""

import copy

import numpy as np
import pytest

from non_local_detector import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError

# Every nonzero value takes the same branch. A negative value and NaN are kept
# because an ``x > 0`` style guard would let them through.
REJECTED_DAMPING = [0.5, -0.1, np.nan]


@pytest.fixture(scope="module")
def sorted_inputs(sorted_sim):
    time, position, spike_times = sorted_sim
    return {
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "time_edges": time_edges_from_centers(time),
    }


@pytest.fixture(scope="module")
def clusterless_inputs(clusterless_sim):
    sim = clusterless_sim
    return {
        "position_time": sim.position_time,
        "position": sim.position,
        "spike_times": sim.spike_times,
        "spike_waveform_features": sim.spike_waveform_features,
        "time_edges": time_edges_from_centers(sim.position_time),
    }


def _fit(detector, inputs):
    fit_inputs = {k: v for k, v in inputs.items() if k != "time_edges"}
    return detector.fit(**fit_inputs)


@pytest.fixture(scope="module")
def fitted_sorted(sorted_inputs):
    return _fit(NonLocalSortedSpikesDetector(), sorted_inputs)


@pytest.fixture(scope="module")
def fitted_clusterless(clusterless_inputs):
    return _fit(NonLocalClusterlessDetector(), clusterless_inputs)


@pytest.mark.unit
@pytest.mark.parametrize("damping", REJECTED_DAMPING)
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_nonzero_damping_is_rejected_before_mutation(request, family, damping):
    """A rejected value raises ``ValidationError`` and leaves fitted state alone."""
    inputs = request.getfixturevalue(f"{family}_inputs")
    # A copy of the shared fit, since a failing guard would mutate it.
    detector = copy.deepcopy(request.getfixturevalue(f"fitted_{family}"))
    encoding_model_before = detector.encoding_model_
    state_before = set(vars(detector))

    with pytest.raises(ValidationError, match="encoding_update_damping"):
        detector.estimate_parameters(
            **inputs, max_iter=1, encoding_update_damping=damping
        )

    assert detector.encoding_model_ is encoding_model_before
    assert set(vars(detector)) == state_before


@pytest.mark.unit
def test_zero_damping_is_accepted(sorted_inputs):
    """An explicit zero is accepted and matches EM run without the argument."""
    results = [
        NonLocalSortedSpikesDetector().estimate_parameters(
            **sorted_inputs, max_iter=1, estimate_encoding_model=True, **damping
        )
        for damping in ({"encoding_update_damping": 0.0}, {})
    ]
    np.testing.assert_array_equal(
        results[0].acausal_posterior.values, results[1].acausal_posterior.values
    )
