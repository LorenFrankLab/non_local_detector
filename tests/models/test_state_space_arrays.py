"""A fit's state-space arrays cannot be written to.

Decoding reads them, so writing to one (``detector.state_ind_[0] = 3``) would
silently corrupt the fit. They stay read-only through every way a detector
gets a ``StateSpace``, which all hold the fit's arrays without copying them.
"""

import copy
import pickle

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from tests.models._helpers import SORTED_SIM_RATE

pytestmark = pytest.mark.integration

ARRAYS = (
    "state_ind_",
    "bin_sizes_",
    "is_track_interior_state_bins_",
    "continuous_state_transitions_",
)


@pytest.fixture(scope="module")
def recording(sorted_sim):
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )
    return detector.fit(time, position, spike_times), sorted_sim


def _saved(detector, tmp_path):
    path = tmp_path / "model.nld"
    detector.save_model(path)
    return NonLocalSortedSpikesDetector.load_model(path)


ROUTES = {
    "fit": lambda detector, tmp_path: detector,
    "copy": lambda detector, tmp_path: detector.copy(),
    "deepcopy": lambda detector, tmp_path: copy.deepcopy(detector),
    "pickle": lambda detector, tmp_path: pickle.loads(pickle.dumps(detector)),
    "saved model": _saved,
}


@pytest.mark.parametrize("name", ARRAYS)
@pytest.mark.parametrize("route", ROUTES)
def test_a_fit_s_arrays_are_read_only(recording, route, name, tmp_path):
    detector, _ = recording

    array = getattr(ROUTES[route](detector, tmp_path), name)

    with pytest.raises(ValueError, match="read-only"):
        array[0] = 0


def test_a_densified_transition_is_read_only(sorted_sim):
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times, transition_representation="structured")

    dense = detector.state_space_.densified().continuous_transitions

    assert isinstance(dense, np.ndarray)
    with pytest.raises(ValueError, match="read-only"):
        dense[0, 0] = 0.0


def test_the_arrays_stay_read_only_after_em(sorted_sim):
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )
    with pytest.warns(UserWarning, match="EM did not converge"):
        detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time[:100]),
            max_iter=1,
        )

    for name in ARRAYS:
        assert not getattr(detector, name).flags.writeable


@pytest.mark.parametrize("route", ROUTES)
def test_a_restored_detector_decodes_the_same(recording, route, tmp_path):
    detector, (time, position, spike_times) = recording
    decode = {
        "spike_times": spike_times,
        "time_edges": time_edges_from_centers(time[:100]),
        "position_time": time,
        "position": position,
    }

    actual = ROUTES[route](detector, tmp_path).predict(**decode)

    np.testing.assert_array_equal(
        actual.acausal_posterior.values,
        detector.predict(**decode).acausal_posterior.values,
    )


def test_decoding_uses_the_fit_s_arrays_without_copying(recording):
    detector, _ = recording
    space = detector.state_space_

    assert detector.state_ind_ is space.state_ind
    assert detector.bin_sizes_ is space.bin_sizes
    assert detector.is_track_interior_state_bins_ is space.is_track_interior
    assert detector.continuous_state_transitions_ is space.continuous_transitions
