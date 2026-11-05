"""Fitting and decoding check their position inputs against the detector."""

import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.exceptions import DataError, ValidationError

pytestmark = pytest.mark.unit

DECODE_METHODS = ["predict", "most_likely_sequence", "compute_log_likelihood"]


TIME = np.linspace(0, 2, 61)
POSITION = np.column_stack((5 + 4 * np.sin(TIME), 5 + 4 * np.cos(TIME)))
SPIKE_DATA = {
    "sorted": {"spike_times": [np.array([0.1, 0.3, 0.8, 1.2, 1.9])]},
    "clusterless": {
        "spike_times": [np.array([0.1, 0.3, 0.8, 1.2, 1.9])],
        "spike_waveform_features": [np.arange(10.0).reshape(5, 2)],
    },
}
DETECTORS = {
    "sorted": NonLocalSortedSpikesDetector,
    "clusterless": NonLocalClusterlessDetector,
}


@pytest.fixture(scope="module", params=["sorted", "clusterless"])
def fitted(request):
    """A fitted 2-D detector of each family and its decode arguments."""
    detector = DETECTORS[request.param](
        environments=Environment(place_bin_size=2, position_range=((0, 10), (0, 10))),
        infer_track_interior=False,
        sampling_frequency=50.0,
    )
    spikes = SPIKE_DATA[request.param]
    detector.fit(TIME, POSITION, **spikes)
    return detector, {
        "position_time": TIME,
        "position": POSITION,
        "time_edges": np.linspace(0, 2, 101),
        **spikes,
    }


def _decode(detector, method, arguments):
    if method == "compute_log_likelihood":
        spikes = [arguments[name] for name in detector._SPIKE_DATA_NAMES]
        return detector.compute_log_likelihood(
            arguments["position_time"],
            arguments["position"],
            *spikes,
            time_edges=arguments["time_edges"],
        )
    return getattr(detector, method)(**arguments)


@pytest.mark.parametrize("method", DECODE_METHODS)
def test_repeated_position_time_raises(fitted, method):
    """Decoding requires strictly increasing position_time, as fitting does."""
    detector, arguments = fitted
    position_time = arguments["position_time"].copy()
    position_time[10] = position_time[9]
    with pytest.raises(DataError, match="strictly increasing"):
        _decode(detector, method, arguments | {"position_time": position_time})


@pytest.mark.parametrize("method", DECODE_METHODS)
@pytest.mark.parametrize("n_columns", [1, 3])
def test_position_columns_must_match_the_fitted_environment(fitted, method, n_columns):
    """A position with another number of columns than the fitted grid raises
    a ValidationError naming both counts, not a zip or index error."""
    detector, arguments = fitted
    position = np.column_stack([arguments["position"][:, 0]] * n_columns)
    with pytest.raises(ValidationError, match=f"{n_columns} column") as info:
        _decode(detector, method, arguments | {"position": position})
    assert "2 columns" in str(info.value)


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_environment_labels_matching_no_environment_are_named(family):
    """Labels naming no configured environment raise an error naming the
    labels and the environment left without samples."""
    detector = DETECTORS[family](
        environments=Environment(environment_name="track", place_bin_size=2)
    )
    labels = np.array(["trak"] * TIME.size)
    with pytest.raises(ValidationError, match="environment_labels") as info:
        detector.fit(TIME, POSITION, **SPIKE_DATA[family], environment_labels=labels)
    assert "'trak'" in str(info.value)
    assert "'track'" in str(info.value)
