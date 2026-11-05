"""Fitting and decoding check their position inputs against the detector."""

import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
)
from non_local_detector.exceptions import DataError, DataQualityWarning, ValidationError

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


def test_unlabeled_samples_do_not_break_the_label_error():
    """Untracked samples left unlabeled (None) beside a misspelt name still
    give the error naming the labels, not a TypeError from sorting them."""
    detector = NonLocalSortedSpikesDetector(
        environments=Environment(environment_name="track", place_bin_size=2)
    )
    labels = np.array(["trak"] * TIME.size, dtype=object)
    labels[:5] = None
    with pytest.raises(ValidationError, match="environment_labels") as info:
        detector.fit(TIME, POSITION, **SPIKE_DATA["sorted"], environment_labels=labels)
    assert "'trak'" in str(info.value)


# The fitted grid spans (0, 10) x (0, 10) in 2-unit bins.
def _shifted(offset, n_shifted=TIME.size):
    position = POSITION.copy()
    position[:n_shifted] += offset
    return position


@pytest.mark.parametrize("method", DECODE_METHODS)
def test_decode_position_far_outside_the_grid_warns(fitted, method):
    """Positions in other units than the grid are snapped onto it; decoding
    warns with the fraction, the distance and the likely cause."""
    detector, arguments = fitted
    with pytest.warns(DataQualityWarning, match="100.0% of the 61") as record:
        _decode(detector, method, arguments | {"position": _shifted(1000.0)})
    message = str(record[0].message)
    assert "units" in message
    assert "position_range" in message
    assert "1.41e+03" in message  # the farthest sample's distance from the grid


@pytest.mark.parametrize("family", ["sorted", "clusterless"])
@pytest.mark.parametrize(
    "position", [POSITION + 20.0, POSITION * 10.0], ids=["shifted", "rescaled"]
)
def test_fit_position_outside_the_position_range_warns(family, position):
    """Fitting positions that miss the configured position_range (another
    origin or other units) warns once the grid is fit."""
    detector = DETECTORS[family](
        environments=Environment(place_bin_size=2, position_range=((0, 10), (0, 10))),
        infer_track_interior=False,
    )
    with pytest.warns(DataQualityWarning, match="outside the fitted grid"):
        detector.fit(TIME, position, **SPIKE_DATA[family])


@pytest.mark.parametrize("method", DECODE_METHODS)
@pytest.mark.parametrize(
    ("offset", "n_shifted"),
    [(1.9, TIME.size), (1000.0, 0)],
    ids=["within-one-bin", "none-shifted"],
)
def test_position_near_the_grid_does_not_warn(fitted, method, offset, n_shifted):
    """Samples less than one bin outside the grid are ordinary tracking noise."""
    detector, arguments = fitted
    position = np.clip(POSITION, 0.5, 9.5)
    position[:n_shifted] = np.clip(position[:n_shifted] + offset, None, 11.9)
    _decode(detector, method, arguments | {"position": position})


@pytest.mark.parametrize("method", DECODE_METHODS)
def test_one_percent_of_samples_far_outside_does_not_warn(method):
    """A stray sample or two (at most 1%) is not reported."""
    time = np.linspace(0, 2, 201)
    position = np.column_stack((5 + 4 * np.sin(time), 5 + 4 * np.cos(time)))
    detector = NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=2, position_range=((0, 10), (0, 10))),
        infer_track_interior=False,
        sampling_frequency=100.0,
    ).fit(time, position, **SPIKE_DATA["sorted"])
    position[[3, 150]] += 1000.0
    arguments = {
        "position_time": time,
        "position": position,
        "time_edges": np.linspace(-0.005, 2.005, 202),
        **SPIKE_DATA["sorted"],
    }
    _decode(detector, method, arguments)


def test_track_interior_holes_do_not_warn():
    """An open field with an unvisited centre infers a hole inside the grid's
    bounding box; positions around it are on the grid."""
    rng = np.random.default_rng(0)
    time = np.arange(2000) / 500.0
    angle = np.pi * time
    radius = 40 + 10 * np.sin(5 * time)
    position = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    detector = NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=4.0), infer_track_interior=True
    ).fit(time, position, [np.sort(rng.uniform(0, time[-1], 50))])
    assert not detector.environments_[0].is_track_interior_.all()
    detector.predict(
        [np.sort(rng.uniform(0, time[-1], 50))],
        time_edges=np.linspace(-0.001, time[-1] + 0.001, 2001),
        position=position,
        position_time=time,
    )


@pytest.mark.parametrize("method", DECODE_METHODS)
def test_position_a_decoder_ignores_is_not_checked(method):
    """A model whose likelihood never reads position (no Local state, no
    non-local penalty) decodes with a position it would reject otherwise."""
    detector = SortedSpikesDecoder(
        environments=Environment(place_bin_size=2, position_range=((0, 10), (0, 10))),
        infer_track_interior=False,
        sampling_frequency=50.0,
    ).fit(TIME, POSITION, **SPIKE_DATA["sorted"])
    position_time = TIME.copy()
    position_time[10] = position_time[9]
    arguments = {
        "position_time": position_time,
        "position": np.column_stack([POSITION, POSITION[:, 0]]),
        "time_edges": np.linspace(0, 2, 101),
        **SPIKE_DATA["sorted"],
    }
    _decode(detector, method, arguments)
