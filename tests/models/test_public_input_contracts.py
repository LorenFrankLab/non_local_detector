"""Public detector methods coerce and check their array inputs up front.

``fit``, ``fit_encoding_model``, ``predict``, ``estimate_parameters``,
``most_likely_sequence`` and ``compute_log_likelihood`` accept array-likes,
reject malformed arrays with a ``ValidationError`` before any work, and treat
the string and list forms of ``return_outputs`` alike.
"""

import numpy as np
import pandas as pd
import pytest

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.models.base import (
    _normalize_return_outputs,
    _validate_inference_arguments,
)
from tests.models._helpers import _assert_unchanged, _snapshot

N_TIME = 100
EDGES = np.arange(41) * 0.002


def _recording():
    time = np.arange(N_TIME) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spike_times = [np.array([0.021, 0.067, 0.121, 0.167])]
    return time, position, spike_times


def _decoder():
    return SortedSpikesDecoder(
        environments=Environment(place_bin_size=5, position_range=((40, 60),)),
        infer_track_interior=False,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


def _local_detector():
    """A detector with a Local state, which reads position at decode."""
    return NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=5, position_range=((40, 60),)),
        infer_track_interior=False,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


@pytest.fixture(scope="module")
def fitted_local():
    time, position, spike_times = _recording()
    return _local_detector().fit(time, position, spike_times), time, position


@pytest.mark.unit
@pytest.mark.parametrize("method", ["fit", "fit_encoding_model"])
@pytest.mark.parametrize(
    "is_training",
    [
        np.array([True]),  # would broadcast to every sample
        np.ones((N_TIME, 1), dtype=bool),  # not one entry per sample
        np.ones(N_TIME - 1, dtype=bool),
        np.full(N_TIME, 2),  # integers: 2 & True is False
        np.ones(N_TIME),  # floats are not a mask
    ],
    ids=["length_one", "column", "short", "integer", "float"],
)
def test_malformed_is_training_is_rejected(method, is_training):
    time, position, spike_times = _recording()
    detector = _decoder()
    if method == "fit_encoding_model":
        detector.fit(time, position, spike_times)
    with pytest.raises(ValidationError, match="is_training"):
        getattr(detector, method)(time, position, spike_times, is_training=is_training)


@pytest.mark.unit
def test_list_is_training_matches_array():
    time, position, spike_times = _recording()
    mask = np.arange(N_TIME) < N_TIME // 2
    from_array = _decoder().fit(time, position, spike_times, is_training=mask)
    from_list = _decoder().fit(time, position, spike_times, is_training=mask.tolist())
    (array_model,) = from_array.encoding_model_.values()
    (list_model,) = from_list.encoding_model_.values()
    np.testing.assert_array_equal(
        list_model["place_fields"], array_model["place_fields"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "labels", "match"),
    [
        ("environment_labels", np.array([""] * (N_TIME - 1)), "environment_labels"),
        ("environment_labels", np.zeros(N_TIME), "environment_labels"),
        ("encoding_group_labels", np.zeros(N_TIME - 1, dtype=int), "encoding_group"),
        ("encoding_group_labels", np.zeros(N_TIME), "encoding_group"),
        ("encoding_group_labels", np.zeros(N_TIME, dtype=bool), "encoding_group"),
    ],
    ids=["env_short", "env_numeric", "group_short", "group_float", "group_bool"],
)
def test_malformed_labels_are_rejected_before_fitting(name, labels, match):
    time, position, spike_times = _recording()
    detector = _decoder().fit(time, position, spike_times)
    snapshot = _snapshot(detector)
    with pytest.raises(ValidationError, match=match):
        detector.fit(time, position, spike_times, **{name: labels})
    _assert_unchanged(detector, snapshot)


@pytest.mark.unit
def test_list_environment_labels_match_array():
    time, position, spike_times = _recording()
    labels = [""] * N_TIME
    from_array = _decoder().fit(
        time, position, spike_times, environment_labels=np.asarray(labels)
    )
    from_list = _decoder().fit(time, position, spike_times, environment_labels=labels)
    (array_model,) = from_array.encoding_model_.values()
    (list_model,) = from_list.encoding_model_.values()
    np.testing.assert_array_equal(
        list_model["place_fields"], array_model["place_fields"]
    )


@pytest.mark.unit
def test_fit_rejects_position_without_one_row_per_sample():
    time, position, spike_times = _recording()
    with pytest.raises(ValidationError, match="position"):
        _decoder().fit(time, position[:-1], spike_times)


@pytest.mark.unit
def test_fit_accepts_list_position():
    time, position, spike_times = _recording()
    from_array = _decoder().fit(time, position, spike_times)
    from_list = _decoder().fit(time.tolist(), position.tolist(), spike_times)
    (array_model,) = from_array.encoding_model_.values()
    (list_model,) = from_list.encoding_model_.values()
    np.testing.assert_array_equal(
        list_model["place_fields"], array_model["place_fields"]
    )


@pytest.mark.unit
@pytest.mark.parametrize("method", ["predict", "most_likely_sequence"])
def test_decode_accepts_list_position(fitted_local, method):
    detector, time, position = fitted_local
    _, _, spike_times = _recording()
    decode = getattr(detector, method)
    expected = decode(
        spike_times, time_edges=EDGES, position=position, position_time=time
    )
    actual = decode(
        spike_times,
        time_edges=EDGES,
        position=position.tolist(),
        position_time=time.tolist(),
    )
    if method == "predict":
        np.testing.assert_array_equal(
            actual.acausal_posterior, expected.acausal_posterior
        )
    else:
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["predict", "most_likely_sequence"])
def test_decode_requires_position_time_with_position(fitted_local, method):
    detector, _, position = fitted_local
    _, _, spike_times = _recording()
    with pytest.raises(ValidationError, match="position_time is required"):
        getattr(detector, method)(spike_times, time_edges=EDGES, position=position)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["predict", "most_likely_sequence"])
def test_decode_rejects_position_without_one_row_per_sample(fitted_local, method):
    detector, time, position = fitted_local
    _, _, spike_times = _recording()
    with pytest.raises(ValidationError, match="position"):
        getattr(detector, method)(
            spike_times, time_edges=EDGES, position=position[:-1], position_time=time
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("filter", {"filter"}),
        ("predictive", {"predictive"}),
        ("predictive_posterior", {"predictive_posterior"}),
        ("log_likelihood", {"log_likelihood"}),
        (
            "all",
            {"filter", "predictive", "predictive_posterior", "log_likelihood"},
        ),
    ],
)
def test_return_outputs_string_and_list_forms_agree(string, expected):
    assert _normalize_return_outputs(string) == expected
    assert _normalize_return_outputs([string]) == expected
    assert _normalize_return_outputs({string}) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "arguments",
    [
        {"inference_mode": "checkpointed", "cache_likelihood": True},
        {"inference_mode": "checkpointed", "n_chunks": 2},
        {"inference_mode": "dense", "output_mode": "compact"},
        {"inference_mode": "dense", "chunk_size": 10},
    ],
)
def test_inference_argument_conflicts_raise_validation_error(arguments):
    settings = {
        "inference_mode": "dense",
        "output_mode": "spatial",
        "cache_likelihood": False,
        "n_chunks": 1,
        "chunk_size": None,
        "result_path": None,
        "checkpoint_dir": None,
        "selected_intervals": None,
    } | arguments
    with pytest.raises(ValidationError):
        _validate_inference_arguments(**settings)


@pytest.mark.unit
def test_base_time_contract_check_fails_closed():
    """A subclass that does not override the hook still gets the encoding check."""
    from non_local_detector.models.base import _DetectorBase

    with pytest.raises(ValidationError, match="not fitted"):
        _DetectorBase._validate_time_contract(_decoder())
    time, position, spike_times = _recording()
    detector = _decoder().fit(time, position, spike_times)
    _DetectorBase._validate_time_contract(detector)
    detector.time_contract_ = {**detector.time_contract_, "rate_units": "per_bin"}
    with pytest.raises(ValidationError, match="refit"):
        _DetectorBase._validate_time_contract(detector)


@pytest.mark.unit
def test_direct_log_likelihood_accepts_list_position(fitted_local):
    detector, time, position = fitted_local
    _, _, spike_times = _recording()
    expected = detector.compute_log_likelihood(
        time, position, spike_times, time_edges=EDGES
    )
    actual = detector.compute_log_likelihood(
        time.tolist(), position.tolist(), spike_times, time_edges=EDGES
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.unit
def test_direct_log_likelihood_rejects_position_without_position_time(fitted_local):
    detector, _, position = fitted_local
    _, _, spike_times = _recording()
    with pytest.raises(ValidationError, match="position_time is required"):
        detector.compute_log_likelihood(None, position, spike_times, time_edges=EDGES)
