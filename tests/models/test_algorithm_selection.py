"""A detector decodes with the likelihood algorithm that fitted its encoding model.

A detector unpacks the fit's encoding model into the algorithm's predict, so
the predict of another algorithm either rejects the keys or reads keys that
happen to share a name. Each encoding model records the algorithm that fitted
it, and a detector whose algorithm parameter no longer names that algorithm
refuses to decode. Unknown algorithm names are rejected at construction.
Detectors saved before encoding models recorded their algorithm get it back
on load.
"""

import copy
import pickle

import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    ClusterlessDecoder,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
    _record_legacy_encoding_algorithm,
)
from non_local_detector.observation_models import ObservationModel
from tests.models._helpers import ALGORITHMS, _decoder

pytestmark = pytest.mark.unit

_N_DECODE = 200


@pytest.mark.parametrize(
    ("detector_cls", "parameter", "registry"),
    [
        (SortedSpikesDecoder, "sorted_spikes_algorithm", _SORTED_SPIKES_ALGORITHMS),
        (
            NonLocalSortedSpikesDetector,
            "sorted_spikes_algorithm",
            _SORTED_SPIKES_ALGORITHMS,
        ),
        (ClusterlessDecoder, "clusterless_algorithm", _CLUSTERLESS_ALGORITHMS),
        (
            NonLocalClusterlessDetector,
            "clusterless_algorithm",
            _CLUSTERLESS_ALGORITHMS,
        ),
    ],
)
def test_unknown_algorithm_is_rejected_at_construction(
    detector_cls, parameter, registry
):
    """The error names the bad value and every registered algorithm."""
    with pytest.raises(ValidationError, match="'nope'") as error:
        detector_cls(**{parameter: "nope"})
    for name in registry:
        assert name in str(error.value)


def test_removed_log_kde_name_constructs_and_fitting_points_to_clusterless_kde(
    clusterless_sim,
):
    """Stored parameter sets with the removed ``clusterless_kde_log`` still
    build a detector, to inspect; fitting it points to ``clusterless_kde``."""
    detector = ClusterlessDecoder(clusterless_algorithm="clusterless_kde_log")
    with pytest.raises(ValidationError, match='Use "clusterless_kde"'):
        detector.fit(
            clusterless_sim.position_time,
            clusterless_sim.position,
            clusterless_sim.spike_times,
            clusterless_sim.spike_waveform_features,
        )


def _fit_sorted(algorithm, sorted_sim):
    time, position, spike_times = sorted_sim
    params = {"disable_progress_bar": True}
    if algorithm == "sorted_spikes_mrf":
        params["penalty"] = 1.0
    detector = SortedSpikesDecoder(
        sorted_spikes_algorithm=algorithm, sorted_spikes_algorithm_params=params
    )
    return detector.fit(time, position, spike_times)


def _predict_sorted(detector, sorted_sim):
    time, position, spike_times = sorted_sim
    return detector.predict(
        spike_times,
        time_edges=time_edges_from_centers(time[:_N_DECODE]),
        position_time=time,
        position=position,
        return_outputs="log_likelihood",
    )


@pytest.mark.parametrize(
    ("fitted", "switched"),
    [
        # The GLM predict rejects the KDE's keys with a TypeError.
        ("sorted_spikes_kde", "sorted_spikes_glm"),
        # The diffusion predict used to accept the KDE's keys and decode.
        ("sorted_spikes_kde", "sorted_spikes_diffusion"),
        ("sorted_spikes_diffusion", "sorted_spikes_mrf"),
    ],
)
def test_switching_the_algorithm_after_fit_raises(sorted_sim, fitted, switched):
    """Predict names both algorithms and says to refit; switching back decodes."""
    detector = _fit_sorted(fitted, sorted_sim)
    expected = _predict_sorted(detector, sorted_sim)

    detector.set_params(sorted_spikes_algorithm=switched)
    with pytest.raises(ValidationError, match="refit") as error:
        _predict_sorted(detector, sorted_sim)
    assert fitted in str(error.value)
    assert switched in str(error.value)

    detector.set_params(sorted_spikes_algorithm=fitted)
    restored = _predict_sorted(detector, sorted_sim)
    np.testing.assert_array_equal(
        np.asarray(restored.log_likelihood), np.asarray(expected.log_likelihood)
    )


def _decode_with(entry_point, detector, sorted_sim):
    """Decode the first rows through one of the detector's decode methods."""
    time, position, spike_times = sorted_sim
    edges = time_edges_from_centers(time[:_N_DECODE])
    if entry_point == "compute_log_likelihood":
        return detector.compute_log_likelihood(
            time, position, spike_times, time_edges=edges
        )
    decode = {"time_edges": edges, "position_time": time, "position": position}
    if entry_point == "most_likely_sequence":
        return detector.most_likely_sequence(spike_times, **decode)
    n_chunks = 3 if entry_point == "chunked_predict" else 1
    return detector.predict(spike_times, **decode, n_chunks=n_chunks)


@pytest.fixture(scope="module")
def kde_detector(sorted_sim):
    """A non-local detector fitted with sorted KDE, copied by each test."""
    time, position, spike_times = sorted_sim
    return NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm_params={"disable_progress_bar": True}
    ).fit(time, position, spike_times)


@pytest.mark.parametrize(
    "entry_point",
    ["predict", "chunked_predict", "compute_log_likelihood", "most_likely_sequence"],
)
def test_every_decode_method_checks_the_recorded_algorithm(
    kde_detector, sorted_sim, entry_point
):
    """Dense and chunked prediction, the likelihood and Viterbi all refuse."""
    detector = copy.deepcopy(kde_detector)
    _decode_with(entry_point, detector, sorted_sim)
    detector.set_params(sorted_spikes_algorithm="sorted_spikes_diffusion")
    with pytest.raises(ValidationError, match="'sorted_spikes_kde'"):
        _decode_with(entry_point, detector, sorted_sim)


def test_no_spike_only_entries_are_not_checked(sorted_sim):
    """No-Spike reads no encoding model, so its own group's entry is skipped.

    An entry that only a No-Spike state uses decodes whatever algorithm it
    records; the same mismatch on a spiking state's entry raises.
    """
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        observation_models=[
            ObservationModel(is_local=True),
            ObservationModel(is_no_spike=True, encoding_group=1),
            ObservationModel(),
            ObservationModel(),
        ],
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )
    groups = (np.arange(time.size) >= time.size // 2).astype(int)
    detector.fit(time, position, spike_times, encoding_group_labels=groups)
    assert set(detector.encoding_model_) == {("", 0), ("", 1)}

    detector.encoding_model_[("", 1)]["algorithm"] = "sorted_spikes_glm"
    _decode_with("predict", detector, sorted_sim)

    detector.encoding_model_[("", 0)]["algorithm"] = "sorted_spikes_glm"
    with pytest.raises(ValidationError, match="'sorted_spikes_glm'"):
        _decode_with("predict", detector, sorted_sim)


def test_switching_the_clusterless_algorithm_after_fit_raises(clusterless_sim):
    """The clusterless detectors check the recorded algorithm too."""
    detector = ClusterlessDecoder(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        clusterless_sim.position_time,
        clusterless_sim.position,
        clusterless_sim.spike_times,
        clusterless_sim.spike_waveform_features,
    )
    detector.set_params(clusterless_algorithm="clusterless_diffusion")
    with pytest.raises(ValidationError, match="clusterless_kde"):
        detector.predict(
            clusterless_sim.spike_times,
            clusterless_sim.spike_waveform_features,
            time_edges=time_edges_from_centers(
                clusterless_sim.position_time[:_N_DECODE]
            ),
        )


def test_encoding_model_without_a_recorded_algorithm_raises(sorted_sim):
    """A model whose encoding entry lacks the algorithm must be refit."""
    detector = _fit_sorted("sorted_spikes_kde", sorted_sim)
    (encoding_model,) = detector.encoding_model_.values()
    del encoding_model["algorithm"]
    with pytest.raises(ValidationError, match="refit") as error:
        _predict_sorted(detector, sorted_sim)
    assert "=None" not in str(error.value)


@pytest.mark.filterwarnings("ignore:EM did not converge:UserWarning")
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_em_refit_does_not_go_through_the_public_fit_encoding_model(
    family, sorted_sim, clusterless_sim, monkeypatch
):
    """Both families' EM refits call the private fit body, so an override of
    the public ``fit_encoding_model`` changes EM refits in neither."""
    if family == "sorted":
        time, position, spike_times = sorted_sim
        detector = SortedSpikesDecoder(
            sorted_spikes_algorithm_params={"disable_progress_bar": True}
        )
        data = {"spike_times": spike_times}
    else:
        time = clusterless_sim.position_time
        position = clusterless_sim.position
        detector = ClusterlessDecoder(
            clusterless_algorithm_params={"disable_progress_bar": True}
        )
        data = {
            "spike_times": clusterless_sim.spike_times,
            "spike_waveform_features": clusterless_sim.spike_waveform_features,
        }
    public_calls = []
    original = detector.fit_encoding_model

    def spy(*args, **kwargs):
        public_calls.append(kwargs.get("weights"))
        return original(*args, **kwargs)

    monkeypatch.setattr(detector, "fit_encoding_model", spy)
    detector.estimate_parameters(
        position_time=time,
        position=position,
        **data,
        time_edges=time_edges_from_centers(time),
        max_iter=1,
        estimate_encoding_model=True,
        min_encoding_local_mass=0.0,
        min_encoding_local_ess=0.0,
    )
    # The initial fit is public; the posterior-weighted refit is not.
    assert public_calls == [None]


# Algorithms whose encoding models recorded no mark_dimensions until encoding
# models recorded their algorithm.
_WITHOUT_MARK_DIMENSIONS = ("clusterless_kde", "clusterless_diffusion")


def _round_trip_fit(algorithm):
    """A fitted one-state decoder, its decode, and the decode's arguments."""
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    args = ([np.array([0.021, 0.067, 0.121, 0.167])],)
    if algorithm.startswith("clusterless"):
        args = (*args, [np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    detector = _decoder(algorithm).fit(time, position, *args)
    kwargs = {
        "time_edges": np.arange(81) * 0.002,
        "position_time": time,
        "position": position,
        "return_outputs": "log_likelihood",
    }
    return detector, detector.predict(*args, **kwargs), args, kwargs


@pytest.mark.filterwarnings(
    r"ignore:position_std \(.*\) is below the characteristic bin spacing:UserWarning"
)
@pytest.mark.parametrize(
    "algorithm",
    # The GLMs' patsy spline basis (DesignInfo) cannot be pickled.
    [
        name
        for name in ALGORITHMS
        if name not in ("sorted_spikes_glm", "clusterless_mark_glm")
    ],
)
def test_detector_saved_before_algorithms_were_recorded_decodes_after_loading(
    algorithm,
):
    """Loading identifies the algorithm from the encoding model's keys and
    restores ``mark_dimensions``, so the detector decodes as before."""
    detector, expected, args, kwargs = _round_trip_fit(algorithm)
    (entry,) = detector.encoding_model_.values()
    recorded = {
        key: entry[key] for key in ("algorithm", "mark_dimensions") if key in entry
    }
    del entry["algorithm"]
    if algorithm in _WITHOUT_MARK_DIMENSIONS:
        del entry["mark_dimensions"]

    restored = pickle.loads(pickle.dumps(detector))

    (restored_entry,) = restored.encoding_model_.values()
    assert {key: restored_entry[key] for key in recorded} == recorded
    xr.testing.assert_identical(restored.predict(*args, **kwargs), expected)


def test_saved_encoding_model_of_no_registered_type_still_asks_for_a_refit():
    """An entry whose keys match no registered algorithm, such as one from a
    0.6 release, is left without an algorithm."""
    detector, _, args, kwargs = _round_trip_fit("sorted_spikes_kde")
    (entry,) = detector.encoding_model_.values()
    del entry["algorithm"]
    entry["key_of_an_older_release"] = None

    restored = pickle.loads(pickle.dumps(detector))

    (restored_entry,) = restored.encoding_model_.values()
    assert "algorithm" not in restored_entry
    with pytest.raises(ValidationError, match="does not record the algorithm"):
        restored.predict(*args, **kwargs)


@pytest.mark.parametrize(
    ("configured", "recorded"),
    [("first", "first"), ("second", "second"), ("other", None), (None, None)],
)
def test_encoding_model_matching_two_algorithms_takes_the_configured_one(
    configured, recorded
):
    """Should two registered algorithms share the saved keys, the detector's
    algorithm decides, and only if it is one of them."""
    record = _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"]
    entry = dict.fromkeys(set(record.encoding_model_type.__required_keys__))
    del entry["algorithm"]

    _record_legacy_encoding_algorithm(
        entry, {"first": record, "second": record}, configured
    )

    assert entry.get("algorithm") == recorded
