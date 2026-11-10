"""``read_model_summary`` reads what a model learned without building it."""

import zipfile

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.model_format import read_model_summary
from tests.model_format.conftest import algorithm_params

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sorted_model(sorted_recording, tmp_path_factory):
    time, position, spike_times = sorted_recording
    detector = nld.NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde")
    ).fit(time, position, spike_times)
    path = tmp_path_factory.mktemp("summary") / "sorted.nld"
    detector.save_model(path)
    return detector, path


def test_summary_holds_the_grid_place_fields_and_transitions(sorted_model):
    detector, path = sorted_model

    summary = read_model_summary(path)

    assert summary["detector_type"] == "nld.NonLocalSortedSpikesDetector"
    assert summary["fitted"]
    assert summary["config"]["sorted_spikes_algorithm"] == "sorted_spikes_kde"
    assert summary["config"]["environments"][0]["type"] == "nld.Environment"
    (environment,) = summary["environments"]
    fitted = detector.environments_[0]
    np.testing.assert_array_equal(
        environment["place_bin_centers"], fitted.place_bin_centers_
    )
    np.testing.assert_array_equal(
        environment["is_track_interior"], fitted.is_track_interior_
    )
    (model,) = summary["encoding_models"]
    expected = detector.encoding_model_[("", 0)]
    assert (model["environment_name"], model["encoding_group"]) == ("", 0)
    assert model["algorithm"] == "sorted_spikes_kde"
    np.testing.assert_array_equal(model["place_fields"], expected["place_fields"])
    np.testing.assert_array_equal(model["mean_rates"], expected["mean_rates"])
    assert "occupancy_model" not in model
    np.testing.assert_array_equal(
        summary["discrete_state_transitions"], detector.discrete_state_transitions_
    )
    np.testing.assert_array_equal(
        summary["initial_conditions"], detector.initial_conditions_
    )


def test_summary_does_not_read_clusterless_spike_samples(
    clusterless_recording, tmp_path
):
    """The summary leaves the encoding spikes' marks unread: it works on a
    file missing them, which ``load_model`` refuses."""
    time, position, spike_times, features = clusterless_recording
    detector = nld.NonLocalClusterlessDetector(
        clusterless_algorithm_params=algorithm_params("clusterless_kde")
    ).fit(time, position, spike_times, features)
    path = tmp_path / "clusterless.nld"
    detector.save_model(path)
    trimmed = tmp_path / "trimmed.nld"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(trimmed, "w") as target:
        for item in source.infolist():
            if "encoding_spike_waveform_features" not in item.filename:
                target.writestr(item, source.read(item.filename))

    summary = read_model_summary(trimmed)

    (model,) = summary["encoding_models"]
    assert "encoding_spike_waveform_features" not in model
    np.testing.assert_array_equal(
        model["mean_rates"], detector.encoding_model_[("", 0)]["mean_rates"]
    )
    with pytest.raises(DataError, match="missing"):
        nld.NonLocalClusterlessDetector.load_model(trimmed)


def test_summary_of_an_unfitted_detector(tmp_path):
    path = tmp_path / "unfitted.nld"
    nld.NonLocalSortedSpikesDetector().save_model(path)

    summary = read_model_summary(path)

    assert not summary["fitted"]
    assert "environments" not in summary


def test_summary_refuses_a_pickle(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"\x80\x05N.")

    with pytest.raises(ValidationError, match="load_model"):
        read_model_summary(path)
