"""A clusterless GMM detector keeps its occupancy threshold through storage.

The fit zeroes the stored bin intensities where the occupancy is unresolved
and records ``log_occupancy_threshold``, which the spike terms and the local
likelihood read at decode. A copy that lost the key would decode a mix of
thresholded bin intensities and unthresholded spike terms, so pickling and
``save_model`` must keep it and decode identically.
"""

import pickle

import numpy as np
import pytest

from non_local_detector import Environment, NonLocalClusterlessDetector

pytestmark = pytest.mark.integration

DT = 0.002


@pytest.fixture(scope="module")
def partial_coverage_detector():
    """A detector fit on 0-150 cm of a 0-300 cm track, and a decode leaving it."""
    rng = np.random.default_rng(1)
    time = np.arange(0.0, 40.0, DT)
    position = (75 + 75 * np.sin(2 * np.pi * time / 20))[:, None]
    spike_times, marks = [], []
    for center, amplitude in ((40.0, 60.0), (120.0, 90.0)):
        rate = 15 * np.exp(-0.5 * ((position[:, 0] - center) / 10) ** 2) + 0.25
        spiking = rng.random(time.size) < rate * DT
        spike_times.append(time[spiking])
        marks.append(amplitude + rng.normal(0, 5, (spiking.sum(), 4)))
    detector = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=3.0, position_range=((0.0, 300.0),)),
        infer_track_interior=False,
        sampling_frequency=1 / DT,
        clusterless_algorithm="clusterless_gmm",
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times, marks)

    decode_time = np.arange(0.0, 1.0, DT)
    decode_position = (140.0 + 110.0 * decode_time / decode_time[-1])[:, None]
    decode_spikes = [np.sort(rng.uniform(0.0, 0.99, 15)) for _ in range(2)]
    decode_marks = [amplitude + rng.normal(0, 5, (15, 4)) for amplitude in (60.0, 90.0)]
    decode = {
        "time_edges": detector.calculate_time_edges((decode_time[0], decode_time[-1])),
        "position_time": decode_time,
        "position": decode_position,
        "return_outputs": "log_likelihood",
    }
    return detector, (decode_spikes, decode_marks), decode


@pytest.mark.parametrize("save", ["pickle", "save_model"])
def test_round_trip_keeps_threshold_and_decode(
    partial_coverage_detector, save, tmp_path
):
    detector, data, decode = partial_coverage_detector
    (entry,) = detector.encoding_model_.values()
    threshold = entry["log_occupancy_threshold"]
    assert np.any(np.asarray(entry["log_occupancy"]) < threshold)
    assert np.any(np.asarray(entry["summed_ground_process_intensity"]) == 0.0)
    expected = detector.predict(*data, **decode)
    assert np.all(np.isfinite(expected.log_likelihood.values))

    path = tmp_path / "model"
    if save == "pickle":
        path.write_bytes(pickle.dumps(detector))
    else:
        detector.save_model(path)
    restored = NonLocalClusterlessDetector.load_model(path)

    (restored_entry,) = restored.encoding_model_.values()
    assert type(restored_entry["log_occupancy_threshold"]) is float
    assert restored_entry["log_occupancy_threshold"] == threshold
    decoded = restored.predict(*data, **decode)
    for name in ("log_likelihood", "acausal_posterior", "acausal_state_probabilities"):
        np.testing.assert_array_equal(decoded[name].values, expected[name].values)
