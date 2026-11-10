"""Diffusion detectors saved by main 11d9b3de, before the finite-volume operator.

``tests/compat_data/main-11d9b3de`` holds ``NonLocal*Detector``s fitted by that
main, the last whose graph diffusion coupled every pair of arm end bins at a
junction with ``1 / distance**2`` and ignored bin widths, on a T-maze with
unequal bins (``generate_diffusion_track_artifacts.py``), each as a plain
pickle (as ``save_model`` wrote models up to 0.6.9) and as that main's saved
model file. Sorted-spikes diffusion and MRF models decode from their stored
place fields, so they must decode as main did. A clusterless diffusion model
decodes by rebuilding the heat kernel from its environment, which would mix
the earlier operator's fitted occupancy with the new kernel, so decoding it
asks for a refit; fitting it again gives the new operator's model.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from non_local_detector import time_edges_from_centers
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.diffusion import DIFFUSION_OPERATOR
from non_local_detector.models.base import ClusterlessDetector, SortedSpikesDetector
from tests.conftest import em_stops_at_max_iter

pytestmark = pytest.mark.integration

ARTIFACTS = Path(__file__).parents[1] / "compat_data" / "main-11d9b3de"
SORTED = ("nonlocal_sorted_spikes_diffusion_tmaze", "nonlocal_sorted_spikes_mrf_tmaze")
CLUSTERLESS = "nonlocal_clusterless_diffusion_tmaze"
FORMS = ("pickle", "model")


@pytest.fixture(scope="module")
def recording():
    return pickle.loads((ARTIFACTS / "diffusion_tmaze_inputs.pkl").read_bytes())


def _load(name, form, detector_class):
    return detector_class.load_model(ARTIFACTS / f"{name}_{form}.pkl")


def _window(times, low, high):
    return [t[(t >= low) & (t < high)] for t in times]


def _window_features(times, features, low, high):
    return [f[(t >= low) & (t < high)] for t, f in zip(times, features, strict=True)]


def _decode(detector, recording, clusterless=False):
    """Decode the samples after the training ones, as main did."""
    n_train = recording["n_train"]
    decode_time = recording["time"][n_train:]
    time_edges = time_edges_from_centers(decode_time)
    low, high = decode_time[0], time_edges[-1] + 1e-9
    spikes = (
        (
            _window(recording["electrode_spike_times"], low, high),
            _window_features(
                recording["electrode_spike_times"],
                recording["electrode_features"],
                low,
                high,
            ),
        )
        if clusterless
        else (_window(recording["spike_times"], low, high),)
    )
    return detector.predict(
        *spikes,
        time_edges=time_edges,
        position_time=decode_time,
        position=recording["position"][n_train:],
    ).acausal_posterior.values


def _training_data(recording):
    n_train = recording["n_train"]
    time = recording["time"]
    end = time[n_train]
    return (
        time[:n_train],
        recording["position"][:n_train],
        _window(recording["electrode_spike_times"], 0.0, end),
        _window_features(
            recording["electrode_spike_times"],
            recording["electrode_features"],
            0.0,
            end,
        ),
    )


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("name", SORTED)
def test_sorted_diffusion_detector_saved_by_main_decodes_as_main_did(
    name, form, recording
):
    """A sorted-spikes diffusion or MRF detector fitted by 11d9b3de decodes,
    Local state included, to 11d9b3de's own posterior."""
    detector = _load(name, form, SortedSpikesDetector)

    posterior = _decode(detector, recording)

    reference = pickle.loads(
        (ARTIFACTS / "diffusion_tmaze_reference.pkl").read_bytes()
    )[name]
    np.testing.assert_allclose(posterior, reference, rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize("form", FORMS)
def test_clusterless_diffusion_detector_saved_by_main_asks_for_a_refit(form, recording):
    """A clusterless diffusion detector fitted by 11d9b3de loads, records no
    diffusion operator, and decoding it asks for a refit."""
    detector = _load(CLUSTERLESS, form, ClusterlessDetector)
    for encoding_model in detector.encoding_model_.values():
        assert "diffusion_operator" not in encoding_model

    with pytest.raises(ValidationError, match="diffusion operator") as error:
        _decode(detector, recording, clusterless=True)
    assert "Refit the model" in str(error.value)


def test_refitting_the_encoding_of_a_main_clusterless_detector_uses_the_new_operator(
    recording,
):
    """``fit_encoding_model`` on the detector fitted by 11d9b3de, whose
    environment still caches the earlier operator, gives the encoding model a
    fresh fit gives, with the operator recorded, and it decodes."""
    detector = _load(CLUSTERLESS, "pickle", ClusterlessDetector)
    fresh = clone(detector).fit(*_training_data(recording))

    detector.fit_encoding_model(*_training_data(recording))

    for key, encoding_model in detector.encoding_model_.items():
        assert encoding_model["diffusion_operator"] == DIFFUSION_OPERATOR
        expected = fresh.encoding_model_[key]
        for array in ("occupancy", "summed_ground_process_intensity", "bin_sizes"):
            np.testing.assert_array_equal(encoding_model[array], expected[array])
    np.testing.assert_array_equal(
        _decode(detector, recording, clusterless=True),
        _decode(fresh, recording, clusterless=True),
    )


@em_stops_at_max_iter
@pytest.mark.parametrize("estimate_encoding_model", [False, True])
def test_estimating_parameters_of_a_main_clusterless_detector_refits_it_first(
    estimate_encoding_model, recording
):
    """``estimate_parameters`` fits the detector, encoding included, before
    any E-step, so on the detector fitted by 11d9b3de it runs as on a fresh
    detector: the earlier operator's statistics are never decoded."""
    detector = _load(CLUSTERLESS, "pickle", ClusterlessDetector)
    fresh = clone(detector)
    time, position, spike_times, features = _training_data(recording)
    options = {
        "time_edges": time_edges_from_centers(time),
        "max_iter": 1,
        "estimate_encoding_model": estimate_encoding_model,
    }

    results = detector.estimate_parameters(
        time, position, spike_times, features, **options
    )
    fresh_results = fresh.estimate_parameters(
        time, position, spike_times, features, **options
    )

    for encoding_model in detector.encoding_model_.values():
        assert encoding_model["diffusion_operator"] == DIFFUSION_OPERATOR
    np.testing.assert_array_equal(
        results.acausal_posterior.values, fresh_results.acausal_posterior.values
    )
