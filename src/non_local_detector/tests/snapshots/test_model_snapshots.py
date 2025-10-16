"""Snapshot tests for decoder models using simulated data.

These tests use the syrupy snapshot testing framework to detect unintended
changes in model behavior. Snapshots capture expected outputs for regression
testing during refactoring and development.

The tests are based on the simulation framework from test_simulation.ipynb.
"""

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from non_local_detector import (
    ContFragSortedSpikesClassifier,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture
def simulated_data():
    """Generate simulated data with fixed random seed for reproducibility."""
    np.random.seed(42)  # Fixed seed for reproducibility
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        sampling_frequency,
        is_event,
        place_fields,
    ) = make_simulated_data(n_neurons=30)
    return {
        "speed": speed,
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "event_times": event_times,
        "sampling_frequency": sampling_frequency,
        "is_event": is_event,
        "place_fields": place_fields,
    }


def serialize_xarray_summary(data_array):
    """Serialize xarray DataArray to summary statistics for snapshot comparison.

    For large arrays, we snapshot summary statistics rather than full data
    to keep snapshot files manageable and readable.
    """
    arr = np.asarray(data_array)
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        # Sample a few values for additional verification
        "first_5": arr.ravel()[:5].tolist() if arr.size >= 5 else arr.ravel().tolist(),
        "last_5": arr.ravel()[-5:].tolist() if arr.size >= 5 else arr.ravel().tolist(),
    }


def serialize_state_probabilities(state_probs):
    """Serialize state probabilities for snapshot comparison."""
    return {
        "shape": state_probs.shape,
        "states": list(state_probs.states.values),
        "mean_per_state": {
            str(state): float(np.mean(state_probs.sel(states=state)))
            for state in state_probs.states.values
        },
        "max_per_state": {
            str(state): float(np.max(state_probs.sel(states=state)))
            for state in state_probs.states.values
        },
        "min_per_state": {
            str(state): float(np.min(state_probs.sel(states=state)))
            for state in state_probs.states.values
        },
    }


@pytest.mark.snapshot
def test_nonlocal_sorted_spikes_detector_snapshot(
    simulated_data: dict, snapshot: SnapshotAssertion
):
    """Snapshot test for NonLocalSortedSpikesDetector.

    This test verifies that the NonLocalSortedSpikesDetector produces
    consistent outputs across code changes. It tests:
    - State probability predictions
    - Posterior distributions (summary statistics)
    - Model fitting results
    """
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        simulated_data["time"],
        simulated_data["position"],
        simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )

    results = detector.predict(
        spike_times=simulated_data["spike_times"],
        time=simulated_data["time"],
        position=simulated_data["position"],
        position_time=simulated_data["time"],
        save_log_likelihood_to_results=True,
    )

    # Snapshot state probabilities
    state_probs_summary = serialize_state_probabilities(
        results.acausal_state_probabilities
    )
    assert state_probs_summary == snapshot(name="state_probabilities")

    # Snapshot posterior summary (full posterior is too large)
    posterior_summary = serialize_xarray_summary(results.acausal_posterior)
    assert posterior_summary == snapshot(name="posterior_summary")

    # Snapshot model attributes
    model_summary = {
        "n_position_bins": detector.environments[0].place_bin_centers_.shape[0],
        "discrete_state_transitions_shape": detector.discrete_state_transitions_.shape,
        "discrete_state_transitions_sum_axis1": detector.discrete_state_transitions_.sum(
            axis=1
        ).tolist(),
        "initial_conditions_sum": float(detector.initial_conditions_.sum()),
        "initial_conditions_shape": detector.initial_conditions_.shape,
    }
    assert model_summary == snapshot(name="model_summary")


@pytest.mark.snapshot
def test_contfrag_sorted_spikes_classifier_snapshot(
    simulated_data: dict, snapshot: SnapshotAssertion
):
    """Snapshot test for ContFragSortedSpikesClassifier.

    This test verifies that the ContFragSortedSpikesClassifier produces
    consistent continuous vs. fragmented classification results.
    """
    classifier = ContFragSortedSpikesClassifier(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        position_time=simulated_data["time"],
        position=simulated_data["position"],
        spike_times=simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )

    results = classifier.predict(
        spike_times=simulated_data["spike_times"],
        time=simulated_data["time"],
    )

    # Snapshot state probabilities
    state_probs_summary = serialize_state_probabilities(
        results.acausal_state_probabilities
    )
    assert state_probs_summary == snapshot(name="state_probabilities")

    # Snapshot posterior summary
    posterior_summary = serialize_xarray_summary(results.acausal_posterior)
    assert posterior_summary == snapshot(name="posterior_summary")

    # Snapshot continuous state transitions (should sum to 2 along axis 1)
    transitions_summary = {
        "shape": classifier.continuous_state_transitions_.shape,
        "sum_axis1_mean": float(
            np.mean(classifier.continuous_state_transitions_.sum(axis=1))
        ),
        "sum_axis1_std": float(
            np.std(classifier.continuous_state_transitions_.sum(axis=1))
        ),
        "sum_axis1_min": float(
            np.min(classifier.continuous_state_transitions_.sum(axis=1))
        ),
        "sum_axis1_max": float(
            np.max(classifier.continuous_state_transitions_.sum(axis=1))
        ),
    }
    assert transitions_summary == snapshot(name="transitions_summary")


@pytest.mark.snapshot
def test_sorted_spikes_decoder_snapshot(
    simulated_data: dict, snapshot: SnapshotAssertion
):
    """Snapshot test for SortedSpikesDecoder.

    This test verifies that the basic SortedSpikesDecoder produces
    consistent position decoding results.
    """
    decoder = SortedSpikesDecoder(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        position_time=simulated_data["time"],
        position=simulated_data["position"],
        spike_times=simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )

    results = decoder.predict(
        spike_times=simulated_data["spike_times"],
        time=simulated_data["time"],
        position=simulated_data["position"],
        position_time=simulated_data["time"],
        save_log_likelihood_to_results=True,
    )

    # Snapshot state probabilities (should be high for continuous state)
    state_probs_summary = {
        "shape": results.acausal_state_probabilities.shape,
        "mean": float(np.mean(results.acausal_state_probabilities)),
        "min": float(np.min(results.acausal_state_probabilities)),
        "max": float(np.max(results.acausal_state_probabilities)),
    }
    assert state_probs_summary == snapshot(name="state_probabilities")

    # Snapshot posterior summary
    posterior_summary = serialize_xarray_summary(results.acausal_posterior)
    assert posterior_summary == snapshot(name="posterior_summary")


@pytest.mark.snapshot
@pytest.mark.parametrize(
    "algorithm,algorithm_params",
    [
        ("sorted_spikes_kde", {"position_std": 6.0, "block_size": int(2**12)}),
    ],
)
def test_nonlocal_detector_different_algorithms_snapshot(
    simulated_data: dict,
    algorithm: str,
    algorithm_params: dict,
    snapshot: SnapshotAssertion,
):
    """Snapshot test for NonLocalSortedSpikesDetector with different algorithms.

    This test verifies consistent behavior across different likelihood algorithms.
    Note: GLM algorithm is excluded as it requires additional environment parameters.
    """
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm=algorithm,
        sorted_spikes_algorithm_params=algorithm_params,
    ).fit(
        simulated_data["time"],
        simulated_data["position"],
        simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )

    results = detector.predict(
        spike_times=simulated_data["spike_times"],
        time=simulated_data["time"],
        position=simulated_data["position"],
        position_time=simulated_data["time"],
    )

    # Snapshot state probabilities
    state_probs_summary = serialize_state_probabilities(
        results.acausal_state_probabilities
    )
    assert state_probs_summary == snapshot


@pytest.mark.snapshot
def test_detector_encoding_model_snapshot(
    simulated_data: dict, snapshot: SnapshotAssertion
):
    """Snapshot test for encoding model properties.

    This test verifies that the fitted encoding models have expected properties.
    """
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(
        simulated_data["time"],
        simulated_data["position"],
        simulated_data["spike_times"],
        is_training=~simulated_data["is_event"],
    )

    # Get encoding model for first environment
    enc_model = detector.encoding_model_[("", 0)]

    # Snapshot place fields summary
    place_fields = enc_model["place_fields"]
    place_fields_summary = {
        "n_neurons": len(place_fields),
        "place_field_shape": place_fields[0].shape if len(place_fields) > 0 else None,
        "mean_rates": [float(np.mean(pf)) for pf in place_fields[:5]],  # First 5
        "max_rates": [float(np.max(pf)) for pf in place_fields[:5]],
        "argmax_positions": [int(np.argmax(pf)) for pf in place_fields[:5]],
    }
    assert place_fields_summary == snapshot(name="place_fields_summary")

    # Snapshot occupancy
    occupancy_summary = {
        "shape": enc_model["occupancy"].shape,
        "mean": float(np.mean(enc_model["occupancy"])),
        "min": float(np.min(enc_model["occupancy"])),
        "max": float(np.max(enc_model["occupancy"])),
        "sum": float(np.sum(enc_model["occupancy"])),
    }
    assert occupancy_summary == snapshot(name="occupancy_summary")
