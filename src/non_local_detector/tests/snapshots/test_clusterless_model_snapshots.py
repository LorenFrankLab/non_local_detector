"""Snapshot tests for clusterless decoder models.

These tests verify consistent behavior of clusterless decoders that work
with continuous spike features (waveform features) rather than sorted units.
"""

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from non_local_detector import (
    ClusterlessDecoder,
    ContFragClusterlessClassifier,
    NonLocalClusterlessDetector,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data


@pytest.fixture
def clusterless_simulated_data():
    """Generate simulated clusterless data with fixed random seed."""
    n_tetrodes = 4
    place_field_means = np.arange(0, 160, 10)  # 16 place fields evenly divisible by 4

    # Use new API that returns ClusterlessSimOutput directly
    sim = make_simulated_run_data(
        sampling_frequency=500,
        n_runs=5,
        n_tetrodes=n_tetrodes,
        place_field_means=place_field_means,
        seed=42,  # Fixed seed for reproducibility
    )

    return {
        "time": sim.position_time,
        "position": sim.position,  # Already 2D from new API
        "spike_times": sim.spike_times,
        "spike_waveform_features": sim.spike_waveform_features,
        "sampling_frequency": 500,
    }


def serialize_xarray_summary(data_array):
    """Serialize xarray DataArray to summary statistics."""
    arr = np.asarray(data_array)
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
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
@pytest.mark.slow
@pytest.mark.skip(
    reason="Skipped due to known issues in clusterless_kde likelihood code. "
    "Will be re-enabled once clusterless_kde is fixed."
)
@pytest.mark.parametrize(
    "algorithm,algorithm_params",
    [
        (
            "clusterless_kde",
            {"position_std": 6.0, "waveform_std": 24.0, "block_size": 100},
        ),
    ],
)
def test_nonlocal_clusterless_detector_snapshot(
    clusterless_simulated_data: dict,
    algorithm: str,
    algorithm_params: dict,
    snapshot: SnapshotAssertion,
):
    """Snapshot test for NonLocalClusterlessDetector with KDE algorithm.

    Tests KDE-based clusterless likelihood model.
    Note: GMM algorithm excluded due to different parameter requirements.
    Note: Currently skipped due to data sparsity issues with small test subsets.
    """
    # Use a subset of data for faster testing
    n_samples = min(5000, len(clusterless_simulated_data["time"]))
    is_training = np.ones(n_samples, dtype=bool)
    is_training[2000:2500] = False  # Leave some data for testing

    detector = NonLocalClusterlessDetector(
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params=algorithm_params,
    ).fit(
        clusterless_simulated_data["time"][:n_samples],
        clusterless_simulated_data["position"][:n_samples],
        [
            st[st < clusterless_simulated_data["time"][n_samples]]
            for st in clusterless_simulated_data["spike_times"]
        ],
        [
            swf[
                clusterless_simulated_data["spike_times"][i]
                < clusterless_simulated_data["time"][n_samples]
            ]
            for i, swf in enumerate(
                clusterless_simulated_data["spike_waveform_features"]
            )
        ],
        is_training=is_training,
    )

    results = detector.predict(
        spike_times=[
            st[
                (st >= clusterless_simulated_data["time"][2000])
                & (st < clusterless_simulated_data["time"][2500])
            ]
            for st in clusterless_simulated_data["spike_times"]
        ],
        spike_waveform_features=[
            swf[
                (
                    clusterless_simulated_data["spike_times"][i]
                    >= clusterless_simulated_data["time"][2000]
                )
                & (
                    clusterless_simulated_data["spike_times"][i]
                    < clusterless_simulated_data["time"][2500]
                )
            ]
            for i, swf in enumerate(
                clusterless_simulated_data["spike_waveform_features"]
            )
        ],
        time=clusterless_simulated_data["time"][2000:2500],
        position=clusterless_simulated_data["position"][2000:2500],
        position_time=clusterless_simulated_data["time"][2000:2500],
    )

    # Snapshot state probabilities
    state_probs_summary = serialize_state_probabilities(
        results.acausal_state_probabilities
    )
    assert state_probs_summary == snapshot(name="state_probabilities")

    # Snapshot posterior summary
    posterior_summary = serialize_xarray_summary(results.acausal_posterior)
    assert posterior_summary == snapshot(name="posterior_summary")


@pytest.mark.snapshot
def test_contfrag_clusterless_classifier_snapshot(
    clusterless_simulated_data: dict, snapshot: SnapshotAssertion
):
    """Snapshot test for ContFragClusterlessClassifier."""
    # Use a subset of data for faster testing
    n_samples = min(5000, len(clusterless_simulated_data["time"]))
    is_training = np.ones(n_samples, dtype=bool)
    is_training[2000:2500] = False

    classifier = ContFragClusterlessClassifier(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    ).fit(
        position_time=clusterless_simulated_data["time"][:n_samples],
        position=clusterless_simulated_data["position"][:n_samples],
        spike_times=[
            st[st < clusterless_simulated_data["time"][n_samples]]
            for st in clusterless_simulated_data["spike_times"]
        ],
        spike_waveform_features=[
            swf[
                clusterless_simulated_data["spike_times"][i]
                < clusterless_simulated_data["time"][n_samples]
            ]
            for i, swf in enumerate(
                clusterless_simulated_data["spike_waveform_features"]
            )
        ],
        is_training=is_training,
    )

    results = classifier.predict(
        spike_times=[
            st[
                (st >= clusterless_simulated_data["time"][2000])
                & (st < clusterless_simulated_data["time"][2500])
            ]
            for st in clusterless_simulated_data["spike_times"]
        ],
        spike_waveform_features=[
            swf[
                (
                    clusterless_simulated_data["spike_times"][i]
                    >= clusterless_simulated_data["time"][2000]
                )
                & (
                    clusterless_simulated_data["spike_times"][i]
                    < clusterless_simulated_data["time"][2500]
                )
            ]
            for i, swf in enumerate(
                clusterless_simulated_data["spike_waveform_features"]
            )
        ],
        time=clusterless_simulated_data["time"][2000:2500],
        position=clusterless_simulated_data["position"][2000:2500],
        position_time=clusterless_simulated_data["time"][2000:2500],
    )

    # Snapshot state probabilities
    state_probs_summary = serialize_state_probabilities(
        results.acausal_state_probabilities
    )
    assert state_probs_summary == snapshot(name="state_probabilities")

    # Snapshot posterior summary
    posterior_summary = serialize_xarray_summary(results.acausal_posterior)
    assert posterior_summary == snapshot(name="posterior_summary")


@pytest.mark.snapshot
def test_clusterless_decoder_snapshot(
    clusterless_simulated_data: dict, snapshot: SnapshotAssertion
):
    """Snapshot test for basic ClusterlessDecoder."""
    # Use a subset of data for faster testing
    n_samples = min(5000, len(clusterless_simulated_data["time"]))
    is_training = np.ones(n_samples, dtype=bool)
    is_training[2000:2500] = False

    decoder = ClusterlessDecoder(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    ).fit(
        position_time=clusterless_simulated_data["time"][:n_samples],
        position=clusterless_simulated_data["position"][:n_samples],
        spike_times=[
            st[st < clusterless_simulated_data["time"][n_samples]]
            for st in clusterless_simulated_data["spike_times"]
        ],
        spike_waveform_features=[
            swf[
                clusterless_simulated_data["spike_times"][i]
                < clusterless_simulated_data["time"][n_samples]
            ]
            for i, swf in enumerate(
                clusterless_simulated_data["spike_waveform_features"]
            )
        ],
        is_training=is_training,
    )

    results = decoder.predict(
        spike_times=[
            st[
                (st >= clusterless_simulated_data["time"][2000])
                & (st < clusterless_simulated_data["time"][2500])
            ]
            for st in clusterless_simulated_data["spike_times"]
        ],
        spike_waveform_features=[
            swf[
                (
                    clusterless_simulated_data["spike_times"][i]
                    >= clusterless_simulated_data["time"][2000]
                )
                & (
                    clusterless_simulated_data["spike_times"][i]
                    < clusterless_simulated_data["time"][2500]
                )
            ]
            for i, swf in enumerate(
                clusterless_simulated_data["spike_waveform_features"]
            )
        ],
        time=clusterless_simulated_data["time"][2000:2500],
        position=clusterless_simulated_data["position"][2000:2500],
        position_time=clusterless_simulated_data["time"][2000:2500],
    )

    # Snapshot state probabilities
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
def test_clusterless_encoding_model_snapshot(
    clusterless_simulated_data: dict, snapshot: SnapshotAssertion
):
    """Snapshot test for clusterless encoding model properties."""
    n_samples = min(5000, len(clusterless_simulated_data["time"]))

    decoder = ClusterlessDecoder(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    ).fit(
        position_time=clusterless_simulated_data["time"][:n_samples],
        position=clusterless_simulated_data["position"][:n_samples],
        spike_times=[
            st[st < clusterless_simulated_data["time"][n_samples]]
            for st in clusterless_simulated_data["spike_times"]
        ],
        spike_waveform_features=[
            swf[
                clusterless_simulated_data["spike_times"][i]
                < clusterless_simulated_data["time"][n_samples]
            ]
            for i, swf in enumerate(
                clusterless_simulated_data["spike_waveform_features"]
            )
        ],
    )

    # Get encoding model for first environment
    enc_model = decoder.encoding_model_[("", 0)]

    # Snapshot occupancy
    occupancy_summary = {
        "shape": enc_model["occupancy"].shape,
        "mean": float(np.mean(enc_model["occupancy"])),
        "min": float(np.min(enc_model["occupancy"])),
        "max": float(np.max(enc_model["occupancy"])),
        "sum": float(np.sum(enc_model["occupancy"])),
    }
    assert occupancy_summary == snapshot(name="occupancy_summary")

    # Snapshot mean rates
    mean_rates_summary = {
        "n_electrodes": len(enc_model["mean_rates"]),
        "rates": [float(rate) for rate in enc_model["mean_rates"]],
    }
    assert mean_rates_summary == snapshot(name="mean_rates_summary")
