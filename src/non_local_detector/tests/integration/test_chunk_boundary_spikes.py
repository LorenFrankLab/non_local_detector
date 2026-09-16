"""Regression tests: decoding spikes must survive likelihood chunk boundaries.

``predict(n_chunks > 1)`` recomputes the likelihood once per chunk. When the
likelihood callback receives only ``time[chunk]``, every backend clips the
decoding spikes to that chunk's first and last timestamp, so a spike falling
strictly between the last timestamp of chunk ``k`` and the first timestamp of
chunk ``k + 1`` is dropped by both chunks. These tests place spikes exactly in
those gaps and require the chunked prediction to match the unchunked one.

Row ownership is defined on the full decoding timeline: a spike is in range iff
``time[0] <= t <= time[-1]`` and lands in row ``np.digitize(t, time[1:-1])``.
"""

import numpy as np
import pytest

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

N_CHUNKS = 5
PARITY_KWARGS = {"rtol": 1e-5, "atol": 1e-6}


def chunk_boundary_midpoints(time: np.ndarray, n_chunks: int) -> np.ndarray:
    """Times strictly between the last row of one chunk and the first of the next.

    Mirrors the core's ``np.array_split(np.arange(n_time), n_chunks)`` row split,
    so each returned time is inside a gap that chunk-local clipping discards.
    """
    row_chunks = np.array_split(np.arange(len(time)), n_chunks)
    first_rows = [chunk[0] for chunk in row_chunks[1:]]
    return np.array([0.5 * (time[row - 1] + time[row]) for row in first_rows])


def insert_spike_times(spike_times: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Merge ``extra`` into ``spike_times``, keeping ascending order."""
    return np.sort(np.concatenate([np.asarray(spike_times), extra]))


def missing_rows_straddling_boundaries(time: np.ndarray, n_chunks: int) -> np.ndarray:
    """Mark the last row of each chunk and the first row of the next as missing.

    The core slices ``is_missing`` per chunk while the row-aware callback keeps
    the full ``time``; a mask that straddles every boundary catches a mask that
    is mis-aligned with the rows it is applied to.
    """
    row_chunks = np.array_split(np.arange(len(time)), n_chunks)
    is_missing = np.zeros(len(time), dtype=bool)
    for chunk in row_chunks[1:]:
        is_missing[chunk[0] - 1 : chunk[0] + 1] = True
    return is_missing


def insert_spikes_with_features(
    spike_times: np.ndarray, features: np.ndarray, extra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge ``extra`` spikes (mean waveform features) keeping times/features aligned."""
    spike_times = np.asarray(spike_times)
    features = np.asarray(features)
    extra_features = np.repeat(
        features.mean(axis=0, keepdims=True), len(extra), axis=0
    ).astype(features.dtype)
    all_times = np.concatenate([spike_times, extra])
    all_features = np.concatenate([features, extra_features])
    order = np.argsort(all_times, kind="stable")
    return all_times[order], all_features[order]


@pytest.mark.unit
def test_row_sliced_spike_counts_match_full_time_counts():
    """The recorded chunk-boundary example: row slices must tile the full counts.

    ``time = [0..5]`` with spikes at every bin midpoint. Chunk-local binning
    yields ``[1, 1, 0, 1, 1, 0]`` (the spike at 2.5 is lost); global binning
    must reproduce ``[1, 1, 1, 1, 1, 0]``.
    """
    time = np.arange(6.0)
    spikes = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    full_counts = get_spikecount_per_time_bin(spikes, time)
    np.testing.assert_array_equal(full_counts, [1, 1, 1, 1, 1, 0])

    chunked_counts = np.concatenate(
        [
            get_spikecount_per_time_bin(spikes, time, row_slice=slice(0, 3)),
            get_spikecount_per_time_bin(spikes, time, row_slice=slice(3, 6)),
        ]
    )
    np.testing.assert_array_equal(chunked_counts, full_counts)


@pytest.fixture(scope="module")
def sorted_setup():
    """A fitted sorted-spikes detector plus a decode window with boundary spikes.

    Module-scoped: the tests only call ``predict`` and never mutate the detector.
    """
    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=6)

    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    ).fit(time, position, spike_times, is_training=~is_event)

    decode = slice(10_000, 10_300)
    decode_time = time[decode]
    boundary_times = chunk_boundary_midpoints(decode_time, N_CHUNKS)
    decode_spike_times = [
        insert_spike_times(neuron_spike_times, boundary_times)
        for neuron_spike_times in spike_times
    ]

    return {
        "detector": detector,
        "spike_times": decode_spike_times,
        "time": decode_time,
        "position": position[decode],
        # The full recording, for tests that need to fit an encoding model
        # themselves (the decode window alone is a stationary pause).
        "fit_time": time,
        "fit_position": position,
        "is_training": ~is_event,
        "n_boundary_spikes": len(boundary_times) * len(decode_spike_times),
    }


@pytest.fixture(scope="module")
def clusterless_setup():
    """A fitted clusterless detector plus a decode window with boundary spikes."""
    sim = make_simulated_run_data(n_tetrodes=2, n_runs=2)
    n_encode = int(0.7 * len(sim.position_time))
    encode_end = sim.position_time[n_encode]

    detector = NonLocalClusterlessDetector(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    ).fit(
        sim.position_time[:n_encode],
        sim.position[:n_encode],
        [times[times <= encode_end] for times in sim.spike_times],
        [
            features[times <= encode_end]
            for times, features in zip(
                sim.spike_times, sim.spike_waveform_features, strict=True
            )
        ],
    )

    decode = slice(n_encode, n_encode + 300)
    decode_time = sim.position_time[decode]
    boundary_times = chunk_boundary_midpoints(decode_time, N_CHUNKS)

    decode_spike_times = []
    decode_features = []
    for times, features in zip(
        sim.spike_times, sim.spike_waveform_features, strict=True
    ):
        merged_times, merged_features = insert_spikes_with_features(
            times, features, boundary_times
        )
        decode_spike_times.append(merged_times)
        decode_features.append(merged_features)

    return {
        "detector": detector,
        "spike_times": decode_spike_times,
        "spike_waveform_features": decode_features,
        "time": decode_time,
        "position": sim.position[decode],
        "n_boundary_spikes": len(boundary_times) * len(decode_spike_times),
    }


def assert_results_match(reference, chunked):
    """Posterior, discrete state probabilities and evidence must all agree."""
    np.testing.assert_allclose(
        chunked.acausal_posterior.to_numpy(),
        reference.acausal_posterior.to_numpy(),
        **PARITY_KWARGS,
    )
    np.testing.assert_allclose(
        chunked.acausal_state_probabilities.to_numpy(),
        reference.acausal_state_probabilities.to_numpy(),
        **PARITY_KWARGS,
    )
    np.testing.assert_allclose(
        chunked.attrs["marginal_log_likelihoods"],
        reference.attrs["marginal_log_likelihoods"],
        **PARITY_KWARGS,
    )


@pytest.mark.integration
def test_sorted_chunked_predict_matches_unchunked(sorted_setup):
    """Uncached chunked prediction must not drop boundary spikes (sorted spikes)."""
    detector = sorted_setup["detector"]
    assert sorted_setup["n_boundary_spikes"] > 0

    predict_kwargs = {
        "spike_times": sorted_setup["spike_times"],
        "time": sorted_setup["time"],
        "position": sorted_setup["position"],
        "position_time": sorted_setup["time"],
    }
    reference = detector.predict(**predict_kwargs, n_chunks=1)
    chunked = detector.predict(
        **predict_kwargs, n_chunks=N_CHUNKS, cache_likelihood=False
    )

    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_sorted_chunked_predict_matches_unchunked_with_missing_rows(sorted_setup):
    """Chunked parity must also hold when missing rows straddle the boundaries.

    ``is_missing`` is the one input the core keeps slicing per chunk while the
    likelihood callback now receives the full ``time``, so the mask and the rows
    it zeroes have to stay aligned.
    """
    detector = sorted_setup["detector"]
    time = sorted_setup["time"]
    is_missing = missing_rows_straddling_boundaries(time, N_CHUNKS)
    assert is_missing.sum() == 2 * (N_CHUNKS - 1)

    predict_kwargs = {
        "spike_times": sorted_setup["spike_times"],
        "time": time,
        "position": sorted_setup["position"],
        "position_time": time,
        "is_missing": is_missing,
    }
    reference = detector.predict(**predict_kwargs, n_chunks=1)
    chunked = detector.predict(
        **predict_kwargs, n_chunks=N_CHUNKS, cache_likelihood=False
    )

    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_clusterless_chunked_predict_matches_unchunked(clusterless_setup):
    """Uncached chunked prediction must not drop boundary spikes (clusterless)."""
    detector = clusterless_setup["detector"]
    assert clusterless_setup["n_boundary_spikes"] > 0

    predict_kwargs = {
        "spike_times": clusterless_setup["spike_times"],
        "spike_waveform_features": clusterless_setup["spike_waveform_features"],
        "time": clusterless_setup["time"],
        "position": clusterless_setup["position"],
        "position_time": clusterless_setup["time"],
    }
    reference = detector.predict(**predict_kwargs, n_chunks=1)
    chunked = detector.predict(
        **predict_kwargs, n_chunks=N_CHUNKS, cache_likelihood=False
    )

    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_chunked_predict_returns_requested_log_likelihood(sorted_setup):
    """``return_outputs='log_likelihood'`` must deliver all rows in global order."""
    detector = sorted_setup["detector"]
    predict_kwargs = {
        "spike_times": sorted_setup["spike_times"],
        "time": sorted_setup["time"],
        "position": sorted_setup["position"],
        "position_time": sorted_setup["time"],
        "return_outputs": "log_likelihood",
    }
    reference = detector.predict(**predict_kwargs, n_chunks=1)
    chunked = detector.predict(
        **predict_kwargs, n_chunks=N_CHUNKS, cache_likelihood=False
    )

    assert "log_likelihood" in chunked
    assert chunked.log_likelihood.shape == reference.log_likelihood.shape
    np.testing.assert_allclose(
        chunked.log_likelihood.to_numpy(),
        reference.log_likelihood.to_numpy(),
        **PARITY_KWARGS,
    )


@pytest.mark.integration
def test_covariate_dependent_chunked_predict_matches_unchunked():
    """The covariate-dependent core path must also own the boundary spikes.

    A time-varying discrete transition routes ``_predict`` through
    ``chunked_filter_smoother_covariate_dependent``, a separate chunk loop from
    the stationary driver.
    """
    (
        speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=4)

    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
        discrete_transition_type=DiscreteNonStationaryDiagonal(
            diagonal_values=np.full((4,), 0.98), formula="1 + speed"
        ),
    ).fit(
        time,
        position,
        spike_times,
        is_training=~is_event,
        discrete_transition_covariate_data={"speed": speed},
    )
    assert detector.discrete_state_transitions_.ndim == 3

    decode = slice(10_000, 10_200)
    decode_time = time[decode]
    boundary_times = chunk_boundary_midpoints(decode_time, N_CHUNKS)
    decode_spike_times = [
        insert_spike_times(neuron_spike_times, boundary_times)
        for neuron_spike_times in spike_times
    ]

    predict_kwargs = {
        "spike_times": decode_spike_times,
        "time": decode_time,
        "position": position[decode],
        "position_time": decode_time,
        "discrete_transition_covariate_data": {"speed": speed[decode]},
    }
    reference = detector.predict(**predict_kwargs, n_chunks=1)
    chunked = detector.predict(
        **predict_kwargs, n_chunks=N_CHUNKS, cache_likelihood=False
    )

    assert_results_match(reference, chunked)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:EM did not converge")
def test_chunked_estimate_parameters_returns_requested_log_likelihood(sorted_setup):
    """EM must also deliver a requested log likelihood when chunked.

    ``estimate_parameters`` reaches the chunked drivers through the same
    ``_DetectorBase._estimate_parameters`` final E-step for both detector
    families, so the sorted-spikes case covers the shared code path.
    """
    time = sorted_setup["time"]
    spike_times = sorted_setup["spike_times"]

    # A fresh detector: estimate_parameters refits and mutates the model, so the
    # module-scoped fixture detector must not be used here.
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    )
    results = detector.estimate_parameters(
        position_time=sorted_setup["fit_time"],
        position=sorted_setup["fit_position"],
        spike_times=spike_times,
        time=time,
        is_training=sorted_setup["is_training"],
        max_iter=1,
        n_chunks=N_CHUNKS,
        cache_likelihood=False,
        store_log_likelihood=True,
        return_outputs="log_likelihood",
    )

    n_state_bins = results.acausal_posterior.shape[1]
    assert "log_likelihood" in results
    assert results.log_likelihood.shape == (len(time), n_state_bins)
    assert detector.log_likelihood_ is not None
    assert detector.log_likelihood_.shape == (len(time), n_state_bins)

    # The final E-step runs after the last M-step, so recomputing the likelihood
    # from the fitted model over the full timeline must reproduce the returned
    # rows exactly, in global order.
    expected = np.asarray(
        detector.compute_log_likelihood(
            time, sorted_setup["fit_time"], sorted_setup["fit_position"], spike_times
        )
    )
    np.testing.assert_allclose(
        results.log_likelihood.to_numpy(), expected, **PARITY_KWARGS
    )
