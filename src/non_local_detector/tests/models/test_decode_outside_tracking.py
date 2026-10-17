"""Decode bins outside the tracked period are missing position, and say so.

Likelihoods read the animal's position by linear interpolation of the tracking
samples, which extrapolates from the first or last interval: tracking 0-9.5 s
at 10 cm/s puts the animal at 300 cm at 30 s. Local models would score that
invented position. A bin that owns any time more than half a sample beyond the
first or last tracking sample is therefore decoded as missing, like a bin with
NaN position, and predict warns once. Models that never read position at
decode (no Local state, no non-local position penalty) ignore the rule.
"""

import warnings

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.environment import Environment
from non_local_detector.models.base import _missing_bins

MESSAGE = "outside the tracked period"
SAMPLING_FREQUENCY = 100.0
TRACKING_END = 9.5


def _outside_tracking_warnings(caught):
    return [w for w in caught if MESSAGE in str(w.message)]


@pytest.mark.unit
def test_missing_bins_marks_bins_more_than_half_a_sample_outside_tracking():
    position_time = np.arange(10) * 0.1  # 0.0 ... 0.9 s
    position = position_time[:, None]
    # Sample-centered bins on 0.1 s steps from -0.1 s to 1.0 s: the first and last
    # reach a full sample beyond the tracking, the rest at most half a sample.
    time_edges = time_edges_from_centers(np.arange(-1, 11) * 0.1)

    is_missing = _missing_bins(time_edges, None, position_time, position)

    expected = np.zeros(len(time_edges) - 1, dtype=bool)
    expected[[0, -1]] = True
    np.testing.assert_array_equal(is_missing, expected)


@pytest.mark.unit
def test_missing_bins_keeps_sample_centered_bins_over_the_tracking():
    position_time = np.arange(10) * 0.1
    position = position_time[:, None]

    is_missing = _missing_bins(
        time_edges_from_centers(position_time), None, position_time, position
    )

    assert is_missing is None


@pytest.mark.unit
def test_missing_bins_combines_outside_tracking_with_caller_mask():
    position_time = np.arange(10) * 0.1
    position = position_time[:, None]
    time_edges = time_edges_from_centers(np.arange(0, 12) * 0.1)
    caller = np.zeros(len(time_edges) - 1, dtype=bool)
    caller[3] = True

    is_missing = _missing_bins(time_edges, caller, position_time, position)

    expected = caller.copy()
    expected[[-2, -1]] = True  # centers 1.0 s and 1.1 s
    np.testing.assert_array_equal(is_missing, expected)


@pytest.fixture(scope="module")
def tracking():
    """Tracking from 0 to 9.5 s at 10 cm/s; spikes continue to 12 s."""
    time = np.arange(0.0, TRACKING_END + 1e-9, 1.0 / SAMPLING_FREQUENCY)
    position = (10.0 * time)[:, None]
    rng = np.random.default_rng(0)
    spike_times = [np.sort(rng.uniform(0.0, 12.0, 120)) for _ in range(3)]
    marks = [rng.normal(size=(len(times), 2)) for times in spike_times]
    return time, position, spike_times, marks


def _detector(kind, tracking, *, reads_position=True):
    """Fit a detector with a Local state, or a decoder that never reads position."""
    time, position, spike_times, marks = tracking
    training_spikes = [times <= TRACKING_END for times in spike_times]
    environment = Environment(place_bin_size=5.0, position_range=((0.0, 100.0),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if kind == "sorted":
            detector_cls = (
                NonLocalSortedSpikesDetector if reads_position else SortedSpikesDecoder
            )
            detector = detector_cls(
                environments=environment,
                sorted_spikes_algorithm_params={
                    "position_std": 6.0,
                    "disable_progress_bar": True,
                },
                sampling_frequency=SAMPLING_FREQUENCY,
            )
            detector.fit(
                time,
                position,
                [
                    times[keep]
                    for times, keep in zip(spike_times, training_spikes, strict=True)
                ],
            )
            return detector, (spike_times,)
        detector_cls = (
            NonLocalClusterlessDetector if reads_position else ClusterlessDecoder
        )
        detector = detector_cls(
            environments=environment,
            clusterless_algorithm_params={
                "position_std": 6.0,
                "waveform_std": 1.0,
                "disable_progress_bar": True,
            },
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        detector.fit(
            time,
            position,
            [
                times[keep]
                for times, keep in zip(spike_times, training_spikes, strict=True)
            ],
            [mark[keep] for mark, keep in zip(marks, training_spikes, strict=True)],
        )
    return detector, (spike_times, marks)


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["sorted", "clusterless"])
@pytest.mark.parametrize("n_chunks", [1, 3])
def test_predict_beyond_tracking_warns_once_and_marks_bins_missing(
    tracking, kind, n_chunks
):
    time, position, _, _ = tracking
    detector, spikes = _detector(kind, tracking)
    decode_time = np.arange(0.0, 12.0, 1.0 / SAMPLING_FREQUENCY)
    time_edges = time_edges_from_centers(decode_time)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = detector.predict(
            *spikes,
            time_edges=time_edges,
            position_time=time,
            position=position,
            n_chunks=n_chunks,
            return_outputs="log_likelihood",
        )

    assert len(_outside_tracking_warnings(caught)) == 1
    log_likelihood = results.log_likelihood.values
    half_sample = 0.5 / SAMPLING_FREQUENCY
    outside = time_edges[1:] > TRACKING_END + half_sample * (1 + 1e-6)
    assert outside.sum() == 249  # bins centered at 9.51 ... 11.99 s
    np.testing.assert_array_equal(log_likelihood[outside], 0.0)
    assert not np.any(np.all(log_likelihood[~outside] == 0.0, axis=1))
    posterior = results.acausal_posterior.values
    assert np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0, rtol=1e-5)


@pytest.mark.integration
def test_predict_over_the_tracking_does_not_warn(tracking):
    time, position, _, _ = tracking
    detector, spikes = _detector("sorted", tracking)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector.predict(
            *spikes,
            time_edges=time_edges_from_centers(time),
            position_time=time,
            position=position,
        )

    assert not _outside_tracking_warnings(caught)


@pytest.mark.integration
def test_compute_log_likelihood_beyond_tracking_warns_and_masks(tracking):
    time, position, spike_times, _ = tracking
    detector, _ = _detector("sorted", tracking)
    time_edges = time_edges_from_centers(np.arange(0.0, 12.0, 1.0 / SAMPLING_FREQUENCY))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log_likelihood = np.asarray(
            detector.compute_log_likelihood(
                time, position, spike_times, time_edges=time_edges
            )
        )

    assert len(_outside_tracking_warnings(caught)) == 1
    np.testing.assert_array_equal(log_likelihood[-249:], 0.0)


@pytest.mark.integration
def test_estimate_parameters_beyond_tracking_warns_once(tracking):
    time, position, spike_times, _ = tracking
    detector, _ = _detector("sorted", tracking)
    time_edges = time_edges_from_centers(np.arange(0.0, 12.0, 1.0 / SAMPLING_FREQUENCY))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector.estimate_parameters(
            time,
            position,
            [times[times <= TRACKING_END] for times in spike_times],
            time_edges=time_edges,
            max_iter=2,
        )

    assert len(_outside_tracking_warnings(caught)) == 1


def _decode_edges():
    return time_edges_from_centers(np.arange(0.0, 12.0, 1.0 / SAMPLING_FREQUENCY))


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["sorted", "clusterless"])
def test_predict_without_position_reads_ignores_the_tracking_range(tracking, kind):
    """A model that never reads position decodes every bin, as without it."""
    time, position, _, _ = tracking
    detector, spikes = _detector(kind, tracking, reads_position=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_position = detector.predict(
            *spikes,
            time_edges=_decode_edges(),
            position_time=time,
            position=position,
            return_outputs="log_likelihood",
        )
    without_position = detector.predict(
        *spikes, time_edges=_decode_edges(), return_outputs="log_likelihood"
    )

    assert not _outside_tracking_warnings(caught)
    np.testing.assert_array_equal(
        with_position.log_likelihood.values, without_position.log_likelihood.values
    )
    np.testing.assert_array_equal(
        with_position.acausal_posterior.values,
        without_position.acausal_posterior.values,
    )


@pytest.mark.integration
def test_compute_log_likelihood_without_position_reads_ignores_the_tracking_range(
    tracking,
):
    time, position, spike_times, _ = tracking
    detector, _ = _detector("sorted", tracking, reads_position=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_position = detector.compute_log_likelihood(
            time, position, spike_times, time_edges=_decode_edges()
        )
    without_position = detector.compute_log_likelihood(
        None, None, spike_times, time_edges=_decode_edges()
    )

    assert not _outside_tracking_warnings(caught)
    np.testing.assert_array_equal(
        np.asarray(with_position), np.asarray(without_position)
    )


@pytest.mark.integration
def test_most_likely_sequence_without_position_reads_ignores_the_tracking_range(
    tracking,
):
    time, position, spike_times, _ = tracking
    detector, _ = _detector("sorted", tracking, reads_position=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_position = detector.most_likely_sequence(
            spike_times,
            time_edges=_decode_edges(),
            position_time=time,
            position=position,
        )
    without_position = detector.most_likely_sequence(
        spike_times, time_edges=_decode_edges()
    )

    assert not _outside_tracking_warnings(caught)
    np.testing.assert_array_equal(
        np.asarray(with_position), np.asarray(without_position)
    )


@pytest.mark.integration
def test_estimate_parameters_without_position_reads_decodes_every_bin(tracking):
    """EM still fits on the tracking but its E-step masks no bin."""
    time, position, spike_times, _ = tracking
    detector, _ = _detector("sorted", tracking, reads_position=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = detector.estimate_parameters(
            time,
            position,
            [times[times <= TRACKING_END] for times in spike_times],
            time_edges=_decode_edges(),
            max_iter=1,
            return_outputs="log_likelihood",
        )

    assert not _outside_tracking_warnings(caught)
    # A missing bin's log likelihood is 0; a decoded one includes the
    # negative no-spike term.
    assert not np.any(np.all(results.log_likelihood.values == 0.0, axis=1))
