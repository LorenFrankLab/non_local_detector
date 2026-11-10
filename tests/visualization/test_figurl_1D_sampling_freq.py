"""The 1-D figurl reads its sampling rate from the posterior's time stamps."""

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.visualization.figurl_1D import get_sampling_freq

pytestmark = pytest.mark.unit

UNIX_START = 1_634_567_890.123


@pytest.mark.parametrize(
    "sampling_frequency", [250, 300, 400, 500, 600, 1000, 1500, 2000, 3000]
)
@pytest.mark.parametrize("start_time", [0.0, UNIX_START])
def test_sampling_rate_is_recovered(sampling_frequency, start_time):
    """Rates whose sample interval is not a whole millisecond are exact too."""
    times = start_time + np.arange(20_000) / sampling_frequency

    np.testing.assert_allclose(get_sampling_freq(times), sampling_frequency, rtol=2e-8)


def test_a_gap_in_the_times_does_not_change_the_rate():
    times = UNIX_START + np.arange(20_000) / 600.0
    times = np.delete(times, np.arange(5_000, 6_000))

    np.testing.assert_allclose(get_sampling_freq(times), 600.0, rtol=2e-8)


def test_a_long_pause_does_not_change_the_rate():
    """A 100 s pause between two 10 s recordings at 500 Hz."""
    segment = np.arange(5_000) / 500.0
    times = UNIX_START + np.concatenate([segment, segment + 110.0])

    np.testing.assert_allclose(get_sampling_freq(times), 500.0, rtol=2e-8)


@pytest.mark.parametrize(
    ("sampling_frequency", "resolution", "rounding"),
    [
        (30.0, 1e-3, np.round),  # video frames stamped to the millisecond
        (300.0, 1e-3, np.floor),
        (30_000.0, 1e-5, np.round),
    ],
)
def test_timestamps_coarser_than_the_interval_give_the_rate(
    sampling_frequency, resolution, rounding
):
    """Rounding the stamps changes single intervals, not their average.

    At 30 Hz stamped to 1 ms the intervals are 33 or 34 ms; the rate is
    their average, 30 Hz, not that of the more common 33 ms interval.
    """
    times = rounding(np.arange(20_000) / sampling_frequency / resolution) * resolution

    # Rounding moves the first and last stamps by up to one resolution step.
    span = times[-1] - times[0]
    np.testing.assert_allclose(
        get_sampling_freq(times), sampling_frequency, rtol=2 * resolution / span
    )


def test_dropped_frames_give_the_frame_rate():
    """Every third frame of a 500 Hz recording dropped.

    The intervals alternate 2 ms and 4 ms. The answer is the rate at which
    frames are taken, 500 Hz, not the average rate of the kept frames
    (333.3 Hz): a 4 ms interval is one missing frame.
    """
    times = np.arange(20_000) / 500.0
    times = np.delete(times, np.arange(0, times.size, 3))

    np.testing.assert_allclose(get_sampling_freq(times), 500.0, rtol=2e-8)


def test_randomly_dropped_frames_give_the_frame_rate():
    rng = np.random.default_rng(0)
    frames = np.arange(20_000) / 500.0
    times = np.sort(rng.choice(frames, 8_000, replace=False))

    np.testing.assert_allclose(get_sampling_freq(times), 500.0, rtol=2e-8)


def test_jittered_timestamps_give_the_rate():
    """Each stamp jittered with a standard deviation of 5% of the interval."""
    rng = np.random.default_rng(0)
    frames = np.arange(20_000) / 500.0
    times = UNIX_START + frames + rng.normal(0.0, 0.05 / 500.0, frames.size)

    # The span's error is the end stamps' jitter: about 0.07 / 20,000.
    np.testing.assert_allclose(get_sampling_freq(times), 500.0, rtol=2e-5)


@pytest.mark.parametrize(
    "times",
    [
        np.array([]),
        np.array([0.0]),
        np.array([0.0, 0.002, 0.002, 0.004]),
        np.array([0.004, 0.002, 0.0]),
        np.array([0.0, np.nan, 0.004]),
    ],
    ids=["empty", "one sample", "repeated time", "decreasing", "nan"],
)
def test_times_that_give_no_rate_are_refused(times):
    with pytest.raises(ValidationError, match="times"):
        get_sampling_freq(times)
