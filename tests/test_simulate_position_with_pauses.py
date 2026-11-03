"""``simulate_position_with_pauses`` pauses once per arrival at the track end."""

import numpy as np
import pytest

from non_local_detector.simulate import simulate_position_with_pauses

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sampling_frequency", [50, 500])
def test_one_pause_per_arrival_at_track_end(sampling_frequency):
    track, speed, pause = 180.0, 30.0, 5.0
    n_seconds = 240
    time = np.arange(n_seconds * sampling_frequency) / sampling_frequency
    position = simulate_position_with_pauses(
        time, track, speed, pause, sampling_frequency
    )

    assert position.shape == time.shape
    # Without pauses the animal reaches the far end every 2*track/speed seconds,
    # first at track/speed. Each arrival adds `pause` seconds of waiting, so
    # arrival k (from 0) starts at track/speed + k*(2*track/speed + pause).
    sweep = track / speed
    cycle = 2 * sweep + pause
    starts = sweep + cycle * np.arange(int(n_seconds // cycle) + 1)
    expected_paused = np.zeros(time.size, dtype=bool)
    for start in starts:
        expected_paused |= (time >= start) & (time < start + pause)

    # Samples next to the end are within isclose tolerance of it, not equal.
    is_at_end = np.isclose(position, track)
    np.testing.assert_array_equal(is_at_end[expected_paused], True)
    assert abs(is_at_end.mean() - expected_paused.mean()) < 0.01
    assert is_at_end.mean() < 0.4


def test_each_arrival_is_a_single_run_of_the_pause_length():
    sampling_frequency, pause = 100, 2.0
    time = np.arange(100 * sampling_frequency) / sampling_frequency
    position = simulate_position_with_pauses(
        time, 60.0, 30.0, pause, sampling_frequency
    )

    at_end = np.isclose(position, 60.0)
    starts = np.flatnonzero(at_end & ~np.r_[False, at_end[:-1]])
    stops = np.flatnonzero(at_end & ~np.r_[at_end[1:], False])
    # Arrivals at 2 s (60 / 30), then every 2 * 2 + 2 = 6 s.
    np.testing.assert_allclose(time[starts], 2.0 + 6.0 * np.arange(starts.size))
    assert starts.size == 17  # arrivals at 2, 8, ..., 98 s
    run_lengths = (stops - starts + 1)[:-1] / sampling_frequency
    np.testing.assert_allclose(run_lengths, pause, atol=0.05)
    assert position.min() >= 0.0
    assert position.max() <= 60.0
