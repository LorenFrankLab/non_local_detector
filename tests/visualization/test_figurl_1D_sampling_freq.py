"""The 1-D figurl reads its sampling rate from the posterior's time stamps."""

import numpy as np
import pytest

from non_local_detector.visualization.figurl_1D import get_sampling_freq

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "sampling_frequency", [250, 300, 400, 500, 600, 1000, 1500, 2000, 3000]
)
@pytest.mark.parametrize("start_time", [0.0, 1_634_567_890.123])
def test_sampling_rate_is_recovered(sampling_frequency, start_time):
    """Rates whose sample interval is not a whole millisecond are exact too."""
    times = start_time + np.arange(20_000) / sampling_frequency

    np.testing.assert_allclose(get_sampling_freq(times), sampling_frequency, rtol=1e-6)


def test_a_gap_in_the_times_does_not_change_the_rate():
    times = 1_634_567_890.123 + np.arange(20_000) / 600.0
    times = np.delete(times, np.arange(5_000, 6_000))

    np.testing.assert_allclose(get_sampling_freq(times), 600.0, rtol=1e-6)
