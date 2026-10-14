"""Multiunit rate rows are centered on the timestamps they are plotted at."""

import numpy as np
import pytest

from non_local_detector.visualization.static import get_multiunit_firing_rate


@pytest.mark.unit
def test_each_row_counts_the_spikes_nearest_its_timestamp():
    """Row ``i`` covers the half intervals on either side of ``time[i]``, so a
    spike just before a timestamp and one just after it both count there, and
    the first and last rows can own spikes."""
    time = np.arange(5.0)
    spikes = [np.array([-0.4, 0.9, 1.1, 3.6, 4.4])]
    rate = np.asarray(get_multiunit_firing_rate(spikes, time, smoothing_sigma=1e-6))
    np.testing.assert_allclose(rate.ravel(), [1.0, 2.0, 0.0, 0.0, 2.0])


@pytest.mark.unit
def test_single_timestamp_is_rejected():
    """One timestamp does not determine a row duration, so the rate is
    undefined; the caller gets a ``ValidationError`` rather than an index or
    smoothing error."""
    from non_local_detector.exceptions import ValidationError

    with pytest.raises(ValidationError, match="at least two"):
        get_multiunit_firing_rate([np.array([0.5])], np.array([0.5]))


@pytest.mark.unit
def test_irregular_rows_use_their_own_exposure():
    """The rate times each bin's duration recovers its spike count."""
    time = np.array([0.0, 0.1, 1.1])
    spikes = [time.copy(), np.array([0.1, 1.1])]

    rate = np.asarray(
        get_multiunit_firing_rate(spikes, time, smoothing_sigma=1e-6)
    ).ravel()

    durations = np.array([0.1, 0.55, 1.0])
    np.testing.assert_allclose(rate, [10.0, 2.0 / 0.55, 2.0])
    np.testing.assert_allclose(rate * durations, [1.0, 2.0, 2.0])
