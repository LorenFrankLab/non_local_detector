"""``EncodingSupport``: the exposure, event support and interpolation of tracking."""

import numpy as np
import pytest

from non_local_detector.encoding_time import EncodingSupport
from non_local_detector.exceptions import DataError, ValidationError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ([-0.5, 1.5], [1.0, 1.0]),
        ([0.25, 0.75], [0.25, 0.25]),
        ([0.0, 0.25], [0.21875, 0.03125]),
    ],
)
def test_exposure_integrates_the_original_interpolation_basis(bounds, expected):
    support = EncodingSupport(
        np.array([0.0, 1.0]), np.zeros((2, 1)), encoding_time_range=bounds
    )
    np.testing.assert_allclose(support.exposure, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "valid_position_intervals", [None, [[-0.5, 4.5]]], ids=["inferred", "declared"]
)
def test_missing_sample_splits_tracking(valid_position_intervals):
    """A NaN sample has no exposure and splits tracking around it, also inside
    a declared interval that spans it."""
    position = np.array([0.0, 1.0, np.nan, 3.0, 4.0])[:, None]
    support = EncodingSupport(
        np.arange(5.0), position, valid_position_intervals=valid_position_intervals
    )
    np.testing.assert_allclose(support.exposure, [1.0, 1.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        support.contains(np.array([1.25, 1.6, 2.0, 2.4, 2.75])),
        [True, False, False, False, True],
    )
    np.testing.assert_allclose(
        support.interpolate(position[:, 0], np.array([1.25, 2.75])), [1.0, 3.0]
    )


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_outside_nan_does_not_shorten_declared_tracking_support(side):
    time = np.arange(-1.0, 4.0)
    position = np.zeros((5, 1))
    if side in ("left", "both"):
        position[0] = np.nan
    if side in ("right", "both"):
        position[-1] = np.nan
    support = EncodingSupport(time, position, valid_position_intervals=[[-0.75, 2.75]])
    np.testing.assert_allclose(support.exposure, [0.0, 1.25, 1.0, 1.25, 0.0])
    events = np.array([-0.6, 0.0, 1.0, 2.0, 2.6])
    np.testing.assert_array_equal(support.contains(events), np.ones(5, dtype=bool))
    assert support.event_counts(events, np.ones(5)).sum() == 5.0


def test_acquisition_clip_keeps_original_basis_inside_declared_interval():
    support = EncodingSupport(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0], [1.0], [np.nan]]),
        valid_position_intervals=[[-0.5, 1.5]],
        encoding_time_range=[0.0, 0.25],
    )
    np.testing.assert_allclose(support.exposure, [0.21875, 0.03125, 0.0])
    np.testing.assert_allclose(
        support.interpolate(np.array([0.0, 1.0, np.nan]), np.array([0.125])),
        [0.125],
    )


def test_explicit_tracking_segments_do_not_bridge_epochs():
    time = np.array([0.0, 1.0, 10.0, 11.0])
    support = EncodingSupport(
        time, time[:, None], valid_position_intervals=[[-0.5, 1.5], [9.5, 11.5]]
    )
    np.testing.assert_allclose(support.exposure, np.ones(4))
    np.testing.assert_allclose(
        support.interpolate(time, np.array([1.25, 9.75])), [1.0, 10.0]
    )
    assert not support.contains(np.array([5.0]))[0]


def test_adjacent_tracking_segments_keep_their_own_endpoint_basis():
    time = np.arange(4.0)
    support = EncodingSupport(
        time, time[:, None], valid_position_intervals=[[-0.5, 1.1], [1.1, 3.5]]
    )
    np.testing.assert_allclose(support.exposure, [1.0, 0.6, 1.4, 1.0])
    np.testing.assert_allclose(
        support.interpolate(time, np.array([1.05, 1.1])), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        support.event_counts(np.array([1.1]), np.ones(4)), [0.0, 0.0, 1.0, 0.0]
    )


def test_single_sample_requires_explicit_recording_duration():
    with pytest.raises(ValidationError, match="encoding_time_range"):
        EncodingSupport(np.array([1.0]), np.array([[3.0]]))
    support = EncodingSupport(
        np.array([1.0]), np.array([[3.0]]), encoding_time_range=[0.5, 1.5]
    )
    np.testing.assert_array_equal(support.exposure, [1.0])


def test_acquisition_bounds_do_not_authorize_interpolation_across_a_gap():
    with pytest.raises(DataError, match="continuous tracking"):
        EncodingSupport(
            np.array([0.0, 1.0, 10.0, 11.0]),
            np.zeros((4, 1)),
            encoding_time_range=[0.0, 12.0],
        )
