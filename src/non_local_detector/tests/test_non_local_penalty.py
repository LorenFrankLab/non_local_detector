"""Parameter validation for the non-local position penalty.

Covers ``_validate_penalty_params`` directly and through the sorted-spikes and
clusterless non-local detector constructors (negative penalty, non-positive
sigma, valid values). The penalty's effect on the posterior is tested in
test_non_local_penalty_effect.py.
"""

import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models.non_local_model import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    _validate_penalty_params,
)

DETECTOR_CLASSES = [NonLocalSortedSpikesDetector, NonLocalClusterlessDetector]
DETECTOR_IDS = ["sorted", "clusterless"]


@pytest.mark.unit
class TestValidatePenaltyParams:
    """Test _validate_penalty_params directly."""

    @pytest.mark.parametrize(
        "penalty, sigma",
        [(0.0, 1.0), (1.5, 2.0), (1.0, 1e-15)],
        ids=["defaults", "positive-penalty", "very-small-sigma"],
    )
    def test_valid_params_pass(self, penalty, sigma):
        """Non-negative penalty with positive (even tiny) sigma should pass."""
        _validate_penalty_params(penalty, sigma)

    @pytest.mark.parametrize(
        "penalty, sigma, match",
        [
            (-1.0, 1.0, "non_local_position_penalty must be >= 0"),
            # Zero sigma would divide by zero
            (1.0, 0.0, "non_local_penalty_std must be > 0"),
            (1.0, -1.0, "non_local_penalty_std must be > 0"),
        ],
        ids=["negative-penalty", "zero-sigma", "negative-sigma"],
    )
    def test_invalid_params_raise(self, penalty, sigma, match):
        """Negative penalty or non-positive sigma should raise ValidationError."""
        with pytest.raises(ValidationError, match=match):
            _validate_penalty_params(penalty, sigma)


@pytest.mark.unit
class TestDetectorPenaltyValidation:
    """Test that detector constructors validate penalty params."""

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"non_local_position_penalty": -0.5}, "non_local_position_penalty"),
            ({"non_local_penalty_std": 0.0}, "non_local_penalty_std"),
        ],
        ids=["negative-penalty", "zero-sigma"],
    )
    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
    def test_invalid_params_raise(self, detector_cls, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            detector_cls(**kwargs)

    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
    def test_valid_params_no_raise(self, detector_cls):
        """Constructor with valid penalty params should not raise."""
        detector = detector_cls(
            non_local_position_penalty=0.0, non_local_penalty_std=1.0
        )
        assert detector.non_local_position_penalty == 0.0
        assert detector.non_local_penalty_std == 1.0

    def test_sorted_spikes_positive_penalty_accepted(self):
        """Constructor with positive penalty should succeed."""
        detector = NonLocalSortedSpikesDetector(
            non_local_position_penalty=5.0, non_local_penalty_std=2.0
        )
        assert detector.non_local_position_penalty == 5.0
        assert detector.non_local_penalty_std == 2.0
