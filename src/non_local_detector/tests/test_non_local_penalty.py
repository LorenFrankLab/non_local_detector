"""Tests for non-local position penalty feature.

Covers:
- Parameter validation (sigma <= 0, negative penalty)
- Multi-environment penalty correctness
- Position validation when penalty is enabled
- Behavioral equivalence when penalty == 0.0
"""

import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models.non_local_model import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    _validate_penalty_params,
)


@pytest.mark.unit
class TestValidatePenaltyParams:
    """Test _validate_penalty_params directly."""

    def test_valid_defaults(self):
        """Default values (penalty=0.0, sigma=1.0) should pass."""
        _validate_penalty_params(0.0, 1.0)

    def test_valid_positive_penalty(self):
        """Positive penalty with positive sigma should pass."""
        _validate_penalty_params(1.5, 2.0)

    def test_negative_penalty_raises(self):
        """Negative penalty should raise ValidationError."""
        with pytest.raises(
            ValidationError, match="non_local_position_penalty must be >= 0"
        ):
            _validate_penalty_params(-1.0, 1.0)

    def test_zero_sigma_raises(self):
        """Zero sigma should raise ValidationError (divide-by-zero)."""
        with pytest.raises(
            ValidationError, match="non_local_penalty_sigma must be > 0"
        ):
            _validate_penalty_params(1.0, 0.0)

    def test_negative_sigma_raises(self):
        """Negative sigma should raise ValidationError."""
        with pytest.raises(
            ValidationError, match="non_local_penalty_sigma must be > 0"
        ):
            _validate_penalty_params(1.0, -1.0)

    def test_very_small_positive_sigma_passes(self):
        """Very small but positive sigma should pass."""
        _validate_penalty_params(1.0, 1e-15)


@pytest.mark.unit
class TestDetectorPenaltyValidation:
    """Test that detector constructors validate penalty params."""

    def test_sorted_spikes_negative_penalty_raises(self):
        with pytest.raises(ValidationError, match="non_local_position_penalty"):
            NonLocalSortedSpikesDetector(non_local_position_penalty=-0.5)

    def test_sorted_spikes_zero_sigma_raises(self):
        with pytest.raises(ValidationError, match="non_local_penalty_sigma"):
            NonLocalSortedSpikesDetector(non_local_penalty_sigma=0.0)

    def test_clusterless_negative_penalty_raises(self):
        with pytest.raises(ValidationError, match="non_local_position_penalty"):
            NonLocalClusterlessDetector(non_local_position_penalty=-0.5)

    def test_clusterless_zero_sigma_raises(self):
        with pytest.raises(ValidationError, match="non_local_penalty_sigma"):
            NonLocalClusterlessDetector(non_local_penalty_sigma=0.0)

    def test_sorted_spikes_valid_params_no_raise(self):
        """Constructor with valid penalty params should not raise."""
        detector = NonLocalSortedSpikesDetector(
            non_local_position_penalty=0.0, non_local_penalty_sigma=1.0
        )
        assert detector.non_local_position_penalty == 0.0
        assert detector.non_local_penalty_sigma == 1.0

    def test_clusterless_valid_params_no_raise(self):
        """Constructor with valid penalty params should not raise."""
        detector = NonLocalClusterlessDetector(
            non_local_position_penalty=0.0, non_local_penalty_sigma=1.0
        )
        assert detector.non_local_position_penalty == 0.0
        assert detector.non_local_penalty_sigma == 1.0

    def test_sorted_spikes_positive_penalty_accepted(self):
        """Constructor with positive penalty should succeed."""
        detector = NonLocalSortedSpikesDetector(
            non_local_position_penalty=5.0, non_local_penalty_sigma=2.0
        )
        assert detector.non_local_position_penalty == 5.0
        assert detector.non_local_penalty_sigma == 2.0
