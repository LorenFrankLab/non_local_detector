"""Parameter validation for the non-local position penalty.

Covers ``_validate_penalty_params`` directly and through the sorted-spikes and
clusterless non-local detector constructors (negative penalty, non-positive
sigma, valid values), and again when decoding, so values set later with
``set_params`` are checked too. The penalty's effect on the posterior is tested in
test_non_local_penalty_effect.py.
"""

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models.non_local_model import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    _validate_penalty_params,
)
from non_local_detector.time_edges import time_edges_from_centers

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


def _fitted_detector(detector_cls):
    """A non-local detector fit to two seconds of a simulated linear track.

    Returns
    -------
    detector, predict_kwargs
    """
    time = np.arange(0.0, 2.0, 0.002)
    position = 50.0 + 40.0 * np.sin(2 * np.pi * time / 2.0)
    rng = np.random.default_rng(0)
    spike_times = [np.sort(rng.uniform(0.0, 2.0, 100)) for _ in range(3)]
    data = {"spike_times": spike_times}
    if detector_cls is NonLocalClusterlessDetector:
        data["spike_waveform_features"] = [
            rng.normal(size=(len(times), 2)) for times in spike_times
        ]
    detector = detector_cls(non_local_position_penalty=1.0)
    detector.fit(time, position, **data)
    predict_kwargs = {
        **data,
        "time_edges": time_edges_from_centers(time),
        "position": position,
        "position_time": time,
    }
    return detector, predict_kwargs


@pytest.mark.integration
class TestPenaltyValidatedWhenDecoding:
    """Values set after construction (``set_params``) are validated at use."""

    @pytest.mark.parametrize(
        "params, match",
        [
            ({"non_local_position_penalty": -1.0}, "non_local_position_penalty"),
            ({"non_local_penalty_std": 0.0}, "non_local_penalty_std"),
        ],
        ids=["negative-penalty", "zero-sigma"],
    )
    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
    def test_set_params_invalid_raises_at_predict(self, detector_cls, params, match):
        detector, predict_kwargs = _fitted_detector(detector_cls)
        detector.set_params(**params)
        with pytest.raises(ValidationError, match=match):
            detector.predict(**predict_kwargs)
