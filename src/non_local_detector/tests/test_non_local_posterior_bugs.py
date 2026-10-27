"""Tests for get_conditional_non_local_posterior and position_time validation.

- The non-local states are selected by position (third and fourth state), so
  custom ``state_names`` work.
- Marginalization covers every position level, so 2D environments
  (``x_position``/``y_position``) work.
- ``predict()`` raises a clear ValidationError when ``position`` is given
  without ``position_time``.
"""

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models.non_local_model import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.tests.conftest import make_state_bins_results
from non_local_detector.time_edges import time_edges_from_centers

pytestmark = pytest.mark.unit


def _make_results_dataset_1d(state_names, n_time=10, n_bins_per_state=5):
    """Minimal 1D results xr.Dataset mimicking predict() output."""
    return make_state_bins_results(
        state_names,
        {"position": np.arange(n_bins_per_state, dtype=float)},
        n_time=n_time,
    )


def _make_results_dataset_2d(state_names, n_time=10, n_bins_per_state=4):
    """Minimal 2D results xr.Dataset mimicking predict() output."""
    return make_state_bins_results(
        state_names,
        {
            "x_position": np.arange(n_bins_per_state, dtype=float),
            "y_position": np.arange(n_bins_per_state, dtype=float) * 10,
        },
        n_time=n_time,
    )


def _expected_conditional_non_local(results):
    """Hand-compute (P[state 2] + P[state 3]) / sum over position bins.

    Parameters
    ----------
    results : xr.Dataset
        Output of ``make_state_bins_results`` with four equally sized state
        blocks along ``state_bins``.

    Returns
    -------
    expected : np.ndarray, shape (n_time, n_bins_per_state)
    """
    posterior = results.acausal_posterior.values
    n_time = posterior.shape[0]
    per_state = posterior.reshape(n_time, 4, -1)
    non_local = per_state[:, 2] + per_state[:, 3]
    return non_local / non_local.sum(axis=1, keepdims=True)


def _assert_matches_expected(posterior, results):
    """Assert the conditional posterior equals the hand-computed one."""
    assert posterior.dims[0] == "time"
    values = posterior.values.reshape(posterior.shape[0], -1)
    np.testing.assert_allclose(
        values, _expected_conditional_non_local(results), rtol=1e-12
    )


# ============================================================
# Custom state names
# ============================================================


class TestCustomStateNames:
    """The third and fourth states are the non-local ones, whatever their names."""

    CUSTOM_NAMES = ["Here", "Quiet", "Replay-Smooth", "Replay-Jump"]

    @pytest.mark.parametrize(
        "detector_cls", [NonLocalSortedSpikesDetector, NonLocalClusterlessDetector]
    )
    def test_custom_names(self, detector_cls):
        results = _make_results_dataset_1d(self.CUSTOM_NAMES)
        posterior = detector_cls.get_conditional_non_local_posterior(results)
        _assert_matches_expected(posterior, results)

    def test_default_names(self):
        default_names = [
            "Local",
            "No-Spike",
            "Non-Local Continuous",
            "Non-Local Fragmented",
        ]
        results = _make_results_dataset_1d(default_names)
        posterior = NonLocalSortedSpikesDetector.get_conditional_non_local_posterior(
            results
        )
        _assert_matches_expected(posterior, results)


# ============================================================
# 2D position
# ============================================================


class TestTwoDimensionalPosition:
    """The conditional posterior marginalizes over x/y position levels in 2D."""

    DEFAULT_NAMES = [
        "Local",
        "No-Spike",
        "Non-Local Continuous",
        "Non-Local Fragmented",
    ]

    @pytest.mark.parametrize(
        "detector_cls", [NonLocalSortedSpikesDetector, NonLocalClusterlessDetector]
    )
    def test_2d(self, detector_cls):
        results = _make_results_dataset_2d(self.DEFAULT_NAMES)
        posterior = detector_cls.get_conditional_non_local_posterior(results)
        _assert_matches_expected(posterior, results)

    def test_2d_with_custom_names(self):
        """Custom state names and a 2D environment together."""
        results = _make_results_dataset_2d(["A", "B", "C", "D"])
        posterior = NonLocalClusterlessDetector.get_conditional_non_local_posterior(
            results
        )
        _assert_matches_expected(posterior, results)


# ============================================================
# State count guard
# ============================================================


class TestStateCountGuard:
    """get_conditional_non_local_posterior raises on results with < 4 states."""

    def test_3_state_dataset_raises(self):
        results = _make_results_dataset_1d(["A", "B", "C"])
        with pytest.raises(ValidationError, match="at least 4 unique states"):
            NonLocalSortedSpikesDetector.get_conditional_non_local_posterior(results)

    def test_2_state_dataset_raises(self):
        results = _make_results_dataset_1d(["A", "B"])
        with pytest.raises(ValidationError, match="at least 4 unique states"):
            NonLocalClusterlessDetector.get_conditional_non_local_posterior(results)


# ============================================================
# position_time validation
# ============================================================


class TestPositionTimeValidation:
    """predict() should raise a clear error when position is given without position_time."""

    def test_clusterless_position_without_position_time(self):
        """Providing position but omitting position_time should raise ValidationError.

        Validation fires at the start of predict(), before fit() is needed.
        """
        detector = NonLocalClusterlessDetector()
        time = np.linspace(0, 1, 100)
        position = np.linspace(0, 10, 100)
        spike_times = [np.array([0.1, 0.5])]
        spike_features = [np.array([[1.0], [2.0]])]

        with pytest.raises(ValidationError, match="position_time is required"):
            detector.predict(
                spike_times=spike_times,
                spike_waveform_features=spike_features,
                time_edges=time_edges_from_centers(time),
                position=position,
                position_time=None,
            )

    def test_sorted_spikes_position_without_position_time(self):
        """Providing position but omitting position_time should raise ValidationError.

        Validation fires at the start of predict(), before fit() is needed.
        """
        detector = NonLocalSortedSpikesDetector()
        time = np.linspace(0, 1, 100)
        position = np.linspace(0, 10, 100)
        spike_times = [np.array([0.1, 0.5])]

        with pytest.raises(ValidationError, match="position_time is required"):
            detector.predict(
                spike_times=spike_times,
                time_edges=time_edges_from_centers(time),
                position=position,
                position_time=None,
            )
