"""Tests for get_conditional_non_local_posterior bugs and position_time validation.

Bug 1: Hard-coded state names ("Non-Local Continuous", "Non-Local Fragmented")
       break when callers use custom state_names= override.

Bug 2: .sum("position") fails for 2D environments where the coordinate
       levels are x_position/y_position, not "position".

Bug 3: predict() does not validate position_time when position is provided,
       leading to confusing low-level errors instead of a clear ValidationError.
"""

import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.models.non_local_model import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.tests.conftest import make_state_bins_results


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


# ============================================================
# Bug 1: custom state names
# ============================================================


class TestCustomStateNames:
    """get_conditional_non_local_posterior must work with custom state_names."""

    CUSTOM_NAMES = ["Here", "Quiet", "Replay-Smooth", "Replay-Jump"]

    def test_sorted_spikes_custom_names(self):
        results = _make_results_dataset_1d(self.CUSTOM_NAMES)
        # Should NOT raise KeyError
        posterior = NonLocalSortedSpikesDetector.get_conditional_non_local_posterior(
            results
        )
        assert posterior is not None
        # Should be normalised over position for each time step
        # After sel(state=...) on 1D, the position dim replaces state_bins
        non_time_dims = [d for d in posterior.dims if d != "time"]
        sums = posterior.sum(non_time_dims).values
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_clusterless_custom_names(self):
        results = _make_results_dataset_1d(self.CUSTOM_NAMES)
        posterior = NonLocalClusterlessDetector.get_conditional_non_local_posterior(
            results
        )
        assert posterior is not None
        non_time_dims = [d for d in posterior.dims if d != "time"]
        sums = posterior.sum(non_time_dims).values
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_default_names_still_work(self):
        """Verify the fix doesn't break the default state names."""
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
        assert posterior is not None
        non_time_dims = [d for d in posterior.dims if d != "time"]
        sums = posterior.sum(non_time_dims).values
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)


# ============================================================
# Bug 2: 2D position
# ============================================================


class TestTwoDimensionalPosition:
    """get_conditional_non_local_posterior must work with 2D environments."""

    DEFAULT_NAMES = [
        "Local",
        "No-Spike",
        "Non-Local Continuous",
        "Non-Local Fragmented",
    ]

    def test_sorted_spikes_2d(self):
        results = _make_results_dataset_2d(self.DEFAULT_NAMES)
        # Should NOT raise ValueError about 'position' dimension
        posterior = NonLocalSortedSpikesDetector.get_conditional_non_local_posterior(
            results
        )
        assert posterior is not None
        non_time_dims = [d for d in posterior.dims if d != "time"]
        sums = posterior.sum(non_time_dims).values
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_clusterless_2d(self):
        results = _make_results_dataset_2d(self.DEFAULT_NAMES)
        posterior = NonLocalClusterlessDetector.get_conditional_non_local_posterior(
            results
        )
        assert posterior is not None
        non_time_dims = [d for d in posterior.dims if d != "time"]
        sums = posterior.sum(non_time_dims).values
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_2d_with_custom_names(self):
        """Both bugs combined: custom names + 2D."""
        custom = ["A", "B", "C", "D"]
        results = _make_results_dataset_2d(custom)
        posterior = NonLocalClusterlessDetector.get_conditional_non_local_posterior(
            results
        )
        assert posterior is not None
        non_time_dims = [d for d in posterior.dims if d != "time"]
        sums = posterior.sum(non_time_dims).values
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)


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
# Bug 3: position_time validation
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
                time=time,
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
                time=time,
                position=position,
                position_time=None,
            )
