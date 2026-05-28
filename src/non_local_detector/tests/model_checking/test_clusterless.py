"""Unit tests for ``model_checking/clusterless.py``.

Covers the three implemented public functions (``interval_rescaling_transform``,
``empirical_cdf``, ``rosenblatt_transform``) and captures the status of the
seven goodness-of-fit functions that currently raise ``NotImplementedError``.

A genuine bug is documented in
``test_interval_rescaling_transform_shape_mismatch_is_a_bug``: the mark-rescaling
step divides ``joint_mark_intensity`` (``n_spikes, n_features``) by
``ground_process_intensity`` (``n_time,``), which only broadcasts in the
degenerate ``n_features == n_time`` case.
"""

import numpy as np
import pytest

from non_local_detector.model_checking.clusterless import (
    discrepancy_test,
    distance_to_boundary_test,
    empirical_cdf,
    interval_rescaling_transform,
    ks_test,
    mark_conditional_intensity_transform,
    minimal_spanning_tree_test,
    pearson_chi_squared_test,
    ripley_stats_test,
    rosenblatt_transform,
)

# ---------------------------------------------------------------------------
# empirical_cdf
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmpiricalCDF:
    def test_known_values(self):
        """Hand-computed ECDF for a small sample with a duplicate."""
        x, cdf = empirical_cdf(np.array([3.0, 1.0, 2.0, 2.0]))
        np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
        # counts 1, 2, 1 over 4 samples -> cumulative 0.25, 0.75, 1.0
        np.testing.assert_allclose(cdf, np.array([0.25, 0.75, 1.0]))

    def test_cdf_ends_at_one(self):
        rng = np.random.default_rng(0)
        _, cdf = empirical_cdf(rng.standard_normal(100))
        assert cdf[-1] == pytest.approx(1.0)
        assert np.all(np.diff(cdf) > 0)  # strictly increasing for unique values

    def test_single_value(self):
        """A single repeated value collapses to one unique point with CDF 1."""
        x, cdf = empirical_cdf(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_array_equal(x, np.array([5.0]))
        np.testing.assert_allclose(cdf, np.array([1.0]))


# ---------------------------------------------------------------------------
# rosenblatt_transform
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRosenblattTransform:
    def test_output_in_unit_range_and_shape(self):
        rng = np.random.default_rng(1)
        samples = rng.standard_normal((200, 3))
        transformed = rosenblatt_transform(samples)
        assert transformed.shape == samples.shape
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0

    def test_marginals_approximately_uniform(self):
        """Each transformed dimension should have mean ~0.5 (Uniform(0, 1))."""
        rng = np.random.default_rng(2)
        samples = rng.standard_normal((5000, 2))
        transformed = rosenblatt_transform(samples)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.5, atol=0.02)

    def test_monotone_within_column(self):
        """The transform is the ECDF, so it preserves the rank order per column."""
        samples = np.array([[3.0], [1.0], [2.0]])
        transformed = rosenblatt_transform(samples)
        # Smallest input maps to smallest output, largest to largest.
        assert np.argmin(transformed[:, 0]) == np.argmin(samples[:, 0])
        assert np.argmax(transformed[:, 0]) == np.argmax(samples[:, 0])

    def test_single_feature_column(self):
        """Edge case: a single feature dimension (n_samples, 1)."""
        samples = np.array([[0.0], [1.0], [2.0], [3.0]])
        transformed = rosenblatt_transform(samples)
        assert transformed.shape == (4, 1)
        # Distinct values -> ECDF values 0.25, 0.5, 0.75, 1.0.
        np.testing.assert_allclose(
            np.sort(transformed[:, 0]), np.array([0.25, 0.5, 0.75, 1.0])
        )


# ---------------------------------------------------------------------------
# interval_rescaling_transform
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntervalRescalingTransform:
    def test_happy_path_degenerate_shapes(self):
        """End-to-end run in the only shape combination that broadcasts.

        Because of the broadcasting bug documented below, the function only
        runs when ``n_features == n_time``. We use that degenerate case to
        exercise the full code path (ISI rescaling + Rosenblatt transform) and
        check the outputs are valid uniform-transformed quantities.
        """
        rng = np.random.default_rng(3)
        n_time = 6  # also serves as n_features (degenerate)
        time = np.linspace(0.0, 1.0, n_time)
        ground_process_intensity = np.full(n_time, 2.0)
        # Two spikes, on the time grid so np.interp is exact.
        electrode_spike_times = time[[1, 4]]
        n_features = n_time
        features = rng.standard_normal((2, n_features))
        joint_mark_intensity = rng.random((2, n_features)) + 1.0

        u_isi, u_mark = interval_rescaling_transform(
            time,
            electrode_spike_times,
            features,
            ground_process_intensity,
            joint_mark_intensity,
        )

        assert u_isi.shape == (2,)
        assert u_mark.shape == (2, n_features)
        assert np.all((u_isi >= 0.0) & (u_isi <= 1.0))
        assert np.all((u_mark >= 0.0) & (u_mark <= 1.0))

    def test_permute_waveform_features_runs(self):
        """The optional feature permutation path executes without error."""
        rng = np.random.default_rng(4)
        n_time = 5
        time = np.linspace(0.0, 1.0, n_time)
        ground_process_intensity = np.full(n_time, 1.5)
        electrode_spike_times = time[[0, 2, 4]]
        joint_mark_intensity = rng.random((3, n_time)) + 1.0
        features = rng.standard_normal((3, n_time))

        u_isi, u_mark = interval_rescaling_transform(
            time,
            electrode_spike_times,
            features,
            ground_process_intensity,
            joint_mark_intensity,
            permute_waveform_features=True,
        )
        assert u_isi.shape == (3,)
        assert u_mark.shape == (3, n_time)

    def test_shape_mismatch_is_a_bug(self):
        """Documented bug: realistic shapes (n_features != n_time) cannot run.

        ``clusterless.py`` line 67 divides ``joint_mark_intensity``
        ``(n_spikes, n_features)`` by ``ground_process_intensity`` ``(n_time,)``.
        These broadcast only when ``n_features == n_time``. With the docstring's
        intended shapes (n_time much larger than n_features) the division
        raises ``ValueError``. This test pins that broken behavior so a future
        source fix is forced to update the test.
        """
        rng = np.random.default_rng(5)
        n_time = 100
        n_features = 3
        time = np.linspace(0.0, 5.0, n_time)
        ground_process_intensity = np.full(n_time, 2.0)
        electrode_spike_times = time[[10, 50, 90]]
        features = rng.standard_normal((3, n_features))
        joint_mark_intensity = rng.random((3, n_features)) + 1.0

        with pytest.raises(ValueError, match="broadcast"):
            interval_rescaling_transform(
                time,
                electrode_spike_times,
                features,
                ground_process_intensity,
                joint_mark_intensity,
            )


# ---------------------------------------------------------------------------
# Unimplemented goodness-of-fit functions (status capture)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnimplementedStubs:
    """Pin the ``NotImplementedError`` status of the unfinished stubs.

    These functions are documented but not yet implemented. The tests assert
    they raise ``NotImplementedError`` so the stub status is captured and a
    future implementation is forced to update the test (rather than silently
    returning wrong results).
    """

    def test_mark_conditional_intensity_transform_not_implemented(self):
        with pytest.raises(NotImplementedError):
            mark_conditional_intensity_transform(np.zeros((3, 2)))

    def test_pearson_chi_squared_test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            pearson_chi_squared_test(np.array([1, 2]), np.array([1, 2]))

    def test_ks_test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ks_test(np.linspace(0, 1, 10))

    def test_distance_to_boundary_test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            distance_to_boundary_test(np.zeros((3, 2)), np.zeros((4, 2)))

    def test_discrepancy_test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            discrepancy_test(np.zeros(3), np.zeros((5, 3)))

    def test_ripley_stats_test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ripley_stats_test(np.zeros((3, 2)), np.linspace(0, 1, 5), 1.0)

    def test_minimal_spanning_tree_test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            minimal_spanning_tree_test(np.zeros((3, 2)))
