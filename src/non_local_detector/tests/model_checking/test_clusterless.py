"""Unit tests for ``model_checking/clusterless.py``.

Covers the three implemented public functions (``interval_rescaling_transform``,
``empirical_cdf``, ``rosenblatt_transform``) and captures the status of the
seven goodness-of-fit functions that currently raise ``NotImplementedError``.

``interval_rescaling_transform`` is tested with realistic shapes
(``n_time`` much larger than ``n_features``): the mark-rescaling step
evaluates the ``(n_time,)`` ground intensity at the spike times before
dividing the ``(n_spikes, n_features)`` joint mark intensity by it.
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
    def test_realistic_shapes_run(self):
        """Runs with realistic shapes (n_time much larger than n_features).

        Exercises the full code path (ISI rescaling + Rosenblatt transform)
        and checks the outputs are valid uniform-transformed quantities. This
        is the shape regime the docstring describes and that previously
        raised a broadcasting ValueError.
        """
        rng = np.random.default_rng(3)
        n_time = 100
        n_features = 3
        time = np.linspace(0.0, 5.0, n_time)
        ground_process_intensity = np.full(n_time, 2.0)
        electrode_spike_times = time[[10, 50, 90]]
        features = rng.standard_normal((3, n_features))
        joint_mark_intensity = rng.random((3, n_features)) + 1.0

        u_isi, u_mark = interval_rescaling_transform(
            time,
            electrode_spike_times,
            features,
            ground_process_intensity,
            joint_mark_intensity,
        )

        assert u_isi.shape == (3,)
        assert u_mark.shape == (3, n_features)
        assert np.all((u_isi >= 0.0) & (u_isi <= 1.0))
        assert np.all((u_mark >= 0.0) & (u_mark <= 1.0))

    def test_conditional_mark_uses_ground_intensity_at_spike_times(self):
        """The mark intensity is divided by the ground intensity at the spikes.

        With a time-varying ground intensity, the conditional mark intensity
        fed to the Rosenblatt transform must use the ground intensity
        interpolated at the spike times (shape ``(n_spikes,)``), not the raw
        ``(n_time,)`` array. We verify by reproducing the pre-Rosenblatt
        quotient: since ``rosenblatt_transform`` applies a per-column
        empirical CDF (rank transform), feeding ``joint / ground_at_spikes``
        and the hand-computed equivalent must yield identical uniform output.
        """
        n_time = 50
        n_features = 2
        time = np.linspace(0.0, 1.0, n_time)
        # Time-varying ground intensity so spike-time evaluation matters.
        ground_process_intensity = np.linspace(1.0, 3.0, n_time)
        electrode_spike_times = time[[5, 20, 40]]
        features = np.zeros((3, n_features))
        joint_mark_intensity = np.array(
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype=float
        )

        _, u_mark = interval_rescaling_transform(
            time,
            electrode_spike_times,
            features,
            ground_process_intensity,
            joint_mark_intensity,
        )

        # Reproduce the expected pre-Rosenblatt quotient and rank-transform it.
        ground_at_spikes = np.interp(
            electrode_spike_times, time, ground_process_intensity
        )
        expected_quotient = joint_mark_intensity / ground_at_spikes[:, None]
        expected_u_mark = rosenblatt_transform(expected_quotient)

        np.testing.assert_allclose(u_mark, expected_u_mark)

    def test_permute_waveform_features_runs(self):
        """The optional feature permutation path executes without error."""
        rng = np.random.default_rng(4)
        n_time = 60
        n_features = 4
        time = np.linspace(0.0, 1.0, n_time)
        ground_process_intensity = np.full(n_time, 1.5)
        electrode_spike_times = time[[0, 25, 55]]
        joint_mark_intensity = rng.random((3, n_features)) + 1.0
        features = rng.standard_normal((3, n_features))

        u_isi, u_mark = interval_rescaling_transform(
            time,
            electrode_spike_times,
            features,
            ground_process_intensity,
            joint_mark_intensity,
            permute_waveform_features=True,
        )
        assert u_isi.shape == (3,)
        assert u_mark.shape == (3, n_features)

    def test_n_features_equals_n_time_does_not_silently_broadcast(self):
        """``n_features == n_time`` is the case the old broadcast bug hid.

        The previous implementation divided ``(n_spikes, n_features)`` by the
        raw ``(n_time,)`` ground intensity. That raised for the common
        ``n_features != n_time`` shapes, but silently *succeeded* with a wrong
        per-feature divisor when ``n_features == n_time``. With the spike-time
        interpolation the divisor is per-spike, so this shape produces the same
        correct quotient as the hand-computed reference.
        """
        n_time = 3
        n_features = 3  # deliberately equal to n_time
        time = np.linspace(0.0, 1.0, n_time)
        ground_process_intensity = np.array([1.0, 2.0, 4.0])
        electrode_spike_times = time[[0, 1, 2]]
        features = np.zeros((3, n_features))
        joint_mark_intensity = np.array(
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0], [14.0, 16.0, 18.0]], dtype=float
        )

        _, u_mark = interval_rescaling_transform(
            time,
            electrode_spike_times,
            features,
            ground_process_intensity,
            joint_mark_intensity,
        )

        ground_at_spikes = np.interp(
            electrode_spike_times, time, ground_process_intensity
        )
        expected = rosenblatt_transform(
            joint_mark_intensity / ground_at_spikes[:, None]
        )
        np.testing.assert_allclose(u_mark, expected)

    @pytest.mark.parametrize(
        "bad_value",
        [0.0, -1.0, np.nan, np.inf],
        ids=["zero", "negative", "nan", "+inf"],
    )
    def test_invalid_ground_intensity_at_spike_raises(self, bad_value):
        """A non-finite or <= 0 ground intensity at a spike raises, not silent inf.

        Dividing the joint mark intensity by such a value yields ``inf``/``nan``;
        the Rosenblatt rank transform would launder that into a plausible value
        in ``[0, 1]``, corrupting the goodness-of-fit result. A ``<= 0`` check
        alone misses ``nan`` and ``+inf`` (both compare ``False`` against 0), so
        all four cases must raise.
        """
        time = np.array([0.0, 1.0, 2.0])
        ground_process_intensity = np.array([1.0, bad_value, 1.0])  # bad at t=1
        electrode_spike_times = np.array([1.0])  # spike sits on the bad value
        features = np.zeros((1, 2))
        joint_mark_intensity = np.ones((1, 2))

        with pytest.raises(ValueError, match="non-finite or <= 0"):
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
