"""Unit tests for ``interval_rescaling_transform`` in ``model_checking/clusterless.py``.

The transform rescales interspike intervals by a sampled ground intensity and
evaluates a supplied conditional mark CDF at the observed marks. Tests use
realistic shapes (``n_time`` much larger than ``n_features``).
"""

import numpy as np
import pytest
from scipy.stats import norm

from non_local_detector.model_checking.clusterless import (
    interval_rescaling_transform,
)


def _standard_normal_cdf(times, observed, order):
    order = np.arange(observed.shape[1]) if order is None else np.asarray(order)
    return norm.cdf(observed[:, order])


@pytest.mark.unit
class TestIntervalRescalingTransform:
    def test_realistic_shapes_run(self):
        """Outputs have one value per spike and lie in the unit interval."""
        rng = np.random.default_rng(3)
        time = np.linspace(0.0, 5.0, 100)
        spike_times = time[[10, 50, 90]]
        features = rng.standard_normal((3, 3))

        u, v = interval_rescaling_transform(
            time, spike_times, features, np.full(100, 2.0), _standard_normal_cdf
        )

        assert u.shape == (3,) and v.shape == (3, 3)
        assert np.all((u >= 0.0) & (u <= 1.0))
        assert np.all((v >= 0.0) & (v <= 1.0))

    def test_rescaled_intervals_integrate_the_ground_intensity(self):
        """u = 1 - exp(-integral of the rate since the previous spike)."""
        time = np.linspace(0.0, 2.0, 201)
        spike_times = np.array([0.5, 1.5])
        u, _ = interval_rescaling_transform(
            time,
            spike_times,
            np.zeros((2, 1)),
            np.full(201, 3.0),
            _standard_normal_cdf,
        )
        np.testing.assert_allclose(u, 1.0 - np.exp(-3.0 * np.array([0.5, 1.0])))

    def test_fitted_cdf_receives_observed_marks_at_spike_times(self):
        """The CDF is evaluated at the observed marks and spike times."""
        time = np.linspace(0.0, 1.0, 50)
        spike_times = time[[5, 20, 40]]
        features = np.zeros((3, 2))
        calls = []

        def conditional(times, observed, order):
            calls.append((times, observed, order))
            return norm.cdf(observed - times[:, None])

        _, v = interval_rescaling_transform(
            time, spike_times, features, np.linspace(1.0, 3.0, 50), conditional
        )

        np.testing.assert_allclose(v, norm.cdf(features - spike_times[:, None]))
        np.testing.assert_array_equal(calls[0][0], spike_times)
        np.testing.assert_array_equal(calls[0][1], features)
        assert calls[0][2] is None

    def test_feature_order_is_passed_to_the_cdf(self):
        rng = np.random.default_rng(4)
        time = np.linspace(0.0, 1.0, 60)
        features = rng.standard_normal((3, 4))
        _, v = interval_rescaling_transform(
            time,
            time[[0, 25, 55]],
            features,
            np.full(60, 1.5),
            _standard_normal_cdf,
            feature_order=[3, 1, 0, 2],
        )
        np.testing.assert_allclose(v, norm.cdf(features[:, [3, 1, 0, 2]]))

    def test_n_features_equals_n_time_does_not_silently_broadcast(self):
        """Equal time and feature counts still evaluate CDFs at observed marks."""
        time = np.linspace(0.0, 1.0, 3)
        features = np.zeros((3, 3))
        _, v = interval_rescaling_transform(
            time, time, features, np.array([1.0, 2.0, 4.0]), _standard_normal_cdf
        )
        np.testing.assert_allclose(v, norm.cdf(features))

    @pytest.mark.parametrize(
        "bad_value",
        [0.0, -1.0, np.nan, np.inf],
        ids=["zero", "negative", "nan", "+inf"],
    )
    def test_invalid_ground_intensity_at_spike_raises(self, bad_value):
        """A spike where the ground intensity is not finite and positive raises."""
        with pytest.raises(ValueError, match="zero or invalid"):
            interval_rescaling_transform(
                np.array([0.0, 1.0, 2.0]),
                np.array([1.0]),
                np.zeros((1, 2)),
                np.array([1.0, bad_value, 1.0]),
                _standard_normal_cdf,
            )

    def test_cdf_values_outside_the_unit_interval_raise(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            interval_rescaling_transform(
                np.array([0.0, 1.0]),
                np.array([0.5]),
                np.zeros((1, 1)),
                np.ones(2),
                lambda times, observed, order: observed + 2.0,
            )
