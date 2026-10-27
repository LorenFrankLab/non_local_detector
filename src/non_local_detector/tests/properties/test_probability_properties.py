"""Property-based tests for probability functions.

These tests use Hypothesis to verify mathematical invariants and properties
that should hold for all valid inputs. Property-based testing helps catch
edge cases that manual testing might miss.
"""

import hypothesis.extra.numpy as npst
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from non_local_detector.core import (
    _condition_on,
    _divide_safe,
    _normalize,
    _safe_log,
)
from non_local_detector.tests.properties.strategies import probability_distribution


@st.composite
def positive_array(draw, min_size=2, max_size=20):
    """Generate an UN-normalized array of strictly positive values.

    Unlike ``probability_distribution``, this does not pre-normalize, so a
    test consuming it actually exercises the normalization under test rather
    than the strategy's own division. ``min_value`` keeps the sum well above
    the ``_normalize`` eps floor, so the normalized sum is always ~1.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(
        npst.arrays(
            dtype=np.float64,
            shape=(size,),
            elements=st.floats(
                min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )


@pytest.mark.property
class TestProbabilityProperties:
    """Property-based tests for probability functions."""

    @given(positive_array())
    def test_normalize_yields_unit_sum(self, values):
        """Property: _normalize of any positive array sums to 1.

        Feeds UN-normalized positive input (so this exercises _normalize
        itself, not the strategy's pre-normalization). The eps floor in
        _normalize is negligible here because the strategy's min element is
        0.01, so the sum is always >> eps. Tolerance is generous for the
        float32 accumulation across up to 20 wide-dynamic-range elements.
        """
        normalized, _ = _normalize(jnp.asarray(values))
        assert abs(float(normalized.sum()) - 1.0) < 1e-4

    @given(probability_distribution())
    def test_normalize_preserves_proportions(self, dist):
        """Property: normalization preserves relative proportions."""
        # Skip if values are too similar (would cause numerical issues in ratio)
        if np.std(dist) < 1e-10:
            return

        normalized, _ = _normalize(jnp.asarray(dist))

        # Check that ratios are preserved for distinct elements
        # Find two distinct elements
        sorted_indices = np.argsort(dist)
        if len(sorted_indices) >= 2:
            i, j = sorted_indices[0], sorted_indices[-1]
            if dist[j] > dist[i] * 1.01:  # At least 1% different
                original_ratio = dist[j] / (dist[i] + 1e-10)
                normalized_ratio = normalized[j] / (normalized[i] + 1e-10)
                assert jnp.allclose(original_ratio, normalized_ratio, rtol=1e-4)

    @settings(deadline=None)  # JAX JIT compilation can exceed any fixed deadline on CI
    @given(probability_distribution(), st.integers(min_value=0, max_value=1000000))
    def test_condition_on_preserves_probability_property(self, prior, seed):
        """Property: conditioning always produces valid distribution."""
        # Generate log likelihoods of same size as prior with deterministic seed
        n = len(prior)
        rng = np.random.RandomState(seed)
        log_likes = rng.randn(n) * 2.0  # Random log likelihoods

        posterior, _ = _condition_on(jnp.asarray(prior), jnp.asarray(log_likes))

        assert jnp.allclose(posterior.sum(), 1.0, rtol=1e-5)
        assert jnp.all(posterior >= 0)
        assert jnp.all(posterior <= 1)

    def test_safe_log_zeros_produce_neg_inf(self):
        """_safe_log(0) must return -inf, not NaN."""
        x = jnp.array([0.0, 0.0, 1.0, 0.0])
        result = _safe_log(x)
        assert jnp.all(result[jnp.array([0, 1, 3])] == -jnp.inf)
        assert jnp.isfinite(result[2])

    def test_safe_log_positive_values_match_log(self):
        """_safe_log on positive values must equal jnp.log."""
        x = jnp.array([1e-30, 1e-10, 1.0, 1e10, 1e30])
        result = _safe_log(x)
        expected = jnp.log(x)
        assert jnp.allclose(result, expected)

    def test_safe_log_mixed_zeros_and_positives(self):
        """_safe_log handles a mix of zeros and positives without NaN."""
        x = jnp.array([0.0, 0.5, 0.0, 1.0])
        result = _safe_log(x)
        assert not jnp.any(jnp.isnan(result))
        assert result[0] == -jnp.inf
        assert result[2] == -jnp.inf
        assert jnp.allclose(result[1], jnp.log(jnp.array(0.5)))
        assert result[3] == 0.0

    @given(
        st.lists(
            st.floats(
                min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=50,
        )
    )
    def test_safe_log_monotonicity_property(self, values):
        """Property: safe_log should preserve order (monotonicity)."""
        # Need sorted values for monotonicity test
        values = sorted(values)
        x = jnp.array(values)

        result = _safe_log(x)

        # Check monotonically non-decreasing
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] or jnp.allclose(result[i], result[i + 1])

    def test_divide_safe_zero_denominator_returns_zero(self):
        """_divide_safe must return 0.0 where denominator is 0."""
        a = jnp.array([1.0, 2.0, 3.0, 0.0])
        b = jnp.array([0.0, 1.0, 0.0, 0.0])
        result = _divide_safe(a, b)
        expected = jnp.array([0.0, 2.0, 0.0, 0.0])
        assert jnp.allclose(result, expected)
        assert jnp.all(jnp.isfinite(result))

    @given(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_divide_safe_equals_standard_division_for_nonzero(self, a, b):
        """Property: divide_safe should equal standard division when divisor is nonzero."""
        a_arr = jnp.array([a])
        b_arr = jnp.array([b])

        result = _divide_safe(a_arr, b_arr)
        expected = a_arr / b_arr

        # JAX defaults to float32; rtol must accommodate single-op rounding.
        assert jnp.allclose(result, expected, rtol=1e-5)

    @given(probability_distribution())
    def test_normalize_is_idempotent(self, dist):
        """Property: normalizing twice should give same result as normalizing once."""
        first_normalized, _ = _normalize(jnp.asarray(dist))
        second_normalized, _ = _normalize(first_normalized)

        # float32 (JAX default): loosen rtol so wide-dynamic-range inputs
        # don't trip on benign last-bit rounding.
        assert jnp.allclose(first_normalized, second_normalized, rtol=1e-4)

    @given(
        probability_distribution(),
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_normalize_scales_correctly(self, dist, scale_factor):
        """Property: scaling then normalizing should give same result as normalizing."""
        scaled = dist * scale_factor

        normalized_original, _ = _normalize(jnp.asarray(dist))
        normalized_scaled, _ = _normalize(jnp.asarray(scaled))

        # float32 (JAX default): scale-then-normalize vs normalize differ by
        # benign last-bit rounding for wide-dynamic-range inputs; rtol=1e-6
        # was below float32 precision and flaked under Hypothesis exploration.
        assert jnp.allclose(
            normalized_original, normalized_scaled, rtol=1e-4, atol=1e-7
        )
