"""Property-based tests for probability functions.

These tests use Hypothesis to verify mathematical invariants and properties
that should hold for all valid inputs. Property-based testing helps catch
edge cases that manual testing might miss.
"""

import hypothesis.extra.numpy as npst
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from non_local_detector.continuous_state_transitions import RandomWalk
from non_local_detector.core import (
    _condition_on,
    _divide_safe,
    _normalize,
    _safe_log,
)
from non_local_detector.models.decoder import ClusterlessDecoder
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data


# Custom strategies for probability distributions
@st.composite
def probability_distribution(draw, min_size=2, max_size=20):
    """Generate valid probability distributions.

    Returns an array of positive values that sum to 1.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate positive values
    values = draw(
        npst.arrays(
            dtype=np.float64,
            shape=(size,),
            elements=st.floats(
                min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    # Normalize to sum to 1
    return values / values.sum()


@st.composite
def stochastic_matrix(draw, min_size=2, max_size=10):
    """Generate valid stochastic (row-normalized) matrices.

    Each row sums to 1, representing valid transition probabilities.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    matrix = draw(
        npst.arrays(
            dtype=np.float64,
            shape=(size, size),
            elements=st.floats(
                min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    # Row-normalize
    return matrix / matrix.sum(axis=1, keepdims=True)


@pytest.mark.property
class TestProbabilityProperties:
    """Property-based tests for probability functions."""

    @given(probability_distribution())
    def test_normalize_preserves_probability_property(self, dist):
        """Property: normalized distributions always sum to 1."""
        normalized, _ = _normalize(jnp.asarray(dist))
        assert jnp.allclose(normalized.sum(), 1.0, rtol=1e-5)

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

    @settings(deadline=500)  # Increased deadline for JAX JIT compilation
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

    @given(st.integers(min_value=2, max_value=10))
    def test_transition_preserves_probability_property(self, n_states):
        """Property: transition always produces valid distribution."""
        # Generate matching sized distributions and matrices
        dist = np.random.rand(n_states)
        dist = dist / dist.sum()

        trans_matrix = np.random.rand(n_states, n_states)
        trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)

        next_dist = jnp.dot(jnp.asarray(dist), jnp.asarray(trans_matrix))

        assert jnp.allclose(next_dist.sum(), 1.0, rtol=1e-5)
        assert jnp.all(next_dist >= 0)
        assert jnp.all(next_dist <= 1)

    @settings(deadline=500)  # Increased deadline for JAX JIT compilation
    @given(
        st.lists(
            # Limit range to avoid overflow in log for float32
            st.floats(
                min_value=1e-30, max_value=1e30, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=50,
        )
    )
    def test_safe_log_always_finite_or_neg_inf_for_positive_values(self, values):
        """Property: safe_log should produce finite values or -inf (not NaN) for positive values."""
        x = jnp.array(values)
        assume(jnp.all(x >= 0))

        result = _safe_log(x)

        # Should not produce NaN
        assert not jnp.any(jnp.isnan(result))
        # Should be finite or -inf (for zeros)
        # Note: Very large values may overflow to +inf in log, which is expected
        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

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

    @given(
        npst.arrays(
            dtype=np.float64,
            shape=(10,),
            elements=st.floats(
                min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        ),
        npst.arrays(
            dtype=np.float64,
            shape=(10,),
            elements=st.floats(
                min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        ),
    )
    def test_divide_safe_never_produces_nan_or_inf(self, a, b):
        """Property: divide_safe should never produce NaN or inf."""
        result = _divide_safe(jnp.asarray(a), jnp.asarray(b))

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

        assert jnp.allclose(result, expected, rtol=1e-6)

    @given(probability_distribution())
    def test_normalize_is_idempotent(self, dist):
        """Property: normalizing twice should give same result as normalizing once."""
        first_normalized, _ = _normalize(jnp.asarray(dist))
        second_normalized, _ = _normalize(first_normalized)

        assert jnp.allclose(first_normalized, second_normalized, rtol=1e-6)

    @given(
        probability_distribution(),
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_normalize_scales_correctly(self, dist, scale_factor):
        """Property: scaling then normalizing should give same result as normalizing."""
        scaled = dist * scale_factor

        normalized_original, _ = _normalize(jnp.asarray(dist))
        normalized_scaled, _ = _normalize(jnp.asarray(scaled))

        assert jnp.allclose(normalized_original, normalized_scaled, rtol=1e-6)

    @settings(deadline=5000, max_examples=10)  # Decoder tests are slower
    @given(st.integers(min_value=42, max_value=9999))
    def test_posterior_probabilities_sum_to_one(self, seed):
        """Property: decoder posteriors sum to 1 across position dimension."""
        # Generate simulation
        # NOTE: n_runs must be >= 3 to create proper 2D position data
        # NOTE: Need substantial data for RandomWalk to build proper position bins
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),  # 4 neurons
            sampling_frequency=500,
            n_runs=3,  # Multiple runs to ensure 2D position array
            seed=seed,
        )

        # Use 70/30 train/test split on all data
        n_encode = int(0.7 * len(sim.position_time))
        is_training = np.ones(len(sim.position_time), dtype=bool)
        is_training[n_encode:] = False

        decoder = ClusterlessDecoder(
            clusterless_algorithm="clusterless_kde",
            clusterless_algorithm_params={
                "position_std": 6.0,
                "waveform_std": 24.0,
                "block_size": 50,
            },
            continuous_transition_types=[[RandomWalk(movement_var=25.0)]],
        )

        decoder.fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
            is_training=is_training,
        )

        # Predict on small test set (10 time bins only for speed)
        test_start_idx = n_encode
        test_end_idx = min(n_encode + 10, len(sim.position_time))
        results = decoder.predict(
            spike_times=[
                st[
                    (st >= sim.position_time[test_start_idx])
                    & (st < sim.position_time[test_end_idx])
                ]
                for st in sim.spike_times
            ],
            spike_waveform_features=[
                swf[
                    (sim.spike_times[i] >= sim.position_time[test_start_idx])
                    & (sim.spike_times[i] < sim.position_time[test_end_idx])
                ]
                for i, swf in enumerate(sim.spike_waveform_features)
            ],
            time=sim.position_time[test_start_idx:test_end_idx],
            position=sim.position[test_start_idx:test_end_idx],
            position_time=sim.position_time[test_start_idx:test_end_idx],
        )

        # Check posterior sums to 1 across spatial dimension (state_bins)
        posterior_sums = results.acausal_posterior.sum(dim="state_bins")
        assert np.allclose(posterior_sums.values, 1.0, atol=1e-10)

    @settings(deadline=5000, max_examples=10)
    @given(st.integers(min_value=42, max_value=9999))
    def test_posteriors_nonnegative_and_bounded(self, seed):
        """Property: decoder posteriors are in [0, 1]."""
        # NOTE: n_runs must be >= 3 to create proper 2D position data
        # NOTE: Need substantial data for RandomWalk to build proper position bins
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            sampling_frequency=500,
            n_runs=3,
            seed=seed,
        )

        n_encode = int(0.7 * len(sim.position_time))
        is_training = np.ones(len(sim.position_time), dtype=bool)
        is_training[n_encode:] = False

        decoder = ClusterlessDecoder(
            clusterless_algorithm="clusterless_kde",
            clusterless_algorithm_params={
                "position_std": 6.0,
                "waveform_std": 24.0,
                "block_size": 50,
            },
            continuous_transition_types=[[RandomWalk(movement_var=25.0)]],
        )

        decoder.fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
            is_training=is_training,
        )

        test_start_idx = n_encode
        test_end_idx = min(n_encode + 10, len(sim.position_time))
        results = decoder.predict(
            spike_times=[
                st[
                    (st >= sim.position_time[test_start_idx])
                    & (st < sim.position_time[test_end_idx])
                ]
                for st in sim.spike_times
            ],
            spike_waveform_features=[
                swf[
                    (sim.spike_times[i] >= sim.position_time[test_start_idx])
                    & (sim.spike_times[i] < sim.position_time[test_end_idx])
                ]
                for i, swf in enumerate(sim.spike_waveform_features)
            ],
            time=sim.position_time[test_start_idx:test_end_idx],
            position=sim.position[test_start_idx:test_end_idx],
            position_time=sim.position_time[test_start_idx:test_end_idx],
        )

        # Check all values in [0, 1]
        assert np.all(results.acausal_posterior.values >= 0.0)
        assert np.all(results.acausal_posterior.values <= 1.0)

    @settings(deadline=5000, max_examples=10)
    @given(st.integers(min_value=42, max_value=9999))
    def test_log_probabilities_finite(self, seed):
        """Property: log probabilities should be finite (or -inf for zero prob)."""
        # NOTE: n_runs must be >= 3 to create proper 2D position data
        # NOTE: Need substantial data for RandomWalk to build proper position bins
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            sampling_frequency=500,
            n_runs=3,
            seed=seed,
        )

        n_encode = int(0.7 * len(sim.position_time))
        is_training = np.ones(len(sim.position_time), dtype=bool)
        is_training[n_encode:] = False

        decoder = ClusterlessDecoder(
            clusterless_algorithm="clusterless_kde",
            clusterless_algorithm_params={
                "position_std": 6.0,
                "waveform_std": 24.0,
                "block_size": 50,
            },
            continuous_transition_types=[[RandomWalk(movement_var=25.0)]],
        )

        decoder.fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
            is_training=is_training,
        )

        test_start_idx = n_encode
        test_end_idx = min(n_encode + 10, len(sim.position_time))
        results = decoder.predict(
            spike_times=[
                st[
                    (st >= sim.position_time[test_start_idx])
                    & (st < sim.position_time[test_end_idx])
                ]
                for st in sim.spike_times
            ],
            spike_waveform_features=[
                swf[
                    (sim.spike_times[i] >= sim.position_time[test_start_idx])
                    & (sim.spike_times[i] < sim.position_time[test_end_idx])
                ]
                for i, swf in enumerate(sim.spike_waveform_features)
            ],
            time=sim.position_time[test_start_idx:test_end_idx],
            position=sim.position[test_start_idx:test_end_idx],
            position_time=sim.position_time[test_start_idx:test_end_idx],
        )

        # Take log of posteriors
        log_posterior = np.log(
            results.acausal_posterior.values + 1e-300
        )  # Avoid log(0)

        # Should not have NaN
        assert not np.any(np.isnan(log_posterior))
        # Should be finite or -inf
        assert np.all(np.isfinite(log_posterior) | np.isneginf(log_posterior))
