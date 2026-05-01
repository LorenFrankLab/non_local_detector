"""High-priority tests for discrete state transition estimation functions.

Tests the core EM algorithm functions that are currently untested:
- estimate_non_stationary_state_transition
- estimate_stationary_state_transition
- _estimate_discrete_transition
"""

from unittest.mock import patch

import numpy as np
import pytest

from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteStationaryCustom,
    _estimate_discrete_transition,
    _state_aggregation_matrix,
    estimate_discrete_transition_counts_from_expanded_posteriors,
    estimate_discrete_transition_counts_from_factorized_posteriors,
    estimate_discrete_transition_responses_from_expanded_posteriors,
    estimate_discrete_transition_responses_from_factorized_posteriors,
    estimate_joint_distribution,
    estimate_non_stationary_state_transition,
    estimate_non_stationary_state_transition_from_responses,
    estimate_stationary_state_transition,
    estimate_stationary_state_transition_from_counts,
)
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.models.base import _DetectorBase
from non_local_detector.observation_models import ObservationModel
from non_local_detector.tests.conftest import assert_stochastic_matrix


class _FixedTransition:
    """Continuous transition object with deterministic test matrix."""

    def __init__(self, transition_matrix: np.ndarray):
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)

    def make_state_transition(self, environments):
        return self.transition_matrix


class _FixedLikelihoodDetector(_DetectorBase):
    """Minimal detector that simulates observations as fixed log likelihoods."""

    def __init__(
        self,
        log_likelihoods: np.ndarray,
        discrete_initial_conditions: np.ndarray | None = None,
        discrete_transition: np.ndarray | None = None,
        discrete_transition_type=None,
        continuous_transition_blocks: list[list[np.ndarray]] | None = None,
    ):
        self._fixed_log_likelihoods = np.asarray(log_likelihoods, dtype=float)

        if discrete_initial_conditions is None:
            discrete_initial_conditions = np.array([0.8, 0.2, 0.0])
        if discrete_transition is None:
            discrete_transition = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        if continuous_transition_blocks is None:
            source_a_to_target_bin_0 = np.array([[1.0, 0.0], [1.0, 0.0]])
            source_b_to_target_bin_1 = np.array([[0.0, 1.0], [0.0, 1.0]])
            identity = np.eye(2)
            uniform = np.ones((2, 2)) / 2.0
            continuous_transition_blocks = [
                [identity, uniform, source_a_to_target_bin_0],
                [uniform, identity, source_b_to_target_bin_1],
                [uniform, uniform, identity],
            ]

        continuous_transition_types = [
            [_FixedTransition(block) for block in row]
            for row in continuous_transition_blocks
        ]

        super().__init__(
            discrete_initial_conditions=discrete_initial_conditions,
            continuous_initial_conditions_types=[UniformInitialConditions()] * 3,
            discrete_transition_type=(
                DiscreteStationaryCustom(values=discrete_transition)
                if discrete_transition_type is None
                else discrete_transition_type
            ),
            discrete_transition_concentration=1.0,
            discrete_transition_stickiness=np.zeros(3),
            discrete_transition_regularization=0.0,
            continuous_transition_types=continuous_transition_types,
            observation_models=[
                ObservationModel(),
                ObservationModel(),
                ObservationModel(),
            ],
            environments=Environment(
                place_bin_size=1.0,
                position_range=[(0.0, 2.0)],
            ),
            infer_track_interior=False,
            state_names=["Source A", "Source B", "Target"],
        )

    def compute_log_likelihood(self, time, *args, is_missing=None):
        return self._fixed_log_likelihoods

    def fit_encoding_model(self, *args, **kwargs):
        return None


def _sample_categorical(rng: np.random.Generator, probabilities: np.ndarray) -> int:
    """Sample one index from a probability vector."""
    return int(rng.choice(probabilities.size, p=probabilities))


def _simulate_expanded_hmm(
    rng: np.random.Generator,
    initial_conditions: np.ndarray,
    discrete_transition: np.ndarray,
    continuous_transition_blocks: list[list[np.ndarray]],
    n_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate discrete states and within-state bins from an expanded HMM."""
    states = np.zeros(n_time, dtype=int)
    bins = np.zeros(n_time, dtype=int)
    states[0] = _sample_categorical(rng, initial_conditions)
    bins[0] = int(
        rng.integers(continuous_transition_blocks[states[0]][states[0]].shape[0])
    )

    for t in range(1, n_time):
        previous_state = states[t - 1]
        previous_bin = bins[t - 1]
        states[t] = _sample_categorical(rng, discrete_transition[previous_state])
        bins[t] = _sample_categorical(
            rng,
            continuous_transition_blocks[previous_state][states[t]][previous_bin],
        )

    return states, bins


def _simulate_nonstationary_expanded_hmm(
    rng: np.random.Generator,
    initial_conditions: np.ndarray,
    discrete_transition: np.ndarray,
    continuous_transition_blocks: list[list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate expanded states with time-varying discrete transitions."""
    n_time = discrete_transition.shape[0]
    states = np.zeros(n_time, dtype=int)
    bins = np.zeros(n_time, dtype=int)
    states[0] = _sample_categorical(rng, initial_conditions)
    bins[0] = int(
        rng.integers(continuous_transition_blocks[states[0]][states[0]].shape[0])
    )

    for t in range(1, n_time):
        previous_state = states[t - 1]
        previous_bin = bins[t - 1]
        states[t] = _sample_categorical(
            rng,
            discrete_transition[t - 1, previous_state],
        )
        bins[t] = _sample_categorical(
            rng,
            continuous_transition_blocks[previous_state][states[t]][previous_bin],
        )

    return states, bins


def _log_likelihoods_from_expanded_states(
    states: np.ndarray,
    bins: np.ndarray,
    n_states: int,
    n_bins: int,
    off_target_log_likelihood: float = -6.0,
) -> np.ndarray:
    """Construct informative simulated log likelihoods from expanded states."""
    log_likelihoods = np.full(
        (states.size, n_states * n_bins), off_target_log_likelihood
    )
    log_likelihoods[np.arange(states.size), states * n_bins + bins] = 0.0
    return log_likelihoods


def _empirical_discrete_transition(states: np.ndarray, n_states: int) -> np.ndarray:
    """Estimate row-stochastic transition probabilities from state samples."""
    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def _centered_softmax_forward_numpy(linear_predictor: np.ndarray) -> np.ndarray:
    """Apply centered softmax with an implicit zero logit for the last state."""
    logits = np.concatenate(
        (
            linear_predictor,
            np.zeros((*linear_predictor.shape[:-1], 1)),
        ),
        axis=-1,
    )
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _expanded_responses_numpy_reference(
    causal_posterior: np.ndarray,
    predictive_posterior: np.ndarray,
    acausal_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Reference implementation matching the original NumPy loop."""
    aggregation = _state_aggregation_matrix(state_ind)
    n_time = causal_posterior.shape[0]
    n_states = aggregation.shape[1]
    response = np.zeros((n_time - 1, n_states, n_states))

    for t in range(n_time - 1):
        ratio = np.divide(
            acausal_posterior[t + 1],
            predictive_posterior[t + 1],
            out=np.zeros_like(acausal_posterior[t + 1]),
            where=~np.isclose(predictive_posterior[t + 1], 0.0),
        )
        transition_t = (
            transition_matrix if transition_matrix.ndim == 2 else transition_matrix[t]
        )
        xi = causal_posterior[t, :, np.newaxis] * transition_t * ratio[np.newaxis, :]
        response[t] = aggregation.T @ xi @ aggregation

    return response


def _expanded_hmm_parameters() -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """Return a small identifiable expanded HMM for transition-recovery tests."""
    initial_conditions = np.full(3, 1.0 / 3.0)
    discrete_transition = np.array(
        [
            [0.72, 0.18, 0.10],
            [0.12, 0.75, 0.13],
            [0.16, 0.14, 0.70],
        ]
    )
    same = np.array([[0.90, 0.10], [0.10, 0.90]])
    flip = np.array([[0.20, 0.80], [0.80, 0.20]])
    left = np.array([[0.85, 0.15], [0.85, 0.15]])
    right = np.array([[0.15, 0.85], [0.15, 0.85]])
    continuous_transition_blocks = [
        [same, left, right],
        [right, same, left],
        [left, right, flip],
    ]
    return initial_conditions, discrete_transition, continuous_transition_blocks


def _estimate_simulated_discrete_transition(
    seed: int,
    n_time: int,
    off_target_log_likelihood: float,
    max_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate an expanded HMM and estimate its discrete transition matrix."""
    rng = np.random.default_rng(seed)
    n_states = 3
    n_bins = 2
    (
        initial_conditions,
        true_discrete_transition,
        continuous_transition_blocks,
    ) = _expanded_hmm_parameters()
    states, bins = _simulate_expanded_hmm(
        rng,
        initial_conditions,
        true_discrete_transition,
        continuous_transition_blocks,
        n_time,
    )
    log_likelihoods = _log_likelihoods_from_expanded_states(
        states,
        bins,
        n_states,
        n_bins,
        off_target_log_likelihood=off_target_log_likelihood,
    )
    detector = _FixedLikelihoodDetector(
        log_likelihoods,
        discrete_initial_conditions=initial_conditions,
        discrete_transition=np.full((n_states, n_states), 1.0 / n_states),
        continuous_transition_blocks=continuous_transition_blocks,
    )
    detector._fit(position=np.array([[0.25], [1.25]]))
    results = detector.estimate_parameters(
        time=np.arange(n_time, dtype=float),
        estimate_initial_conditions=False,
        estimate_discrete_transition=True,
        estimate_encoding_model=False,
        max_iter=max_iter,
        tolerance=0.0,
    )

    return (
        detector.discrete_state_transitions_,
        _empirical_discrete_transition(states, n_states),
        true_discrete_transition,
        results.attrs["marginal_log_likelihoods"],
    )


@pytest.mark.unit
class TestExpandedDiscreteTransitionCounts:
    """Test exact discrete transition counts from expanded-bin posteriors."""

    def test_expanded_responses_match_numpy_reference(self):
        """The JAX response kernel should match the original NumPy formula."""
        rng = np.random.default_rng(5)
        n_time = 6
        state_ind = np.array([0, 0, 1, 2, 2])
        n_bins = state_ind.size
        causal = rng.random((n_time, n_bins))
        causal /= causal.sum(axis=1, keepdims=True)
        transition = rng.random((n_time, n_bins, n_bins))
        transition /= transition.sum(axis=2, keepdims=True)
        predictive = np.einsum("tk,tkl->tl", causal, transition)
        acausal = predictive * rng.uniform(0.8, 1.2, size=predictive.shape)
        acausal /= acausal.sum(axis=1, keepdims=True)

        response = estimate_discrete_transition_responses_from_expanded_posteriors(
            causal,
            predictive,
            acausal,
            transition,
            state_ind,
        )
        expected = _expanded_responses_numpy_reference(
            causal,
            predictive,
            acausal,
            transition,
            state_ind,
        )

        np.testing.assert_allclose(response, expected, atol=1e-6)

    def test_factorized_counts_match_numpy_reference(self):
        """The stationary factorized JAX count helper should preserve answers."""
        rng = np.random.default_rng(6)
        n_time = 7
        state_ind = np.array([0, 0, 1, 2, 2])
        n_bins = state_ind.size
        n_states = 3
        causal = rng.random((n_time, n_bins))
        causal /= causal.sum(axis=1, keepdims=True)
        continuous_transition = rng.random((n_bins, n_bins))
        continuous_transition /= continuous_transition.sum(axis=1, keepdims=True)
        discrete_transition = rng.random((n_states, n_states))
        discrete_transition /= discrete_transition.sum(axis=1, keepdims=True)
        full_transition = (
            continuous_transition * discrete_transition[np.ix_(state_ind, state_ind)]
        )
        predictive = causal @ full_transition
        acausal = predictive * rng.uniform(0.8, 1.2, size=predictive.shape)
        acausal /= acausal.sum(axis=1, keepdims=True)

        counts = estimate_discrete_transition_counts_from_factorized_posteriors(
            causal,
            predictive,
            acausal,
            continuous_transition,
            discrete_transition,
            state_ind,
        )
        expected = _expanded_responses_numpy_reference(
            causal,
            predictive,
            acausal,
            full_transition,
            state_ind,
        ).sum(axis=0)

        np.testing.assert_allclose(counts, expected, atol=1e-6)

    def test_stationary_factorized_responses_match_materialized_full_transition(self):
        """Stationary factorized responses should avoid changing the math."""
        rng = np.random.default_rng(8)
        n_time = 7
        state_ind = np.array([0, 0, 1, 2, 2])
        n_bins = state_ind.size
        n_states = 3
        causal = rng.random((n_time, n_bins))
        causal /= causal.sum(axis=1, keepdims=True)
        continuous_transition = rng.random((n_bins, n_bins))
        continuous_transition /= continuous_transition.sum(axis=1, keepdims=True)
        discrete_transition = rng.random((n_states, n_states))
        discrete_transition /= discrete_transition.sum(axis=1, keepdims=True)
        full_transition = (
            continuous_transition * discrete_transition[np.ix_(state_ind, state_ind)]
        )
        predictive = causal @ full_transition
        acausal = predictive * rng.uniform(0.8, 1.2, size=predictive.shape)
        acausal /= acausal.sum(axis=1, keepdims=True)

        factorized = estimate_discrete_transition_responses_from_factorized_posteriors(
            causal,
            predictive,
            acausal,
            continuous_transition,
            discrete_transition,
            state_ind,
        )
        materialized = estimate_discrete_transition_responses_from_expanded_posteriors(
            causal,
            predictive,
            acausal,
            full_transition,
            state_ind,
        )

        np.testing.assert_allclose(factorized, materialized, atol=1e-6)

    def test_factorized_counts_requires_stationary_discrete_transition(self):
        """The count helper should reject time-varying discrete transitions."""
        causal = np.array([[0.5, 0.5], [0.25, 0.75]])
        predictive = np.array([[0.5, 0.5], [0.25, 0.75]])
        acausal = np.array([[0.5, 0.5], [0.25, 0.75]])
        continuous_transition = np.eye(2)
        discrete_transition = np.tile(np.eye(2), (2, 1, 1))
        state_ind = np.array([0, 1])

        with pytest.raises(ValueError, match="must be stationary"):
            estimate_discrete_transition_counts_from_factorized_posteriors(
                causal,
                predictive,
                acausal,
                continuous_transition,
                discrete_transition,
                state_ind,
            )

    def test_expanded_counts_rejects_broadcastable_transition_shape(self):
        """Broadcastable expanded transitions should not produce silent counts."""
        causal = np.ones((3, 3)) / 3.0
        predictive = causal.copy()
        acausal = causal.copy()
        state_ind = np.array([0, 1, 2])
        transition = np.array([[0.2, 0.3, 0.5]])

        with pytest.raises(ValueError, match="transition_matrix"):
            estimate_discrete_transition_counts_from_expanded_posteriors(
                causal,
                predictive,
                acausal,
                transition,
                state_ind,
            )

    def test_expanded_responses_rejects_bad_time_transition_shape(self):
        """Time-varying expanded transitions must match n_time and n_state_bins."""
        causal = np.ones((3, 3)) / 3.0
        predictive = causal.copy()
        acausal = causal.copy()
        state_ind = np.array([0, 1, 2])
        transition = np.ones((3, 1, 3)) / 3.0

        with pytest.raises(ValueError, match="transition_matrix"):
            estimate_discrete_transition_responses_from_expanded_posteriors(
                causal,
                predictive,
                acausal,
                transition,
                state_ind,
            )

    def test_factorized_counts_rejects_broadcastable_discrete_transition_shape(self):
        """Broadcastable stationary discrete transitions should raise."""
        causal = np.ones((3, 3)) / 3.0
        predictive = causal.copy()
        acausal = causal.copy()
        continuous_transition = np.eye(3)
        discrete_transition = np.array([[0.2, 0.3, 0.5]])
        state_ind = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="discrete_transition_matrix"):
            estimate_discrete_transition_counts_from_factorized_posteriors(
                causal,
                predictive,
                acausal,
                continuous_transition,
                discrete_transition,
                state_ind,
            )

    def test_factorized_responses_rejects_bad_time_discrete_transition_shape(self):
        """Time-varying discrete transitions must match n_time and n_states."""
        causal = np.ones((3, 3)) / 3.0
        predictive = causal.copy()
        acausal = causal.copy()
        continuous_transition = np.eye(3)
        discrete_transition = np.ones((3, 1, 3)) / 3.0
        state_ind = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="discrete_transition_matrix"):
            estimate_discrete_transition_responses_from_factorized_posteriors(
                causal,
                predictive,
                acausal,
                continuous_transition,
                discrete_transition,
                state_ind,
            )

    def test_factorized_responses_rejects_bad_continuous_transition_shape(self):
        """Continuous transitions must match expanded state bins exactly."""
        causal = np.ones((3, 3)) / 3.0
        predictive = causal.copy()
        acausal = causal.copy()
        continuous_transition = np.array([[0.2, 0.3, 0.5]])
        discrete_transition = np.eye(3)
        state_ind = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="continuous_transition_matrix"):
            estimate_discrete_transition_responses_from_factorized_posteriors(
                causal,
                predictive,
                acausal,
                continuous_transition,
                discrete_transition,
                state_ind,
            )

    def test_pure_discrete_hmm_matches_legacy_joint_sum(self, posterior_data):
        """One expanded bin per state should match the existing discrete formula."""
        post = posterior_data
        state_ind = np.arange(post["n_states"])

        exact_counts = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal_posterior=post["causal_posterior"],
            predictive_posterior=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            state_ind=state_ind,
        )
        legacy_counts = estimate_joint_distribution(
            post["causal_posterior"],
            post["predictive_distribution"],
            post["transition_matrix"],
            post["acausal_posterior"],
        ).sum(axis=0)

        np.testing.assert_allclose(exact_counts, legacy_counts, atol=1e-5)

    def test_uniform_continuous_transitions_match_aggregated_counts(self):
        """Source-independent target-bin predictions reduce to the aggregate path."""
        state_ind = np.array([0, 1, 1])
        discrete_transition = np.array(
            [
                [0.4, 0.6],
                [0.3, 0.7],
            ]
        )
        continuous_transition = np.array(
            [
                [1.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
            ]
        )
        full_transition = (
            continuous_transition * discrete_transition[np.ix_(state_ind, state_ind)]
        )
        causal = np.array(
            [
                [0.2, 0.5, 0.3],
                [0.3, 0.4, 0.3],
                [0.6, 0.2, 0.2],
            ]
        )
        predictive = causal @ full_transition
        acausal = np.array(
            [
                [0.2, 0.5, 0.3],
                [0.4, 0.3, 0.3],
                [0.5, 0.25, 0.25],
            ]
        )

        exact_counts = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal, predictive, acausal, full_transition, state_ind
        )

        causal_state = np.column_stack((causal[:, 0], causal[:, 1:].sum(axis=1)))
        predictive_state = np.column_stack(
            (predictive[:, 0], predictive[:, 1:].sum(axis=1))
        )
        acausal_state = np.column_stack((acausal[:, 0], acausal[:, 1:].sum(axis=1)))
        legacy_counts = estimate_joint_distribution(
            causal_state,
            predictive_state,
            discrete_transition,
            acausal_state,
        ).sum(axis=0)

        np.testing.assert_allclose(exact_counts, legacy_counts, atol=1e-5)

    def test_source_specific_spatial_prediction_changes_transition_credit(self):
        """Exact counts credit the source that predicts the supported target bin."""
        state_ind = np.array([0, 1, 2, 2])
        full_transition = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        causal = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        predictive = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2],
            ]
        )
        acausal = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        exact_counts = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal, predictive, acausal, full_transition, state_ind
        )

        discrete_transition = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        causal_state = np.column_stack((causal[:, :2], causal[:, 2:].sum(axis=1)))
        predictive_state = np.column_stack(
            (predictive[:, :2], predictive[:, 2:].sum(axis=1))
        )
        acausal_state = np.column_stack((acausal[:, :2], acausal[:, 2:].sum(axis=1)))
        legacy_counts = estimate_joint_distribution(
            causal_state,
            predictive_state,
            discrete_transition,
            acausal_state,
        ).sum(axis=0)

        assert exact_counts[1, 2] > exact_counts[0, 2]
        assert legacy_counts[0, 2] > legacy_counts[1, 2]

    def test_zero_predictive_bins_are_finite(self):
        """Zero predictive bins should not create NaN or inf counts."""
        state_ind = np.array([0, 1])
        transition_matrix = np.eye(2)
        causal = np.array([[1.0, 0.0], [0.0, 1.0]])
        predictive = np.array([[1.0, 0.0], [0.0, 0.0]])
        acausal = np.array([[1.0, 0.0], [0.0, 1.0]])

        counts = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal, predictive, acausal, transition_matrix, state_ind
        )

        assert np.all(np.isfinite(counts))
        np.testing.assert_array_equal(counts, np.zeros((2, 2)))

    def test_nonstationary_responses_sum_to_stationary_counts(self, posterior_data):
        """The response helper should reduce to the count helper when summed."""
        post = posterior_data
        state_ind = np.arange(post["n_states"])
        transition_matrix = np.tile(post["transition_matrix"], (post["n_time"], 1, 1))

        response = estimate_discrete_transition_responses_from_expanded_posteriors(
            post["causal_posterior"],
            post["predictive_distribution"],
            post["acausal_posterior"],
            transition_matrix,
            state_ind,
        )
        counts = estimate_discrete_transition_counts_from_expanded_posteriors(
            post["causal_posterior"],
            post["predictive_distribution"],
            post["acausal_posterior"],
            post["transition_matrix"],
            state_ind,
        )

        np.testing.assert_allclose(response.sum(axis=0), counts, atol=1e-5)

    def test_factorized_responses_match_materialized_full_transition(self):
        """The streaming nonstationary helper should avoid changing the math."""
        state_ind = np.array([0, 1, 2, 2])
        discrete_transition = np.tile(
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            (2, 1, 1),
        )
        continuous_transition = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        full_transition = (
            continuous_transition[np.newaxis]
            * discrete_transition[:, state_ind][:, :, state_ind]
        )
        causal = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        predictive = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2],
            ]
        )
        acausal = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        factorized = estimate_discrete_transition_responses_from_factorized_posteriors(
            causal,
            predictive,
            acausal,
            continuous_transition,
            discrete_transition,
            state_ind,
        )
        materialized = estimate_discrete_transition_responses_from_expanded_posteriors(
            causal,
            predictive,
            acausal,
            full_transition,
            state_ind,
        )

        np.testing.assert_allclose(factorized, materialized, atol=1e-5)

    def test_nonstationary_factorized_responses_reduce_to_expanded_counts(self):
        """Summed factorized responses should match materialized expanded counts."""
        rng = np.random.default_rng(7)
        n_time = 6
        state_ind = np.array([0, 0, 1, 2, 2])
        n_bins = state_ind.size
        n_states = 3
        causal = rng.random((n_time, n_bins))
        causal /= causal.sum(axis=1, keepdims=True)
        continuous_transition = rng.random((n_bins, n_bins))
        continuous_transition /= continuous_transition.sum(axis=1, keepdims=True)
        discrete_transition = rng.random((n_time, n_states, n_states))
        discrete_transition /= discrete_transition.sum(axis=2, keepdims=True)
        full_transition = (
            continuous_transition[np.newaxis]
            * discrete_transition[:, state_ind][:, :, state_ind]
        )
        predictive = np.einsum("tk,tkl->tl", causal, full_transition)
        acausal = predictive * rng.uniform(0.8, 1.2, size=predictive.shape)
        acausal /= acausal.sum(axis=1, keepdims=True)

        factorized_response = (
            estimate_discrete_transition_responses_from_factorized_posteriors(
                causal,
                predictive,
                acausal,
                continuous_transition,
                discrete_transition,
                state_ind,
            )
        )
        materialized_counts = (
            estimate_discrete_transition_counts_from_expanded_posteriors(
                causal,
                predictive,
                acausal,
                full_transition,
                state_ind,
            )
        )

        np.testing.assert_allclose(
            factorized_response.sum(axis=0),
            materialized_counts,
            atol=1e-5,
        )


@pytest.mark.integration
class TestExpandedTransitionMstepEndToEnd:
    """Detector-level tests for exact expanded-state transition learning."""

    def test_simulated_likelihoods_shift_learned_transition_to_matching_source(self):
        """The learned matrix should credit the source with matching target bins."""
        log_likelihoods = np.full((2, 6), -1000.0)
        log_likelihoods[0, 0:4] = 0.0
        log_likelihoods[1, 5] = 0.0

        detector = _FixedLikelihoodDetector(log_likelihoods)
        detector._fit(position=np.array([[0.25], [1.25]]))
        detector.estimate_parameters(
            time=np.array([0.0, 1.0]),
            estimate_initial_conditions=False,
            estimate_discrete_transition=True,
            estimate_encoding_model=False,
            max_iter=1,
        )

        learned_transition = detector.discrete_state_transitions_

        assert learned_transition[1, 2] > learned_transition[0, 2]
        assert learned_transition[1, 2] > 0.99
        assert_stochastic_matrix(learned_transition)

    def test_exact_update_beats_aggregated_negative_control(self):
        """The detector update should avoid aggregate-state source credit."""
        log_likelihoods = np.full((2, 6), -1000.0)
        log_likelihoods[0, 0:4] = 0.0
        log_likelihoods[1, 5] = 0.0

        detector = _FixedLikelihoodDetector(log_likelihoods)
        detector._fit(position=np.array([[0.25], [1.25]]))
        initial_discrete_transition = detector.discrete_state_transitions_.copy()
        results = detector.estimate_parameters(
            time=np.array([0.0, 1.0]),
            estimate_initial_conditions=False,
            estimate_discrete_transition=True,
            estimate_encoding_model=False,
            max_iter=1,
            return_outputs="all",
        )

        aggregate_counts = estimate_joint_distribution(
            results.causal_state_probabilities.values,
            results.predictive_state_probabilities.values,
            initial_discrete_transition,
            results.acausal_state_probabilities.values,
        ).sum(axis=0)
        learned_transition = detector.discrete_state_transitions_

        assert aggregate_counts[0, 2] > aggregate_counts[1, 2]
        assert learned_transition[1, 2] > learned_transition[0, 2]
        assert learned_transition[1, 2] > 0.99
        assert_stochastic_matrix(learned_transition)

    def test_simulated_hmm_recovers_stationary_discrete_transition(self):
        """With informative simulated emissions, the M-step recovers A."""
        learned_transition, empirical_transition, true_discrete_transition, _ = (
            _estimate_simulated_discrete_transition(
                seed=11,
                n_time=5_000,
                off_target_log_likelihood=-12.0,
            )
        )

        assert_stochastic_matrix(learned_transition)
        np.testing.assert_allclose(
            learned_transition,
            empirical_transition,
            atol=0.015,
        )
        np.testing.assert_allclose(
            empirical_transition,
            true_discrete_transition,
            atol=0.04,
        )

    def test_recovery_improves_with_emission_strength_and_sample_size(self):
        """Recovery should improve as emissions and sample size become stronger."""
        emission_errors = []
        for off_target_log_likelihood in (-3.0, -6.0, -12.0):
            learned_transition, empirical_transition, _, _ = (
                _estimate_simulated_discrete_transition(
                    seed=11,
                    n_time=2_000,
                    off_target_log_likelihood=off_target_log_likelihood,
                )
            )
            emission_errors.append(
                np.max(np.abs(learned_transition - empirical_transition))
            )

        short_learned, _, short_true, _ = _estimate_simulated_discrete_transition(
            seed=11,
            n_time=300,
            off_target_log_likelihood=-12.0,
        )
        long_learned, _, long_true, _ = _estimate_simulated_discrete_transition(
            seed=11,
            n_time=5_000,
            off_target_log_likelihood=-12.0,
        )
        short_error = np.max(np.abs(short_learned - short_true))
        long_error = np.max(np.abs(long_learned - long_true))

        assert emission_errors[0] > emission_errors[1] > emission_errors[2]
        assert emission_errors[2] < 0.001
        assert long_error < short_error
        assert long_error < 0.02

    def test_exact_discrete_transition_em_is_monotonic(self):
        """Repeated exact transition M-steps should not decrease likelihood."""
        _, _, _, marginal_log_likelihoods = _estimate_simulated_discrete_transition(
            seed=12,
            n_time=600,
            off_target_log_likelihood=-6.0,
            max_iter=5,
        )

        assert len(marginal_log_likelihoods) == 5
        assert np.all(np.diff(marginal_log_likelihoods) >= -1e-6)

    def test_nonstationary_detector_recovers_covariate_direction(self):
        """Exact responses should learn a simple covariate-dependent transition."""
        rng = np.random.default_rng(13)
        n_time = 1_200
        n_states = 3
        n_bins = 2
        covariate = np.sin(np.linspace(0.0, 12.0 * np.pi, n_time))
        design_matrix = np.column_stack((np.ones(n_time), covariate))
        true_coefficients = np.zeros((2, n_states, n_states - 1))
        true_coefficients[:, 0, :] = np.array([[1.5, -0.5], [-2.0, 2.0]])
        true_coefficients[:, 1, :] = np.array([[-0.5, 1.5], [2.0, -2.0]])
        true_coefficients[:, 2, :] = np.array([[0.5, 0.0], [1.2, -1.2]])
        true_transition = np.zeros((n_time, n_states, n_states))
        for from_state in range(n_states):
            true_transition[:, from_state] = _centered_softmax_forward_numpy(
                design_matrix @ true_coefficients[:, from_state]
            )

        initial_conditions, _, continuous_transition_blocks = _expanded_hmm_parameters()
        states, bins = _simulate_nonstationary_expanded_hmm(
            rng,
            initial_conditions,
            true_transition,
            continuous_transition_blocks,
        )
        log_likelihoods = _log_likelihoods_from_expanded_states(
            states,
            bins,
            n_states,
            n_bins,
            off_target_log_likelihood=-12.0,
        )
        detector = _FixedLikelihoodDetector(
            log_likelihoods,
            discrete_initial_conditions=initial_conditions,
            discrete_transition_type=DiscreteNonStationaryCustom(
                values=np.full((n_states, n_states), 1.0 / n_states),
                formula="1 + covariate",
            ),
            continuous_transition_blocks=continuous_transition_blocks,
        )
        covariate_data = {"covariate": covariate}
        detector._fit(
            position=np.array([[0.25], [1.25]]),
            discrete_transition_covariate_data=covariate_data,
        )
        results = detector.estimate_parameters(
            time=np.arange(n_time, dtype=float),
            estimate_initial_conditions=False,
            estimate_discrete_transition=True,
            estimate_encoding_model=False,
            max_iter=1,
            tolerance=0.0,
        )

        learned_transition = detector.discrete_state_transitions_
        low_covariate = covariate[:-1] < -0.75
        high_covariate = covariate[:-1] > 0.75
        learned_low = learned_transition[:-1][low_covariate].mean(axis=0)
        learned_high = learned_transition[:-1][high_covariate].mean(axis=0)
        true_low = true_transition[:-1][low_covariate].mean(axis=0)
        true_high = true_transition[:-1][high_covariate].mean(axis=0)

        assert learned_low[0, 0] > learned_high[0, 0]
        assert learned_high[0, 1] > learned_low[0, 1]
        assert learned_high[1, 0] > learned_low[1, 0]
        assert learned_low[1, 1] > learned_high[1, 1]
        np.testing.assert_allclose(learned_low[:2], true_low[:2], atol=0.16)
        np.testing.assert_allclose(learned_high[:2], true_high[:2], atol=0.16)
        assert len(results.attrs["marginal_log_likelihoods"]) == 1


@pytest.mark.unit
class TestEstimateNonStationaryStateTransition:
    """Test non-stationary state transition estimation (EM algorithm)."""

    def test_returns_correct_shapes(self, posterior_data, design_matrix_data):
        """Verify coefficient and transition matrix shapes are correct."""
        # Arrange
        post = posterior_data
        dm = design_matrix_data

        # Act
        coeffs, trans_matrix = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=0.0,  # No stickiness (uniform prior)
            transition_regularization=1e-5,
            maxiter=10,  # Limit iterations for speed
        )

        # Assert
        n_coeffs, n_states = dm["n_coefficients"], post["n_states"]
        assert coeffs.shape == (n_coeffs, n_states, n_states - 1)
        assert trans_matrix.shape == (post["n_time"], n_states, n_states)

    def test_produces_valid_probabilities(self, posterior_data, design_matrix_data):
        """Check all transition matrices are valid stochastic matrices."""
        # Arrange
        post = posterior_data
        dm = design_matrix_data

        # Act
        _, trans_matrix = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Assert - check each time step
        for t in range(trans_matrix.shape[0]):
            assert_stochastic_matrix(trans_matrix[t])

    def test_concentration_below_one_raises(self, posterior_data, design_matrix_data):
        """The nonstationary MAP update rejects negative pseudo-counts."""
        post = posterior_data
        dm = design_matrix_data

        with pytest.raises(ValueError, match="prior parameters >= 1.0"):
            estimate_non_stationary_state_transition(
                causal_posterior=post["causal_posterior"],
                predictive_distribution=post["predictive_distribution"],
                acausal_posterior=post["acausal_posterior"],
                transition_matrix=post["transition_matrix"],
                design_matrix=dm["design_matrix"][: post["n_time"]],
                transition_coefficients=dm["transition_coefficients"],
                concentration=0.5,
                stickiness=0.0,
                transition_regularization=1e-5,
                maxiter=10,
            )

    def test_from_responses_matches_posterior_wrapper(
        self, posterior_data, design_matrix_data
    ):
        """Precomputed responses should match the existing aggregate wrapper."""
        post = posterior_data
        dm = design_matrix_data
        design_matrix = dm["design_matrix"][: post["n_time"]]

        expected = estimate_joint_distribution(
            post["causal_posterior"],
            post["predictive_distribution"],
            post["transition_matrix"],
            post["acausal_posterior"],
        )

        coeffs_from_response, trans_from_response = (
            estimate_non_stationary_state_transition_from_responses(
                transition_coefficients=dm["transition_coefficients"],
                design_matrix=design_matrix,
                response=expected,
                concentration=1.0,
                stickiness=0.0,
                transition_regularization=1e-5,
                maxiter=10,
            )
        )
        coeffs_from_wrapper, trans_from_wrapper = (
            estimate_non_stationary_state_transition(
                causal_posterior=post["causal_posterior"],
                predictive_distribution=post["predictive_distribution"],
                acausal_posterior=post["acausal_posterior"],
                transition_matrix=post["transition_matrix"],
                design_matrix=design_matrix,
                transition_coefficients=dm["transition_coefficients"],
                concentration=1.0,
                stickiness=0.0,
                transition_regularization=1e-5,
                maxiter=10,
            )
        )

        np.testing.assert_allclose(coeffs_from_response, coeffs_from_wrapper)
        np.testing.assert_allclose(trans_from_response, trans_from_wrapper)

    def test_from_responses_recovers_covariate_dependent_transition(self):
        """Sampled responses should recover a known nonstationary transition."""
        rng = np.random.default_rng(0)
        n_time = 600
        n_states = 3
        n_coefficients = 2
        covariate = np.linspace(-1.0, 1.0, n_time)
        design_matrix = np.column_stack((np.ones(n_time), covariate))
        true_coefficients = np.zeros((n_coefficients, n_states, n_states - 1))
        true_coefficients[:, 0, :] = np.array([[1.0, -0.5], [1.2, -0.8]])
        true_coefficients[:, 1, :] = np.array([[-0.5, 1.0], [-1.0, 0.9]])
        true_coefficients[:, 2, :] = np.array([[0.4, 0.2], [-0.7, 1.1]])
        true_transition = np.zeros((n_time, n_states, n_states))
        for from_state in range(n_states):
            true_transition[:, from_state] = _centered_softmax_forward_numpy(
                design_matrix @ true_coefficients[:, from_state, :]
            )

        response = np.zeros((n_time - 1, n_states, n_states))
        for t in range(n_time - 1):
            for from_state in range(n_states):
                response[t, from_state] = rng.multinomial(
                    50,
                    true_transition[t, from_state],
                )

        _, estimated_transition = (
            estimate_non_stationary_state_transition_from_responses(
                transition_coefficients=np.zeros_like(true_coefficients),
                design_matrix=design_matrix,
                response=response,
                concentration=1.0,
                stickiness=0.0,
                transition_regularization=1e-8,
                maxiter=200,
            )
        )

        np.testing.assert_allclose(
            estimated_transition[:-1],
            true_transition[:-1],
            atol=0.02,
        )
        assert estimated_transition[-2, 0, 0] > estimated_transition[0, 0, 0]
        assert estimated_transition[-2, 1, 1] > estimated_transition[0, 1, 1]

    def test_with_different_concentrations(self, posterior_data, design_matrix_data):
        """Test with different concentration (prior strength) values."""
        post = posterior_data
        dm = design_matrix_data

        # Test with weak prior (concentration=1.0)
        _, trans_weak = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Test with strong prior (concentration=10.0)
        _, trans_strong = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=10.0,
            stickiness=0.0,  # uniform prior
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Both should be valid
        for t in range(min(5, trans_weak.shape[0])):  # Check first 5 timesteps
            assert_stochastic_matrix(trans_weak[t])
            assert_stochastic_matrix(trans_strong[t])

    def test_with_diagonal_stickiness(self, posterior_data, design_matrix_data):
        """Test with diagonal stickiness prior."""
        post = posterior_data
        dm = design_matrix_data

        # Act
        _, trans_matrix = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=1.0,  # Diagonal stickiness - favors self-transitions
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Assert - diagonal should be larger than off-diagonal on average
        diagonal_mean = np.mean(
            [
                trans_matrix[t, i, i]
                for t in range(trans_matrix.shape[0])
                for i in range(post["n_states"])
            ]
        )
        off_diagonal_mean = np.mean(
            [
                trans_matrix[t, i, j]
                for t in range(trans_matrix.shape[0])
                for i in range(post["n_states"])
                for j in range(post["n_states"])
                if i != j
            ]
        )

        assert diagonal_mean > off_diagonal_mean, (
            "Diagonal stickiness should favor self-transitions"
        )

    def test_keeps_previous_coefficients_on_optimizer_failure(
        self, posterior_data, design_matrix_data
    ):
        """When optimizer fails, previous coefficients should be retained."""
        post = posterior_data
        dm = design_matrix_data
        n_coeffs = dm["n_coefficients"]
        n_states = post["n_states"]

        # Use known initial coefficients
        rng = np.random.default_rng(99)
        initial_coeffs = rng.standard_normal((n_coeffs, n_states, n_states - 1)) * 0.1

        # Mock minimize to always return failure
        class FakeResult:
            success = False
            message = "mock failure"
            x = np.full(n_coeffs * (n_states - 1), 999.0)  # garbage values

        with patch(
            "non_local_detector.discrete_state_transitions.minimize",
            return_value=FakeResult(),
        ):
            coeffs, trans_matrix = estimate_non_stationary_state_transition(
                causal_posterior=post["causal_posterior"],
                predictive_distribution=post["predictive_distribution"],
                acausal_posterior=post["acausal_posterior"],
                transition_matrix=post["transition_matrix"],
                design_matrix=dm["design_matrix"][: post["n_time"]],
                transition_coefficients=initial_coeffs,
                maxiter=10,
            )

        # Coefficients should be unchanged from input (not garbage)
        np.testing.assert_array_equal(coeffs, initial_coeffs)

        # Transition matrix should still be valid stochastic matrices
        for t in range(trans_matrix.shape[0]):
            assert_stochastic_matrix(trans_matrix[t])


@pytest.mark.unit
class TestEstimateStationaryStateTransition:
    """Test stationary state transition estimation."""

    def test_returns_stochastic_matrix(self, posterior_data):
        """Verify output is a valid stochastic matrix."""
        # Arrange
        post = posterior_data

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
        )

        # Assert
        assert trans_matrix.shape == (post["n_states"], post["n_states"])
        assert_stochastic_matrix(trans_matrix)

    def test_respects_uniform_prior(self, posterior_data):
        """Test with uniform prior (concentration=1.0)."""
        post = posterior_data

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
        )

        # Assert - should be a valid stochastic matrix
        assert_stochastic_matrix(trans_matrix)
        # All probabilities should be reasonable (not extreme)
        assert np.all(trans_matrix > 1e-6), (
            "No probability should be exactly zero with uniform prior"
        )

    def test_respects_diagonal_prior(self, posterior_data):
        """Test with diagonal stickiness prior."""
        post = posterior_data

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,  # Diagonal stickiness
        )

        # Assert
        assert_stochastic_matrix(trans_matrix)
        # Diagonal should be favored
        diagonal_mean = np.mean(np.diag(trans_matrix))
        off_diagonal_mean = np.mean(trans_matrix[~np.eye(post["n_states"], dtype=bool)])
        assert diagonal_mean > off_diagonal_mean

    def test_numerical_stability_with_small_probabilities(self):
        """Test with very small posterior probabilities."""
        # Arrange - create extreme case with very small probabilities
        n_time, n_states = 20, 4
        causal_posterior = np.ones((n_time, n_states)) * 1e-8
        causal_posterior[:, 0] = 1.0 - 3e-8
        acausal_posterior = causal_posterior.copy()

        transition_matrix = np.eye(n_states) * 0.9 + 0.025
        predictive_distribution = np.zeros((n_time, n_states))
        for t in range(n_time):
            predictive_distribution[t] = causal_posterior[t] @ transition_matrix

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=causal_posterior,
            predictive_distribution=predictive_distribution,
            acausal_posterior=acausal_posterior,
            transition_matrix=transition_matrix,
            concentration=1.0,
            stickiness=0.0,  # uniform prior
        )

        # Assert - should not contain NaN or inf
        assert np.all(np.isfinite(trans_matrix))
        assert_stochastic_matrix(trans_matrix)

    def test_from_counts_matches_posterior_wrapper(self, posterior_data):
        """The count-based estimator should preserve legacy posterior behavior."""
        post = posterior_data
        joint_sum = estimate_joint_distribution(
            post["causal_posterior"],
            post["predictive_distribution"],
            post["transition_matrix"],
            post["acausal_posterior"],
        ).sum(axis=0)

        from_counts = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )
        from_posteriors = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )

        np.testing.assert_allclose(from_counts, from_posteriors, atol=1e-12)

    def test_concentration_below_one_raises(self, posterior_data):
        """The MAP pseudo-count update rejects sparse Dirichlet parameters."""
        post = posterior_data

        with pytest.raises(ValueError, match="prior parameters >= 1.0"):
            estimate_stationary_state_transition(
                causal_posterior=post["causal_posterior"],
                predictive_distribution=post["predictive_distribution"],
                acausal_posterior=post["acausal_posterior"],
                transition_matrix=post["transition_matrix"],
                concentration=0.5,
                stickiness=0.0,
            )


@pytest.mark.unit
class TestPriorWeightScaling:
    """Test data-adaptive prior weight for stationary transition estimation."""

    def test_prior_weight_zero_uses_legacy_prior(self, posterior_data):
        """prior_weight=0 should use the legacy fixed-count prior (unchanged behavior)."""
        post = posterior_data

        legacy_result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
        )

        pw_result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.0,
        )

        # prior_weight=0 should produce identical results to legacy behavior
        np.testing.assert_allclose(pw_result, legacy_result, atol=1e-12)

    def test_prior_weight_increases_diagonal(self, posterior_data):
        """prior_weight > 0 with stickiness should increase diagonal relative to MLE."""
        post = posterior_data

        mle_result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=0.0,
        )

        pw_result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )

        assert_stochastic_matrix(pw_result)
        # Diagonal should be larger with prior_weight + stickiness
        assert np.mean(np.diag(pw_result)) > np.mean(np.diag(mle_result))

    def test_prior_weight_approximately_invariant_to_T(self, posterior_data):
        """Prior effect should be similar regardless of number of time bins.

        With prior_weight, doubling T (by repeating data) should produce
        approximately the same transition matrix, unlike the fixed-count prior
        which gets diluted.
        """
        post = posterior_data

        # Estimate with original T
        result_T = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )

        # Estimate with 2T (repeat data)
        result_2T = estimate_stationary_state_transition(
            causal_posterior=np.tile(post["causal_posterior"], (2, 1)),
            predictive_distribution=np.tile(post["predictive_distribution"], (2, 1)),
            acausal_posterior=np.tile(post["acausal_posterior"], (2, 1)),
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )

        assert_stochastic_matrix(result_T)
        assert_stochastic_matrix(result_2T)
        # Should be close — prior scales with data, so relative effect is constant
        np.testing.assert_allclose(result_T, result_2T, atol=0.05)

    def test_prior_weight_produces_valid_stochastic_matrix(self, posterior_data):
        """Result should always be a valid stochastic matrix."""
        post = posterior_data

        result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.5,
            stickiness=3.0,
            prior_weight=0.5,
        )

        assert_stochastic_matrix(result)
        assert np.all(np.isfinite(result))

    def test_prior_weight_handles_unvisited_state(self):
        """A state with zero posterior mass should not produce NaN."""
        n_time, n_states = 20, 3

        # State 2 is never visited — all mass on states 0 and 1
        causal_posterior = np.zeros((n_time, n_states))
        causal_posterior[:, 0] = 0.6
        causal_posterior[:, 1] = 0.4
        acausal_posterior = causal_posterior.copy()

        transition_matrix = np.eye(n_states) * 0.8 + 0.2 / n_states
        predictive_distribution = np.zeros((n_time, n_states))
        for t in range(n_time):
            predictive_distribution[t] = causal_posterior[t] @ transition_matrix

        result = estimate_stationary_state_transition(
            causal_posterior=causal_posterior,
            predictive_distribution=predictive_distribution,
            acausal_posterior=acausal_posterior,
            transition_matrix=transition_matrix,
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )

        assert np.all(np.isfinite(result)), f"NaN/Inf in result: {result}"
        assert_stochastic_matrix(result)

        # Unvisited state (index 2) should reflect the sticky prior direction,
        # not uniform — diagonal should be larger than off-diagonal
        assert result[2, 2] > result[2, 0], (
            "Unvisited state should reflect sticky prior, not uniform"
        )

    def test_per_state_prior_weight_scalar_equivalence(self, posterior_data):
        """Passing scalar prior_weight should match passing uniform array."""
        post = posterior_data
        n_states = post["n_states"]

        scalar_result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )
        array_result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,
            prior_weight=np.full(n_states, 0.1),
        )

        np.testing.assert_allclose(scalar_result, array_result, atol=1e-12)

    def test_per_state_prior_weight_mixed_modes(self, posterior_data):
        """Rows with prior_weight=0 use legacy; rows with >0 use data-adaptive."""
        post = posterior_data
        n_states = post["n_states"]

        # State 0 uses legacy path (strong fixed prior), others use adaptive
        prior_weight = np.array([0.0] + [0.1] * (n_states - 1))
        sticky = np.array([1e6] + [2.0] * (n_states - 1))

        result = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=sticky,
            prior_weight=prior_weight,
        )

        assert_stochastic_matrix(result)
        # State 0 (frozen) should have near-1 diagonal due to huge stickiness
        assert result[0, 0] > 0.99

    def test_per_state_prior_weight_row_0_frozen_matches_legacy(self, posterior_data):
        """For frozen row (prior_weight=0), result should match pure legacy call."""
        post = posterior_data
        n_states = post["n_states"]

        sticky = np.array([1e6] + [2.0] * (n_states - 1))

        legacy = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=sticky,
            prior_weight=0.0,
        )
        mixed = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=sticky,
            prior_weight=np.array([0.0] + [0.1] * (n_states - 1)),
        )

        # Row 0 (legacy) should match in both calls
        np.testing.assert_allclose(mixed[0], legacy[0], atol=1e-12)

    def test_per_state_prior_weight_negative_raises(self, posterior_data):
        """Negative per-state prior_weight should raise ValueError."""
        post = posterior_data
        n_states = post["n_states"]

        with pytest.raises(ValueError, match="non-negative"):
            estimate_stationary_state_transition(
                causal_posterior=post["causal_posterior"],
                predictive_distribution=post["predictive_distribution"],
                acausal_posterior=post["acausal_posterior"],
                transition_matrix=post["transition_matrix"],
                concentration=1.0,
                stickiness=2.0,
                prior_weight=np.array([0.1, -0.1] + [0.1] * (n_states - 2)),
            )

    def test_per_state_prior_weight_wrong_shape_raises(self, posterior_data):
        """prior_weight array with wrong length should raise ValueError."""
        post = posterior_data

        with pytest.raises(ValueError, match="prior_weight"):
            estimate_stationary_state_transition(
                causal_posterior=post["causal_posterior"],
                predictive_distribution=post["predictive_distribution"],
                acausal_posterior=post["acausal_posterior"],
                transition_matrix=post["transition_matrix"],
                concentration=1.0,
                stickiness=2.0,
                prior_weight=np.array([0.1, 0.1]),  # wrong length
            )


@pytest.mark.unit
class TestEstimateDiscreteTransition:
    """Test _estimate_discrete_transition wrapper function."""

    def test_stationary_diagonal(self, posterior_data):
        """Test with stationary diagonal transition type."""
        # Arrange
        post = posterior_data

        # Act - _estimate_discrete_transition uses transition matrix directly
        new_trans, _ = _estimate_discrete_transition(
            causal_state_probabilities=post["causal_posterior"],
            predictive_state_probabilities=post["predictive_distribution"],
            acausal_state_probabilities=post["acausal_posterior"],
            discrete_transition=post["transition_matrix"],
            discrete_transition_coefficients=None,
            discrete_transition_design_matrix=None,
            transition_concentration=1.0,
            transition_stickiness=1.0,  # Diagonal stickiness
            transition_regularization=1e-5,
        )

        # Assert
        assert new_trans.shape == (post["n_states"], post["n_states"])
        assert_stochastic_matrix(new_trans)

    def test_stationary_uses_expanded_counts_when_provided(self):
        """Expanded inputs should override the approximate aggregate path."""
        state_ind = np.array([0, 1, 2, 2])
        continuous_transition = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        discrete_transition = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        causal_posterior = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        predictive_posterior = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2],
            ]
        )
        acausal_posterior = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        causal_state = np.column_stack(
            (causal_posterior[:, :2], causal_posterior[:, 2:].sum(axis=1))
        )
        predictive_state = np.column_stack(
            (
                predictive_posterior[:, :2],
                predictive_posterior[:, 2:].sum(axis=1),
            )
        )
        acausal_state = np.column_stack(
            (acausal_posterior[:, :2], acausal_posterior[:, 2:].sum(axis=1))
        )

        new_trans, _ = _estimate_discrete_transition(
            causal_state_probabilities=causal_state,
            predictive_state_probabilities=predictive_state,
            acausal_state_probabilities=acausal_state,
            discrete_transition=discrete_transition,
            discrete_transition_coefficients=None,
            discrete_transition_design_matrix=None,
            transition_concentration=1.0,
            transition_stickiness=0.0,
            transition_regularization=1e-5,
            causal_posterior=causal_posterior,
            predictive_posterior=predictive_posterior,
            acausal_posterior=acausal_posterior,
            continuous_transition=continuous_transition,
            state_ind=state_ind,
        )

        assert new_trans[1, 2] > new_trans[0, 2]

    def test_non_stationary_diagonal(self, posterior_data, design_matrix_data):
        """Test with non-stationary diagonal transition type."""
        # Arrange
        post = posterior_data
        dm = design_matrix_data

        # Create time-varying transition matrix
        time_varying_trans = np.tile(post["transition_matrix"], (post["n_time"], 1, 1))

        # Act - returns (transition_matrix, coefficients)
        new_trans, new_coeffs = _estimate_discrete_transition(
            causal_state_probabilities=post["causal_posterior"],
            predictive_state_probabilities=post["predictive_distribution"],
            acausal_state_probabilities=post["acausal_posterior"],
            discrete_transition=time_varying_trans,
            discrete_transition_coefficients=dm["transition_coefficients"],
            discrete_transition_design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_concentration=1.0,
            transition_stickiness=1.0,  # Diagonal stickiness
            transition_regularization=1e-5,
        )

        # Assert
        assert new_trans.shape == (post["n_time"], post["n_states"], post["n_states"])
        assert new_coeffs.shape == dm["transition_coefficients"].shape
        for t in range(min(5, new_trans.shape[0])):
            assert_stochastic_matrix(new_trans[t])

    def test_nonstationary_uses_expanded_responses_when_provided(self):
        """Expanded nonstationary inputs should reach the exact response path."""
        state_ind = np.array([0, 1, 2, 2])
        continuous_transition = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        discrete_transition = np.tile(
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            (2, 1, 1),
        )
        causal_posterior = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        predictive_posterior = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2],
            ]
        )
        acausal_posterior = np.array(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        causal_state = np.column_stack(
            (causal_posterior[:, :2], causal_posterior[:, 2:].sum(axis=1))
        )
        predictive_state = np.column_stack(
            (
                predictive_posterior[:, :2],
                predictive_posterior[:, 2:].sum(axis=1),
            )
        )
        acausal_state = np.column_stack(
            (acausal_posterior[:, :2], acausal_posterior[:, 2:].sum(axis=1))
        )
        transition_coefficients = np.zeros((1, 3, 2))
        design_matrix = np.ones((2, 1))
        captured = {}

        def fake_from_responses(
            transition_coefficients,
            design_matrix,
            response,
            **kwargs,
        ):
            captured["response"] = response
            return transition_coefficients, np.tile(np.eye(3), (2, 1, 1))

        with patch(
            "non_local_detector.discrete_state_transitions."
            "estimate_non_stationary_state_transition_from_responses",
            side_effect=fake_from_responses,
        ):
            new_trans, new_coeffs = _estimate_discrete_transition(
                causal_state_probabilities=causal_state,
                predictive_state_probabilities=predictive_state,
                acausal_state_probabilities=acausal_state,
                discrete_transition=discrete_transition,
                discrete_transition_coefficients=transition_coefficients,
                discrete_transition_design_matrix=design_matrix,
                transition_concentration=1.0,
                transition_stickiness=0.0,
                transition_regularization=1e-5,
                causal_posterior=causal_posterior,
                predictive_posterior=predictive_posterior,
                acausal_posterior=acausal_posterior,
                continuous_transition=continuous_transition,
                state_ind=state_ind,
            )

        assert new_trans.shape == (2, 3, 3)
        assert new_coeffs.shape == transition_coefficients.shape
        assert captured["response"][0, 1, 2] > captured["response"][0, 0, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
