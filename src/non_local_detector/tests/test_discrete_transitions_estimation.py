"""High-priority tests for discrete state transition estimation functions.

Tests the core EM algorithm functions:
- estimate_non_stationary_state_transition_from_responses
- estimate_stationary_state_transition_from_counts
- _discrete_transition_statistics and _discrete_transition_from_statistics
"""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax, xlogy

from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    _discrete_transition_from_statistics,
    _discrete_transition_statistics,
    _state_aggregation_matrix,
    _stationary_counts_block,
    _transition_log_prior,
    dirichlet_gradient,
    dirichlet_hessian,
    dirichlet_neg_log_likelihood,
    estimate_discrete_transition_counts_from_expanded_posteriors,
    estimate_discrete_transition_counts_from_factorized_posteriors,
    estimate_discrete_transition_responses_from_expanded_posteriors,
    estimate_discrete_transition_responses_from_factorized_posteriors,
    estimate_non_stationary_state_transition_from_responses,
    estimate_stationary_state_transition_from_counts,
    get_transition_prior,
    jax_centered_log_softmax_forward,
)
from non_local_detector.environment import Environment
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.models.base import _DetectorBase
from non_local_detector.observation_models import ObservationModel
from non_local_detector.tests.conftest import assert_stochastic_matrix
from non_local_detector.time_edges import time_edges_from_centers


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

    def compute_log_likelihood(self, time_edges, *args, is_missing=None):
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
            where=predictive_posterior[t + 1] != 0.0,
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
        time_edges=time_edges_from_centers(np.arange(n_time, dtype=float)),
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


def _nonstationary_transition_parameters(
    n_time: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return covariate, coefficients, and transitions for simulation tests."""
    n_states = 3
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

    return covariate, true_coefficients, true_transition


def _estimate_simulated_nonstationary_discrete_transition(
    seed: int,
    n_time: int,
    off_target_log_likelihood: float,
    max_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate and estimate a covariate-dependent expanded HMM."""
    rng = np.random.default_rng(seed)
    n_states = 3
    n_bins = 2
    covariate, _, true_transition = _nonstationary_transition_parameters(n_time)
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
        off_target_log_likelihood=off_target_log_likelihood,
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
        time_edges=time_edges_from_centers(np.arange(n_time, dtype=float)),
        estimate_initial_conditions=False,
        estimate_discrete_transition=True,
        estimate_encoding_model=False,
        max_iter=max_iter,
        tolerance=0.0,
    )

    return (
        detector.discrete_state_transitions_,
        true_transition,
        covariate,
        results.attrs["marginal_log_likelihoods"],
    )


def _low_high_covariate_transition_error(
    learned_transition: np.ndarray,
    true_transition: np.ndarray,
    covariate: np.ndarray,
) -> float:
    """Return max low/high covariate transition error for identifiable rows."""
    low_covariate = covariate[:-1] < -0.75
    high_covariate = covariate[:-1] > 0.75
    learned_low = learned_transition[:-1][low_covariate].mean(axis=0)
    learned_high = learned_transition[:-1][high_covariate].mean(axis=0)
    true_low = true_transition[:-1][low_covariate].mean(axis=0)
    true_high = true_transition[:-1][high_covariate].mean(axis=0)
    return max(
        np.max(np.abs(learned_low[:2] - true_low[:2])),
        np.max(np.abs(learned_high[:2] - true_high[:2])),
    )


def _joint_sum_from_post(post):
    """Return stationary joint counts via the exact singleton-bin helper."""
    n_states = post["n_states"]
    return estimate_discrete_transition_counts_from_expanded_posteriors(
        causal_posterior=post["causal_posterior"],
        predictive_posterior=post["predictive_distribution"],
        acausal_posterior=post["acausal_posterior"],
        transition_matrix=post["transition_matrix"],
        state_ind=np.arange(n_states),
    )


def _response_from_post(post):
    """Return per-time response tensor via the exact singleton-bin helper."""
    n_states = post["n_states"]
    return estimate_discrete_transition_responses_from_expanded_posteriors(
        causal_posterior=post["causal_posterior"],
        predictive_posterior=post["predictive_distribution"],
        acausal_posterior=post["acausal_posterior"],
        transition_matrix=post["transition_matrix"],
        state_ind=np.arange(n_states),
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

    @staticmethod
    def _stationary_posteriors(rng, n_time, n_bins, state_ind, n_states):
        causal = rng.random((n_time, n_bins))
        causal /= causal.sum(axis=1, keepdims=True)
        continuous_transition = rng.random((n_bins, n_bins)) ** 4
        # An unreachable target column gives exact zero predictive probability.
        continuous_transition[:, -1] = 0.0
        continuous_transition /= continuous_transition.sum(axis=1, keepdims=True)
        discrete_transition = rng.random((n_states, n_states))
        discrete_transition /= discrete_transition.sum(axis=1, keepdims=True)
        full_transition = (
            continuous_transition * discrete_transition[np.ix_(state_ind, state_ind)]
        )
        predictive = causal @ full_transition
        acausal = predictive * rng.uniform(0.8, 1.2, size=predictive.shape)
        acausal /= acausal.sum(axis=1, keepdims=True)
        return (
            causal,
            predictive,
            acausal,
            continuous_transition,
            discrete_transition,
            full_transition,
        )

    @pytest.mark.parametrize("is_factorized", [True, False])
    def test_stationary_counts_do_not_loop_over_time(self, is_factorized):
        """Stationary counts are contractions over rows, not a scan per row.

        A 709k-row scan spent 11 s per EM iteration on an A100 for 514 state
        bins, bound by per-step loop overhead. The contraction may loop over
        partial products of many rows each.
        """
        import non_local_detector.discrete_state_transitions as transitions

        rng = np.random.default_rng(31)
        n_time = 601
        state_ind = np.array([0, 0, 1, 2, 2])
        causal, predictive, acausal, continuous, discrete, full = (
            self._stationary_posteriors(rng, n_time, 5, state_ind, 3)
        )
        transition = (
            continuous * discrete[np.ix_(state_ind, state_ind)]
            if is_factorized
            else full
        )
        block = jax.make_jaxpr(
            lambda *posteriors: _stationary_counts_block(
                *posteriors, jnp.asarray(transition), jnp.asarray(state_ind), 3
            )
        )(causal[:-1], predictive[1:], acausal[1:])

        def loops(jaxpr):
            for eqn in jaxpr.eqns:
                if eqn.primitive.name in ("scan", "while"):
                    yield eqn
                for value in eqn.params.values():
                    for sub in value if isinstance(value, (tuple, list)) else (value,):
                        inner = getattr(sub, "jaxpr", sub)
                        if hasattr(inner, "eqns"):
                            yield from loops(inner)

        found = list(loops(block.jaxpr))
        assert all(eqn.primitive.name == "scan" for eqn in found), found
        n_partial = -(-(n_time - 1) // transitions._CONTRACTION_SUB_ROWS)
        assert all(eqn.params["length"] <= n_partial for eqn in found)

    @pytest.mark.parametrize("factorized", [True, False])
    def test_stationary_counts_stream_bounded_row_blocks(self, factorized, monkeypatch):
        """Rows reach the device in byte-bounded blocks; counts are unchanged.

        Full posteriors of a 23.6 min recording on a 2-D grid (709k rows x
        3,298 bins) did not fit on an 80 GB A100 as one contraction.
        """
        import non_local_detector.discrete_state_transitions as transitions

        rng = np.random.default_rng(33)
        n_time, n_bins, n_states = 2_000, 30, 3
        state_ind = np.repeat(np.arange(n_states), n_bins // n_states)
        posteriors = [
            np.asarray(array, np.float32)
            for array in self._stationary_posteriors(
                rng, n_time, n_bins, state_ind, n_states
            )
        ]

        def counts():
            if factorized:
                return estimate_discrete_transition_counts_from_factorized_posteriors(
                    *posteriors[:3], posteriors[3], posteriors[4], state_ind
                )
            return estimate_discrete_transition_counts_from_expanded_posteriors(
                *posteriors[:3], posteriors[5], state_ind
            )

        whole = counts()
        monkeypatch.setattr(transitions, "_PAIR_SUM_BLOCK_BYTES", 300 * n_bins * 4)
        original = transitions._stationary_counts_block
        block_rows = []

        def record(causal, *args):
            block_rows.append(causal.shape[0])
            return original(causal, *args)

        monkeypatch.setattr(transitions, "_stationary_counts_block", record)
        blocked = counts()

        assert max(block_rows) == 300 and sum(block_rows) == n_time - 1
        np.testing.assert_allclose(blocked, whole, rtol=1e-6)

    @pytest.mark.parametrize("factorized", [True, False])
    def test_stationary_counts_over_many_rows_match_float64(self, factorized):
        """A long float32 sum over rows stays close to the float64 counts."""
        rng = np.random.default_rng(32)
        n_time, n_bins, n_states = 20_000, 30, 3
        state_ind = np.repeat(np.arange(n_states), n_bins // n_states)
        posteriors = self._stationary_posteriors(
            rng, n_time, n_bins, state_ind, n_states
        )
        causal, predictive, acausal, continuous, discrete, full = posteriors
        assert np.any(predictive == 0.0)
        expected = _expanded_responses_numpy_reference(
            causal, predictive, acausal, full, state_ind
        ).sum(axis=0)
        as32 = [np.asarray(array, np.float32) for array in posteriors]
        if factorized:
            counts = estimate_discrete_transition_counts_from_factorized_posteriors(
                *as32[:3], as32[3], as32[4], state_ind
            )
        else:
            counts = estimate_discrete_transition_counts_from_expanded_posteriors(
                *as32[:3], as32[5], state_ind
            )

        np.testing.assert_allclose(counts, expected, rtol=1e-5)

    @pytest.mark.parametrize("factorized", [True, False])
    def test_stationary_counts_do_not_overflow_with_rare_transitions(self, factorized):
        """Ratios near 1 / P(transition) must be weighted before they are summed.

        A 1e-34 transition taken at every other row of 100,001 gives smoother
        ratios near 1e34; their float32 sum over rows overflows although the
        weighted count is 50,000.
        """
        from non_local_detector.core import filter, smoother

        n_time = 100_001
        transition = np.array([[1 - 1e-34, 1e-34], [1.0, 0.0]], dtype=np.float32)
        log_likelihoods = np.zeros((n_time, 2), np.float32)
        log_likelihoods[0::2, 1] = -1e4
        log_likelihoods[1::2, 0] = -1e4
        (_, _), (causal, predictive) = filter(
            np.array([1.0, 0.0], np.float32), transition, log_likelihoods
        )
        acausal = smoother(transition, causal)
        state_ind = np.array([0, 1])
        if factorized:
            counts = estimate_discrete_transition_counts_from_factorized_posteriors(
                causal,
                predictive,
                acausal,
                np.ones((2, 2), dtype=np.float32),
                transition,
                state_ind,
            )
        else:
            counts = estimate_discrete_transition_counts_from_expanded_posteriors(
                causal, predictive, acausal, transition, state_ind
            )
        np.testing.assert_allclose(
            counts, [[0.0, 50_000.0], [50_000.0, 0.0]], rtol=1e-5, atol=1e-3
        )

    def test_rare_transition_does_not_erase_ordinary_counts(self):
        """One extreme ratio in a column must not scale its other ratios away.

        A 2e-38 transition gives a ratio near 2.5e37 in column 1. Dividing that
        column by its maximum put an ordinary ratio of 1 near 4e-38, and its
        0.01 expected count underflowed float32, turning a 0.01 transition
        into a deterministic one.
        """
        transition = np.array([[1 - 2e-38, 2e-38], [0.99, 0.01]], np.float32)
        causal = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01]], np.float32)
        predictive = np.vstack(
            [[0.5, 0.5], causal[0] @ transition, causal[1] @ transition]
        ).astype(np.float32)
        acausal = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01]], np.float32)
        state_ind = np.array([0, 1])
        expected = _expanded_responses_numpy_reference(
            causal.astype(float),
            predictive.astype(float),
            acausal.astype(float),
            transition.astype(float),
            state_ind,
        ).sum(axis=0)
        counts = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal, predictive, acausal, transition, state_ind
        )
        assert expected[1, 1] == pytest.approx(0.01, rel=1e-5)
        np.testing.assert_allclose(counts, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_statistics_are_rejected(self, bad):
        """A NaN or infinite count would silently give NaN transitions."""
        from non_local_detector.discrete_state_transitions import (
            _discrete_transition_from_statistics,
        )

        counts = np.array([[5.0, 1.0], [2.0, bad]])
        with pytest.raises(ValueError, match=r"\(1, 1\)"):
            _discrete_transition_from_statistics(counts, None, None, 1.0, 0.0, 0.0)

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

    def test_tiny_nonzero_predictive_bin_is_not_dropped(self):
        """A tiny but nonzero predictive bin must keep its transition count.

        Regression guard for `_safe_ratio_jax`: an `isclose(., 0)` threshold
        would zero the acausal/predictive ratio for predictive mass below
        ~1e-8, silently dropping a real transition when the smoother strongly
        revises a low-probability prediction. Only exact zeros should be
        treated as unreachable.
        """
        eps = 1e-10
        state_ind = np.array([0, 1])
        transition_matrix = np.array([[1.0 - eps, eps], [0.5, 0.5]])
        causal = np.array([[1.0, 0.0], [0.0, 1.0]])
        # predictive[1] = causal[0] @ transition_matrix; the second bin has mass
        # eps << 1e-8 but is genuinely reachable.
        predictive = np.array([[1.0, 0.0], [1.0 - eps, eps]])
        # Smoother says the second bin was actually visited: ratio = 1 / eps.
        acausal = np.array([[1.0, 0.0], [0.0, 1.0]])

        counts = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal, predictive, acausal, transition_matrix, state_ind
        )

        # The eps in the transition cancels the 1/eps ratio, leaving a full
        # unit transition 0 -> 1. An isclose threshold would zero this to 0.
        np.testing.assert_allclose(
            counts, np.array([[0.0, 1.0], [0.0, 0.0]]), atol=1e-6
        )

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
            time_edges=time_edges_from_centers(np.array([0.0, 1.0])),
            estimate_initial_conditions=False,
            estimate_discrete_transition=True,
            estimate_encoding_model=False,
            max_iter=1,
        )

        learned_transition = detector.discrete_state_transitions_

        assert learned_transition[1, 2] > learned_transition[0, 2]
        assert learned_transition[1, 2] > 0.99
        # Source 0 must be actively *not* credited for target bin 2, not merely
        # credited less than source 1. This magnitude guard fails for a
        # source-blind kernel that splits transition credit evenly across
        # sources, whereas the inequality above can hold weakly even under
        # partial credit bleed.
        assert learned_transition[0, 2] < 0.5
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

        # max_iter=5 EM iterations + 1 final E-step after the loop
        assert len(marginal_log_likelihoods) == 6
        # Tolerance accommodates float32 noise on the converged tail
        # (LL magnitude ~700, float32 epsilon ~1e-7 → ~1e-4 noise).
        assert np.all(np.diff(marginal_log_likelihoods) >= -1e-3)

    def test_nonstationary_detector_recovers_covariate_direction(self):
        """Exact responses should learn a simple covariate-dependent transition."""
        n_time = 1_200
        learned_transition, true_transition, covariate, marginal_log_likelihoods = (
            _estimate_simulated_nonstationary_discrete_transition(
                seed=13,
                n_time=n_time,
                off_target_log_likelihood=-12.0,
            )
        )
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
        # 1 EM iteration + 1 final E-step after the loop
        assert len(marginal_log_likelihoods) == 2

    def test_nonstationary_recovery_improves_with_emission_strength(self):
        """Nonstationary recovery should improve with more informative emissions."""
        weak_learned, weak_true, weak_covariate, _ = (
            _estimate_simulated_nonstationary_discrete_transition(
                seed=17,
                n_time=900,
                off_target_log_likelihood=-3.0,
            )
        )
        strong_learned, strong_true, strong_covariate, _ = (
            _estimate_simulated_nonstationary_discrete_transition(
                seed=17,
                n_time=900,
                off_target_log_likelihood=-12.0,
            )
        )

        weak_error = _low_high_covariate_transition_error(
            weak_learned,
            weak_true,
            weak_covariate,
        )
        strong_error = _low_high_covariate_transition_error(
            strong_learned,
            strong_true,
            strong_covariate,
        )

        assert strong_error < weak_error
        assert strong_error < 0.10

    def test_nonstationary_strong_emissions_do_not_freeze_rows(self):
        """Strong emissions should still recover rows that are nearly separated."""
        learned_transition, true_transition, covariate, _ = (
            _estimate_simulated_nonstationary_discrete_transition(
                seed=14,
                n_time=900,
                off_target_log_likelihood=-12.0,
            )
        )
        error = _low_high_covariate_transition_error(
            learned_transition,
            true_transition,
            covariate,
        )

        assert error < 0.10


def _noisy_expanded_hmm_observations(
    seed: int, n_time: int, emission_sharpness: float
) -> np.ndarray:
    """Log likelihoods of categorical observations that only partly reveal
    the expanded state, shape (n_time, n_states * n_bins)."""
    initial_conditions, discrete_transition, blocks = _expanded_hmm_parameters()
    n_expanded, n_symbols = 6, 4
    emissions = softmax(
        emission_sharpness
        * np.random.default_rng(123).normal(size=(n_expanded, n_symbols)),
        axis=1,
    )
    rng = np.random.default_rng(seed)
    states, bins = _simulate_expanded_hmm(
        rng, initial_conditions, discrete_transition, blocks, n_time
    )
    symbols = np.array(
        [rng.choice(n_symbols, p=emissions[e]) for e in states * 2 + bins]
    )
    return np.log(emissions[:, symbols].T)


def _expanded_hmm_marginal_log_likelihood(
    discrete_transition: np.ndarray, log_likelihoods: np.ndarray
) -> float:
    """Scaled forward algorithm in float64, independent of the package."""
    initial_conditions, _, blocks = _expanded_hmm_parameters()
    expanded_transition = np.block(
        [[discrete_transition[i, j] * blocks[i][j] for j in range(3)] for i in range(3)]
    )
    scale = log_likelihoods.max(axis=1)
    likelihoods = np.exp(log_likelihoods - scale[:, None])
    alpha = np.repeat(initial_conditions, 2) / 2 * likelihoods[0]
    log_marginal = scale.sum()
    for t in range(1, len(likelihoods)):
        norm = alpha.sum()
        log_marginal += np.log(norm)
        alpha = (alpha / norm) @ expanded_transition * likelihoods[t]
    return float(log_marginal + np.log(alpha.sum()))


def _map_objective(
    discrete_transition: np.ndarray,
    log_likelihoods: np.ndarray,
    pseudo_counts: np.ndarray,
    free_rows: np.ndarray,
) -> float:
    """Marginal log-likelihood plus the Dirichlet log prior of the free rows."""
    log_prior = np.sum(xlogy(pseudo_counts[free_rows], discrete_transition[free_rows]))
    return _expanded_hmm_marginal_log_likelihood(
        discrete_transition, log_likelihoods
    ) + float(log_prior)


def _direct_transition_map(
    log_likelihoods: np.ndarray,
    pseudo_counts: np.ndarray,
    frozen_row: int | None = None,
    frozen_values: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Maximize ``_map_objective`` over the free rows' softmax logits, with
    ``frozen_row`` held at ``frozen_values``. Zero pseudo-counts and no frozen
    row give the maximum-likelihood estimate."""
    free_rows = np.arange(3) != frozen_row
    n_free = int(free_rows.sum())

    def to_transition(logits):
        transition = np.empty((3, 3))
        transition[free_rows] = softmax(
            np.column_stack((logits.reshape(n_free, 2), np.zeros(n_free))), axis=1
        )
        if frozen_row is not None:
            transition[frozen_row] = frozen_values
        return transition

    def negative_objective(logits):
        return -_map_objective(
            to_transition(logits), log_likelihoods, pseudo_counts, free_rows
        )

    result = minimize(
        negative_objective, np.zeros(2 * n_free), method="BFGS", options={"gtol": 1e-7}
    )
    return to_transition(result.x), -float(result.fun)


@pytest.fixture(scope="module")
def noisy_hmm():
    """Observations that only partly reveal the expanded state, and the
    maximum-likelihood transitions found by direct maximization."""
    log_likelihoods = _noisy_expanded_hmm_observations(
        seed=1, n_time=1_000, emission_sharpness=2.0
    )
    mle, mle_log_likelihood = _direct_transition_map(log_likelihoods, np.zeros((3, 3)))
    return SimpleNamespace(
        log_likelihoods=log_likelihoods,
        mle=mle,
        mle_log_likelihood=mle_log_likelihood,
    )


def _em_estimate(
    log_likelihoods: np.ndarray, discrete_transition: np.ndarray, **attributes
):
    """Run EM on the discrete transitions alone, from ``discrete_transition``,
    with ``attributes`` set on the detector before it is fit."""
    initial_conditions, _, blocks = _expanded_hmm_parameters()
    detector = _FixedLikelihoodDetector(
        log_likelihoods,
        discrete_initial_conditions=initial_conditions,
        discrete_transition=discrete_transition,
        continuous_transition_blocks=blocks,
    )
    for name, value in attributes.items():
        setattr(detector, name, value)
    detector._fit(position=np.array([[0.25], [1.25]]))
    results = detector.estimate_parameters(
        time_edges=time_edges_from_centers(
            np.arange(len(log_likelihoods), dtype=float)
        ),
        estimate_initial_conditions=False,
        estimate_discrete_transition=True,
        estimate_encoding_model=False,
        max_iter=2_000,
        tolerance=1e-9,
    )
    assert detector.converged_
    assert detector.em_monotonicity_violations_ == []
    return np.asarray(detector.discrete_state_transitions_), results


@pytest.mark.slow
@pytest.mark.integration
class TestEMConvergesToMaximumLikelihood:
    """Iterated EM reaches the maximum-likelihood transition matrix.

    Noisy emissions leave the states hidden, so the answer depends on EM
    iterating to its fixed point, not on one M-step over near-observed states.
    The maximum is found independently by maximizing a float64 forward
    algorithm's marginal log-likelihood.
    """

    def test_em_fixed_point_is_the_mle_from_any_start(self, noisy_hmm):
        estimates = []
        for start in (
            np.full((3, 3), 1.0 / 3.0),
            np.full((3, 3), 0.05) + 0.85 * np.eye(3),
        ):
            estimate, results = _em_estimate(noisy_hmm.log_likelihoods, start)
            estimate_log_likelihood = _expanded_hmm_marginal_log_likelihood(
                estimate, noisy_hmm.log_likelihoods
            )

            # The package's final E-step evaluates the estimate as the
            # independent forward algorithm does (float32 likelihoods).
            np.testing.assert_allclose(
                results.attrs["marginal_log_likelihoods"][-1],
                estimate_log_likelihood,
                atol=1e-3,
            )
            # Within EM's slow final approach of the maximum (measured
            # 1.6e-3 and 2.4e-3 nats), never above it.
            gap = noisy_hmm.mle_log_likelihood - estimate_log_likelihood
            assert -1e-4 < gap < 1e-2
            np.testing.assert_allclose(estimate, noisy_hmm.mle, atol=0.02)
            estimates.append(estimate)

        np.testing.assert_allclose(estimates[0], estimates[1], atol=0.02)


@pytest.mark.slow
@pytest.mark.integration
class TestEMConvergesToMapEstimate:
    """With a fixed Dirichlet prior and a frozen row, iterated EM reaches the
    maximum of marginal log-likelihood + log prior over the free rows.

    The maximum is found independently, by maximizing a float64 forward
    algorithm's marginal log-likelihood plus the prior directly, so this
    checks the whole loop rather than each M-step or the penalty formula.
    """

    def test_em_fixed_point_is_the_map_estimate_from_any_start(self, noisy_hmm):
        log_likelihoods = noisy_hmm.log_likelihoods
        concentration, stickiness = 2.0, np.array([40.0, 20.0, 0.0])
        pseudo_counts = (
            get_transition_prior(concentration, stickiness, n_states=3) - 1.0
        )
        frozen_row, frozen_values = 2, np.array([0.16, 0.14, 0.70])
        free_rows = np.arange(3) != frozen_row
        map_estimate, map_objective = _direct_transition_map(
            log_likelihoods, pseudo_counts, frozen_row, frozen_values
        )
        # The prior moves the free rows far from the maximum-likelihood
        # estimate (0.24 measured), so matching the MAP estimate is not
        # matching the MLE.
        assert np.abs(map_estimate - noisy_hmm.mle)[free_rows].max() > 0.1

        estimates = []
        for start in (
            np.full((2, 3), 1.0 / 3.0),
            np.full((2, 3), 0.05) + 0.85 * np.eye(2, 3),
        ):
            estimate, results = _em_estimate(
                log_likelihoods,
                np.vstack((start, frozen_values)),
                discrete_transition_concentration=concentration,
                discrete_transition_stickiness=stickiness,
                frozen_discrete_transition_rows=[frozen_row],
            )
            estimate_objective = _map_objective(
                estimate, log_likelihoods, pseudo_counts, free_rows
            )

            np.testing.assert_array_equal(estimate[frozen_row], frozen_values)
            # EM's final objective is the one maximized here (float32
            # likelihoods; 1.9e-6 and 1.3e-5 measured).
            np.testing.assert_allclose(
                results.attrs["em_objectives"][-1], estimate_objective, atol=1e-3
            )
            # Within EM's slow final approach (8.2e-5 and 4.7e-5 measured).
            assert -1e-4 < map_objective - estimate_objective < 1e-3
            # 6.6e-4 and 5.0e-4 measured.
            np.testing.assert_allclose(estimate, map_estimate, atol=5e-3)
            estimates.append(estimate)

        np.testing.assert_allclose(estimates[0], estimates[1], atol=5e-3)


@pytest.mark.unit
class TestEstimateNonStationaryStateTransition:
    """Test non-stationary state transition estimation (EM algorithm)."""

    def test_returns_correct_shapes(self, posterior_data, design_matrix_data):
        """Verify coefficient and transition matrix shapes are correct."""
        post = posterior_data
        dm = design_matrix_data
        response = _response_from_post(post)

        coeffs, trans_matrix = estimate_non_stationary_state_transition_from_responses(
            transition_coefficients=dm["transition_coefficients"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            response=response,
            concentration=1.0,
            stickiness=0.0,
            transition_regularization=1e-5,
            maxiter=10,
        )

        n_coeffs, n_states = dm["n_coefficients"], post["n_states"]
        assert coeffs.shape == (n_coeffs, n_states, n_states - 1)
        assert trans_matrix.shape == (post["n_time"], n_states, n_states)

    def test_produces_valid_probabilities(self, posterior_data, design_matrix_data):
        """Check all transition matrices are valid stochastic matrices."""
        post = posterior_data
        dm = design_matrix_data
        response = _response_from_post(post)

        _, trans_matrix = estimate_non_stationary_state_transition_from_responses(
            transition_coefficients=dm["transition_coefficients"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            response=response,
            concentration=1.0,
            stickiness=0.0,
            transition_regularization=1e-5,
            maxiter=10,
        )

        for t in range(trans_matrix.shape[0]):
            assert_stochastic_matrix(trans_matrix[t])

    def test_concentration_below_one_raises(self, posterior_data, design_matrix_data):
        """The nonstationary MAP update rejects negative pseudo-counts."""
        post = posterior_data
        dm = design_matrix_data
        response = _response_from_post(post)

        with pytest.raises(ValueError, match="prior parameters >= 1.0"):
            estimate_non_stationary_state_transition_from_responses(
                transition_coefficients=dm["transition_coefficients"],
                design_matrix=dm["design_matrix"][: post["n_time"]],
                response=response,
                concentration=0.5,
                stickiness=0.0,
                transition_regularization=1e-5,
                maxiter=10,
            )

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

    @staticmethod
    def _fit(post, dm, concentration, stickiness):
        """Fit the fixture's responses with the given Dirichlet prior."""
        _, transition = estimate_non_stationary_state_transition_from_responses(
            transition_coefficients=dm["transition_coefficients"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            response=_response_from_post(post),
            concentration=concentration,
            stickiness=stickiness,
            transition_regularization=1e-5,
            maxiter=10,
        )
        return transition

    def test_with_different_concentrations(self, posterior_data, design_matrix_data):
        """A larger concentration pulls every row toward uniform relative to
        the same fit without a prior (concentration=1, stickiness=0)."""
        post, dm = posterior_data, design_matrix_data
        n_states = post["n_states"]

        trans_none = self._fit(post, dm, concentration=1.0, stickiness=0.0)
        trans_strong = self._fit(post, dm, concentration=10.0, stickiness=0.0)

        for t in range(trans_strong.shape[0]):
            assert_stochastic_matrix(trans_strong[t])
        # Mean distance from uniform of each row, averaged over time.
        distance_none = np.abs(trans_none - 1.0 / n_states).mean(axis=(0, 2))
        distance_strong = np.abs(trans_strong - 1.0 / n_states).mean(axis=(0, 2))
        assert np.all(distance_strong < distance_none), (
            distance_strong,
            distance_none,
        )

    def test_with_diagonal_stickiness(self, posterior_data, design_matrix_data):
        """Stickiness raises every state's time-averaged self-transition
        probability above the same fit without a prior. The fixture's data
        are already sticky, so comparing diagonal to off-diagonal would not
        show the prior's effect."""
        post, dm = posterior_data, design_matrix_data
        is_diagonal = np.eye(post["n_states"], dtype=bool)

        trans_none = self._fit(post, dm, concentration=1.0, stickiness=0.0)
        trans_sticky = self._fit(post, dm, concentration=1.0, stickiness=1.0)

        for t in range(trans_sticky.shape[0]):
            assert_stochastic_matrix(trans_sticky[t])
        stay_none = trans_none[:, is_diagonal].mean(axis=0)
        stay_sticky = trans_sticky[:, is_diagonal].mean(axis=0)
        assert np.all(stay_sticky > stay_none), (stay_sticky, stay_none)

    def test_keeps_previous_coefficients_when_optimizer_result_raises_the_loss(
        self, posterior_data, design_matrix_data
    ):
        """An optimizer result that raises the M-step loss is discarded and
        the previous coefficients kept, even when the optimizer reports
        success: the choice depends on the loss, not on ``success``."""
        post = posterior_data
        dm = design_matrix_data
        n_coeffs = dm["n_coefficients"]
        n_states = post["n_states"]

        rng = np.random.default_rng(99)
        initial_coeffs = rng.standard_normal((n_coeffs, n_states, n_states - 1)) * 0.1
        response = _response_from_post(post)

        class FakeResult:
            success = True
            message = "mock result far from the optimum"
            x = np.full(n_coeffs * (n_states - 1), 999.0)

        with patch(
            "non_local_detector.discrete_state_transitions.minimize",
            return_value=FakeResult(),
        ):
            coeffs, trans_matrix = (
                estimate_non_stationary_state_transition_from_responses(
                    transition_coefficients=initial_coeffs,
                    design_matrix=dm["design_matrix"][: post["n_time"]],
                    response=response,
                    maxiter=10,
                )
            )

        np.testing.assert_array_equal(coeffs, initial_coeffs)
        for t in range(trans_matrix.shape[0]):
            assert_stochastic_matrix(trans_matrix[t])

    def test_reaches_the_float64_minimum_of_its_objective(self):
        """The default stickiness's 1e5 pseudo-counts, against about 1e4
        expected transitions per row, push two rows' stay probabilities to
        about 0.9995. Each row's M-step still never raises its objective, and
        ends at the minimum an independent float64 optimizer of the same
        objective finds."""
        rng = np.random.default_rng(0)
        n_time, n_states = 20_000, 4
        speed = np.abs(np.cumsum(rng.normal(0, 0.5, n_time)))
        speed = 50 * speed / speed.max()
        design_matrix = np.column_stack((np.ones(n_time), speed))
        # Expected transitions: each state's occupancy times a sticky row.
        leave = 1e-3 * (1 + speed[:-1, None, None] / 50)
        response = rng.uniform(0.1, 0.9, (n_time - 1, n_states, 1)) * np.where(
            np.eye(n_states, dtype=bool), 1 - 3 * leave, leave
        )
        stickiness = np.array([1e5, 1e5, 0.0, 0.0])
        alpha = get_transition_prior(1.0, stickiness, n_states)
        start = np.zeros((2, n_states, n_states - 1))
        for row in range(n_states - 1):
            start[0, row, row] = 5.0
        start[0, n_states - 1] = -5.0

        coefficients, transition = (
            estimate_non_stationary_state_transition_from_responses(
                start,
                design_matrix,
                response,
                concentration=1.0,
                stickiness=stickiness,
                transition_regularization=1e-5,
            )
        )

        def objective(flat_coefficients, row):
            """The row's M-step loss in float64: summed over the transitions,
            with the Dirichlet prior counted once."""
            coefficients_row = flat_coefficients.reshape(2, -1)
            # The last target state is the reference category.
            linear_predictor = np.column_stack(
                (design_matrix[:-1] @ coefficients_row, np.zeros(n_time - 1))
            )
            log_probs = linear_predictor - logsumexp(
                linear_predictor, axis=1, keepdims=True
            )
            weights = response[:, row] + (alpha[row] - 1.0) / (n_time - 1)
            penalty = 1e-5 * np.sum(coefficients_row[1:] ** 2)
            return -np.sum(weights * log_probs) + penalty

        for row in range(n_states):
            before = objective(start[:, row].ravel(), row)
            after = objective(coefficients[:, row].ravel(), row)
            # Started from the estimate, so it can only improve on it.
            best = minimize(
                objective,
                coefficients[:, row].ravel(),
                args=(row,),
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000},
            )
            assert after <= before, row
            assert after - best.fun < 1e-3, (row, after - best.fun)
        np.testing.assert_allclose(transition.sum(axis=-1), 1.0, atol=1e-12)

    @pytest.mark.parametrize("method", ["trust-exact", "L-BFGS-B"])
    def test_single_state_has_nothing_to_fit(self, method):
        """One state has no free transition coefficients, so the M-step keeps
        them and the only transition stays at probability 1."""
        n_time = 20
        design_matrix = np.column_stack((np.ones(n_time), np.linspace(0, 1, n_time)))
        coefficients, transition = (
            estimate_non_stationary_state_transition_from_responses(
                np.zeros((2, 1, 0)),
                design_matrix,
                np.ones((n_time - 1, 1, 1)),
                stickiness=5.0,
                optimization_method=method,
            )
        )
        assert coefficients.shape == (2, 1, 0)
        np.testing.assert_array_equal(transition, np.ones((n_time, 1, 1)))

    def test_newton_methods_reach_the_same_estimate(self):
        """L-BFGS-B and the default trust-exact, a Newton method that uses the
        Hessian, reach the same estimate."""
        rng = np.random.default_rng(6)
        n_time, n_states = 500, 3
        design_matrix = np.column_stack(
            (np.ones(n_time), np.sin(np.arange(n_time) / 50))
        )
        response = rng.dirichlet(np.ones(n_states) * 2, size=(n_time - 1, n_states))
        start = np.zeros((2, n_states, n_states - 1))
        estimates = [
            estimate_non_stationary_state_transition_from_responses(
                start,
                design_matrix,
                response,
                concentration=1.5,
                stickiness=5.0,
                transition_regularization=1e-3,
                optimization_method=method,
                maxiter=500,
            )[0]
            for method in ("L-BFGS-B", "trust-exact")
        ]
        np.testing.assert_allclose(estimates[1], estimates[0], atol=1e-4)

    def test_uses_improved_coefficients_when_optimizer_reports_failure(self):
        """A non-converged optimizer result should be used if it improves loss."""
        n_time = 5
        n_states = 3
        design_matrix = np.ones((n_time, 1))
        initial_coefficients = np.zeros((1, n_states, n_states - 1))
        improved_row = np.array([2.0, -1.0])
        response = np.tile(np.array([0.95, 0.04, 0.01]), (n_time - 1, 1))
        response = response[:, np.newaxis, :] * np.ones((1, n_states, 1))

        class FakeResult:
            success = False
            message = "iteration limit"
            x = improved_row.copy()
            fun = -1.0

        with patch(
            "non_local_detector.discrete_state_transitions.minimize",
            return_value=FakeResult(),
        ):
            coefficients, transition_matrix = (
                estimate_non_stationary_state_transition_from_responses(
                    transition_coefficients=initial_coefficients,
                    design_matrix=design_matrix,
                    response=response,
                    concentration=1.0,
                    stickiness=0.0,
                    transition_regularization=0.0,
                )
            )

        np.testing.assert_allclose(coefficients[:, 0, :], improved_row[np.newaxis])
        expected_row = np.exp(
            jax_centered_log_softmax_forward(improved_row[np.newaxis])
        )[0]
        np.testing.assert_allclose(transition_matrix[0, 0], expected_row, atol=1e-6)


def _sampled_transition_responses(
    rng: np.random.Generator, transition: np.ndarray
) -> np.ndarray:
    """One-hot next states drawn independently for every row and time bin.

    Parameters
    ----------
    transition : np.ndarray, shape (n_time - 1, n_states, n_states)

    Returns
    -------
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
    """
    cumulative = np.cumsum(transition, axis=-1)
    draws = rng.uniform(size=transition.shape[:2] + (1,))
    next_state = np.minimum((draws > cumulative).sum(axis=-1), transition.shape[-1] - 1)
    return np.eye(transition.shape[-1])[next_state]


@pytest.mark.unit
class TestNonStationaryPriorAndConvergence:
    """The covariate M-step's prior counts once per row, as for stationary
    transitions, and its optimizer reaches the optimum on long recordings."""

    def test_intercept_only_design_matches_stationary_map_estimate(self):
        """With only an intercept, a covariate transition is stationary, so
        the M-step must give the stationary MAP estimate for the same prior.
        A prior added at every time bin instead swamps the data."""
        rng = np.random.default_rng(0)
        n_time, n_states = 20_001, 3
        true_transition = np.array(
            [[0.97, 0.02, 0.01], [0.05, 0.90, 0.05], [0.10, 0.10, 0.80]]
        )
        response = _sampled_transition_responses(
            rng, np.broadcast_to(true_transition, (n_time - 1, n_states, n_states))
        )
        stickiness = np.array([300.0, 100.0, 0.0])

        _, covariate_transition = (
            estimate_non_stationary_state_transition_from_responses(
                np.zeros((1, n_states, n_states - 1)),
                np.ones((n_time, 1)),
                response,
                concentration=1.0,
                stickiness=stickiness,
                transition_regularization=0.0,
            )
        )
        stationary_transition = estimate_stationary_state_transition_from_counts(
            response.sum(axis=0), stickiness=stickiness, concentration=1.0
        )

        np.testing.assert_allclose(
            covariate_transition[0], stationary_transition, atol=1e-6
        )
        np.testing.assert_allclose(covariate_transition[-1], covariate_transition[0])

    def test_gradient_and_hessian_match_finite_differences(self):
        rng = np.random.default_rng(3)
        design_matrix = np.column_stack((np.ones(200), rng.normal(size=(200, 2))))
        response = rng.dirichlet(np.ones(4), size=200)
        args = (design_matrix, response, np.array([5.0, 1.0, 2.0, 1.0]), 0.3)
        coefficients = rng.normal(0, 0.5, 3 * 3)
        step = 1e-6
        identity = np.eye(coefficients.size)

        numerical_gradient = [
            (
                dirichlet_neg_log_likelihood(coefficients + step * e, *args)
                - dirichlet_neg_log_likelihood(coefficients - step * e, *args)
            )
            / (2 * step)
            for e in identity
        ]
        numerical_hessian = np.stack(
            [
                (
                    dirichlet_gradient(coefficients + step * e, *args)
                    - dirichlet_gradient(coefficients - step * e, *args)
                )
                / (2 * step)
                for e in identity
            ]
        )

        np.testing.assert_allclose(
            dirichlet_gradient(coefficients, *args), numerical_gradient, rtol=1e-6
        )
        np.testing.assert_allclose(
            dirichlet_hessian(coefficients, *args),
            numerical_hessian,
            rtol=1e-6,
            atol=1e-8,
        )

    def test_m_step_functions_default_to_the_detectors_ridge(self):
        """Direct calls get the same L2 penalty as a detector's M-step."""
        from non_local_detector import NonLocalSortedSpikesDetector

        detector_default = (
            NonLocalSortedSpikesDetector().discrete_transition_regularization
        )
        defaults = [
            inspect.signature(estimate_non_stationary_state_transition_from_responses)
            .parameters["transition_regularization"]
            .default
        ] + [
            inspect.signature(function).parameters["l2_penalty"].default
            for function in (
                dirichlet_neg_log_likelihood,
                dirichlet_gradient,
                dirichlet_hessian,
            )
        ]
        assert defaults == [detector_default] * 4

    @pytest.mark.parametrize(
        ("regularization", "bounded"), [(0.1, True), (1e-10, False)]
    )
    def test_ridge_keeps_rare_transitions_off_zero(
        self, regularization, bounded, caplog
    ):
        """Rare transitions leave some speed bands without counts, and
        tracking outliers make a basis column with few samples. Unpenalized,
        the converged optimum sends those transitions to ~0; the default
        ridge (0.1) keeps them bounded. Either way the optimizer converges."""
        rng = np.random.default_rng(4)
        n_time, n_states = 50_001, 4
        speed = rng.gamma(1.5, 12.0, n_time)
        speed[rng.uniform(size=n_time) < 0.6] = 0.0
        speed[rng.choice(n_time, 20, replace=False)] = rng.uniform(300, 2000, 20)
        stay = np.array([0.99, 0.98, 0.96, 0.95])
        transition = (
            np.full((n_states, n_states), 0.0) + ((1 - stay) / (n_states - 1))[:, None]
        )
        np.fill_diagonal(transition, stay)
        response = _sampled_transition_responses(
            rng, np.broadcast_to(transition, (n_time - 1, n_states, n_states))
        )
        _, initial_coefficients, design_matrix = DiscreteNonStationaryDiagonal(
            stay
        ).make_state_transition({"speed": speed})

        with caplog.at_level("WARNING"):
            _, estimated = estimate_non_stationary_state_transition_from_responses(
                initial_coefficients,
                design_matrix,
                response,
                stickiness=np.array([1e5, 1e5, 300.0, 300.0]),
                transition_regularization=regularization,
            )

        assert not [r for r in caplog.records if "optimization" in r.getMessage()]
        assert bool(estimated.min() > 1e-5) is bounded

    @pytest.mark.slow
    def test_reaches_optimum_on_long_recording(self):
        """On a long recording the returned coefficients are at the optimum of
        the summed objective, not stopped early by a tolerance scaled for a
        per-bin average."""
        rng = np.random.default_rng(1)
        n_time, n_states = 200_001, 3
        covariate = rng.gamma(2.0, 10.0, size=n_time)
        design_matrix = np.column_stack(
            (np.ones(n_time), np.log1p(covariate), np.sqrt(covariate))
        )
        true_coefficients = rng.normal(0, 0.5, (3, n_states, n_states - 1))
        true_coefficients[0] += 2.0 * np.eye(n_states)[:, : n_states - 1]
        logits = np.einsum("tc,csk->tsk", design_matrix[:-1], true_coefficients)
        true_transition = softmax(
            np.concatenate((logits, np.zeros(logits.shape[:-1] + (1,))), axis=-1),
            axis=-1,
        )
        response = _sampled_transition_responses(rng, true_transition)
        prior = {
            "concentration": 1.0,
            "stickiness": np.array([50.0, 0.0, 10.0]),
            "transition_regularization": 1e-3,
        }

        coefficients, _ = estimate_non_stationary_state_transition_from_responses(
            np.zeros_like(true_coefficients), design_matrix, response, **prior
        )
        alpha = 1.0 + np.diag(prior["stickiness"])
        n_samples = n_time - 1

        def negative_log_posterior(flat, row):
            """Summed float64 objective: expected log-likelihood, Dirichlet
            prior counted once, L2 on non-intercept coefficients."""
            beta = flat.reshape(3, n_states - 1)
            logits = design_matrix[:-1] @ beta
            log_probs = (
                logits
                - np.logaddexp.reduce(
                    np.column_stack((logits, np.zeros(n_samples))), axis=1
                )[:, None]
            )
            log_probs = np.column_stack((log_probs, log_probs[:, :1] - logits[:, :1]))
            weights = response[:, row] + (alpha[row] - 1.0) / n_samples
            penalty = prior["transition_regularization"] * np.sum(beta[1:] ** 2)
            return -np.sum(weights * log_probs) + penalty

        for row in range(n_states):
            reference = minimize(
                negative_log_posterior,
                coefficients[:, row].ravel(),
                args=(row,),
                method="BFGS",
                options={"gtol": 1e-8},
            )
            gap = negative_log_posterior(coefficients[:, row].ravel(), row) - min(
                reference.fun, negative_log_posterior(coefficients[:, row].ravel(), row)
            )
            # Summed over ~200,000 transitions: within 1e-3 nats of optimal.
            assert gap < 1e-3, f"row {row} stopped {gap:.3g} nats short"


def _float64_joint_sum(post):
    """Writable float64 copy of the fixture's stationary expected counts, so
    package and closed-form estimates share their inputs exactly."""
    return np.array(_joint_sum_from_post(post), dtype=np.float64)


def _closed_form_stationary_map(joint_sum, concentration, stickiness, prior_weight=0.0):
    """Stationary MAP transition matrix, computed independently of the package.

    Each row is ``counts + pseudo_counts`` normalized, where the Dirichlet
    pseudo-counts are ``alpha - 1`` with ``alpha = concentration +
    stickiness * I``. A row with ``prior_weight > 0`` keeps the direction of
    ``alpha - 1`` but rescales its total to ``prior_weight`` times that row's
    expected transition count.

    Parameters
    ----------
    joint_sum : np.ndarray, shape (n_states, n_states)
    concentration : float
    stickiness : float or np.ndarray, shape (n_states,)
    prior_weight : float or np.ndarray, shape (n_states,)

    Returns
    -------
    transition : np.ndarray, shape (n_states, n_states)
    """
    joint_sum = np.asarray(joint_sum, dtype=np.float64)
    n_states = joint_sum.shape[0]
    pseudo_counts = (concentration - 1.0) * np.ones((n_states, n_states))
    pseudo_counts += np.diag(np.broadcast_to(stickiness, (n_states,)))
    prior_weight = np.broadcast_to(prior_weight, (n_states,))
    for row in range(n_states):
        if prior_weight[row] > 0:
            pseudo_counts[row] *= (
                prior_weight[row] * joint_sum[row].sum() / pseudo_counts[row].sum()
            )
    unnormalized = joint_sum + pseudo_counts
    return unnormalized / unnormalized.sum(axis=1, keepdims=True)


@pytest.mark.unit
class TestEstimateStationaryStateTransition:
    """Test stationary state transition estimation."""

    def test_without_prior_is_row_normalized_counts(self, posterior_data):
        """concentration=1 and stickiness=0 give ``alpha - 1 = 0``: no prior,
        so the estimate is the maximum-likelihood row-normalized counts."""
        joint_sum = _float64_joint_sum(posterior_data)

        trans_matrix = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=0.0,
        )

        assert trans_matrix.shape == joint_sum.shape
        np.testing.assert_allclose(
            trans_matrix,
            _closed_form_stationary_map(joint_sum, 1.0, 0.0),
            rtol=1e-12,
        )

    def test_concentration_keeps_unobserved_transitions_nonzero(self, posterior_data):
        """With concentration > 1 a transition with zero expected count gets
        probability ``(concentration - 1) / row total``, not zero."""
        joint_sum = _float64_joint_sum(posterior_data)
        joint_sum[0, 3] = 0.0

        trans_matrix = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=2.0,
            stickiness=0.0,
        )

        assert_stochastic_matrix(trans_matrix)
        assert trans_matrix[0, 3] > 0.0
        np.testing.assert_allclose(
            trans_matrix,
            _closed_form_stationary_map(joint_sum, 2.0, 0.0),
            rtol=1e-12,
        )

    def test_respects_diagonal_prior(self, posterior_data):
        """Stickiness adds pseudo-counts to the diagonal: the estimate is the
        closed-form MAP, and every self-transition rises above the no-prior
        estimate (the fixture's data alone already favor the diagonal)."""
        joint_sum = _float64_joint_sum(posterior_data)

        no_prior = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=0.0,
        )
        trans_matrix = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
        )

        np.testing.assert_allclose(
            trans_matrix,
            _closed_form_stationary_map(joint_sum, 1.0, 2.0),
            rtol=1e-12,
        )
        assert np.all(np.diag(trans_matrix) > np.diag(no_prior))

    def test_numerical_stability_with_small_probabilities(self):
        """Test with very small posterior probabilities."""
        n_time, n_states = 20, 4
        causal_posterior = np.ones((n_time, n_states)) * 1e-8
        causal_posterior[:, 0] = 1.0 - 3e-8
        acausal_posterior = causal_posterior.copy()

        transition_matrix = np.eye(n_states) * 0.9 + 0.025
        # predictive[t] = causal[t - 1] @ T, as core.filter defines it.
        predictive_distribution = np.vstack(
            [causal_posterior[:1], causal_posterior[:-1] @ transition_matrix]
        )
        joint_sum = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal_posterior=causal_posterior,
            predictive_posterior=predictive_distribution,
            acausal_posterior=acausal_posterior,
            transition_matrix=transition_matrix,
            state_ind=np.arange(n_states),
        )

        trans_matrix = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=0.0,
        )

        assert np.all(np.isfinite(trans_matrix))
        assert_stochastic_matrix(trans_matrix)

    def test_concentration_below_one_raises(self, posterior_data):
        """The MAP pseudo-count update rejects sparse Dirichlet parameters."""
        post = posterior_data
        joint_sum = _joint_sum_from_post(post)

        with pytest.raises(ValueError, match="prior parameters >= 1.0"):
            estimate_stationary_state_transition_from_counts(
                joint_sum,
                concentration=0.5,
                stickiness=0.0,
            )


@pytest.mark.unit
class TestPriorWeightScaling:
    """Test data-adaptive prior weight for stationary transition estimation."""

    def test_prior_weight_zero_uses_legacy_prior(self, posterior_data):
        """prior_weight=0 should use the legacy fixed-count prior (unchanged behavior)."""
        post = posterior_data
        joint_sum = _joint_sum_from_post(post)

        legacy_result = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
        )

        pw_result = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.0,
        )

        np.testing.assert_allclose(pw_result, legacy_result, atol=1e-12)

    def test_prior_weight_matches_closed_form_adaptive_prior(self, posterior_data):
        """prior_weight > 0 rescales the sticky pseudo-counts to
        ``prior_weight`` times each row's expected count: the estimate is the
        closed-form adaptive MAP, it differs from the fixed-count prior with
        the same stickiness, and it raises every self-transition above the
        no-prior estimate."""
        joint_sum = _float64_joint_sum(posterior_data)

        no_prior = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=0.0,
        )
        legacy_result = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
        )
        pw_result = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )

        np.testing.assert_allclose(
            pw_result,
            _closed_form_stationary_map(joint_sum, 1.0, 2.0, prior_weight=0.1),
            rtol=1e-12,
        )
        assert np.max(np.abs(pw_result - legacy_result)) > 1e-3
        assert np.all(np.diag(pw_result) > np.diag(no_prior))

    def test_prior_weight_invariant_to_T(self, posterior_data):
        """Doubling every expected count (as repeating the data would) leaves
        the adaptive-prior estimate unchanged up to float rounding, while the
        fixed-count prior is diluted and its estimate moves."""
        joint_sum_T = _float64_joint_sum(posterior_data)
        joint_sum_2T = 2.0 * joint_sum_T

        def estimate(joint_sum, prior_weight):
            return estimate_stationary_state_transition_from_counts(
                joint_sum,
                concentration=1.0,
                stickiness=2.0,
                prior_weight=prior_weight,
            )

        np.testing.assert_allclose(
            estimate(joint_sum_T, 0.1), estimate(joint_sum_2T, 0.1), atol=1e-12
        )
        legacy_change = np.max(
            np.abs(estimate(joint_sum_T, 0.0) - estimate(joint_sum_2T, 0.0))
        )
        assert legacy_change > 1e-2

    def test_prior_weight_produces_valid_stochastic_matrix(self, posterior_data):
        """Result should always be a valid stochastic matrix."""
        post = posterior_data

        result = estimate_stationary_state_transition_from_counts(
            _joint_sum_from_post(post),
            concentration=1.5,
            stickiness=3.0,
            prior_weight=0.5,
        )

        assert_stochastic_matrix(result)
        assert np.all(np.isfinite(result))

    def test_prior_weight_handles_unvisited_state(self):
        """A state with zero posterior mass should not produce NaN."""
        n_time, n_states = 20, 3

        causal_posterior = np.zeros((n_time, n_states))
        causal_posterior[:, 0] = 0.6
        causal_posterior[:, 1] = 0.4
        acausal_posterior = causal_posterior.copy()

        transition_matrix = np.eye(n_states) * 0.8 + 0.2 / n_states
        # predictive[t] = causal[t - 1] @ T, as core.filter defines it.
        predictive_distribution = np.vstack(
            [causal_posterior[:1], causal_posterior[:-1] @ transition_matrix]
        )
        joint_sum = estimate_discrete_transition_counts_from_expanded_posteriors(
            causal_posterior=causal_posterior,
            predictive_posterior=predictive_distribution,
            acausal_posterior=acausal_posterior,
            transition_matrix=transition_matrix,
            state_ind=np.arange(n_states),
        )

        result = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )

        assert np.all(np.isfinite(result)), f"NaN/Inf in result: {result}"
        assert_stochastic_matrix(result)

        assert result[2, 2] > result[2, 0], (
            "Unvisited state should reflect sticky prior, not uniform"
        )

    def test_per_state_prior_weight_scalar_equivalence(self, posterior_data):
        """Passing scalar prior_weight should match passing uniform array."""
        post = posterior_data
        n_states = post["n_states"]
        joint_sum = _joint_sum_from_post(post)

        scalar_result = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
            prior_weight=0.1,
        )
        array_result = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=2.0,
            prior_weight=np.full(n_states, 0.1),
        )

        np.testing.assert_allclose(scalar_result, array_result, atol=1e-12)

    def test_per_state_prior_weight_mixes_legacy_and_adaptive_rows(
        self, posterior_data
    ):
        """A row with prior_weight=0 (here a frozen-like row with 1e6
        stickiness) uses the fixed-count prior and matches the all-legacy
        call; rows with prior_weight > 0 use the adaptive prior and differ
        from the all-legacy call."""
        n_states = posterior_data["n_states"]
        joint_sum = _float64_joint_sum(posterior_data)
        sticky = np.array([1e6] + [2.0] * (n_states - 1))
        prior_weight = np.array([0.0] + [0.1] * (n_states - 1))

        legacy = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=sticky,
            prior_weight=0.0,
        )
        mixed = estimate_stationary_state_transition_from_counts(
            joint_sum,
            concentration=1.0,
            stickiness=sticky,
            prior_weight=prior_weight,
        )

        assert_stochastic_matrix(mixed)
        assert mixed[0, 0] > 0.99
        np.testing.assert_allclose(mixed[0], legacy[0], atol=1e-12)
        np.testing.assert_allclose(
            mixed,
            _closed_form_stationary_map(joint_sum, 1.0, sticky, prior_weight),
            rtol=1e-12,
        )
        assert np.all(np.max(np.abs(mixed[1:] - legacy[1:]), axis=1) > 1e-3)

    def test_per_state_prior_weight_negative_raises(self, posterior_data):
        """Negative per-state prior_weight should raise ValueError."""
        post = posterior_data
        n_states = post["n_states"]

        with pytest.raises(ValueError, match="non-negative"):
            estimate_stationary_state_transition_from_counts(
                _joint_sum_from_post(post),
                concentration=1.0,
                stickiness=2.0,
                prior_weight=np.array([0.1, -0.1] + [0.1] * (n_states - 2)),
            )

    def test_per_state_prior_weight_wrong_shape_raises(self, posterior_data):
        """prior_weight array with wrong length should raise ValueError."""
        post = posterior_data

        with pytest.raises(ValueError, match="prior_weight"):
            estimate_stationary_state_transition_from_counts(
                _joint_sum_from_post(post),
                concentration=1.0,
                stickiness=2.0,
                prior_weight=np.array([0.1, 0.1]),
            )


@pytest.mark.unit
class TestDiscreteTransitionMStep:
    """The EM discrete M-step: expected statistics from the expanded
    posteriors, then the transition update from them."""

    def test_stationary_diagonal(self, posterior_data):
        """Test with stationary diagonal transition type using singleton bins."""
        post = posterior_data
        n_states = post["n_states"]
        # Singleton-state expanded HMM: one bin per discrete state, identity C.
        # Reduces the exact path to the discrete-only EM update.
        state_ind = np.arange(n_states)
        continuous_transition = np.eye(n_states)

        statistics = _discrete_transition_statistics(
            causal_posterior=post["causal_posterior"],
            predictive_posterior=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            continuous_transition=continuous_transition,
            state_ind=state_ind,
            discrete_transition=post["transition_matrix"],
            discrete_transition_coefficients=None,
            discrete_transition_design_matrix=None,
        )
        new_trans, _ = _discrete_transition_from_statistics(
            statistics,
            None,
            None,
            transition_concentration=1.0,
            transition_stickiness=1.0,
            transition_regularization=1e-5,
        )

        assert new_trans.shape == (n_states, n_states)
        assert_stochastic_matrix(new_trans)

    def test_stationary_uses_exact_counts(self):
        """Expanded inputs reach the exact path."""
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

        statistics = _discrete_transition_statistics(
            causal_posterior=causal_posterior,
            predictive_posterior=predictive_posterior,
            acausal_posterior=acausal_posterior,
            continuous_transition=continuous_transition,
            state_ind=state_ind,
            discrete_transition=discrete_transition,
            discrete_transition_coefficients=None,
            discrete_transition_design_matrix=None,
        )
        new_trans, _ = _discrete_transition_from_statistics(
            statistics,
            None,
            None,
            transition_concentration=1.0,
            transition_stickiness=0.0,
            transition_regularization=1e-5,
        )

        assert new_trans[1, 2] > new_trans[0, 2]

    def test_non_stationary_diagonal(self, posterior_data, design_matrix_data):
        """Test with non-stationary diagonal transition type using singleton bins."""
        post = posterior_data
        dm = design_matrix_data
        n_states = post["n_states"]
        n_time = post["n_time"]
        state_ind = np.arange(n_states)
        continuous_transition = np.eye(n_states)
        time_varying_trans = np.tile(post["transition_matrix"], (n_time, 1, 1))

        statistics = _discrete_transition_statistics(
            causal_posterior=post["causal_posterior"],
            predictive_posterior=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            continuous_transition=continuous_transition,
            state_ind=state_ind,
            discrete_transition=time_varying_trans,
            discrete_transition_coefficients=dm["transition_coefficients"],
            discrete_transition_design_matrix=dm["design_matrix"][:n_time],
        )
        new_trans, new_coeffs = _discrete_transition_from_statistics(
            statistics,
            dm["transition_coefficients"],
            dm["design_matrix"][:n_time],
            transition_concentration=1.0,
            transition_stickiness=1.0,
            transition_regularization=1e-5,
        )

        assert new_trans.shape == (n_time, n_states, n_states)
        assert new_coeffs.shape == dm["transition_coefficients"].shape
        for t in range(min(5, new_trans.shape[0])):
            assert_stochastic_matrix(new_trans[t])

    def test_nonstationary_uses_expanded_responses(self):
        """Expanded nonstationary inputs reach the exact response path."""
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
            statistics = _discrete_transition_statistics(
                causal_posterior=causal_posterior,
                predictive_posterior=predictive_posterior,
                acausal_posterior=acausal_posterior,
                continuous_transition=continuous_transition,
                state_ind=state_ind,
                discrete_transition=discrete_transition,
                discrete_transition_coefficients=transition_coefficients,
                discrete_transition_design_matrix=design_matrix,
            )
            new_trans, new_coeffs = _discrete_transition_from_statistics(
                statistics,
                transition_coefficients,
                design_matrix,
                transition_concentration=1.0,
                transition_stickiness=0.0,
                transition_regularization=1e-5,
            )

        assert new_trans.shape == (2, 3, 3)
        assert new_coeffs.shape == transition_coefficients.shape
        assert captured["response"][0, 1, 2] > captured["response"][0, 0, 2]


def _random_stochastic(rng: np.random.Generator, n_states: int) -> np.ndarray:
    matrix = rng.uniform(0.05, 1.0, (n_states, n_states))
    return matrix / matrix.sum(axis=1, keepdims=True)


@pytest.mark.unit
class TestTransitionLogPrior:
    """``_transition_log_prior`` is the penalty the discrete M-step maximizes.

    EM with a MAP M-step never decreases marginal log-likelihood + this
    penalty, which is what the EM convergence check monitors.
    """

    def test_stationary_matches_dirichlet_formula_over_free_rows(self):
        rng = np.random.default_rng(0)
        transition = _random_stochastic(rng, 3)
        counts = rng.uniform(0, 50, (3, 3))
        stickiness = np.array([5.0, 0.0, 2.0])
        free_rows = np.array([True, False, True])

        log_prior = _transition_log_prior(
            counts, transition, None, None, 1.5, stickiness, 0.0, 0.0, free_rows
        )

        pseudo_counts = 0.5 + np.diag(stickiness)
        expected = np.sum((pseudo_counts * np.log(transition))[free_rows])
        np.testing.assert_allclose(log_prior, expected, rtol=1e-12)

    @pytest.mark.parametrize("prior_weight", [0.0, 0.5])
    def test_stationary_m_step_maximizes_counts_plus_log_prior(self, prior_weight):
        """The M-step's matrix beats every other stochastic matrix on
        expected complete-data log-likelihood + log prior."""
        rng = np.random.default_rng(1)
        counts = rng.uniform(0, 50, (4, 4))
        stickiness = np.array([200.0, 0.0, 30.0, 0.0])
        prior = (None, None, 2.0, stickiness, 0.0, prior_weight)

        def objective(transition):
            return np.sum(counts * np.log(transition)) + _transition_log_prior(
                counts, transition, *prior
            )

        m_step = estimate_stationary_state_transition_from_counts(
            counts, stickiness=stickiness, concentration=2.0, prior_weight=prior_weight
        )
        best = objective(m_step)
        for _ in range(50):
            assert objective(_random_stochastic(rng, 4)) < best
            nearby = m_step * np.exp(rng.normal(0, 1e-3, m_step.shape))
            assert objective(nearby / nearby.sum(axis=1, keepdims=True)) <= best

    def test_non_stationary_matches_m_step_loss(self):
        """Expected log-likelihood + log prior equals minus the summed
        per-row loss that the non-stationary M-step minimizes."""
        rng = np.random.default_rng(2)
        n_time, n_coefficients, n_states = 40, 3, 3
        design_matrix = np.column_stack(
            (np.ones(n_time), rng.normal(size=(n_time, n_coefficients - 1)))
        )
        coefficients = rng.normal(0, 0.5, (n_coefficients, n_states, n_states - 1))
        response = rng.uniform(0, 1, (n_time - 1, n_states, n_states))
        stickiness = np.array([20.0, 0.0, 5.0])
        concentration, regularization = 1.5, 0.3
        alpha = concentration + np.diag(stickiness)

        log_prior = _transition_log_prior(
            response,
            None,
            coefficients,
            design_matrix,
            concentration,
            stickiness,
            regularization,
            0.0,
        )

        linear_predictor = np.concatenate(
            (
                np.einsum("tc,csk->tsk", design_matrix[:-1], coefficients),
                np.zeros((n_time - 1, n_states, 1)),
            ),
            axis=-1,
        )
        log_probs = linear_predictor - logsumexp(
            linear_predictor, axis=-1, keepdims=True
        )
        expected_log_likelihood = np.sum(response * log_probs)
        summed_loss = sum(
            dirichlet_neg_log_likelihood(
                coefficients[:, row].ravel(),
                design_matrix[:-1],
                response[:, row],
                alpha[row],
                regularization,
            )
            for row in range(n_states)
        )
        # Both in float64, as the M-step evaluates its loss.
        np.testing.assert_allclose(
            expected_log_likelihood + log_prior, -summed_loss, rtol=1e-12
        )

    def test_non_stationary_without_transitions_is_zero(self):
        """A one-bin recording has no transitions, so no covariate prior."""
        log_prior = _transition_log_prior(
            np.empty((0, 3, 3)),
            None,
            np.zeros((2, 3, 2)),
            np.ones((1, 2)),
            1.0,
            np.array([5.0, 0.0, 0.0]),
            1e-5,
        )
        assert log_prior == 0.0

    def test_non_stationary_change_is_float64_accurate(self):
        """The covariate log prior changes between nearby coefficients as a
        float64 reference does, far inside ``EM_MONOTONICITY_SLACK``, which a
        float32 evaluation does not resolve."""
        rng = np.random.default_rng(3)
        n_time, n_coefficients, n_states = 200_000, 2, 4
        design_matrix = np.column_stack(
            (np.ones(n_time), np.sin(np.arange(n_time) / 500.0))
        )
        # Stay probability about 0.999, the regime of real recordings; the
        # last target state is the reference category.
        coefficients = rng.normal(0, 0.1, (n_coefficients, n_states, n_states - 1))
        for row in range(n_states - 1):
            coefficients[0, row, row] += 8.0
        coefficients[0, n_states - 1] -= 8.0
        nearby = coefficients + 1e-4 * rng.normal(size=coefficients.shape)
        response = np.zeros((n_time - 1, n_states, n_states))
        concentration, stickiness, regularization = 1.0, 1e3, 1e-5
        alpha = get_transition_prior(concentration, stickiness, n_states)

        def reference(coefficients):
            total = 0.0
            for row in range(n_states):
                logits = design_matrix[:-1] @ coefficients[:, row]
                logits = np.column_stack((logits, np.zeros(n_time - 1)))
                log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
                # The prior counts once per row: alpha - 1 times the mean log
                # probability over the transitions.
                total += np.sum((alpha[row] - 1.0) * log_probs) / (n_time - 1)
                total -= regularization * np.sum(coefficients[1:, row] ** 2)
            return total

        def log_prior(coefficients):
            return _transition_log_prior(
                response,
                None,
                coefficients,
                design_matrix,
                concentration,
                stickiness,
                regularization,
            )

        scale = abs(reference(coefficients))
        change = log_prior(nearby) - log_prior(coefficients)
        reference_change = reference(nearby) - reference(coefficients)
        assert abs(change - reference_change) < 1e-10 * scale

    def test_zero_probability_entries(self):
        """A zero entry with no pseudo-count contributes nothing; a zero entry
        the prior puts mass on makes the log prior ``-inf``."""
        transition = np.array([[1.0, 0.0], [0.5, 0.5]])
        counts = np.ones((2, 2))

        no_prior = _transition_log_prior(counts, transition, None, None, 1.0, 0.0, 0.0)
        assert no_prior == 0.0
        sticky_second = _transition_log_prior(
            counts, transition, None, None, 1.0, np.array([0.0, 3.0]), 0.0
        )
        np.testing.assert_allclose(sticky_second, 3.0 * np.log(0.5))
        uniform_prior = _transition_log_prior(
            counts, transition, None, None, 2.0, 0.0, 0.0
        )
        assert uniform_prior == -np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
