"""Reference-value tests for HMM conditioning and normalization in ``core``.

Every assertion here compares against an *independent* float64 calculation
(NumPy/SciPy log-sum-exp, or explicit enumeration of state paths), never
against another caller of the same JAX implementation. Two historical
defects motivate the one-step cases:

1. ``_normalize`` added a fixed ``1e-15`` to the normalizer, so a positive
   total mass far below that epsilon produced a posterior summing to ~0.
2. ``_condition_on`` shifted the log-likelihoods by their global maximum, which
   may sit on a zero-prior state; the reachable states then underflowed and a
   possible observation was reported as impossible (prior returned, ``-inf``
   evidence).
"""

import itertools
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from non_local_detector import core as core_module
from non_local_detector.core import (
    _condition_on,
    _normalize,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    filter,
    filter_covariate_dependent,
    smoother,
    smoother_covariate_dependent,
)

# Tolerance for comparing float32 JAX output with the float64 reference.
RTOL = 1e-5
ATOL = 1e-6

# (prior, log-likelihood) pairs reproducing the two historical defects.
CASE_SMALL_NORMALIZER = ([1e-20, 1.0], [0.0, -100.0])
CASE_UNREACHABLE_MAXIMUM = ([0.5, 0.5, 0.0], [-1000.0, -1001.0, 0.0])
ONE_STEP_CASES = {
    "small_normalizer": CASE_SMALL_NORMALIZER,
    "unreachable_maximum": CASE_UNREACHABLE_MAXIMUM,
}


def _as_f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def bayes_reference(prior, log_likelihood) -> tuple[np.ndarray, float]:
    """Posterior and log evidence of one Bayes update, in float64 log space.

    Parameters
    ----------
    prior : array-like, shape (n_states,)
    log_likelihood : array-like, shape (n_states,)

    Returns
    -------
    posterior : np.ndarray, shape (n_states,)
    log_evidence : float
    """
    with np.errstate(divide="ignore"):
        log_joint = np.log(np.asarray(prior, dtype=np.float64)) + np.asarray(
            log_likelihood, dtype=np.float64
        )
    log_evidence = logsumexp(log_joint)
    return np.exp(log_joint - log_evidence), float(log_evidence)


def enumerate_reference(
    initial: np.ndarray, transitions: np.ndarray, log_likelihoods: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Exact HMM marginals by enumerating every state path in float64 log space.

    Parameters
    ----------
    initial : np.ndarray, shape (n_states,)
    transitions : np.ndarray, shape (n_time, n_states, n_states)
        ``transitions[t]`` moves ``s_t -> s_{t+1}``; the last entry is unused,
        matching the indexing convention of ``core.filter``.
    log_likelihoods : np.ndarray, shape (n_time, n_states)

    Returns
    -------
    dict with keys ``filtered``, ``predicted``, ``smoothed`` (each
    ``(n_time, n_states)``) and ``log_evidence`` (float).
    """
    initial = np.asarray(initial, dtype=np.float64)
    transitions = np.asarray(transitions, dtype=np.float64)
    log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
    n_time, n_states = log_likelihoods.shape
    with np.errstate(divide="ignore"):
        log_initial = np.log(initial)
        log_transitions = np.log(transitions)

    # log weight of every prefix path s_0..s_t, with and without the final
    # observation term, for each t.
    filtered = np.zeros((n_time, n_states))
    predicted = np.zeros((n_time, n_states))
    smoothed = np.zeros((n_time, n_states))
    log_evidence = None
    for t in range(n_time):
        paths = list(itertools.product(range(n_states), repeat=t + 1))
        log_w_predict = np.empty(len(paths))
        log_w_filter = np.empty(len(paths))
        for i, path in enumerate(paths):
            log_w = log_initial[path[0]]
            for k in range(1, t + 1):
                log_w += log_transitions[k - 1, path[k - 1], path[k]]
            log_w += log_likelihoods[np.arange(t), path[:-1]].sum()
            log_w_predict[i] = log_w
            log_w_filter[i] = log_w + log_likelihoods[t, path[-1]]
        last = np.array([path[-1] for path in paths])
        for s in range(n_states):
            sel = last == s
            predicted[t, s] = logsumexp(log_w_predict[sel]) - logsumexp(log_w_predict)
            filtered[t, s] = logsumexp(log_w_filter[sel]) - logsumexp(log_w_filter)
        if t == n_time - 1:
            log_evidence = float(logsumexp(log_w_filter))
            path_arr = np.array(paths)
            for tt in range(n_time):
                for s in range(n_states):
                    sel = path_arr[:, tt] == s
                    smoothed[tt, s] = logsumexp(log_w_filter[sel]) - log_evidence
    return {
        "filtered": np.exp(filtered),
        "predicted": np.exp(predicted),
        "smoothed": np.exp(smoothed),
        "log_evidence": log_evidence,
    }


def _tiny_model(seed: int = 0, n_time: int = 5, n_states: int = 3):
    """Random tiny HMM with well-conditioned rows; float64 NumPy arrays."""
    rng = np.random.default_rng(seed)
    initial = rng.dirichlet(np.ones(n_states))
    transitions = np.stack(
        [rng.dirichlet(np.ones(n_states), size=n_states) for _ in range(n_time)]
    )
    log_likelihoods = rng.standard_normal((n_time, n_states)) * 2.0
    return initial, transitions, log_likelihoods


def _covariate_model(
    seed: int = 1, n_time: int = 5, n_discrete: int = 2, n_bins: int = 2
):
    """Tiny covariate-dependent model: (discrete, continuous, state_ind) pieces.

    Returns the pieces plus the equivalent expanded ``(n_time, K, K)`` joint
    transition used by the enumeration reference, where ``K = n_discrete * n_bins``.
    """
    rng = np.random.default_rng(seed)
    n_state_bins = n_discrete * n_bins
    state_ind = np.repeat(np.arange(n_discrete), n_bins)
    discrete = np.stack(
        [rng.dirichlet(np.ones(n_discrete), size=n_discrete) for _ in range(n_time)]
    )
    continuous = np.zeros((n_state_bins, n_state_bins))
    for d in range(n_discrete):
        for e in range(n_discrete):
            block = rng.dirichlet(np.ones(n_bins), size=n_bins)
            continuous[d * n_bins : (d + 1) * n_bins, e * n_bins : (e + 1) * n_bins] = (
                block
            )
    initial = rng.dirichlet(np.ones(n_state_bins))
    log_likelihoods = rng.standard_normal((n_time, n_state_bins)) * 2.0
    expanded = np.stack(
        [continuous * discrete[t][np.ix_(state_ind, state_ind)] for t in range(n_time)]
    )
    return initial, discrete, continuous, state_ind, log_likelihoods, expanded


@pytest.mark.unit
class TestNormalizeSmallMass:
    """``_normalize`` must divide by the true sum, not ``sum + eps``."""

    def test_small_positive_sum_normalizes_to_one(self):
        # Total mass 4e-20 is far below the old 1e-15 epsilon.
        u = jnp.asarray(_as_f32([1e-20, 3e-20]))

        normalized, const = _normalize(u)

        np.testing.assert_allclose(np.asarray(normalized), [0.25, 0.75], rtol=RTOL)
        assert float(normalized.sum()) == pytest.approx(1.0, rel=RTOL)
        assert float(const) == pytest.approx(4e-20, rel=RTOL)

    def test_zero_input_keeps_zero_contract(self):
        u = jnp.zeros(4)

        normalized, const = _normalize(u)

        assert float(const) == 0.0
        assert np.all(np.asarray(normalized) == 0.0)

    def test_nan_input_stays_visible(self):
        u = jnp.asarray([0.5, jnp.nan])

        normalized, const = _normalize(u)

        assert jnp.isnan(const)
        assert jnp.any(jnp.isnan(normalized))


@pytest.mark.unit
class TestConditionOnReference:
    """One-step conditioning reproduces the float64 reference."""

    @pytest.mark.parametrize("case", ONE_STEP_CASES, ids=list(ONE_STEP_CASES))
    def test_direct_conditioning_matches_reference(self, case):
        prior, ll = (_as_f32(v) for v in ONE_STEP_CASES[case])
        expected_posterior, expected_evidence = bayes_reference(prior, ll)

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll))

        np.testing.assert_allclose(
            np.asarray(posterior), expected_posterior, rtol=RTOL, atol=ATOL
        )
        assert float(np.asarray(posterior).sum()) == pytest.approx(1.0, rel=RTOL)
        assert float(log_norm) == pytest.approx(expected_evidence, rel=1e-6)

    @pytest.mark.parametrize("case", ONE_STEP_CASES, ids=list(ONE_STEP_CASES))
    def test_filter_matches_reference_without_false_warning(self, case, caplog):
        prior, ll = (_as_f32(v) for v in ONE_STEP_CASES[case])
        expected_posterior, expected_evidence = bayes_reference(prior, ll)

        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            (evidence, _), (posterior, _) = filter(
                jnp.asarray(prior),
                jnp.eye(len(prior), dtype=prior.dtype),
                jnp.asarray(ll[None, :]),
            )

        np.testing.assert_allclose(
            np.asarray(posterior)[0], expected_posterior, rtol=RTOL, atol=ATOL
        )
        assert float(evidence) == pytest.approx(expected_evidence, rel=1e-6)
        assert not caplog.records, [r.getMessage() for r in caplog.records]

    @pytest.mark.parametrize("case", ONE_STEP_CASES, ids=list(ONE_STEP_CASES))
    def test_filter_covariate_dependent_matches_reference(self, case, caplog):
        prior, ll = (_as_f32(v) for v in ONE_STEP_CASES[case])
        n_states = len(prior)
        expected_posterior, expected_evidence = bayes_reference(prior, ll)

        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            (evidence, _), (posterior, _) = filter_covariate_dependent(
                jnp.asarray(prior),
                jnp.eye(n_states, dtype=prior.dtype)[None],
                jnp.eye(n_states, dtype=prior.dtype),
                jnp.arange(n_states),
                jnp.asarray(ll[None, :]),
            )

        np.testing.assert_allclose(
            np.asarray(posterior)[0], expected_posterior, rtol=RTOL, atol=ATOL
        )
        assert float(evidence) == pytest.approx(expected_evidence, rel=1e-6)
        assert not caplog.records, [r.getMessage() for r in caplog.records]


@pytest.mark.unit
class TestConditionOnInvariants:
    """Structural properties that hold for any correct Bayes update."""

    @pytest.mark.parametrize("case", ONE_STEP_CASES, ids=list(ONE_STEP_CASES))
    def test_permuting_states_permutes_posterior(self, case):
        prior, ll = (_as_f32(v) for v in ONE_STEP_CASES[case])
        perm = np.arange(len(prior))[::-1]

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll))
        posterior_perm, log_norm_perm = _condition_on(
            jnp.asarray(prior[perm]), jnp.asarray(ll[perm])
        )

        np.testing.assert_allclose(
            np.asarray(posterior_perm),
            np.asarray(posterior)[perm],
            rtol=RTOL,
            atol=ATOL,
        )
        assert float(log_norm_perm) == pytest.approx(float(log_norm), rel=1e-6)

    @pytest.mark.parametrize("case", ONE_STEP_CASES, ids=list(ONE_STEP_CASES))
    def test_common_offset_shifts_evidence_only(self, case):
        prior, ll = (_as_f32(v) for v in ONE_STEP_CASES[case])
        offset = np.float32(37.5)

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll))
        posterior_shift, log_norm_shift = _condition_on(
            jnp.asarray(prior), jnp.asarray(ll + offset)
        )

        np.testing.assert_allclose(
            np.asarray(posterior_shift), np.asarray(posterior), rtol=RTOL, atol=ATOL
        )
        assert float(log_norm_shift) - float(log_norm) == pytest.approx(
            float(offset), abs=1e-3
        )

    def test_extreme_finite_likelihoods_do_not_overflow(self):
        """``ll - ll_shift`` can overflow to ``+inf`` on an unreachable state.

        Finite inputs must still give the finite reference answer (the
        overflow is an artifact of the shift, not invalid input).
        """
        prior, ll = _as_f32([1.0, 0.0]), _as_f32([-3e38, 3e38])
        expected_posterior, expected_evidence = bayes_reference(prior, ll)

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll))

        np.testing.assert_allclose(
            np.asarray(posterior), expected_posterior, rtol=RTOL, atol=ATOL
        )
        assert float(log_norm) == pytest.approx(expected_evidence, rel=1e-6)

    @pytest.mark.parametrize("unreachable_ll", [0.0, 500.0, -3000.0, 3e38, -3e38])
    def test_unreachable_state_likelihood_has_no_effect(self, unreachable_ll):
        prior, ll = (_as_f32(v) for v in CASE_UNREACHABLE_MAXIMUM)
        ll_variant = ll.copy()
        ll_variant[2] = unreachable_ll
        expected_posterior, expected_evidence = bayes_reference(prior, ll)

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll_variant))

        np.testing.assert_allclose(
            np.asarray(posterior), expected_posterior, rtol=RTOL, atol=ATOL
        )
        assert float(np.asarray(posterior)[2]) == 0.0
        assert float(log_norm) == pytest.approx(expected_evidence, rel=1e-6)


@pytest.mark.unit
class TestGenuineImpossibilityAndNaN:
    """Zero support keeps the prior fallback; NaN stays visible everywhere."""

    @pytest.mark.parametrize(
        "prior, ll",
        [
            ([0.3, 0.7], [-np.inf, -np.inf]),  # all impossible
            ([1.0, 0.0], [-np.inf, 0.0]),  # disjoint support
        ],
        ids=["all_inf", "disjoint_support"],
    )
    def test_zero_support_returns_prior_and_neg_inf(self, prior, ll):
        prior, ll = _as_f32(prior), _as_f32(ll)

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll))

        np.testing.assert_array_equal(np.asarray(posterior), prior)
        assert jnp.isneginf(log_norm)

    @pytest.mark.parametrize(
        "prior, ll",
        [
            ([0.4, 0.6], [-0.5, np.nan]),  # NaN on a reachable state
            ([1.0, 0.0], [-0.5, np.nan]),  # NaN on a zero-prior state
            ([0.5, 0.5], [-np.inf, np.nan]),  # -inf and NaN mixed
            # NaN on the zero-prior state while every reachable state is -inf:
            # the reachable normalizer is 0, so the NaN must not be hidden by
            # the zero-support fallback.
            ([1.0, 0.0], [-np.inf, np.nan]),
        ],
        ids=[
            "nan_reachable",
            "nan_zero_prior",
            "inf_and_nan",
            "nan_zero_prior_zero_norm",
        ],
    )
    def test_nan_propagates(self, prior, ll):
        prior, ll = _as_f32(prior), _as_f32(ll)

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll))

        assert jnp.any(jnp.isnan(posterior))
        assert jnp.isnan(log_norm)


@pytest.mark.unit
class TestPositiveInfiniteLikelihood:
    """A ``+inf`` log-likelihood is invalid input and must stay visible."""

    @pytest.mark.parametrize(
        "prior, ll",
        [
            ([0.5, 0.5], [np.inf, -1.0]),  # +inf on a reachable state
            ([1.0, 0.0], [-1.0, np.inf]),  # +inf on a zero-prior state
            (
                [1.0, 0.0],
                [-np.inf, np.inf],
            ),  # +inf on a zero-prior state, reachable -inf
        ],
        ids=["reachable", "zero_prior", "zero_prior_zero_norm"],
    )
    def test_posinf_is_not_laundered_into_a_finite_posterior(self, prior, ll):
        prior, ll = _as_f32(prior), _as_f32(ll)

        posterior, log_norm = _condition_on(jnp.asarray(prior), jnp.asarray(ll))

        assert not jnp.all(jnp.isfinite(posterior)), posterior
        assert not jnp.isfinite(log_norm)


@pytest.mark.unit
class TestConditionOnGradients:
    """Gradients through the Bayes update match the closed forms.

    For ``log_norm = log sum_k p_k exp(ll_k)``: ``d log_norm / d ll_i =
    posterior_i``, and ``d posterior_i / d ll_k = posterior_i (delta_ik -
    posterior_k)``. The stabilizing shifts must not distort these — in
    particular at exact ties, where a clip or ``max`` splits its gradient.
    """

    @pytest.mark.parametrize(
        "prior, ll",
        [
            ([0.2, 0.8], [0.0, 0.0]),  # tied likelihoods
            ([0.3, 0.7], [-1.0, -2.0]),
            ([0.5, 0.5, 0.0], [-1000.0, -1001.0, 0.0]),  # unreachable maximum
        ],
        ids=["tied", "distinct", "unreachable_maximum"],
    )
    def test_gradients_match_closed_form(self, prior, ll):
        prior, ll = _as_f32(prior), _as_f32(ll)
        posterior, _ = bayes_reference(prior, ll)

        evidence_grad = jax.grad(lambda x: _condition_on(jnp.asarray(prior), x)[1])(
            jnp.asarray(ll)
        )
        posterior_jac = jax.jacobian(lambda x: _condition_on(jnp.asarray(prior), x)[0])(
            jnp.asarray(ll)
        )

        np.testing.assert_allclose(
            np.asarray(evidence_grad), posterior, rtol=RTOL, atol=ATOL
        )
        expected_jac = np.diag(posterior) - np.outer(posterior, posterior)
        np.testing.assert_allclose(
            np.asarray(posterior_jac), expected_jac, rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize(
        "prior, ll",
        [
            ([0.3, 0.7], [-1.0, -2.0]),
            ([1.0, 0.0], [0.0, -2.0]),  # zero-prior state: d log Z / d p_1 = e^-2 / Z
            ([0.5, 0.5, 0.0], [-1000.0, -1001.0, 0.0]),  # unreachable maximum
            # exp(exponent) overflows float32 but exp(exponent) / Z = e^88.5 does not
            ([0.5, 0.5, 0.0], [-88.5, -88.5, 0.0]),
        ],
        ids=[
            "distinct",
            "zero_prior",
            "unreachable_maximum",
            "representable_after_norm",
        ],
    )
    def test_evidence_gradient_wrt_prior_matches_closed_form(self, prior, ll):
        """``d log_norm / d p_i = exp(ll_i) / Z`` — also at ``p_i == 0``.

        Gradients with respect to the transition matrix flow through the
        predicted probabilities, so this must hold at zero entries too.
        """
        prior, ll = _as_f32(prior), _as_f32(ll)
        _, log_evidence = bayes_reference(prior, ll)
        expected = np.exp(ll.astype(np.float64) - log_evidence)

        prior_grad = np.asarray(
            jax.grad(lambda p: _condition_on(p, jnp.asarray(ll))[1])(jnp.asarray(prior))
        )

        # A derivative that overflows the dtype (an unreachable state whose
        # likelihood dwarfs the reachable ones) is saturated to a finite value
        # rather than becoming inf/NaN; only representable entries are compared.
        representable = expected < np.finfo(np.float32).max
        np.testing.assert_allclose(
            prior_grad[representable], expected[representable], rtol=RTOL, atol=ATOL
        )
        assert np.all(np.isfinite(prior_grad[~representable]))
        assert np.all(prior_grad[~representable] > 0)

    def test_transition_gradient_through_zero_predicted_probability(self):
        """``d evidence / d T[0, 1]`` where ``T[0, 1] == 0`` equals the closed form.

        With ``init = [1, 0]`` and identity ``T``, the second step's predicted
        distribution is ``[1, 0]``; nudging ``T[0, 1]`` moves mass onto state 1,
        so ``d log Z_2 / d T[0, 1] = exp(ll_2[1]) / Z_2 = exp(ll_2[1] - ll_2[0])``.
        """
        init = jnp.asarray([1.0, 0.0])
        transition = jnp.eye(2)
        ll = jnp.asarray([[0.0, 0.0], [0.0, -2.0]])

        # The public ``filter`` runs host-side diagnostics that need a concrete
        # evidence value, so differentiate the jitted core directly.
        grad = jax.grad(lambda t: core_module._filter_jit(init, t, ll)[0][0])(
            transition
        )

        assert float(grad[0, 1]) == pytest.approx(float(np.exp(-2.0)), rel=RTOL)

    @pytest.mark.parametrize("offset", [0.0, 10000.0, -10000.0, -1000000.0])
    def test_joint_jvp_matches_reference(self, offset):
        """Perturb priors and likelihoods together, including zero prior mass."""
        prior = _as_f32([0.2, 0.8, 0.0])
        ll = _as_f32([0.0, -1.0, 1.0]) + np.float32(offset)
        prior_dot = _as_f32([-0.3, 0.1, 0.2])
        ll_dot = _as_f32([0.7, -0.4, 0.3])
        posterior, evidence = bayes_reference(prior, ll)
        ratio = np.exp(ll.astype(np.float64) - evidence)
        weighted_dot = ratio * prior_dot + posterior * ll_dot
        expected_evidence_dot = weighted_dot.sum()
        expected_posterior_dot = weighted_dot - posterior * expected_evidence_dot

        _, (posterior_dot, evidence_dot) = jax.jit(
            lambda p, likelihood, dp, dl: jax.jvp(
                _condition_on, (p, likelihood), (dp, dl)
            )
        )(*(jnp.asarray(x) for x in (prior, ll, prior_dot, ll_dot)))

        np.testing.assert_allclose(
            np.asarray(posterior_dot), expected_posterior_dot, rtol=RTOL, atol=ATOL
        )
        assert float(evidence_dot) == pytest.approx(
            expected_evidence_dot, rel=RTOL, abs=ATOL
        )

    def test_batched_prior_gradient_with_large_likelihood_offsets(self):
        """Center likelihoods before calculating derivatives in float32."""
        prior = _as_f32([0.2, 0.8, 0.0])
        likelihoods = (
            _as_f32([0.0, -1.0, 1.0]) + _as_f32([0.0, 10000.0, -10000.0])[:, None]
        )
        expected = []
        for ll in likelihoods:
            _, evidence = bayes_reference(prior, ll)
            expected.append(np.exp(ll.astype(np.float64) - evidence))

        gradient = jax.grad(lambda p, ll: _condition_on(p, ll)[1])
        actual = jax.jit(jax.vmap(gradient, in_axes=(None, 0)))(
            jnp.asarray(prior), jnp.asarray(likelihoods)
        )

        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize("offset", [0.0, 10000.0, -1000000.0])
    def test_second_derivatives_at_zero_prior_match_closed_form(self, offset):
        """Differentiating the derivative rule preserves prior/likelihood terms."""
        prior = _as_f32([0.2, 0.8, 0.0])
        ll = _as_f32([0.0, -1.0, 1.0]) + np.float32(offset)
        posterior, evidence = bayes_reference(prior, ll)
        ratio = np.exp(ll.astype(np.float64) - evidence)
        gradient = jax.grad(
            lambda p, likelihood: _condition_on(p, likelihood)[1], argnums=0
        )

        prior_hessian, mixed_hessian = jax.jit(jax.jacfwd(gradient, argnums=(0, 1)))(
            jnp.asarray(prior), jnp.asarray(ll)
        )

        np.testing.assert_allclose(
            np.asarray(prior_hessian), -np.outer(ratio, ratio), rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            np.asarray(mixed_hessian),
            np.diag(ratio) - np.outer(ratio, posterior),
            rtol=RTOL,
            atol=ATOL,
        )

    @pytest.mark.parametrize("ll", [[-np.inf, -np.inf], [-np.inf, 0.0]])
    def test_zero_support_fallback_has_identity_prior_derivative(self, ll):
        """Both differentiation directions follow the selected prior fallback."""
        prior = jnp.asarray([1.0, 0.0])
        ll = jnp.asarray(ll)

        def posterior_func(p):
            return _condition_on(p, ll)[0]

        for differentiate in (jax.jacfwd, jax.jacrev):
            actual = jax.jit(differentiate(posterior_func))(prior)
            np.testing.assert_array_equal(np.asarray(actual), np.eye(2))

        evidence_grad = jax.jit(jax.grad(lambda p: _condition_on(p, ll)[1]))(prior)
        np.testing.assert_array_equal(np.asarray(evidence_grad), np.zeros(2))

    def test_prior_gradient_with_integer_likelihoods(self):
        """Nondifferentiable likelihood inputs contribute a zero tangent."""
        prior = jnp.asarray([0.3, 0.7])
        ll = jnp.asarray([0, -2], dtype=jnp.int32)
        _, evidence = bayes_reference(np.asarray(prior), np.asarray(ll))
        expected = np.exp(np.asarray(ll, dtype=np.float64) - evidence)

        actual = jax.jit(jax.grad(lambda p: _condition_on(p, ll)[1]))(prior)

        np.testing.assert_allclose(np.asarray(actual), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.unit
class TestMultiStepEnumerationReference:
    """Filtered/predicted/smoothed marginals and evidence vs path enumeration."""

    def test_stationary_filter_and_smoother(self):
        initial, transitions, ll = _tiny_model()
        ref = enumerate_reference(
            initial, np.broadcast_to(transitions[0], transitions.shape), ll
        )

        (evidence, _), (filtered, predicted) = filter(
            jnp.asarray(initial), jnp.asarray(transitions[0]), jnp.asarray(ll)
        )
        smoothed = smoother(jnp.asarray(transitions[0]), filtered)

        np.testing.assert_allclose(np.asarray(filtered), ref["filtered"], atol=1e-5)
        np.testing.assert_allclose(np.asarray(predicted), ref["predicted"], atol=1e-5)
        np.testing.assert_allclose(np.asarray(smoothed), ref["smoothed"], atol=1e-5)
        assert float(evidence) == pytest.approx(ref["log_evidence"], abs=1e-4)

    def test_covariate_dependent_filter_and_smoother(self):
        initial, discrete, continuous, state_ind, ll, expanded = _covariate_model()
        ref = enumerate_reference(initial, expanded, ll)

        (evidence, _), (filtered, predicted) = filter_covariate_dependent(
            jnp.asarray(initial),
            jnp.asarray(discrete),
            jnp.asarray(continuous),
            jnp.asarray(state_ind),
            jnp.asarray(ll),
        )
        smoothed = smoother_covariate_dependent(
            jnp.asarray(discrete),
            jnp.asarray(continuous),
            jnp.asarray(state_ind),
            filtered,
        )

        np.testing.assert_allclose(np.asarray(filtered), ref["filtered"], atol=1e-5)
        np.testing.assert_allclose(np.asarray(predicted), ref["predicted"], atol=1e-5)
        np.testing.assert_allclose(np.asarray(smoothed), ref["smoothed"], atol=1e-5)
        assert float(evidence) == pytest.approx(ref["log_evidence"], abs=1e-4)

    def test_constant_covariate_matches_stationary_expanded(self):
        initial, discrete, continuous, state_ind, ll, expanded = _covariate_model()
        discrete_const = np.broadcast_to(discrete[0], discrete.shape).copy()
        expanded_const = expanded[0]

        (ev_cov, _), (filt_cov, pred_cov) = filter_covariate_dependent(
            jnp.asarray(initial),
            jnp.asarray(discrete_const),
            jnp.asarray(continuous),
            jnp.asarray(state_ind),
            jnp.asarray(ll),
        )
        (ev_stat, _), (filt_stat, pred_stat) = filter(
            jnp.asarray(initial), jnp.asarray(expanded_const), jnp.asarray(ll)
        )

        np.testing.assert_allclose(
            np.asarray(filt_cov), np.asarray(filt_stat), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(pred_cov), np.asarray(pred_stat), rtol=1e-5, atol=1e-6
        )
        assert float(ev_cov) == pytest.approx(float(ev_stat), rel=1e-6)

    @pytest.mark.parametrize("path", ["stationary", "covariate"])
    @pytest.mark.parametrize("n_chunks", [1, 3, 5])
    @pytest.mark.parametrize("cache_log_likelihoods", [True, False])
    def test_chunked_drivers_match_enumeration(
        self, path, n_chunks, cache_log_likelihoods
    ):
        # Missing rows carry no evidence (log-likelihood 0), in both the cached
        # array and the callback. The callback returns rows by global index,
        # keeping this independent of the spike-binning code.
        n_time = 5
        is_missing = np.zeros(n_time, dtype=bool)
        is_missing[[1, 3]] = True
        time = np.arange(n_time)

        if path == "stationary":
            initial, transitions, ll = _tiny_model(n_time=n_time)
            state_ind = np.arange(ll.shape[1])
            expanded = np.broadcast_to(transitions[0], transitions.shape)
        else:
            initial, discrete, continuous, state_ind, ll, expanded = _covariate_model(
                n_time=n_time
            )
        ll = ll.copy()
        ll[is_missing] = 0.0
        ref = enumerate_reference(initial, expanded, ll)

        def log_likelihood_func(time_idx, is_missing=None):
            return ll[np.asarray(time_idx, dtype=int)]

        common = {
            "time": time,
            "state_ind": state_ind,
            "initial_distribution": initial,
            "log_likelihood_func": log_likelihood_func,
            "log_likelihood_args": (),
            "is_missing": is_missing,
            "n_chunks": n_chunks,
            "log_likelihoods": ll if cache_log_likelihoods else None,
            "cache_log_likelihoods": cache_log_likelihoods,
        }
        if path == "stationary":
            result = chunked_filter_smoother(transition_matrix=transitions[0], **common)
        else:
            result = chunked_filter_smoother_covariate_dependent(
                discrete_transition_matrix=discrete,
                continuous_transition_matrix=continuous,
                **common,
            )
        acausal, _, marginal, _, _, _, causal, predictive = result

        np.testing.assert_allclose(acausal, ref["smoothed"], atol=1e-5)
        np.testing.assert_allclose(causal, ref["filtered"], atol=1e-5)
        np.testing.assert_allclose(predictive, ref["predicted"], atol=1e-5)
        assert marginal == pytest.approx(ref["log_evidence"], abs=1e-4)


@pytest.mark.unit
class TestDtypes:
    """Outputs carry the requested dtype; float64 is not silently truncated."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_filter_output_dtype_matches_request(self, dtype):
        if dtype == jnp.float64 and not jax.config.jax_enable_x64:
            pytest.skip("float64 requires JAX_ENABLE_X64=1")
        prior, ll = (np.asarray(v, dtype=dtype) for v in CASE_UNREACHABLE_MAXIMUM)
        expected_posterior, expected_evidence = bayes_reference(prior, ll)

        (evidence, _), (posterior, _) = filter(
            jnp.asarray(prior),
            jnp.eye(len(prior), dtype=dtype),
            jnp.asarray(ll[None, :]),
        )

        assert posterior.dtype == dtype
        assert evidence.dtype == dtype
        np.testing.assert_allclose(
            np.asarray(posterior)[0], expected_posterior, rtol=RTOL, atol=ATOL
        )
        assert float(evidence) == pytest.approx(expected_evidence, rel=1e-6)
