"""Estimation of discrete state transitions from expected transition counts.

The M-step of a discrete transition: the stationary estimate with Dirichlet
priors and stickiness, and the covariate-dependent estimate, a penalized
(optionally spline-smoothed) multinomial regression of each transition row on
the design matrix of a formula, with the smoothness strengths fixed or chosen
by restricted maximum likelihood. :mod:`non_local_detector.discrete_state_transitions`
defines the transitions themselves; this module imports from it, never the
reverse.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from patsy import DesignMatrix
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar
from scipy.special import xlogy

from non_local_detector.discrete_state_transitions import (
    _float64_jax,
    _require_spline_term,
    _transitions_on_design,
    _validate_smoothness,
    get_transition_prior,
    jax_centered_log_softmax_forward,
)

logger = logging.getLogger(__name__)

# ``(design, response)`` blocks of one transition row; see ``_row_problem``.
_Blocks: TypeAlias = list[tuple[jax.Array, jax.Array]]


# Transitions per block of the covariate M-step loss and its log prior, so
# their float64 temporaries stay bounded however long the recording is.
_COVARIATE_BLOCK_ROWS = 65_536


def _assert_map_compatible_alpha(alpha: np.ndarray) -> None:
    if np.any(alpha < 1.0):
        msg = (
            "concentration and stickiness must produce prior parameters >= 1.0 "
            "for this MAP transition update"
        )
        raise ValueError(msg)


class CovariateTransitionFit(NamedTuple):
    """A covariate-dependent discrete transition.

    It is fit by :func:`_fit_covariate_transition`.

    Attributes
    ----------
    coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
    transition : np.ndarray, shape (n_time, n_states, n_states)
    smoothness : np.ndarray, shape (n_states, n_terms)
        Curvature-penalty strength of each row for each spline term.
    covariance : np.ndarray, shape (n_states, n_parameters, n_parameters)
        Laplace (posterior) covariance of each row's flattened coefficients,
        ``n_parameters = n_coefficients * (n_states - 1)``: the inverse
        Hessian of the row's penalized loss at the fit. Zero for frozen rows;
        NaN for a row that is not identified (no transitions, or a singular
        Hessian).
    """

    coefficients: np.ndarray
    transition: np.ndarray
    smoothness: np.ndarray
    covariance: np.ndarray | None


# REML searches the curvature-penalty strength over this grid of log10
# strengths, then refines the best grid point within one decade to this
# log10 tolerance. Above 1e6 the fitted curve is already its linear limit
# (transitions move by < 0.002) and the LAML is flat (within 4e-4 nats to
# 1e7 in one test). From about 3e7 the inner fit stops converging (its
# Hessian's condition number passes 1e14), so the LAML it gives is wrong:
# by 0.2 nats at 3e8 and 37 at 1e10 in that test.
# The grid reaches 1e-12 because the penalty is on the covariate rescaled
# between the spline's boundaries: a boundary far beyond the data (e.g. set
# by tracking glitches) moves the useful strengths down by decades, and the
# LAML can have maxima there and near 1 (0.2 nats apart in one test).
# Per-term searches around a current value scan down to the floor too.
_REML_LOG10_FLOOR = -12.0
_REML_LOG10_GRID = np.arange(_REML_LOG10_FLOOR, 6.5, 1.0)
_REML_LOG10_TOLERANCE = 0.02
# With several spline terms, REML then refines each term's strength in turn,
# in at most this many sweeps over the terms.
_REML_MAX_SWEEPS = 5


def estimate_non_stationary_state_transition_from_responses(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
    transition_regularization: float = 0.1,
    optimization_method: str = "trust-exact",
    maxiter: int | None = 100,
    *,
    disp: bool = False,
    frozen_rows: np.ndarray | None = None,
    smoothness: float | np.ndarray | str = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a non-stationary transition model from expected counts.

    Each row maximizes its expected complete-data log-likelihood, summed over
    time bins, plus a Dirichlet prior counted once per row (as for stationary
    transitions, so ``stickiness`` means the same pseudo-counts for both), an
    L2 penalty and an optional curvature penalty, in float64. See
    :func:`dirichlet_neg_log_likelihood`.

    Parameters
    ----------
    transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Initial estimate of the transition coefficients.
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
        Covariate design matrix.
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
        Expected transition counts for each time and transition row.
    concentration : float, optional
        Dirichlet prior concentration parameter (uniform part), by default 1.0.
    stickiness : float or np.ndarray, optional
        Dirichlet prior stickiness parameter (diagonal enhancement), by default 0.0.
    transition_regularization : float, optional
        L2 penalty on coefficients (excluding intercept), by default 0.1, as
        for the detectors' ``discrete_transition_regularization``.
    optimization_method : str, optional
        Optimization method for `scipy.optimize.minimize`, by default
        "trust-exact": Newton steps with the exact float64 Hessian, which
        converge in a few iterations on spline designs whose Hessians have
        condition numbers of 1e5 to 1e7, where "L-BFGS-B" stops short.
    maxiter : int, optional
        Maximum iterations for optimizer, by default 100.
    disp : bool, optional
        Display optimizer convergence messages, by default False.
    frozen_rows : np.ndarray of bool, shape (n_states,), optional
        Rows that keep their ``transition_coefficients`` instead of being
        estimated. None (default) estimates every row.
    smoothness : float, np.ndarray, or "reml", optional
        Strength of the curvature penalty on each ``quantile_bs`` and
        ``cyclic_bs`` term: one for every row and term, one per row (shape
        (n_states,)), or one per row and term (shape (n_states, n_terms)); by
        default 0.0 (none). "reml" chooses each row's strength for each term
        by maximizing the row's Laplace approximate marginal likelihood.
    smoothness_matrix : np.ndarray, optional
        Curvature penalties from :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients), needed when
        ``smoothness`` is not 0.

    Returns
    -------
    estimated_transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Optimized transition coefficients.
    estimated_transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Resulting non-stationary transition matrix.
    """
    fit = _fit_covariate_transition(
        transition_coefficients,
        design_matrix,
        response,
        concentration=concentration,
        stickiness=stickiness,
        transition_regularization=transition_regularization,
        smoothness=smoothness,
        smoothness_matrix=smoothness_matrix,
        optimization_method=optimization_method,
        maxiter=maxiter,
        disp=disp,
        frozen_rows=frozen_rows,
        compute_covariance=False,
    )
    return fit.coefficients, fit.transition


def _term_penalties(
    smoothness_matrix: np.ndarray | None, n_coefficients: int
) -> np.ndarray:
    """Curvature penalties as a stack, one per spline term.

    Parameters
    ----------
    smoothness_matrix : np.ndarray or None
        A stack from :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients); one term's matrix, shape
        (n_coefficients, n_coefficients); or None (no spline terms).
    n_coefficients : int
        Number of coefficients per logit (columns of the design matrix).

    Returns
    -------
    penalties : np.ndarray, shape (n_terms, n_coefficients, n_coefficients)
    """
    if smoothness_matrix is None:
        return np.zeros((0, n_coefficients, n_coefficients))
    return np.reshape(
        np.asarray(smoothness_matrix, dtype=np.float64),
        (-1, n_coefficients, n_coefficients),
    )


def _row_strengths(
    smoothness: float | np.ndarray, n_states: int, n_terms: int
) -> np.ndarray:
    """Broadcast curvature-penalty strengths to (n_states, n_terms).

    Accepts a scalar for every row and term, shape (n_states,) per row, or
    (n_states, n_terms).
    """
    values = np.asarray(smoothness, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return np.broadcast_to(values, (n_states, n_terms)).astype(np.float64)


def _row_smoothness(
    smoothness: float | np.ndarray | str,
    initial_smoothness: np.ndarray | None,
    smoothness_matrix: np.ndarray | None,
    n_states: int,
) -> tuple[np.ndarray, bool]:
    """Return validated per-row, per-term curvature-penalty strengths to start from.

    Returns
    -------
    smoothness : np.ndarray, shape (n_states, n_terms)
        The fixed strengths, or the REML starting strengths
        (``initial_smoothness``, else 1).
    estimate : bool
        Whether REML estimates the strengths.
    """
    estimate = _validate_smoothness(smoothness)
    _require_spline_term(smoothness, smoothness_matrix)
    if estimate:
        values = 1.0 if initial_smoothness is None else initial_smoothness
        _validate_smoothness(values)
    else:
        values = cast(float | np.ndarray, smoothness)
    n_terms = (
        0
        if smoothness_matrix is None
        else _term_penalties(smoothness_matrix, np.shape(smoothness_matrix)[-1]).shape[
            0
        ]
    )
    return _row_strengths(values, n_states, n_terms), estimate


def _fit_covariate_transition(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float = 1.0,
    stickiness: float | np.ndarray = 0.0,
    transition_regularization: float = 0.1,
    smoothness: float | np.ndarray | str = 0.0,
    smoothness_matrix: np.ndarray | None = None,
    initial_smoothness: np.ndarray | None = None,
    optimization_method: str = "trust-exact",
    maxiter: int | None = 100,
    *,
    disp: bool = False,
    frozen_rows: np.ndarray | None = None,
    compute_covariance: bool = True,
) -> CovariateTransitionFit:
    """Fit a covariate-dependent transition, its smoothness and covariance.

    With ``smoothness="reml"``, each row's curvature-penalty strengths, one
    per spline term, maximize the row's Laplace approximate marginal
    likelihood (LAML, the
    REML criterion of Wood, 2011, J. R. Stat. Soc. B 73:3 for non-Gaussian
    models) ``-L(beta) + log|Q|_+ / 2 - log|A| / 2``, with ``beta`` the
    penalized fit, ``L`` the penalized loss, ``Q`` the prior precision (twice
    the penalty) and ``A`` the Hessian of ``L``, given the E-step's expected
    transitions. The LAML can have more than one local maximum (the L2
    penalty and the curvature penalty both shrink the curve), so the
    strength is searched over a grid of decades and then refined. The L2
    penalty stays fixed and keeps rare transitions from running off along the
    curvature penalty's unpenalized linear trends. See :func:`_reml_row` for
    the search over several terms' strengths.

    Parameters
    ----------
    transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Initial estimate of the transition coefficients.
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
        Covariate design matrix.
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
        Expected transition counts for each time and transition row.
    concentration : float, optional
        Dirichlet prior concentration parameter (uniform part).
    stickiness : float or np.ndarray, optional
        Dirichlet prior stickiness parameter (diagonal enhancement).
    transition_regularization : float, optional
        L2 penalty on coefficients (excluding intercept).
    smoothness : float, np.ndarray, or "reml", optional
        Strength of the curvature penalty on each spline term, or "reml" to
        choose it per row and term; see
        :func:`estimate_non_stationary_state_transition_from_responses`.
    smoothness_matrix : np.ndarray, optional
        Curvature penalties from :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients).
    optimization_method : str, optional
        Optimization method for `scipy.optimize.minimize`.
    maxiter : int, optional
        Maximum iterations for the optimizer; None uses the optimizer default.
    disp : bool, optional
        Display optimizer convergence messages.
    frozen_rows : np.ndarray of bool, shape (n_states,), optional
        Rows that keep their ``transition_coefficients`` instead of being
        estimated. None estimates every row.
    initial_smoothness : np.ndarray, shape (n_states, n_terms), optional
        Strengths kept by rows REML does not fit (frozen rows), e.g. the
        previous EM iteration's; by default 1. REML searches its full range
        for every fitted row. Ignored for fixed ``smoothness``.
    compute_covariance : bool, optional
        Whether to compute the fit's covariance, by default True. EM skips
        it in every M-step and computes it once, after its last, with
        :func:`_covariate_transition_covariance`.

    Returns
    -------
    fit : CovariateTransitionFit
        ``covariance`` is None without ``compute_covariance``.
    """
    n_coefficients, n_states = transition_coefficients.shape[:2]
    estimated_transition_coefficients = np.zeros(
        (n_coefficients, n_states, (n_states - 1))
    )

    design_matrix = np.asarray(design_matrix, dtype=np.float64)
    n_time = design_matrix.shape[0]
    estimated_transition_matrix = np.zeros((n_time, n_states, n_states))

    alpha = get_transition_prior(concentration, stickiness, n_states)
    _assert_map_compatible_alpha(alpha)
    row_smoothness, estimate_smoothness = _row_smoothness(
        smoothness, initial_smoothness, smoothness_matrix, n_states
    )

    options: dict[str, Any] = {}
    if maxiter is not None:
        options["maxiter"] = maxiter
    # scipy deprecated `disp`/`iprint` for L-BFGS-B (removal in 1.18), so
    # only forward `disp` to methods that still accept it.
    if optimization_method != "L-BFGS-B":
        options["disp"] = disp
    else:
        # If L-BFGS-B is requested, stop on the projected gradient, not on
        # the relative change of the summed (large) objective.
        options["ftol"] = 0.0
        options["gtol"] = 1e-7

    penalties = _term_penalties(smoothness_matrix, n_coefficients)

    def penalty(strengths: np.ndarray) -> np.ndarray:
        return _penalty_matrix(
            n_coefficients, transition_regularization, strengths, penalties
        )

    with _float64_jax():
        # Every row's fit shares the design matrix's blocks.
        design_blocks = _transition_blocks(design_matrix[:-1])
        for from_state, row_alpha in enumerate(alpha):
            row_coefficients = transition_coefficients[:, from_state, :]
            if frozen_rows is None or not frozen_rows[from_state]:
                blocks, prior_per_bin = _row_problem(
                    design_blocks, response[:, from_state, :], row_alpha
                )

                def fit_row(
                    start: np.ndarray,
                    strengths: np.ndarray,
                    blocks: _Blocks = blocks,
                    prior_per_bin: jax.Array = prior_per_bin,
                    from_state: int = from_state,
                    *,
                    warn: bool = True,
                ) -> np.ndarray:
                    return _fit_covariate_row(
                        start,
                        blocks,
                        prior_per_bin,
                        penalty(strengths),
                        optimization_method,
                        options,
                        from_state,
                        warn=warn,
                    )

                strength = row_smoothness[from_state]
                if estimate_smoothness and blocks:
                    fitted, strength = _reml_row(
                        row_coefficients,
                        fit_row,
                        blocks,
                        prior_per_bin,
                        penalty,
                        penalties,
                        transition_regularization,
                    )
                    # The search's fits each start from a neighbouring one,
                    # so check the result against the row's start too.
                    if _blocked_loss(
                        fitted, blocks, prior_per_bin, penalty(strength)
                    ) > _blocked_loss(
                        row_coefficients, blocks, prior_per_bin, penalty(strength)
                    ):
                        logger.warning(
                            "Transition optimization did not lower the loss for "
                            "state %d at its REML smoothness. Keeping previous "
                            "coefficients.",
                            from_state,
                        )
                        fitted = row_coefficients
                    elif (
                        _newton_decrement(
                            np.ravel(fitted), blocks, prior_per_bin, penalty(strength)
                        )
                        > 1e-6
                    ):
                        logger.warning(
                            "Transition optimization did not converge for state %d "
                            "at its REML smoothness. Using improved coefficients.",
                            from_state,
                        )
                else:
                    fitted = fit_row(row_coefficients, strength)
                row_coefficients = np.reshape(fitted, row_coefficients.shape)
                row_smoothness[from_state] = strength
            estimated_transition_coefficients[:, from_state, :] = row_coefficients
            device_coefficients = jnp.asarray(row_coefficients, dtype=jnp.float64)
            for start in range(0, n_time, _COVARIATE_BLOCK_ROWS):
                rows = slice(start, start + _COVARIATE_BLOCK_ROWS)
                estimated_transition_matrix[rows, from_state, :] = jnp.exp(
                    jax_centered_log_softmax_forward(
                        jnp.asarray(design_matrix[rows], dtype=jnp.float64)
                        @ device_coefficients
                    )
                )

    return CovariateTransitionFit(
        estimated_transition_coefficients,
        estimated_transition_matrix,
        row_smoothness,
        _covariate_transition_covariance(
            estimated_transition_coefficients,
            design_matrix,
            response,
            concentration,
            stickiness,
            transition_regularization,
            row_smoothness,
            smoothness_matrix,
            frozen_rows,
        )
        if compute_covariance
        else None,
    )


def _covariate_transition_covariance(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    concentration: float,
    stickiness: float | np.ndarray,
    transition_regularization: float,
    smoothness: np.ndarray,
    smoothness_matrix: np.ndarray | None,
    frozen_rows: np.ndarray | None,
) -> np.ndarray:
    """Return the Laplace covariance of each row's coefficients at a fit.

    This is the inverse Hessian of the row's penalized loss.

    Parameters
    ----------
    transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        The fitted coefficients.
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
        Covariate design matrix.
    response : np.ndarray, shape (n_time - 1, n_states, n_states)
        Expected transition counts for each time and transition row.
    concentration : float
        Dirichlet prior concentration parameter (uniform part).
    stickiness : float or np.ndarray
        Dirichlet prior stickiness parameter (diagonal enhancement).
    transition_regularization : float
        L2 penalty on coefficients (excluding intercept).
    smoothness_matrix : np.ndarray or None
        Curvature penalties, shape (n_terms, n_coefficients, n_coefficients).
    frozen_rows : np.ndarray of bool, shape (n_states,), or None
        Rows whose covariance is left at zero.
    smoothness : np.ndarray, shape (n_states, n_terms)
        The fit's strengths, :attr:`CovariateTransitionFit.smoothness`.

    Returns
    -------
    covariance : np.ndarray, shape (n_states, n_parameters, n_parameters)
        As :attr:`CovariateTransitionFit.covariance`.
    """
    n_coefficients, n_states, n_free = transition_coefficients.shape
    n_parameters = n_coefficients * n_free
    covariance = np.zeros((n_states, n_parameters, n_parameters))
    alpha = get_transition_prior(concentration, stickiness, n_states)
    penalties = _term_penalties(smoothness_matrix, n_coefficients)
    with _float64_jax():
        design_blocks = _transition_blocks(
            np.asarray(design_matrix, dtype=np.float64)[:-1]
        )
        for from_state, row_alpha in enumerate(alpha):
            if frozen_rows is not None and frozen_rows[from_state]:
                continue
            blocks, prior_per_bin = _row_problem(
                design_blocks, response[:, from_state, :], row_alpha
            )
            # Without transitions (a single time bin) the row is not
            # identified.
            covariance[from_state] = (
                _inverse_symmetric(
                    _blocked_hessian(
                        transition_coefficients[:, from_state, :],
                        blocks,
                        prior_per_bin,
                        _penalty_matrix(
                            n_coefficients,
                            transition_regularization,
                            smoothness[from_state],
                            penalties,
                        ),
                    )
                )
                if blocks
                else np.nan
            )
    return covariance


def _max_transition_change(
    design_matrix: np.ndarray,
    old_coefficients: np.ndarray,
    new_coefficients: np.ndarray,
) -> float:
    """Return the largest change of any covariate-dependent transition probability.

    The change is between two sets of coefficients.

    Evaluated a block of time bins at a time, so no copy of the
    (n_time, n_states, n_states) transitions is held.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
        Covariate design matrix.
    old_coefficients, new_coefficients : np.ndarray
        Shape (n_coefficients, n_states, n_states - 1).

    Returns
    -------
    change : float
    """
    design_matrix = np.asarray(design_matrix, dtype=np.float64)
    change = 0.0
    for start in range(0, design_matrix.shape[0], _COVARIATE_BLOCK_ROWS):
        design = design_matrix[start : start + _COVARIATE_BLOCK_ROWS]
        moved = _transitions_on_design(design, new_coefficients) - (
            _transitions_on_design(design, old_coefficients)
        )
        change = max(change, float(np.max(np.abs(moved))))
    return change


def _inverse_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Invert a symmetric positive definite matrix; NaN if it is singular.

    A singular penalized Hessian leaves some direction of the coefficients
    unidentified (e.g. a row without expected transitions and with a flat
    prior), so its covariance is unknown; a pseudo-inverse would give that
    direction zero variance and the intervals false certainty.
    """
    if matrix.shape[0] == 0:
        # A single state has no free coefficients (scipy < 1.13's LAPACK
        # wrappers reject empty matrices).
        return np.zeros_like(matrix)
    try:
        factor = cho_factor(matrix)
    except np.linalg.LinAlgError:
        return np.full_like(matrix, np.nan)
    return cast(np.ndarray, cho_solve(factor, np.eye(matrix.shape[0])))


def _precision_log_pdet(
    penalties: np.ndarray, strengths: np.ndarray, l2_penalty: float
) -> float:
    """Log pseudo-determinant of one logit's prior precision.

    The precision is twice :func:`_penalty_matrix`; the pseudo-determinant
    is over its penalized directions, whose number the strengths do not
    change: every coefficient but the intercept with an L2 penalty, else the
    curvature penalties' ranges. The spline terms sit on disjoint columns, so
    the precision is block diagonal and its determinant factors by term.
    Taking the largest eigenvalues of the summed precision instead would let
    one strongly penalized term's round-off null directions outrank a weakly
    penalized term's genuine ones.

    Parameters
    ----------
    penalties : np.ndarray, shape (n_terms, n_coefficients, n_coefficients)
        Each spline term's curvature penalty.
    strengths : np.ndarray, shape (n_terms,)
        Positive strength of each term.
    l2_penalty : float
        The L2 penalty on every coefficient but the intercept.

    Returns
    -------
    log_pdet : float
        Without an L2 penalty, up to a constant that does not depend on the
        strengths.
    """
    n_coefficients = penalties.shape[-1]
    log_pdet = 0.0
    n_term_columns = 0
    for strength, term in zip(strengths, penalties, strict=True):
        columns = np.flatnonzero(np.any(term != 0.0, axis=0))
        n_term_columns += columns.size
        if l2_penalty > 0.0:
            eigenvalues = np.linalg.eigvalsh(term[np.ix_(columns, columns)])
            log_pdet += np.sum(
                np.log(2.0 * (l2_penalty + strength * np.maximum(eigenvalues, 0.0)))
            )
        else:
            # A range stretched by outliers puts genuine eigenvalues of the
            # penalty at 1e-16 of its largest, among its null directions'
            # round-off. Scaled by its diagonal (a congruence, which keeps the
            # rank) they sit near 1e-2, far above round-off. The log
            # pseudo-determinant of the scaled penalty plus that of the
            # scaling is exact for a full-rank penalty and otherwise off by a
            # constant in the strength, which the LAML's comparisons cancel.
            block = term[np.ix_(columns, columns)]
            diagonal = np.diag(block)
            scaled = block / np.sqrt(np.outer(diagonal, diagonal))
            rank = np.linalg.matrix_rank(scaled, hermitian=True)
            scaled_eigenvalues = np.linalg.eigvalsh(scaled)
            log_pdet += (
                rank * np.log(2.0 * strength)
                + np.sum(np.log(scaled_eigenvalues[scaled_eigenvalues.size - rank :]))
                + np.sum(np.log(diagonal))
            )
    if l2_penalty > 0.0:
        # Penalized columns outside every spline term carry the L2 penalty.
        log_pdet += (n_coefficients - 1 - n_term_columns) * np.log(2.0 * l2_penalty)
    return float(log_pdet)


def _reml_row(
    coefficients: np.ndarray,
    fit_row: Callable[..., np.ndarray],
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: Callable[[np.ndarray], np.ndarray],
    penalties: np.ndarray,
    l2_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One row's curvature-penalty strengths maximizing its LAML.

    The strengths of all spline terms are first searched together, as one
    shared strength; with several terms each is then searched in turn,
    around its current value with the others fixed, until none moves by more
    than the search tolerance. A search keeps its starting value unless it
    finds a higher LAML, so the LAML never decreases. Call with JAX in
    float64. See :func:`_fit_covariate_transition`.

    Parameters
    ----------
    coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
        Starting coefficients.
    fit_row : callable
        ``fit_row(start, strengths, warn=False)`` fits the row's coefficients
        without logging.
    blocks, prior_per_bin
        From :func:`_row_problem`.
    penalty : callable
        ``penalty(strengths)``, the row's :func:`_penalty_matrix`.
    penalties : np.ndarray, shape (n_terms, n_coefficients, n_coefficients)
        Each spline term's curvature penalty.
    l2_penalty : float
        The L2 penalty on every coefficient but the intercept.

    Returns
    -------
    coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
        The fit at the returned strengths.
    strengths : np.ndarray, shape (n_terms,)
    """
    n_terms, n_coefficients = penalties.shape[:2]
    n_free = np.size(coefficients) // n_coefficients
    fits: dict[tuple[float, ...], tuple[np.ndarray, float]] = {}

    def laml(log10_strengths: np.ndarray, start: np.ndarray) -> float:
        key = tuple(np.asarray(log10_strengths, dtype=np.float64))
        if key in fits:
            return fits[key][1]
        strengths = 10.0 ** np.asarray(log10_strengths, dtype=np.float64)
        beta = np.reshape(
            fit_row(start, strengths, warn=False), (n_coefficients, n_free)
        )
        row_penalty = penalty(strengths)
        log_pdet = n_free * _precision_log_pdet(penalties, strengths, l2_penalty)
        # A singular Hessian is handled below; slogdet need not warn about it.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, log_det = np.linalg.slogdet(
                _blocked_hessian(beta, blocks, prior_per_bin, row_penalty)
            )
        value: float = (
            -_blocked_loss(beta, blocks, prior_per_bin, row_penalty)
            + 0.5 * log_pdet
            - 0.5 * log_det
        )
        # A singular or indefinite Hessian (e.g. a row without expected
        # transitions) has no Laplace approximation.
        if sign <= 0 or not np.isfinite(value):
            value = -np.inf
        fits[key] = (beta, value)
        return value

    def search(
        point: Callable[[float], np.ndarray], grid: np.ndarray, start: np.ndarray
    ) -> float:
        """Return the log10 strength ``x`` maximizing ``laml(point(x))``.

        The search is a grid, strongest first with each fit starting from the
        next-stronger one, then a bounded refinement within a decade of the
        best.
        """
        scores: dict[float, tuple[float, ...]] = {}

        def score(x: float, start: np.ndarray) -> float:
            value = laml(point(x), start)
            scores[float(x)] = tuple(point(x))
            return value

        for x in np.sort(grid)[::-1]:
            score(x, start)
            start = fits[scores[float(x)]][0]

        def best() -> float:
            return max(scores, key=lambda x: fits[scores[x]][1])

        if not np.isfinite(fits[scores[best()]][1]):
            # No strength has a Laplace approximation: keep the strongest
            # (searched first), whose curve is flattest.
            return best()

        # When the weakest strength searched scores best, scan every decade
        # to the floor: the LAML can dip and rise again (a 0.2-nat dip hid a
        # maximum 13 nats higher three decades down).
        lowest = float(np.min(grid))
        if best() == lowest:
            while lowest > _REML_LOG10_FLOOR:
                above, lowest = lowest, lowest - 1.0
                score(lowest, fits[scores[above]][0])
        best_start = fits[scores[best()]][0]
        minimize_scalar(
            lambda x: -score(x, best_start),
            bounds=(max(best() - 1.0, lowest), min(best() + 1.0, float(np.max(grid)))),
            method="bounded",
            options={"xatol": _REML_LOG10_TOLERANCE},
        )
        return best()

    start = np.reshape(coefficients, (n_coefficients, n_free))
    shared = search(lambda x: np.full(n_terms, x), _REML_LOG10_GRID, start)
    log10_strengths = np.full(n_terms, shared)
    if n_terms > 1:
        for _ in range(_REML_MAX_SWEEPS):
            previous = log10_strengths.copy()
            for term in range(n_terms):

                def point(
                    x: float,
                    term: int = term,
                    base: np.ndarray = log10_strengths.copy(),
                ) -> np.ndarray:
                    moved = base.copy()
                    moved[term] = x
                    return moved

                current = log10_strengths[term]
                # Within two decades of the current value, which is on the
                # grid so the search never returns a lower LAML.
                grid = np.unique(
                    np.clip(
                        current + np.arange(-2.0, 3.0),
                        _REML_LOG10_FLOOR,
                        _REML_LOG10_GRID[-1],
                    )
                )
                log10_strengths = point(
                    search(point, grid, fits[tuple(log10_strengths)][0])
                )
            if np.max(np.abs(log10_strengths - previous)) < _REML_LOG10_TOLERANCE:
                break
    return fits[tuple(log10_strengths)][0], 10.0**log10_strengths


def _fit_covariate_row(
    initial_coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
    optimization_method: str,
    options: dict[str, Any],
    from_state: int,
    *,
    warn: bool = True,
) -> np.ndarray:
    """Minimize one row's :func:`dirichlet_neg_log_likelihood` over its blocks.

    The blocks are of transitions. Call with JAX in float64.

    Parameters
    ----------
    initial_coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
        Starting point of the optimizer.
    blocks, prior_per_bin
        From :func:`_row_problem`.
    penalty : np.ndarray, shape (n_coefficients, n_coefficients)
        From :func:`_penalty_matrix`.
    optimization_method : str
        Optimization method for `scipy.optimize.minimize`.
    options : dict
        Options for `scipy.optimize.minimize`.
    from_state : int
        The row, for log messages.
    warn : bool, optional
        Log when the optimizer does not lower the loss or stops short, by
        default True. REML's candidate fits are not logged; the chosen fit is
        checked once.

    Returns
    -------
    coefficients : np.ndarray, shape (n_coefficients, n_states - 1)
        The initial coefficients when there are no transitions or no
        coefficients to fit (one state, whose only transition is certain), or
        when the optimizer does not lower the loss.
    """
    x0 = np.asarray(initial_coefficients, dtype=np.float64).ravel()
    if not blocks or x0.size == 0:
        return initial_coefficients
    args = (blocks, prior_per_bin, penalty)
    minimize_kwargs = {
        "fun": _blocked_loss_and_gradient,
        "x0": x0,
        "method": optimization_method,
        "jac": True,
        "args": args,
        "options": options,
    }
    if optimization_method in {"Newton-CG", "trust-ncg", "dogleg", "trust-exact"}:
        minimize_kwargs["hess"] = _blocked_hessian
    try:
        # scipy 1.15's trust-exact takes the square root of a negative bound
        # product on some Hessians and warns; the loss check below guards the
        # result whatever the optimizer did.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "invalid value encountered in sqrt",
                RuntimeWarning,
                r"scipy\.optimize\._trustregion_exact",
            )
            result = minimize(**minimize_kwargs)
    except (ValueError, np.linalg.LinAlgError) as error:
        # trust-exact's subproblem can fail on a badly conditioned Hessian
        # (seen with scipy 1.12 at strong curvature penalties): fall back to
        # gradient steps, which the loss check below still guards.
        if warn:
            logger.warning(
                "%s failed for state %d (%s); retrying with L-BFGS-B.",
                optimization_method,
                from_state,
                error,
            )
        minimize_kwargs.pop("hess", None)
        result = minimize(
            **{
                **minimize_kwargs,
                "method": "L-BFGS-B",
                "options": {
                    "maxiter": options.get("maxiter", 100),
                    "ftol": 0.0,
                    "gtol": 1e-7,
                },
            }
        )

    # EM never lowers its penalized log-likelihood only if no M-step raises
    # its own loss, so the optimizer's coefficients replace the previous ones
    # only when they do not, whatever the optimizer reports.
    result_loss = _blocked_loss(result.x, *args)
    if not (np.isfinite(result_loss) and result_loss <= _blocked_loss(x0, *args)):
        if not warn:
            return initial_coefficients
        logger.warning(
            "Transition optimization did not lower the loss for state %d: %s. "
            "Keeping previous coefficients.",
            from_state,
            result.message,
        )
        return initial_coefficients
    if warn and not result.success and _newton_decrement(result.x, *args) > 1e-6:
        logger.warning(
            "Transition optimization did not report convergence for state %d: "
            "%s. Using improved coefficients.",
            from_state,
            result.message,
        )
    return cast(np.ndarray, result.x.reshape(np.shape(initial_coefficients)))


def _newton_decrement(coefficients: np.ndarray, *args: Any) -> float:
    """Loss decrease a Newton step predicts, ``g @ inv(H) @ g / 2``.

    Near the optimum of a badly conditioned row (strong penalties give
    condition numbers of 1e10) the gradient stalls well above scipy's
    tolerance while the remaining decrease is negligible; this measures the
    latter. ``inf`` if the Hessian is singular. Call with JAX in float64.
    """
    _, gradient = _blocked_loss_and_gradient(coefficients, *args)
    try:
        step = np.linalg.solve(_blocked_hessian(coefficients, *args), gradient)
    except np.linalg.LinAlgError:
        return np.inf
    return float(gradient @ step) / 2.0


def _row_prior_weights(prior_weight: float | np.ndarray, n_states: int) -> np.ndarray:
    """Validate ``prior_weight`` and broadcast it to one weight per row.

    Parameters
    ----------
    prior_weight : float or np.ndarray, shape (n_states,)
        Data-adaptive prior weight: one for every row, or one per row.
    n_states : int
        Number of discrete states.

    Returns
    -------
    prior_weight : np.ndarray, shape (n_states,)
    """
    prior_weight_arr = np.atleast_1d(np.asarray(prior_weight, dtype=float))
    if prior_weight_arr.size == 1:
        prior_weight_arr = np.full(n_states, prior_weight_arr.item())
    if prior_weight_arr.shape != (n_states,):
        msg = (
            f"prior_weight must be a scalar or array of shape ({n_states},), "
            f"got shape {prior_weight_arr.shape}"
        )
        raise ValueError(msg)
    if np.any(prior_weight_arr < 0):
        msg = f"prior_weight must be non-negative, got {prior_weight_arr}"
        raise ValueError(msg)
    return prior_weight_arr


def _stationary_pseudo_counts(
    joint_sum: np.ndarray, alpha: np.ndarray, prior_weight: np.ndarray
) -> np.ndarray:
    """Dirichlet pseudo-counts the stationary M-step adds to the expected counts.

    Rows with ``prior_weight[i] == 0`` use the fixed ``alpha - 1``. Rows with
    ``prior_weight[i] > 0`` keep the direction of ``alpha - 1`` but scale its
    total to ``prior_weight[i]`` times the expected transitions out of state
    ``i``, so these pseudo-counts change with the E-step.

    Parameters
    ----------
    joint_sum : np.ndarray, shape (n_states, n_states)
        Expected transition counts.
    alpha : np.ndarray, shape (n_states, n_states)
        Dirichlet parameters from :func:`get_transition_prior`.
    prior_weight : np.ndarray, shape (n_states,)
        Validated per-row weights from :func:`_row_prior_weights`.

    Returns
    -------
    pseudo_counts : np.ndarray, shape (n_states, n_states)
    """
    legacy_prior = alpha - 1.0
    if not np.any(prior_weight > 0):
        return legacy_prior
    # Normalize to get prior direction per row
    row_sums = legacy_prior.sum(axis=-1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
    tilde_alpha = np.where(row_sums == 0, 0.0, legacy_prior / safe_row_sums)
    # N_i = expected transitions from state i
    n_i = joint_sum.sum(axis=-1, keepdims=True)
    adaptive_prior = prior_weight[:, np.newaxis] * n_i * tilde_alpha
    return np.where((prior_weight > 0)[:, np.newaxis], adaptive_prior, legacy_prior)


def estimate_stationary_state_transition_from_counts(
    joint_sum: np.ndarray,
    stickiness: float | np.ndarray = 0.0,
    concentration: float = 1.0,
    prior_weight: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Estimate a stationary transition matrix from expected transition counts.

    Parameters
    ----------
    joint_sum : np.ndarray, shape (n_states, n_states)
        Expected transition counts from each source state to each target state.
    stickiness : float, optional
        Diagonal stickiness parameter, by default 0.0. If array-like, one
        stickiness value per state.
    concentration : float, optional
        Dirichlet prior concentration parameter, by default 1.0.
    prior_weight : float or np.ndarray, shape (n_states,), optional
        Dimensionless weight for data-adaptive prior scaling. When > 0,
        the effective prior pseudo-counts for that row are scaled by the
        expected transition count ``N_i = joint_sum[i].sum()``, making the
        prior influence approximately invariant to the number of time bins.

        Can be a scalar (same weight for all rows) or an array with one
        value per state. Rows with ``prior_weight[i] == 0`` use the legacy
        fixed-count prior from ``concentration`` and ``stickiness``
        directly. This allows mixing frozen rows (e.g., structural states
        with very high stickiness) with data-adaptive rows in the same
        call. When 0.0 (default), all rows use legacy behavior.

    Returns
    -------
    new_transition_matrix : np.ndarray, shape (n_states, n_states)
        Row-stochastic transition matrix estimated from the counts.
    """
    n_states = joint_sum.shape[0]
    alpha = get_transition_prior(concentration, stickiness, n_states)
    _assert_map_compatible_alpha(alpha)
    prior_weight_arr = _row_prior_weights(prior_weight, n_states)

    # Warn when an adaptive row has a uniform prior direction (alpha - 1 == 0,
    # e.g. concentration=1.0 and stickiness=0.0): prior_weight then has no
    # effect because there is nothing to scale.
    no_op_rows = (prior_weight_arr > 0) & ((alpha - 1.0).sum(axis=-1) == 0)
    if np.any(no_op_rows):
        logger.warning(
            "prior_weight > 0 has no effect for states %s because the prior "
            "direction is uniform (alpha - 1 == 0). Increase concentration or "
            "stickiness to regularize these rows.",
            np.flatnonzero(no_op_rows).tolist(),
        )

    effective_prior = _stationary_pseudo_counts(joint_sum, alpha, prior_weight_arr)
    new_transition_matrix = joint_sum + effective_prior

    # Normalize rows to get transition probabilities.
    # Guard against all-zero rows (e.g., unvisited states) to avoid NaN.
    # Fall back to normalized Dirichlet prior direction for unvisited states,
    # preserving stickiness semantics. If prior is also uniform, this gives 1/n.
    prior_fallback = alpha / alpha.sum(axis=-1, keepdims=True)
    row_totals = new_transition_matrix.sum(axis=-1, keepdims=True)
    safe_row_totals = np.where(row_totals == 0, 1.0, row_totals)
    return cast(
        np.ndarray,
        np.where(
            row_totals == 0,
            prior_fallback,
            new_transition_matrix / safe_row_totals,
        ),
    )


def _block_loss(
    coefficients: jax.Array,
    design_matrix: jax.Array,
    response: jax.Array,
    prior_per_bin: jax.Array,
) -> jax.Array:
    """``-sum_t (r_t + prior_per_bin) . log p_t`` over one block of time bins.

    Parameters
    ----------
    coefficients : jax.Array, shape (n_coefficients * (n_states - 1),)
        Ordered (coefficient, logit).
    design_matrix : jax.Array, shape (n_block, n_coefficients)
        Design matrix rows of the block.
    response : jax.Array, shape (n_block, n_states)
        Expected transition counts of the block, for this row.
    prior_per_bin : jax.Array, shape (n_states,)
        The row's Dirichlet pseudo-counts ``alpha - 1`` divided by the number
        of time bins of the whole row, so that the prior counts once.

    Returns
    -------
    loss : jax.Array, shape ()
    """
    log_probabilities = jax_centered_log_softmax_forward(
        design_matrix @ coefficients.reshape((design_matrix.shape[1], -1))
    )
    return -jnp.sum((response + prior_per_bin) * log_probabilities)


@jax.jit
def _block_hessian(
    coefficients: jax.Array,
    design_matrix: jax.Array,
    response: jax.Array,
    prior_per_bin: jax.Array,
) -> jax.Array:
    """Hessian of :func:`_block_loss` in closed form.

    With weights ``w_t = r_t + prior_per_bin`` summing to ``W_t`` and free
    probabilities ``p_t`` (all but the last state), the loss's Hessian in
    the logits ``z_t`` is ``W_t (diag(p_t) - p_t p_t')``, so the Hessian in
    the coefficients is ``sum_t x_t x_t' (x) W_t (diag(p_t) - p_t p_t')``.
    About 4x faster than ``jax.hessian``, which differentiates the gradient
    once per coefficient.

    Parameters
    ----------
    coefficients : jax.Array, shape (n_coefficients * (n_states - 1),)
        Ordered (coefficient, logit).
    design_matrix : jax.Array, shape (n_block, n_coefficients)
        Design matrix rows of the block.
    response : jax.Array, shape (n_block, n_states)
        Expected transition counts of the block, for this row.
    prior_per_bin : jax.Array, shape (n_states,)
        The row's Dirichlet pseudo-counts per time bin, as in
        :func:`_block_loss`.

    Returns
    -------
    hessian : jax.Array, shape (n_coefficients * (n_states - 1),) * 2
    """
    n_coefficients = design_matrix.shape[1]
    free = jnp.exp(
        jax_centered_log_softmax_forward(
            design_matrix @ coefficients.reshape((n_coefficients, -1))
        )
    )[:, :-1]
    n_free = free.shape[1]
    weight = jnp.sum(response + prior_per_bin, axis=1)
    curvature = weight[:, None, None] * (
        free[:, :, None] * jnp.eye(n_free) - free[:, :, None] * free[:, None, :]
    )
    hessian = jnp.einsum("tc,tjk,td->cjdk", design_matrix, curvature, design_matrix)
    return hessian.reshape((n_coefficients * n_free, n_coefficients * n_free))


_jit_block_loss = jax.jit(_block_loss)
_block_loss_and_gradient = jax.jit(jax.value_and_grad(_block_loss))


def _transition_blocks(
    values: np.ndarray, dtype: type | None = np.float64
) -> list[jax.Array]:
    """Device blocks of ``_COVARIATE_BLOCK_ROWS`` rows of ``values``.

    Call with JAX in float64; ``dtype=None`` keeps the values' own type.
    """
    return [
        jnp.asarray(values[start : start + _COVARIATE_BLOCK_ROWS], dtype=dtype)
        for start in range(0, len(values), _COVARIATE_BLOCK_ROWS)
    ]


def _row_problem(
    design_blocks: list[jax.Array],
    response: np.ndarray,
    alpha: float | np.ndarray,
) -> tuple[_Blocks, jax.Array]:
    """One row's blocks of transitions and its prior pseudo-counts per bin.

    Call with JAX in float64.

    Parameters
    ----------
    design_blocks : list of jax.Array
        :func:`_transition_blocks` of the design matrix's transition rows.
    response : np.ndarray, shape (n_samples, n_states)
        Expected transition counts of this row for each transition.
    alpha : float or np.ndarray, shape (n_states,)
        Dirichlet prior of this row.

    Returns
    -------
    blocks : list of (jax.Array, jax.Array)
        ``(design, response)`` blocks; empty without transitions.
    prior_per_bin : jax.Array, shape (n_states,)
        ``(alpha - 1) / n_samples``, zero without transitions.
    """
    n_samples, n_states = np.shape(response)
    blocks = list(
        zip(design_blocks, _transition_blocks(response, dtype=None), strict=True)
    )
    pseudo_counts = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (n_states,))
    prior_per_bin = (pseudo_counts - 1.0) / max(n_samples, 1)
    return blocks, jnp.asarray(prior_per_bin, dtype=jnp.float64)


def _penalty_matrix(
    n_coefficients: int,
    l2_penalty: float,
    smoothness: float | np.ndarray = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Quadratic penalty on one row's coefficients, per free logit.

    ``l2_penalty`` on every coefficient but the intercept (the first) plus
    each spline term's curvature penalty times its strength.

    Parameters
    ----------
    n_coefficients : int
        Number of coefficients per logit.
    l2_penalty : float
        Penalty strength on every coefficient but the intercept.
    smoothness : float or np.ndarray, shape (n_terms,), optional
        One strength for every term, or one per term.
    smoothness_matrix : np.ndarray, optional
        From :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients), or one term's
        (n_coefficients, n_coefficients); None adds no curvature penalty.

    Returns
    -------
    penalty : np.ndarray, shape (n_coefficients, n_coefficients)
        The row's penalty is ``sum_k beta[:, k] @ penalty @ beta[:, k]`` over
        its ``n_states - 1`` logits.
    """
    penalty = np.diag(np.r_[0.0, np.full(n_coefficients - 1, float(l2_penalty))])
    penalties = _term_penalties(smoothness_matrix, n_coefficients)
    strengths = np.broadcast_to(
        np.asarray(smoothness, dtype=np.float64), (penalties.shape[0],)
    )
    if np.any(strengths != 0.0):
        penalty = penalty + np.tensordot(strengths, penalties, axes=1)
    return penalty


def _penalty_value(coefficients: np.ndarray, penalty: np.ndarray) -> float:
    """``sum_k beta[:, k] @ penalty @ beta[:, k]`` of flattened coefficients."""
    beta = np.reshape(coefficients, (penalty.shape[0], -1))
    return float(np.sum(beta * (penalty @ beta)))


def _blocked_loss(
    coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
) -> float:
    """:func:`dirichlet_neg_log_likelihood` over blocks of transitions.

    Call with JAX in float64. ``penalty`` is from :func:`_penalty_matrix`.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
    # Summed on the device, so the blocks are not each copied to the host.
    device_coefficients = jnp.asarray(coefficients)
    loss = sum(
        _jit_block_loss(device_coefficients, design, response, prior_per_bin)
        for design, response in blocks
    )
    return float(loss) + _penalty_value(coefficients, penalty)


def _blocked_loss_and_gradient(
    coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return :func:`dirichlet_neg_log_likelihood` and its gradient, for scipy.

    Evaluated over blocks of transitions. Call with JAX in float64.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
    # Summed on the device, so the blocks are not each copied to the host.
    device_coefficients = jnp.asarray(coefficients)
    device_loss, device_gradient = jnp.zeros(()), jnp.zeros(coefficients.shape)
    for design, response in blocks:
        block_loss, block_gradient = _block_loss_and_gradient(
            device_coefficients, design, response, prior_per_bin
        )
        device_loss = device_loss + block_loss
        device_gradient = device_gradient + block_gradient
    gradient = np.array(device_gradient, dtype=np.float64)
    loss = float(device_loss) + _penalty_value(coefficients, penalty)
    beta = coefficients.reshape((penalty.shape[0], -1))
    gradient += (2.0 * penalty @ beta).ravel()
    return loss, gradient


def _blocked_hessian(
    coefficients: np.ndarray,
    blocks: _Blocks,
    prior_per_bin: jax.Array,
    penalty: np.ndarray,
) -> np.ndarray:
    """Return the Hessian of :func:`dirichlet_neg_log_likelihood`, for scipy.

    Evaluated over blocks of transitions. Call with JAX in float64.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
    # Summed on the device, so the blocks are not each copied to the host.
    device_coefficients = jnp.asarray(coefficients)
    device_hessian = jnp.zeros((coefficients.size, coefficients.size))
    for design, response in blocks:
        device_hessian = device_hessian + _block_hessian(
            device_coefficients, design, response, prior_per_bin
        )
    hessian = np.array(device_hessian, dtype=np.float64)
    # Flattened coefficients are ordered (coefficient, logit), so each logit
    # gets its own copy of the penalty.
    n_free = coefficients.size // penalty.shape[0]
    return hessian + np.kron(2.0 * penalty, np.eye(n_free))


def dirichlet_neg_log_likelihood(
    coefficients: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    alpha: float | np.ndarray = 1.0,
    l2_penalty: float = 0.1,
    smoothness: float = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> float:
    """Negative log posterior of one covariate-dependent transition row.

    ``-sum_t (r_t + (alpha - 1) / n) . log p_t + l2_penalty * |beta[1:]|^2
    + sum_j smoothness_j * sum_k beta[:, k] @ S_j @ beta[:, k]``, summed over the ``n``
    time bins: the expected complete-data log-likelihood of the row plus a
    Dirichlet prior counted once, as for a stationary row, an L2 penalty and
    a curvature penalty ``S_j`` on each spline term. Evaluated with
    JAX in float64 over blocks of ``_COVARIATE_BLOCK_ROWS`` time bins, as the
    M-step evaluates it.

    Parameters
    ----------
    coefficients : np.ndarray, shape (n_coefficients * (n_states - 1),)
        Flattened regression coefficients.
    design_matrix : np.ndarray, shape (n_samples, n_coefficients)
        Covariate design matrix.
    response : np.ndarray, shape (n_samples, n_states)
        Expected transition counts from this row at each time bin.
    alpha : float or np.ndarray, shape (n_states,), optional
        Dirichlet prior parameters for this row. Defaults to 1.0 (no prior).
    l2_penalty : float, optional
        L2 penalty on coefficients (excluding intercept), by default 0.1, as
        for the detectors' ``discrete_transition_regularization``.
        The intercept is assumed to be the first coefficient row
        (``coefficients[0]``), as produced by patsy ``"1 + ..."`` formulas.
        A ``"0 + ..."`` formula or hand-built design matrix without a leading
        intercept column would penalize the wrong rows.
    smoothness : float or np.ndarray, shape (n_terms,), optional
        Strength of each spline term's curvature penalty (one value for
        every term), by default 0.0.
    smoothness_matrix : np.ndarray, optional
        Curvature penalties from :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients), or one term's
        (n_coefficients, n_coefficients); None (default) adds none.

    Returns
    -------
    negative_log_posterior : float
        Value of the penalized negative log posterior.
    """
    with _float64_jax():
        return _blocked_loss(
            coefficients,
            *_row_problem(_transition_blocks(design_matrix), response, alpha),
            _penalty_matrix(
                np.shape(design_matrix)[1], l2_penalty, smoothness, smoothness_matrix
            ),
        )


def _transition_log_prior(
    statistics: np.ndarray,
    discrete_transition: np.ndarray | None,
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | np.ndarray | None,
    transition_concentration: float,
    transition_stickiness: float | np.ndarray,
    transition_regularization: float,
    transition_prior_weight: float | np.ndarray = 0.0,
    free_rows: np.ndarray | None = None,
    smoothness: float | np.ndarray = 0.0,
    smoothness_matrix: np.ndarray | None = None,
) -> float:
    """Log prior of discrete transition parameters, as the M-step weighs it.

    The discrete M-step maximizes the expected complete-data log-likelihood
    plus this term, so EM never decreases the marginal log-likelihood plus
    this term. Both are evaluated with the prior of one M-step (the
    data-adaptive ``prior_weight`` pseudo-counts depend on ``statistics``).

    Stationary: ``sum_ij c_ij log A_ij`` with the pseudo-counts ``c`` the
    stationary M-step adds once per row. Covariate-dependent:
    ``sum_ij (alpha_ij - 1) mean_t log A_t,ij - lambda * |beta_i[1:]|^2``
    ``- sum_j smoothness_ij * sum_k beta_i[:, k] @ S_j @ beta_i[:, k]`` over the
    ``n_time - 1`` transitions, the prior and penalty terms of
    :func:`dirichlet_neg_log_likelihood`, negated.

    Parameters
    ----------
    statistics : np.ndarray
        Expected transition counts, shape (n_states, n_states), or per-row
        responses, shape (n_time - 1, n_states, n_states), as passed to
        :func:`_discrete_transition_from_statistics`.
    discrete_transition : np.ndarray, shape (n_states, n_states), or None
        Stationary transition matrix to evaluate. Ignored for a
        covariate-dependent transition, which is evaluated from its
        coefficients.
    discrete_transition_coefficients : np.ndarray or None
        Shape (n_coefficients, n_states, n_states - 1) for a
        covariate-dependent transition.
    discrete_transition_design_matrix : DesignMatrix, np.ndarray or None
        Shape (n_time, n_coefficients) for a covariate-dependent transition.
    transition_concentration : float
        Dirichlet prior concentration parameter (uniform part).
    transition_stickiness : float or np.ndarray
        Dirichlet prior stickiness parameter (diagonal enhancement).
    transition_regularization : float
        L2 penalty on covariate-dependent coefficients (excluding intercept).
    transition_prior_weight : float or np.ndarray, optional
        Data-adaptive prior weight for stationary transitions, as in
        :func:`_discrete_transition_from_statistics`. By default 0.0.
    free_rows : np.ndarray of bool, shape (n_states,), optional
        Rows the M-step estimates. Frozen rows are constant and left out.
        All rows by default.
    smoothness : float or np.ndarray, optional
        Curvature-penalty strength of each covariate-dependent row and spline
        term, shape (n_states, n_terms), or one per row, shape (n_states,);
        by default 0.0.
    smoothness_matrix : np.ndarray, optional
        From :func:`_smoothness_matrix`, shape
        (n_terms, n_coefficients, n_coefficients); None adds no curvature
        penalty.

    Returns
    -------
    log_prior : float
        ``-inf`` if the prior puts pseudo-counts on a zero-probability entry.
    """
    statistics = np.asarray(statistics)
    n_states = statistics.shape[-1]
    alpha = get_transition_prior(
        transition_concentration, transition_stickiness, n_states
    )
    rows = np.ones(n_states, dtype=bool) if free_rows is None else free_rows
    if _is_nonstationary(
        discrete_transition_coefficients, discrete_transition_design_matrix
    ):
        n_samples = statistics.shape[0]
        if n_samples == 0:
            # A recording without transitions has no covariate prior, and the
            # M-step keeps its coefficients.
            return 0.0
        coefficients = np.asarray(discrete_transition_coefficients, dtype=np.float64)
        design_matrix = np.asarray(discrete_transition_design_matrix)
        free_states = np.flatnonzero(rows)
        prior_per_bin = (alpha - 1.0) / n_samples
        # Minus the M-step's loss with no expected transitions: its prior and
        # penalties alone, with one block of the design matrix on the device
        # at a time.
        penalties = _term_penalties(smoothness_matrix, coefficients.shape[0])
        row_smoothness = _row_strengths(smoothness, n_states, penalties.shape[0])
        log_prior = -sum(
            _penalty_value(
                coefficients[:, row],
                _penalty_matrix(
                    coefficients.shape[0],
                    transition_regularization,
                    row_smoothness[row],
                    penalties,
                ),
            )
            for row in free_states
        )
        with _float64_jax():
            for start in range(0, n_samples, _COVARIATE_BLOCK_ROWS):
                stop = min(start + _COVARIATE_BLOCK_ROWS, n_samples)
                design = jnp.asarray(design_matrix[start:stop], dtype=jnp.float64)
                no_counts = jnp.zeros((stop - start, n_states))
                for row in free_states:
                    log_prior -= float(
                        _jit_block_loss(
                            coefficients[:, row].ravel(),
                            design,
                            no_counts,
                            jnp.asarray(prior_per_bin[row]),
                        )
                    )
        return log_prior

    pseudo_counts = _stationary_pseudo_counts(
        statistics, alpha, _row_prior_weights(transition_prior_weight, n_states)
    )
    # xlogy gives 0 * log(0) = 0: an impossible transition costs nothing
    # without pseudo-counts, and -inf with them.
    terms = xlogy(pseudo_counts, np.asarray(discrete_transition, dtype=np.float64))
    return float(np.sum(terms[rows]))


def _is_nonstationary(
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | None,
) -> bool:
    """Whether the discrete transition depends on covariates."""
    return (
        discrete_transition_coefficients is not None
        and discrete_transition_design_matrix is not None
    )


def _discrete_transition_from_statistics(
    statistics: np.ndarray,
    discrete_transition_coefficients: np.ndarray | None,
    discrete_transition_design_matrix: DesignMatrix | None,
    transition_concentration: float,
    transition_stickiness: float | np.ndarray,
    transition_regularization: float,
    transition_prior_weight: float | np.ndarray = 0.0,
    frozen_rows: np.ndarray | None = None,
    smoothness: float | np.ndarray | str = 0.0,
    smoothness_matrix: np.ndarray | None = None,
    initial_smoothness: np.ndarray | None = None,
) -> "_DiscreteTransitionMStep":
    """Discrete transition M-step from expected transition statistics.

    Parameters
    ----------
    statistics : np.ndarray
        Expected transition counts, shape (n_states, n_states), for a
        stationary transition; per-row responses, shape
        (n_time - 1, n_states, n_states), for a covariate transition.
    discrete_transition_coefficients : np.ndarray or None
        Current coefficients, shape (n_coefficients, n_states, n_states - 1),
        of a covariate transition; None for a stationary one.
    discrete_transition_design_matrix : patsy.DesignMatrix or None
        Design matrix of a covariate transition; None for a stationary one.
    transition_concentration : float
        Dirichlet prior concentration.
    transition_stickiness : float or np.ndarray
        Dirichlet prior stickiness.
    transition_regularization : float
        L2 penalty for non-stationary coefficients.
    transition_prior_weight : float, optional
        Data-adaptive prior weight for stationary transitions. When > 0,
        pseudo-counts scale with expected transition counts. Only applies
        to the stationary path. By default 0.0.
    frozen_rows : np.ndarray of bool, shape (n_states,), optional
        Covariate-dependent rows that keep their coefficients. Stationary
        frozen rows are restored by the caller.
    smoothness, smoothness_matrix, initial_smoothness
        Curvature penalty of a covariate transition, as in
        :func:`_fit_covariate_transition`. Ignored for a stationary one. The
        covariance is not computed: EM computes it once, after its last
        M-step, with :func:`_covariate_transition_covariance`.

    Returns
    -------
    m_step : _DiscreteTransitionMStep

    Raises
    ------
    ValueError
        If any statistic is NaN or infinite, which would otherwise give NaN
        transitions without an error.
    """
    statistics = np.asarray(statistics)
    bad = ~np.isfinite(statistics)
    if bad.any():
        pairs = sorted(
            {tuple(int(i) for i in index[-2:]) for index in np.argwhere(bad)}
        )
        msg = (
            "Expected discrete transition statistics are not finite for state "
            f"pairs {pairs}. This usually follows impossible likelihood rows; "
            "see the degenerate timestep warning."
        )
        raise ValueError(msg)
    if _is_nonstationary(
        discrete_transition_coefficients, discrete_transition_design_matrix
    ):
        fit = _fit_covariate_transition(
            cast(np.ndarray, discrete_transition_coefficients),
            cast(np.ndarray, discrete_transition_design_matrix),
            statistics,
            concentration=transition_concentration,
            stickiness=transition_stickiness,
            transition_regularization=transition_regularization,
            smoothness=smoothness,
            smoothness_matrix=smoothness_matrix,
            initial_smoothness=initial_smoothness,
            frozen_rows=frozen_rows,
            compute_covariance=False,
        )
        return _DiscreteTransitionMStep(
            fit.transition, fit.coefficients, fit.smoothness
        )
    stickiness_value: float | np.ndarray
    if (
        isinstance(transition_stickiness, np.ndarray)
        and transition_stickiness.size == 1
    ):
        stickiness_value = float(transition_stickiness.item())
    else:
        stickiness_value = transition_stickiness
    discrete_transition = estimate_stationary_state_transition_from_counts(
        statistics,
        concentration=transition_concentration,
        stickiness=stickiness_value,
        prior_weight=transition_prior_weight,
    )

    return _DiscreteTransitionMStep(
        discrete_transition, discrete_transition_coefficients
    )


class _DiscreteTransitionMStep(NamedTuple):
    """Output of :func:`_discrete_transition_from_statistics`.

    ``smoothness`` is that of :class:`CovariateTransitionFit`, and None for
    a stationary transition.
    """

    transition: np.ndarray
    coefficients: np.ndarray | None
    smoothness: np.ndarray | None = None
