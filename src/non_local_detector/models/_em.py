"""Parameter estimation (EM) of the detectors.

The Expectation-Maximization engine behind ``estimate_parameters``: the
internal fit's choice of dense or structured EM, the option checks, the
E-steps, the M-steps for the encoding model, discrete transitions and
initial conditions, the convergence and monotonicity checks, and the final
E-step and results. ``_DetectorBase`` inherits the methods of ``_EMMixin``;
the detectors' ``estimate_parameters`` validate and fit the data, then call
``_run_em``.

Methods that read the detector's attributes annotate ``self`` as
``_DetectorBase``, which mypy only accepts with ``type: ignore[misc]`` on a
mixin.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

import non_local_detector
from non_local_detector._transition_estimation import (
    _covariate_transition_covariance,
    _discrete_transition_from_statistics,
    _max_transition_change,
    _transition_log_prior,
)
from non_local_detector._warnings import user_stacklevel
from non_local_detector.checkpointed_inference import (
    _available_host_bytes,
    default_chunk_size,
    default_replay_cache_bytes,
)
from non_local_detector.core import check_converged
from non_local_detector.discrete_state_transitions import (
    estimate_discrete_transition_counts_from_factorized_posteriors,
)
from non_local_detector.exceptions import (
    DataError,
    NumericalWarning,
    ParameterWarning,
    ValidationError,
)
from non_local_detector.models._checkpointed import _driver_outputs, _PosteriorOutputs
from non_local_detector.models._input_validation import _normalize_return_outputs
from non_local_detector.models._results import convert_results_to_xarray
from non_local_detector.result_store import DEFAULT_MAX_READ_BYTES
from non_local_detector.time_edges import (
    _uniformity_tolerance,
    decode_bin_centers,
    uniform_time_edges,
)
from non_local_detector.types import (
    ReturnOutputs,
    TransitionRepresentation,
    TransitionStatistics,
)

if TYPE_CHECKING:
    import non_local_detector.models.base

# EM logs under the detector module's logger, as it did when it lived there.
logger = getLogger("non_local_detector.models.base")


def _validate_encoding_update_damping(encoding_update_damping: float) -> None:
    """Reject any nonzero ``encoding_update_damping``.

    Blending only ``place_fields`` cannot keep a refit encoding model
    self-consistent: the GLM and KDE local paths read other fitted state, the
    diffusion/MRF spike term reads ``interior_log_place_fields``, and the
    clusterless models have no ``place_fields`` at all. Estimation wrappers call
    this before their initial ``fit`` so a rejected value changes nothing.

    Parameters
    ----------
    encoding_update_damping : float
        Requested damping of encoding updates; only 0.0 is accepted.

    Raises
    ------
    ValidationError
        If the value is not zero (including NaN).
    """
    if encoding_update_damping != 0.0:
        msg = "encoding_update_damping is not supported"
        raise ValidationError(
            msg,
            expected="encoding_update_damping=0.0",
            got=str(encoding_update_damping),
            hint=(
                "Damped encoding updates are not implemented for any likelihood: "
                "blending place fields leaves the other fitted encoding state "
                "stale. Use min_encoding_local_mass / min_encoding_local_ess to "
                "skip poorly supported encoding updates instead."
            ),
        )


# EM with transition_representation="auto" runs dense EM up to this many bytes
# of float32 interior transition. A dense step reads the whole matrix, so its
# cost grows with this size; a structured step costs about the same at any
# size. Three warm EM iterations on a 23.6 min clusterless 2-D recording (A100)
# took 181/216 s dense/structured at 5.7 MB (1,190 interior state bins),
# 250/276 s at 13.5 MB, 348/255 s at 21.9 MB and 501/362 s at 43.5 MB (3,298
# bins); interpolating puts the crossover near 15 MB.
_AUTO_DENSE_TRANSITION_BYTES = 16 * 2**20


# Dense EM holds about this many float32 (time bins x state bins) arrays on the
# device (measured device peak 10.0 GB, about seven 709k x 498 arrays, on the
# clusterless track recording).
_DENSE_EM_WORKING_ARRAYS = 8


# EM that estimates the discrete transitions also requires the last M-step to
# move every transition probability by less than this: a rarely occupied state
# barely changes the objective, so a flat objective does not show that the
# transitions have settled. It is a probability, so it is separate from
# ``tolerance`` (nats per time bin).
_TRANSITION_CHANGE_TOLERANCE = 1e-4


def _device_memory_limit() -> int | None:
    """Bytes JAX may allocate on the default device, or None if unknown (CPU)."""
    stats = jax.local_devices()[0].memory_stats()
    return None if not stats else stats.get("bytes_limit")


def _structured_em_options(
    chunk_size: int | None,
    result_path: str | Path | None,
    replay_cache_bytes: int | None,
) -> list[str]:
    """Names of the structured-EM options that were given."""
    return [
        name
        for name, value in (
            ("chunk_size", chunk_size),
            ("result_path", result_path),
            ("replay_cache_bytes", replay_cache_bytes),
        )
        if value is not None
    ]


def _validate_em_options(
    *,
    structured: bool,
    n_chunks: int,
    chunk_size: int | None,
    result_path: str | Path | None = None,
    store_log_likelihood: bool = False,
    replay_cache_bytes: int | None = None,
) -> None:
    """Reject options that do not apply to the EM path.

    Parameters
    ----------
    structured : bool
        Whether EM runs on a structured transition operator.
    n_chunks : int
        Dense-EM row chunks.
    chunk_size : int or None
        Checkpoint chunk rows for structured EM.
    result_path : path-like or None
        Store for structured EM's final outputs.
    store_log_likelihood : bool
        Whether EM keeps the full likelihood in memory.
    replay_cache_bytes : int or None
        Structured EM's replay cache budget.
    """
    for name, value, minimum in (
        ("chunk_size", chunk_size, 1),
        ("replay_cache_bytes", replay_cache_bytes, 0),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))  # type: ignore[redundant-expr]  # runtime check of caller input
            or value < minimum
        ):
            msg = f"Invalid {name} for estimate_parameters"
            raise ValidationError(
                msg,
                expected=f"an integer >= {minimum}, or None for the default",
                got=f"{value!r} ({type(value).__name__})",
                hint=(
                    "Pass whole rows, e.g. chunk_size=20_000."
                    if name == "chunk_size"
                    else "Pass bytes as an integer, e.g. replay_cache_bytes=8 * 2**30."
                ),
            )
    if structured and n_chunks != 1:
        msg = "n_chunks applies only to dense EM"
        raise ValidationError(
            msg,
            got=f"n_chunks={n_chunks} with structured EM",
            hint=(
                "Structured EM already bounds memory with checkpoint chunks: set "
                "chunk_size (rows per chunk) instead, or pass "
                'transition_representation="dense" to estimate_parameters to '
                "run dense EM in n_chunks chunks."
            ),
        )
    if not structured:
        checkpoint_options = _structured_em_options(
            chunk_size, result_path, replay_cache_bytes
        )
        if checkpoint_options:
            msg = f"{', '.join(checkpoint_options)} applies only to structured EM"
            raise ValidationError(
                msg,
                got=f"{', '.join(checkpoint_options)} with dense EM",
                hint=(
                    'Pass transition_representation="structured" or "auto" to '
                    "estimate_parameters, or drop "
                    f"{', '.join(checkpoint_options)} for dense EM."
                ),
            )
    if result_path is not None and store_log_likelihood:
        msg = "store_log_likelihood cannot be combined with result_path"
        raise ValidationError(
            msg,
            got="store_log_likelihood=True and a result_path",
            hint=(
                "store_log_likelihood keeps the full likelihood in memory; with "
                "result_path, request return_outputs='log_likelihood' to write "
                "it to the store instead."
            ),
        )


def _dense_em_memory(n_time: int, n_bins: int) -> tuple[int, int | None, str]:
    """Bytes one-chunk dense EM holds, the memory it draws on, and where.

    ``_DENSE_EM_WORKING_ARRAYS`` float32 time-by-state-bin arrays, against the
    device limit, or the available host memory on CPU (None if unknown).

    Parameters
    ----------
    n_time : int
        Decode bins.
    n_bins : int
        Interior state bins.

    Returns
    -------
    needed : int
    limit : int or None
    where : {"device", "host"}
    """
    needed = _DENSE_EM_WORKING_ARRAYS * n_time * n_bins * 4
    limit = _device_memory_limit()
    if limit is not None:
        return needed, limit, "device"
    return needed, _available_host_bytes(), "host"


def _warn_if_dense_em_exceeds_memory(n_time: int, n_bins: int) -> None:
    """Warn before one-chunk dense EM whose arrays exceed half the memory.

    The line "auto" uses (see ``_dense_em_memory``). On the 23.6 min 2-D
    recording (A100) the estimate was 27.0 and 41.8 GB at 4 and 3 cm
    (measured peaks 23.7 and 36.6 GB) and 53.1 GB at 2.5 cm, which ran out of
    a 63.8 GB limit, so the full limit is not a safe line. The estimate is
    approximate, so this warns rather than raising.

    Parameters
    ----------
    n_time : int
        Decode bins.
    n_bins : int
        Interior state bins.
    """
    needed, limit, where = _dense_em_memory(n_time, n_bins)
    if limit is None or needed <= limit // 2:
        return
    warnings.warn(
        f"Dense EM needs about {needed / 1e9:.1f} GB of {where} memory for "
        f"{n_time} time bins x {n_bins} state bins, more than half of the "
        f"{limit / 1e9:.1f} GB available, and may run out of memory. "
        'Pass transition_representation="auto" (or "structured") to '
        "estimate_parameters to bound EM's memory, or n_chunks > 1 for dense EM.",
        ParameterWarning,
        stacklevel=user_stacklevel(),
    )


def _resolve_em_options(
    transition_representation: TransitionRepresentation,
    n_chunks: int,
    chunk_size: int | None,
    result_path: str | Path | None,
    *,
    store_log_likelihood: bool,
    replay_cache_bytes: int | None,
) -> TransitionRepresentation:
    """Resolve "auto" with ``n_chunks`` to dense EM, then validate the options.

    ``n_chunks`` is a dense-EM option and ``chunk_size``, ``result_path`` and
    ``replay_cache_bytes`` are structured-EM options, so with "auto" the one
    given decides the path; both together are rejected.

    Returns
    -------
    transition_representation : str
    """
    checkpoint_options = _structured_em_options(
        chunk_size, result_path, replay_cache_bytes
    )
    if transition_representation == "auto" and n_chunks != 1:
        if checkpoint_options:
            msg = (
                "n_chunks (dense EM) cannot be combined with "
                f"{', '.join(checkpoint_options)} (structured EM)"
            )
            raise ValidationError(
                msg,
                got=f"n_chunks={n_chunks} and {', '.join(checkpoint_options)}",
                hint=(
                    "Drop n_chunks to run structured EM, or drop "
                    f"{', '.join(checkpoint_options)} to run dense EM in chunks."
                ),
            )
        transition_representation = "dense"
    _validate_em_options(
        structured=transition_representation != "dense",
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        result_path=result_path,
        store_log_likelihood=store_log_likelihood,
        replay_cache_bytes=replay_cache_bytes,
    )
    return transition_representation


def _validate_estimation_arguments(
    time_edges: ArrayLike,
    n_chunks: int,
    return_outputs: ReturnOutputs,
    *,
    min_encoding_local_mass: float,
    min_encoding_local_ess: float,
) -> tuple[np.ndarray, float]:
    """Check ``estimate_parameters`` arguments that need no fitted state.

    The estimation wrappers call this before their initial ``fit`` so an invalid
    argument cannot leave a fitted detector partly rebuilt.

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
        Time bin edges; must be uniform.
    n_chunks : int
        Number of chunks the bins are split into; at most ``n_bins``.
    return_outputs : str, collection of str, or None, optional
        Requested outputs.
    min_encoding_local_mass : float
        Minimum local mass for an encoding update; must be >= 0.
    min_encoding_local_ess : float
        Minimum local effective sample size for an encoding update; must be >= 0.

    Returns
    -------
    time_edges : np.ndarray, shape (n_bins + 1,)
        The validated edges (integers converted to float64).
    time_bin_width : float

    Raises
    ------
    ValidationError, DataError
        If ``time_edges`` are invalid or not uniform (see
        :func:`~non_local_detector.time_edges.uniform_time_edges`).
    ValidationError
        If ``n_chunks`` exceeds the number of bins, ``return_outputs`` is
        invalid, or an encoding-update threshold is negative.
    """
    time_edges, time_bin_width = uniform_time_edges(time_edges)
    n_bins = len(time_edges) - 1
    if n_chunks > n_bins:
        msg = (
            f"n_chunks ({n_chunks}) cannot exceed n_bins ({n_bins}). "
            f"Each chunk must contain at least one time bin."
        )
        raise ValidationError(
            msg,
            expected=f"n_chunks <= {n_bins}",
            got=str(n_chunks),
            hint="Lower n_chunks or pass more time bins.",
        )
    _normalize_return_outputs(return_outputs)
    if min_encoding_local_mass < 0:
        msg = "min_encoding_local_mass must be >= 0"
        raise ValidationError(msg, expected=">= 0", got=str(min_encoding_local_mass))
    if min_encoding_local_ess < 0:
        msg = "min_encoding_local_ess must be >= 0"
        raise ValidationError(msg, expected=">= 0", got=str(min_encoding_local_ess))
    return time_edges, time_bin_width


@dataclass(frozen=True)
class _EMSettings:
    """Options of one ``_DetectorBase._run_em`` run, resolved before its loop.

    Fields not documented here are ``_run_em`` arguments, after validation and
    defaults.

    Attributes
    ----------
    time_bin_width : float
    time_centers : np.ndarray, shape (n_time,)
    n_observed_bins : int
        Decode bins not marked missing; the units of ``tolerance``.
    checkpointed : bool
        Whether EM runs on the structured transition operator.
    requested_outputs : set of str
        Canonical ``return_outputs``.
    em_attrs : dict
        EM path and chunking, recorded on the results.
    covariate_transitions : bool
        Whether the discrete transitions depend on covariates.
    transition_statistics : {None, "counts", "responses"}
        What each E-step returns for the discrete M-step.
    prior_tracks_counts : bool
        Whether the transition prior scales with each E-step's expected
        counts (``discrete_transition_prior_weight > 0``, stationary).
    prior_set_in_m_step : bool
        Whether REML sets the transition prior in each M-step.
    smoothness_setting : float or str
        ``discrete_transition_type.smoothness``.
    prior : tuple
        Concentration, stickiness, regularization and prior weight.
    frozen_rows, free_rows : np.ndarray, shape (n_states,), or None
        Discrete-transition rows pinned and estimated.
    interior_state_ind : np.ndarray, shape (n_interior_state_bins,), or None
        Dense EM with estimated stationary transitions: state of each
        interior bin.
    interior_continuous_transition : np.ndarray or None
        Dense EM with estimated stationary transitions: the continuous
        transition over the interior state bins.
    encoding_fit_arguments : dict or None
        Keyword arguments of ``_fit_encoding_model`` that the encoding M-step
        refits with, besides the posterior weights.
    """

    time_edges: np.ndarray
    time_bin_width: float
    time_centers: np.ndarray
    log_likelihood_args: tuple[Any, ...]
    is_missing: np.ndarray | None
    n_observed_bins: int
    checkpointed: bool
    chunk_size: int | None
    replay_cache_bytes: int | None
    n_chunks: int
    cache_likelihood: bool
    store_log_likelihood: bool
    result_path: str | Path | None
    requested_outputs: set[str]
    em_attrs: dict[str, Any]
    estimate_initial_conditions: bool
    estimate_discrete_transition: bool
    estimate_encoding_model: bool
    max_iter: int
    tolerance: float
    min_encoding_local_mass: float
    min_encoding_local_ess: float
    covariate_transitions: bool
    transition_statistics: TransitionStatistics | None
    prior_tracks_counts: bool
    prior_set_in_m_step: bool
    smoothness_setting: float | str
    prior: tuple[Any, Any, Any, Any]
    frozen_rows: np.ndarray | None
    free_rows: np.ndarray | None
    interior_state_ind: np.ndarray | None
    interior_continuous_transition: np.ndarray | None
    encoding_fit_arguments: dict[str, Any] | None


@dataclass
class _EMState:
    """Values one EM iteration hands the next.

    EM monitors the marginal log-likelihood plus the log prior of the
    discrete transitions, the objective a MAP M-step never decreases; the
    likelihood alone can fall when the prior pulls the transitions. Each check
    compares two E-steps under the prior of the M-step between them
    (data-adaptive pseudo-counts change per E-step), so an iteration hands the
    next one its objective and the log prior of its M-step's output.

    Attributes
    ----------
    marginal_log_likelihoods : list of float
        Each in-loop E-step's marginal log-likelihood.
    objectives : list of float
        Each in-loop E-step's penalized log-likelihood.
    previous_objectives : list of float
        The objective each check compared ``objectives[k + 1]`` with.
    next_log_prior : float
        The log prior of the next E-step's parameters under the last M-step's
        prior; there is none before the first M-step.
    n_iter : int
    converged, objective_converged, is_increasing : bool
        The last check's outcome: converged (objective and, when they are
        estimated, transitions), objective within tolerance, and not decreasing.
    objective_change : float
        The last check's change in the objective.
    transition_change : float
        The last M-step's largest transition-probability change.
    run_log_likelihoods : np.ndarray, shape (n_time, n_interior_state_bins), or None
        A likelihood computed in this run that the next E-step can reuse.
    transition_statistics : np.ndarray or None
        The last discrete M-step's expected transition statistics.
    """

    marginal_log_likelihoods: list[float] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)
    previous_objectives: list[float] = field(default_factory=list)
    next_log_prior: float = 0.0
    n_iter: int = 0
    converged: bool = False
    objective_converged: bool = False
    is_increasing: bool = True
    objective_change: float = np.inf
    transition_change: float = np.inf
    run_log_likelihoods: np.ndarray | None = None
    transition_statistics: np.ndarray | None = None

    def traces(self, final_log_likelihood: float) -> dict[str, list[float]]:
        """EM's traces, given the final E-step's marginal log-likelihood.

        The final E-step evaluates the returned parameters, the last M-step's
        output, under that M-step's prior (none with max_iter=0).
        """
        return {
            "marginal_log_likelihoods": [
                *self.marginal_log_likelihoods,
                final_log_likelihood,
            ],
            "em_objectives": [
                *self.objectives,
                final_log_likelihood + self.next_log_prior,
            ],
            "em_previous_objectives": list(self.previous_objectives),
        }


@dataclass(frozen=True)
class _EMEStep:
    """What an in-loop E-step hands the M-steps.

    Attributes
    ----------
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
    marginal_log_likelihood : float
    first_smoothed : np.ndarray, shape (n_interior_state_bins,), or None
        Smoothed posterior of the first bin (None when the checkpointed
        driver did not return it).
    transition_statistics : np.ndarray or None
        Expected discrete-transition statistics, when they are estimated.
    """

    acausal_state_probabilities: np.ndarray
    marginal_log_likelihood: float
    first_smoothed: np.ndarray | None
    transition_statistics: np.ndarray | None


@dataclass(frozen=True)
class _EMFinalEStep:
    """The final E-step's outputs.

    With ``result_path``, ``stored`` is the lazily loaded store and the
    posterior holds only the marginal log-likelihood and degenerate bins.
    """

    posterior: _PosteriorOutputs
    stored: xr.Dataset | None = None


def _require_observed_bins(is_missing: np.ndarray | None) -> None:
    """Reject EM when every decode bin is missing: there is nothing to fit.

    Run anyway, EM moves the transitions on the prior dynamics alone (on a
    simulated recording every stay probability went to 1) and then reports
    non-convergence, pointing at ``max_iter`` instead of the mask.

    Parameters
    ----------
    is_missing : np.ndarray of bool, shape (n_bins,), or None
        Missing bins from ``_missing_bins``.

    Raises
    ------
    DataError
        If every bin is missing.
    """
    if is_missing is not None and is_missing.size and bool(np.all(is_missing)):
        msg = (
            "Cannot estimate parameters: every decode time bin is missing, so "
            "EM has no observations to fit"
        )
        raise DataError(
            msg,
            data_name="is_missing",
            hint=(
                "Check that is_missing is True only for missing bins, and that "
                "position is tracked (finite) during time_edges, since bins "
                "without tracked position are missing too."
            ),
        )


class _EMMixin:
    """EM methods of ``_DetectorBase``."""

    # mypy declares an attribute on the class whose method assigns it, even
    # through ``self: _DetectorBase``; these match ``_DetectorBase``.
    em_monotonicity_violations_: list[int]
    transition_time_bin_width_: float | None

    def _refit_encoding_model(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        weights: np.ndarray,
        encoding_fit_arguments: dict[str, Any],
    ) -> None:
        """EM M-step: refit the encoding model with posterior ``weights``.

        Parameters
        ----------
        weights : np.ndarray, shape (n_time_position,)
            Posterior weight of each time sample in the refit.
        encoding_fit_arguments : dict
            The other keyword arguments of ``_fit_encoding_model``: the
            estimation's position, spike data, training mask and labels.
        """
        self._fit_encoding_model(**encoding_fit_arguments, weights=weights)

    def _fit_em_model(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        fit: Callable[[str], Any],
        transition_representation: TransitionRepresentation,
        n_time: int,
        *,
        structured_required: bool,
    ) -> None:
        """Run EM's internal fit and, for "auto", choose dense or structured EM.

        "auto" cannot build some transitions in bounded form that the dense
        constructor accepts (opaque custom constructors with deferred graph
        distances, or dense fallback blocks over ``max_dense_transition_bytes``).
        Unless a structured-EM option requires the operator, those run dense EM;
        otherwise ``_choose_em_representation`` picks the path.

        Parameters
        ----------
        fit : callable
            ``fit(transition_representation)`` runs the detector's ``fit``.
        transition_representation : str
            Resolved by ``_resolve_em_options``.
        n_time : int
            Decode bins.
        structured_required : bool
            Whether ``chunk_size``, ``result_path`` or ``replay_cache_bytes``
            was given.
        """
        from non_local_detector.transition_operators import (
            DenseTransitionBudgetError,
            UnsupportedTransitionError,
        )

        try:
            fit(transition_representation)
        except (UnsupportedTransitionError, DenseTransitionBudgetError) as error:
            if transition_representation != "auto" or structured_required:
                raise
            logger.info("EM with transition_representation='auto': dense (%s)", error)
            fit("dense")
            return
        if transition_representation == "auto" and not structured_required:
            self._choose_em_representation(n_time)

    def _choose_em_representation(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase", n_time: int
    ) -> None:
        """Run EM dense when it is faster and fits; otherwise keep the operator.

        Called after an ``"auto"`` fit. Dense EM is chosen when the float32
        interior transition is at most ``_AUTO_DENSE_TRANSITION_BYTES``, the
        full dense transition it stores (every state bin, float64, as the
        dense constructor builds it) fits ``max_dense_transition_bytes``, and
        dense EM's arrays fit in half the device memory, or half the available
        host memory on CPU (see ``_dense_em_memory``; unchecked when unknown).
        The dense matrix comes from the fitted operator, which matches the
        dense constructor. When memory alone rules dense EM out, it warns:
        structured EM recomputes the likelihood on every E-step, which made EM
        about 9x slower per E-step on a 567k-bin clusterless track decode.

        Parameters
        ----------
        n_time : int
            Decode bins.
        """
        state_space = self.state_space_
        n_bins = int(np.count_nonzero(state_space.is_track_interior))
        working, limit, where = _dense_em_memory(n_time, n_bins)
        n_total = state_space.n_state_bins
        dense_preferred = (
            n_bins * n_bins * 4 <= _AUTO_DENSE_TRANSITION_BYTES
            and n_total * n_total * 8 <= state_space.max_dense_transition_bytes
        )
        dense_fits = limit is None or working <= limit // 2
        use_dense = dense_preferred and dense_fits
        if dense_preferred and not dense_fits:
            assert limit is not None  # dense_fits is True whenever limit is None
            warnings.warn(
                'EM with transition_representation="auto" runs structured EM '
                f"because dense EM needs about {working / 1e9:.1f} GB of {where} "
                f"memory for {n_time} time bins x {n_bins} state bins, more than "
                f"half of the {limit / 1e9:.1f} GB available. Structured EM "
                "bounds memory by recomputing the likelihood on every E-step, "
                "which can make EM several times slower. The estimate is "
                "approximate: if dense EM fits, pass "
                'transition_representation="dense" to estimate_parameters.',
                ParameterWarning,
                stacklevel=user_stacklevel(),
            )
        logger.info(
            "EM with transition_representation='auto': %s (%d interior state "
            "bins, %d time bins)",
            "dense" if use_dense else "structured",
            n_bins,
            n_time,
        )
        if use_dense:
            self.state_space_ = state_space.densified()

    def _run_em(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        time_edges: ArrayLike,
        log_likelihood_args: tuple[Any, ...] | None = None,
        is_missing: np.ndarray | None = None,
        *,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = False,
        max_iter: int = 100,
        tolerance: float = 1e-5,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: ReturnOutputs = None,
        min_encoding_local_mass: float = 1.0,
        min_encoding_local_ess: float = 1.0,
        chunk_size: int | None = None,
        result_path: str | Path | None = None,
        replay_cache_bytes: int | None = None,
        encoding_fit_arguments: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        """EM engine: estimate the parameters from a likelihood's arguments.

        The detectors' ``estimate_parameters`` validate and fit the data, then
        call this with the ``compute_log_likelihood`` arguments. It estimates
        the initial conditions, the discrete transitions and, optionally, the
        encoding model with the Expectation-Maximization algorithm.

        Parameters
        ----------
        time_edges : array_like, shape (n_bins + 1,)
            Uniform decode bin edges. Results have one row per bin, at the bin
            centers, and the bin width is recorded with the learned transitions
            as ``transition_time_bin_width_``.
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default None.
        is_missing : np.ndarray, shape (n_bins,), optional
            Boolean array indicating missing data, by default None.
        estimate_initial_conditions : bool, optional
            Whether to estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Whether to estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Re-fit the place fields in each M-step, weighted by the summed
            posterior of observations with ``is_local=True``, independently
            of their state names, by default False. Without local observations
            this update is skipped. Every state shares the encoding model, so
            weighting it by local observations alone is not
            an exact M-step: it can lower the penalized log-likelihood, which
            EM reports as a monotonicity violation and never counts as
            convergence. Off by default for that reason.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100. EM stops earlier once it converges.
        tolerance : float, optional
            Convergence tolerance for the EM algorithm, in nats per time
            bin, by default 1e-5. EM stops when the penalized log-likelihood
            (marginal log-likelihood plus the discrete-transition log prior)
            rises by less than ``tolerance`` times the number of time bins in
            one iteration, without decreasing. When EM estimates the discrete
            transitions, the last M-step must also change every transition
            probability by less than 1e-4.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
            Caching is disabled automatically when ``n_chunks > 1``; asking to
            keep the likelihood there (``store_log_likelihood=True`` or
            ``return_outputs='log_likelihood'``) makes the FINAL E-step
            accumulate the per-chunk rows into the full (n_time, n_state_bins)
            array instead. In-loop E-steps never retain a chunk array.
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihood_, by default False.
        n_chunks : int, optional
            Dense EM only: the number of row chunks the filter and smoother
            process, by default 1. Structured EM uses ``chunk_size`` (rows per
            chunk) instead.
        return_outputs : str, collection of str, or None, optional
            Controls which optional outputs are returned.

            Options:
            - 'filter' : causal (filtered) posterior and state probabilities
            - 'predictive' : predictive state probabilities
            - 'predictive_posterior' : predictive posterior over state bins
            - 'log_likelihood' : log likelihoods
            - 'all' : all of the above

            A name adds the same outputs alone or in a list or set.

            By default None (only smoother posterior and state probabilities).
        min_encoding_local_mass : float, optional
            Minimum sum of local state weights required to update the encoding
            model. If the total mass is below this threshold, the encoding
            M-step is skipped for that iteration. By default 1.0.
        min_encoding_local_ess : float, optional
            Minimum effective sample size (ESS) of local state weights required
            to update the encoding model. ESS = sum(w)^2 / sum(w^2). If ESS
            is below this threshold, the encoding M-step is skipped. By default 1.0.
        chunk_size : int, optional
            Rows per checkpoint chunk when the model was fit with structured
            transitions (``transition_representation="structured"`` or
            ``"auto"``). Such models run every E-step on the checkpointed
            driver: one forward pass that keeps a checkpoint per chunk and a
            backward replay, so working memory scales with ``chunk_size``
            times the state bins instead of the recording length, and the
            M-steps use the expected transition statistics accumulated in
            the replay. The final E-step returns the same outputs as dense
            EM, held in memory unless ``result_path`` is given. ``n_chunks``
            must then be 1 and ``cache_likelihood`` has no effect: each
            uncached chunk evaluates the likelihood twice (see
            ``replay_cache_bytes``). By default None, which sizes chunks so one
            chunk of float32 likelihoods takes about 256 MiB.
        result_path : str or Path, optional
            Structured-transition EM only: write the final E-step's outputs
            (the requested ``return_outputs`` plus the acausal posterior) to
            an atomic directory store here, as checkpointed ``predict`` does,
            and return it lazily, so host memory does not hold the full
            posterior. ``marginal_log_likelihoods`` in the returned attributes
            lists every iteration; the store itself records the final one.
            ``store_log_likelihood`` cannot be combined with it.
        replay_cache_bytes : int, optional
            Structured-transition EM only: host memory for keeping each
            E-step's forward-pass rows (filtered rows, or likelihoods when the
            final outputs need them), so the backward replay does not filter
            and evaluate them again. Chunks beyond the budget are recomputed.
            By default None, which uses 16 GiB or half the available host
            memory (counting a container's limit), whichever is smaller.
        encoding_fit_arguments : dict, optional
            Keyword arguments of ``_fit_encoding_model``, except ``weights``,
            that the encoding M-step refits with. Required with
            ``estimate_encoding_model=True`` when a state is local (raises
            ``ValidationError`` before any attribute is set otherwise); the
            detectors' ``estimate_parameters`` pass their data. By default
            None.

        Returns
        -------
        results : xr.Dataset
            Results of the decoding, including posterior probabilities.
            ``attrs["marginal_log_likelihoods"]`` and ``attrs["em_objectives"]``
            give each E-step's marginal log-likelihood and penalized
            log-likelihood (plus the discrete-transition log prior), the
            quantity EM's convergence and monotonicity checks use. Each check
            compares two E-steps under the prior of the M-step between them:
            ``check_converged(em_objectives[k + 1], em_previous_objectives[k],
            tolerance, n_time_bins=n_time_bins)`` reproduces it, with
            ``attrs["em_previous_objectives"]`` the earlier E-step's objective
            under that prior and ``n_time_bins`` the number of decode bins
            not marked missing. With
            ``discrete_transition_prior_weight > 0`` or REML smoothness the
            prior changes with each E-step, so ``em_previous_objectives[k]``
            can differ from ``em_objectives[k]``. With ``max_iter=0`` there is no M-step
            prior; the sole objective is the final marginal log-likelihood
            and ``em_previous_objectives`` is empty. A ``result_path`` store
            keeps these attributes, so ``load_results`` returns them too.
        """
        settings = self._em_settings(
            time_edges,
            log_likelihood_args,
            is_missing,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            n_chunks=n_chunks,
            return_outputs=return_outputs,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
            chunk_size=chunk_size,
            result_path=result_path,
            replay_cache_bytes=replay_cache_bytes,
            encoding_fit_arguments=encoding_fit_arguments,
        )
        self._em_reset_fitted_attributes()
        state = _EMState()

        while not state.converged and (state.n_iter < settings.max_iter):
            e_step = self._em_e_step(settings, state)

            # Maximization step
            logger.info("Maximization step...")
            if settings.estimate_encoding_model:
                self._em_update_encoding_model(settings, state, e_step)
            if settings.estimate_discrete_transition:
                log_prior_before, log_prior_after = (
                    self._em_update_discrete_transitions(settings, state, e_step)
                )
            else:
                log_prior_before = log_prior_after = 0.0
            if settings.estimate_initial_conditions:
                self._em_update_initial_conditions(e_step)

            self._em_record_iteration(
                settings,
                state,
                e_step.marginal_log_likelihood,
                log_prior_before,
                log_prior_after,
            )

        if (
            settings.estimate_discrete_transition
            and settings.covariate_transitions
            and state.n_iter > 0
        ):
            self._em_update_transition_covariance(settings, state)

        # converged_ is set after the final E-step, which evaluates the
        # returned parameters and can still find a decrease.
        self.n_iter_ = int(state.n_iter)

        if not state.converged:
            self._em_warn_not_converged(settings, state)

        final = self._em_final_e_step(settings, state)
        traces = state.traces(final.posterior.marginal_log_likelihood)

        # The final E-step's degenerate (all-impossible) timesteps, from its
        # forward pass rather than the returned log-likelihoods, so they are
        # complete even when caching is disabled (n_chunks > 1), where the
        # returned log-likelihoods are None.
        self.degenerate_timesteps_ = final.posterior.degenerate_indices

        self._em_check_final_e_step(settings, state, traces["em_objectives"][-1])

        if settings.store_log_likelihood:
            self.log_likelihood_ = final.posterior.log_likelihood

        if settings.result_path is not None:
            # The store already holds the EM attributes and traces.
            assert final.stored is not None
            return final.stored
        return self._em_results(settings, final.posterior, traces)

    def _em_settings(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        time_edges: ArrayLike,
        log_likelihood_args: tuple[Any, ...] | None,
        is_missing: np.ndarray | None,
        *,
        estimate_initial_conditions: bool,
        estimate_discrete_transition: bool,
        estimate_encoding_model: bool,
        max_iter: int,
        tolerance: float,
        cache_likelihood: bool,
        store_log_likelihood: bool,
        n_chunks: int,
        return_outputs: ReturnOutputs,
        min_encoding_local_mass: float,
        min_encoding_local_ess: float,
        chunk_size: int | None,
        result_path: str | Path | None,
        replay_cache_bytes: int | None,
        encoding_fit_arguments: dict[str, Any] | None,
    ) -> _EMSettings:
        """Validate ``_run_em``'s arguments and resolve its options.

        Sets no attribute, so a validation failure leaves the detector as it
        was. Parameters are those of ``_run_em``.

        Returns
        -------
        settings : _EMSettings
        """
        time_edges, time_bin_width = _validate_estimation_arguments(
            time_edges,
            n_chunks,
            return_outputs,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
        )
        if (
            estimate_encoding_model
            and encoding_fit_arguments is None
            and any(obs.is_local for obs in self.observation_models)
        ):
            msg = "estimate_encoding_model=True needs encoding_fit_arguments"
            raise ValidationError(
                msg,
                expected="the _fit_encoding_model arguments the encoding M-step refits with",
                got="encoding_fit_arguments=None",
                hint="Call estimate_parameters, which passes its data, or pass "
                "estimate_encoding_model=False.",
            )
        state_space = self.state_space_
        checkpointed = state_space.transition_operator is not None
        _validate_em_options(
            structured=checkpointed,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            result_path=result_path,
            store_log_likelihood=store_log_likelihood,
            replay_cache_bytes=replay_cache_bytes,
        )
        # Resolved once and recorded on the results, so users can tell which
        # EM path ran and with what chunks.
        em_attrs: dict[str, Any]
        if checkpointed:
            if chunk_size is None:
                chunk_size = default_chunk_size(
                    cast("int", np.count_nonzero(state_space.is_track_interior)),
                    np.float32,
                )
            if replay_cache_bytes is None:
                replay_cache_bytes = default_replay_cache_bytes()
            em_attrs = {
                "em_transition_representation": "structured",
                "em_chunk_size": int(chunk_size),
                "em_replay_cache_bytes": int(replay_cache_bytes),
            }
        else:
            em_attrs = {
                "em_transition_representation": "dense",
                "em_n_chunks": n_chunks,
            }
            if n_chunks == 1:
                _warn_if_dense_em_exceeds_memory(
                    len(time_edges) - 1,
                    int(np.count_nonzero(state_space.is_track_interior)),
                )
        # Subclasses have already combined ``is_missing`` with the bins
        # without tracked position via ``_missing_bins``.
        time_centers = decode_bin_centers(time_edges, 0, len(time_edges) - 1)
        # Units of ``tolerance``. Subclasses refuse to estimate with every bin
        # missing (``_require_observed_bins``), so at least one is observed.
        n_missing = 0 if is_missing is None else int(np.count_nonzero(is_missing))
        n_observed_bins = len(time_centers) - n_missing

        if log_likelihood_args is None:
            log_likelihood_args = ()

        # Normalize return_outputs to canonical set
        requested_outputs = _normalize_return_outputs(return_outputs)

        # A requested log likelihood has to be materialized. Unchunked that
        # means caching the full-time evaluation; with n_chunks > 1 caching is
        # disabled by _predict and the final E-step accumulates the per-chunk
        # rows instead (see _em_final_e_step).
        if (
            "log_likelihood" in requested_outputs
            and not cache_likelihood
            and n_chunks == 1
        ):
            cache_likelihood = True

        covariate_transitions = not self.discrete_transition_type.is_stationary
        # The discrete M-step's statistics: per-row responses for covariate
        # transitions, accumulated in the smoother's scan (structured EM's
        # replay or dense EM's smoother); else counts, which dense EM sums from
        # the posteriors in one product over all rows.
        transition_statistics: TransitionStatistics | None = None
        if estimate_discrete_transition:
            transition_statistics = "responses" if covariate_transitions else "counts"
        # Data-adaptive pseudo-counts (prior_weight > 0 on a stationary
        # transition) scale with each E-step's expected counts. Any other prior
        # is fixed, so an M-step's input has the log prior the previous M-step
        # computed for its output.
        prior_tracks_counts = (
            estimate_discrete_transition
            and not covariate_transitions
            and bool(np.any(np.asarray(self.discrete_transition_prior_weight) > 0))
        )
        # REML chooses each covariate row's smoothness in its M-step, so that
        # M-step's prior is known only after it, and EM checks the M-step's
        # input against its output under it.
        prior_set_in_m_step = (
            estimate_discrete_transition
            and covariate_transitions
            and isinstance(self.discrete_transition_type.smoothness, str)
        )
        interior_state_ind: np.ndarray | None = None
        interior_continuous_transition: np.ndarray | None = None
        if transition_statistics == "counts" and not checkpointed:
            interior_state_bins = state_space.is_track_interior
            interior_state_ind = state_space.interior_state_ind
            interior_continuous_transition = state_space.continuous_transitions[
                np.ix_(interior_state_bins, interior_state_bins)
            ]
        prior = (
            self.discrete_transition_concentration,
            self.discrete_transition_stickiness,
            self.discrete_transition_regularization,
            self.discrete_transition_prior_weight,
        )
        # Frozen rows are not estimated (covariate-dependent rows are skipped,
        # stationary rows restored after each M-step), so they do not enter the
        # objective it maximizes.
        frozen_rows = self._frozen_discrete_transition_rows_mask_
        free_rows = None if frozen_rows is None else ~frozen_rows
        return _EMSettings(
            time_edges=time_edges,
            time_bin_width=time_bin_width,
            time_centers=time_centers,
            log_likelihood_args=log_likelihood_args,
            is_missing=is_missing,
            n_observed_bins=n_observed_bins,
            checkpointed=checkpointed,
            chunk_size=chunk_size,
            replay_cache_bytes=replay_cache_bytes,
            n_chunks=n_chunks,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            result_path=result_path,
            requested_outputs=requested_outputs,
            em_attrs=em_attrs,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
            covariate_transitions=covariate_transitions,
            transition_statistics=transition_statistics,
            prior_tracks_counts=prior_tracks_counts,
            prior_set_in_m_step=prior_set_in_m_step,
            smoothness_setting=self.discrete_transition_type.smoothness,
            prior=prior,
            frozen_rows=frozen_rows,
            free_rows=free_rows,
            interior_state_ind=interior_state_ind,
            interior_continuous_transition=interior_continuous_transition,
            encoding_fit_arguments=encoding_fit_arguments,
        )

    def _em_reset_fitted_attributes(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
    ) -> None:
        """Start EM's result attributes and drop an earlier run's likelihood."""
        # Initialize the EM-result attributes together, after validation, so a
        # validation failure leaves NONE of them set (all-or-nothing).
        # converged_/n_iter_ are updated to their final values after the loop;
        # degenerate_timesteps_ is filled from the final E-step's likelihoods.
        self.converged_ = False
        self.n_iter_ = 0
        self.em_monotonicity_violations_ = []
        self.degenerate_timesteps_ = np.array([], dtype=int)
        # Any likelihood stored by an EARLIER run describes that run's data and
        # encoding model, not this one's. Drop it now, and reuse within this run
        # only a likelihood computed inside this run
        # (``_EMState.run_log_likelihoods``).
        self._invalidate_stored_log_likelihood()
        # Fitted initial-condition distribution starts at the user-supplied
        # constructor argument; overwritten with the fitted distribution in
        # each M-step when ``estimate_initial_conditions`` is True.
        self.discrete_initial_conditions_ = np.asarray(self.discrete_initial_conditions)

    def _em_transition_log_prior(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        statistics: np.ndarray,
        coefficients: np.ndarray | None = None,
    ) -> float:
        """Return the log prior of the current transitions under one M-step's prior.

        With covariate ``coefficients``, the prior of those.
        """
        return _transition_log_prior(
            statistics,
            self.discrete_state_transitions_,
            (
                self.discrete_transition_coefficients_
                if coefficients is None
                else coefficients
            ),
            self.discrete_transition_design_matrix_,
            *settings.prior,
            free_rows=settings.free_rows,
            smoothness=(
                0.0
                if self.discrete_transition_smoothness_ is None
                else self.discrete_transition_smoothness_
            ),
            smoothness_matrix=self.discrete_transition_smoothness_matrix_,
        )

    def _em_e_step(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        state: _EMState,
    ) -> _EMEStep:
        """One in-loop E-step, dense or checkpointed.

        Updates ``state.run_log_likelihoods`` with a likelihood the next
        E-step can reuse.
        """
        logger.info("Expectation step...")
        if settings.checkpointed:
            e_step = self._checkpointed_e_step(
                settings.time_edges,
                settings.log_likelihood_args,
                settings.is_missing,
                cast("int", settings.chunk_size),  # resolved when checkpointed
                transition_statistics=settings.transition_statistics,
                replay_cache_bytes=cast("int", settings.replay_cache_bytes),
            )
            acausal_state_probabilities = e_step.dataset[
                "acausal_state_probabilities"
            ].values
            marginal_log_likelihood = e_step.marginal_log_likelihood
            first_smoothed = e_step.first_smoothed
            log_likelihood = None
            transition_statistics = e_step.transition_statistics
        else:
            # Device arrays.
            posterior = self._predict(
                time_edges=settings.time_edges,
                log_likelihood_args=settings.log_likelihood_args,
                is_missing=settings.is_missing,
                cache_likelihood=settings.cache_likelihood,
                log_likelihoods=state.run_log_likelihoods,
                n_chunks=settings.n_chunks,
                transition_responses=settings.transition_statistics == "responses",
            )
            acausal_state_probabilities = cast(
                "np.ndarray", posterior.acausal_state_probabilities
            )
            marginal_log_likelihood = posterior.marginal_log_likelihood
            log_likelihood = posterior.log_likelihood
            first_smoothed = np.asarray(posterior.acausal_posterior[0])
            # Covariate responses come from the smoother. Stationary counts
            # need only these posteriors, so take them now; the device arrays
            # are released on return, before the M-step (an encoding refit
            # allocates on the device too).
            transition_statistics = (
                estimate_discrete_transition_counts_from_factorized_posteriors(
                    posterior.causal_posterior,
                    posterior.predictive_posterior,
                    posterior.acausal_posterior,
                    cast("np.ndarray", settings.interior_continuous_transition),
                    self.discrete_state_transitions_,
                    cast("np.ndarray", settings.interior_state_ind),
                )
                if settings.transition_statistics == "counts"
                else posterior.transition_responses
            )
        # Reusable by the next iteration: the likelihood depends only on the
        # data and the encoding model, and the encoding M-step clears this when
        # it updates the encoding model. ``None`` when the likelihood was not
        # materialized (n_chunks > 1 disables caching), i.e. recompute.
        if settings.cache_likelihood:
            state.run_log_likelihoods = log_likelihood
        return _EMEStep(
            acausal_state_probabilities=acausal_state_probabilities,
            marginal_log_likelihood=marginal_log_likelihood,
            first_smoothed=first_smoothed,
            transition_statistics=transition_statistics,
        )

    def _em_update_encoding_model(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        state: _EMState,
        e_step: _EMEStep,
    ) -> None:
        """Run the encoding M-step: refit with the local observations' posterior.

        Skipped without local observations, or when their posterior mass or
        effective sample size is below the thresholds.
        """
        local_state_indices = [
            index
            for index, observation in enumerate(self.observation_models)
            if observation.is_local
        ]

        if local_state_indices:
            # _em_settings requires them whenever a local state is refit.
            encoding_fit_arguments = cast(
                "dict[str, Any]", settings.encoding_fit_arguments
            )
            logger.info("Estimating encoding model...")
            local_state_weights = e_step.acausal_state_probabilities[
                :, local_state_indices
            ].sum(axis=1)
            # Weights live on the decode bins; the encoding model is
            # fit on the position samples. Interpolate from the bin
            # centers even when the counts match: equal lengths do not
            # mean equal coordinates. Samples outside the decoded edges
            # have no posterior and get no weight.
            position_time = encoding_fit_arguments.get("position_time")
            if position_time is not None:
                position_time = np.asarray(position_time)
                local_state_weights = np.where(
                    (position_time >= settings.time_edges[0])
                    & (position_time <= settings.time_edges[-1]),
                    np.interp(
                        position_time, settings.time_centers, local_state_weights
                    ),
                    0.0,
                )

            # ESS / mass guard: skip encoding update when local
            # state has effectively disappeared
            mass = float(np.sum(local_state_weights))
            sum_sq = float(np.sum(local_state_weights**2))
            ess = (mass**2 / sum_sq) if sum_sq > 0 else 0.0

            if (
                mass < settings.min_encoding_local_mass
                or ess < settings.min_encoding_local_ess
            ):
                logger.info(
                    "Skipping encoding update: local mass=%.4g, "
                    "ESS=%.4g (thresholds: mass>=%.4g, ESS>=%.4g)",
                    mass,
                    ess,
                    settings.min_encoding_local_mass,
                    settings.min_encoding_local_ess,
                )
            else:
                # Re-fit the encoding model using the posterior weights.
                # Known limitation (issue #31): the encoding model (the
                # emission) is shared across states but is re-fit here
                # using only the Local-state marginal as weights, so this
                # M-step does not strictly maximize the full-model
                # expected complete-data log-likelihood and can break the
                # EM monotonicity guarantee. The monotonicity-violation /
                # final-E-step warnings now surface this when it occurs.
                self._refit_encoding_model(local_state_weights, encoding_fit_arguments)

                # The encoding model changed, so the likelihood
                # computed from the previous one is stale.
                # (``fit_encoding_model`` also drops any stored
                # ``log_likelihood_``.)
                state.run_log_likelihoods = None

    def _em_update_discrete_transitions(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        state: _EMState,
        e_step: _EMEStep,
    ) -> tuple[float, float]:
        """Discrete-transition M-step.

        Records the step's statistics in ``state.transition_statistics`` and,
        when the prior changes with each E-step, how far the transitions moved
        in ``state.transition_change``.

        Returns
        -------
        log_prior_before : float
            Log prior of the M-step's input under its prior.
        log_prior_after : float
            Log prior of the M-step's output under its prior.
        """
        statistics = e_step.transition_statistics
        # Requested from the E-step whenever the transition is estimated.
        assert statistics is not None
        state.transition_statistics = statistics
        log_prior_before = (
            self._em_transition_log_prior(settings, statistics)
            if state.n_iter == 0 or settings.prior_tracks_counts
            else state.next_log_prior
        )
        coefficients_before = self.discrete_transition_coefficients_
        # Covariate transitions are compared from their coefficients, without
        # copying (n_time, n_states, n_states).
        if not settings.covariate_transitions:
            transitions_before = np.array(self.discrete_state_transitions_)
        m_step = _discrete_transition_from_statistics(
            statistics,
            self.discrete_transition_coefficients_,
            self.discrete_transition_design_matrix_,
            *settings.prior,
            frozen_rows=settings.frozen_rows,
            smoothness=settings.smoothness_setting,
            smoothness_matrix=self.discrete_transition_smoothness_matrix_,
            initial_smoothness=self.discrete_transition_smoothness_,
        )
        self.discrete_state_transitions_ = m_step.transition
        self.discrete_transition_coefficients_ = m_step.coefficients
        if settings.covariate_transitions:
            self.discrete_transition_smoothness_ = m_step.smoothness
        if settings.prior_set_in_m_step:
            log_prior_before = self._em_transition_log_prior(
                settings, statistics, coefficients_before
            )
        # The learned transitions are per bin at this width.
        self.transition_time_bin_width_ = settings.time_bin_width
        self._transition_time_bin_width_tolerance = _uniformity_tolerance(
            settings.time_edges, settings.time_bin_width
        )
        # Restore frozen rows: overwrite the M-step result with
        # the initial snapshot for rows the user wants pinned.
        # This is a constrained M-step; EM converges to the
        # constrained fixed point rather than the unconstrained
        # maximizer, which is the desired behavior here.
        if self._frozen_discrete_transition_rows_baseline_ is not None:
            self.discrete_state_transitions_[
                self._frozen_discrete_transition_rows_mask_
            ] = self._frozen_discrete_transition_rows_baseline_
        log_prior_after = self._em_transition_log_prior(settings, statistics)
        if settings.covariate_transitions:
            # Covariate transitions have a design matrix and coefficients.
            assert self.discrete_transition_design_matrix_ is not None
            assert coefficients_before is not None
            assert self.discrete_transition_coefficients_ is not None
            state.transition_change = _max_transition_change(
                self.discrete_transition_design_matrix_,
                coefficients_before,
                self.discrete_transition_coefficients_,
            )
        else:
            moved = self.discrete_state_transitions_ - transitions_before
            state.transition_change = float(np.max(np.abs(moved)))
        return log_prior_before, log_prior_after

    def _em_update_initial_conditions(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase", e_step: _EMEStep
    ) -> None:
        """Initial-condition M-step: the first bin's smoothed posterior."""
        self.initial_conditions_[self.state_space_.is_track_interior] = (
            e_step.first_smoothed
        )
        self.discrete_initial_conditions_ = np.asarray(
            e_step.acausal_state_probabilities[0]
        )

        expanded_discrete_ic = e_step.acausal_state_probabilities[0][
            self.state_space_.state_ind
        ]
        is_zero = np.isclose(expanded_discrete_ic, 0.0)
        safe_discrete_ic = np.where(is_zero, 1.0, expanded_discrete_ic)
        self.continuous_initial_conditions_ = np.where(
            is_zero,
            0.0,
            self.initial_conditions_ / safe_discrete_ic,
        )

    def _em_record_iteration(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        state: _EMState,
        marginal_log_likelihood: float,
        log_prior_before: float,
        log_prior_after: float,
    ) -> None:
        """Record an iteration's objective and check convergence.

        Warns and records the iteration in ``em_monotonicity_violations_``
        when the penalized log-likelihood decreased.

        Parameters
        ----------
        settings : _EMSettings
            Resolved options of this EM run.
        state : _EMState
            Mutable EM loop state; updated in place.
        marginal_log_likelihood : float
            This iteration's E-step marginal log-likelihood.
        log_prior_before, log_prior_after : float
            Log prior of this iteration's M-step input and output under its
            prior (0.0 when the transitions are not estimated).
        """
        logger.info("Computing stats..")
        state.n_iter += 1

        state.marginal_log_likelihoods.append(marginal_log_likelihood)
        # This E-step evaluated the last M-step's output, under that
        # M-step's prior; the first E-step has only its own M-step's.
        state.objectives.append(
            marginal_log_likelihood
            + (state.next_log_prior if state.n_iter > 1 else log_prior_before)
        )
        if state.n_iter > 1:
            state.objective_change = (
                state.objectives[-1] - state.previous_objectives[-1]
            )
            state.objective_converged, state.is_increasing = check_converged(
                state.objectives[-1],
                state.previous_objectives[-1],
                settings.tolerance,
                n_time_bins=settings.n_observed_bins,
            )
            state.converged = state.objective_converged and (
                not settings.estimate_discrete_transition
                or state.transition_change < _TRANSITION_CHANGE_TOLERANCE
            )

            if not state.is_increasing:
                # Surfaced via warnings.warn(NumericalWarning) to match the
                # other EM/GMM/GLM fit-quality diagnostics (decode-time
                # degenerate-timestep warnings use logger.warning instead;
                # see core._report_degenerate_and_nan_timesteps).
                warnings.warn(
                    f"EM iteration {state.n_iter}: penalized log-likelihood "
                    f"decreased from {state.previous_objectives[-1]:.6f} to "
                    f"{state.objectives[-1]:.6f} "
                    f"(change={state.objective_change:.6e}; "
                    f"marginal log-likelihood + transition log prior). "
                    f"A MAP M-step cannot lower it; re-fitting the encoding "
                    f"model inside EM, a covariate transition fit stopping "
                    f"early, or a numerical instability can.",
                    NumericalWarning,
                    stacklevel=user_stacklevel(),
                )
                self.em_monotonicity_violations_.append(state.n_iter)

            logger.info(
                "iteration %s, likelihood: %s, penalized: %s, change: %s%s",
                state.n_iter,
                state.marginal_log_likelihoods[-1],
                state.objectives[-1],
                state.objective_change,
                (
                    f", transition change: {state.transition_change}"
                    if settings.estimate_discrete_transition
                    else ""
                ),
            )
        else:
            logger.info(
                "iteration %s, likelihood: %s",
                state.n_iter,
                state.marginal_log_likelihoods[-1],
            )
        # The next E-step evaluates this M-step's output; it is checked
        # against this E-step, both under this M-step's prior.
        state.previous_objectives.append(marginal_log_likelihood + log_prior_before)
        state.next_log_prior = log_prior_after

    def _em_update_transition_covariance(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        state: _EMState,
    ) -> None:
        """Covariance of the covariate transition coefficients after EM."""
        # The last M-step's statistics and output (covariate transitions
        # have a design matrix, coefficients and smoothness).
        assert self.discrete_transition_coefficients_ is not None
        assert self.discrete_transition_design_matrix_ is not None
        assert self.discrete_transition_smoothness_ is not None
        self.discrete_transition_coefficient_covariance_ = (
            _covariate_transition_covariance(
                self.discrete_transition_coefficients_,
                self.discrete_transition_design_matrix_,
                np.asarray(state.transition_statistics),
                self.discrete_transition_concentration,
                self.discrete_transition_stickiness,
                self.discrete_transition_regularization,
                self.discrete_transition_smoothness_,
                self.discrete_transition_smoothness_matrix_,
                settings.frozen_rows,
            )
        )

    def _em_warn_not_converged(self, settings: _EMSettings, state: _EMState) -> None:
        """Warn that EM stopped at ``max_iter``, with advice for the cause."""
        if not state.is_increasing:
            advice = (
                "The penalized log-likelihood decreased in the last "
                "iteration, and EM never counts a decrease as convergence, "
                "so a looser tolerance will not help; see the monotonicity "
                "warnings above for likely causes."
            )
        elif state.n_iter > 1 and state.objectives[-1] == -np.inf:
            advice = (
                "The penalized log-likelihood is -inf, so its change cannot "
                "be judged; this usually means degenerate timesteps (every "
                "state impossible), listed in degenerate_timesteps_ after "
                "the fit."
            )
        elif state.objective_converged:
            advice = (
                "The penalized log-likelihood changed by less than "
                "tolerance, but the last M-step must also move every "
                "transition probability by less than "
                f"{_TRANSITION_CHANGE_TOLERANCE:g}, and it "
                f"moved one by {state.transition_change:.3e}; consider "
                "increasing max_iter."
            )
        else:
            advice = "Consider increasing max_iter or relaxing tolerance."
        warnings.warn(
            f"EM did not converge after max_iter={settings.max_iter} iterations "
            f"(final penalized log-likelihood change per time bin="
            f"{state.objective_change / settings.n_observed_bins:.3e}, "
            f"tolerance={settings.tolerance:.3e}). Fitted parameters may be biased. "
            f"{advice}",
            NumericalWarning,
            stacklevel=user_stacklevel(),
        )

    def _em_final_e_step(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        state: _EMState,
    ) -> _EMFinalEStep:
        """E-step on the fitted parameters, with the requested outputs.

        Run after the last M-step so the returned posterior reflects the final
        fitted parameters. Without it, the returned arrays are the E-step
        computed *before* the last M-step and disagree with predict() on the
        same data.
        """
        logger.info("Final E-step after EM...")
        requested_outputs = settings.requested_outputs
        if settings.result_path is not None:
            em_attrs = settings.em_attrs
            stored, degenerate_indices = self._predict_checkpointed(
                time_edges=settings.time_edges,
                log_likelihood_args=settings.log_likelihood_args,
                is_missing=settings.is_missing,
                discrete_transitions=None,
                output_mode="spatial",
                chunk_size=settings.chunk_size,
                result_path=settings.result_path,
                checkpoint_dir=None,
                selected_intervals=None,
                requested_outputs=requested_outputs,
                max_read_bytes=DEFAULT_MAX_READ_BYTES,
                replay_cache_bytes=settings.replay_cache_bytes,
                # The store is complete before its marginal log-likelihood
                # reaches this method, so it writes the traces itself.
                em_attrs=lambda marginal: {**em_attrs, **state.traces(marginal)},
            )
            return _EMFinalEStep(
                _PosteriorOutputs(
                    marginal_log_likelihood=float(
                        stored.attrs["marginal_log_likelihoods"][-1]
                    ),
                    degenerate_indices=degenerate_indices,
                ),
                stored=stored,
            )
        if settings.checkpointed:
            outputs = _driver_outputs(
                requested_outputs
                | ({"log_likelihood"} if settings.store_log_likelihood else set()),
                spatial=True,
            )
            final = self._checkpointed_e_step(
                settings.time_edges,
                settings.log_likelihood_args,
                settings.is_missing,
                cast("int", settings.chunk_size),  # resolved when checkpointed
                outputs=outputs,
                replay_cache_bytes=cast("int", settings.replay_cache_bytes),
            )
            values = {name: final.dataset[name].values for name in final.dataset}
            return _EMFinalEStep(
                _PosteriorOutputs(
                    marginal_log_likelihood=final.marginal_log_likelihood,
                    degenerate_indices=final.diagnostics["degenerate_indices"],
                    acausal_posterior=values["acausal_posterior"],
                    acausal_state_probabilities=values["acausal_state_probabilities"],
                    causal_state_probabilities=values.get("causal_state_probabilities"),
                    predictive_state_probabilities=values.get(
                        "predictive_state_probabilities"
                    ),
                    log_likelihood=values.get("log_likelihood"),
                    causal_posterior=values.get("causal_posterior"),
                    predictive_posterior=values.get("predictive_posterior"),
                )
            )
        posterior = self._predict(
            time_edges=settings.time_edges,
            log_likelihood_args=settings.log_likelihood_args,
            is_missing=settings.is_missing,
            cache_likelihood=settings.cache_likelihood,
            log_likelihoods=state.run_log_likelihoods,
            n_chunks=settings.n_chunks,
            # Final E-step only: this is the likelihood the caller can ask to
            # keep. With n_chunks > 1 caching is unavailable, so the chunked
            # path accumulates the per-chunk rows (the documented T x N
            # exception) instead of returning None. The in-loop E-steps never
            # need the array, so they keep the per-chunk allocation.
            accumulate_log_likelihoods=(
                settings.store_log_likelihood or "log_likelihood" in requested_outputs
            ),
        )
        return _EMFinalEStep(posterior)

    def _em_check_final_e_step(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        state: _EMState,
        final_objective: float,
    ) -> None:
        """Check the final E-step like an in-loop iteration and set ``converged_``.

        There is no check when max_iter=0. The final E-step counts as
        iteration n_iter + 1, and a decrease here means the fit has not
        converged.
        """
        final_is_increasing = True
        if state.n_iter > 0:
            _, final_is_increasing = check_converged(
                final_objective,
                state.previous_objectives[-1],
                settings.tolerance,
                n_time_bins=settings.n_observed_bins,
            )
            if not final_is_increasing:
                self.em_monotonicity_violations_.append(state.n_iter + 1)
                decrease = state.previous_objectives[-1] - final_objective
                warnings.warn(
                    f"Final E-step penalized log-likelihood (marginal log-likelihood "
                    f"+ transition log prior) decreased by {decrease:.3e}. This "
                    f"indicates an inconsistency between the E-step and the M-step "
                    f"output and is unusual; the returned posterior may not reflect "
                    f"the fitted parameters. converged_ is False and iteration "
                    f"{state.n_iter + 1} is recorded in em_monotonicity_violations_.",
                    NumericalWarning,
                    stacklevel=user_stacklevel(),
                )
        self.converged_ = bool(state.converged) and final_is_increasing

    def _em_results(  # type: ignore[misc]
        self: "non_local_detector.models.base._DetectorBase",
        settings: _EMSettings,
        final: _PosteriorOutputs,
        traces: dict[str, list[float]],
    ) -> xr.Dataset:
        """Return the in-memory results of EM: the final E-step and EM's traces."""
        # Every final E-step without a result_path returns them.
        assert final.acausal_state_probabilities is not None
        em_attrs = dict(settings.em_attrs)
        em_attrs["em_objectives"] = np.asarray(traces["em_objectives"])
        em_attrs["em_previous_objectives"] = np.asarray(
            traces["em_previous_objectives"]
        )
        results = convert_results_to_xarray(
            self,
            settings.time_centers,
            np.asarray(final.acausal_posterior),
            final.acausal_state_probabilities,
            traces["marginal_log_likelihoods"],
            time_edges=settings.time_edges,
            is_missing=settings.is_missing,
            **final.requested_results(settings.requested_outputs),
        )
        results.attrs.update(em_attrs)
        return results
