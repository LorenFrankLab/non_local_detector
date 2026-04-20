import abc
import copy
import inspect
import pickle
import warnings
from logging import getLogger

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy.ndimage  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
import xarray as xr
from patsy import DesignMatrix, build_design_matrices  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from track_linearization import get_linearized_position  # type: ignore[import-untyped]

from non_local_detector import _validation as val
from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    Uniform,
)
from non_local_detector.core import (
    check_converged,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    most_likely_sequence,
    most_likely_sequence_covariate_dependent,
)
from non_local_detector.discrete_state_transitions import (
    _estimate_discrete_transition,
    centered_softmax_forward,
    predict_discrete_state_transitions,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ConfigurationError, ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
    predict_no_spike_log_likelihood,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.types import (
    ContinuousInitialConditions,
    ContinuousTransitions,
    DiscreteTransitions,
    Environments,
    Observations,
    StateNames,
    Stickiness,
)

logger = getLogger(__name__)
sklearn.set_config(print_changed_only=False)

_DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS = {
    "waveform_std": 24.0,
    "position_std": 6.0,
    "block_size": 10_000,
}
_DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS = {
    "position_std": 6.0,
    "block_size": 10_000,
}

# Valid options for return_outputs parameter
VALID_OUTPUTS: set[str] = {
    "filter",
    "predictive",
    "predictive_posterior",
    "log_likelihood",
    "all",
}

# Mapping of single string options to sets of outputs
OUTPUT_INCLUDES: dict[str, set[str]] = {
    "filter": {"filter"},
    "predictive": {"predictive", "predictive_posterior"},
    "predictive_posterior": {"predictive_posterior"},
    "log_likelihood": {"log_likelihood"},
    "all": {"filter", "predictive", "predictive_posterior", "log_likelihood"},
}


def _snapshot_encoding_model(encoding_model: dict | None) -> dict | None:
    """Create a restorable snapshot of the encoding model.

    Deep-copies numpy/jax arrays but keeps non-picklable objects (e.g.
    patsy DesignInfo, KDEModel) by reference so that ``copy.deepcopy``
    failures are avoided.
    """
    if encoding_model is None:
        return None
    snapshot = {}
    for key, entry in encoding_model.items():
        entry_copy = {}
        for k, v in entry.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                entry_copy[k] = (
                    np.array(v) if isinstance(v, np.ndarray) else jnp.array(v)
                )
            else:
                entry_copy[k] = v
        snapshot[key] = entry_copy
    return snapshot


def _normalize_frozen_discrete_transition_rows(
    frozen_rows: np.ndarray | list[int] | tuple[int, ...] | None,
    n_states: int,
) -> np.ndarray | None:
    """Normalize ``frozen_discrete_transition_rows`` to a boolean row mask.

    Accepts a sequence of integer row indices, a boolean mask of shape
    ``(n_states,)``, or ``None``/empty (no frozen rows). Returns a
    boolean mask of shape ``(n_states,)`` where ``True`` marks rows to
    freeze, or ``None`` if no rows are frozen.

    Parameters
    ----------
    frozen_rows : np.ndarray, list of int, tuple of int, or None
        Row indices to freeze, a boolean mask, or None/empty.
    n_states : int
        Total number of discrete states.

    Returns
    -------
    np.ndarray of bool, shape (n_states,), or None

    Raises
    ------
    ValueError
        If the input has the wrong shape, wrong dtype, or out-of-range
        indices.
    """
    if frozen_rows is None:
        return None
    arr = np.asarray(frozen_rows)
    if arr.size == 0:
        return None
    if arr.dtype == bool:
        if arr.shape != (n_states,):
            raise ValueError(
                f"frozen_discrete_transition_rows boolean mask must have "
                f"shape ({n_states},), got {arr.shape}"
            )
        return arr.copy()
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"frozen_discrete_transition_rows must be integer indices or "
            f"a boolean mask, got dtype {arr.dtype}"
        )
    if arr.ndim != 1:
        raise ValueError(
            f"frozen_discrete_transition_rows index array must be 1D, "
            f"got shape {arr.shape}"
        )
    if np.any(arr < 0) or np.any(arr >= n_states):
        raise ValueError(
            f"frozen_discrete_transition_rows indices must be in "
            f"[0, {n_states}), got {arr.tolist()}"
        )
    mask = np.zeros(n_states, dtype=bool)
    mask[arr] = True
    return mask


def _validate_covariate_time_length(
    predicted_transitions: np.ndarray, time: np.ndarray
) -> None:
    """Validate that covariate-derived transitions match decode time length.

    Parameters
    ----------
    predicted_transitions : np.ndarray, shape (n_covariate_time, n_states, n_states)
    time : np.ndarray, shape (n_decode_time,)

    Raises
    ------
    ValueError
        If the number of covariate time steps does not match decode time.
    """
    n_covariate_time = predicted_transitions.shape[0]
    n_decode_time = len(time)
    if n_covariate_time != n_decode_time:
        raise ValueError(
            f"Covariate data has {n_covariate_time} time steps but "
            f"decode time has {n_decode_time} time steps. "
            f"These must match."
        )


def _normalize_return_outputs(
    return_outputs: str | list[str] | set[str] | None,
) -> set[str]:
    """Convert return_outputs to canonical set of output names.

    Parameters
    ----------
    return_outputs : str, list of str, set of str, or None
        Controls which optional outputs are included.

    Returns
    -------
    set of str
        Normalized set containing any of: 'filter', 'predictive',
        'predictive_posterior', 'log_likelihood'

    Raises
    ------
    ValueError
        If return_outputs contains invalid option names.
    TypeError
        If return_outputs is not str, list, set, or None.
    """
    if return_outputs is None:
        return set()

    if isinstance(return_outputs, str):
        if return_outputs not in VALID_OUTPUTS:
            raise ValueError(
                f"Invalid return_outputs='{return_outputs}'. "
                f"Must be one of: {sorted(VALID_OUTPUTS)}"
            )
        return OUTPUT_INCLUDES.get(return_outputs, {return_outputs})

    if isinstance(return_outputs, list | set):
        outputs_set = set(return_outputs)
        invalid = outputs_set - VALID_OUTPUTS
        if invalid:
            raise ValueError(
                f"Invalid outputs: {sorted(invalid)}. "
                f"Valid options are: {sorted(VALID_OUTPUTS)}"
            )
        # Expand 'all' if present
        if "all" in outputs_set:
            return OUTPUT_INCLUDES["all"]
        return outputs_set

    raise TypeError(
        f"return_outputs must be str, list of str, set of str, or None. "
        f"Got {type(return_outputs).__name__}"
    )


class _DetectorBase(BaseEstimator, abc.ABC):
    """Base class for detector objects."""

    # Type annotations for attributes assigned during fit
    discrete_state_transitions_: np.ndarray
    discrete_transition_coefficients_: np.ndarray | None
    discrete_transition_design_matrix_: DesignMatrix | None

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: (
            np.ndarray | list[int] | tuple[int, ...] | None
        ) = None,
        local_position_std: float | None = None,
    ) -> None:
        """
        Initialize the _DetectorBase class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Sampling frequency, by default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        discrete_transition_prior_weight : float or np.ndarray, optional
            Dimensionless weight for data-adaptive prior scaling. When > 0,
            the Dirichlet prior pseudo-counts are scaled by expected transition
            counts, making regularization strength approximately invariant to
            temporal resolution. Can be a scalar (same weight for all rows) or
            an array of shape ``(n_states,)`` for per-row control. When 0.0
            (default), the fixed-count prior from concentration and stickiness
            is used directly.
        frozen_discrete_transition_rows : np.ndarray, list, tuple, or None, optional
            Rows of the discrete transition matrix to freeze during EM
            re-estimation. Accepts integer row indices, a boolean mask of
            shape ``(n_states,)``, or ``None``/empty (no frozen rows,
            default). Frozen rows are snapshotted from the initial
            transition matrix and restored after every M-step, so they
            retain the values specified by ``discrete_transition_type``
            regardless of what the data suggests. Use this to pin
            structural states (e.g., a No-Spike self-transition) that
            should not be learned from data. Only applies to stationary
            (2D) transition matrices.
        local_position_std : float or None, optional
            Controls the Local state's spatial representation:

            - ``None`` (default): legacy behavior. Local state occupies a
              single bin; the likelihood is evaluated at the animal's
              exact interpolated position (continuous).
            - ``0.0``: delta-kernel multi-bin local. Local state spans all
              position bins, with all mass concentrated at the single bin
              containing the animal (one-hot). Likelihood is evaluated at
              bin centers (discrete).
            - ``> 0``: multi-bin local with a Gaussian kernel of standard
              deviation ``local_position_std`` (same units as ``position``,
              typically centimeters) on the shortest-path track-graph
              distance; models spatial uncertainty.
        """
        # Validate all parameters early (Tier 1 & 2)
        self._validate_initial_conditions(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            continuous_transition_types,
            discrete_transition_stickiness,
        )
        self._validate_probability_distributions(discrete_initial_conditions)
        self._validate_numerical_parameters(
            discrete_transition_concentration,
            discrete_transition_regularization,
            sampling_frequency,
            no_spike_rate,
        )
        self._validate_discrete_transition_type(
            discrete_transition_type, len(discrete_initial_conditions)
        )

        # Initial conditions parameters
        self.discrete_initial_conditions = discrete_initial_conditions
        self.continuous_initial_conditions_types = continuous_initial_conditions_types

        # Discrete state transition parameters
        self.discrete_transition_concentration = discrete_transition_concentration
        self.discrete_transition_stickiness = discrete_transition_stickiness
        self.discrete_transition_regularization = discrete_transition_regularization
        self.discrete_transition_type = discrete_transition_type
        self.discrete_transition_prior_weight = discrete_transition_prior_weight
        # Normalize and validate frozen rows against n_states up front
        self.frozen_discrete_transition_rows = frozen_discrete_transition_rows
        self._frozen_discrete_transition_rows_mask_ = (
            _normalize_frozen_discrete_transition_rows(
                frozen_discrete_transition_rows,
                n_states=len(discrete_initial_conditions),
            )
        )

        # Continuous state transition parameters
        self.continuous_transition_types = continuous_transition_types

        # Environment parameters
        self.environments = self._initialize_environments(environments)
        self.infer_track_interior = infer_track_interior

        # Observation model parameters
        self.observation_models = self._initialize_observation_models(
            observation_models, continuous_transition_types, environments
        )

        # State names
        self.state_names = self._initialize_state_names(
            state_names, discrete_initial_conditions
        )

        self.sampling_frequency = sampling_frequency
        self.no_spike_rate = no_spike_rate

        # Local position uncertainty parameter
        if local_position_std is not None and local_position_std < 0:
            raise ValidationError(
                "local_position_std must be non-negative",
                expected="float >= 0 or None",
                got=str(local_position_std),
                hint=(
                    "Use None for legacy single-bin local (likelihood at "
                    "the animal's exact interpolated position), 0.0 for "
                    "multi-bin local with a delta kernel at the animal's "
                    "bin, or > 0 for a Gaussian kernel."
                ),
            )
        self.local_position_std = local_position_std

    def _validate_position_dimensionality(
        self, position: np.ndarray, context: str = "fit"
    ) -> None:
        """Check that position shape matches the environment type.

        When any environment has a ``track_graph``, positions must be
        raw 2D ``(x, y)`` coordinates — the detector linearizes
        internally via ``get_position_at_time``. Passing already-
        linearized 1D position in this case causes the internal
        linearization step to silently produce garbage coordinates,
        which mis-centers the local kernel and non-local penalty.

        This check raises a clear ``ValidationError`` when the shape
        does not match the environment type, rather than allowing
        silent numerical corruption downstream.

        Parameters
        ----------
        position : np.ndarray
            Position array to validate.
        context : str, optional
            Name of the calling context (``"fit"`` or ``"predict"``)
            for error messages.
        """
        is_1d = position.ndim == 1 or (position.ndim == 2 and position.shape[1] == 1)
        if not is_1d:
            return

        has_track_graph = any(env.track_graph is not None for env in self.environments)
        if has_track_graph:
            raise ValidationError(
                "Environment has a track_graph but position is 1D "
                f"(shape {position.shape}). Track-graph environments "
                "require raw 2D (x, y) position — the detector "
                "linearizes internally. Passing pre-linearized 1D "
                "position silently corrupts internal coordinates.",
                expected="shape (n_time, 2) or (n_time, n_dims) with n_dims >= 2",
                got=f"shape {position.shape}",
                hint=(
                    "If you already called get_linearized_position() to "
                    "get 1D position, pass the raw 2D (x, y) array "
                    "instead — the detector linearizes via the track_graph "
                    "during both fit() and predict()."
                ),
                example=(
                    "    # Raw 2D position, not pre-linearized\n"
                    f"    detector.{context}(..., position=position_2d)"
                ),
            )

    def _compute_non_local_position_penalty(
        self, time, position_time, position, environment
    ):
        """Compute position-dependent penalty for non-local states.

        Returns a ``(n_time, n_interior_bins)`` array of log-likelihood
        penalties that suppress non-local likelihood near the animal's
        current position. The penalty is a negative Gaussian centered on
        the animal's position.

        Uses topology-aware distances via
        :meth:`Environment.get_distances_to_interior_bins`:

        - 1D track graph: shortest-path distance along the linearized
          track (handles multi-arm mazes correctly so the penalty at a
          bin reflects its graph distance, not its linearized coordinate
          difference).
        - N-D environment: shortest-path distance on ``track_graphDD``
          (respects holes and obstacles — the penalty follows the
          reachable interior, not straight-line Euclidean).
        - No precomputed distance matrix: Euclidean fallback.

        NaN/inf distances (off-track animal position, unreachable bins)
        are treated as infinite distance so the penalty at those bins is
        zero — consistent with the intuition "cannot compute → no
        penalty applied".

        Parameters
        ----------
        time : jnp.ndarray, shape (n_time,)
            Decoding time bins.
        position_time : jnp.ndarray, shape (n_time_position,)
            Position sampling times.
        position : jnp.ndarray, shape (n_time_position, n_position_dims)
            Position samples.
        environment : Environment
            The spatial environment with place_bin_centers_ and
            is_track_interior_.

        Returns
        -------
        penalty : jnp.ndarray, shape (n_time, n_interior_bins)
            Non-positive penalty values (to be added to log-likelihood).
        """
        from non_local_detector.likelihoods.common import get_position_at_time

        animal_pos = get_position_at_time(position_time, position, time, environment)

        # Topology-aware distance (graph shortest path when available).
        dist = environment.get_distances_to_interior_bins(np.asarray(animal_pos))

        # NaN (off-track animal) and inf (unreachable bins) → treat as
        # infinitely far so the penalty evaluates to zero at those cells,
        # avoiding NaN propagation into the log-likelihood.
        sq_dist = jnp.nan_to_num(jnp.asarray(dist) ** 2, nan=jnp.inf, posinf=jnp.inf)

        return -self.non_local_position_penalty * jnp.exp(
            -0.5 * sq_dist / (self.non_local_penalty_std**2)
        )

    def _compute_local_position_kernel(
        self,
        time: jnp.ndarray,
        position_time: jnp.ndarray,
        position: jnp.ndarray,
        environment: "Environment",
    ) -> jnp.ndarray:
        """Compute log position kernel for the local state.

        Dispatches to one of two paths based on ``self.local_position_std``:

        - ``== 0``: delta kernel. ``log(n_bins)`` at the animal's bin,
          ``-inf`` elsewhere (one-hot).
        - ``> 0``: Gaussian kernel over shortest-path track-graph
          distance (Euclidean fallback when no graph is fitted), then
          normalized so ``exp(log_kernel)`` sums to ``n_interior_bins``
          per time step. Unreachable bins on disconnected graph
          components get zero probability naturally via ``exp(-inf)``.

        Gap-bin positions (e.g. positions exactly on an arm-boundary
        edge of a linearized track) are snapped to the nearest interior
        bin by :meth:`Environment.get_bin_ind`, so the kernel is always
        defined on a real interior bin.

        NaN animal positions (tracking dropouts) fall back to a uniform
        kernel.

        Both paths satisfy the invariant ``exp(log_kernel).sum(axis=1)
        == n_bins``. Combined with the multi-bin local state's
        ``1/n_bins`` uniform continuous initial conditions, this makes
        the effective per-bin prior exactly ``exp(log_kernel)`` — so
        the total Local state mass after the HMM forward step equals
        ``p_local * P(spikes | animal_bin)`` in the delta limit.

        In the sharp-sigma limit, the Gaussian concentrates at the
        animal's bin and matches the delta-kernel behavior numerically.

        Parameters
        ----------
        time : jnp.ndarray, shape (n_time,)
            Decoding time bins.
        position_time : jnp.ndarray, shape (n_time_position,)
            Position sampling times.
        position : jnp.ndarray, shape (n_time_position, n_position_dims)
            Position samples.
        environment : Environment
            The spatial environment.

        Returns
        -------
        log_kernel : jnp.ndarray, shape (n_time, n_interior_bins)
            Log kernel. exp(log_kernel) sums to n_interior_bins per
            time step (not 1).
        """
        from non_local_detector.likelihoods.common import get_position_at_time

        assert self.local_position_std is not None  # narrowing for type checker

        animal_pos = get_position_at_time(position_time, position, time, environment)

        n_bins = int(environment.is_track_interior_.sum())
        log_n_bins = jnp.log(jnp.array(n_bins, dtype=jnp.float32))

        # Detect NaN positions (from gaps in position data)
        if animal_pos.ndim == 1:
            nan_mask = jnp.isnan(animal_pos)
        else:
            nan_mask = jnp.any(jnp.isnan(animal_pos), axis=-1)

        # Delta kernel path: σ=0 concentrates the Local state on the
        # single bin containing the animal. Emits log(n_bins) at that bin
        # and -inf elsewhere, so after the multi-bin state's 1/n_bins
        # uniform continuous IC, the total Local mass equals p_local ×
        # P(spikes | animal_bin_center). Environment.get_bin_ind snaps
        # gap-bin assignments to the nearest interior bin, so the only
        # remaining case needing uniform fallback is a truly NaN animal
        # position (tracking dropout).
        if self.local_position_std == 0:
            interior_bin_indices = np.where(environment.is_track_interior_.ravel())[0]
            # Replace NaN animal positions with 0 before get_bin_ind (it
            # has no NaN guard); nan_mask below masks those rows.
            safe_animal_pos = np.nan_to_num(np.asarray(animal_pos), nan=0.0)
            animal_bin_inds = environment.get_bin_ind(safe_animal_pos)
            is_animal_bin = jnp.asarray(
                animal_bin_inds[:, np.newaxis] == interior_bin_indices[np.newaxis, :]
            )
            log_kernel = jnp.where(is_animal_bin, log_n_bins, -jnp.inf)
            log_kernel = jnp.where(nan_mask[:, jnp.newaxis], 0.0, log_kernel)
            return log_kernel

        # Gaussian kernel path (σ > 0). Uses topology-aware distance via
        # Environment.get_distances_to_interior_bins(). After the
        # get_bin_ind snap, the animal bin is always interior, so the
        # distance-matrix row is always finite (no NaN rows). Unreachable
        # bins on disconnected graph components stay as inf, which through
        # exp(-0.5 * inf² / σ²) naturally yields zero probability there —
        # the Gaussian concentrates on the reachable interior.
        dist = environment.get_distances_to_interior_bins(np.asarray(animal_pos))
        sq_dist = jnp.asarray(dist) ** 2

        # Compute log-kernel with -inf at unreachable bins. jnp.where
        # avoids inf/NaN propagation through the arithmetic ops below.
        reachable = jnp.isfinite(sq_dist)
        log_kernel = jnp.where(
            reachable, -0.5 * sq_dist / (self.local_position_std**2), -jnp.inf
        )
        # Normalize to sum to n_bins (not 1) per time step. This compensates
        # for the 1/n_bins uniform continuous IC of the multi-bin local state
        # so the effective per-bin prior matches the kernel itself. Without
        # this scaling, non-local states dominate by a factor of n_bins.
        log_kernel = (
            log_kernel
            - jax.scipy.special.logsumexp(log_kernel, axis=1, keepdims=True)
            + log_n_bins
        )

        # NaN animal positions (tracking dropout) fall back to a uniform
        # kernel: each bin's exp value is 1, so log_kernel = 0 per bin.
        log_kernel = jnp.where(nan_mask[:, jnp.newaxis], 0.0, log_kernel)

        return log_kernel

    def _validate_initial_conditions(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        continuous_transition_types: ContinuousTransitions,
        discrete_transition_stickiness: Stickiness,
    ) -> None:
        """
        Validate the initial conditions.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.

        Raises
        ------
        ValidationError
            If the number of discrete initial conditions does not match the number of continuous initial conditions or transition types.
        """
        n_discrete = len(discrete_initial_conditions)
        n_continuous_init = len(continuous_initial_conditions_types)
        n_continuous_trans = len(continuous_transition_types)

        if n_discrete != n_continuous_init:
            raise ValidationError(
                "Mismatch between discrete initial conditions and continuous initial conditions",
                expected=f"{n_continuous_init} discrete initial condition(s) (one per continuous initial condition type)",
                got=f"{n_discrete} discrete initial condition(s)",
                hint="Each continuous initial condition type needs a corresponding discrete initial probability",
                example=f"    discrete_initial_conditions=np.array([{', '.join(['1.0/' + str(n_continuous_init)] * n_continuous_init)}])",
            )

        if n_discrete != n_continuous_trans:
            state_names_str = (
                f" ({self.state_names})"
                if hasattr(self, "state_names") and self.state_names
                else ""
            )
            raise ValidationError(
                "Mismatch between discrete initial conditions and continuous transition types",
                expected=f"{n_continuous_trans} discrete initial condition(s) (one per state{state_names_str})",
                got=f"{n_discrete} discrete initial condition(s)",
                hint="Each state needs an initial probability. Check that your discrete_initial_conditions array has the same length as continuous_transition_types list.",
                example=f"    # For {n_continuous_trans} states:\n    discrete_initial_conditions=np.array([{', '.join(['1.0/' + str(n_continuous_trans)] * n_continuous_trans)}])",
            )

        if not isinstance(discrete_transition_stickiness, float) and len(
            discrete_initial_conditions
        ) != len(discrete_transition_stickiness):
            raise ValidationError(
                f"Discrete transition stickiness must be set for all {n_discrete} states or be a single float",
                expected=f"Either a float or array of length {n_discrete}",
                got=f"Array of length {len(discrete_transition_stickiness)}",
                hint="Use a float for uniform stickiness across all states, or an array with one value per state",
                example=f"    discrete_transition_stickiness=0.0  # uniform\n    # OR\n    discrete_transition_stickiness=np.array([{', '.join(['0.0'] * n_discrete)}])  # per-state",
            )

    def _validate_probability_distributions(
        self, discrete_initial_conditions: np.ndarray
    ) -> None:
        """Validate that probability distributions sum to 1.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states

        Raises
        ------
        ValidationError
            If discrete_initial_conditions is not 1D, contains invalid values,
            or does not sum to 1
        DataError
            If array contains NaN or Inf values
        """
        # Check array is 1D
        val.ensure_array_1d(discrete_initial_conditions, "discrete_initial_conditions")

        # Check for NaN/Inf
        val.ensure_all_finite(
            discrete_initial_conditions, "discrete_initial_conditions"
        )

        # Check non-negative
        val.ensure_all_non_negative(
            discrete_initial_conditions, "discrete_initial_conditions"
        )

        # Check sums to 1
        val.ensure_probability_distribution(
            discrete_initial_conditions, "discrete_initial_conditions"
        )

    def _validate_numerical_parameters(
        self,
        discrete_transition_concentration: float,
        discrete_transition_regularization: float,
        sampling_frequency: float,
        no_spike_rate: float,
    ) -> None:
        """Validate numerical parameters are in valid ranges.

        Parameters
        ----------
        discrete_transition_concentration : float
            Concentration parameter (must be > 0)
        discrete_transition_regularization : float
            Regularization parameter (must be >= 0)
        sampling_frequency : float
            Sampling frequency in Hz (must be > 0)
        no_spike_rate : float
            No-spike rate (must be > 0)

        Raises
        ------
        ValidationError
            If any parameter is outside its valid range
        """
        val.ensure_positive_scalar(
            discrete_transition_concentration,
            "discrete_transition_concentration",
            minimum=0.0,
            strict=True,
        )

        val.ensure_positive_scalar(
            discrete_transition_regularization,
            "discrete_transition_regularization",
            minimum=0.0,
            strict=False,  # Can be exactly 0
        )

        val.ensure_positive_scalar(
            sampling_frequency,
            "sampling_frequency",
            minimum=0.0,
            strict=True,
        )

        val.ensure_positive_scalar(
            no_spike_rate,
            "no_spike_rate",
            minimum=0.0,
            strict=True,
        )

    def _validate_discrete_transition_type(
        self, discrete_transition_type, n_states: int
    ) -> None:
        """Validate discrete transition type configuration.

        Parameters
        ----------
        discrete_transition_type : DiscreteTransitions
            Discrete transition type object
        n_states : int
            Number of states in the model

        Raises
        ------
        ValidationError
            If transition matrix is invalid (wrong shape, doesn't sum to 1, etc.)
        DataError
            If transition matrix contains NaN or Inf
        """
        # Deferred to avoid circular import: base.py -> discrete_state_transitions.py -> base.py
        from non_local_detector.discrete_state_transitions import (
            DiscreteStationaryCustom,
        )

        # If it's a custom transition matrix, validate it
        if isinstance(discrete_transition_type, DiscreteStationaryCustom):
            matrix = discrete_transition_type.values

            # Check it's a numpy array
            val.ensure_ndarray(matrix, "discrete_transition_type.values")

            # Check for NaN/Inf
            val.ensure_all_finite(matrix, "discrete_transition_type.values")

            # Check square
            val.ensure_square_matrix(matrix, "discrete_transition_type.values")

            # Check correct size
            if matrix.shape[0] != n_states:
                raise ValidationError(
                    "Discrete transition matrix size must match number of states",
                    expected=f"matrix with shape ({n_states}, {n_states})",
                    got=f"matrix with shape {matrix.shape}",
                    hint=f"Your model has {n_states} states, so transition matrix needs {n_states}x{n_states} shape",
                )

            # Check non-negative
            val.ensure_all_non_negative(matrix, "discrete_transition_type.values")

            # Check values in [0, 1]
            val.ensure_in_range(matrix, "discrete_transition_type.values", 0.0, 1.0)

            # Check row-stochastic (rows sum to 1)
            val.ensure_stochastic_matrix(matrix, "discrete_transition_type.values")

    def _initialize_environments(
        self, environments: Environments
    ) -> tuple[Environment, ...]:
        """
        Initialize environments.

        Parameters
        ----------
        environments : Environments
            Environments in which the detector operates.

        Returns
        -------
        tuple[Environment, ...]
            Initialized environments as a tuple.
        """
        if environments is None:
            environments = (Environment(),)
        if not hasattr(environments, "__iter__"):
            environments = (environments,)
        return environments

    def _initialize_observation_models(
        self,
        observation_models: Observations,
        continuous_transition_types: ContinuousTransitions,
        environments: Environments,
    ) -> tuple[ObservationModel, ...]:
        """
        Initialize observation models.

        Parameters
        ----------
        observation_models : Observations
            Observation models for the detector.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        environments : Environments
            Environments in which the detector operates.

        Returns
        -------
        tuple[ObservationModel, ...]
            Initialized observation models as a tuple.
        """
        if observation_models is None:
            n_states = len(continuous_transition_types)
            env_name = environments[0].environment_name
            observation_models = (ObservationModel(env_name),) * n_states
        elif isinstance(observation_models, ObservationModel):
            observation_models = (observation_models,) * len(
                continuous_transition_types
            )
        return observation_models

    def _initialize_state_names(
        self, state_names: StateNames, discrete_initial_conditions: np.ndarray
    ) -> list[str]:
        """
        Initialize state names.

        Parameters
        ----------
        state_names : StateNames, optional
            Names of the states.
        discrete_initial_conditions : np.ndarray
            Initial conditions for discrete states.

        Returns
        -------
        state_names : list[str]

        Raises
        ------
        ValueError
            If the number of state names does not match the number of discrete initial conditions.
        """
        if state_names is None:
            state_names = [
                f"state {state_ind}"
                for state_ind in range(len(discrete_initial_conditions))
            ]
        if len(state_names) != len(discrete_initial_conditions):
            raise ValidationError(
                "Number of state names must match number of states",
                expected=f"{len(discrete_initial_conditions)} state name(s)",
                got=f"{len(state_names)} state name(s)",
                hint="Provide one name per state in your model",
                example=f"    state_names={state_names[: len(discrete_initial_conditions)]}  # truncate to {len(discrete_initial_conditions)}",
            )
        return state_names

    def initialize_environments(
        self, position: np.ndarray, environment_labels: np.ndarray | None = None
    ) -> None:
        """
        Fits the Environment class on the position data to get information about the spatial environment.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        environment_labels : np.ndarray, optional, shape (n_time,)
            Labels for each time points about which environment it corresponds to, by default None
        """
        for environment in self.environments:
            if environment_labels is None:
                is_environment = np.ones((position.shape[0],), dtype=bool)
            else:
                is_environment = environment_labels == environment.environment_name

            env_position = position[is_environment]
            if environment.track_graph is not None:
                # convert to 1D
                env_position = get_linearized_position(
                    env_position,
                    environment.track_graph,
                    edge_order=environment.edge_order,
                    edge_spacing=environment.edge_spacing,
                ).linear_position.to_numpy()

            environment.fit_place_grid(
                env_position, infer_track_interior=self.infer_track_interior
            )

    def initialize_state_index(self) -> None:
        """Initialize indices and parameters related to the combined state space.

        Determines the total number of bins across all discrete states (spatial
        bins for continuous states, 1 for discrete states like 'Local' or
        'No-Spike') and creates mappings between these combined bins and the
        original discrete states. Also identifies which combined bins correspond
        to the track interior.

        Attributes
        ----------
        n_discrete_states_ : int
            Total number of discrete states defined in the model.
        state_ind_ : np.ndarray, shape (n_total_bins,)
            Index mapping each combined bin to its corresponding discrete state index.
        n_state_bins_ : int
            Total number of bins across all states (sum of spatial bins for
            continuous states and 1 for discrete states). Referred to as
            `n_total_bins` in the shape description here for clarity.
        bin_sizes_ : np.ndarray, shape (n_discrete_states_,)
             Number of bins associated with each discrete state (e.g., number
             of place bins for spatial states, 1 for non-spatial states).
        is_track_interior_state_bins_ : np.ndarray, shape (n_total_bins,)
             Boolean array indicating if a combined state bin corresponds to the
             track interior. For non-spatial states, this is typically True.

        """
        self.n_discrete_states_ = len(self.state_names)
        bin_sizes = []
        state_ind = []
        is_track_interior = []
        for ind, obs in enumerate(self.observation_models):
            if obs.is_no_spike or (obs.is_local and self.local_position_std is None):
                bin_sizes.append(1)
                state_ind.append(np.full(1, ind, dtype=int))
                is_track_interior.append(np.ones(1, dtype=bool))
            else:
                environment = self._get_environment_by_name(obs.environment_name)
                if environment.place_bin_centers_ is not None:
                    bin_sizes.append(environment.place_bin_centers_.shape[0])
                    state_ind.append(np.full(bin_sizes[-1], ind, dtype=int))
                else:
                    raise ValueError("Environment place_bin_centers_ cannot be None")
                if environment.is_track_interior_ is not None:
                    is_track_interior.append(environment.is_track_interior_.ravel())
                else:
                    # Default fallback: all positions are track interior
                    is_track_interior.append(np.ones(bin_sizes[-1], dtype=bool))

        self.state_ind_ = np.concatenate(state_ind)
        self.n_state_bins_ = len(self.state_ind_)
        self.bin_sizes_ = np.array(bin_sizes)
        self.is_track_interior_state_bins_ = np.concatenate(is_track_interior)

    def initialize_initial_conditions(self) -> None:
        """Constructs the initial probability for the state and each spatial bin.

        Attributes
        ----------
        continuous_initial_conditions_ : np.ndarray, shape (n_state_bins,)
            Initial probability distribution over the bins within each state.
        initial_conditions_ : np.ndarray, shape (n_state_bins,)
            Overall initial probability distribution across all state bins.
        """
        logger.info("Fitting initial conditions...")
        from dataclasses import replace

        ic_parts = []
        for obs, cont_ic in zip(
            self.observation_models,
            self.continuous_initial_conditions_types,
            strict=False,
        ):
            # Multi-bin local uses full spatial initial conditions
            effective_obs = obs
            if obs.is_local and self.local_position_std is not None:
                effective_obs = replace(obs, is_local=False)
            ic_parts.append(
                cont_ic.make_initial_conditions(effective_obs, self.environments)
            )
        self.continuous_initial_conditions_ = np.concatenate(ic_parts)
        self.initial_conditions_ = (
            self.continuous_initial_conditions_
            * self.discrete_initial_conditions[self.state_ind_]
        )

    def initialize_continuous_state_transition(
        self,
        continuous_transition_types: ContinuousTransitions,
        position: np.ndarray | None = None,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
    ) -> None:
        """
        Constructs the transition matrices for the continuous states.

        Parameters
        ----------
        continuous_transition_types : ContinuousTransitions
            Types of transition models.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        is_training : np.ndarray, optional, shape (n_time,)
            Boolean array that determines what data to train the place fields on, by default None.
        encoding_group_labels : np.ndarray, shape (n_time,), optional
            If place fields should correspond to each state, label each time point with the group name, by default None.
        environment_labels : np.ndarray, shape (n_time,), optional
            If there are multiple environments, label each time point with the environment name, by default None.

        Attributes
        ----------
        continuous_state_transitions_ : np.ndarray, shape (n_state_bins, n_state_bins)
            Probability of transitioning between bins, assuming a transition between the corresponding discrete states occurs.
        continuous_transition_types : ContinuousTransitions
            Stores the continuous transition types used.
        """
        logger.info("Fitting continuous state transition...")

        self.continuous_transition_types = continuous_transition_types

        n_total_bins = len(self.state_ind_)
        self.continuous_state_transitions_ = np.zeros((n_total_bins, n_total_bins))

        for from_state, row in enumerate(self.continuous_transition_types):
            for to_state, transition in enumerate(row):
                inds = np.ix_(
                    self.state_ind_ == from_state, self.state_ind_ == to_state
                )

                if isinstance(transition, EmpiricalMovement):
                    if is_training is None:
                        if position is not None:
                            n_time = position.shape[0]
                            is_training = np.ones((n_time,), dtype=bool)
                        else:
                            raise ValueError(
                                "Position cannot be None when is_training is None"
                            )

                    if encoding_group_labels is None:
                        if position is not None:
                            n_time = position.shape[0]
                            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)
                        else:
                            raise ValueError(
                                "Position cannot be None when encoding_group_labels is None"
                            )

                    is_training = np.asarray(is_training).squeeze()
                    if position is None:
                        raise ValueError(
                            "position must not be None for EmpiricalMovement transitions"
                        )
                    self.continuous_state_transitions_[inds] = (
                        transition.make_state_transition(
                            self.environments,
                            position,
                            is_training,
                            encoding_group_labels,
                            environment_labels,
                        )
                    )
                else:
                    # Auto-upgrade Discrete() for multi-bin local states.
                    # Must happen BEFORE make_state_transition() to avoid
                    # (1,1) -> (n,n) broadcast corruption.
                    if (
                        isinstance(transition, Discrete)
                        and self.local_position_std is not None
                    ):
                        from_obs = self.observation_models[from_state]
                        to_obs = self.observation_models[to_state]
                        if from_obs.is_local or to_obs.is_local:
                            # Cross-environment transitions with multi-bin
                            # local are not supported in v1.
                            if from_obs.environment_name != to_obs.environment_name:
                                raise ValueError(
                                    f"Cross-environment Discrete() transition "
                                    f"between '{from_obs.environment_name}' and "
                                    f"'{to_obs.environment_name}' is not "
                                    f"supported with multi-bin local "
                                    f"(local_position_std is set). Use explicit "
                                    f"Uniform(environment_name=..., "
                                    f"environment2_name=...) instead."
                                )
                            local_obs = from_obs if from_obs.is_local else to_obs
                            transition = Uniform(
                                environment_name=local_obs.environment_name
                            )

                    n_row_bins = np.max(inds[0].shape)
                    n_col_bins = np.max(inds[1].shape)

                    if np.logical_and(n_row_bins == 1, n_col_bins > 1):
                        # transition from discrete to continuous
                        # ASSUME uniform for now
                        if hasattr(transition, "environment_name"):
                            environment = self._get_environment_by_name(
                                transition.environment_name
                            )
                        else:
                            raise ValueError(
                                "Transition must have an environment_name attribute for discrete to continuous transitions"
                            )
                        if environment.is_track_interior_ is not None:
                            self.continuous_state_transitions_[inds] = (
                                environment.is_track_interior_.ravel()
                                / environment.is_track_interior_.sum()
                            ).astype(float)
                        else:
                            # Default fallback: uniform transition
                            if environment.place_bin_centers_ is not None:
                                n_bins = environment.place_bin_centers_.shape[0]
                                self.continuous_state_transitions_[inds] = 1.0 / n_bins
                            else:
                                raise ValueError(
                                    "Environment must have place_bin_centers_ for discrete to continuous transitions"
                                )
                    elif n_row_bins > 1 and n_col_bins == 1:
                        # Spatial to non-spatial: each source bin transitions
                        # to the single target bin with probability 1.
                        self.continuous_state_transitions_[inds] = np.ones(
                            (n_row_bins, 1)
                        )
                    else:
                        self.continuous_state_transitions_[inds] = (
                            transition.make_state_transition(self.environments)
                        )

    def initialize_discrete_state_transition(
        self, covariate_data: pd.DataFrame | dict | None = None
    ) -> None:
        """
        Constructs the transition matrix for the discrete states.

        Parameters
        ----------
        covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Attributes
        ----------
        discrete_state_transitions_ : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)
            Probability of transitioning between discrete states. Shape is (n_states, n_states) if no covariates, otherwise depends on covariate data.
        discrete_transition_coefficients_ : np.ndarray or None
            Fitted coefficients for covariate-dependent transitions.
        discrete_transition_design_matrix_ : patsy.DesignMatrix or None
            Design matrix information for covariate-dependent transitions.
        """
        logger.info("Fitting discrete state transition")
        (
            self.discrete_state_transitions_,
            self.discrete_transition_coefficients_,
            self.discrete_transition_design_matrix_,
        ) = self.discrete_transition_type.make_state_transition(covariate_data)

        # Snapshot frozen rows from the initial matrix so the EM M-step
        # can restore them after every update. Only meaningful for
        # stationary (2D) transition matrices — if the user requests
        # frozen rows with a non-stationary transition type, warn.
        if self._frozen_discrete_transition_rows_mask_ is not None:
            if self.discrete_state_transitions_.ndim == 2:
                self._frozen_discrete_transition_rows_baseline_ = (
                    self.discrete_state_transitions_[
                        self._frozen_discrete_transition_rows_mask_
                    ].copy()
                )
            else:
                warnings.warn(
                    "frozen_discrete_transition_rows is set but the "
                    "discrete transition matrix is non-stationary "
                    f"(ndim={self.discrete_state_transitions_.ndim}). "
                    "Row freezing is only applied to stationary matrices "
                    "and will be ignored here.",
                    stacklevel=2,
                )
                self._frozen_discrete_transition_rows_baseline_ = None
        else:
            self._frozen_discrete_transition_rows_baseline_ = None

    def plot_discrete_state_transition(
        self,
        cmap: str = "Oranges",
        ax: matplotlib.axes.Axes | None = None,
        convert_to_seconds: bool = False,
        sampling_frequency: int = 1,
        covariate_data: pd.DataFrame | dict | None = None,
    ) -> None:
        """
        Plot heatmap of discrete transition matrix.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap, by default "Oranges".
        ax : matplotlib.axes.Axes, optional
            Plotting axis, by default plots to current axis.
        convert_to_seconds : bool, optional
            Convert the probabilities of state to expected duration of state, by default False.
        sampling_frequency : int, optional
            Number of samples per second, by default 1.
        covariate_data: dict or pd.DataFrame, optional
            Dictionary or DataFrame of covariate data, by default None.
            Keys are covariate names and values are 1D arrays.
        """

        if self.discrete_state_transitions_.ndim == 2:
            if ax is None:
                ax = plt.gca()

            if convert_to_seconds:
                discrete_state_transition = (
                    1 / (1 - self.discrete_state_transitions_)
                ) / sampling_frequency
                vmin, vmax, fmt = 0.0, None, "0.03f"
                label = "Seconds"
            else:
                discrete_state_transition = self.discrete_state_transitions_
                vmin, vmax, fmt = 0.0, 1.0, "0.03f"
                label = "Probability"

            sns.heatmap(
                data=discrete_state_transition,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                xticklabels=self.state_names,
                yticklabels=self.state_names,
                ax=ax,
                cbar_kws={"label": label},
            )
            ax.set_ylabel("Previous State", fontsize=12)
            ax.set_xlabel("Current State", fontsize=12)
            ax.set_title("Discrete State Transition", fontsize=16)
        else:
            discrete_transition_design_matrix = self.discrete_transition_design_matrix_
            discrete_transition_coefficients = self.discrete_transition_coefficients_
            state_names = self.state_names

            if discrete_transition_design_matrix is None:
                raise ValueError(
                    "discrete_transition_design_matrix_ is None for covariate-dependent prediction"
                )

            predict_matrix = build_design_matrices(
                [discrete_transition_design_matrix.design_info],
                covariate_data,  # type: ignore[union-attr]
            )[0]

            n_states = len(state_names)

            for covariate in covariate_data:  # type: ignore[union-attr]
                fig, axes = plt.subplots(
                    1, n_states, sharex=True, constrained_layout=True, figsize=(10, 5)
                )

                for from_state_ind, (ax, from_state) in enumerate(
                    zip(axes.flat, state_names, strict=False)
                ):
                    from_local_transition = centered_softmax_forward(
                        predict_matrix
                        @ discrete_transition_coefficients[:, from_state_ind]
                    )

                    ax.plot(covariate_data[covariate], from_local_transition)
                    ax.set_xlabel(covariate)
                    ax.set_ylabel("Prob.")
                    if from_state_ind == n_states - 1:
                        ax.legend(state_names)
                    ax.set_title(f"From {from_state}")
                fig.suptitle(f"Predicted transitions: {covariate}")

    def plot_continuous_state_transition(
        self, figsize_scaling: float = 1.5, vmax: float = 0.3
    ) -> None:
        """
        Plot heatmap of continuous state transition matrices.

        Parameters
        ----------
        figsize_scaling : float, optional
            Scaling factor for figure size, by default 1.5.
        vmax : float, optional
            Maximum value for color scale, by default 0.3.
        """
        GOLDEN_RATIO = 1.618

        fig, axes = plt.subplots(
            self.n_discrete_states_,
            self.n_discrete_states_,
            gridspec_kw={
                "width_ratios": self.bin_sizes_,
                "height_ratios": self.bin_sizes_,
            },
            constrained_layout=True,
            figsize=(
                self.n_discrete_states_ * figsize_scaling * GOLDEN_RATIO,
                (self.n_discrete_states_ * figsize_scaling),
            ),
        )

        try:
            for from_state, ax_row in enumerate(axes):
                for to_state, ax in enumerate(ax_row):
                    ind = np.ix_(
                        self.state_ind_ == from_state, self.state_ind_ == to_state
                    )
                    ax.pcolormesh(
                        self.continuous_state_transitions_[ind], vmin=0.0, vmax=vmax
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if to_state == 0:
                        ax.set_ylabel(
                            self.state_names[from_state],
                            rotation=0,
                            ha="right",
                            va="center",
                        )

                    if from_state == self.n_discrete_states_ - 1:
                        ax.set_xlabel(
                            self.state_names[to_state],
                            rotation=45,
                            ha="right",
                            va="top",
                            labelpad=1.0,
                        )
            fig.supylabel("From State")
            fig.supxlabel("To State")
        except TypeError:
            axes.pcolormesh(self.continuous_state_transitions_, vmin=0.0, vmax=vmax)
            axes.set_xticks([])
            axes.set_yticks([])

    def plot_initial_conditions(
        self, figsize_scaling: float = 1.5, vmax: float = 0.3
    ) -> None:
        """
        Plot heatmap of initial conditions.

        Parameters
        ----------
        figsize_scaling : float, optional
            Scaling factor for figure size, by default 1.5.
        vmax : float, optional
            Maximum value for color scale, by default 0.3.
        """
        GOLDEN_RATIO = 1.618
        fig, axes = plt.subplots(
            1,
            self.n_discrete_states_,
            gridspec_kw={"width_ratios": self.bin_sizes_},
            constrained_layout=True,
            figsize=(self.n_discrete_states_ * figsize_scaling * GOLDEN_RATIO, 1.1),
        )

        try:
            for state, ax in enumerate(axes):
                ind = self.state_ind_ == state
                ax.pcolormesh(
                    self.initial_conditions_[ind][:, np.newaxis], vmin=0.0, vmax=vmax
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(
                    self.state_names[state],
                    rotation=45,
                    ha="right",
                    va="top",
                    labelpad=0.0,
                )
        except TypeError:
            axes.pcolormesh(
                self.initial_conditions_[:, np.newaxis], vmin=0.0, vmax=vmax
            )
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xlabel(
                self.state_names[0],
                rotation=45,
                ha="right",
                va="top",
                labelpad=0.0,
            )

        fig.suptitle("Initial Conditions")

    def _fit(
        self,
        position: np.ndarray | None = None,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict | None = None,
    ) -> "_DetectorBase":
        """
        Fit the model to the data.

        Parameters
        ----------
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        is_training : np.ndarray, optional, shape (n_time,)
            Boolean array that determines what data to train the place fields on, by default None.
        encoding_group_labels : np.ndarray, optional, shape (n_time,)
            If place fields should correspond to each state, label each time point with the group name, by default None.
        environment_labels : np.ndarray, optional, shape (n_time,)
            If there are multiple environments, label each time point with the environment name, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Returns
        -------
        _DetectorBase
            Fitted model.
        """
        # Validate required parameters
        if position is None:
            raise ValidationError(
                "Missing required parameter: position",
                expected="position array with shape (n_time, n_dims)",
                got="None",
                hint="Provide the animal's position data during training",
                example="    detector.fit(position=position_train, spikes=spikes_train, time=time_train)",
            )

        # Tier 2: Validate data types and properties
        val.ensure_ndarray(position, "position")
        val.ensure_all_finite(position, "position")

        # Raises ValidationError if position is 1D but env has track_graph.
        self._validate_position_dimensionality(position, context="fit")

        if position.ndim == 1 or (position.ndim == 2 and position.shape[1] == 1):
            has_track_graph = any(
                env.track_graph is not None for env in self.environments
            )
            if not has_track_graph:
                warnings.warn(
                    "Position data appears to be 1D but no track_graph was "
                    "provided. This will be treated as a 1D open field. If "
                    "you are working with a linear track, you likely want to "
                    "pass 2D position (x, y) along with a track_graph, "
                    "edge_order, and edge_spacing in your Environment so "
                    "that the track topology is properly linearized. See the "
                    "Environment docstring for details.",
                    UserWarning,
                    stacklevel=3,
                )

        position = position[:, np.newaxis] if position.ndim == 1 else position
        self.initialize_environments(
            position=position, environment_labels=environment_labels
        )
        self.initialize_state_index()
        self.initialize_initial_conditions()
        self.initialize_discrete_state_transition(
            covariate_data=discrete_transition_covariate_data
        )
        self.initialize_continuous_state_transition(
            continuous_transition_types=self.continuous_transition_types,
            position=position,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
        )

        return self

    @abc.abstractmethod
    def compute_log_likelihood(self):
        """Compute the log likelihood. To be implemented by inheriting class."""

    def _predict(
        self,
        time: np.ndarray,
        log_likelihood_args: tuple = (),
        is_missing: np.ndarray | None = None,
        log_likelihoods: np.ndarray | None = None,
        cache_likelihood: bool = True,
        n_chunks: int = 1,
        discrete_state_transitions: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Compute the posterior probabilities.

        Parameters
        ----------
        time : np.ndarray, optional
            Time points for decoding, by default None
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default ()
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None
        log_likelihoods : np.ndarray, optional
            Precomputed log likelihoods, by default None
        cache_likelihood : bool, optional
            Whether to cache the log likelihoods, by default True
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        discrete_state_transitions : np.ndarray, shape (n_time, n_states, n_states) or None, optional
            Covariate-driven transition matrices to use instead of
            self.discrete_state_transitions_. When None, falls back to the
            fitted attribute. By default None.

        Returns
        -------
        acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
        acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
        marginal_log_likelihood : float
        causal_state_probabilities : np.ndarray, shape (n_time, n_states)
        predictive_state_probabilities : np.ndarray, shape (n_time, n_states)
        log_likelihoods : np.ndarray, shape (n_time, n_state_bins)
        causal_posterior : np.ndarray, shape (n_time, n_state_bins)
        predictive_posterior : np.ndarray, shape (n_time, n_state_bins)
        """
        # Disable caching when using multiple chunks (memory optimization)
        if n_chunks > 1 and cache_likelihood:
            logger.info("Disabling likelihood caching for chunked processing")
            cache_likelihood = False

        logger.info("Computing posterior...")
        is_track_interior = self.is_track_interior_state_bins_
        cross_is_track_interior = np.ix_(is_track_interior, is_track_interior)
        state_ind = self.state_ind_[is_track_interior]

        # Use provided transitions or fall back to fitted attribute
        discrete_transitions = (
            discrete_state_transitions
            if discrete_state_transitions is not None
            else self.discrete_state_transitions_
        )

        if discrete_transitions.ndim == 2:
            return chunked_filter_smoother(
                time=time,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                transition_matrix=(
                    self.continuous_state_transitions_[cross_is_track_interior]
                    * discrete_transitions[np.ix_(state_ind, state_ind)]
                ),
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=cache_likelihood,
            )
        else:
            return chunked_filter_smoother_covariate_dependent(
                time=time,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                discrete_transition_matrix=discrete_transitions,
                continuous_transition_matrix=self.continuous_state_transitions_[
                    cross_is_track_interior
                ],
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                n_chunks=n_chunks,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=cache_likelihood,
            )

    def fit_predict(self) -> xr.Dataset:
        """Fit the model and predict the posterior probabilities. To be implemented by inheriting class."""
        raise NotImplementedError

    @abc.abstractmethod
    def fit_encoding_model(self):
        """Fit the encoding model. To be implemented by inheriting class."""

    @staticmethod
    def _apply_encoding_damping(
        new_model: dict,
        old_model: dict,
        damping: float,
    ) -> None:
        """Blend new encoding model place fields with old ones in-place.

        For each key present in both models, computes:
            place_fields = (1 - damping) * new + damping * old
        and recomputes no_spike_part_log_likelihood accordingly.
        """
        for key in new_model:
            if key not in old_model:
                continue
            new_entry = new_model[key]
            old_entry = old_model[key]
            if "place_fields" not in new_entry or "place_fields" not in old_entry:
                continue
            blended = (1 - damping) * new_entry["place_fields"] + damping * old_entry[
                "place_fields"
            ]
            new_entry["place_fields"] = blended
            new_entry["no_spike_part_log_likelihood"] = jnp.sum(blended, axis=0)

    def estimate_parameters(
        self,
        time: np.ndarray | None = None,
        log_likelihood_args: tuple | None = None,
        is_missing: np.ndarray | None = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: str | list[str] | set[str] | None = None,
        save_log_likelihood_to_results: bool | None = None,
        min_encoding_local_mass: float = 1.0,
        min_encoding_local_ess: float = 1.0,
        encoding_update_damping: float = 0.0,
    ) -> xr.Dataset:
        """
        Estimate the initial conditions and transition probabilities using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        time : np.ndarray, optional, shape (n_time,)
            Time points for decoding, by default None.
        log_likelihood_args : tuple, optional
            Arguments for the log likelihood function, by default None.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        estimate_initial_conditions : bool, optional
            Whether to estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Whether to estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Estimate the place fields based on the Local state, by default True.
        max_iter : int, optional
            Maximum number of EM iterations, by default 20.
        tolerance : float, optional
            Convergence tolerance for the EM algorithm, by default 1e-4.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihoods_, by default False.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        return_outputs : str, list of str, set of str, or None, optional
            Controls which optional outputs are returned.

            Options:
            - 'filter' : causal (filtered) posterior and state probabilities
            - 'predictive' : predictive state probabilities and posterior
            - 'predictive_posterior' : predictive posterior only
            - 'log_likelihood' : log likelihoods
            - 'all' : all of the above

            By default None (only smoother posterior and state probabilities).
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead.
            Whether to save the log likelihood to the results, by default None.
        min_encoding_local_mass : float, optional
            Minimum sum of local state weights required to update the encoding
            model. If the total mass is below this threshold, the encoding
            M-step is skipped for that iteration. By default 1.0.
        min_encoding_local_ess : float, optional
            Minimum effective sample size (ESS) of local state weights required
            to update the encoding model. ESS = sum(w)^2 / sum(w^2). If ESS
            is below this threshold, the encoding M-step is skipped. By default 1.0.
        encoding_update_damping : float, optional
            Damping factor in [0, 1) for encoding model updates. When > 0, the
            new place fields are blended with the old ones:
            new = (1 - damping) * refit + damping * old. Set higher (e.g. 0.5)
            when local ESS is expected to be low. By default 0.0 (no damping).

        Returns
        -------
        results : xr.Dataset
            Results of the decoding, including posterior probabilities and marginal log likelihoods.
        """
        marginal_log_likelihoods = []
        n_iter = 0
        converged = False

        # Validate required parameters
        if time is None:
            raise ValidationError(
                "Missing required parameter: time",
                expected="time array with shape (n_time,)",
                got="None",
                hint="Provide timestamps corresponding to your data",
                example="    results = detector.predict(spikes=spikes_test, time=time_test)",
            )

        # Tier 2: Validate time properties
        val.ensure_ndarray(time, "time")
        val.ensure_all_finite(time, "time")
        val.ensure_monotonic_increasing(time, "time", strict=False)

        if log_likelihood_args is None:
            log_likelihood_args = ()

        # Handle deprecated boolean flag
        if save_log_likelihood_to_results is not None:
            warnings.warn(
                "save_log_likelihood_to_results is deprecated. "
                "Use return_outputs='log_likelihood' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if save_log_likelihood_to_results:
                if return_outputs is not None:
                    raise ValueError(
                        "Cannot specify both return_outputs and deprecated "
                        "save_log_likelihood_to_results flag. "
                        "Use return_outputs only."
                    )
                return_outputs = "log_likelihood"

        # Normalize return_outputs to canonical set
        requested_outputs = _normalize_return_outputs(return_outputs)

        # Automatically enable caching if log_likelihood is requested
        if "log_likelihood" in requested_outputs and not cache_likelihood:
            cache_likelihood = True

        # Validate encoding update parameters
        if not 0.0 <= encoding_update_damping < 1.0:
            raise ValueError(
                f"encoding_update_damping must be in [0, 1), got {encoding_update_damping}"
            )
        if min_encoding_local_mass < 0:
            raise ValueError(
                f"min_encoding_local_mass must be >= 0, got {min_encoding_local_mass}"
            )
        if min_encoding_local_ess < 0:
            raise ValueError(
                f"min_encoding_local_ess must be >= 0, got {min_encoding_local_ess}"
            )

        while not converged and (n_iter < max_iter):
            # Expectation step
            logger.info("Expectation step...")
            (
                acausal_posterior,
                acausal_state_probabilities,
                marginal_log_likelihood,
                causal_state_probabilities,
                predictive_state_probabilities,
                log_likelihood,
                causal_posterior,
                predictive_posterior,
            ) = self._predict(
                time=time,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                cache_likelihood=cache_likelihood,
                log_likelihoods=getattr(self, "log_likelihood_", None),
                n_chunks=n_chunks,
            )
            # Maximization step
            logger.info("Maximization step...")

            if estimate_encoding_model:
                try:
                    local_state_index = self.state_names.index("Local")
                except ValueError:
                    local_state_index = None

                if local_state_index is not None:
                    logger.info("Estimating encoding model...")
                    local_state_weights = acausal_state_probabilities[
                        :, local_state_index
                    ]
                    # Align weights from decoding time grid to position_time
                    position_time = self._encoding_model_data.get("position_time")
                    if position_time is not None and len(local_state_weights) != len(
                        position_time
                    ):
                        local_state_weights = np.interp(
                            position_time, time, local_state_weights
                        )

                    # ESS / mass guard: skip encoding update when local
                    # state has effectively disappeared
                    mass = float(np.sum(local_state_weights))
                    sum_sq = float(np.sum(local_state_weights**2))
                    ess = (mass**2 / sum_sq) if sum_sq > 0 else 0.0

                    if mass < min_encoding_local_mass or ess < min_encoding_local_ess:
                        logger.info(
                            "Skipping encoding update: local mass=%.4g, "
                            "ESS=%.4g (thresholds: mass>=%.4g, ESS>=%.4g)",
                            mass,
                            ess,
                            min_encoding_local_mass,
                            min_encoding_local_ess,
                        )
                    else:
                        # Snapshot for damping (only when needed)
                        prev_encoding_model = (
                            _snapshot_encoding_model(
                                getattr(self, "encoding_model_", None)
                            )
                            if encoding_update_damping > 0
                            else None
                        )

                        # Re-fit the encoding model using the posterior weights
                        self.fit_encoding_model(
                            **self._encoding_model_data,
                            weights=local_state_weights,
                        )

                        # Apply damping: blend new place fields with old
                        if prev_encoding_model is not None:
                            self._apply_encoding_damping(
                                self.encoding_model_,
                                prev_encoding_model,
                                encoding_update_damping,
                            )

                        if cache_likelihood:
                            try:
                                del self.log_likelihood_
                            except AttributeError:
                                pass

            if estimate_discrete_transition:
                (
                    self.discrete_state_transitions_,
                    self.discrete_transition_coefficients_,
                ) = _estimate_discrete_transition(
                    causal_state_probabilities,
                    predictive_state_probabilities,
                    acausal_state_probabilities,
                    self.discrete_state_transitions_,
                    self.discrete_transition_coefficients_,
                    self.discrete_transition_design_matrix_,
                    self.discrete_transition_concentration,
                    self.discrete_transition_stickiness,
                    self.discrete_transition_regularization,
                    self.discrete_transition_prior_weight,
                )
                # Restore frozen rows: overwrite the M-step result with
                # the initial snapshot for rows the user wants pinned.
                # This is a constrained M-step; EM converges to the
                # constrained fixed point rather than the unconstrained
                # maximizer, which is the desired behavior here.
                if (
                    self._frozen_discrete_transition_rows_baseline_ is not None
                    and self.discrete_state_transitions_.ndim == 2
                ):
                    self.discrete_state_transitions_[
                        self._frozen_discrete_transition_rows_mask_
                    ] = self._frozen_discrete_transition_rows_baseline_

            if estimate_initial_conditions:
                self.initial_conditions_[self.is_track_interior_state_bins_] = (
                    acausal_posterior[0]
                )
                self.discrete_initial_conditions = acausal_state_probabilities[0]

                expanded_discrete_ic = acausal_state_probabilities[0][self.state_ind_]
                is_zero = np.isclose(expanded_discrete_ic, 0.0)
                safe_discrete_ic = np.where(is_zero, 1.0, expanded_discrete_ic)
                self.continuous_initial_conditions_ = np.where(
                    is_zero,
                    0.0,
                    self.initial_conditions_ / safe_discrete_ic,
                )

            # Stats
            logger.info("Computing stats..")
            n_iter += 1

            marginal_log_likelihoods.append(marginal_log_likelihood)
            if n_iter > 1:
                log_likelihood_change = (
                    marginal_log_likelihoods[-1] - marginal_log_likelihoods[-2]
                )
                converged, _ = check_converged(
                    marginal_log_likelihoods[-1],
                    marginal_log_likelihoods[-2],
                    tolerance,
                )

                logger.info(
                    f"iteration {n_iter}, "
                    f"likelihood: {marginal_log_likelihoods[-1]}, "
                    f"change: {log_likelihood_change}"
                )
            else:
                logger.info(
                    f"iteration {n_iter}, likelihood: {marginal_log_likelihoods[-1]}"
                )

        if store_log_likelihood:
            self.log_likelihood_ = log_likelihood

        if hasattr(self, "encoding_model_data_"):
            del self.encoding_model_data_

        return self._convert_results_to_xarray(
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihoods,
            log_likelihood=(
                log_likelihood if "log_likelihood" in requested_outputs else None
            ),
            causal_posterior=(
                causal_posterior if "filter" in requested_outputs else None
            ),
            causal_state_probabilities=(
                causal_state_probabilities if "filter" in requested_outputs else None
            ),
            predictive_state_probabilities=(
                predictive_state_probabilities
                if "predictive" in requested_outputs
                else None
            ),
            predictive_posterior=(
                predictive_posterior
                if "predictive_posterior" in requested_outputs
                else None
            ),
        )

    def most_likely_sequence(
        self,
        time: np.ndarray,
        log_likelihood_args: tuple | None = None,
        is_missing: np.ndarray | None = None,
        n_chunks: int = 1,
    ) -> np.ndarray:
        """Find the most likely sequence of states.

        Returns
        -------
        pd.DataFrame, shape (n_time, n_columns)
            DataFrame containing the most likely sequence of states
            and corresponding positions/metadata at each time step.

        """
        # Validate parameters
        if log_likelihood_args is None:
            log_likelihood_args = ()

        is_track_interior = self.is_track_interior_state_bins_
        cross_is_track_interior = np.ix_(is_track_interior, is_track_interior)
        state_ind = self.state_ind_[is_track_interior]
        if self.discrete_state_transitions_.ndim == 2:
            sequence_ind, _ = most_likely_sequence(
                time=time,
                initial_distribution=self.initial_conditions_[is_track_interior],
                transition_matrix=(
                    self.continuous_state_transitions_[cross_is_track_interior]
                    * self.discrete_state_transitions_[np.ix_(state_ind, state_ind)]
                ),
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                log_likelihoods=getattr(self, "log_likelihood_", None),
                n_chunks=n_chunks,
            )
        else:
            sequence_ind, _ = most_likely_sequence_covariate_dependent(
                time=time,
                state_ind=state_ind,
                initial_distribution=self.initial_conditions_[is_track_interior],
                discrete_transition_matrix=self.discrete_state_transitions_,
                continuous_transition_matrix=self.continuous_state_transitions_[
                    cross_is_track_interior
                ],
                log_likelihood_func=self.compute_log_likelihood,
                log_likelihood_args=log_likelihood_args,
                is_missing=is_missing,
                log_likelihoods=getattr(self, "log_likelihood_", None),
                n_chunks=n_chunks,
            )

        return self._convert_seq_to_df(sequence_ind, time)

    def calculate_time_bins(self, time_range: np.ndarray) -> np.ndarray:
        """
        Calculate time bins based on the provided time range.

        Parameters
        ----------
        time_range : np.ndarray, shape (2,)
            Array specifying the range of time.

        Returns
        -------
        time : np.ndarray, shape (n_time_bins,)
            Array of time bins.
        """
        n_time_bins = int(
            np.ceil((time_range[-1] - time_range[0]) * self.sampling_frequency)
        )
        return time_range[0] + np.arange(n_time_bins) / self.sampling_frequency

    def save_model(self, filename: str = "model.pkl") -> None:
        """
        Save the detector to a pickled file.

        Parameters
        ----------
        filename : str, optional
            Filename to save the model, by default "model.pkl".
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(filename: str = "model.pkl") -> "_DetectorBase":
        """
        Load the detector from a file.

        .. warning::

            Only load models from trusted sources. Pickle files can execute
            arbitrary code during deserialization.

        Parameters
        ----------
        filename : str, optional
            Filename to load the model from, by default "model.pkl"

        Returns
        -------
        _DetectorBase
            Loaded detector instance
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_results(results: xr.Dataset, filename: str = "results.nc") -> None:
        """
        Save the results to a netcdf file.

        `state_bins`is a multiindex, which is not supported by netcdf so
        it is converted before saving.

        Parameters
        ----------
        results : xr.Dataset
            Decoding results
        filename : str, optional
            Name to save, by default "results.nc"
        """
        results.reset_index("state_bins").to_netcdf(filename)

    @staticmethod
    def load_results(filename: str = "results.nc") -> xr.Dataset:
        """
        Loads the results from a netcdf file and converts the
        index back to a multiindex.

        Parameters
        ----------
        filename : str, optional
            File containing results, by default "results.nc"

        Returns
        -------
        xr.Dataset
            Decoding results
        """
        results = xr.open_dataset(filename)
        coord_names = list(results["state_bins"].coords)
        return results.set_index(state_bins=coord_names)

    def copy(self) -> "_DetectorBase":
        """
        Makes a copy of the detector.

        Returns
        -------
        _DetectorBase
            Deep copy of the detector instance.
        """
        return copy.deepcopy(self)

    def _get_environment_by_name(self, environment_name: str) -> Environment:
        """
        Get environment by name with O(1) lookup using cached dictionary.

        Parameters
        ----------
        environment_name : str
            Name of the environment to retrieve.

        Returns
        -------
        Environment
            The environment object.

        Raises
        ------
        ValueError
            If environment not found.
        """
        # Create cached lookup dict if not exists
        if not hasattr(self, "_env_lookup_cache_"):
            self._env_lookup_cache_ = {
                env.environment_name: env for env in self.environments
            }

        environment = self._env_lookup_cache_.get(environment_name)
        if environment is None:
            raise ValueError(
                f"Environment '{environment_name}' not found in environments list"
            )
        return environment

    @staticmethod
    def _create_masked_posterior(
        data: np.ndarray, is_track_interior: np.ndarray, n_total_bins: int
    ) -> np.ndarray:
        """
        Create full posterior array with NaN for non-interior bins.

        Parameters
        ----------
        data : np.ndarray, shape (n_time, n_interior_bins)
            Posterior data for interior bins only.
        is_track_interior : np.ndarray, shape (n_total_bins,)
            Boolean mask indicating which bins are track interior.
        n_total_bins : int
            Total number of bins including non-interior.

        Returns
        -------
        full_array : np.ndarray, shape (n_time, n_total_bins)
            Full array with NaN for non-interior bins.
        """
        n_time = data.shape[0]
        full_array = np.full((n_time, n_total_bins), np.nan, dtype=np.float32)
        full_array[:, is_track_interior] = data
        return full_array

    def _convert_results_to_xarray(
        self,
        time: np.ndarray,
        acausal_posterior: np.ndarray,
        acausal_state_probabilities: np.ndarray,
        marginal_log_likelihoods: list[float],
        log_likelihood: np.ndarray | None = None,
        causal_posterior: np.ndarray | None = None,
        causal_state_probabilities: np.ndarray | None = None,
        predictive_state_probabilities: np.ndarray | None = None,
        predictive_posterior: np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Convert the results to an xarray Dataset.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
            Acausal posterior probabilities.
        acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
            Acausal state probabilities.
        marginal_log_likelihoods : list of float
            Marginal log likelihoods for each iteration.
        log_likelihood : np.ndarray, optional, shape (n_time, n_state_bins)
            Log likelihoods, by default None.
        causal_posterior : np.ndarray, optional, shape (n_time, n_state_bins)
            Causal (filtered) posterior probabilities, by default None.
        causal_state_probabilities : np.ndarray, optional, shape (n_time, n_states)
            Causal state probabilities, by default None.
        predictive_state_probabilities : np.ndarray, optional, shape (n_time, n_states)
            One-step-ahead predicted state probabilities, by default None.
        predictive_posterior : np.ndarray, optional, shape (n_time, n_state_bins)
            One-step-ahead predicted posterior probabilities over state bins, by default None.

        Returns
        -------
        xr.Dataset
            Decoding results in an xarray Dataset.
        """
        is_track_interior = self.is_track_interior_state_bins_

        # Extract environment and encoding group names in single pass
        environment_names = [obs.environment_name for obs in self.observation_models]
        encoding_group_names = [obs.encoding_group for obs in self.observation_models]

        # Create environment lookup dict for O(1) access
        env_dict = {env.environment_name: env for env in self.environments}

        # Get position dimensionality
        if not self.environments:
            raise ConfigurationError(
                "No environments found in model",
                hint="Environments are set up during fit(). Either fit the model first, or check that position data was provided",
            )
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]  # type: ignore[union-attr]

        # Build position array
        position = []
        for obs in self.observation_models:
            if obs.is_no_spike or (obs.is_local and self.local_position_std is None):
                position.append(np.full((1, n_position_dims), np.nan))
            else:
                environment = env_dict.get(obs.environment_name)
                if environment is None:
                    raise ValueError(
                        f"Environment '{obs.environment_name}' not found in environments list"
                    )
                if environment.place_bin_centers_ is None:
                    raise ValueError(
                        f"place_bin_centers_ is None for environment {obs.environment_name}"
                    )
                position.append(environment.place_bin_centers_)
        position = np.concatenate(position, axis=0)

        states = np.asarray(self.state_names)

        # Generate position dimension names
        if n_position_dims == 1:
            position_names = ["position"]
        else:
            # Support arbitrary number of dimensions
            dim_labels = ["x", "y", "z", "w", "v", "u"]  # Up to 6 dimensions
            if n_position_dims <= len(dim_labels):
                position_names = [
                    f"{dim_labels[i]}_position" for i in range(n_position_dims)
                ]
            else:
                # Fall back to numbered dimensions if > 6
                position_names = [f"dim{i}_position" for i in range(n_position_dims)]
        # Create MultiIndex for state_bins coordinate
        state_bins_mindex = pd.MultiIndex.from_arrays(
            ((states[self.state_ind_], *position.T)),
            names=("state", *position_names),
        )

        coords = {
            "time": time,
            "state_ind": self.state_ind_,
            "states": states,
            "environments": ("states", environment_names),
            "encoding_groups": ("states", encoding_group_names),
        }

        # Handle MultiIndex: use new API if available (xarray >= 2022.06.0),
        # otherwise use old direct assignment (for backward compatibility)
        if hasattr(xr.Coordinates, "from_pandas_multiindex"):
            # New method (preferred, avoids FutureWarning)
            mindex_coords = xr.Coordinates.from_pandas_multiindex(
                state_bins_mindex, "state_bins"
            )
        else:
            # Old method (backward compatible)
            coords["state_bins"] = state_bins_mindex
            mindex_coords = None

        attrs = {"marginal_log_likelihoods": np.asarray(marginal_log_likelihoods)}

        # Build data_vars dict with masked posteriors
        n_total_bins = len(is_track_interior)
        data_vars = {
            "acausal_posterior": (
                ("time", "state_bins"),
                self._create_masked_posterior(
                    acausal_posterior, is_track_interior, n_total_bins
                ),
            ),
            "acausal_state_probabilities": (
                ("time", "states"),
                acausal_state_probabilities,
            ),
        }

        if log_likelihood is not None:
            data_vars["log_likelihood"] = (
                ("time", "state_bins"),
                self._create_masked_posterior(
                    log_likelihood, is_track_interior, n_total_bins
                ),
            )

        if causal_posterior is not None:
            data_vars["causal_posterior"] = (
                ("time", "state_bins"),
                self._create_masked_posterior(
                    causal_posterior, is_track_interior, n_total_bins
                ),
            )

        if causal_state_probabilities is not None:
            data_vars["causal_state_probabilities"] = (
                ("time", "states"),
                causal_state_probabilities,
            )

        if predictive_state_probabilities is not None:
            data_vars["predictive_state_probabilities"] = (
                ("time", "states"),
                predictive_state_probabilities,
            )

        if predictive_posterior is not None:
            data_vars["predictive_posterior"] = (
                ("time", "state_bins"),
                self._create_masked_posterior(
                    predictive_posterior, is_track_interior, n_total_bins
                ),
            )

        # Create Dataset with MultiIndex coordinates
        results = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Assign MultiIndex coordinates if using new API
        if mindex_coords is not None:
            results = results.assign_coords(mindex_coords)

        return results.squeeze()

    def _convert_seq_to_df(
        self, sequence_ind: np.ndarray, time: np.ndarray
    ) -> pd.DataFrame:
        """Converts the sequence indices to a DataFrame.

        Parameters
        ----------
        sequence_ind : np.ndarray, shape (n_time,)
            Most likely sequence indices.
        time : np.ndarray, shape (n_time,)

        Returns
        -------
        results : pd.DataFrame, shape (n_time, n_cols)
        """
        position = []
        n_position_dims = self.environments[0].place_bin_centers_.shape[1]  # type: ignore[union-attr]
        environment_names = []
        encoding_group_names = []
        for obs in self.observation_models:
            if obs.is_no_spike or (obs.is_local and self.local_position_std is None):
                position.append(np.full((1, n_position_dims), np.nan))
                environment_names.append([None])
                encoding_group_names.append([None])
            else:
                environment = self._get_environment_by_name(obs.environment_name)
                if environment.place_bin_centers_ is None:
                    raise ValueError(
                        f"place_bin_centers_ is None for environment {obs.environment_name}"
                    )
                position.append(environment.place_bin_centers_)
                environment_names.append(
                    [obs.environment_name] * environment.place_bin_centers_.shape[0]
                )
                encoding_group_names.append(
                    [obs.encoding_group] * environment.place_bin_centers_.shape[0]
                )

        position = np.concatenate(position, axis=0)
        environment_names = np.concatenate(environment_names, axis=0)
        encoding_group_names = np.concatenate(encoding_group_names, axis=0)

        states = np.asarray(self.state_names)
        if n_position_dims == 1:
            position_names = ["position"]
        else:
            position_names = [
                f"{name}_position"
                for name, _ in zip(["x", "y", "z", "w"], position.T, strict=False)
            ]
        state_bins = pd.DataFrame(
            {
                "state": states[self.state_ind_],
                **dict(zip(position_names, position.T, strict=False)),
                "environment": environment_names,
                "encoding_group_names": encoding_group_names,
            }
        ).iloc[self.is_track_interior_state_bins_]

        return state_bins.iloc[sequence_ind].set_index(pd.Index(time, name="time"))


class ClusterlessDetector(_DetectorBase):
    """
    Detector class for clusterless spikes.
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        clusterless_algorithm: str = "clusterless_kde",
        clusterless_algorithm_params: dict = _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: (
            np.ndarray | list[int] | tuple[int, ...] | None
        ) = None,
        local_position_std: float | None = None,
    ) -> None:
        """
        Initialize the ClusterlessDetector class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        clusterless_algorithm : str, optional
            Algorithm for clusterless spikes, by default "clusterless_kde".
        clusterless_algorithm_params : dict, optional
            Parameters for the clusterless algorithm, by default _DEFAULT_CLUSTERLESS_ALGORITHM_PARAMS.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Sampling frequency, by default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        discrete_transition_prior_weight : float or np.ndarray, optional
            Data-adaptive prior weight, by default 0.0. See
            ``_DetectorBase`` for details.
        frozen_discrete_transition_rows : array-like or None, optional
            Rows of the discrete transition matrix to freeze during EM
            re-estimation, by default None. See ``_DetectorBase`` for
            details.
        local_position_std : float or None, optional
            Standard deviation of the position uncertainty kernel for the
            local state. See ``_DetectorBase`` for details.
        """
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            infer_track_interior,
            state_names,
            sampling_frequency,
            no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        self.clusterless_algorithm = clusterless_algorithm
        self.clusterless_algorithm_params = clusterless_algorithm_params

    def _get_group_spike_data(
        self,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_group: np.ndarray,
        position_time: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Get group spike data based on is_group mask.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_group : np.ndarray, shape (n_time_position,)
            Boolean mask indicating group membership.
        position_time : np.ndarray
            Time points for position data.

        Returns
        -------
        group_spike_times : list of np.ndarray
        group_spike_waveform_features : list of np.ndarray
        """
        # get consecutive runs in each group
        group_labels, n_groups = scipy.ndimage.label(is_group)

        time_delta = position_time[1] - position_time[0]

        group_spike_times = []
        group_spike_waveform_features = []
        for electrode_spike_times, electrode_spike_waveform_features in zip(
            spike_times, spike_waveform_features, strict=False
        ):
            group_electrode_spike_times = []
            group_electrode_waveform_features = []
            # get spike times for each run
            for group in range(1, n_groups + 1):
                start_time, stop_time = position_time[group_labels == group][[0, -1]]
                # Add half a time bin to the start and stop times
                # to ensure that the spike times are within the group
                start_time -= time_delta
                stop_time += time_delta
                is_valid_spike_time = np.logical_and(
                    electrode_spike_times >= start_time,
                    electrode_spike_times <= stop_time,
                )
                group_electrode_spike_times.append(
                    electrode_spike_times[is_valid_spike_time]
                )
                group_electrode_waveform_features.append(
                    electrode_spike_waveform_features[is_valid_spike_time]
                )
            if group_electrode_spike_times:
                group_spike_times.append(np.concatenate(group_electrode_spike_times))
                group_spike_waveform_features.append(
                    np.concatenate(group_electrode_waveform_features, axis=0)
                )
            else:
                # No training coverage for this obs group; return empty arrays
                # with the right dtype/feature-dim so downstream encoding can
                # fit an (empty) model without crashing on np.concatenate([]).
                group_spike_times.append(
                    np.asarray(
                        electrode_spike_times[:0], dtype=electrode_spike_times.dtype
                    )
                )
                group_spike_waveform_features.append(
                    electrode_spike_waveform_features[:0]
                )

        return group_spike_times, group_spike_waveform_features

    def fit_encoding_model(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        """
        Fit the encoding model to the data.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array or weights indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,) optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, optional
            Environment labels, by default None.
        weights : np.ndarray, optional, shape (n_time_position,)
            Weights for training data, by default None.

        Attributes
        ----------
        encoding_model_ : dict
            Dictionary holding the fitted encoding models for each unique observation model
            configuration (environment, encoding group).
            The values depend on the chosen `clusterless_algorithm`.
        """
        logger.info("Fitting clusterless spikes...")
        n_time = position.shape[0]
        position = position if position.ndim > 1 else position[:, np.newaxis]

        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time
            )

        is_training = np.asarray(is_training).squeeze()

        is_nan = np.any(np.isnan(position), axis=1)
        position = position[~is_nan]
        position_time = position_time[~is_nan]
        is_training = is_training[~is_nan]
        encoding_group_labels = encoding_group_labels[~is_nan]
        environment_labels = environment_labels[~is_nan]
        if weights is not None:
            weights = weights[~is_nan]

        kwargs = self.clusterless_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = {}

        for obs in np.unique(self.observation_models):
            environment = self._get_environment_by_name(obs.environment_name)

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = environment_labels == obs.environment_name
            likelihood_name = (obs.environment_name, obs.encoding_group)

            encoding_algorithm, _ = _CLUSTERLESS_ALGORITHMS[self.clusterless_algorithm]
            is_group = is_training & is_encoding & is_environment
            (
                group_spike_times,
                group_spike_waveform_features,
            ) = self._get_group_spike_data(
                spike_times, spike_waveform_features, is_group, position_time
            )
            self.encoding_model_[likelihood_name] = encoding_algorithm(
                position_time=position_time[is_group],
                position=position[is_group],
                spike_times=group_spike_times,
                spike_waveform_features=group_spike_waveform_features,
                environment=environment,
                sampling_frequency=self.sampling_frequency,
                weights=weights[is_group] if weights is not None else None,
                **kwargs,
            )

    def fit(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict | None = None,
    ) -> "ClusterlessDetector":
        """
        Fit the detector to the data.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Environment labels, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Returns
        -------
        ClusterlessDetector
            Fitted detector instance.
        """
        self._fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        self.fit_encoding_model(
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        return self

    def compute_log_likelihood(
        self,
        time: np.ndarray,
        position_time: np.ndarray,
        position: np.ndarray | None,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        is_missing: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Compute the log likelihood for the given data.

        Notes
        -----
        When ``local_position_std`` is set, the per-bin value returned
        for local-state entries is not a pure observation likelihood.
        It combines the spatial spike likelihood with the position
        uncertainty kernel::

            log_likelihood[local, b, t] =
                log P(spikes_t | bin b)                   # spatial likelihood
              + log_kernel(b | animal_pos_t)              # Gaussian anchor
              + log(n_interior_bins)                      # mass-balance

        The kernel term is a time-varying state-specific spatial prior.
        Mathematically, injecting it into the likelihood is equivalent
        to injecting it into the transition matrix — both multiply into
        the HMM forward step — but avoids breaking the static-transition
        assumption used by ``jax.lax.scan``. The ``log(n_interior_bins)``
        constant compensates for the ``1/n_bins`` uniform continuous IC
        so the effective per-bin prior is ``exp(log_kernel)`` itself.

        The kernel is part of the likelihood model, not an ad-hoc
        post-processing step, so the posterior returned by ``predict()``
        is a valid probability distribution under the modified
        generative model. Users comparing ``local_position_std=None`` to
        a finite value will see different posteriors (the models
        differ); users reading the raw ``log_likelihood`` (via
        ``return_outputs='log_likelihood'``) should be aware that
        local-state entries are *not* pure ``log P(spikes | state, bin)``
        — they include the kernel and mass-balance terms.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.

        Returns
        -------
        log_likelihood : jnp.ndarray, shape (n_time, n_state_bins)
        """
        logger.info("Computing log likelihood...")
        non_local_penalty = getattr(self, "non_local_position_penalty", 0.0)
        needs_position = (
            np.any([obs.is_local for obs in self.observation_models])
            or non_local_penalty > 0
            or self.local_position_std is not None
        )
        if position is None and needs_position:
            reason = []
            if np.any([obs.is_local for obs in self.observation_models]):
                reason.append("local observation models")
            if non_local_penalty > 0:
                reason.append("non_local_position_penalty > 0")
            if self.local_position_std is not None:
                reason.append("local_position_std is set")
            raise ValidationError(
                f"Missing required parameter: position (needed for {', '.join(reason)})",
                expected="position array with shape (n_time, n_dims)",
                got="None",
                hint="Provide position data or set non_local_position_penalty=0.0 to disable the penalty",
                example="    results = detector.predict(spikes=spikes_test, time=time_test, position=position_test)",
            )
        if position is not None:
            self._validate_position_dimensionality(position, context="predict")

        n_time = len(time)
        if is_missing is None:
            is_missing = jnp.zeros((n_time,), dtype=bool)

        _, likelihood_func = _CLUSTERLESS_ALGORITHMS[self.clusterless_algorithm]

        # Pre-compute state bins mask for all states (avoid recomputation in loop)
        interior_state_ind = self.state_ind_[self.is_track_interior_state_bins_]
        state_bin_masks = {
            state_id: interior_state_ind == state_id
            for state_id in range(len(self.observation_models))
        }

        # Use dict for O(1) lookup instead of list
        computed_likelihoods = {}

        # Compute all likelihoods first (stays in JAX/GPU if available)
        likelihood_results = {}
        for state_id, obs in enumerate(self.observation_models):
            # Multi-bin local computes full spatial likelihood (is_local=False)
            # then adds position kernel afterward
            effective_is_local = obs.is_local and self.local_position_std is None

            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
                effective_is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                likelihood_results[state_id] = predict_no_spike_log_likelihood(
                    time, spike_times, self.no_spike_rate
                )
            elif likelihood_name not in computed_likelihoods:
                likelihood_results[state_id] = likelihood_func(
                    time,
                    position_time,
                    position,
                    spike_times,
                    spike_waveform_features,
                    **self.encoding_model_[likelihood_name[:2]],
                    is_local=effective_is_local,
                )
                computed_likelihoods[likelihood_name] = state_id
            else:
                # Will reuse previously computed likelihood
                likelihood_results[state_id] = computed_likelihoods[likelihood_name]

        # Assemble final array (single pass, stays in JAX)
        log_likelihood = jnp.zeros(
            (n_time, self.is_track_interior_state_bins_.sum()), dtype=jnp.float32
        )

        for state_id in range(len(self.observation_models)):
            is_state_bin = state_bin_masks[state_id]
            result = likelihood_results[state_id]

            if isinstance(result, int):
                # Reuse from previously computed state
                source_mask = state_bin_masks[result]
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    log_likelihood[:, source_mask]
                )
            else:
                # Set new likelihood values
                log_likelihood = log_likelihood.at[:, is_state_bin].set(result)

        # Apply non-local position penalty if configured
        if non_local_penalty > 0:
            # Cache penalties per environment to avoid recomputation
            env_penalties = {}
            for state_id, obs in enumerate(self.observation_models):
                if not obs.is_local and not obs.is_no_spike:
                    env_name = obs.environment_name
                    if env_name not in env_penalties:
                        env = self._get_environment_by_name(env_name)
                        env_penalties[env_name] = (
                            self._compute_non_local_position_penalty(
                                time, position_time, position, env
                            )
                        )
                    is_state_bin = state_bin_masks[state_id]
                    log_likelihood = log_likelihood.at[:, is_state_bin].add(
                        env_penalties[env_name]
                    )

        # Apply local position kernel if configured
        if self.local_position_std is not None:
            env_kernels = {}
            for state_id, obs in enumerate(self.observation_models):
                if obs.is_local:
                    env_name = obs.environment_name
                    if env_name not in env_kernels:
                        env = self._get_environment_by_name(env_name)
                        env_kernels[env_name] = self._compute_local_position_kernel(
                            time, position_time, position, env
                        )
                    is_state_bin = state_bin_masks[state_id]
                    log_likelihood = log_likelihood.at[:, is_state_bin].add(
                        env_kernels[env_name]
                    )

        # Apply missing data mask
        return jnp.where(is_missing[:, jnp.newaxis], 0.0, log_likelihood)

    def predict(
        self,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        time: np.ndarray,
        position: np.ndarray | None = None,
        position_time: np.ndarray | None = None,
        is_missing: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict | None = None,
        cache_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: str | list[str] | set[str] | None = None,
        save_log_likelihood_to_results: bool | None = None,
        save_causal_posterior_to_results: bool | None = None,
    ) -> xr.Dataset:
        """
        Predict the posterior probabilities for the given data.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : np.ndarray, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        discrete_transition_covariate_data : dict-like, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        return_outputs : str, list of str, set of str, or None, optional
            Controls which optional outputs are returned.

            Options:
            - None: smoother only (default, minimal memory)
            - 'filter': filtered (causal) posterior and state probabilities
            - 'predictive': both aggregated and full predictive distributions
            - 'predictive_posterior': only full predictive posterior (state bins)
            - 'log_likelihood': per-timepoint log likelihoods
            - 'all': all outputs above
            - List/set: e.g., ['filter', 'log_likelihood'] for multiple outputs

            The smoother (acausal_posterior, acausal_state_probabilities) and
            marginal_log_likelihood are ALWAYS included.

            When to use each output:
            - 'filter': Online/causal decoding, debugging forward pass
            - 'predictive': Model evaluation, predictive checks (includes both formats)
            - 'predictive_posterior': When you only need full distribution, not aggregated
            - 'log_likelihood': Diagnostics, per-timepoint metrics, model comparison

            Memory warning: 'log_likelihood', 'filter', 'predictive', and
            'predictive_posterior' can be very large (~400 GB for 1M timepoints × 100k
            spatial bins). Use None for minimal memory (smoother only).
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead.
            Whether to save the log likelihood to the results, by default None.
        save_causal_posterior_to_results : bool, optional
            DEPRECATED. Use return_outputs='filter' instead.
            Whether to save the causal (filtered) posterior to the results, by default None.

        Returns
        -------
        xr.Dataset
            Dataset containing decoded posterior distributions.

            Always included:
            - acausal_posterior : (n_time, n_state_bins)
                Smoothed posterior over state bins
            - acausal_state_probabilities : (n_time, n_states)
                Smoothed discrete state probabilities
            - marginal_log_likelihoods : float (in attrs)
                Total log evidence for the model

            Conditionally included based on return_outputs:
            - causal_posterior : (n_time, n_state_bins) - if 'filter'
                Filtered (forward-only) posterior over state bins
            - causal_state_probabilities : (n_time, n_states) - if 'filter'
                Filtered discrete state probabilities
            - predictive_state_probabilities : (n_time, n_states) - if 'predictive'
                One-step-ahead predictive distributions over discrete states
            - predictive_posterior : (n_time, n_state_bins) - if 'predictive_posterior'
                One-step-ahead predictive distributions over state bins
                (Warning: can be very large, ~same size as causal_posterior)
            - log_likelihood : (n_time, n_state_bins) - if 'log_likelihood'
                Per-timepoint observation log likelihoods

        Examples
        --------
        Get only smoother (default, minimal memory):

        >>> results = model.predict(spike_times, time)
        >>> results.acausal_posterior.shape
        (10000, 50000)

        Include filtered posterior for online decoding:

        >>> results = model.predict(spike_times, time, return_outputs='filter')
        >>> results.causal_posterior.shape
        (10000, 50000)

        Get multiple outputs for analysis:

        >>> results = model.predict(
        ...     spike_times, time,
        ...     return_outputs=['filter', 'predictive']
        ... )

        Get everything for debugging:

        >>> results = model.predict(spike_times, time, return_outputs='all')
        """
        if position is not None and position_time is None:
            raise ValidationError(
                "position_time is required when position is provided",
                expected="position_time array with shape (n_time_position,)",
                got="None",
                hint="Provide position_time corresponding to the position samples",
                example="    results = detector.predict(\n"
                "        spike_times=spike_times, spike_waveform_features=features,\n"
                "        time=time, position=position, position_time=position_time\n"
                "    )",
            )

        if position is not None:
            position = position[:, np.newaxis] if position.ndim == 1 else position
            nan_position = np.any(np.isnan(position), axis=1)
            if np.any(nan_position) and is_missing is None:
                is_missing = nan_position
            elif np.any(nan_position) and is_missing is not None:
                is_missing = np.logical_or(is_missing, nan_position)

        if is_missing is not None and len(is_missing) != len(time):
            raise ValueError(
                f"Length of is_missing must match length of time. Time is n_samples: {len(time)}"
            )

        # Handle deprecated boolean flags
        if (
            save_log_likelihood_to_results is not None
            or save_causal_posterior_to_results is not None
        ):
            warnings.warn(
                "save_log_likelihood_to_results and save_causal_posterior_to_results "
                "are deprecated. Use return_outputs parameter instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Convert old flags to new format
            outputs_from_flags = set()
            if save_log_likelihood_to_results:
                outputs_from_flags.add("log_likelihood")
            if save_causal_posterior_to_results:
                outputs_from_flags.add("filter")

            if return_outputs is not None:
                raise ValueError(
                    "Cannot specify both return_outputs and deprecated "
                    "save_*_to_results flags. Use return_outputs only."
                )
            return_outputs = outputs_from_flags if outputs_from_flags else None

        # Normalize return_outputs to canonical set
        requested_outputs = _normalize_return_outputs(return_outputs)

        # Automatically enable caching if log_likelihood is requested
        if "log_likelihood" in requested_outputs and not cache_likelihood:
            cache_likelihood = True

        predicted_transitions = None
        if discrete_transition_covariate_data is not None:
            if self.discrete_transition_coefficients_ is None:
                raise ValueError(
                    "discrete_transition_coefficients_ is None but covariate data provided"
                )
            predicted_transitions = predict_discrete_state_transitions(
                self.discrete_transition_design_matrix_,
                self.discrete_transition_coefficients_,
                discrete_transition_covariate_data,
            )
            _validate_covariate_time_length(predicted_transitions, time)
        (
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            causal_state_probabilities,
            predictive_state_probabilities,
            log_likelihood,
            causal_posterior,
            predictive_posterior,
        ) = self._predict(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
                spike_waveform_features,
            ),
            is_missing=is_missing,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
            discrete_state_transitions=predicted_transitions,
        )

        return self._convert_results_to_xarray(
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            log_likelihood=(
                log_likelihood if "log_likelihood" in requested_outputs else None
            ),
            causal_posterior=(
                causal_posterior if "filter" in requested_outputs else None
            ),
            causal_state_probabilities=(
                causal_state_probabilities if "filter" in requested_outputs else None
            ),
            predictive_state_probabilities=(
                predictive_state_probabilities
                if "predictive" in requested_outputs
                else None
            ),
            predictive_posterior=(
                predictive_posterior
                if "predictive_posterior" in requested_outputs
                else None
            ),
        )

    def estimate_parameters(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        time: np.ndarray,
        is_missing: np.ndarray | None = None,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict | None = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: str | list[str] | set[str] | None = None,
        save_log_likelihood_to_results: bool | None = None,
        min_encoding_local_mass: float = 1.0,
        min_encoding_local_ess: float = 1.0,
        encoding_update_damping: float = 0.0,
    ) -> xr.Dataset:
        """
        Estimate the initial conditions and transition probabilities using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Environment labels, by default None.
        discrete_transition_covariate_data : dict-like, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        estimate_initial_conditions : bool, optional
            Whether to estimate the initial conditions, by default True.
        estimate_discrete_transition : bool, optional
            Whether to estimate the discrete transition matrix, by default True.
        estimate_encoding_model : bool, optional
            Estimate the place fields based on the Local state, by default True
        max_iter : int, optional
            Maximum number of EM iterations, by default 20.
        tolerance : float, optional
            Convergence tolerance for the EM algorithm, by default 1e-4.
        cache_likelihood : bool, optional
            If True, log likelihoods are cached instead of recomputed for each chunk, by default True
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihoods_, by default False.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        return_outputs : str, list of str, set of str, or None, optional
            Controls which optional outputs are returned. See predict() for full
            documentation of options. By default None.
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead. By default None.
        min_encoding_local_mass : float, optional
            Minimum sum of local state weights to update encoding model. By default 1.0.
        min_encoding_local_ess : float, optional
            Minimum effective sample size of local weights to update encoding. By default 1.0.
        encoding_update_damping : float, optional
            Damping factor in [0, 1) for encoding updates. By default 0.0.

        Returns
        -------
        results : xr.Dataset
            Results of the decoding.
        """
        self._encoding_model_data = {
            "position_time": position_time,
            "position": position,
            "spike_times": spike_times,
            "spike_waveform_features": spike_waveform_features,
            "is_training": is_training,
            "encoding_group_labels": encoding_group_labels,
            "environment_labels": environment_labels,
        }
        # Mirror predict(): treat NaN positions as missing observations during
        # the E-step so EM and predict use the same local-likelihood handling.
        if position is not None:
            position_2d = position[:, np.newaxis] if position.ndim == 1 else position
            nan_position = np.any(np.isnan(position_2d), axis=1)
            if np.any(nan_position):
                if is_missing is None:
                    is_missing = nan_position
                else:
                    is_missing = np.logical_or(is_missing, nan_position)
        self.fit(
            position_time,
            position,
            spike_times,
            spike_waveform_features,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
        )

        return super().estimate_parameters(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
                spike_waveform_features,
            ),
            is_missing=is_missing,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            n_chunks=n_chunks,
            return_outputs=return_outputs,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
            encoding_update_damping=encoding_update_damping,
        )

    def most_likely_sequence(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        spike_waveform_features: list[np.ndarray],
        time: np.ndarray,
        is_missing: np.ndarray | None = None,
        n_chunks: int = 1,
    ) -> pd.DataFrame:
        """Find the most likely sequence of states.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        spike_times : list of np.ndarray
            Spike times for each neuron.
        spike_waveform_features : list of np.ndarray
            Spike waveform features for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1

        Returns
        -------
        most_likely_sequence : pd.DataFrame, shape (n_time, n_cols)
        """
        return super().most_likely_sequence(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
                spike_waveform_features,
            ),
            is_missing=is_missing,
            n_chunks=n_chunks,
        )


class SortedSpikesDetector(_DetectorBase):
    """
    Detector class for sorted spikes.
    """

    def __init__(
        self,
        discrete_initial_conditions: np.ndarray,
        continuous_initial_conditions_types: ContinuousInitialConditions,
        discrete_transition_type: DiscreteTransitions,
        discrete_transition_concentration: float,
        discrete_transition_stickiness: Stickiness,
        discrete_transition_regularization: float,
        continuous_transition_types: ContinuousTransitions,
        observation_models: Observations,
        environments: Environments,
        sorted_spikes_algorithm: str = "sorted_spikes_kde",
        sorted_spikes_algorithm_params: dict = _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS,
        infer_track_interior: bool = True,
        state_names: StateNames = None,
        sampling_frequency: float = 500.0,
        no_spike_rate: float = 1e-10,
        discrete_transition_prior_weight: float | np.ndarray = 0.0,
        frozen_discrete_transition_rows: (
            np.ndarray | list[int] | tuple[int, ...] | None
        ) = None,
        local_position_std: float | None = None,
    ) -> None:
        """
        Initialize the SortedSpikesDetector class.

        Parameters
        ----------
        discrete_initial_conditions : np.ndarray, shape (n_states,)
            Initial conditions for discrete states.
        continuous_initial_conditions_types : ContinuousInitialConditions
            Types of continuous initial conditions.
        discrete_transition_type : DiscreteTransitions
            Type of discrete state transition.
        discrete_transition_concentration : float
            Concentration parameter for discrete state transitions.
        discrete_transition_stickiness : Stickiness
            Stickiness parameter for discrete state transitions.
        discrete_transition_regularization : float
            Regularization parameter for discrete state transitions.
        continuous_transition_types : ContinuousTransitions
            Types of continuous state transitions.
        observation_models : Observations
            Observation models for the detector.
        environments : Environments
            Environments in which the detector operates.
        sorted_spikes_algorithm : str, optional
            Algorithm for sorted spikes, by default "sorted_spikes_kde".
        sorted_spikes_algorithm_params : dict, optional
            Parameters for the sorted spikes algorithm, by default _DEFAULT_SORTED_SPIKES_ALGORITHM_PARAMS.
        infer_track_interior : bool, optional
            Whether to infer track interior, by default True.
        state_names : StateNames, optional
            Names of the states, by default None.
        sampling_frequency : float, optional
            Sampling frequency, by default 500.0.
        no_spike_rate : float, optional
            No spike rate, by default 1e-10.
        discrete_transition_prior_weight : float or np.ndarray, optional
            Data-adaptive prior weight, by default 0.0. See
            ``_DetectorBase`` for details.
        frozen_discrete_transition_rows : array-like or None, optional
            Rows of the discrete transition matrix to freeze during EM
            re-estimation, by default None. See ``_DetectorBase`` for
            details.
        local_position_std : float or None, optional
            Standard deviation of the position uncertainty kernel for the
            local state. See ``_DetectorBase`` for details.
        """
        super().__init__(
            discrete_initial_conditions,
            continuous_initial_conditions_types,
            discrete_transition_type,
            discrete_transition_concentration,
            discrete_transition_stickiness,
            discrete_transition_regularization,
            continuous_transition_types,
            observation_models,
            environments,
            infer_track_interior,
            state_names,
            sampling_frequency,
            no_spike_rate,
            discrete_transition_prior_weight=discrete_transition_prior_weight,
            frozen_discrete_transition_rows=frozen_discrete_transition_rows,
            local_position_std=local_position_std,
        )
        self.sorted_spikes_algorithm = sorted_spikes_algorithm
        self.sorted_spikes_algorithm_params = sorted_spikes_algorithm_params

    @staticmethod
    def _get_group_spikes(
        spike_times: list[np.ndarray], is_group: np.ndarray, position_time: np.ndarray
    ) -> list[np.ndarray]:
        """
        Get group spike times based on is_group mask.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_group : np.ndarray
            Boolean mask indicating group membership.
        position_time : np.ndarray
            Time points for position data.

        Returns
        -------
        list of np.ndarray
            Grouped spike times.
        """
        # get consecutive runs in each group
        group_labels, n_groups = scipy.ndimage.label(is_group)

        time_delta = position_time[1] - position_time[0]

        group_spike_times = []
        for neuron_spike_times in spike_times:
            group_neuron_spike_times = []
            # get spike times for each run
            for group in range(1, n_groups + 1):
                start_time, stop_time = position_time[group_labels == group][[0, -1]]
                # Add half a time bin to the start and stop times
                # to ensure that the spike times are within the group
                start_time -= time_delta
                stop_time += time_delta
                group_neuron_spike_times.append(
                    neuron_spike_times[
                        np.logical_and(
                            neuron_spike_times >= start_time,
                            neuron_spike_times <= stop_time,
                        )
                    ]
                )
            if group_neuron_spike_times:
                group_spike_times.append(np.concatenate(group_neuron_spike_times))
            else:
                # No training coverage for this obs group; return empty spike
                # array so downstream encoding can still fit a (uninformative)
                # model without crashing on np.concatenate([]).
                group_spike_times.append(
                    np.asarray(neuron_spike_times[:0], dtype=neuron_spike_times.dtype)
                )

        return group_spike_times

    def fit_encoding_model(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        """
        Fit place fields to the data.

        Parameters
        ----------
        position_time : np.ndarray
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_training : np.ndarray, optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, optional
            Environment labels, by default None.
        weights : np.ndarray, optional, shape (n_time_position,)
            Weights for training data, by default None.

        Attributes
        ----------
        encoding_model_ : dict
            Dictionary holding the fitted encoding models (place fields) for each unique
            observation model configuration (environment, encoding group).
            The values depend on the chosen `sorted_spikes_algorithm`.
        """
        logger.info("Fitting place fields...")
        n_time = position.shape[0]
        position = position if position.ndim > 1 else position[:, np.newaxis]
        if is_training is None:
            is_training = np.ones((n_time,), dtype=bool)

        if encoding_group_labels is None:
            encoding_group_labels = np.zeros((n_time,), dtype=np.int32)

        if environment_labels is None:
            environment_labels = np.asarray(
                [self.environments[0].environment_name] * n_time
            )

        is_training = np.asarray(is_training).squeeze()
        is_nan = np.any(np.isnan(position), axis=1)
        position = position[~is_nan]
        position_time = position_time[~is_nan]
        is_training = is_training[~is_nan]
        encoding_group_labels = encoding_group_labels[~is_nan]
        environment_labels = environment_labels[~is_nan]
        if weights is not None:
            weights = weights[~is_nan]

        kwargs = self.sorted_spikes_algorithm_params
        if kwargs is None:
            kwargs = {}

        self.encoding_model_ = {}

        for obs in np.unique(self.observation_models):
            environment = self._get_environment_by_name(obs.environment_name)

            is_encoding = np.isin(encoding_group_labels, obs.encoding_group)
            is_environment = environment_labels == obs.environment_name
            likelihood_name = (obs.environment_name, obs.encoding_group)
            encoding_algorithm, _ = _SORTED_SPIKES_ALGORITHMS[
                self.sorted_spikes_algorithm
            ]
            is_group = is_training & is_encoding & is_environment
            if weights is not None:
                # Zero out weights for non-group data so position array
                # stays aligned with weights (required by KDE)
                group_weights = np.where(is_group, weights, 0.0)
            else:
                group_weights = None

            # GLM requires environment geometry that KDE derives internally
            glm_kwargs = {}
            if self.sorted_spikes_algorithm == "sorted_spikes_glm":
                glm_kwargs = {
                    "place_bin_edges": environment.place_bin_edges_,
                    "edges": environment.edges_,
                    "is_track_interior": environment.is_track_interior_,
                    "is_track_boundary": environment.is_track_boundary_,
                }

            # Filter algorithm_params to only those accepted by the
            # encoding function (KDE and GLM have different signatures).
            # Exclude glm_kwargs keys so user params can't override
            # environment geometry.
            sig = inspect.signature(encoding_algorithm)
            valid_params = set(sig.parameters.keys()) - set(glm_kwargs.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

            self.encoding_model_[likelihood_name] = encoding_algorithm(
                position_time=position_time,
                position=position,
                spike_times=self._get_group_spikes(
                    spike_times, is_group, position_time
                ),
                environment=environment,
                sampling_frequency=self.sampling_frequency,
                weights=group_weights,
                **glm_kwargs,
                **filtered_kwargs,
            )

    def fit(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict | None = None,
    ) -> "SortedSpikesDetector":
        """
        Fit the detector to the data.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims)
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array indicating training data, by default None.
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            Group labels for encoding, by default None.
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Environment labels, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.

        Returns
        -------
        SortedSpikesDetector
            Fitted detector instance.
        """
        self._fit(
            position,
            is_training,
            encoding_group_labels,
            environment_labels,
            discrete_transition_covariate_data,
        )
        self.fit_encoding_model(
            position_time,
            position,
            spike_times,
            is_training,
            encoding_group_labels,
            environment_labels,
        )
        return self

    def compute_log_likelihood(
        self,
        time: np.ndarray,
        position_time: np.ndarray,
        position: np.ndarray | None,
        spike_times: list[np.ndarray],
        is_missing: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Compute the log likelihood for the given data.

        Notes
        -----
        When ``local_position_std`` is set, the per-bin value returned
        for local-state entries is not a pure observation likelihood.
        It combines the spatial spike likelihood with the position
        uncertainty kernel::

            log_likelihood[local, b, t] =
                log P(spikes_t | bin b)                   # spatial likelihood
              + log_kernel(b | animal_pos_t)              # Gaussian anchor
              + log(n_interior_bins)                      # mass-balance

        The kernel term is a time-varying state-specific spatial prior.
        Mathematically, injecting it into the likelihood is equivalent
        to injecting it into the transition matrix — both multiply into
        the HMM forward step — but avoids breaking the static-transition
        assumption used by ``jax.lax.scan``. The ``log(n_interior_bins)``
        constant compensates for the ``1/n_bins`` uniform continuous IC
        so the effective per-bin prior is ``exp(log_kernel)`` itself.

        The kernel is part of the likelihood model, not an ad-hoc
        post-processing step, so the posterior returned by ``predict()``
        is a valid probability distribution under the modified
        generative model. Users comparing ``local_position_std=None`` to
        a finite value will see different posteriors (the models
        differ); users reading the raw ``log_likelihood`` (via
        ``return_outputs='log_likelihood'``) should be aware that
        local-state entries are *not* pure ``log P(spikes | state, bin)``
        — they include the kernel and mass-balance terms.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position_time : np.ndarray, shape (n_time_position,)
            Time points for position data.
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data.
        spike_times : list of np.ndarray
            Spike times for each neuron.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.

        Returns
        -------
        log_likelihood : jnp.ndarray, shape (n_time, n_state_bins)
        """
        logger.info("Computing log likelihood...")
        n_time = len(time)

        non_local_penalty = getattr(self, "non_local_position_penalty", 0.0)
        needs_position = (
            np.any([obs.is_local for obs in self.observation_models])
            or non_local_penalty > 0
            or self.local_position_std is not None
        )
        if position is None and needs_position:
            reason = []
            if np.any([obs.is_local for obs in self.observation_models]):
                reason.append("local observation models")
            if non_local_penalty > 0:
                reason.append("non_local_position_penalty > 0")
            if self.local_position_std is not None:
                reason.append("local_position_std is set")
            raise ValidationError(
                f"Missing required parameter: position (needed for {', '.join(reason)})",
                expected="position array with shape (n_time, n_dims)",
                got="None",
                hint="Provide position data or set non_local_position_penalty=0.0 to disable the penalty",
                example="    results = detector.predict(spikes=spikes_test, time=time_test, position=position_test)",
            )
        if position is not None:
            self._validate_position_dimensionality(position, context="predict")

        if is_missing is None:
            is_missing = np.zeros((n_time,), dtype=bool)

        _, likelihood_func = _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm]

        # Pre-compute state bins mask for all states (avoid recomputation in loop)
        interior_state_ind = self.state_ind_[self.is_track_interior_state_bins_]
        state_bin_masks = {
            state_id: interior_state_ind == state_id
            for state_id in range(len(self.observation_models))
        }

        # Use dict for O(1) lookup instead of list
        computed_likelihoods = {}

        # Compute all likelihoods first (stays in JAX/GPU if available)
        likelihood_results = {}
        for state_id, obs in enumerate(self.observation_models):
            # Multi-bin local computes full spatial likelihood (is_local=False)
            # then adds position kernel afterward
            effective_is_local = obs.is_local and self.local_position_std is None

            likelihood_name = (
                obs.environment_name,
                obs.encoding_group,
                effective_is_local,
                obs.is_no_spike,
            )

            if obs.is_no_spike:
                likelihood_results[state_id] = predict_no_spike_log_likelihood(
                    time, spike_times, self.no_spike_rate
                )
            elif likelihood_name not in computed_likelihoods:
                likelihood_results[state_id] = likelihood_func(
                    time,
                    position_time,
                    position,
                    spike_times,
                    **self.encoding_model_[likelihood_name[:2]],
                    is_local=effective_is_local,
                )
                computed_likelihoods[likelihood_name] = state_id
            else:
                # Will reuse previously computed likelihood
                likelihood_results[state_id] = computed_likelihoods[likelihood_name]

        # Assemble final array (single pass, stays in JAX)
        log_likelihood = jnp.zeros(
            (n_time, self.is_track_interior_state_bins_.sum()), dtype=jnp.float32
        )

        for state_id in range(len(self.observation_models)):
            is_state_bin = state_bin_masks[state_id]
            result = likelihood_results[state_id]

            if isinstance(result, int):
                # Reuse from previously computed state
                source_mask = state_bin_masks[result]
                log_likelihood = log_likelihood.at[:, is_state_bin].set(
                    log_likelihood[:, source_mask]
                )
            else:
                # Set new likelihood values
                log_likelihood = log_likelihood.at[:, is_state_bin].set(result)

        # Apply non-local position penalty if configured
        if non_local_penalty > 0:
            # Cache penalties per environment to avoid recomputation
            env_penalties = {}
            for state_id, obs in enumerate(self.observation_models):
                if not obs.is_local and not obs.is_no_spike:
                    env_name = obs.environment_name
                    if env_name not in env_penalties:
                        env = self._get_environment_by_name(env_name)
                        env_penalties[env_name] = (
                            self._compute_non_local_position_penalty(
                                time, position_time, position, env
                            )
                        )
                    is_state_bin = state_bin_masks[state_id]
                    log_likelihood = log_likelihood.at[:, is_state_bin].add(
                        env_penalties[env_name]
                    )

        # Apply local position kernel if configured
        if self.local_position_std is not None:
            env_kernels = {}
            for state_id, obs in enumerate(self.observation_models):
                if obs.is_local:
                    env_name = obs.environment_name
                    if env_name not in env_kernels:
                        env = self._get_environment_by_name(env_name)
                        env_kernels[env_name] = self._compute_local_position_kernel(
                            time, position_time, position, env
                        )
                    is_state_bin = state_bin_masks[state_id]
                    log_likelihood = log_likelihood.at[:, is_state_bin].add(
                        env_kernels[env_name]
                    )

        # Apply missing data mask
        is_missing = jnp.asarray(is_missing)
        return jnp.where(is_missing[:, jnp.newaxis], 0.0, log_likelihood)

    def predict(
        self,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        position: np.ndarray | None = None,
        position_time: np.ndarray | None = None,
        is_missing: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict | None = None,
        cache_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: str | list[str] | set[str] | None = None,
        save_log_likelihood_to_results: bool | None = None,
        save_causal_posterior_to_results: bool | None = None,
    ) -> xr.Dataset:
        """
        Predict the posterior probabilities for the given data.

        Parameters
        ----------
        spike_times : list of np.ndarray
            Spike times for each neuron.
        time : np.ndarray, shape (n_time,)
            Time points for decoding.
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        position_time : np.ndarray, shape (n_time_position,), optional
            Time points for position data, by default None.
        is_missing : np.ndarray, shape (n_time,), optional
            Boolean array indicating missing data, by default None.
        discrete_transition_covariate_data : dict or pd.DataFrame, optional
            Covariate data for covariate-dependent discrete transition, by default None.
        cache_likelihood : bool, optional
            Whether to cache the log likelihoods, by default False.
        n_chunks : int, optional
            Splits data into chunks for processing, by default 1
        return_outputs : str, list of str, set of str, or None, optional
            Controls which optional outputs are returned. See ClusterlessDetector.predict
            for full documentation. By default None.
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead. By default None.
        save_causal_posterior_to_results : bool, optional
            DEPRECATED. Use return_outputs='filter' instead. By default None.

        Returns
        -------
        xr.Dataset
            Predicted posterior probabilities.
        """
        if position is not None and position_time is None:
            raise ValidationError(
                "position_time is required when position is provided",
                expected="position_time array with shape (n_time_position,)",
                got="None",
                hint="Provide position_time corresponding to the position samples",
                example="    results = detector.predict(\n"
                "        spike_times=spike_times, time=time,\n"
                "        position=position, position_time=position_time\n"
                "    )",
            )

        if position is not None:
            position = position[:, np.newaxis] if position.ndim == 1 else position
            nan_position = np.any(np.isnan(position), axis=1)
            if np.any(nan_position) and is_missing is None:
                is_missing = nan_position
            elif np.any(nan_position) and is_missing is not None:
                is_missing = np.logical_or(is_missing, nan_position)

        if is_missing is not None and len(is_missing) != len(time):
            raise ValueError(
                f"Length of is_missing must match length of time. Time is n_samples: {len(time)}"
            )

        # Handle deprecated boolean flags
        if (
            save_log_likelihood_to_results is not None
            or save_causal_posterior_to_results is not None
        ):
            warnings.warn(
                "save_log_likelihood_to_results and save_causal_posterior_to_results "
                "are deprecated. Use return_outputs parameter instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Convert old flags to new format
            outputs_from_flags = set()
            if save_log_likelihood_to_results:
                outputs_from_flags.add("log_likelihood")
            if save_causal_posterior_to_results:
                outputs_from_flags.add("filter")

            if return_outputs is not None:
                raise ValueError(
                    "Cannot specify both return_outputs and deprecated "
                    "save_*_to_results flags. Use return_outputs only."
                )
            return_outputs = outputs_from_flags if outputs_from_flags else None

        # Normalize return_outputs to canonical set
        requested_outputs = _normalize_return_outputs(return_outputs)

        # Automatically enable caching if log_likelihood is requested
        if "log_likelihood" in requested_outputs and not cache_likelihood:
            cache_likelihood = True

        predicted_transitions = None
        if discrete_transition_covariate_data is not None:
            if self.discrete_transition_coefficients_ is None:
                raise ValueError(
                    "discrete_transition_coefficients_ is None but covariate data provided"
                )
            predicted_transitions = predict_discrete_state_transitions(
                self.discrete_transition_design_matrix_,
                self.discrete_transition_coefficients_,
                discrete_transition_covariate_data,
            )
            _validate_covariate_time_length(predicted_transitions, time)

        (
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            causal_state_probabilities,
            predictive_state_probabilities,
            log_likelihood,
            causal_posterior,
            predictive_posterior,
        ) = self._predict(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
            ),
            is_missing=is_missing,
            cache_likelihood=cache_likelihood,
            n_chunks=n_chunks,
            discrete_state_transitions=predicted_transitions,
        )

        return self._convert_results_to_xarray(
            time,
            acausal_posterior,
            acausal_state_probabilities,
            marginal_log_likelihood,
            log_likelihood=(
                log_likelihood if "log_likelihood" in requested_outputs else None
            ),
            causal_posterior=(
                causal_posterior if "filter" in requested_outputs else None
            ),
            causal_state_probabilities=(
                causal_state_probabilities if "filter" in requested_outputs else None
            ),
            predictive_state_probabilities=(
                predictive_state_probabilities
                if "predictive" in requested_outputs
                else None
            ),
            predictive_posterior=(
                predictive_posterior
                if "predictive_posterior" in requested_outputs
                else None
            ),
        )

    def estimate_parameters(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        is_missing: np.ndarray | None = None,
        is_training: np.ndarray | None = None,
        encoding_group_labels: np.ndarray | None = None,
        environment_labels: np.ndarray | None = None,
        discrete_transition_covariate_data: pd.DataFrame | dict | None = None,
        estimate_initial_conditions: bool = True,
        estimate_discrete_transition: bool = True,
        estimate_encoding_model: bool = True,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        cache_likelihood: bool = True,
        store_log_likelihood: bool = False,
        n_chunks: int = 1,
        return_outputs: str | list[str] | set[str] | None = None,
        save_log_likelihood_to_results: bool | None = None,
        min_encoding_local_mass: float = 1.0,
        min_encoding_local_ess: float = 1.0,
        encoding_update_damping: float = 0.0,
    ) -> xr.Dataset:
        """
        Estimate the initial conditions and transition probabilities
         using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time of each position sample
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        spike_times : list of np.ndarray, len (n_neurons,)
            Each element of the list is an array of spike times for a neuron
        time : np.ndarray, shape (n_time,)
            Decoding time points
        is_missing : np.ndarray, shape (n_time,), optional
            Denote missing samples, None includes all samples, by default None
        is_training : np.ndarray, shape (n_time_position,), optional
            Boolean array where True values include the sample in estimating the firing rate by
            position, None includes all samples, by default None
        encoding_group_labels : np.ndarray, shape (n_time_position,), optional
            If place fields should correspond to each state, label each time point with the group name
            For example, some points could correspond to inbound trajectories and some outbound, by default None
        environment_labels : np.ndarray, shape (n_time_position,), optional
            Labels denoting which environment the sample corresponds to, by default None
        discrete_transition_covariate_data : dict, optional
            Covariate data for a covariate dependent discrete transition.
            A dict-like object that can be used to look up variables, by default None
        estimate_initial_conditions : bool, optional
            Estimate the initial conditions, by default True
        estimate_discrete_transition : bool, optional
            Estimate the discrete transition matrix, by default True
        estimate_encoding_model : bool, optional
            Estimate the place fields based on the Local state, by default True.
        max_iter : int, optional
            Maximuim number of EM iterations, by default 20
        tolerance : float, optional
            Convergence tolerance for EM, by default 0.0001
        cache_likelihood : bool, optional
            Store the likelihood for faster iterations, by default True
        store_log_likelihood : bool, optional
            Whether to store the log likelihoods in self.log_likelihoods_, by default False.
        n_chunks : int, optional
            Number of chunks for processing, by default 1
        return_outputs : str, list of str, set of str, or None, optional
            Controls which optional outputs are returned. See predict() for full
            documentation of options. By default None.
        save_log_likelihood_to_results : bool, optional
            DEPRECATED. Use return_outputs='log_likelihood' instead. By default None.
        min_encoding_local_mass : float, optional
            Minimum sum of local state weights to update encoding model. By default 1.0.
        min_encoding_local_ess : float, optional
            Minimum effective sample size of local weights to update encoding. By default 1.0.
        encoding_update_damping : float, optional
            Damping factor in [0, 1) for encoding updates. By default 0.0.

        Returns
        -------
        xr.Dataset
            Results of the decoding
        """
        position = position[:, np.newaxis] if position.ndim == 1 else position
        self._encoding_model_data = {
            "position_time": position_time,
            "position": position,
            "spike_times": spike_times,
            "is_training": is_training,
            "encoding_group_labels": encoding_group_labels,
            "environment_labels": environment_labels,
        }
        # Mirror predict(): treat NaN positions as missing observations during
        # the E-step so EM and predict use the same local-likelihood handling.
        nan_position = np.any(np.isnan(position), axis=1)
        if np.any(nan_position):
            if is_missing is None:
                is_missing = nan_position
            else:
                is_missing = np.logical_or(is_missing, nan_position)
        self.fit(
            position_time,
            position,
            spike_times,
            is_training=is_training,
            encoding_group_labels=encoding_group_labels,
            environment_labels=environment_labels,
            discrete_transition_covariate_data=discrete_transition_covariate_data,
        )

        return super().estimate_parameters(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
            ),
            is_missing=is_missing,
            estimate_initial_conditions=estimate_initial_conditions,
            estimate_discrete_transition=estimate_discrete_transition,
            estimate_encoding_model=estimate_encoding_model,
            max_iter=max_iter,
            tolerance=tolerance,
            cache_likelihood=cache_likelihood,
            store_log_likelihood=store_log_likelihood,
            n_chunks=n_chunks,
            return_outputs=return_outputs,
            save_log_likelihood_to_results=save_log_likelihood_to_results,
            min_encoding_local_mass=min_encoding_local_mass,
            min_encoding_local_ess=min_encoding_local_ess,
            encoding_update_damping=encoding_update_damping,
        )

    def most_likely_sequence(
        self,
        position_time: np.ndarray,
        position: np.ndarray,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        is_missing: np.ndarray | None = None,
        n_chunks: int = 1,
    ) -> pd.DataFrame:
        """Find the most likely sequence of states.

        Parameters
        ----------
        position_time : np.ndarray, shape (n_time_position,)
            Time of each position sample
        position : np.ndarray, shape (n_time_position, n_position_dims), optional
            Position data, by default None.
        spike_times : list of np.ndarray, len (n_neurons,)
            Each element of the list is an array of spike times for a neuron
        time : np.ndarray, shape (n_time,)
            Decoding time points
        is_missing : np.ndarray, shape (n_time,), optional
            Denote missing samples, None includes all samples, by default None
        n_chunks : int, optional
            Number of chunks for processing, by default 1

        Returns
        -------
        most_likely_sequence : pd.DataFrame, shape (n_time, n_cols)
        """
        return super().most_likely_sequence(
            time=time,
            log_likelihood_args=(
                position_time,
                position,
                spike_times,
            ),
            is_missing=is_missing,
            n_chunks=n_chunks,
        )
