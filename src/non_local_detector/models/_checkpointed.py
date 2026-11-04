"""Checkpointed prediction of the detectors.

Decoding on the checkpointed forward-backward driver
(``non_local_detector.checkpointed_inference``), which keeps one checkpoint
per chunk of decode bins and replays the backward pass, so working memory
scales with the chunk rather than the recording: ``predict`` with
``inference_mode="checkpointed"`` and the E-steps of structured EM.
``_DetectorBase`` inherits the methods of ``_CheckpointedPredictionMixin``.
Also here are the outputs of one forward-backward pass
(``_PosteriorOutputs``), which dense prediction and EM return too, and the
driver outputs ``return_outputs`` names (``_driver_outputs``).

Methods that read the detector's attributes annotate ``self`` as
``_DetectorBase``, which mypy only accepts with ``type: ignore[misc]`` on a
mixin.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

from non_local_detector._position_dims import get_position_dim_names
from non_local_detector.checkpointed_inference import (
    CheckpointedResult,
    default_chunk_size,
    default_replay_cache_bytes,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.models._input_validation import _decode_time_edges
from non_local_detector.models._results import (
    label_checkpointed_result,
    state_bin_positions,
)
from non_local_detector.types import OutputMode

try:
    from non_local_detector._version import __version__ as _PACKAGE_VERSION
except ImportError:
    _PACKAGE_VERSION = "unknown"

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


def _driver_outputs(requested_outputs: set[str], *, spatial: bool) -> set[str]:
    """Forward-backward outputs to compute for canonical ``return_outputs``.

    Always the smoothed state probabilities, and with ``spatial`` the
    smoothed posterior over state bins too. ``spatial=False`` (compact
    checkpointed output) computes the filtered state probabilities but not
    the filtered posterior.

    Parameters
    ----------
    requested_outputs : set of str
        Names from ``_normalize_return_outputs``.
    spatial : bool
        Whether posteriors over state bins are computed.

    Returns
    -------
    outputs : set of str
        Field names of ``_PosteriorOutputs``, as the checkpointed driver's
        ``return_outputs`` names them.
    """
    outputs = {"acausal_state_probabilities"}
    if spatial:
        outputs.add("acausal_posterior")
    if "filter" in requested_outputs:
        outputs.add("causal_state_probabilities")
        if spatial:
            outputs.add("causal_posterior")
    if "predictive" in requested_outputs:
        outputs.add("predictive_state_probabilities")
    if "predictive_posterior" in requested_outputs:
        outputs.add("predictive_posterior")
    if "log_likelihood" in requested_outputs:
        outputs.add("log_likelihood")
    return outputs


@dataclass(frozen=True)
class _PosteriorOutputs:
    """Outputs of one forward-backward pass over the decode bins.

    Outputs the pass did not compute are None. Posteriors over state bins can
    be device arrays: callers copy to the host only those they need there.

    Attributes
    ----------
    marginal_log_likelihood : float
    degenerate_indices : np.ndarray of int, shape (n_degenerate,)
        Bins whose log likelihoods are ``-inf`` in every state bin. Found
        during the forward pass, so they are complete whether or not the log
        likelihoods were kept.
    acausal_posterior : array, shape (n_time, n_state_bins), or None
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states), or None
    causal_state_probabilities : np.ndarray, shape (n_time, n_states), or None
    predictive_state_probabilities : np.ndarray, shape (n_time, n_states), or None
    log_likelihood : np.ndarray, shape (n_time, n_state_bins), or None
    causal_posterior : array, shape (n_time, n_state_bins), or None
    predictive_posterior : array, shape (n_time, n_state_bins), or None
    """

    marginal_log_likelihood: float
    degenerate_indices: np.ndarray
    acausal_posterior: Any = None
    acausal_state_probabilities: np.ndarray | None = None
    causal_state_probabilities: np.ndarray | None = None
    predictive_state_probabilities: np.ndarray | None = None
    log_likelihood: np.ndarray | None = None
    causal_posterior: Any = None
    predictive_posterior: Any = None

    def requested_results(self, requested_outputs: set[str]) -> dict[str, Any]:
        """The optional ``convert_results_to_xarray`` arrays that were requested.

        Parameters
        ----------
        requested_outputs : set of str
            Names from ``_normalize_return_outputs``.

        Returns
        -------
        arrays : dict
            ``convert_results_to_xarray`` keyword arguments; outputs that were
            not requested are None, posteriors are host arrays.
        """
        outputs = _driver_outputs(requested_outputs, spatial=True)
        return {
            "log_likelihood": (
                self.log_likelihood if "log_likelihood" in outputs else None
            ),
            "causal_posterior": (
                np.asarray(self.causal_posterior)
                if "causal_posterior" in outputs
                else None
            ),
            "causal_state_probabilities": (
                self.causal_state_probabilities
                if "causal_state_probabilities" in outputs
                else None
            ),
            "predictive_state_probabilities": (
                self.predictive_state_probabilities
                if "predictive_state_probabilities" in outputs
                else None
            ),
            "predictive_posterior": (
                np.asarray(self.predictive_posterior)
                if "predictive_posterior" in outputs
                else None
            ),
        }


class _CheckpointedPredictionMixin:
    """Checkpointed-prediction methods of ``_DetectorBase``."""

    def _checkpointed_likelihood(  # type: ignore[misc]
        self: "_DetectorBase",
        edges: np.ndarray,
        centers: np.ndarray,
        log_likelihood_args: tuple[Any, ...],
    ) -> Callable[..., Any]:
        """Row-slice likelihood callback for the checkpointed driver."""
        prepared = self._decode_likelihood_callback(edges, log_likelihood_args)

        def likelihood(
            full_edges: np.ndarray,  # noqa: ARG001 -- likelihood callback signature
            *,
            row_slice: slice | None,
            is_missing: np.ndarray | None,
        ) -> Any:
            return prepared(
                centers,
                *log_likelihood_args,
                row_slice=row_slice,
                is_missing=is_missing,
            )

        return likelihood

    def _checkpointed_transition_args(  # type: ignore[misc]
        self: "_DetectorBase", discrete: np.ndarray, n_rows: int
    ) -> dict[str, Any]:
        """Checkpointed-driver transition arguments over the interior state bins.

        Structured fits pass the fused operator; dense fits pass the dense
        interior matrices. Covariate transitions pass their per-row discrete
        matrices separately and must have one row per decode bin.
        """
        if discrete.ndim == 3 and discrete.shape[0] != n_rows:
            msg = "Predict covariate transitions must have one row per decode bin"
            raise ValidationError(msg)
        interior = self.is_track_interior_state_bins_
        states = self.state_ind_[interior]
        transition_args: dict[str, Any] = {}
        operator = getattr(self, "_continuous_transition_operator_", None)
        if operator is not None:
            operator = operator.restricted(interior)
            if discrete.ndim == 2:
                operator = operator.bind_discrete(discrete)
            else:
                transition_args["discrete_transition_matrix"] = discrete
            transition_args["transition_operator"] = operator.fused(
                self._max_dense_transition_bytes_
            )
        else:
            continuous = self.continuous_state_transitions_[np.ix_(interior, interior)]
            if discrete.ndim == 2:
                transition_args["transition_matrix"] = (
                    continuous * discrete[np.ix_(states, states)]
                )
            else:
                transition_args.update(
                    continuous_transition_matrix=continuous,
                    discrete_transition_matrix=discrete,
                )
        return transition_args

    def _checkpointed_e_step(  # type: ignore[misc]
        self: "_DetectorBase",
        time_edges: np.ndarray,
        log_likelihood_args: tuple[Any, ...],
        is_missing: np.ndarray | None,
        chunk_size: int,
        *,
        outputs: Iterable[str] = ("acausal_state_probabilities",),
        transition_statistics: str | None = None,
        replay_cache_bytes: int,
    ) -> CheckpointedResult:
        """One whole-recording E-step on the checkpointed driver.

        Parameters
        ----------
        time_edges : np.ndarray, shape (n_time + 1,)
        log_likelihood_args : tuple
        is_missing : np.ndarray, shape (n_time,), or None
        chunk_size : int
            Rows per checkpoint chunk.
        outputs : iterable of str
            Driver outputs. Spatial outputs are held in memory.
        transition_statistics : {None, "counts", "responses"}
        replay_cache_bytes : int
            Host bytes for forward-pass rows reused by the replay.

        Returns
        -------
        result : CheckpointedResult
        """
        from non_local_detector.checkpointed_inference import (
            _SPATIAL,
            checkpointed_forward_backward,
        )

        edges, centers = _decode_time_edges(
            time_edges, *self._learned_transition_time_bin_width()
        )
        interior = self.is_track_interior_state_bins_
        output_names = set(outputs)
        return checkpointed_forward_backward(
            edges,
            self.initial_conditions_[interior],
            self._checkpointed_likelihood(edges, centers, log_likelihood_args),
            state_ind=self.state_ind_[interior],
            n_states=len(self.state_names),
            is_missing=is_missing,
            chunk_size=chunk_size,
            output_mode="spatial" if output_names & _SPATIAL else "compact",
            return_outputs=output_names,
            dtype=np.float32,
            transition_statistics=transition_statistics,
            replay_cache_bytes=replay_cache_bytes,
            **self._checkpointed_transition_args(
                self.discrete_state_transitions_, len(centers)
            ),
        )

    def _predict_checkpointed(  # type: ignore[misc]
        self: "_DetectorBase",
        *,
        time_edges: np.ndarray,
        log_likelihood_args: tuple[Any, ...],
        is_missing: np.ndarray | None,
        discrete_transitions: np.ndarray | None,
        output_mode: OutputMode,
        chunk_size: int | None,
        result_path: str | Path | None,
        checkpoint_dir: str | Path | None,
        selected_intervals: np.ndarray | None,
        requested_outputs: set[str],
        max_read_bytes: int,
        replay_cache_bytes: int | None,
        em_attrs: Callable[[float], dict[str, Any]] | None = None,
    ) -> tuple[xr.Dataset, np.ndarray]:
        """Decode with complete conditioning and bounded spatial working memory.

        ``replay_cache_bytes`` of None uses ``default_replay_cache_bytes()``.
        ``em_attrs`` maps this pass's marginal log-likelihood to EM's
        attributes, which a result store keeps with the rest so that
        ``load_results`` returns them.

        Returns
        -------
        results : xr.Dataset
            The labelled results, lazily loaded from ``result_path`` when given.
        degenerate_indices : np.ndarray of int, shape (n_degenerate,)
            Recording rows whose log likelihoods are ``-inf`` in every state
            bin, including rows outside ``selected_intervals``.
        """
        from non_local_detector.checkpointed_inference import (
            checkpointed_forward_backward,
        )

        if replay_cache_bytes is None:
            replay_cache_bytes = default_replay_cache_bytes()

        if output_mode == "spatial" and result_path is None:
            msg = "Checkpointed spatial prediction requires result_path"
            raise ValidationError(
                msg,
                hint="Pass a new directory as result_path, or use output_mode='compact'.",
            )
        if output_mode == "compact" and requested_outputs & {
            "log_likelihood",
            "predictive_posterior",
        }:
            msg = "Spatial outputs (including return_outputs='all') require output_mode='spatial'"
            raise ValidationError(
                msg,
                got=f"return_outputs requesting {sorted(requested_outputs)}",
                hint="Compact output holds state probabilities only; request "
                "'filter' and 'predictive', or use output_mode='spatial' with a result_path.",
            )
        edges, centers = _decode_time_edges(
            time_edges, *self._learned_transition_time_bin_width()
        )
        interior = self.is_track_interior_state_bins_
        states = self.state_ind_[interior]
        if chunk_size is None:
            chunk_size = default_chunk_size(
                cast("int", np.count_nonzero(interior)), np.float32
            )
        discrete = (
            self.discrete_state_transitions_
            if discrete_transitions is None
            else discrete_transitions
        )
        transition_args = self._checkpointed_transition_args(discrete, len(centers))
        likelihood = self._checkpointed_likelihood(edges, centers, log_likelihood_args)
        selected: np.ndarray | None = None
        if selected_intervals is not None:
            intervals = np.asarray(selected_intervals, dtype=float)
            if (
                intervals.ndim != 2
                or intervals.shape[1] != 2
                or not np.isfinite(intervals).all()
                or np.any(intervals[:, 0] >= intervals[:, 1])
                or np.any(intervals[1:, 0] < intervals[:-1, 1])
            ):
                msg = "selected_intervals must be ordered non-overlapping start/stop pairs"
                raise ValidationError(msg)
            mask = np.zeros(len(centers), dtype=bool)
            # The tolerances are under 1% of a bin width, so bins more than one
            # bin outside an interval can never be selected; apply the rule only
            # from one bin before each start to one bin after each stop.
            firsts = np.maximum(np.searchsorted(edges[:-1], intervals[:, 0]) - 1, 0)
            lasts = np.minimum(
                np.searchsorted(edges[1:], intervals[:, 1], side="right") + 1,
                len(centers),
            )
            for (start, stop), first, last in zip(
                intervals, firsts, lasts, strict=True
            ):
                lower, upper = edges[first:last], edges[first + 1 : last + 1]
                width = upper - lower
                lower_tol = np.minimum(
                    4 * np.maximum(np.spacing(abs(start)), np.spacing(abs(lower))),
                    width * 0.01,
                )
                upper_tol = np.minimum(
                    4 * np.maximum(np.spacing(abs(stop)), np.spacing(abs(upper))),
                    width * 0.01,
                )
                mask[first:last] |= (lower >= start - lower_tol) & (
                    upper <= stop + upper_tol
                )
            selected = np.flatnonzero(mask)
            if selected.size == 0:
                msg = "selected_intervals select no complete decode bins"
                raise ValidationError(
                    msg,
                    got=f"{len(intervals)} interval(s)",
                    hint=(
                        "Intervals are in seconds and select bins whose start and "
                        f"stop edges both lie inside one; the recording spans "
                        f"[{edges[0]}, {edges[-1]}] in bins of {edges[1] - edges[0]}."
                    ),
                )
        outputs = _driver_outputs(requested_outputs, spatial=output_mode == "spatial")
        position = np.concatenate(state_bin_positions(self))
        metadata: dict[str, Any] = {
            "state_names": list(self.state_names),
            "environment_names": [
                obs.environment_name for obs in self.observation_models
            ],
            "encoding_groups": [
                obs.encoding_group.item()
                if isinstance(obs.encoding_group, np.generic)
                else obs.encoding_group
                for obs in self.observation_models
            ],
            "position": position.tolist(),
            "position_names": get_position_dim_names(position.shape[1]),
            "state_ind": self.state_ind_.tolist(),
            "interior_mask": interior.tolist(),
            "max_read_bytes": int(max_read_bytes),
        }
        attrs: dict[str, Any] = {
            "non_local_detector_version": _PACKAGE_VERSION,
            "time_units": "seconds",
            "time_bin_convention": "left_closed_right_open_final_closed",
            "conditioning": "whole_recording",
            "_native_metadata": metadata,
            "recording_rows": len(centers),
            "transition_representation": getattr(
                self, "_transition_representation_", "dense"
            ),
        }
        result = checkpointed_forward_backward(
            edges,
            self.initial_conditions_[interior],
            likelihood,
            state_ind=states,
            n_states=len(self.state_names),
            is_missing=is_missing,
            chunk_size=chunk_size,
            output_mode=output_mode,
            result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            return_outputs=outputs,
            selected_rows=selected,
            dtype=np.float32,
            max_read_bytes=max_read_bytes,
            result_attrs=(
                attrs
                if em_attrs is None
                else lambda marginal: {**attrs, **em_attrs(marginal)}
            ),
            replay_cache_bytes=replay_cache_bytes,
            **transition_args,
        )
        degenerate_indices = np.asarray(
            result.diagnostics.get("degenerate_indices", []), dtype=int
        )
        return label_checkpointed_result(result.dataset, metadata), degenerate_indices
