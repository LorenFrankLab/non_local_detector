"""Core view-model dataclasses for the interactive viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr

from non_local_detector._validation import (
    ensure_array_1d,
    ensure_matching_lengths,
    ensure_monotonic_increasing,
)
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
    find_duplicate_overlay_names,
)
from non_local_detector.visualization.interactive.view_models.series import (
    MetricSpec,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


ExtraMetricValue = Union[MetricSpec, "pd.Series"]


@dataclass(frozen=True)
class ViewState:
    """Per-load request snapshot.

    The window-load worker tags its result with ``request_id``; the
    viewer drops a result if its ``request_id`` is older than the latest
    committed one. Mirrors the statespacecheck pattern.

    Carries only what's needed to fulfil one window-load request.
    Pinned-event state, overlay state, and active-run state live on
    ``ViewerCore`` (those don't influence which window the worker
    fetches).
    """

    request_id: int
    t_center: float
    t_width: float
    load_acausal: bool = False


@dataclass(frozen=True)
class PositionGrid:
    """Position-axis description, abstracted over 1D / 2D.

    v1 ships only the 1D path (``ndim == 1``). v3+ extends to 2D
    decoders (``ndim == 2``) by populating the second-axis fields.
    Panels that visualize position-distributions take a
    ``PositionGrid`` and dispatch on ``.ndim``.
    """

    ndim: int
    centers: np.ndarray
    is_interior: np.ndarray | None = None
    centers_y: np.ndarray | None = None
    is_interior_y: np.ndarray | None = None

    @classmethod
    def from_environment(cls, environment) -> PositionGrid:
        """Build a ``PositionGrid`` from a fitted ``Environment``."""
        centers = np.asarray(environment.place_bin_centers_)
        n_pos_dims = centers.shape[1]
        is_interior = (
            np.asarray(environment.is_track_interior_).ravel()
            if environment.is_track_interior_ is not None
            else None
        )
        if n_pos_dims == 1:
            return cls(ndim=1, centers=centers.squeeze(-1), is_interior=is_interior)
        if n_pos_dims == 2:
            return cls(
                ndim=2,
                centers=centers[:, 0],
                centers_y=centers[:, 1],
                is_interior=is_interior,
            )
        raise ValueError(
            f"PositionGrid supports 1D or 2D environments, got n_pos_dims={n_pos_dims}."
        )


@dataclass(frozen=True)
class WindowPayload:
    """Output of a window load — what a ``TimeAxisPanel.update_window`` consumes.

    Each field carries data already shaped to ``(n_visible, ...)`` for
    the window the worker resolved. ``request_id`` lets the viewer
    drop stale results.
    """

    request_id: int
    time: np.ndarray
    indices: slice
    posterior: np.ndarray | None = None
    likelihood: np.ndarray | None = None
    predictive: np.ndarray | None = None
    state_probabilities: np.ndarray | None = None
    # 1D position (true behaviour) interpolated onto ``time``. Used
    # for the white trace overlaid on the posterior + likelihood
    # heatmaps. ``None`` when the bundle has no position or when the
    # position is 2D (heatmap overlay is 1D only in v1).
    position: np.ndarray | None = None


@dataclass(frozen=True)
class CellSlice:
    """Per-cell row payload for the SlicePanel."""

    cell_id: int
    place_field_norm: np.ndarray
    spike_count: int = 0


@dataclass(frozen=True)
class BinPayload:
    """Output of a single-bin load — what a ``BinSyncedPanel.update_for_index`` consumes.

    ``top_curve`` is the population-likelihood (or fallback collapsed
    posterior) over position; ``predictive_curve`` is the predictive
    overlay; ``cells`` are the per-cell rows for cells that fired in
    this bin.
    """

    t_idx: int
    t: float
    top_curve: np.ndarray | None = None
    top_curve_label: str = ""
    predictive_curve: np.ndarray | None = None
    cells: tuple[CellSlice, ...] = ()


@dataclass
class RunBundle:
    """User-facing viewer input bundle.

    Mutable so users can attach ``event_overlays`` / ``extra_metrics``
    after construction. ``__post_init__`` validates internal
    consistency once at construction; the data source revalidates the
    cross-bundle invariants (time-grid alignment, overlay schema)
    when bundles are loaded together.

    Parameters
    ----------
    results : xr.Dataset
        Decoder output. Must contain ``acausal_posterior`` and
        ``acausal_state_probabilities``. May optionally contain
        ``log_likelihood`` and ``predictive_posterior`` (panels that
        consume them disable themselves with a clear title-bar
        message when the array is missing).
    detector : _DetectorBase
        Fitted detector. Must expose ``state_ind_``, ``state_names``,
        ``encoding_model_``, and ``environments[0]``.
    spike_times : list[np.ndarray]
        Per-cell spike times in absolute seconds. Length must match
        the detector's encoding-model neuron count.
    position_time : np.ndarray, shape (n_position_time,)
        Monotonic absolute time in seconds for each position sample.
    position : np.ndarray
        Shape ``(n_position_time,)`` for 1D detectors,
        ``(n_position_time, 2)`` for 2D detectors.
    speed : np.ndarray, optional
        Shape ``(n_position_time,)``.
    events : pd.DataFrame, optional
        Per-spike event metrics (HPD overlap, KL divergence, spike
        prob).
    extra_metrics : dict[str, MetricSpec | pd.Series]
    event_overlays : list[EventOverlay]
        Names must be unique within the bundle (so the navigator's
        ``active_overlay_name`` resolves unambiguously).
    """

    results: xr.Dataset
    detector: _DetectorBase
    spike_times: list[np.ndarray]
    position_time: np.ndarray
    position: np.ndarray
    speed: np.ndarray | None = None
    events: pd.DataFrame | None = None
    extra_metrics: dict[str, ExtraMetricValue] = field(default_factory=dict)
    event_overlays: list[EventOverlay] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Required results variables.
        for var in ("acausal_posterior", "acausal_state_probabilities"):
            if var not in self.results:
                raise ValueError(
                    f"RunBundle.results is missing required variable {var!r}. "
                    "Got variables: "
                    f"{sorted(self.results.data_vars)!r}"
                )

        position_time = np.asarray(self.position_time)
        ensure_array_1d(position_time, "RunBundle.position_time")
        ensure_monotonic_increasing(
            position_time, "RunBundle.position_time", strict=True
        )

        env = self.detector.environments[0]
        n_pos_dims = env.place_bin_centers_.shape[1]
        position = np.asarray(self.position)
        # Accept 1D position arrays for 1D detectors (the model layer
        # auto-newaxes them); compare against the column count for 2D.
        position_n_cols = 1 if position.ndim == 1 else position.shape[1]
        if position_n_cols != n_pos_dims:
            raise ValueError(
                f"RunBundle.position has {position_n_cols} spatial "
                f"dim(s) but the detector's environment expects "
                f"{n_pos_dims}."
            )
        ensure_matching_lengths(
            position,
            position_time,
            "RunBundle.position",
            "RunBundle.position_time",
        )

        # spike_times length matches encoding-model neuron count.
        n_neurons_expected = self._infer_n_neurons()
        if n_neurons_expected is not None and (
            len(self.spike_times) != n_neurons_expected
        ):
            raise ValueError(
                f"RunBundle.spike_times has {len(self.spike_times)} "
                f"entries but the detector's encoding model carries "
                f"{n_neurons_expected} neurons."
            )

        duplicates = find_duplicate_overlay_names(self.event_overlays)
        if duplicates:
            raise ValueError(
                f"RunBundle.event_overlays has duplicate names: {duplicates}. "
                "Each overlay name must be unique within a bundle so that "
                "`active_overlay_name` resolves unambiguously."
            )

    def _infer_n_neurons(self) -> int | None:
        """Return per-encoding-model-entry n_neurons, or None if ambiguous."""
        encoding = getattr(self.detector, "encoding_model_", None)
        if not encoding:
            return None
        keys = list(encoding.keys())
        if len(keys) != 1:
            return None  # multi-environment / multi-group: skip strict check.
        place_fields = encoding[keys[0]].get("place_fields")
        if place_fields is None:
            return None
        return int(place_fields.shape[0])

    @property
    def available_outputs(self) -> set[str]:
        """Return the set of optional output arrays present in ``results``.

        ``acausal_posterior`` and ``acausal_state_probabilities`` are
        always required (validated in ``__post_init__``); this set
        reports which of the optional arrays
        (``log_likelihood``, ``predictive_posterior``,
        ``causal_posterior``, etc.) are also present.
        """
        return set(self.results.data_vars)

    @classmethod
    def from_predict(
        cls,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        time: np.ndarray,
        position: np.ndarray,
        position_time: np.ndarray,
        speed: np.ndarray | None = None,
        return_outputs: str | list[str] | set[str] | None = "all",
        events: pd.DataFrame | None = None,
        extra_metrics: dict[str, ExtraMetricValue] | None = None,
        event_overlays: list[EventOverlay] | None = None,
    ) -> RunBundle:
        """Convenience constructor: call ``detector.predict(...)`` then wrap.

        Defaults ``return_outputs="all"`` so the bundle ships with the
        full panel set (likelihood, predictive overlay, smoothed
        posterior, state probabilities) by default; pass a tighter
        value for memory savings.
        """
        results = detector.predict(  # type: ignore[attr-defined]
            spike_times=spike_times,
            time=time,
            position=position,
            position_time=position_time,
            return_outputs=return_outputs,
        )
        return cls(
            results=results,
            detector=detector,
            spike_times=spike_times,
            position_time=position_time,
            position=position,
            speed=speed,
            events=events,
            extra_metrics=extra_metrics or {},
            event_overlays=event_overlays or [],
        )
