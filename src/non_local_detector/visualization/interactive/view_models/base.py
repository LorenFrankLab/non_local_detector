"""Core view-model dataclasses (Phase 1b: ``RunBundle`` only).

Phase 1c will add ``ViewState``, ``PositionGrid``, ``WindowPayload``,
``BinPayload``, and ``CellSlice`` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr

from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
)
from non_local_detector.visualization.interactive.view_models.series import (
    MetricSpec,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


ExtraMetricValue = Union[MetricSpec, "pd.Series"]


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

        # Monotonic position_time.
        position_time = np.asarray(self.position_time)
        if position_time.ndim != 1:
            raise ValueError(
                f"RunBundle.position_time must be 1D. Got shape "
                f"{position_time.shape}."
            )
        if not np.all(np.diff(position_time) > 0):
            raise ValueError(
                "RunBundle.position_time must be strictly monotonic "
                "increasing."
            )

        # Position dimensionality vs detector environment.
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
        if position.shape[0] != position_time.shape[0]:
            raise ValueError(
                f"RunBundle.position has {position.shape[0]} samples "
                f"but position_time has {position_time.shape[0]}."
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

        # Within-bundle overlay-name uniqueness.
        overlay_names = [ovl.name for ovl in self.event_overlays]
        duplicates = sorted({n for n in overlay_names if overlay_names.count(n) > 1})
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
