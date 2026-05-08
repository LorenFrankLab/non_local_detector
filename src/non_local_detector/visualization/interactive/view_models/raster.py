"""``RasterModel`` — per-cell spike-time slicing + place-field-peak sort."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.analysis.place_fields import extract_per_cell_place_fields

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.view_models.base import (
        SpikeEventIndex,
    )


@dataclass(frozen=True)
class RasterPayload:
    """One row per (sorted) cell of spike times within the visible window."""

    cell_label: str
    sort_indices: np.ndarray
    spike_times_per_cell: list[np.ndarray]
    event_ids_per_cell: list[np.ndarray]


class RasterModel:
    """Per-cell raster + place-field-peak sort.

    On bind we compute the sort order from each cell's place-field
    peak (in centimeters); this matches the static-plot convention.
    Detectors that don't expose per-cell place fields (multi-group /
    clusterless) fall back to insertion order with an "Electrode
    Group" label.

    ``update_window`` slices each cell's spike-time array to the
    visible time window and returns them in sorted order.
    """

    def __init__(
        self,
        detector: _DetectorBase,
        spike_times: list[np.ndarray],
        event_index: SpikeEventIndex | None = None,
    ) -> None:
        self._spike_times = list(spike_times)
        self._event_index = event_index
        self._bind(detector)

    def _bind(self, detector: _DetectorBase) -> None:
        self._detector = detector
        env = detector.environments[0]
        try:
            place_fields = extract_per_cell_place_fields(detector)
            self._sort_indices = np.argsort(
                env.place_bin_centers_[np.nanargmax(place_fields, axis=1)].squeeze()
            )
            self._cell_label = "Neuron"
        except (KeyError, ValueError):
            self._sort_indices = np.arange(len(self._spike_times))
            self._cell_label = "Electrode\nGroup"

    @property
    def detector(self) -> _DetectorBase:
        return self._detector

    @property
    def sort_indices(self) -> np.ndarray:
        return self._sort_indices

    @property
    def cell_label(self) -> str:
        return self._cell_label

    def set_active_run(
        self,
        detector: _DetectorBase,
        spike_times: list[np.ndarray] | None = None,
        event_index: SpikeEventIndex | None = None,
    ) -> None:
        """Rebind to a new detector + (optionally) new spike-times."""
        if spike_times is not None:
            self._spike_times = list(spike_times)
        self._event_index = event_index
        self._bind(detector)

    def update_window(self, t_start: float, t_stop: float) -> RasterPayload:
        """Return per-cell spike times clipped to ``[t_start, t_stop)``."""
        per_cell = []
        per_cell_event_ids = []
        for cell_id in self._sort_indices:
            if self._event_index is None:
                spikes = self._spike_times[cell_id]
                i_start = int(np.searchsorted(spikes, t_start, side="left"))
                i_stop = int(np.searchsorted(spikes, t_stop, side="left"))
                per_cell.append(spikes[i_start:i_stop])
                per_cell_event_ids.append(
                    np.full(i_stop - i_start, -1, dtype=np.int64)
                )
            else:
                event_ids = self._event_index.event_ids_for_cell_window(
                    int(cell_id), t_start, t_stop
                )
                per_cell.append(self._event_index.times[event_ids])
                per_cell_event_ids.append(event_ids)
        return RasterPayload(
            cell_label=self._cell_label,
            sort_indices=self._sort_indices,
            spike_times_per_cell=per_cell,
            event_ids_per_cell=per_cell_event_ids,
        )
