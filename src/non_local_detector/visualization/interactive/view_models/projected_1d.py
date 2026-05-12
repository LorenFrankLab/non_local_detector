"""Projection model for optional 2D-decode-on-1D-track diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from track_linearization import get_linearized_position

from non_local_detector.visualization.interactive.view_models.posterior import (
    PosteriorHeatmapModel,
)

if TYPE_CHECKING:
    from non_local_detector.visualization.interactive.view_models.base import (
        RunBundle,
        WindowPayload,
    )


class Projected1DModel:
    """Project a 2D posterior window onto an explicit 1D track graph.

    This is the mirror of ``Projected2DModel``: the decoder was fit in
    raw 2D coordinates, but the user provides the graph-linearization
    geometry they want to inspect against. The model bins each 2D place
    bin center by its linearized position, then sums posterior mass
    into those 1D bins per time row.
    """

    def __init__(self, run: RunBundle) -> None:
        self._posterior_model = PosteriorHeatmapModel(run.detector)
        self._bind(run)

    def set_active_run(self, run: RunBundle) -> None:
        """Rebind detector schema + projection geometry for a new run."""
        self._posterior_model.set_active_run(run.detector)
        self._bind(run)

    def _bind(self, run: RunBundle) -> None:
        self._detector = run.detector
        self._track_graph = run.projection_track_graph
        self._edge_order = run.projection_edge_order
        self._edge_spacing = run.projection_edge_spacing
        self._available = False
        self._message = "Projected 1D unavailable"
        self._linear_centers = np.empty(0, dtype=np.float64)
        self._linear_edges = np.empty(0, dtype=np.float64)
        self._valid_position_mask = np.empty(0, dtype=bool)
        self._linear_bin_index = np.empty(0, dtype=np.int64)

        if self._track_graph is None or self._edge_order is None:
            self._message = (
                "Projected 1D requires projection_track_graph and "
                "projection_edge_order"
            )
            return
        env = run.detector.environments[0]
        centers = np.asarray(env.place_bin_centers_, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2:
            self._message = "Projected 1D is only defined for 2D decoders"
            return
        try:
            linear = self._linearize_position(centers)
        except Exception as exc:  # pragma: no cover - defensive message path
            self._message = f"Projected 1D setup failed: {exc}"
            return
        if linear is None:
            self._message = "Projected 1D setup produced no finite linear bins"
            return

        valid = np.isfinite(linear)
        interior = getattr(env, "is_track_interior_", None)
        if interior is not None:
            interior_flat = np.asarray(interior, dtype=bool).ravel()
            if interior_flat.size == valid.size:
                valid &= interior_flat
        if not np.any(valid):
            self._message = "Projected 1D setup produced no valid interior bins"
            return

        bin_width = _projection_bin_width(env)
        finite_linear = linear[valid]
        lo = np.floor(float(np.nanmin(finite_linear)) / bin_width) * bin_width
        hi = np.ceil(float(np.nanmax(finite_linear)) / bin_width) * bin_width
        edges = np.arange(lo, hi + 1.5 * bin_width, bin_width, dtype=np.float64)
        if edges.size < 2:
            edges = np.array([lo - 0.5 * bin_width, lo + 0.5 * bin_width])
        idx = np.digitize(linear, edges, right=False) - 1
        idx = np.clip(idx, 0, edges.size - 2)

        self._linear_edges = edges
        self._linear_centers = (edges[:-1] + edges[1:]) / 2.0
        self._valid_position_mask = valid
        self._linear_bin_index = idx.astype(np.int64)
        self._available = True
        self._message = ""

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    @property
    def linear_centers(self) -> np.ndarray:
        return self._linear_centers

    def update_window(
        self, payload: WindowPayload
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(projected_posterior, projected_position)`` for a window."""
        if not self._available or payload.posterior is None:
            return None, None
        collapsed = self._posterior_model.update_window(payload.posterior)
        projected = self.project_curves(collapsed)
        position = self.linearize_window_position(payload.position)
        return projected, position

    def project_curves(self, curves_2d: np.ndarray) -> np.ndarray:
        """Sum ``(n_time, n_2d_bins)`` curves into linear-position bins."""
        if curves_2d.ndim != 2:
            raise ValueError(
                "Projected1DModel.project_curves expects a 2D array "
                f"(n_time, n_position_bins); got shape {curves_2d.shape}."
            )
        if curves_2d.shape[1] != self._valid_position_mask.size:
            raise ValueError(
                "Projected1DModel.project_curves got "
                f"{curves_2d.shape[1]} position bins, expected "
                f"{self._valid_position_mask.size}."
            )
        out = np.zeros((curves_2d.shape[0], self._linear_centers.size), dtype=np.float64)
        values = np.asarray(curves_2d, dtype=np.float64)
        for linear_idx in range(self._linear_centers.size):
            mask = self._valid_position_mask & (self._linear_bin_index == linear_idx)
            if np.any(mask):
                out[:, linear_idx] = np.nansum(values[:, mask], axis=1)
        return out

    def linearize_window_position(self, position: np.ndarray | None) -> np.ndarray | None:
        """Linearize raw 2D animal position for the heatmap trace."""
        if position is None:
            return None
        pos = np.asarray(position, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] == 0:
            return None
        try:
            return self._linearize_position(pos)
        except Exception:
            # The projected posterior remains useful even if the animal
            # trace has a bad sample/window; drop just the trace.
            return None

    def _linearize_position(self, position: np.ndarray) -> np.ndarray | None:
        if self._track_graph is None or self._edge_order is None:
            return None
        result = get_linearized_position(
            np.asarray(position, dtype=np.float64),
            self._track_graph,
            edge_order=self._edge_order,
            edge_spacing=self._edge_spacing,
        )
        return result.linear_position.to_numpy(dtype=np.float64)


def _projection_bin_width(env) -> float:
    place_bin_size = getattr(env, "place_bin_size", 1.0)
    if isinstance(place_bin_size, int | float):
        width = float(place_bin_size)
    else:
        values = np.asarray(place_bin_size, dtype=np.float64).ravel()
        values = values[np.isfinite(values) & (values > 0.0)]
        width = float(np.min(values)) if values.size else 1.0
    return width if np.isfinite(width) and width > 0.0 else 1.0
