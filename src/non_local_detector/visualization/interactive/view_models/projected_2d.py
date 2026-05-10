"""Projection model for the optional 1D-decode-on-2D-track panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from track_linearization import project_1d_to_2d

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )


@dataclass(frozen=True)
class Projected2DPayload:
    """One-bin projected 2D panel payload."""

    available: bool
    message: str = ""
    bin_xy: np.ndarray | None = None
    posterior: np.ndarray | None = None
    animal_xy: np.ndarray | None = None
    graph_segments: tuple[np.ndarray, ...] = ()


class Projected2DModel:
    """Map a collapsed 1D posterior row back onto a track graph.

    The viewer's existing ``PosteriorHeatmapModel`` is reused for the
    per-row collapse; pass the same instance here so a model swap
    rebinds both views at once and we don't pay for two copies of
    ``_validate_rectangular_spatial`` + selected-state caches.
    """

    def __init__(
        self, detector: _DetectorBase, posterior_model: PosteriorHeatmapModel
    ) -> None:
        self._posterior_model = posterior_model
        self._bind(detector)

    def set_active_run(self, detector: _DetectorBase) -> None:
        """Rebind all cached projection geometry for a new detector."""
        self._bind(detector)

    def _bind(self, detector: _DetectorBase) -> None:
        self._detector = detector
        self._available = False
        self._message = "Projected 2D unavailable"
        self._bin_xy: np.ndarray | None = None
        self._graph_segments: tuple[np.ndarray, ...] = ()

        env = detector.environments[0]
        if getattr(env, "track_graph", None) is None:
            self._message = "Projected 2D requires a fitted track_graph environment"
            return
        centers = np.asarray(env.place_bin_centers_)
        if centers.ndim != 2 or centers.shape[1] != 1:
            self._message = "Projected 2D is only defined for 1D graph decoders"
            return
        try:
            self._bin_xy = _project_bin_centers_to_2d(env, centers.squeeze(axis=1))
            self._graph_segments = _graph_segments(env.track_graph)
        except Exception as exc:  # pragma: no cover - defensive message path
            self._message = f"Projected 2D setup failed: {exc}"
            return
        self._available = True
        self._message = ""

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    def geometry_payload(self) -> Projected2DPayload:
        """Return current static graph geometry for panel rebinds."""
        return Projected2DPayload(
            available=self._available,
            message=self._message,
            bin_xy=self._bin_xy,
            graph_segments=self._graph_segments,
        )

    def update_for_index(
        self, payload: WindowPayload, t_idx: int
    ) -> Projected2DPayload:
        """Return projected posterior + animal position for one cursor bin."""
        if not self._available or self._bin_xy is None:
            return Projected2DPayload(available=False, message=self._message)
        if payload.posterior is None:
            return Projected2DPayload(
                available=False,
                message="Projected 2D requires posterior output",
                bin_xy=self._bin_xy,
                graph_segments=self._graph_segments,
            )
        local_idx = int(t_idx - payload.indices.start)
        if local_idx < 0 or local_idx >= payload.posterior.shape[0]:
            return Projected2DPayload(
                available=False,
                message="Cursor is outside the buffered window",
                bin_xy=self._bin_xy,
                graph_segments=self._graph_segments,
            )
        posterior_row = self._posterior_model.collapse_at(
            payload.posterior, [local_idx]
        )[0]
        animal_xy = self._animal_xy(payload, local_idx)
        return Projected2DPayload(
            available=True,
            bin_xy=self._bin_xy,
            posterior=posterior_row,
            animal_xy=animal_xy,
            graph_segments=self._graph_segments,
        )

    def _animal_xy(self, payload: WindowPayload, local_idx: int) -> np.ndarray | None:
        """Return the animal's XY for ``local_idx``.

        Prefers ``payload.position_2d`` (raw track-space coordinates).
        Falls back to ``project_1d_to_2d(payload.position[idx], ...)`` —
        which lays the marker exactly on the linearized graph rather
        than at the raw position. That fallback is intentional: without
        a 2D source, the only XY we can produce is the same projection
        the bin centers used. Callers wanting a true raw-position
        marker must populate ``RunBundle.position_2d``.
        """
        if payload.position_2d is not None and local_idx < payload.position_2d.shape[0]:
            xy = np.asarray(payload.position_2d[local_idx], dtype=float)
            return xy if np.all(np.isfinite(xy)) else None
        if payload.position is None or local_idx >= payload.position.shape[0]:
            return None
        linear_pos = np.asarray([payload.position[local_idx]], dtype=float)
        try:
            xy = project_1d_to_2d(
                linear_pos,
                self._detector.environments[0].track_graph,
                self._detector.environments[0].edge_order,
                self._detector.environments[0].edge_spacing,
            )[0]
        except Exception:
            return None
        return xy if np.all(np.isfinite(xy)) else None


def _project_bin_centers_to_2d(env, centers_1d: np.ndarray) -> np.ndarray:
    """Return one XY coordinate per 1D place bin center."""
    nodes_df = getattr(env, "place_bin_centers_nodes_df_", None)
    if nodes_df is not None and {"x_position", "y_position"}.issubset(nodes_df.columns):
        xy = nodes_df[["x_position", "y_position"]].to_numpy(dtype=float)
        if xy.shape[0] == centers_1d.shape[0] and np.isfinite(xy).any():
            return xy
    return np.asarray(
        project_1d_to_2d(
            centers_1d,
            env.track_graph,
            env.edge_order,
            env.edge_spacing,
        ),
        dtype=float,
    )


def _graph_segments(track_graph) -> tuple[np.ndarray, ...]:
    """Return graph edge endpoints as ``(2, 2)`` arrays."""
    segments: list[np.ndarray] = []
    for node_a, node_b in track_graph.edges:
        pos_a = np.asarray(track_graph.nodes[node_a]["pos"], dtype=float)
        pos_b = np.asarray(track_graph.nodes[node_b]["pos"], dtype=float)
        if pos_a.size >= 2 and pos_b.size >= 2:
            segments.append(np.vstack([pos_a[:2], pos_b[:2]]))
    return tuple(segments)
