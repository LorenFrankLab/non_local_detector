"""Projection model for the optional 1D-decode-on-2D-track panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from track_linearization import project_1d_to_2d

from non_local_detector.analysis.posterior import (
    collapse_posterior_to_position,
    select_reduction,
)

if TYPE_CHECKING:
    from non_local_detector.analysis.posterior import PosteriorReduction
    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.view_models.base import (
        WindowPayload,
    )


@dataclass(frozen=True)
class Projected2DGeometry:
    """Static graph + bin geometry. Computed once per active run."""

    available: bool
    message: str = ""
    # ``(n_pos, 2)`` XY for each 1D bin center, projected through the
    # track graph. ``None`` when the model is unavailable.
    bin_xy: np.ndarray | None = None
    # ``(2, 2)`` endpoint pairs for each graph edge.
    graph_segments: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class Projected2DFrame:
    """Per-cursor-bin projected posterior + animal position.

    The frame is intentionally separate from :class:`Projected2DGeometry`
    so a render pipeline can apply the static graph + bin layout once
    (on swap / setup) and only re-render the posterior + animal-XY
    overlay on each cursor tick.
    """

    available: bool
    message: str = ""
    # Collapsed posterior over the 1D bin centers, shape ``(n_pos,)``.
    posterior: np.ndarray | None = None
    # Raw 2D XY of the animal at the cursor's time bin.
    animal_xy: np.ndarray | None = None


class Projected2DModel:
    """Map a collapsed 1D posterior row back onto a track graph.

    Calls ``analysis.posterior.collapse_posterior_to_position``
    directly so the projected view stays bit-equivalent to the
    posterior heatmap without depending on a sibling
    ``PosteriorHeatmapModel`` instance.
    """

    def __init__(self, detector: _DetectorBase) -> None:
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
        self._reduction: PosteriorReduction | None = None

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
            self._reduction = select_reduction(
                detector.state_names, detector.bin_sizes_
            )
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

    def geometry(self) -> Projected2DGeometry:
        """Return the static graph + bin geometry.

        Render-pipeline lifecycle: call this once on construction and
        once after each ``set_active_run`` to refresh graph lines and
        bin positions. Cursor ticks should use :meth:`frame_at_row`
        (or :meth:`update_for_index` for the buffered convenience)
        instead.
        """
        return Projected2DGeometry(
            available=self._available,
            message=self._message,
            bin_xy=self._bin_xy,
            graph_segments=self._graph_segments,
        )

    def frame_at_row(
        self,
        posterior_row: np.ndarray | None,
        animal_xy: np.ndarray | None,
    ) -> Projected2DFrame:
        """Build a frame from a precomputed posterior row + animal XY.

        Lets callers drive the render without owning a
        :class:`WindowPayload` — useful for the synchronous fallback
        path when the cursor is past the buffered window.

        Parameters
        ----------
        posterior_row : np.ndarray, shape (n_state_bins,) or None
            Single row of ``acausal_posterior``. ``None`` returns an
            "unavailable" frame.
        animal_xy : np.ndarray, shape (2,) or None
            Pre-resolved animal XY at this bin. Callers without a
            raw 2D source can use :meth:`animal_xy_from_linear` to
            project a 1D linearized position back through the graph.
        """
        if not self._available or self._reduction is None:
            return Projected2DFrame(available=False, message=self._message)
        if posterior_row is None:
            return Projected2DFrame(
                available=False,
                message="Projected 2D requires posterior output",
            )
        collapsed = collapse_posterior_to_position(
            np.asarray(posterior_row), self._detector, self._reduction
        )
        return Projected2DFrame(
            available=True,
            posterior=collapsed,
            animal_xy=animal_xy if animal_xy is not None else None,
        )

    def animal_xy_from_linear(self, linear_pos: float | None) -> np.ndarray | None:
        """Project a 1D linearized position back onto the track graph."""
        if linear_pos is None or not np.isfinite(float(linear_pos)):
            return None
        try:
            xy = project_1d_to_2d(
                np.asarray([float(linear_pos)], dtype=float),
                self._detector.environments[0].track_graph,
                self._detector.environments[0].edge_order,
                self._detector.environments[0].edge_spacing,
            )[0]
        except Exception:
            return None
        return xy if np.all(np.isfinite(xy)) else None

    def update_for_index(
        self, payload: WindowPayload, t_idx: int
    ) -> Projected2DFrame:
        """Buffered-window convenience: build a frame from a payload + cursor.

        Use this when the cursor is inside the buffered window;
        out-of-buffer callers should drive :meth:`frame_at_row` directly
        from a synchronous data-source fetch.
        """
        if not self._available or self._bin_xy is None:
            return Projected2DFrame(available=False, message=self._message)
        if payload.posterior is None:
            return Projected2DFrame(
                available=False,
                message="Projected 2D requires posterior output",
            )
        local_idx = int(t_idx - payload.indices.start)
        if local_idx < 0 or local_idx >= payload.posterior.shape[0]:
            return Projected2DFrame(
                available=False,
                message="Cursor is outside the buffered window",
            )
        animal_xy = self._animal_xy_from_payload(payload, local_idx)
        return self.frame_at_row(payload.posterior[local_idx], animal_xy)

    def _animal_xy_from_payload(
        self, payload: WindowPayload, local_idx: int
    ) -> np.ndarray | None:
        """Resolve animal XY from a payload at ``local_idx``.

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
        return self.animal_xy_from_linear(float(payload.position[local_idx]))


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
