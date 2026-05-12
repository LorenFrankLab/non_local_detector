"""Pure-Python tests for ``Projected2DModel`` (no Qt).

Kept separate from the panel smoke test so unit-only CI runs that
filter out ``pytest.mark.gui`` still cover the projection logic.
"""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.visualization.interactive.view_models.base import (
    WindowPayload,
)
from non_local_detector.visualization.interactive.view_models.projected_2d import (
    Projected2DModel,
)


def _graph_detector(n_pos: int = 5):
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    env = Environment(
        track_graph=graph,
        edge_order=[(0, 1)],
        edge_spacing=0.0,
        place_bin_size=10.0 / n_pos,
    )
    env.fit_place_grid(np.array([[0.0, 0.0], [10.0, 0.0]]))
    return SimpleNamespace(
        environments=[env],
        state_names=["Non-Local"],
        bin_sizes_=[env.place_bin_centers_.shape[0]],
        state_ind_=np.zeros(env.place_bin_centers_.shape[0], dtype=int),
    )


@pytest.mark.unit
def test_projected_2d_model_geometry_and_frame_split() -> None:
    """Geometry is static across cursor ticks; frame carries per-bin data."""
    detector = _graph_detector()
    model = Projected2DModel(detector)
    posterior = np.array([[0.0, 0.1, 0.8, 0.1, 0.0]], dtype=np.float64)
    payload = WindowPayload(
        request_id=0,
        time=np.array([0.0]),
        indices=slice(0, 1),
        posterior=posterior,
        position=np.array([5.0]),
        position_2d=np.array([[4.0, 0.5]]),
    )

    geometry = model.geometry()
    assert geometry.available
    assert geometry.bin_xy is not None
    assert geometry.bin_xy.shape == (5, 2)
    # Geometry carries no per-bin frame fields.
    assert not hasattr(geometry, "posterior")
    assert not hasattr(geometry, "animal_xy")

    frame = model.update_for_index(payload, 0)
    assert frame.available
    np.testing.assert_allclose(frame.posterior, posterior[0])
    np.testing.assert_allclose(frame.animal_xy, [4.0, 0.5])
    # Frame doesn't carry geometry — that's only on Projected2DGeometry.
    assert not hasattr(frame, "bin_xy")
    assert not hasattr(frame, "graph_segments")


@pytest.mark.unit
def test_projected_2d_model_falls_back_to_projected_position_without_2d() -> None:
    """Without ``position_2d``, the animal lays on the projected graph."""
    detector = _graph_detector()
    model = Projected2DModel(detector)
    payload = WindowPayload(
        request_id=0,
        time=np.array([0.0]),
        indices=slice(0, 1),
        posterior=np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float64),
        position=np.array([5.0]),
        position_2d=None,
    )

    frame = model.update_for_index(payload, 0)

    assert frame.available
    assert frame.animal_xy is not None
    # Linearized position 5.0 on a (0,0) → (10,0) edge projects to ~(5, 0).
    np.testing.assert_allclose(frame.animal_xy, [5.0, 0.0], atol=1e-6)


@pytest.mark.unit
def test_projected_2d_model_frame_at_row_drives_render_without_payload() -> None:
    """``frame_at_row`` lets callers drive the render without a payload.

    Used by the panel's synchronous-fallback path when the cursor moves
    past the buffered window during fast playback.
    """
    detector = _graph_detector()
    model = Projected2DModel(detector)
    posterior_row = np.array([0.0, 0.1, 0.8, 0.1, 0.0], dtype=np.float64)
    animal_xy = np.array([4.5, 0.0])

    frame = model.frame_at_row(posterior_row, animal_xy)

    assert frame.available
    np.testing.assert_allclose(frame.posterior, posterior_row)
    np.testing.assert_allclose(frame.animal_xy, animal_xy)

    # ``None`` row → unavailable frame with the appropriate message.
    no_data = model.frame_at_row(None, animal_xy)
    assert not no_data.available
    assert "posterior" in no_data.message.lower()


@pytest.mark.unit
def test_projected_2d_model_unavailable_for_2d_decoder() -> None:
    """A 2D decoder (no graph linearization) reports unavailable."""

    class _Env:
        track_graph = None
        place_bin_centers_ = np.zeros((4, 2))

    detector = SimpleNamespace(environments=[_Env()])
    model = Projected2DModel(detector)  # type: ignore[arg-type]
    assert not model.is_available
    assert "track_graph" in model.message
