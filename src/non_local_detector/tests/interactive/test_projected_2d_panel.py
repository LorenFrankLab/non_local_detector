"""GUI smoke test for ``QtProjected2DPanel``."""

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

pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    """Provide a singleton QApplication for projected-panel GUI tests."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        _ensure_qapplication,
    )

    app = _ensure_qapplication()
    yield app


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


def test_projected_2d_panel_row_provider_fires_when_cursor_outside_buffer(
    qapp,
) -> None:
    """Out-of-buffer cursor ticks fall back to the registered provider.

    Before the synchronous-fallback path, the panel froze on the last
    in-buffer frame during fast playback — same lag pattern the 2D
    image panels had. Regression for the contract-drift review.
    """
    from non_local_detector.visualization.interactive.panels.qt.projected_2d import (
        QtProjected2DPanel,
    )

    detector = _graph_detector()
    model = Projected2DModel(detector)
    panel = QtProjected2DPanel(model)

    # Buffered window covers indices [0, 1) only. Cursor at t_idx=10
    # is well outside, so without the provider the panel renders nothing.
    buffer_payload = WindowPayload(
        request_id=0,
        time=np.array([0.0]),
        indices=slice(0, 1),
        posterior=np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
        position=np.array([5.0]),
        position_2d=np.array([[5.0, 0.0]]),
    )
    panel.set_window_buffer(buffer_payload)

    calls = []

    def provider(t_idx: int):
        calls.append(t_idx)
        # Match the (n_state_bins,) shape the model's collapse expects.
        return np.array([0.0, 0.2, 0.6, 0.2, 0.0], dtype=np.float64), np.array(
            [3.0, 0.0]
        )

    panel.set_row_provider(provider)
    panel.update_for_index(10)
    assert calls == [10], "row provider should fire when cursor is outside buffer"
    assert len(panel._posterior_scatter.points()) == 5
    assert len(panel._animal_marker.points()) == 1


def test_projected_2d_panel_smoke_renders_one_bin(qapp) -> None:
    from non_local_detector.visualization.interactive.panels.qt.projected_2d import (
        QtProjected2DPanel,
    )

    detector = _graph_detector()
    model = Projected2DModel(detector)
    panel = QtProjected2DPanel(model)
    payload = WindowPayload(
        request_id=0,
        time=np.array([0.0]),
        indices=slice(0, 1),
        posterior=np.array([[0.0, 0.2, 0.6, 0.2, 0.0]], dtype=np.float64),
        position=np.array([5.0]),
        position_2d=np.array([[5.0, 0.0]]),
    )

    panel.set_window_buffer(payload)
    panel.update_for_index(0)

    assert len(panel._posterior_scatter.points()) == 5
    assert len(panel._animal_marker.points()) == 1
