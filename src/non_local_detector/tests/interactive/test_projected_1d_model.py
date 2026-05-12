"""Pure-Python tests for ``Projected1DModel``."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from non_local_detector.visualization.interactive.view_models.base import (
    WindowPayload,
)
from non_local_detector.visualization.interactive.view_models.projected_1d import (
    Projected1DModel,
)


def _line_graph():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    return graph


def _run_bundle_like(*, with_projection: bool = True):
    centers = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [5.0, 2.0],
        ],
        dtype=np.float64,
    )
    env = SimpleNamespace(
        place_bin_centers_=centers,
        is_track_interior_=np.ones(centers.shape[0], dtype=bool),
        place_bin_size=5.0,
    )
    detector = SimpleNamespace(
        environments=[env],
        state_names=["Decoded"],
        bin_sizes_=np.array([centers.shape[0]]),
        state_ind_=np.zeros(centers.shape[0], dtype=int),
    )
    return SimpleNamespace(
        detector=detector,
        projection_track_graph=_line_graph() if with_projection else None,
        projection_edge_order=[(0, 1)] if with_projection else None,
        projection_edge_spacing=0.0,
    )


@pytest.mark.unit
def test_projected_1d_model_sums_2d_bins_by_linearized_position() -> None:
    model = Projected1DModel(_run_bundle_like())
    posterior = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    payload = WindowPayload(
        request_id=0,
        time=np.array([0.0]),
        indices=slice(0, 1),
        posterior=posterior,
        position=np.array([[2.0, 1.0]]),
    )

    projected, linear_position = model.update_window(payload)

    assert model.is_available
    np.testing.assert_allclose(model.linear_centers, [2.5, 7.5, 12.5])
    assert projected is not None
    # Two 2D bins project to the middle linear bin: 0.2 + 0.4.
    np.testing.assert_allclose(projected, [[0.1, 0.6, 0.3]])
    assert linear_position is not None
    np.testing.assert_allclose(linear_position, [2.0], atol=1e-6)


@pytest.mark.unit
def test_projected_1d_model_unavailable_without_projection_graph() -> None:
    model = Projected1DModel(_run_bundle_like(with_projection=False))
    payload = WindowPayload(
        request_id=0,
        time=np.array([0.0]),
        indices=slice(0, 1),
        posterior=np.ones((1, 4), dtype=np.float64),
        position=np.array([[2.0, 1.0]]),
    )

    projected, linear_position = model.update_window(payload)

    assert not model.is_available
    assert "projection_track_graph" in model.message
    assert projected is None
    assert linear_position is None
