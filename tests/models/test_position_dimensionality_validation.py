"""Validation tests for position dimensionality against environment type.

When an Environment has a ``track_graph``, the detector expects raw 2D
(x, y) position coordinates — it linearizes internally during both
``fit()`` and ``predict()``. Passing already-linearized 1D position in
this case would silently corrupt internal coordinates, so the detector
raises a ValidationError with a clear hint.
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError


def _make_track_graph_env():
    """Build a small 1D linear-track environment with a track graph."""
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(10.0, 0.0))
    g.add_edge(0, 1, distance=10.0, edge_id=0)
    env = Environment(
        environment_name="",
        place_bin_size=1.0,
        track_graph=g,
        edge_order=[(0, 1)],
        edge_spacing=0.0,
    )
    return env.fit_place_grid()


@pytest.mark.unit
class TestPositionDimensionalityValidation:
    """Validate that 1D position with track_graph env raises a clear error."""

    def test_fit_rejects_1d_position_with_track_graph(self):
        """fit() raises ValidationError for 1D position when env has track_graph."""
        env = _make_track_graph_env()
        detector = NonLocalSortedSpikesDetector(environments=[env])

        time = np.linspace(0.0, 10.0, 100)
        pos_1d = np.linspace(0.0, 10.0, 100)  # already linearized

        with pytest.raises(ValidationError, match="track_graph"):
            detector.fit(time, pos_1d, spike_times=[np.array([1.0, 2.0])])

    def test_fit_rejects_column_vector_with_track_graph(self):
        """fit() rejects (n, 1)-shaped 1D position (common accidental form)."""
        env = _make_track_graph_env()
        detector = NonLocalSortedSpikesDetector(environments=[env])

        time = np.linspace(0.0, 10.0, 100)
        pos_col = np.linspace(0.0, 10.0, 100)[:, np.newaxis]  # shape (100, 1)

        with pytest.raises(ValidationError, match="track_graph"):
            detector.fit(time, pos_col, spike_times=[np.array([1.0, 2.0])])

    def test_fit_accepts_2d_position_with_track_graph(self):
        """fit() with 2D (x, y) position proceeds normally when env has track_graph."""
        env = _make_track_graph_env()
        detector = NonLocalSortedSpikesDetector(environments=[env])

        time = np.linspace(0.0, 10.0, 100)
        pos_2d = np.stack(
            [np.linspace(0.0, 10.0, 100), np.zeros(100)],
            axis=-1,
        )

        # Should not raise
        detector.fit(time, pos_2d, spike_times=[np.array([1.0, 2.0])])

    def test_fit_accepts_1d_position_without_track_graph(self):
        """fit() with 1D position + no track_graph warns but proceeds."""
        detector = NonLocalSortedSpikesDetector()  # default env: no track_graph

        time = np.linspace(0.0, 10.0, 100)
        pos_1d = np.linspace(0.0, 10.0, 100)

        # Should not raise (a UserWarning is emitted but that's OK)
        with pytest.warns(UserWarning, match="no track_graph"):
            detector.fit(time, pos_1d, spike_times=[np.array([1.0, 2.0])])

    def test_error_message_includes_shape_and_fix_hint(self):
        """Error message tells the user what shape they passed and how to fix it."""
        env = _make_track_graph_env()
        detector = NonLocalSortedSpikesDetector(environments=[env])

        time = np.linspace(0.0, 10.0, 100)
        pos_1d = np.linspace(0.0, 10.0, 100)

        with pytest.raises(ValidationError) as excinfo:
            detector.fit(time, pos_1d, spike_times=[np.array([1.0, 2.0])])

        msg = str(excinfo.value)
        # Should mention the actual shape
        assert "(100,)" in msg
        # Should hint at raw 2D coordinates
        assert "2D" in msg
        # Should point to the fix
        assert "get_linearized_position" in msg or "raw 2D" in msg
