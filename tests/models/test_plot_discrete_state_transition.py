"""``plot_discrete_state_transition`` draws the transition matrix as a heatmap."""

import matplotlib as mpl
import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector

mpl.use("Agg")
import matplotlib.pyplot as plt

pytestmark = pytest.mark.unit

STATE_NAMES = ["Here", "Quiet", "Replay-Smooth", "Replay-Jump"]
TRANSITIONS = np.array(
    [
        [0.9, 0.05, 0.03, 0.02],
        [0.1, 0.8, 0.05, 0.05],
        [0.2, 0.1, 0.6, 0.1],
        [0.3, 0.05, 0.05, 0.6],
    ]
)


@pytest.fixture
def detector():
    detector = NonLocalSortedSpikesDetector(state_names=STATE_NAMES)
    detector.discrete_state_transitions_ = TRANSITIONS
    return detector


def _plot(detector, **kwargs):
    fig, ax = plt.subplots()
    detector.plot_discrete_state_transition(ax=ax, **kwargs)
    return fig, ax


def test_probability_heatmap_content(detector):
    fig, ax = _plot(detector)
    try:
        mesh = ax.collections[0]
        np.testing.assert_allclose(mesh.get_array().reshape(4, 4), TRANSITIONS)
        assert mesh.get_clim() == (0.0, 1.0)
        assert mesh.get_cmap().name == "Oranges"
        assert [t.get_text() for t in ax.get_xticklabels()] == STATE_NAMES
        assert [t.get_text() for t in ax.get_yticklabels()] == STATE_NAMES
        np.testing.assert_allclose(ax.get_xticks(), np.arange(4) + 0.5)
        np.testing.assert_allclose(ax.get_yticks(), np.arange(4) + 0.5)
        # Row 0 is at the top.
        assert ax.get_ylim() == (4.0, 0.0)
        assert [t.get_text() for t in ax.texts] == [
            f"{v:.3f}" for v in TRANSITIONS.ravel()
        ]
        # Annotation at (row, col) sits at the center of that cell.
        assert ax.texts[1].get_position() == (1.5, 0.5)
        assert ax.texts[4].get_position() == (0.5, 1.5)
        assert fig.axes[1].get_ylabel() == "Probability"
        assert ax.get_xlabel() == "Current State"
        assert ax.get_ylabel() == "Previous State"
        assert ax.get_title() == "Discrete State Transition"
    finally:
        plt.close(fig)


def test_expected_duration_heatmap_content(detector):
    fig, ax = _plot(detector, convert_to_seconds=True, sampling_frequency=500)
    try:
        expected = (1 / (1 - TRANSITIONS)) / 500
        mesh = ax.collections[0]
        np.testing.assert_allclose(mesh.get_array().reshape(4, 4), expected)
        vmin, vmax = mesh.get_clim()
        assert vmin == 0.0
        assert vmax == pytest.approx(expected.max())
        assert [t.get_text() for t in ax.texts] == [
            f"{v:.3f}" for v in expected.ravel()
        ]
        assert fig.axes[1].get_ylabel() == "Seconds"
    finally:
        plt.close(fig)


def test_annotation_color_contrasts_with_cell(detector):
    fig, ax = _plot(detector)
    try:
        # 0.9 is a dark cell (white text); 0.05 is a light cell (dark text).
        assert ax.texts[0].get_color() == "w"
        assert ax.texts[1].get_color() == ".15"
    finally:
        plt.close(fig)
