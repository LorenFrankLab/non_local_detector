"""``plot_discrete_state_transition`` draws the transition matrix as a heatmap."""

import warnings
from types import SimpleNamespace

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
    detector.state_space_ = SimpleNamespace(state_names=STATE_NAMES)
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


def _expected_dwell_seconds(transitions, sampling_frequency):
    """Mean of the geometric number of bins spent in a state, in seconds.

    A state left with probability ``1 - p_ii`` per bin is occupied for
    ``1 / (1 - p_ii)`` bins on average.
    """
    return 1.0 / (1.0 - np.diag(transitions)) / sampling_frequency


@pytest.mark.parametrize(
    ("kwargs", "expected_diagonal"),
    [
        # The detector's own 500 Hz decode bins: 10, 5, 2.5 and 2.5 bins.
        ({}, [0.02, 0.01, 0.005, 0.005]),
        ({"sampling_frequency": 1000.0}, [0.01, 0.005, 0.0025, 0.0025]),
    ],
)
def test_expected_duration_heatmap_shows_dwell_seconds_on_diagonal(
    detector, kwargs, expected_diagonal
):
    assert detector.sampling_frequency == 500.0
    fig, ax = _plot(detector, convert_to_seconds=True, **kwargs)
    try:
        np.testing.assert_allclose(
            _expected_dwell_seconds(
                TRANSITIONS, kwargs.get("sampling_frequency", 500.0)
            ),
            expected_diagonal,
        )
        mesh = ax.collections[0]
        cells = np.ma.asarray(mesh.get_array()).reshape(4, 4)
        np.testing.assert_allclose(np.diag(cells), expected_diagonal)
        # A cell off the diagonal has no duration: it is left blank.
        off_diagonal = ~np.eye(4, dtype=bool)
        assert np.ma.getmaskarray(cells)[off_diagonal].all()
        assert not np.ma.getmaskarray(cells)[~off_diagonal].any()
        assert mesh.get_clim() == pytest.approx((0.0, max(expected_diagonal)))
        assert [t.get_text() for t in ax.texts] == [
            f"{v:.3f}" for v in expected_diagonal
        ]
        assert [t.get_position() for t in ax.texts] == [
            (i + 0.5, i + 0.5) for i in range(4)
        ]
        assert fig.axes[1].get_ylabel() == "Expected dwell time (s)"
    finally:
        plt.close(fig)


def test_expected_duration_uses_detector_sampling_frequency(detector):
    detector.sampling_frequency = 250.0
    fig, ax = _plot(detector, convert_to_seconds=True)
    try:
        cells = np.ma.asarray(ax.collections[0].get_array()).reshape(4, 4)
        np.testing.assert_allclose(np.diag(cells), [0.04, 0.02, 0.01, 0.01])
    finally:
        plt.close(fig)


def test_absorbing_state_has_infinite_dwell(detector):
    detector.state_space_ = SimpleNamespace(state_names=["Stay", "Move"])
    detector.discrete_state_transitions_ = np.array([[1.0, 0.0], [0.2, 0.8]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, ax = _plot(detector, convert_to_seconds=True)
    try:
        mesh = ax.collections[0]
        cells = np.ma.getdata(mesh.get_array()).reshape(2, 2)
        assert np.isposinf(cells[0, 0])
        np.testing.assert_allclose(cells[1, 1], 5 / 500)
        # The finite dwell sets the color scale; the infinite one is drawn at
        # its top.
        assert mesh.get_clim() == pytest.approx((0.0, 5 / 500))
        np.testing.assert_allclose(
            mesh.to_rgba(cells[0, 0]), mesh.cmap(mesh.cmap.N - 1)
        )
        assert [t.get_text() for t in ax.texts] == ["inf", "0.010"]
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
