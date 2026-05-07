"""Tests for ``QtRasterPanel``'s non-local-state shading.

Covers the pure ``_contiguous_spans`` helper unit-style and the
panel's full render path against a real fixture (GUI-marked).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Force offscreen Qt platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
)
from non_local_detector.visualization.interactive.panels.qt.raster import (
    _contiguous_spans,
)
from non_local_detector.visualization.interactive.view_models.base import WindowPayload
from non_local_detector.visualization.interactive.view_models.raster import RasterModel


@pytest.mark.unit
def test_qt_raster_panel_class_methods_are_accessible() -> None:
    """Regression guard: methods stayed nested inside ``_contiguous_spans``."""
    from non_local_detector.visualization.interactive.panels.qt.raster import (
        QtRasterPanel,
    )

    assert callable(QtRasterPanel.update_for_window)
    assert callable(QtRasterPanel.rebind_after_swap)


@pytest.mark.unit
class TestContiguousSpans:
    def test_empty_mask(self) -> None:
        assert _contiguous_spans(np.array([], dtype=bool)) == []

    def test_all_false(self) -> None:
        assert _contiguous_spans(np.zeros(5, dtype=bool)) == []

    def test_all_true(self) -> None:
        assert _contiguous_spans(np.ones(3, dtype=bool)) == [(0, 3)]

    def test_single_span_in_middle(self) -> None:
        mask = np.array([False, True, True, False, False])
        assert _contiguous_spans(mask) == [(1, 3)]

    def test_two_spans(self) -> None:
        mask = np.array([True, False, True, True, False, True])
        assert _contiguous_spans(mask) == [(0, 1), (2, 4), (5, 6)]


@pytest.mark.gui
def test_qt_raster_panel_shades_non_local_active_bins(
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """When state probabilities mark a high-non-local span, the panel
    adds at least one ``LinearRegionItem`` to overlay it.

    Constructs a real RasterModel + QtRasterPanel against the NL
    fixture, then synthesizes a payload whose state probabilities
    place 100% mass on a non-local state for the middle of the
    window. Panel must render exactly one shaded span at that
    location.
    """
    from PySide6 import QtWidgets

    from non_local_detector.analysis.posterior import _non_local_state_ids
    from non_local_detector.visualization.interactive.panels.qt.raster import (
        QtRasterPanel,
    )

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    detector = nl_fitted.detector
    nl_ids = _non_local_state_ids(detector)
    assert nl_ids.size > 0

    model = RasterModel(detector, sim_session.spike_times)
    panel = QtRasterPanel(model, non_local_threshold=0.5)

    n_visible = 20
    n_states = len(detector.state_names)
    time = np.linspace(10.0, 11.0, n_visible)
    probs = np.zeros((n_visible, n_states))
    probs[5:15, nl_ids[0]] = 1.0  # non-local 100% in middle 10 bins
    payload = WindowPayload(
        request_id=0,
        time=time,
        indices=slice(0, n_visible),
        state_probabilities=probs,
    )
    panel.update_window(payload)
    assert len(panel._non_local_regions) == 1
    region = panel._non_local_regions[0]
    lo, hi = region.getRegion()
    assert lo == pytest.approx(time[5])
    assert hi == pytest.approx(time[14])

    # Update with state_probabilities=None — the previous shading
    # should be cleared, no new regions added.
    payload_no_probs = WindowPayload(
        request_id=1,
        time=time,
        indices=slice(0, n_visible),
        state_probabilities=None,
    )
    panel.update_window(payload_no_probs)
    assert panel._non_local_regions == []
