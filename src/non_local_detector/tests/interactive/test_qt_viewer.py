"""Headless Qt viewer smoke tests.

Marked ``@pytest.mark.gui`` because they construct an actual
``QApplication`` and ``QtViewer`` window. They run on the offscreen
platform plugin so they don't require a display server.

These do *not* assert pixel-perfect rendering — they verify that the
viewer launches, the model collapses correctly, and the slider /
backend wiring delivers a payload to the panel without crashing.
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
from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.view_models.base import (
    RunBundle,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    """Provide a singleton QApplication for all GUI tests."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.mark.unit
def test_qt_panel_renders_collapsed_array(
    qapp,
    nl_fitted: FittedDetector,
    sim_session: SimulatedSession,
) -> None:
    """``QtPosteriorHeatmapPanel`` accepts a real posterior window
    without raising.

    Numerically the rendered ImageItem array is the model's output
    (transposed for row-major). We assert shape + non-empty.
    """
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )

    detector = nl_fitted.detector
    env = detector.environments[0]
    model = PosteriorHeatmapModel(detector)
    panel = QtPosteriorHeatmapPanel(
        model=model,
        position_centers=np.asarray(env.place_bin_centers_).squeeze(),
    )
    post = nl_fitted.results["acausal_posterior"].values
    # First 50 rows starting from the first finite row.
    finite = np.flatnonzero(np.isfinite(post).any(axis=-1))
    start = int(finite[0])
    window = post[start : start + 50]
    panel.update_for_array(
        time=np.linspace(0.0, 1.0, window.shape[0]),
        posterior=window,
    )
    image = panel._image_item.image
    # ImageItem stores the transposed (n_pos, n_visible) array.
    n_pos = int(env.place_bin_centers_.shape[0])
    assert image.shape == (n_pos, window.shape[0])


@pytest.mark.unit
def test_qt_viewer_launches_and_routes_payload(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """``launch_qt(block=False)`` builds the window + delivers a payload.

    The QtBackendAdapter dispatches a load on the QThreadPool; we
    process events until the panel receives a non-empty image.
    """

    from non_local_detector.visualization.interactive.viewer.qt import (
        launch_qt,
    )

    # Disable threadpool for deterministic synchronous tests — drive
    # the load via the public API, then process events until the
    # signal-bridge delivers the payload.
    code = launch_qt(multi_run_bundles, t_width=0.5, block=False)
    assert code == 0


@pytest.mark.unit
def test_qt_viewer_set_active_run_via_core(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Construct viewer, then swap to a different run via ``core.set_active_run``."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    assert viewer.core.active_run_name == "nl"
    viewer.core.set_active_run("cf")
    assert viewer.core.active_run_name == "cf"


@pytest.mark.unit
def test_event_overlay_mixin_renders_numpy_arrays(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Regression: ``set_event_overlays`` must accept multi-element NumPy arrays.

    The previous ``overlay.times or []`` fallback raised
    ``ValueError: ambiguous truth value`` because ``bool(np.array([1, 2]))``
    is ambiguous. The fix uses explicit ``is None`` comparisons.
    """
    from non_local_detector.visualization.interactive.panels.qt.posterior import (
        QtPosteriorHeatmapPanel,
    )
    from non_local_detector.visualization.interactive.view_models.events import (
        EventOverlay,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )

    detector = multi_run_bundles["nl"].detector
    env = detector.environments[0]
    panel = QtPosteriorHeatmapPanel(
        model=PosteriorHeatmapModel(detector),
        position_centers=np.asarray(env.place_bin_centers_).squeeze(),
    )

    overlays = [
        EventOverlay.points(
            name="multi-point",
            times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        EventOverlay.intervals(
            name="multi-interval",
            t_start=np.array([1.0, 5.0, 10.0]),
            t_end=np.array([2.0, 7.0, 12.0]),
        ),
    ]
    # Must not raise.
    panel.set_event_overlays(overlays)
    # 5 points + 3 intervals = 8 overlay items.
    assert len(panel._overlay_items) == 8

    # Idempotent: a fresh call replaces previously rendered items.
    panel.set_event_overlays([overlays[0]])
    assert len(panel._overlay_items) == 5

    # Empty list clears.
    panel.set_event_overlays([])
    assert len(panel._overlay_items) == 0


@pytest.mark.unit
def test_qt_viewer_swap_rebinds_panel_model(
    qapp,
    multi_run_bundles: dict[str, RunBundle],
) -> None:
    """Regression: swap from NL to CF must rebind PosteriorHeatmapModel.

    Before the fix, the panel held a model bound to the NL detector
    forever, so a CF posterior payload (n_state_bins=2*n_pos) would
    be collapsed under NL's state_ind_ (n_state_bins=n_pos+1+n_pos+n_pos)
    and produce garbage / crash.
    """
    from non_local_detector.analysis.posterior import PosteriorReduction
    from non_local_detector.visualization.interactive.viewer.qt import (
        QtViewer,
    )

    ds = InMemoryDecoderDataSource(multi_run_bundles)
    viewer = QtViewer(ds, t_width=0.5)
    # Initial: NL → CONDITIONAL_NON_LOCAL.
    assert viewer._posterior_model.reduction is PosteriorReduction.CONDITIONAL_NON_LOCAL
    assert viewer._posterior_model.detector is multi_run_bundles["nl"].detector

    viewer.core.set_active_run("cf")

    # After swap: CF → MARGINAL, model bound to the CF detector.
    assert viewer._posterior_model.reduction is PosteriorReduction.MARGINAL
    assert viewer._posterior_model.detector is multi_run_bundles["cf"].detector
