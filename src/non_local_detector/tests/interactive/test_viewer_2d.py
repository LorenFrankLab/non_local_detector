"""Phase 1 smoke test for the 2D-detector viewer path.

A 2D detector fits + predicts on a tiny synthetic open-field session,
the QtViewer is constructed against the resulting bundle, and the
viewer is checked for:

- ``_posterior_at_cursor_panel`` present (Qt2DImagePanel)
- 1D heatmap panels (``_panel`` / ``_likelihood_panel``) absent
- 1D slice panel + slice overlay combo absent
- ``projected_2d`` rejected even when requested
- A cursor update doesn't crash and populates the ImageItem
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def qapp():
    """Provide a singleton QApplication for the 2D viewer GUI tests."""
    from non_local_detector.visualization.interactive.viewer.qt import (
        _ensure_qapplication,
    )

    app = _ensure_qapplication()
    yield app


@pytest.fixture(scope="module")
def fitted_2d_bundle():
    """Minimal 2D ``SortedSpikesDecoder`` fit + bundle for the viewer."""
    from non_local_detector import SortedSpikesDecoder
    from non_local_detector.environment import Environment
    from non_local_detector.visualization.interactive.view_models.base import (
        RunBundle,
    )

    rng = np.random.default_rng(0)
    sampling_frequency = 100
    n_time = 800
    time = np.arange(n_time) / sampling_frequency

    # Simple back-and-forth animal trajectory across a 2D box.
    x = 25.0 + 20.0 * np.sin(2 * np.pi * time / 4.0)
    y = 25.0 + 15.0 * np.cos(2 * np.pi * time / 3.0)
    position = np.column_stack([x, y])

    # Three cells with Gaussian place fields at distinct (x, y) centers.
    centers_xy = np.array([[15.0, 15.0], [35.0, 35.0], [25.0, 25.0]])
    place_std = 6.0
    spike_times: list[np.ndarray] = []
    for cx, cy in centers_xy:
        rate = 12.0 * np.exp(
            -((position[:, 0] - cx) ** 2 + (position[:, 1] - cy) ** 2)
            / (2 * place_std**2)
        )
        # Inhomogeneous Poisson sampling per bin.
        spikes = rng.poisson(rate / sampling_frequency)
        spike_times.append(time[spikes > 0])

    env = Environment(
        place_bin_size=5.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    )
    detector = SortedSpikesDecoder(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 4.0,
            "block_size": 4096,
        },
        environments=[env],
    )
    detector.fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
    )
    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        return_outputs="all",
    )
    bundle = RunBundle(
        results=results,
        detector=detector,
        spike_times=spike_times,
        position_time=time,
        position=position,
    )
    return bundle


def test_viewer_constructs_2d_at_cursor_panels(qapp, fitted_2d_bundle) -> None:
    """The 2D path swaps in two ``Qt2DImagePanel`` instances and drops the 1D panels."""
    from non_local_detector.visualization.interactive.data_source import (
        InMemoryDecoderDataSource,
    )
    from non_local_detector.visualization.interactive.panels.qt.image_2d import (
        Qt2DImagePanel,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    viewer = QtViewer(
        InMemoryDecoderDataSource.from_single(fitted_2d_bundle),
        t_width=0.5,
    )

    # 1D panels not built for 2D.
    assert viewer._panel is None
    assert viewer._likelihood_panel is None
    assert viewer._slice_panel is None
    assert viewer._slice_overlay_combo is None
    assert viewer._per_cell_checkbox is None
    # 2D top images: posterior + likelihood. Both present in the
    # bin-synced dispatch list so each cursor tick updates both.
    assert isinstance(viewer._posterior_at_cursor_panel, Qt2DImagePanel)
    assert isinstance(viewer._likelihood_at_cursor_panel, Qt2DImagePanel)
    assert viewer._posterior_at_cursor_panel in viewer._bin_synced_panels
    assert viewer._likelihood_at_cursor_panel in viewer._bin_synced_panels
    # Right-column layout stacks them (posterior on top, likelihood
    # below) in the same column-panel registry.
    assert viewer._right_column_panels == [
        viewer._posterior_at_cursor_panel,
        viewer._likelihood_at_cursor_panel,
    ]


def test_viewer_2d_rejects_projected_2d(qapp, fitted_2d_bundle) -> None:
    """``show_projected_2d=True`` is no-op for true 2D detectors."""
    from non_local_detector.visualization.interactive.data_source import (
        InMemoryDecoderDataSource,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    viewer = QtViewer(
        InMemoryDecoderDataSource.from_single(fitted_2d_bundle),
        t_width=0.5,
        show_projected_2d=True,
    )

    # ``show_projected_2d`` is silently disabled — projected-2D is for
    # 1D graph-linearized decoders only.
    assert viewer._projected_2d_panel is None
    assert viewer._projected_2d_model is None


def test_viewer_2d_cursor_update_renders_images(qapp, fitted_2d_bundle) -> None:
    """Driving a cursor tick populates both 2D ImageItems without crashing."""
    from non_local_detector.visualization.interactive.data_source import (
        InMemoryDecoderDataSource,
    )
    from non_local_detector.visualization.interactive.view_models.base import (
        ViewState,
    )
    from non_local_detector.visualization.interactive.viewer.qt import QtViewer

    viewer = QtViewer(
        InMemoryDecoderDataSource.from_single(fitted_2d_bundle),
        t_width=0.5,
    )

    # Synchronously build a payload covering a mid-session window and
    # push it to the bin-synced panels.
    state = ViewState(request_id=0, t_center=4.0, t_width=0.5)
    payload = viewer._backend.build_payload(state)
    # Pick a t_idx inside the buffered window and render.
    t_idx = (payload.indices.start + payload.indices.stop) // 2

    for panel in (
        viewer._posterior_at_cursor_panel,
        viewer._likelihood_at_cursor_panel,
    ):
        assert panel is not None
        panel.set_window_buffer(payload)
        panel.update_for_index(t_idx)
        # Each ImageItem should hold a populated RGBA frame.
        assert panel._image_item.image is not None
        assert panel._image_item.image.shape[-1] == 4
