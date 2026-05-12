"""Construction helpers for the viewer's dimension-specific panel stacks.

``QtViewer.__init__`` previously inlined three separate panel
construction blocks: 1D (posterior + likelihood heatmaps + slice
panel), 2D (posterior + likelihood at-cursor images + cell-grid),
and the optional projected-2D third column. Extracting them here
keeps the viewer constructor focused on assembly + interaction
wiring; future panel-set variations (e.g. a marginals overlay,
a continuous-state-only path) get a sibling builder rather than
another conditional branch inside ``QtViewer.__init__``.

Each builder returns a typed bundle of the panels it constructed.
The viewer plucks individual panels off the bundle into its own
attribute slots so downstream methods (model swap, layout
insertion, click wiring) continue to reference ``self._panel`` /
``self._slice_panel`` / etc. unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from non_local_detector.visualization.interactive.panels.qt.cell_grid_2d import (
    Qt2DCellGridPanel,
)
from non_local_detector.visualization.interactive.panels.qt.image_2d import (
    Qt2DImagePanel,
)
from non_local_detector.visualization.interactive.panels.qt.likelihood import (
    QtLikelihoodHeatmapPanel,
)
from non_local_detector.visualization.interactive.panels.qt.posterior import (
    QtPosteriorHeatmapPanel,
)
from non_local_detector.visualization.interactive.panels.qt.projected_2d import (
    QtProjected2DPanel,
)
from non_local_detector.visualization.interactive.panels.qt.slice import (
    QtSlicePanel,
)
from non_local_detector.visualization.interactive.view_models.projected_2d import (
    Projected2DModel,
)
from non_local_detector.visualization.interactive.viewer.cursor_row_service import (
    StateBinsField,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.view_models.base import (
        PositionGrid,
    )
    from non_local_detector.visualization.interactive.view_models.likelihood import (
        LikelihoodHeatmapModel,
    )
    from non_local_detector.visualization.interactive.view_models.posterior import (
        PosteriorHeatmapModel,
    )
    from non_local_detector.visualization.interactive.view_models.slice import (
        SliceModel,
    )


@dataclass(frozen=True)
class OneDPanels:
    """Right- and left-column panels for 1D decoders."""

    posterior: QtPosteriorHeatmapPanel
    likelihood: QtLikelihoodHeatmapPanel
    slice: QtSlicePanel


@dataclass(frozen=True)
class TwoDPanels:
    """At-cursor panels for 2D decoders."""

    posterior_at_cursor: Qt2DImagePanel
    likelihood_at_cursor: Qt2DImagePanel
    cell_grid: Qt2DCellGridPanel


# Row-provider signature: takes a decoder time index, returns the
# tuple the panel's ``set_row_provider`` consumer expects. The exact
# shape is per-panel (see ``CursorRowService`` callers in qt.py).
SliceRowProvider = Callable[[int], tuple | None]
ImageRowProvider = Callable[
    [int, StateBinsField], tuple[np.ndarray | None, np.ndarray | None]
]
ProjectedRowProvider = Callable[[int], tuple[np.ndarray | None, np.ndarray | None]]


def build_1d_panels(
    *,
    posterior_model: PosteriorHeatmapModel,
    likelihood_model: LikelihoodHeatmapModel,
    slice_model: SliceModel,
    grid: PositionGrid,
    slice_row_provider: SliceRowProvider,
) -> OneDPanels:
    """Build the 1D-decoder panel set.

    Top-to-bottom in the left column: posterior heatmap + likelihood
    heatmap. Right column: slice panel with its single-row fallback
    provider already attached so out-of-buffer cursor ticks don't
    freeze on the last in-buffer frame.
    """
    posterior_panel = QtPosteriorHeatmapPanel(
        model=posterior_model, position_centers=grid.centers
    )
    likelihood_panel = QtLikelihoodHeatmapPanel(
        model=likelihood_model, position_centers=grid.centers
    )
    slice_panel = QtSlicePanel(model=slice_model, position_centers=grid.centers)
    slice_panel.set_row_provider(slice_row_provider)
    return OneDPanels(
        posterior=posterior_panel,
        likelihood=likelihood_panel,
        slice=slice_panel,
    )


def build_2d_panels(
    *,
    posterior_model: PosteriorHeatmapModel,
    likelihood_model: LikelihoodHeatmapModel,
    slice_model: SliceModel,
    grid: PositionGrid,
    image_row_provider: ImageRowProvider,
) -> TwoDPanels:
    """Build the 2D-decoder at-cursor panel set.

    Right column top-to-bottom: posterior + likelihood ``(x, y)``
    images at the cursor bin, then a stack of per-cell place-field
    thumbnails. ``vmax=None`` (default) on the image panels →
    per-frame peak-normalize so the cursor row's brightest bin
    always lands at the LUT top regardless of absolute magnitude.
    """
    posterior_at_cursor = Qt2DImagePanel(
        model=posterior_model,
        grid=grid,
        payload_field="posterior",
        title="Posterior at cursor",
    )
    posterior_at_cursor.set_row_provider(
        lambda t_idx: image_row_provider(t_idx, "posterior")
    )
    likelihood_at_cursor = Qt2DImagePanel(
        model=likelihood_model,
        grid=grid,
        payload_field="likelihood",
        title="Likelihood at cursor",
    )
    likelihood_at_cursor.set_row_provider(
        lambda t_idx: image_row_provider(t_idx, "likelihood")
    )
    cell_grid = Qt2DCellGridPanel(model=slice_model, grid=grid)
    return TwoDPanels(
        posterior_at_cursor=posterior_at_cursor,
        likelihood_at_cursor=likelihood_at_cursor,
        cell_grid=cell_grid,
    )


def build_projected_2d_panel(
    detector: _DetectorBase,
    *,
    row_provider: ProjectedRowProvider,
) -> tuple[Projected2DModel, QtProjected2DPanel]:
    """Build the optional projected-2D column for 1D graph-linearized decoders.

    Returns the model + panel pair. Callers gate the call on
    ``grid.ndim == 1`` and a ``show_projected_2d`` flag — both
    must hold for the projection to be meaningful.
    """
    model = Projected2DModel(detector)
    panel = QtProjected2DPanel(model)
    panel.set_row_provider(row_provider)
    return model, panel
