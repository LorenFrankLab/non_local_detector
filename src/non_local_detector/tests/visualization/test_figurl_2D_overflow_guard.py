"""Tests for ``figurl_2D``'s uint16 grid-capacity precondition check.

``figurl_2D`` stores linearized 2D position bin indices as ``uint16`` to match
the downstream sortingview ``DecodedPositionData`` schema. The linearization
maps an ``x_count`` x ``y_count`` grid onto indices ``0 .. x_count*y_count - 1``,
so the *largest index that must be stored* is ``n_position_bins - 1`` — not
``n_position_bins``. A grid larger than that silently wrapped, mapping multiple
true bins onto the same display bin. ``validate_grid_within_uint16`` raises a
``ValueError`` at the boundary instead of producing corrupted visualizations.

These unit tests exercise the guard directly so they run without ``sortingview``
(the rest of ``figurl_2D`` is import-guarded behind it). The boundary is checked
against the largest *index* (``n_position_bins - 1`` vs ``uint16`` max), not the
bin count. The wiring test loads ``figurl_2D`` against a stub ``sortingview`` so
the rendering path is exercised without the optional dependency.
"""

import importlib.util
import re
import sys
import types

import numpy as np
import pytest

# validate_grid_within_uint16 lives outside the sortingview try-block, so it is
# importable even when sortingview is not installed.
from non_local_detector.visualization.figurl_2D import (
    _UINT16_MAX,
    validate_grid_within_uint16,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("x_count", "y_count"),
    [
        (255, 257),  # 65535 bins, max index 65534
        (256, 256),  # 65536 bins, max index 65535 — fits exactly (regression case)
        (1, _UINT16_MAX + 1),  # 65536 bins, max index 65535
    ],
)
def test_grid_at_or_below_uint16_index_capacity_passes(x_count, y_count):
    """Grids whose largest linear index fits in uint16 must NOT trip the guard.

    The 256x256 case has 65536 bins (one more than uint16 max) but a maximum
    index of 65535, which fits; a guard on the bin count would reject it.
    """
    assert validate_grid_within_uint16(x_count, y_count) == x_count * y_count


@pytest.mark.parametrize(
    ("x_count", "y_count"),
    [
        (1, _UINT16_MAX + 2),  # 65537 bins, max index 65536 — first true overflow
        (256, 257),  # 65792 bins, max index 65791
        (300, 300),  # 90000 bins
    ],
)
def test_grid_above_uint16_index_capacity_raises(x_count, y_count):
    """Grids whose largest linear index exceeds uint16 must fail loudly."""
    with pytest.raises(ValueError) as excinfo:
        validate_grid_within_uint16(x_count, y_count)

    msg = str(excinfo.value)
    assert str(_UINT16_MAX) in msg
    assert "sortingview" in msg
    # The reported bin count should match the offending grid.
    assert str(x_count * y_count) in msg


def test_error_message_explains_index_vs_count():
    """The error message should report both the bin count and the offending index."""
    with pytest.raises(ValueError) as excinfo:
        validate_grid_within_uint16(300, 300)
    msg = str(excinfo.value)
    assert "90000" in msg  # bin count
    assert re.search(r"index\D+89999", msg)  # max index = 90000 - 1


def _load_figurl_2d_with_stub_sortingview(monkeypatch):
    """Execute a private copy of ``figurl_2D`` against a stub ``sortingview``.

    The copy is not registered in ``sys.modules``, so the real module (whose
    sortingview-dependent functions are absent without the dependency) is left
    untouched for other tests.
    """
    sortingview = types.ModuleType("sortingview")
    views = types.ModuleType("sortingview.views")
    franklab = types.ModuleType("sortingview.views.franklab")
    # Referenced in return annotations, which are evaluated at definition time.
    franklab.TrackPositionAnimationV1 = type("TrackPositionAnimationV1", (), {})
    sortingview.views = views
    views.franklab = franklab
    monkeypatch.setitem(sys.modules, "sortingview", sortingview)
    monkeypatch.setitem(sys.modules, "sortingview.views", views)
    monkeypatch.setitem(sys.modules, "sortingview.views.franklab", franklab)

    spec = importlib.util.find_spec("non_local_detector.visualization.figurl_2D")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_decoded_data_invokes_guard(monkeypatch):
    """``process_decoded_data`` rejects a grid whose indices overflow uint16."""
    import xarray as xr

    figurl_2D = _load_figurl_2d_with_stub_sortingview(monkeypatch)

    x_count, y_count = 300, 300
    probs = np.full((1, x_count, y_count), 1.0 / (x_count * y_count), dtype=float)
    posterior = xr.DataArray(
        probs,
        dims=["time", "x_position", "y_position"],
        coords={
            "time": np.arange(1, dtype=float),
            "x_position": np.arange(x_count, dtype=float),
            "y_position": np.arange(y_count, dtype=float),
        },
    )
    with pytest.raises(ValueError, match="sortingview"):
        figurl_2D.process_decoded_data(posterior)
