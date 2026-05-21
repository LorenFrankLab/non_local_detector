"""Tests for ``figurl_2D.process_decoded_data`` precondition checks.

``figurl_2D`` stores linearized 2D position bin indices as ``uint16`` to match
the downstream sortingview ``DecodedPositionData`` schema. Grids larger than
``np.iinfo(np.uint16).max`` (65535) silently wrapped, mapping multiple
true bins onto the same display bin. The guard tested here raises a
``ValueError`` at the boundary instead of producing corrupted visualizations.
"""

import numpy as np
import pytest
import xarray as xr

# figurl_2D wraps its body in `try: ... except ImportError`, so the
# names it defines only exist when sortingview is installed.
pytest.importorskip("sortingview")

from non_local_detector.visualization.figurl_2D import (  # noqa: E402
    process_decoded_data,
)


def _make_posterior(x_count: int, y_count: int, n_time: int = 1) -> xr.DataArray:
    """Build a uniform-density 3D posterior on an ``x_count`` x ``y_count`` grid."""
    x_positions = np.arange(x_count, dtype=float)
    y_positions = np.arange(y_count, dtype=float)
    probs = np.full(
        (n_time, x_count, y_count),
        1.0 / (x_count * y_count),
        dtype=float,
    )
    return xr.DataArray(
        probs,
        dims=["time", "x_position", "y_position"],
        coords={
            "time": np.arange(n_time, dtype=float),
            "x_position": x_positions,
            "y_position": y_positions,
        },
    )


def test_grid_above_uint16_max_raises():
    """Grids with > 65535 bins must fail loudly, not wrap silently."""
    x_count, y_count = 300, 300  # 90 000 > 65 535
    posterior = _make_posterior(x_count, y_count, n_time=1)

    with pytest.raises(ValueError) as excinfo:
        process_decoded_data(posterior)

    msg = str(excinfo.value)
    assert "65535" in msg
    assert "sortingview" in msg


def test_grid_at_uint16_boundary_passes():
    """Exactly ``n_position_bins == 65535`` must not trigger the guard."""
    # 255 * 257 = 65 535 — the maximum representable uint16 value.
    x_count, y_count = 255, 257
    assert x_count * y_count == np.iinfo(np.uint16).max
    posterior = _make_posterior(x_count, y_count, n_time=1)

    # The function may raise downstream errors (e.g. due to memory), but it
    # must NOT raise the uint16 overflow ValueError defined in the guard.
    try:
        process_decoded_data(posterior)
    except ValueError as exc:
        assert "65535" not in str(exc), (
            "Grids exactly at the uint16 boundary must not trip the overflow guard"
        )
