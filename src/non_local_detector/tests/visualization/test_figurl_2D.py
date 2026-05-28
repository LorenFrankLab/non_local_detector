"""Happy-path smoke tests for ``visualization.figurl_2D``.

``figurl_2D`` wraps its body in ``try: ... except ImportError``, so its
public names only exist when ``sortingview`` is installed; otherwise these
tests skip via ``importorskip``. The uint16 overflow guard is covered
separately in ``test_figurl_2D_overflow_guard.py``; this module covers the
normal construction paths.
"""

import numpy as np
import pytest
import xarray as xr

# figurl_2D defines its functions only when sortingview imports successfully.
pytest.importorskip("sortingview")

from non_local_detector.visualization.figurl_2D import (  # noqa: E402
    create_static_track_animation,
    get_ul_corners,
    make_track,
    process_decoded_data,
)


def _make_2d_posterior(x_count: int = 4, y_count: int = 4, n_time: int = 6):
    """Build a normalized 2D posterior on an ``x_count`` x ``y_count`` grid."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(x_count * y_count), size=n_time).reshape(
        n_time, x_count, y_count
    )
    return xr.DataArray(
        probs,
        dims=["time", "x_position", "y_position"],
        coords={
            "time": np.arange(n_time, dtype=float) / 100.0,
            "x_position": np.arange(x_count, dtype=float),
            "y_position": np.arange(y_count, dtype=float),
        },
    )


def test_get_ul_corners_shape_and_offset():
    """Upper-left corners are returned transposed as (2, n_centers)."""
    centers = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0]])
    corners = get_ul_corners(width=4.0, height=4.0, centers=centers)

    assert corners.shape == (2, len(centers))
    # x is shifted left by half-width; y up by half-height.
    np.testing.assert_allclose(corners[0], centers[:, 0] - 2.0)
    np.testing.assert_allclose(corners[1], centers[:, 1] + 2.0)


def test_make_track_returns_positive_bin_dimensions():
    """``make_track`` returns positive bin width/height and corner array."""
    rng = np.random.default_rng(0)
    position = rng.uniform(0.0, 20.0, size=(200, 2))

    bin_width, bin_height, upper_left_points = make_track(position, bin_size=5.0)

    assert bin_width > 0
    assert bin_height > 0
    assert upper_left_points.shape[0] == 2


def test_create_static_track_animation_has_expected_keys():
    """The returned dict carries the documented TrackAnimation schema keys."""
    n_time = 5
    ul_corners = np.array([[0.0, 5.0, 0.0, 5.0], [0.0, 0.0, 5.0, 5.0]])
    timestamps = np.arange(n_time, dtype=float) / 100.0
    positions = np.stack(
        [np.linspace(0, 10, n_time), np.linspace(0, 10, n_time)], axis=0
    )

    data = create_static_track_animation(
        track_rect_width=5.0,
        track_rect_height=5.0,
        ul_corners=ul_corners,
        timestamps=timestamps,
        positions=positions,
    )

    assert isinstance(data, dict)
    expected_keys = {
        "type",
        "trackBinWidth",
        "trackBinHeight",
        "trackBinULCorners",
        "totalRecordingFrameLength",
        "timestamps",
        "positions",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
    }
    assert expected_keys <= set(data)
    assert data["type"] == "TrackAnimation"
    assert data["totalRecordingFrameLength"] == n_time


def test_process_decoded_data_has_expected_keys():
    """``process_decoded_data`` returns a DecodedPositionData dict."""
    posterior = _make_2d_posterior()

    data = process_decoded_data(posterior)

    assert isinstance(data, dict)
    expected_keys = {
        "type",
        "xmin",
        "binWidth",
        "xcount",
        "ymin",
        "binHeight",
        "ycount",
        "uniqueLocations",
        "values",
        "locations",
        "frameBounds",
    }
    assert expected_keys <= set(data)
    assert data["type"] == "DecodedPositionData"
    assert len(data["frameBounds"]) == posterior.sizes["time"]
