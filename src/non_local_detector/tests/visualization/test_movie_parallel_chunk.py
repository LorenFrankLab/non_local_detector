"""Regression test for the parallel-chunk firing-rate panel (issue #42).

``create_parallel_video`` renders contiguous frame ranges in separate worker
processes; each worker calls the setup function once and then renders its own
*global* frame indices. The bottom multiunit panel's axis limits were gated on
``frame_idx == 0``, so only the worker owning the first chunk set them — every
later chunk left ``ax1`` at default limits and drew the firing-rate line
off-screen. The fix initializes the limits on the first frame each worker
renders (mirroring the mesh pattern), so a non-zero starting frame must still
set ``ax1``'s limits.
"""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

from non_local_detector.visualization.movie import (  # noqa: E402
    _render_single_env_frame,
    _setup_single_env_figure,
)


def _make_frame_data(n_time: int = 100, sampling_frequency: float = 500.0) -> dict:
    rng = np.random.default_rng(0)
    x_position = np.linspace(0.0, 40.0, 5)
    y_position = np.linspace(0.0, 40.0, 5)
    probs = rng.random((n_time, x_position.size, y_position.size))
    probs /= probs.sum(axis=(1, 2), keepdims=True)
    posterior = xr.DataArray(
        probs,
        dims=["time", "x_position", "y_position"],
        coords={
            "time": np.arange(n_time) / sampling_frequency,
            "x_position": x_position,
            "y_position": y_position,
        },
    )
    window_ind = np.arange(-10, 11)
    return {
        "position": rng.random((n_time, 2)) * 40.0,
        "direction": rng.uniform(-np.pi, np.pi, size=n_time),
        "r": 5.0,
        "map_position": rng.random((n_time, 2)) * 40.0,
        "posterior": posterior,
        "vmax": float(probs.max()),
        "xy_limits": (0.0, 40.0, 0.0, 40.0),
        "window_ind": window_ind,
        "window_size": 20,
        "sampling_frequency": sampling_frequency,
        "rate": np.linspace(0.0, 600.0, 2 * n_time),
    }


@pytest.mark.unit
def test_bottom_axis_limits_set_on_worker_first_frame_not_global_frame_zero():
    """A worker whose chunk starts at a non-zero frame must still set ax1 limits.

    Pre-fix, the limits were only set when ``frame_idx == 0``, so this fresh
    (per-worker) figure rendered at ``frame_idx=50`` would keep ax1 at its
    default ``(0, 1)`` limits and the firing-rate line would be off-screen.
    """
    fig, axes = _setup_single_env_figure()
    try:
        data = _make_frame_data()
        sf = data["sampling_frequency"]
        window_ind = data["window_ind"]
        frame_idx = 50  # simulates the first frame of a non-first chunk

        # A freshly set-up figure (as each worker gets) has not initialized ax1.
        assert axes["mesh"] is None
        assert not axes.get("_ax1_init", False)

        _render_single_env_frame(fig, axes, frame_idx, data)

        expected_xlim = (window_ind[0] / sf, window_ind[-1] / sf)
        xlim = axes["ax1"].get_xlim()
        ylim = axes["ax1"].get_ylim()

        np.testing.assert_allclose(xlim, expected_xlim, atol=1e-12)
        np.testing.assert_allclose(ylim[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(ylim[1], data["rate"].max(), atol=1e-9)
        assert axes["_ax1_init"] is True
    finally:
        plt.close(fig)
