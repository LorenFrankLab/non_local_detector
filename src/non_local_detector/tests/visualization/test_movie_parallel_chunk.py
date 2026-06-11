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


@pytest.mark.unit
def test_ax1_limits_initialized_even_when_rate_update_raises():
    """A boundary chunk whose first frame overruns the rate array still sets limits.

    The firing-rate ``set_data`` indexes ``rate[frame_idx + window/2 + window_ind]``,
    which can run past the rate array near the recording boundary and raise
    ``IndexError`` (swallowed on purpose). The axis-limit init must happen
    *before* that indexing so a worker whose opening frame lands in the edge
    region does not keep default ``ax1`` limits — otherwise the off-screen-rate
    bug reappears for boundary chunks.
    """
    fig, axes = _setup_single_env_figure()
    try:
        n_time = 40
        data = _make_frame_data(n_time=n_time)
        # Shorten the rate array so the window indexing overruns it.
        data["rate"] = np.linspace(0.0, 600.0, n_time)
        sf = data["sampling_frequency"]
        window_ind = data["window_ind"]
        frame_idx = n_time - 5  # near the end: rate[frame_idx+10+10] is OOB

        assert not axes.get("_ax1_init", False)

        # Must not raise (IndexError is caught) and must still set the limits.
        _render_single_env_frame(fig, axes, frame_idx, data)

        expected_xlim = (window_ind[0] / sf, window_ind[-1] / sf)
        np.testing.assert_allclose(axes["ax1"].get_xlim(), expected_xlim, atol=1e-12)
        np.testing.assert_allclose(
            axes["ax1"].get_ylim()[1], data["rate"].max(), atol=1e-9
        )
        assert axes["_ax1_init"] is True
    finally:
        plt.close(fig)


@pytest.mark.unit
def test_ax1_limits_set_once_per_worker():
    """Limits are initialized once; later frames in the same worker do not reset them.

    The fix replaces ``frame_idx == 0`` with a per-worker ``_ax1_init`` flag.
    The flag must latch: a second frame must leave the (possibly user-adjusted)
    limits untouched, mirroring the mesh's lazy one-time init.
    """
    fig, axes = _setup_single_env_figure()
    try:
        data = _make_frame_data()
        _render_single_env_frame(fig, axes, 10, data)
        assert axes["_ax1_init"] is True

        # Perturb the limits, then render another frame in the same worker.
        sentinel = (123.0, 456.0)
        axes["ax1"].set_ylim(sentinel)
        _render_single_env_frame(fig, axes, 11, data)

        # The second render must not have re-initialized (overwritten) the limits.
        np.testing.assert_allclose(axes["ax1"].get_ylim(), sentinel, atol=1e-9)
    finally:
        plt.close(fig)
