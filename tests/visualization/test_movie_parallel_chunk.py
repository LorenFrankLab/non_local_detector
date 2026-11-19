"""Movie frames drawn by parallel workers, and the movie's frame data.

``create_parallel_video`` renders contiguous frame ranges in separate worker
processes; each worker calls the setup function once and then renders its own
*global* frame indices. The posterior mesh and the firing-rate panel's axis
limits are therefore set up on the first frame each worker renders, which is
not frame 0 for any chunk but the first.
"""

import matplotlib as mpl
import numpy as np
import pytest
import xarray as xr

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt

from non_local_detector.visualization.movie import (
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
        "sampling_frequency": sampling_frequency,
        "rate": np.linspace(0.0, 600.0, 2 * n_time),
        "rate_center": np.arange(n_time) + 10,
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
def test_worker_first_frame_draws_its_own_posterior():
    """The mesh a worker builds on its first frame shows that frame's posterior.

    Every chunk after the first starts on a fresh figure at a frame other than
    0, so a mesh built from frame 0 would show the wrong posterior there.
    """
    fig, axes = _setup_single_env_figure()
    try:
        data = _make_frame_data()
        # Unequal grid sides, so a transposed mesh would not match either.
        probs = np.random.default_rng(1).random((data["posterior"].sizes["time"], 5, 7))
        data["posterior"] = xr.DataArray(
            probs,
            dims=["time", "x_position", "y_position"],
            coords={
                "time": data["posterior"].time,
                "x_position": np.linspace(0.0, 40.0, 5),
                "y_position": np.linspace(0.0, 40.0, 7),
            },
        )
        frame_idx = 50

        _render_single_env_frame(fig, axes, frame_idx, data)

        # pcolormesh(x="x_position", y="y_position") draws the (y, x) array.
        np.testing.assert_array_equal(
            np.ma.getdata(axes["mesh"].get_array()).reshape(7, 5),
            probs[frame_idx].T,
        )
        assert axes["title"].get_text() == f"time = {frame_idx / 500.0:0.2f}"

        _render_single_env_frame(fig, axes, frame_idx + 1, data)
        np.testing.assert_array_equal(
            np.ma.getdata(axes["mesh"].get_array()).reshape(7, 5),
            probs[frame_idx + 1].T,
        )
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


def _movie_frame_data(monkeypatch, n_results, n_position, time_slice):
    """The frame data ``make_single_environment_movie`` hands the renderer.

    The multiunit rate is replaced by the row number, so the rate drawn on a
    frame names the rows it was taken from.
    """
    from types import SimpleNamespace

    import pandas as pd

    from non_local_detector.visualization import movie

    rows = [("Local", np.nan, np.nan)] + [
        ("NL", x, y) for x in np.arange(3.0) for y in np.arange(2.0)
    ]
    index = pd.MultiIndex.from_tuples(rows, names=["state", "x_position", "y_position"])
    values = np.random.default_rng(0).random((n_results, len(rows)))
    results = xr.Dataset(
        {
            "acausal_posterior": xr.DataArray(
                values / values.sum(axis=1, keepdims=True),
                dims=["time", "state_bins"],
                coords={"time": np.arange(n_results) / 500.0, "state_bins": index},
            )
        }
    )
    classifier = SimpleNamespace(
        environments_=[SimpleNamespace(is_track_interior_=np.ones((3, 2), dtype=bool))]
    )
    position_time = np.arange(n_position) / 500.0
    position_info = pd.DataFrame(
        {
            "head_position_x": np.arange(n_position, dtype=float),
            "head_position_y": np.zeros(n_position),
            "head_orientation": np.zeros(n_position),
        },
        index=pd.Index(position_time, name="time"),
    )
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs["frame_data"])
        return kwargs["output_path"]

    monkeypatch.setattr(movie, "create_parallel_video", capture)
    monkeypatch.setattr(
        movie,
        "get_multiunit_firing_rate",
        lambda spike_times, time: np.arange(len(time), dtype=float)[:, np.newaxis],
    )
    movie.make_single_environment_movie(
        time_slice, classifier, results, position_info, [np.array([0.001])]
    )
    return captured


@pytest.mark.unit
@pytest.mark.parametrize(
    ("time_slice", "frame_idx", "row", "first_row", "last_row"),
    [
        # Away from the edges: the whole 501-row window.
        (slice(300, 800), 0, 300, 300, 799),
        # Starting at row 0 or inside the first 250 rows: the window is cut
        # at row 0 and stays centred on the frame's row.
        (slice(0, 500), 0, 0, 0, 499),
        (slice(0, 500), 10, 10, 0, 499),
        (slice(100, 600), 0, 100, 100, 599),
        (slice(None, 500), 3, 3, 0, 499),
        # Running to the end: the window is cut at the last row.
        (slice(700, None), 499, 1199, 700, 1199),
    ],
)
def test_movie_rate_window_is_centred_on_frame_and_cut_at_recording_edges(
    monkeypatch, time_slice, frame_idx, row, first_row, last_row
):
    from non_local_detector.visualization.movie import (
        _render_single_env_frame,
        _setup_single_env_figure,
    )

    n_rows = 1200
    data = _movie_frame_data(monkeypatch, n_rows, n_rows, time_slice)
    np.testing.assert_array_equal(
        data["position"][:, 0][[0, -1]], [first_row, last_row]
    )

    fig, axes = _setup_single_env_figure()
    try:
        _render_single_env_frame(fig, axes, frame_idx, data)

        offsets = np.arange(-250, 251)
        is_inside = (row + offsets >= 0) & (row + offsets < n_rows)
        line = axes["multiunit_line"]
        np.testing.assert_allclose(line.get_xdata(), offsets[is_inside] / 500.0)
        np.testing.assert_array_equal(line.get_ydata(), row + offsets[is_inside])
        assert axes["ax1"].get_xlim() == (-0.5, 0.5)
        # The y limit is the largest rate in the windows of all frames.
        assert axes["ax1"].get_ylim() == (0.0, min(last_row + 250, n_rows - 1))
    finally:
        plt.close(fig)


@pytest.mark.unit
def test_movie_refuses_position_rows_that_differ_from_result_times(monkeypatch):
    from non_local_detector.exceptions import ValidationError

    with pytest.raises(ValidationError, match="one position_info row per result"):
        _movie_frame_data(monkeypatch, 500, 1000, slice(250, 500))


@pytest.mark.unit
@pytest.mark.parametrize(
    "time_slice",
    [slice(250, 250), slice(600, 700), slice(300, 200)],
    ids=["empty", "past the end", "stop before start"],
)
def test_movie_refuses_a_time_slice_that_selects_no_rows(monkeypatch, time_slice):
    from non_local_detector.exceptions import ValidationError

    with pytest.raises(ValidationError, match="time_slice selects no rows"):
        _movie_frame_data(monkeypatch, 500, 500, time_slice)


def test_movie_posterior_has_no_false_peak_at_last_bin():
    """Local's mass must not be placed on the last x/y bin of the movie posterior."""
    import pandas as pd
    import xarray as xr

    from non_local_detector.visualization.movie import _spatial_posterior_for_movie

    rows = [("Local", np.nan, np.nan)] + [
        ("NL", x, y) for x in np.arange(3.0) for y in np.arange(2.0)
    ]
    index = pd.MultiIndex.from_tuples(rows, names=["state", "x_position", "y_position"])
    values = np.full((2, len(rows)), 0.02)
    values[:, 0] = 0.9
    results = xr.Dataset(
        {
            "acausal_posterior": xr.DataArray(
                values,
                dims=["time", "state_bins"],
                coords={"time": [0.0, 0.1], "state_bins": index},
            )
        }
    )
    interior = np.ones((3, 2), dtype=bool)

    posterior = _spatial_posterior_for_movie(results, slice(0, 2), interior)

    np.testing.assert_allclose(posterior.values, 0.02)


def test_movie_map_position_is_nan_without_spatial_mass(monkeypatch):
    """A frame whose mass is all in position-less states has no MAP dot.

    The argmax of an all-zero frame would otherwise place a confident MAP at
    the first grid bin.
    """
    from types import SimpleNamespace

    import pandas as pd

    from non_local_detector.visualization import movie

    rows = [("Local", np.nan, np.nan)] + [
        ("NL", x, y) for x in np.arange(3.0) for y in np.arange(2.0)
    ]
    index = pd.MultiIndex.from_tuples(rows, names=["state", "x_position", "y_position"])
    values = np.zeros((2, len(rows)))
    values[0, 0] = 0.5
    values[0, 6] = 0.5  # NL peak at (x=2, y=1)
    values[1, 0] = 1.0  # all mass in Local
    time = np.array([0.0, 0.002])
    results = xr.Dataset(
        {
            "acausal_posterior": xr.DataArray(
                values,
                dims=["time", "state_bins"],
                coords={"time": time, "state_bins": index},
            )
        }
    )
    classifier = SimpleNamespace(
        environments_=[SimpleNamespace(is_track_interior_=np.ones((3, 2), dtype=bool))]
    )
    position_info = pd.DataFrame(
        {
            "head_position_x": [0.0, 1.0],
            "head_position_y": [0.0, 1.0],
            "head_orientation": [0.0, 0.0],
        },
        index=pd.Index(time, name="time"),
    )
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs["frame_data"])
        return kwargs["output_path"]

    monkeypatch.setattr(movie, "create_parallel_video", capture)

    movie.make_single_environment_movie(
        slice(0, 2), classifier, results, position_info, [np.array([0.001])]
    )

    np.testing.assert_array_equal(
        captured["map_position"], [[2.0, 1.0], [np.nan, np.nan]]
    )
