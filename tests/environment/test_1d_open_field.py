"""One-dimensional open-field ``Environment``: fitting from ``(n,)`` position and plotting."""

import matplotlib as mpl
import numpy as np
import pytest

from non_local_detector.environment import Environment

mpl.use("Agg")
import matplotlib.pyplot as plt

pytestmark = pytest.mark.unit


def test_fit_place_grid_accepts_1d_position_and_matches_column_position():
    position = np.linspace(0, 100, 1000)
    from_vector = Environment(place_bin_size=5.0).fit_place_grid(position)
    from_column = Environment(place_bin_size=5.0).fit_place_grid(position[:, None])

    np.testing.assert_array_equal(from_vector.edges_[0], from_column.edges_[0])
    np.testing.assert_array_equal(
        from_vector.place_bin_centers_, from_column.place_bin_centers_
    )
    np.testing.assert_array_equal(
        from_vector.is_track_interior_, from_column.is_track_interior_
    )
    assert from_vector.centers_shape_ == from_column.centers_shape_ == (20,)
    assert from_vector.is_track_interior_.all()


def test_fit_place_grid_1d_position_with_nan_matches_column_position():
    position = np.linspace(0, 100, 1000)
    position[::10] = np.nan
    from_vector = Environment(place_bin_size=5.0).fit_place_grid(position)
    from_column = Environment(place_bin_size=5.0).fit_place_grid(position[:, None])
    np.testing.assert_array_equal(
        from_vector.is_track_interior_, from_column.is_track_interior_
    )


def test_plot_grid_1d_open_field_draws_edges_and_interior():
    position = np.linspace(0, 100, 1000)[:, None]
    env = Environment(place_bin_size=5.0).fit_place_grid(position)
    fig, ax_in = plt.subplots()
    try:
        ax = env.plot_grid(ax=ax_in)

        assert ax is ax_in
        mesh, edge_lines = ax.collections
        segments = np.asarray(edge_lines.get_segments())
        np.testing.assert_allclose(segments[:, 0, 0], env.edges_[0])
        np.testing.assert_array_equal(
            np.asarray(mesh.get_array()).ravel(),
            env.is_track_interior_.astype(float),
        )
        assert ax.get_xlim() == (env.edges_[0][0], env.edges_[0][-1])
    finally:
        plt.close(fig)


def test_plot_grid_1d_open_field_creates_axis_when_none_given():
    position = np.linspace(0, 100, 1000)[:, None]
    env = Environment(place_bin_size=5.0).fit_place_grid(position)
    ax = env.plot_grid()
    try:
        assert ax.get_figure() is not None
        mesh, edge_lines = ax.collections
        assert len(edge_lines.get_segments()) == len(env.edges_[0])
        assert mesh.get_array().size == env.is_track_interior_.size
    finally:
        plt.close(ax.get_figure())
