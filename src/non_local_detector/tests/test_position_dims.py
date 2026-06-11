"""Tests for the canonical position-dimension naming helpers.

``_position_dims`` is the single source of truth for the position-dim naming
convention shared by the results constructors, ``ContFrag*.get_posterior``, and
``get_HPD_spatial_coverage``. These tests pin the convention directly — in
particular the 5-6D ``v_position``/``u_position`` names and the >6D
``dim{i}_position`` fallback, the branch whose absence allowed the latent
5-6D column-truncation bug.
"""

import numpy as np
import pytest
import xarray as xr

from non_local_detector._position_dims import (
    get_position_dim_names,
    get_position_dims,
)


@pytest.mark.unit
class TestGetPositionDimNames:
    """``get_position_dim_names`` builds the canonical names."""

    @pytest.mark.parametrize(
        ("n_position_dims", "expected"),
        [
            (1, ["position"]),
            (2, ["x_position", "y_position"]),
            (3, ["x_position", "y_position", "z_position"]),
            (
                5,
                ["x_position", "y_position", "z_position", "w_position", "v_position"],
            ),
            (
                6,
                [
                    "x_position",
                    "y_position",
                    "z_position",
                    "w_position",
                    "v_position",
                    "u_position",
                ],
            ),
            (
                7,
                [f"dim{i}_position" for i in range(7)],
            ),
        ],
    )
    def test_names(self, n_position_dims, expected):
        assert get_position_dim_names(n_position_dims) == expected

    def test_length_always_matches_dimension(self):
        """One name per dimension at every size (the 5-6D truncation regression).

        The pre-refactor ``_convert_seq_to_df`` capped labels at four and
        silently dropped the 5th/6th column; the convention must always return
        exactly ``n_position_dims`` names.
        """
        for n in range(1, 10):
            assert len(get_position_dim_names(n)) == n

    @pytest.mark.parametrize("n", [0, -1, -3])
    def test_rejects_non_positive_dimension(self, n):
        """``n_position_dims < 1`` raises rather than silently returning ``[]``.

        A 0/negative count previously produced an empty name list, which would
        build a degenerate empty MultiIndex level set downstream — exactly the
        silent failure the convention is meant to prevent.
        """
        with pytest.raises(ValueError, match=">= 1"):
            get_position_dim_names(n)


@pytest.mark.unit
class TestGetPositionDims:
    """``get_position_dims`` detects the names on a DataArray, in array order."""

    @pytest.mark.parametrize("n_position_dims", [1, 2, 3, 5, 6, 7])
    def test_round_trip(self, n_position_dims):
        """Names built by get_position_dim_names are recovered, in order."""
        names = get_position_dim_names(n_position_dims)
        dims = ["time", *names]
        shape = (4,) + (2,) * n_position_dims
        da = xr.DataArray(np.zeros(shape), dims=dims)
        assert get_position_dims(da) == names

    def test_excludes_non_position_dims(self):
        """Only ``position`` / ``*_position`` dims are returned; others ignored."""
        da = xr.DataArray(
            np.zeros((2, 3, 3, 3)),
            dims=["time", "x_position", "y_position", "state"],
        )
        assert get_position_dims(da) == ["x_position", "y_position"]

    def test_no_false_positive_on_bare_position_suffix(self):
        """A dim ending in 'position' but not '_position' must not match."""
        da = xr.DataArray(np.zeros((2, 3)), dims=["time", "composition"])
        assert get_position_dims(da) == []

    @pytest.mark.parametrize(
        "stray_dim", ["head_position", "map_position", "linear_position"]
    )
    def test_no_false_positive_on_underscore_position_suffix(self, stray_dim):
        """A non-canonical ``*_position`` dim must not be treated as a position axis.

        Detection matches only the closed vocabulary emitted by
        ``get_position_dim_names`` (``position``, ``x``..``u``\\ ``_position``,
        ``dim{i}_position``). A stray dim like ``head_position`` ends in
        ``_position`` but is not a decoder position axis; matching it would
        wrongly marginalize over it.
        """
        da = xr.DataArray(np.zeros((2, 3, 3)), dims=["time", "x_position", stray_dim])
        assert get_position_dims(da) == ["x_position"]
