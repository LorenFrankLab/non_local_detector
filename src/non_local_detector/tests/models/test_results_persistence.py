"""Tests for ``_DetectorBase.save_results`` / ``load_results`` round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import xarray as xr

from non_local_detector.models.base import _DetectorBase


def _make_results_with_upstream_scalar_coords(
    n_time: int = 4, n_state_bins: int = 6
) -> xr.Dataset:
    """Build a results dataset that mirrors the real-data NetCDF schema.

    Real upstream NetCDFs (``cont_results.nc``, ``cont_frag_results.nc``)
    carry **0-D scalar** coords like ``environments`` and
    ``encoding_groups``. These appear in ``list(ds["state_bins"].coords)``
    even though they have no dim — that's what broke ``set_index``
    before the filter.
    """
    state_ind = np.array([0, 0, 0, 1, 1, 1])
    position_ind = np.array([0, 1, 2, 0, 1, 2])
    state_bins = pd.MultiIndex.from_arrays(
        [state_ind, position_ind], names=("state", "position")
    )
    return xr.Dataset(
        {
            "acausal_posterior": (
                ("time", "state_bins"),
                np.zeros((n_time, n_state_bins), dtype=np.float32),
            ),
        },
        coords={
            "time": np.arange(n_time, dtype=np.float64),
            "state_bins": state_bins,
            "environments": "env0",
            "encoding_groups": "grp0",
        },
    )


@pytest.mark.unit
def test_load_results_preserves_state_bins_multiindex(tmp_path: Path) -> None:
    """Round-trip restores the ``(state, position)`` MultiIndex."""
    original = _make_results_with_upstream_scalar_coords()
    path = tmp_path / "results.nc"
    _DetectorBase.save_results(original, str(path))
    loaded = _DetectorBase.load_results(str(path))
    assert isinstance(loaded.indexes["state_bins"], pd.MultiIndex)
    assert loaded.indexes["state_bins"].names == ["state", "position"]


@pytest.mark.unit
def test_load_results_handles_scalar_coords_on_state_bins(
    tmp_path: Path,
) -> None:
    """Regression: real upstream NetCDFs attach 0-D scalar coords to state_bins.

    Before the filter in ``load_results``, blindly passing every coord
    name on ``state_bins`` to ``set_index`` raised
    ``ValueError: PandasMultiIndex only accepts 1-dimensional variables``
    on the upstream ``cont_results.nc`` because ``environments`` and
    ``encoding_groups`` come back as 0-D scalars after the
    ``reset_index`` round-trip.
    """
    original = _make_results_with_upstream_scalar_coords()
    path = tmp_path / "results.nc"
    _DetectorBase.save_results(original, str(path))

    # Sanity check on the on-disk file: the scalar coords must land
    # back as 0-D before calling load_results, otherwise this test
    # isn't actually exercising the regression.
    raw = xr.open_dataset(str(path))
    assert raw["state_bins"].coords["environments"].dims == ()

    loaded = _DetectorBase.load_results(str(path))

    assert loaded.indexes["state_bins"].names == ["state", "position"]
    assert "environments" in loaded.coords
    assert "encoding_groups" in loaded.coords
