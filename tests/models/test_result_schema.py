"""The layout of the xarray results that ``predict`` returns.

Dense and checkpointed inference build their results separately, so these
tests pin the coordinates both share, including those downstream code
(Spyglass) reads, and check they survive ``save_results``/``load_results``.
The tables of ``docs/result_layout.md`` are checked against real results.
"""

import re
import sys
import warnings
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
)
from non_local_detector.analysis import get_spatial_posterior
from non_local_detector.models._results import has_place_bins, state_bin_positions

pytestmark = pytest.mark.integration

_TIME = np.linspace(0, 2, 61)
_POSITION = np.column_stack((5 + 4 * np.sin(_TIME), 5 + 4 * np.cos(_TIME)))
_SPIKES = [np.array([0.1, 0.3, 0.8, 1.2, 1.9])]
_PREDICT = {
    "position_time": _TIME,
    "position": _POSITION,
    "spike_times": _SPIKES,
    "time_edges": np.linspace(0, 2, 101),
}
# Coordinates only checkpointed results carry.
_CHECKPOINTED_ONLY = {"source_row"}


@pytest.fixture(scope="module", params=["single_state", "multi_state"])
def fitted(request):
    """A fitted decoder (one state) or non-local detector (four states)."""
    model_class = {
        "single_state": SortedSpikesDecoder,
        "multi_state": NonLocalSortedSpikesDetector,
    }[request.param]
    model = model_class(
        environments=Environment(place_bin_size=2, position_range=((0, 10), (0, 10))),
        infer_track_interior=False,
    )
    return model.fit(position_time=_TIME, position=_POSITION, spike_times=_SPIKES)


@pytest.fixture(scope="module")
def dense(fitted):
    return fitted.predict(**_PREDICT, return_outputs="all")


@pytest.fixture(scope="module")
def checkpointed(fitted, tmp_path_factory):
    return fitted.predict(
        **_PREDICT,
        return_outputs="all",
        inference_mode="checkpointed",
        output_mode="spatial",
        result_path=tmp_path_factory.mktemp("store") / "result",
        chunk_size=13,
    )


@pytest.fixture(scope="module")
def em(fitted):
    """A dense EM result (one iteration, so EM does not converge)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "EM did not converge")
        return type(fitted)(**fitted.get_params()).estimate_parameters(
            **_PREDICT, max_iter=1, return_outputs="all"
        )


@pytest.fixture(scope="module", params=["dense", "checkpointed"])
def result(request, dense, checkpointed):
    """The dense or the checkpointed result."""
    return {"dense": dense, "checkpointed": checkpointed}[request.param]


def _layout(result):
    """Dims of every variable and of every coordinate both paths share."""
    return {
        "sizes": dict(result.sizes),
        "data_vars": {name: v.dims for name, v in result.data_vars.items()},
        "coords": {
            name: c.dims
            for name, c in result.coords.items()
            if name not in _CHECKPOINTED_ONLY
        },
        "state_bins_levels": list(result.indexes["state_bins"].names),
    }


def test_state_ind_is_a_state_bins_coordinate(fitted, result):
    assert "state_ind" not in result.dims
    assert result.coords["state_ind"].dims == ("state_bins",)
    np.testing.assert_array_equal(result.state_ind, fitted.state_ind_)
    # Each bin's state index names the state on the MultiIndex.
    np.testing.assert_array_equal(
        np.atleast_1d(result.coords["states"].values)[result.state_ind],
        result.coords["state"].values,
    )


def test_state_bin_positions_follow_the_state_index(fitted, result):
    """One block per state, NaN for states without place bins, as in results."""
    positions = state_bin_positions(fitted)
    np.testing.assert_array_equal([len(p) for p in positions], fitted.bin_sizes_)
    for observation, block in zip(fitted.observation_models, positions, strict=True):
        assert bool(np.isnan(block).all()) is not has_place_bins(fitted, observation)
    position_names = result.indexes["state_bins"].names[1:]
    np.testing.assert_array_equal(
        np.concatenate(positions),
        np.column_stack([result[name].values for name in position_names]),
    )


def test_dense_and_checkpointed_results_share_one_layout(dense, checkpointed):
    assert _layout(checkpointed) == _layout(dense)
    for name in dense.coords:
        if dense.coords[name].dtype.kind in "biuf":
            np.testing.assert_array_equal(
                checkpointed.coords[name].values, dense.coords[name].values
            )


def test_single_state_results_drop_the_states_dimension(fitted, dense):
    is_single_state = len(fitted.state_names) == 1
    assert ("states" in dense.dims) is not is_single_state
    expected = ("time",) if is_single_state else ("time", "states")
    assert dense.acausal_state_probabilities.dims == expected


def test_spyglass_access_patterns(result):
    """The three ways Spyglass (decoding v1) reads results keep working."""
    n_bins = result.sizes["state_bins"]
    # A per-bin array labelled with the result's state_bins coordinate.
    per_bin = xr.DataArray(
        np.arange(n_bins),
        dims=("state_bins",),
        coords={"state_bins": result.coords["state_bins"]},
    )
    assert per_bin.sizes == {"state_bins": n_bins}
    # The posterior over position, summed over states.
    by_position = result.acausal_posterior.unstack("state_bins").sum("state")
    position_names = result.indexes["state_bins"].names[1:]
    assert by_position.dims == ("time", *position_names)
    # State names as an array, for single- and multi-state models.
    states = np.atleast_1d(result.coords["states"].values)
    assert set(result.coords["state"].values) <= set(states)


def test_netcdf_round_trip_keeps_state_ind(fitted, result, tmp_path):
    filename = tmp_path / "results.nc"
    fitted.save_results(result, filename)
    with fitted.load_results(filename) as restored:
        restored = restored.load()
    assert list(restored.indexes["state_bins"].names) == list(
        result.indexes["state_bins"].names
    )
    assert restored.indexes["state_bins"].equals(result.indexes["state_bins"])
    assert restored.coords["state_ind"].dims == ("state_bins",)
    np.testing.assert_array_equal(restored.state_ind, fitted.state_ind_)
    assert restored.state_ind.dtype == result.state_ind.dtype
    for name in result.data_vars:
        np.testing.assert_array_equal(restored[name].values, result[name].values)


@pytest.mark.parametrize("file_format", ["netCDF4", "netCDF3"])
@pytest.mark.parametrize("kind", ["dense", "checkpointed", "em"])
def test_netcdf_round_trip_keeps_every_dtype(
    fitted, kind, file_format, request, tmp_path, monkeypatch
):
    """Integer variables and coordinates keep their dtypes through netCDF.

    xarray writes netCDF4 when netCDF4 (or h5netcdf) is installed, and
    netCDF3 otherwise, which stores int64 as int32. Strings come back as
    object or fixed-width string arrays, depending on the format.
    """
    if file_format == "netCDF4":
        pytest.importorskip("netCDF4")
    result = request.getfixturevalue(kind)
    filename = tmp_path / "results.nc"
    with monkeypatch.context() as patch:
        if file_format == "netCDF3":  # written as without netCDF4, read with it
            for writer in ("netCDF4", "h5netcdf"):
                patch.setitem(sys.modules, writer, None)
        fitted.save_results(result, filename)
    magic = b"CDF" if file_format == "netCDF3" else b"\x89HDF"
    assert filename.read_bytes().startswith(magic)
    with fitted.load_results(filename) as restored:
        restored = restored.load()
    assert set(restored.variables) == set(result.variables)
    assert any(v.dtype.kind in "iu" for v in result.variables.values())
    for name, variable in result.variables.items():
        if "U" in (variable.dtype.kind, restored[name].dtype.kind):
            np.testing.assert_array_equal(restored[name].values, variable.values)
        else:
            assert restored[name].dtype == variable.dtype, name
    assert restored.attrs.keys() == {
        name for name in result.attrs if name != "_native_metadata"
    }


def test_a_recorded_dtype_is_not_forced_on_a_non_integer_variable(
    fitted, dense, tmp_path
):
    """A file edited after saving keeps its non-integer values.

    Making an integer coordinate float (here with a NaN) and rewriting the
    file keeps the recorded dtype; casting back would turn NaN into garbage.
    """
    filename = tmp_path / "results.nc"
    fitted.save_results(dense, filename)
    with xr.open_dataset(filename) as saved:
        edited = saved.load()
    groups = edited["encoding_groups"].values.astype(float)
    groups.flat[0] = np.nan
    edited = edited.assign_coords(
        encoding_groups=(edited["encoding_groups"].dims, groups)
    )
    edited.to_netcdf(tmp_path / "edited.nc")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with fitted.load_results(tmp_path / "edited.nc") as restored:
            np.testing.assert_array_equal(restored["encoding_groups"].values, groups)
    np.testing.assert_array_equal(restored.state_ind, fitted.state_ind_)


@pytest.mark.parametrize("has_state_bins", [True, False])
def test_load_results_releases_the_file(fitted, dense, tmp_path, has_state_bins):
    """Closing the loaded results closes the netCDF file."""
    results = dense if has_state_bins else dense.drop_dims("state_bins")
    filename = tmp_path / "results.nc"
    fitted.save_results(results, filename)
    with fitted.load_results(filename) as restored:
        assert restored._close is not None
        restored.load()
        assert _open_paths() is None or filename.resolve() in _open_paths()
    assert _open_paths() is None or filename.resolve() not in _open_paths()
    # Rewriting the file fails on Windows while it is still open.
    fitted.save_results(results, filename)
    filename.unlink()


def _open_paths():
    """Files this process has open (Linux), or None where that is unknown."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        return None
    paths = set()
    for fd in fd_dir.iterdir():
        with suppress(OSError):
            paths.add(fd.resolve())
    return paths


def test_files_saved_without_recorded_dtypes_restore_state_ind(fitted, dense, tmp_path):
    """Files written before ``save_results`` recorded integer dtypes load."""
    filename = tmp_path / "results.nc"
    fitted.save_results(dense, filename)
    with xr.open_dataset(filename) as saved:
        legacy = saved.load()
    del legacy.attrs["non_local_detector_integer_dtypes"]
    legacy.to_netcdf(tmp_path / "legacy.nc")
    with fitted.load_results(tmp_path / "legacy.nc") as restored:
        restored = restored.load()
    assert restored.state_ind.dtype == dense.state_ind.dtype
    assert restored.indexes["state_bins"].equals(dense.indexes["state_bins"])
    for name in dense.data_vars:
        np.testing.assert_array_equal(restored[name].values, dense[name].values)


def test_files_with_a_state_ind_dimension_load_with_the_current_layout(
    fitted, dense, tmp_path
):
    """Results saved before ``state_ind`` moved onto ``state_bins`` had it as
    its own dimension; loading them gives the current layout."""
    legacy = dense.drop_vars("state_ind").assign_coords(
        state_ind=("state_ind", fitted.state_ind_)
    )
    filename = tmp_path / "legacy.nc"
    fitted.save_results(legacy, filename)
    with fitted.load_results(filename) as restored:
        restored = restored.load()
    assert "state_ind" not in restored.dims
    assert restored.state_ind.dtype == dense.state_ind.dtype
    xr.testing.assert_equal(restored, dense)


def test_time_bin_width_is_each_bins_duration(result):
    np.testing.assert_array_equal(
        result.time_bin_width, result.time_bin_end - result.time_bin_start
    )
    np.testing.assert_array_equal(
        result.time_bin_width, np.diff(_PREDICT["time_edges"])
    )


def test_spatial_posterior_leaves_out_state_ind(dense):
    assert "state_ind" in dense.acausal_posterior.unstack("state_bins").coords
    assert "state_ind" not in get_spatial_posterior(dense.acausal_posterior).coords


_DOCS = Path(__file__).parents[2] / "docs" / "result_layout.md"


def _doc_table(heading):
    """Rows of the first table under ``## heading``, as {column: cell}."""
    section = _DOCS.read_text().split(f"\n## {heading}\n", 1)[1]
    section = section.split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]

    def cells(line):
        return [cell.strip() for cell in line.strip("|").split("|")]

    header = cells(lines[0])
    return [dict(zip(header, cells(line), strict=True)) for line in lines[2:]]


def _names(cell):
    return re.findall(r"`([^`]+)`", cell)


def _dims(text, single_state):
    """``(time, states)`` or ``time`` as a tuple, without ``states`` for a
    single-state model."""
    dims = tuple(d.strip() for d in text.strip("`()").split(","))
    return tuple(d for d in dims if not (single_state and d == "states"))


def _documented_coords(kind, single_state):
    """Coordinate dims the docs give a dense, spatial or compact result."""
    has_state_bins = kind != "compact"
    included = {
        "All": True,
        "Checkpointed": kind != "dense",
        "With `state_bins`": has_state_bins,
    }
    coords = {}
    for row in _doc_table("Coordinates"):
        if included[row["Present in"]]:
            for name in _names(row["Coordinate"]):
                coords[name] = _dims(row["Dimension"], single_state)
    if has_state_bins:
        # The MultiIndex levels of the 2-D test environment.
        for level in ("state", "x_position", "y_position"):
            coords[level] = ("state_bins",)
    return coords


def _documented_variables(option, compact, single_state):
    """Data variable dims the docs give for a string ``return_outputs``."""
    variables = {}
    for row in _doc_table("Data variables"):
        added_by = row["Added by `return_outputs`"]
        options = set(re.findall(r'`"([^"`]+)"`', added_by))
        included = added_by == "Always" or option == "all" or option in options
        if compact and row["In compact results"] != "Yes":
            included = False
        if included:
            (name,) = _names(row["Variable"])
            variables[name] = _dims(row["Dimensions"], single_state)
    return variables


def _documented_attrs(*present_in):
    return {
        name
        for row in _doc_table("Attributes")
        if row["Present in"] in present_in
        for name in _names(row["Attribute"])
    }


def _actual(result):
    return (
        {name: c.dims for name, c in result.coords.items()},
        {name: v.dims for name, v in result.data_vars.items()},
    )


@pytest.mark.parametrize(
    "option",
    [None, "filter", "predictive", "predictive_posterior", "log_likelihood", "all"],
)
def test_docs_tables_match_dense_results(fitted, option):
    single_state = len(fitted.state_names) == 1
    result = fitted.predict(**_PREDICT, return_outputs=option)
    assert _actual(result) == (
        _documented_coords("dense", single_state),
        _documented_variables(option, False, single_state),
    )
    assert set(result.attrs) == _documented_attrs("All")


@pytest.mark.parametrize("option", [None, "filter", "predictive"])
def test_docs_tables_match_compact_results(fitted, option):
    single_state = len(fitted.state_names) == 1
    result = fitted.predict(
        **_PREDICT,
        return_outputs=option,
        inference_mode="checkpointed",
        output_mode="compact",
        chunk_size=13,
    )
    assert _actual(result) == (
        _documented_coords("compact", single_state),
        _documented_variables(option, True, single_state),
    )
    assert set(result.attrs) == _documented_attrs("All", "Checkpointed")


def test_docs_tables_match_checkpointed_spatial_results(fitted, checkpointed):
    single_state = len(fitted.state_names) == 1
    assert _actual(checkpointed) == (
        _documented_coords("spatial", single_state),
        _documented_variables("all", False, single_state),
    )
    assert set(checkpointed.attrs) == _documented_attrs("All", "Checkpointed")


@pytest.mark.filterwarnings("ignore:EM did not converge")
@pytest.mark.parametrize(
    ("representation", "stored", "present_in"),
    [
        ("dense", False, ("Dense EM",)),
        ("structured", False, ("Structured EM",)),
        ("structured", True, ("Structured EM", "Checkpointed")),
    ],
)
def test_docs_tables_match_em_results(
    fitted, tmp_path, representation, stored, present_in
):
    single_state = len(fitted.state_names) == 1
    result = type(fitted)(**fitted.get_params()).estimate_parameters(
        **_PREDICT,
        max_iter=1,
        return_outputs="all",
        transition_representation=representation,
        result_path=tmp_path / "em" if stored else None,
    )
    kind = "spatial" if stored else "dense"
    assert _actual(result) == (
        _documented_coords(kind, single_state),
        _documented_variables("all", False, single_state),
    )
    assert set(result.attrs) == _documented_attrs(
        "All", "`estimate_parameters`", *present_in
    )
