"""The layout of the xarray results that ``predict`` returns.

Dense and checkpointed inference build their results separately, so these
tests pin the coordinates both share, including those downstream code
(Spyglass) reads, and check they survive ``save_results``/``load_results``.
"""

import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
)

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
_CHECKPOINTED_ONLY = {"source_row", "time_bin_width"}


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
    for name in result.data_vars:
        np.testing.assert_array_equal(restored[name].values, result[name].values)


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
    xr.testing.assert_equal(restored, dense)
