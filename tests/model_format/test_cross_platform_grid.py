"""A grid rebuilt a few ulp off, as on another platform, still decodes.

Encoding models record an exact SHA-256 of the grid they were fitted on.
Another platform can rebuild a track graph's bin edges and centers 1 ulp off
(``np.interp`` is FMA-fused on macOS arm64 but not on Linux x86), which
changed the hash and refused every cross-platform decode. The stored grid
arrays are inputs: a rebuild within the fingerprint tolerance takes them
exactly, so the model decodes as the original; a rebuild well beyond it is
still refused.
"""

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector import Environment, environment
from non_local_detector.exceptions import ValidationError
from tests.model_format._compare import assert_same_fitted_state
from tests.model_format.conftest import SORTED_RATE, algorithm_params, decode
from tests.model_format.test_scenarios import _on_track, _track_graph

pytestmark = pytest.mark.integration


def _nudged(values, size):
    """``values`` with every third element moved by ``size`` (in ulp if
    ``size`` is an int, else by that amount)."""
    values = np.array(values, dtype=float)
    picked = values[..., ::3]
    if isinstance(size, int):
        for _ in range(size):
            picked = np.nextafter(picked, np.inf)
    else:
        picked = picked + size
    values[..., ::3] = picked
    return values


def _perturb_track_grid(monkeypatch, size):
    original = environment.get_track_grid

    def perturbed(*args, **kwargs):
        grid = list(original(*args, **kwargs))
        grid[0] = _nudged(grid[0], size)  # place_bin_centers
        grid[1] = _nudged(grid[1], size)  # place_bin_edges
        grid[5] = [_nudged(edge, size) for edge in grid[5]]  # edges
        return tuple(grid)

    monkeypatch.setattr(environment, "get_track_grid", perturbed)


def _perturb_open_field(monkeypatch, size):
    original = environment.get_centers
    monkeypatch.setattr(
        environment, "get_centers", lambda edges: _nudged(original(edges), size)
    )


@pytest.fixture(params=["track_graph", "open_field"])
def saved(request, sorted_recording, tmp_path):
    time, linear, spike_times = sorted_recording
    if request.param == "track_graph":
        position = _on_track(linear)
        env = Environment(
            place_bin_size=2.0,
            track_graph=_track_graph(),
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=0.0,
        )
        perturb = _perturb_track_grid
    else:
        position = np.column_stack([linear, 14.0 + 10.0 * np.sin(time)])
        env = Environment(place_bin_size=4.0)
        perturb = _perturb_open_field
    detector = nld.NonLocalSortedSpikesDetector(
        environments=env,
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
        sampling_frequency=SORTED_RATE,
    ).fit(time, position, spike_times)
    path = tmp_path / "model.nld"
    detector.save_model(path)
    return detector, path, position, perturb


def test_a_grid_rebuilt_an_ulp_off_decodes_as_the_original(
    saved, sorted_recording, monkeypatch
):
    detector, path, position, perturb = saved
    time, _, spike_times = sorted_recording
    perturb(monkeypatch, 1)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    for a, b in zip(
        decode(loaded, time, position, {"spike_times": spike_times}),
        decode(detector, time, position, {"spike_times": spike_times}),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    grid, fitted = loaded.environments_[0], detector.environments_[0]
    np.testing.assert_array_equal(grid.place_bin_centers_, fitted.place_bin_centers_)
    for a, b in zip(grid.edges_, fitted.edges_, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_grid_rebuilt_well_beyond_the_tolerance_is_refused(
    saved, sorted_recording, monkeypatch
):
    _, path, position, perturb = saved
    time, _, spike_times = sorted_recording
    perturb(monkeypatch, 1e-3)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    with pytest.raises(ValidationError, match="environment ''"):
        decode(loaded, time, position, {"spike_times": spike_times})


def test_an_unperturbed_grid_round_trips(saved, tmp_path):
    detector, path, _, _ = saved
    assert_same_fitted_state(
        detector, nld.NonLocalSortedSpikesDetector.load_model(path)
    )
