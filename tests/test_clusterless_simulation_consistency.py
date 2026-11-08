"""Run data and replays from the clusterless simulator must be consistent.

A model fit on ``make_simulated_run_data`` is decoded against the replay
simulators, so the replay marks must sit on the run data's mark clusters and
the default place fields must lie on the simulated track.
"""

import numpy as np
import pytest

from non_local_detector.simulate import clusterless_simulation as cs
from non_local_detector.simulate.clusterless_simulation import (
    make_continuous_replay,
    make_fragmented_continuous_fragmented_replay,
    make_fragmented_hover_fragmented_replay,
    make_fragmented_replay,
    make_hover_continuous_hover_replay,
    make_hover_replay,
    make_simulated_run_data,
)

pytestmark = pytest.mark.unit


def _replay_marks(multiunits, tetrode):
    marks = multiunits[:, 0, tetrode]
    return np.unique(marks[~np.isnan(marks)])


@pytest.fixture(scope="module")
def default_run():
    return make_simulated_run_data(n_runs=2, seed=0)


def test_default_replay_marks_land_on_run_data_mark_clusters(default_run) -> None:
    """Each replay mark is within a few SD of a mark cluster seen in run data."""
    n_tetrodes = len(default_run.spike_times)
    _, replay = make_continuous_replay()
    for tet in range(n_tetrodes):
        run_feats = default_run.spike_waveform_features[tet][:, 0]
        for mark in _replay_marks(replay, tet):
            # mark_variance is 1.0 in the simulator: >100 run spikes within 3 SD
            n_near = np.sum(np.abs(run_feats - mark) < 3.0)
            assert n_near > 100, (tet, mark, n_near)


def _assert_hover_matches_cell(run, replay, cell_ind) -> None:
    """The hover replay spikes with the run-data mark cluster of ``cell_ind``.

    That cluster's run spikes must fire around the cell's place field, which
    lies on the track.
    """
    place_means = cs.PLACE_FIELD_MEANS.reshape((len(run.spike_times), -1))
    tet, cell = np.unravel_index(cell_ind, place_means.shape)
    assert 0.0 <= place_means[tet, cell] <= cs.TRACK_HEIGHT
    for other in set(range(len(run.spike_times))) - {tet}:
        assert _replay_marks(replay, other).size == 0
    mark = _replay_marks(replay, tet)
    assert mark.size == 1
    run_feats = run.spike_waveform_features[tet][:, 0]
    spike_ind = np.nonzero(np.abs(run_feats - mark[0]) < 3.0)[0]
    assert spike_ind.size > 100
    spike_pos = np.interp(
        run.spike_times[tet][spike_ind], run.position_time, run.position[:, 0]
    )
    assert abs(np.median(spike_pos) - place_means[tet, cell]) < 5.0


def test_hover_replay_mark_matches_run_data_cell(default_run) -> None:
    """The default hover replay mark belongs to the cell at the hover position."""
    _, replay = make_hover_replay()
    _assert_hover_matches_cell(default_run, replay, cs.PLACE_FIELD_MEANS.size // 2)


def test_default_place_fields_are_on_the_track() -> None:
    """Every default place field mean lies within [0, TRACK_HEIGHT]."""
    means = cs.PLACE_FIELD_MEANS
    assert means.min() >= 0.0
    assert means.max() <= cs.TRACK_HEIGHT
    assert means.size % cs.N_TETRODES == 0


def test_default_hover_helpers_hover_on_track(default_run) -> None:
    """Each default hover segment is an on-track cell's hover replay."""
    n_cells = cs.PLACE_FIELD_MEANS.size
    _, hover_continuous_hover = make_hover_continuous_hover_replay()
    _, fragmented_hover_fragmented = make_fragmented_hover_fragmented_replay()
    n_hover = make_hover_replay()[1].shape[0]
    n_fragmented = make_fragmented_replay()[1].shape[0]
    segments = {
        0: hover_continuous_hover[:n_hover],
        n_cells - 1: hover_continuous_hover[-n_hover:],
        6: fragmented_hover_fragmented[n_fragmented : n_fragmented + n_hover],
    }
    for cell_ind, segment in segments.items():
        _, replay = make_hover_replay(hover_neuron_ind=cell_ind)
        np.testing.assert_array_equal(segment, replay)
        _assert_hover_matches_cell(default_run, replay, cell_ind)


def test_cells_must_divide_evenly_among_tetrodes() -> None:
    with pytest.raises(ValueError, match=r"10 cells.*3 tetrodes"):
        make_continuous_replay(place_field_means=np.linspace(0, 100, 10), n_tetrodes=3)


def test_mixture_helpers_accept_matching_parameters() -> None:
    """Mixture helpers take the same layout parameters as the run data."""
    kwargs = {
        "place_field_means": np.linspace(0, 100, 12),
        "n_tetrodes": 3,
        "n_features": 2,
        "mark_spacing": 7.0,
    }
    for fn in (
        make_hover_continuous_hover_replay,
        make_fragmented_hover_fragmented_replay,
        make_fragmented_continuous_fragmented_replay,
    ):
        _, mu = fn(**kwargs)
        assert mu.shape[1:] == (2, 3)
        marks = mu[~np.isnan(mu)]
        # four cells per tetrode -> mark centers 0, 7, 14, 21
        assert set(np.unique(marks)) <= {0.0, 7.0, 14.0, 21.0}


@pytest.mark.parametrize("n_tetrodes", [2, 3, 7])
def test_continuous_replay_respects_n_tetrodes(n_tetrodes: int) -> None:
    """Continuous replay works for any n_tetrodes / cells-per-tetrode split."""
    n_per = 3
    means = np.linspace(0, 100, n_tetrodes * n_per)
    _, mu = make_continuous_replay(
        place_field_means=means, n_tetrodes=n_tetrodes, n_features=3
    )
    assert mu.shape[1:] == (3, n_tetrodes)
    # every cell contributes exactly one spike
    n_spiking = np.sum(~np.isnan(mu[:, 0, :]))
    assert n_spiking == means.size
    for tet in range(n_tetrodes):
        assert _replay_marks(mu, tet).size == n_per


def test_run_data_respects_n_features() -> None:
    """Run data feature dimension follows ``n_features``."""
    sim = make_simulated_run_data(
        n_tetrodes=2, place_field_means=np.array([20.0, 60.0]), n_features=3, n_runs=1
    )
    assert all(f.shape[1] == 3 for f in sim.spike_waveform_features)


def test_default_place_fields_are_visited_during_running(default_run) -> None:
    """The animal passes through every default place field."""
    position = default_run.position[:, 0]
    for mean in cs.PLACE_FIELD_MEANS:
        occupancy = np.mean(np.abs(position - mean) < 2.0)
        assert occupancy > 0.01, (mean, occupancy)


def test_sorted_spikes_place_fields_follow_track_height() -> None:
    """Sorted-spikes place fields span the requested track, not a fixed one."""
    from non_local_detector.simulate.sorted_spikes_simulation import (
        make_simulated_data,
    )

    out = make_simulated_data(track_height=100.0, n_neurons=5, sampling_frequency=500)
    place_fields, linear_distance = out[7], out[1]
    peak_position = linear_distance[np.argmax(place_fields, axis=0)]
    assert linear_distance.max() <= 100.5
    assert np.all(np.diff(np.sort(peak_position)) > 0)
    assert peak_position.max() > 90.0
