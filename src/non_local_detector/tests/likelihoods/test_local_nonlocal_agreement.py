"""Local and non-local likelihoods agree at the bin the animal occupies.

The local path scores spikes at the animal's position; the non-local path
scores them at every interior bin. Where the animal holds still at a bin
centre, the local row is the non-local row's entry for that bin, for every
registered backend. Where it holds still off the track (beyond the grid or in
a non-interior bin), the local row is the non-local row's entry for the
nearest interior bin.
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import snap_off_track_positions
from non_local_detector.tests.likelihoods.conftest import (
    ALGORITHMS,
    FLOAT32_ROUNDING,
    fit_registered_backends,
)
from non_local_detector.time_edges import time_edges_from_centers

DECODE_DT = 0.02
N_DECODE_BINS = 10
# Smooth across the 10 cm bins (the default position_std is below one bin).
FIT_PARAMS = {
    "sorted_spikes_diffusion": {"position_std": 10.0},
    "clusterless_diffusion": {"position_std": 10.0},
}


def _encoding_data(position_time, position, environment, field_centers, seed=3):
    """Two units, each firing within 10 cm of its field centre."""
    rng = np.random.default_rng(seed)
    spike_times, features = [], []
    for center in field_centers:
        near = np.linalg.norm(position - center, axis=1) < 10.0
        times = np.sort(rng.choice(position_time[near], 300, replace=False))
        spike_times.append(times + rng.uniform(-0.005, 0.005, times.size))
        features.append(rng.standard_normal((300, 2)) * 3.0 + center[0] / 5.0)
    return {
        "position_time": position_time,
        "position": position,
        "environment": environment,
        "encoding_spike_times": spike_times,
        "encoding_features": features,
    }


def _fit(data):
    return {
        name: (predict, model, is_clusterless)
        for name, predict, model, is_clusterless in fit_registered_backends(
            data, FIT_PARAMS
        )
    }


@pytest.fixture(scope="module")
def line():
    """Fields at 25 and 75 cm on a 0-100 cm, 10 cm-bin track."""
    position_time = np.arange(0.0, 20.0, 0.01)
    position = (50.0 + 45.0 * np.sin(2 * np.pi * position_time / 5.0))[:, None]
    environment = Environment(
        environment_name="line", place_bin_size=10.0, position_range=((0.0, 100.0),)
    ).fit_place_grid(position=position, infer_track_interior=False)
    data = _encoding_data(
        position_time, position, environment, np.array([[25.0], [75.0]])
    )
    return environment, _fit(data)


@pytest.fixture(scope="module")
def box_with_hole():
    """A 0-100 cm box of 10 cm bins whose [40, 60]^2 bins are not interior.

    The 12 x 12 grid also has a non-interior boundary ring of bins. The animal
    runs around square loops 10-45 cm from the centre, so it never enters the
    hole.
    """
    position_time = np.arange(0.0, 200.0, 0.01)
    half_width = 27.5 + 17.5 * np.sin(2 * np.pi * position_time / 37.0)
    side, along = np.divmod(4.0 * ((position_time / 4.0) % 1.0), 1.0)
    offset = half_width * (2.0 * along - 1.0)
    position = 50.0 + np.select(
        [side[:, None] == k for k in range(4)],
        [
            np.column_stack([offset, -half_width]),
            np.column_stack([half_width, offset]),
            np.column_stack([-offset, half_width]),
            np.column_stack([-half_width, -offset]),
        ],
    )
    is_track_interior = np.zeros((12, 12), dtype=bool)
    is_track_interior[1:-1, 1:-1] = True
    is_track_interior[5:7, 5:7] = False
    environment = Environment(
        environment_name="box",
        place_bin_size=10.0,
        position_range=((0.0, 100.0), (0.0, 100.0)),
        is_track_interior=is_track_interior,
    ).fit_place_grid(position=position)
    data = _encoding_data(
        position_time, position, environment, np.array([[25.0, 25.0], [85.0, 55.0]])
    )
    return environment, _fit(data)


def _assert_local_equals_nonlocal(fitted, environment, name, held, expected_center):
    """Hold the animal at ``held``; compare local to the bin at ``expected_center``.

    Tolerance: ``FLOAT32_ROUNDING`` (rtol 1e-6, atol 1e-5). The two paths
    evaluate the same rates and marks through different float32 reductions
    (per-row matmuls or scatter-adds): on the 1-D fixture they differ by at
    most 3.1e-5 absolute on log likelihoods of order 200, 1.7e-7 relative. A
    misread position or a lost spike moves a log likelihood by O(1), and
    every non-local row must span more than 0.5 across bins.
    """
    predict, model, is_clusterless = fitted[name]
    rng = np.random.default_rng(1)

    decode_time = 100.0 + np.arange(N_DECODE_BINS) * DECODE_DT
    time_edges = time_edges_from_centers(decode_time)
    # Hold the animal across the whole stretch, so the local paths read the
    # same position at bin centres and at spike times.
    position_time = np.linspace(time_edges[0] - 1.0, time_edges[-1] + 1.0, 50)
    position = np.tile(held, (position_time.size, 1))
    # Two to four spikes per bin per unit, jittered within the bin.
    spike_times = [
        np.sort(
            np.concatenate(
                [
                    rng.uniform(start, stop, rng.integers(2, 5))
                    for start, stop in zip(time_edges[:-1], time_edges[1:], strict=True)
                ]
            )
        )
        for _ in range(2)
    ]
    args = [position_time, position, spike_times]
    if is_clusterless:
        args.append(
            [
                rng.standard_normal((times.size, 2)) * 3.0 + center
                for times, center in zip(spike_times, (5.0, 15.0), strict=True)
            ]
        )

    local = np.asarray(predict(*args, time_edges=time_edges, is_local=True, **model))
    nonlocal_ = np.asarray(
        predict(*args, time_edges=time_edges, is_local=False, **model)
    )

    interior = environment.is_track_interior_.ravel()
    (interior_bin,) = np.flatnonzero(
        np.all(
            np.isclose(environment.place_bin_centers_[interior], expected_center),
            axis=1,
        )
    )
    assert local.shape == (N_DECODE_BINS, 1)
    assert nonlocal_.shape == (N_DECODE_BINS, interior.sum())
    # The non-local row must discriminate bins, or the identity is vacuous.
    assert np.all(np.ptp(nonlocal_, axis=1) > 0.5)
    np.testing.assert_allclose(
        local[:, 0], nonlocal_[:, interior_bin], **FLOAT32_ROUNDING
    )


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize(
    ("held", "expected_center"),
    [(5.0, 5.0), (25.0, 25.0), (75.0, 75.0), (-30.0, 5.0), (130.0, 95.0)],
    ids=["edge-bin", "bin-25", "bin-75", "below-track", "beyond-track"],
)
def test_local_equals_nonlocal_on_a_line(line, name, held, expected_center):
    """Off the track, the local row is the nearest edge bin's non-local row."""
    environment, fitted = line
    _assert_local_equals_nonlocal(
        fitted, environment, name, np.array([held]), np.array([expected_center])
    )


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize(
    ("held", "expected_center"),
    [
        ((25.0, 25.0), (25.0, 25.0)),
        ((47.0, 52.0), (35.0, 55.0)),
        ((103.0, 52.0), (95.0, 55.0)),
        # Beyond the grid's 110 cm edge, within the 20 cm snap distance past
        # which the diffusion backends' bin lookup warns.
        ((112.0, 52.0), (95.0, 55.0)),
    ],
    ids=["interior-bin", "hole", "boundary-ring", "beyond-grid"],
)
def test_local_equals_nonlocal_in_a_box_with_a_hole(
    box_with_hole, name, held, expected_center
):
    """A position in a hole reads the nearest interior bin, as one beyond it."""
    environment, fitted = box_with_hole
    _assert_local_equals_nonlocal(
        fitted, environment, name, np.array(held), np.array(expected_center)
    )


@pytest.mark.unit
def test_on_track_positions_are_returned_unchanged(line, box_with_hole):
    """Positions in an interior bin, on its closed boundary, or non-finite are
    not moved, so local likelihoods on the track are unchanged."""
    line_environment, _ = line
    positions = np.array([0.0, 4.2, 10.0, 50.0, 95.0, 99.9, 100.0, np.nan])
    np.testing.assert_array_equal(
        snap_off_track_positions(positions, line_environment), positions
    )
    box_environment, _ = box_with_hole
    # Grid corners, the hole's closed boundary and a ring-bin boundary.
    positions = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [40.0, 50.0], [60.0, 60.0]]
    )
    np.testing.assert_array_equal(
        snap_off_track_positions(positions, box_environment), positions
    )


@pytest.mark.unit
def test_track_graph_arm_ends_stay_on_the_track():
    """On a linearized track the end of an arm, at the edge of an inter-arm
    gap bin, is on the track; a position inside the gap is not."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_node(2, pos=(10.0, 8.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    graph.add_edge(1, 2, distance=8.0, edge_id=1)
    environment = Environment(
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=3.0,
    ).fit_place_grid()
    # Linear positions: arm one spans [0, 10], the gap [10, 13], arm two [13, 21].
    on_track = np.array([[0.0], [10.0], [13.0], [21.0]])
    np.testing.assert_array_equal(
        snap_off_track_positions(on_track, environment), on_track
    )
    np.testing.assert_array_equal(
        snap_off_track_positions(np.array([[10.4], [12.8], [25.0]]), environment),
        [[9.5], [13.5], [20.5]],
    )
