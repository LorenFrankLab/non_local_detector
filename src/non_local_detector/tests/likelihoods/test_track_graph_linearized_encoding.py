"""Encoding on a track graph from 2-D tracking equals encoding on linear position.

With a track graph, the KDE and GMM fits linearize 2-D positions (the
occupancy samples and the positions at spike times) onto the graph. That must
give the same model as fitting the already-linearized position on a plain
1-D grid with the same bins: same place fields or ground-process intensity,
occupancy, and non-local log likelihood.

The track is an L (40 cm then 30 cm, no gap between edges), so the second
edge's linear position is not either 2-D coordinate. Spikes fall on tracking
samples, so the position at each spike is exact for both inputs, and the
linear input is the graph's own linearization of the samples, so the KDE
fits agree bit for bit.
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _CLUSTERLESS_MARK_REFERENCE_SCALES,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.tests.likelihoods.conftest import training_mark_references

pytestmark = pytest.mark.unit

FS = 500.0
SPEED = 20.0
LEG_X, LEG_Y = 40.0, 30.0
TRACK = LEG_X + LEG_Y
FIELD_CENTERS = np.array([10.0, 35.0, 45.0, 62.0])  # linear cm, both legs
FIT_PARAMS = {
    "sorted_spikes_kde": {"position_std": 4.0},
    "clusterless_kde": {"position_std": 4.0, "waveform_std": 10.0},
    "clusterless_gmm": {"gmm_components_occupancy": 8, "gmm_components_joint": 16},
}


def _corner_track_environment():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(LEG_X, 0.0))
    graph.add_node(2, pos=(LEG_X, LEG_Y))
    graph.add_edge(0, 1, distance=LEG_X, edge_id=0)
    graph.add_edge(1, 2, distance=LEG_Y, edge_id=1)
    return Environment(
        place_bin_size=2.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
    ).fit_place_grid()


@pytest.fixture(scope="module")
def corner_track():
    """Back-and-forth runs on the L; 2-D and linear positions of each sample."""
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 40.0, 1 / FS)
    phase = (time * SPEED) % (2 * TRACK)
    linear = np.where(phase < TRACK, phase, 2 * TRACK - phase)
    xy = np.where(
        (linear < LEG_X)[:, None],
        np.c_[linear, np.zeros_like(linear)],
        np.c_[np.full_like(linear, LEG_X), linear - LEG_X],
    )
    rates = 0.5 + 30.0 * np.exp(-0.5 * ((linear[:, None] - FIELD_CENTERS) / 4.0) ** 2)
    # One spike per sample at most, placed on the sample time.
    spike_times = [time[rng.random(time.size) < rate / FS] for rate in rates.T]
    marks = [
        rng.normal(30.0 * cell, 5.0, (times.size, 2))
        for cell, times in enumerate(spike_times)
    ]
    # Two electrodes of two cells each.
    electrode_times, electrode_marks = [], []
    for cells in ((0, 2), (1, 3)):
        times = np.concatenate([spike_times[c] for c in cells])
        order = np.argsort(times, kind="stable")
        electrode_times.append(times[order])
        electrode_marks.append(np.concatenate([marks[c] for c in cells])[order])

    graph_environment = _corner_track_environment()
    linear_environment = Environment(
        place_bin_size=2.0, position_range=((0.0, TRACK),)
    ).fit_place_grid(linear[:, None], infer_track_interior=False)
    np.testing.assert_array_equal(
        graph_environment.place_bin_centers_, linear_environment.place_bin_centers_
    )
    linearized = graph_environment._linear_position(xy)
    np.testing.assert_allclose(linearized, linear, atol=1e-10)
    return {
        "time": time,
        "xy": xy,
        "linear": linearized[:, None],
        "spike_times": spike_times,
        "electrode_times": electrode_times,
        "electrode_marks": electrode_marks,
        "graph_environment": graph_environment,
        "linear_environment": linear_environment,
        "time_edges": np.arange(0.0, time[-1], 0.02),
    }


def _fit_and_predict(algorithm, data, environment, position):
    """Fit on ``position`` with ``environment``; non-local log likelihood too."""
    params = FIT_PARAMS[algorithm]
    if algorithm in _SORTED_SPIKES_ALGORITHMS:
        fit, predict = _SORTED_SPIKES_ALGORITHMS[algorithm]
        spikes = (data["spike_times"],)
        encoding = fit(
            data["time"],
            position,
            *spikes,
            environment,
            disable_progress_bar=True,
            **params,
        )
        extra = {}
    else:
        fit, predict = _CLUSTERLESS_ALGORITHMS[algorithm]
        spikes = (data["electrode_times"], data["electrode_marks"])
        encoding = fit(
            data["time"],
            position,
            *spikes,
            environment,
            disable_progress_bar=True,
            **params,
        )
        extra = {
            "mark_reference": training_mark_references(
                data["time"],
                *spikes,
                _CLUSTERLESS_MARK_REFERENCE_SCALES[algorithm](**params),
                position=position,
            )
        }
    log_likelihood = predict(
        position_time=data["time"],
        position=position,
        spike_times=spikes[0],
        **({"spike_waveform_features": spikes[1]} if len(spikes) > 1 else {}),
        time_edges=data["time_edges"],
        is_local=False,
        **{**encoding, **extra, "disable_progress_bar": True},
    )
    return encoding, np.asarray(log_likelihood)


@pytest.mark.parametrize(
    ("algorithm", "fitted_key", "occupancy_key"),
    [
        ("sorted_spikes_kde", "place_fields", "occupancy"),
        ("clusterless_kde", "summed_ground_process_intensity", "occupancy"),
        ("clusterless_gmm", "summed_ground_process_intensity", "log_occupancy"),
    ],
)
def test_two_dimensional_fit_on_graph_equals_linear_fit(
    algorithm, fitted_key, occupancy_key, corner_track
):
    from_xy, xy_log_likelihood = _fit_and_predict(
        algorithm, corner_track, corner_track["graph_environment"], corner_track["xy"]
    )
    from_linear, linear_log_likelihood = _fit_and_predict(
        algorithm,
        corner_track,
        corner_track["linear_environment"],
        corner_track["linear"],
    )
    # The KDE fits are bit-identical. The GMM fit is not: refitting the same
    # input in one process moved its log occupancy by up to 1.1e-3 relative
    # (measured), and it runs EM to a 1e-3 tolerance. A linearization error
    # moves these quantities by O(1).
    tolerance = (
        {"rtol": 5e-3, "atol": 0.0}
        if algorithm == "clusterless_gmm"
        else {"rtol": 0.0, "atol": 0.0}
    )
    np.testing.assert_allclose(
        np.asarray(from_xy[occupancy_key]),
        np.asarray(from_linear[occupancy_key]),
        **tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(from_xy[fitted_key]),
        np.asarray(from_linear[fitted_key]),
        **tolerance,
    )
    np.testing.assert_allclose(xy_log_likelihood, linear_log_likelihood, **tolerance)
    # Non-trivial: the reference varies along the track.
    fitted = np.asarray(from_linear[fitted_key])
    assert np.ptp(fitted) > 0.5 * fitted.mean()
