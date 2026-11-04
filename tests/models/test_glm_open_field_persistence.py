"""Detectors fitted with a GLM likelihood without a track graph can be saved.

The open-field spline basis of ``sorted_spikes_glm`` and
``clusterless_mark_glm`` is a patsy design, which patsy cannot pickle. These
tests save, pickle and deep-copy fitted 1-D (no graph) and 2-D detectors,
including a 2-D maze whose fit drops spline columns, and require the copies to
decode exactly as the original does.
"""

import copy
import pickle

import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)

pytestmark = pytest.mark.integration

SAMPLING_FREQUENCY = 500.0
N_FIT = 10_000
N_PREDICT = 1_000


def _plus_maze_position(rng):
    """Runs out each arm of a plus maze that fills a fifth of a 100 cm box,
    with up to 4 cm of lateral jitter; shape (N_FIT, 2)."""
    center = np.array([50.0, 50.0])
    legs = []
    for end in ([50.0, 0.0], [50.0, 100.0], [0.0, 50.0], [100.0, 50.0]):
        out = np.linspace(center, np.array(end), N_FIT // 8)
        legs += [out, out[::-1]]
    position = np.concatenate(legs)
    along_x = np.abs(position[:, 1] - 50.0) < 1e-9
    jitter = rng.uniform(-4.0, 4.0, len(position))
    position[:, 1] += np.where(along_x, jitter, 0.0)
    position[:, 0] += np.where(along_x, 0.0, jitter)
    return position


def _recording(geometry, seed=0):
    """Positions and Gaussian place cells.

    Parameters
    ----------
    geometry : {"1d", "2d", "plus-maze"}
        Runs over 0-60 cm in 1-D or in each of two dimensions, or a 2-D plus
        maze.

    Returns
    -------
    time : np.ndarray, shape (N_FIT,)
    position : np.ndarray, shape (N_FIT, n_dims)
    spike_times : list of np.ndarray
        Spike times of each of four cells, in seconds.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(N_FIT) / SAMPLING_FREQUENCY
    if geometry == "plus-maze":
        position = _plus_maze_position(rng)
    else:
        n_dims = 1 if geometry == "1d" else 2
        phase = 2 * np.pi * time[:, None] / np.array([7.0, 11.0])[:n_dims]
        position = 30.0 + 28.0 * np.sin(phase + rng.uniform(0, 2 * np.pi, n_dims))
    centers = position[rng.integers(len(position), size=4)]
    distance = np.sum((position[:, None, :] - centers[None]) ** 2, axis=-1)
    rate = 30.0 * np.exp(-distance / (2 * 8.0**2)) + 0.5  # Hz
    is_spike = rng.random(rate.shape) < rate / SAMPLING_FREQUENCY
    spike_times = [time[is_spike[:, cell]] for cell in range(is_spike.shape[1])]
    return time, position, spike_times


@pytest.fixture(
    scope="module",
    params=[
        (algorithm, geometry)
        for algorithm in ("sorted_spikes_glm", "clusterless_mark_glm")
        for geometry in ("1d", "2d", "plus-maze")
    ],
    ids=lambda param: f"{param[0].split('_')[0]}-{param[1]}",
)
def fitted(request):
    """A fitted non-local detector, its spike data, and predict arguments."""
    algorithm, geometry = request.param
    time, position, spike_times = _recording(geometry)
    environment = Environment(place_bin_size=5.0)
    if algorithm == "sorted_spikes_glm":
        detector = NonLocalSortedSpikesDetector(
            environments=environment,
            sorted_spikes_algorithm=algorithm,
            sorted_spikes_algorithm_params={"disable_progress_bar": True},
        )
        spikes = (spike_times,)
    else:
        rng = np.random.default_rng(1)
        # One electrode; each cell's marks cluster around its own amplitude.
        electrode_times = np.concatenate(spike_times)
        order = np.argsort(electrode_times)
        features = np.concatenate(
            [
                np.full((len(times), 2), 10.0 * (cell + 1))
                + rng.normal(scale=1.0, size=(len(times), 2))
                for cell, times in enumerate(spike_times)
            ]
        )
        detector = NonLocalClusterlessDetector(
            environments=environment,
            clusterless_algorithm=algorithm,
            clusterless_algorithm_params={"disable_progress_bar": True},
        )
        spikes = ([electrode_times[order]], [features[order]])
    detector.fit(time, position, *spikes)
    (encoding_model,) = detector.encoding_model_.values()
    # The maze fills a small part of its box, so the fit drops spline columns.
    dropped_columns = encoding_model["emission_design_info"].columns is not None
    assert dropped_columns == (geometry == "plus-maze")
    end = time[N_PREDICT - 1]
    if algorithm == "sorted_spikes_glm":
        predict_spikes = ([times[times < end] for times in spikes[0]],)
    else:
        keep = spikes[0][0] < end
        predict_spikes = ([spikes[0][0][keep]], [spikes[1][0][keep]])
    predict_kwargs = {
        "time_edges": time_edges_from_centers(time[:N_PREDICT]),
        "position_time": time[:N_PREDICT],
        "position": position[:N_PREDICT],
        "return_outputs": "all",
    }
    expected = detector.predict(*predict_spikes, **predict_kwargs)
    return detector, predict_spikes, predict_kwargs, expected


def _assert_same_decode(restored, fitted):
    detector, predict_spikes, predict_kwargs, expected = fitted
    assert type(restored) is type(detector)
    actual = restored.predict(*predict_spikes, **predict_kwargs)
    # The Local and No-Spike states have a NaN position in the state_bins
    # MultiIndex, which an index comparison treats as unequal even between
    # two decodes of one detector; Index.equals treats NaNs as equal.
    assert actual.indexes["state_bins"].equals(expected.indexes["state_bins"])
    xr.testing.assert_identical(
        actual.reset_index("state_bins"), expected.reset_index("state_bins")
    )
    for name in expected.data_vars:
        assert np.array_equal(
            actual[name].to_numpy(), expected[name].to_numpy(), equal_nan=True
        ), name


def test_save_and_load_model_decodes_identically(fitted, tmp_path):
    """``save_model`` / ``load_model`` keep the fitted basis and decode exactly."""
    detector = fitted[0]
    path = tmp_path / "detector.pkl"
    detector.save_model(path)
    _assert_same_decode(detector.load_model(path), fitted)


def test_pickle_round_trip_decodes_identically(fitted):
    """``pickle.dumps`` / ``loads`` keep the fitted basis and decode exactly."""
    _assert_same_decode(pickle.loads(pickle.dumps(fitted[0])), fitted)


def test_deepcopy_decodes_identically(fitted):
    """``copy.deepcopy`` keeps the fitted basis and decodes exactly."""
    _assert_same_decode(copy.deepcopy(fitted[0]), fitted)
