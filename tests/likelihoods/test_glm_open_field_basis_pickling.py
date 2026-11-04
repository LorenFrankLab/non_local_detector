"""The GLM's spline basis without a track graph survives pickling unchanged.

patsy cannot pickle its design, so the basis is rebuilt after unpickling. The
rebuilt basis must give the same design matrix, bit for bit, as the one the
encoding model was fitted with, including when the fit dropped spline columns
without support on the environment.
"""

import copy
import pickle

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    make_spline_predict_matrix,
)

pytestmark = pytest.mark.unit

BIN_SIZE = 3.5
DT = 0.002


def _poisson_spikes(rng, time, position, centers):
    """Spike times of Gaussian place cells (6 cm SD, 15 Hz peak) at ``centers``."""
    spike_times = []
    for center in centers:
        distance = np.sum((position - center) ** 2, axis=1)
        rate = 0.3 + 15.0 * np.exp(-0.5 * distance / 6.0**2)
        spike_times.append(time[rng.random(len(time)) < rate * DT])
    return spike_times


def _line(rng):
    """Back-and-forth runs along a 0-100 cm line."""
    position = (50.0 + 49.0 * np.sin(np.arange(20_000) * DT * 2 * np.pi / 8.0))[:, None]
    return position, np.array([[20.0], [50.0], [80.0]])


def _box(rng):
    """A reflected random walk over a 60 cm square."""
    walk = np.cumsum(rng.normal(0.0, 0.5, (40_000, 2)), axis=0)
    position = 1.0 + np.abs((walk % 116.0) - 58.0)
    return position, np.array([[15.0, 15.0], [30.0, 45.0], [45.0, 20.0]])


def _plus_maze(rng):
    """Runs out each arm of a plus maze filling a fifth of a 100 cm box."""
    center = np.array([50.0, 50.0])
    ends = [[50.0, 0.0], [50.0, 100.0], [0.0, 50.0], [100.0, 50.0]]
    legs = []
    for _ in range(3):
        for end in ends:
            out = np.linspace(center, end, 3000)
            legs += [out, out[::-1]]
    position = np.concatenate(legs)
    along_x = np.abs(position[:, 1] - 50.0) < 1e-9
    jitter = rng.uniform(-4.0, 4.0, len(position))
    position[:, 1] += np.where(along_x, jitter, 0.0)
    position[:, 0] += np.where(along_x, 0.0, jitter)
    return position, np.array([[50.0, 15.0], [15.0, 50.0], [85.0, 50.0]])


@pytest.fixture(
    scope="module",
    params=[_line, _box, _plus_maze],
    ids=["1d-line", "2d-box", "2d-plus-maze"],
)
def encoding(request):
    """A sorted-spikes GLM fit, its environment, and the recording's geometry."""
    rng = np.random.default_rng(0)
    position, centers = request.param(rng)
    time = np.arange(len(position)) * DT
    environment = Environment(place_bin_size=BIN_SIZE).fit_place_grid(position)
    model = fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        _poisson_spikes(rng, time, position, centers),
        environment,
        disable_progress_bar=True,
    )
    return model, environment, request.param.__name__


def _random_interior_positions(environment, n_positions=2_000, seed=1):
    """Uniform positions within randomly chosen interior bins, and one NaN row."""
    rng = np.random.default_rng(seed)
    interior = environment.is_track_interior_.ravel()
    centers = environment.place_bin_centers_[interior]
    positions = centers[rng.integers(len(centers), size=n_positions)]
    positions = positions + rng.uniform(-BIN_SIZE / 2, BIN_SIZE / 2, positions.shape)
    return np.vstack([positions, np.full((1, positions.shape[1]), np.nan)])


@pytest.mark.parametrize(
    "round_trip",
    [lambda basis: pickle.loads(pickle.dumps(basis)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
def test_rebuilt_basis_matches_the_fitted_basis(encoding, round_trip):
    """After a round trip the basis evaluates to the identical design matrix."""
    model, environment, case = encoding
    basis = model["emission_design_info"]
    n_coefficients = np.asarray(model["coefficients"]).shape[1]
    if case == "_plus_maze":
        # The case exercises a basis restricted to its supported columns.
        assert n_coefficients < len(basis.design_info.column_names)
    positions = _random_interior_positions(environment)
    expected = np.asarray(make_spline_predict_matrix(basis, positions))
    assert expected.shape == (len(positions), n_coefficients)
    assert np.all(np.isnan(expected[-1]))
    assert np.all(np.isfinite(expected[:-1]))

    restored = round_trip(basis)

    actual = np.asarray(make_spline_predict_matrix(restored, positions))
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected, equal_nan=True)
