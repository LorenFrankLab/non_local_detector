"""The GLM's spline basis without a track graph survives pickling unchanged.

patsy cannot pickle its design, so the basis is rebuilt after unpickling. The
rebuilt basis must give the same design matrix, bit for bit, as the one the
encoding model was fitted with, including when the fit dropped spline columns
without support on the environment. A rebuild that differs from the fitted
design (as a NumPy or patsy upgrade, or a change to the package's knot
placement, could make it) raises instead of decoding,
and the inputs it is rebuilt from cannot be written to.
"""

import copy
import pickle

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import sorted_spikes_glm
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


def _unpickled(basis):
    return pickle.loads(pickle.dumps(basis))


@pytest.mark.parametrize("relative_change", [1e-4, -1e-5])
def test_rebuild_that_misses_the_fingerprint_raises(encoding, relative_change):
    """A saved fingerprint the rebuilt design does not reproduce raises on use,
    asking for a refit, instead of decoding with a different basis."""
    model, environment, _ = encoding
    restored = _unpickled(model["emission_design_info"])
    object.__setattr__(
        restored,
        "design_fingerprint",
        restored.design_fingerprint * (1.0 + relative_change),
    )
    with pytest.raises(ValidationError, match=r"NumPy or patsy version.*Refit"):
        make_spline_predict_matrix(restored, _random_interior_positions(environment))


def test_rebuild_from_shifted_knots_raises(encoding, monkeypatch):
    """The fingerprint detects a rebuilt design whose knots moved by 1e-6 cm,
    standing in for a version change in how patsy places them."""
    model, environment, _ = encoding
    restored = _unpickled(model["emission_design_info"])
    build = sorted_spikes_glm.make_spline_design_matrix
    monkeypatch.setattr(
        sorted_spikes_glm,
        "make_spline_design_matrix",
        lambda position, *args, **kwargs: build(position + 1e-6, *args, **kwargs),
    )
    with pytest.raises(ValidationError, match="differs from the one it was fitted"):
        make_spline_predict_matrix(restored, _random_interior_positions(environment))


def test_rebuild_with_a_different_knot_count_names_every_cause(encoding, monkeypatch):
    """A rebuild with more knots than the fit, as a change to this package's
    knot placement would give, reports the column counts, names both the
    library versions and the package's own basis code as causes, and asks
    for a refit or a decode with the versions that fitted the model."""
    model, environment, _ = encoding
    restored = _unpickled(model["emission_design_info"])
    inner_knots = sorted_spikes_glm._spline_inner_knots
    monkeypatch.setattr(
        sorted_spikes_glm,
        "_spline_inner_knots",
        lambda position, edges, spacing: inner_knots(position, edges, spacing / 2),
    )
    with pytest.raises(ValidationError) as error:
        make_spline_predict_matrix(restored, _random_interior_positions(environment))
    message = str(error.value)
    assert "columns instead of" in message
    assert "NumPy or patsy version" in message
    assert "non_local_detector code that builds the basis" in message
    assert "Refit the model" in message
    assert "versions that fitted it" in message


def test_rebuild_within_rounding_of_the_fingerprint_decodes(encoding):
    """A fingerprint off by float32 rounding still rebuilds the basis."""
    model, environment, _ = encoding
    basis = model["emission_design_info"]
    restored = _unpickled(basis)
    object.__setattr__(
        restored, "design_fingerprint", restored.design_fingerprint * (1.0 + 1e-7)
    )
    positions = _random_interior_positions(environment)
    assert np.array_equal(
        np.asarray(make_spline_predict_matrix(restored, positions)),
        np.asarray(make_spline_predict_matrix(basis, positions)),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "round_trip",
    [lambda basis: basis, _unpickled, copy.deepcopy],
    ids=["fitted", "pickle", "deepcopy"],
)
def test_basis_inputs_are_read_only(encoding, round_trip):
    """The arrays the design is rebuilt from cannot be changed in place, so a
    basis and its pickle cannot disagree."""
    basis = round_trip(encoding[0]["emission_design_info"])
    arrays = [basis.basis_positions, basis.place_bin_edges, basis.design_fingerprint]
    if basis.columns is not None:
        arrays.append(basis.columns)
    for array in arrays:
        with pytest.raises(ValueError, match="read-only"):
            array[(0,) * array.ndim] = 0
