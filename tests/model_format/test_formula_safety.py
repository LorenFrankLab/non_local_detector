"""A saved model's transition formula never reaches patsy unchecked.

A file whose stored formula was edited to one outside the formula grammar
loads for inspection only, as an unpickled detector with such a formula does:
``load_model`` warns, never calls patsy, and the loaded detector refuses to
decode; ``read_model_summary`` still reads it.
"""

import json
import zipfile

import numpy as np
import patsy
import pytest

import non_local_detector as nld
from non_local_detector import discrete_state_transitions as dst
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import sorted_spikes_glm
from non_local_detector.model_format import read_model_summary
from tests.model_format.conftest import algorithm_params, decode

pytestmark = pytest.mark.integration

ATTACKS = ["1 + C(speed, breakpoint)", "1 + __import__('os').system('x')"]


def _patsy_must_not_run(*args, **kwargs):
    msg = "patsy was asked to evaluate a stored formula"
    raise AssertionError(msg)


@pytest.fixture
def no_patsy(monkeypatch):
    """Fail the test if anything builds a patsy design."""
    for module in (patsy, dst):
        monkeypatch.setattr(module, "dmatrix", _patsy_must_not_run)
        monkeypatch.setattr(module, "build_design_matrices", _patsy_must_not_run)
    monkeypatch.setattr(sorted_spikes_glm, "dmatrix", _patsy_must_not_run)


@pytest.fixture
def fitted(sorted_recording):
    time, position, spike_times = sorted_recording
    speed = np.abs(np.gradient(position, time))
    detector = nld.NonLocalSortedSpikesDetector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.full(4, 0.98), formula="1 + speed"
        ),
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    )
    return detector.fit(
        time, position, spike_times, discrete_transition_covariate_data={"speed": speed}
    )


def _with_formula(path, formula):
    """Rewrite the stored formula of the saved model at ``path``.

    The documents carry no checksum (only arrays do), so the edited file is
    otherwise valid and only the formula check can refuse it.
    """
    with zipfile.ZipFile(path) as source:
        members = [(item, source.read(item.filename)) for item in source.infolist()]
    with zipfile.ZipFile(path, "w") as target:
        for item, data in members:
            if item.filename == "config.json":
                config = json.loads(data)
                config["discrete_transition_type"]["state"]["formula"] = formula
                data = json.dumps(config).encode()
            target.writestr(item, data)


@pytest.mark.parametrize("formula", ATTACKS)
def test_a_refused_stored_formula_loads_for_inspection_only(
    formula, fitted, sorted_recording, tmp_path, no_patsy
):
    time, position, spike_times = sorted_recording
    path = tmp_path / "model.nld"
    fitted.save_model(path)
    _with_formula(path, formula)

    with pytest.warns(UserWarning, match="decoding is refused"):
        loaded = nld.NonLocalSortedSpikesDetector.load_model(path)
    summary = read_model_summary(path)

    assert loaded.discrete_transition_type.formula == formula
    assert loaded.discrete_transition_design_matrix_ is None
    np.testing.assert_array_equal(
        loaded.discrete_transition_coefficients_,
        fitted.discrete_transition_coefficients_,
    )
    assert summary["config"]["discrete_transition_type"]["formula"] == formula
    covariates = {"speed": np.zeros(300)}
    with pytest.raises(ValidationError, match="formula was refused"):
        decode(
            loaded,
            time,
            position,
            {"spike_times": spike_times},
            discrete_transition_covariate_data=covariates,
        )
    with pytest.raises(ValidationError, match="formula was refused"):
        loaded.save_model(tmp_path / "again.nld")


def test_an_allowed_stored_formula_rebuilds_the_design(fitted, tmp_path):
    path = tmp_path / "model.nld"
    fitted.save_model(path)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    np.testing.assert_array_equal(
        loaded.discrete_transition_design_matrix_,
        fitted.discrete_transition_design_matrix_,
    )
