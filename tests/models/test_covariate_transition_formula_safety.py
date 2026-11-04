"""A transition formula outside the allowed grammar never reaches patsy.

patsy evaluates formulas as Python. Each place that hands a formula, or a
design built from one, to patsy checks it first: constructing a transition,
fitting, unpickling (which rebuilds the design), decoding on new covariates,
transition intervals and the transition plot. The patsy calls are replaced by
sentinels that fail the test if they run, so no exploit is executed.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytest

from non_local_detector import discrete_state_transitions as dst
from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    predict_discrete_state_transition_intervals,
    predict_discrete_state_transitions,
)
from non_local_detector.exceptions import ValidationError

pytestmark = pytest.mark.unit

ATTACK = "1 + __import__('os').system('echo formula-allow-list')"


def _patsy_must_not_run(*args, **kwargs):
    msg = "patsy was asked to evaluate a refused formula"
    raise AssertionError(msg)


@pytest.fixture
def no_patsy(monkeypatch):
    """Replace every patsy evaluation the transitions use with a sentinel."""
    monkeypatch.setattr(dst, "dmatrix", _patsy_must_not_run)
    monkeypatch.setattr(dst, "build_design_matrices", _patsy_must_not_run)


@pytest.fixture
def fitted(checkpoint_recording):
    """A small detector fitted with a covariate transition, and decode inputs."""
    model, fit, predict = checkpoint_recording("sorted")
    model.discrete_transition_type = DiscreteNonStationaryDiagonal(
        np.array([0.75, 0.85, 0.9, 0.95]), formula="1 + speed"
    )
    model.fit(
        **fit, discrete_transition_covariate_data={"speed": np.linspace(0, 3, 61)}
    )
    n_decode = len(predict["time_edges"]) - 1
    predict["discrete_transition_covariate_data"] = {"speed": np.zeros(n_decode)}
    return model, fit, predict


def _tamper(design_matrix):
    """Make the design's stored factor code an attack."""
    (factor,) = design_matrix.design_info.factor_infos
    factor.code = "__import__('os').system('echo formula-allow-list')"


@pytest.mark.parametrize(
    "transition",
    [
        lambda formula: DiscreteNonStationaryDiagonal(np.full(4, 0.9), formula),
        lambda formula: DiscreteNonStationaryCustom(np.full((4, 4), 0.25), formula),
    ],
    ids=["diagonal", "custom"],
)
def test_refused_at_construction(transition, no_patsy):
    with pytest.raises(ValidationError, match="not allowed"):
        transition(ATTACK)


def test_refused_when_set_before_fit(checkpoint_recording, no_patsy):
    model, fit, _ = checkpoint_recording("sorted")
    model.discrete_transition_type = DiscreteNonStationaryDiagonal(
        np.array([0.75, 0.85, 0.9, 0.95]), formula="1 + speed"
    )
    model.discrete_transition_type.formula = ATTACK
    with pytest.raises(ValidationError, match="not allowed"):
        model.fit(
            **fit, discrete_transition_covariate_data={"speed": np.linspace(0, 3, 61)}
        )


def test_refused_by_make_state_transition(no_patsy):
    transition = DiscreteNonStationaryCustom(np.full((4, 4), 0.25), "1 + speed")
    transition.formula = ATTACK
    with pytest.raises(ValidationError, match="not allowed"):
        transition.make_state_transition({"speed": np.linspace(0, 3, 61)})


def test_refused_at_decode_when_set_on_a_fitted_detector(fitted, no_patsy):
    model, _, predict = fitted
    model.discrete_transition_type.formula = ATTACK
    with pytest.raises(ValidationError, match="not allowed"):
        model.predict(**predict)


def test_unpickled_detector_with_a_refused_formula_loads_but_does_not_decode(
    fitted, monkeypatch
):
    """A saved detector whose formula is refused still loads and can be
    inspected; its design is not rebuilt, and decoding is refused."""
    model, _, predict = fitted
    model.discrete_transition_type.formula = ATTACK
    saved = pickle.dumps(model)

    monkeypatch.setattr(dst, "dmatrix", _patsy_must_not_run)
    monkeypatch.setattr(dst, "build_design_matrices", _patsy_must_not_run)
    with pytest.warns(UserWarning, match="decoding with covariates is refused"):
        restored = pickle.loads(saved)

    assert restored.discrete_transition_design_matrix_ is None
    np.testing.assert_array_equal(
        restored.discrete_transition_coefficients_,
        model.discrete_transition_coefficients_,
    )
    np.testing.assert_array_equal(
        restored.discrete_state_transitions_, model.discrete_state_transitions_
    )
    with pytest.raises(ValidationError, match="not allowed"):
        restored.predict(**predict)


def test_unpickled_detector_with_an_allowed_formula_rebuilds_its_design(fitted):
    model, _, _ = fitted
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        np.asarray(restored.discrete_transition_design_matrix_),
        np.asarray(model.discrete_transition_design_matrix_),
    )


def test_predicting_transitions_refuses_a_tampered_design(monkeypatch):
    transition = DiscreteNonStationaryDiagonal(np.full(3, 0.9), "1 + speed")
    _, coefficients, design_matrix = transition.make_state_transition(
        {"speed": np.linspace(0, 3, 20)}
    )
    _tamper(design_matrix)
    monkeypatch.setattr(dst, "build_design_matrices", _patsy_must_not_run)
    covariates = {"speed": np.linspace(0, 3, 5)}

    with pytest.raises(ValidationError, match="not allowed"):
        predict_discrete_state_transitions(design_matrix, coefficients, covariates)
    with pytest.raises(ValidationError, match="not allowed"):
        predict_discrete_state_transition_intervals(
            design_matrix, coefficients, np.zeros((3, 6, 6)), covariates
        )


def test_intervals_and_plot_refuse_a_tampered_design(fitted, monkeypatch):
    model, _, _ = fitted
    model.discrete_transition_coefficient_covariance_ = np.zeros((4, 12, 12))
    _tamper(model.discrete_transition_design_matrix_)
    monkeypatch.setattr(dst, "build_design_matrices", _patsy_must_not_run)
    covariates = {"speed": np.linspace(0, 3, 5)}

    with pytest.raises(ValidationError, match="not allowed"):
        model.discrete_transition_intervals(covariates)
    with pytest.raises(ValidationError, match="not allowed"):
        model.plot_discrete_state_transition(covariate_data=covariates)
    plt.close("all")


def test_builtin_contrast_is_refused_at_construction(no_patsy):
    """``C(x, breakpoint)`` would call the builtin as a contrast."""
    with pytest.raises(ValidationError, match="not allowed"):
        DiscreteNonStationaryDiagonal(np.full(2, 0.9), "1 + C(speed, breakpoint)")


def test_name_that_is_not_a_covariate_is_refused_at_fit(no_patsy):
    """patsy would resolve ``exit`` to the Python builtin."""
    transition = DiscreteNonStationaryDiagonal(np.full(2, 0.9), "1 + exit")
    with pytest.raises(ValidationError, match="not a covariate"):
        transition.make_state_transition({"speed": np.linspace(0, 3, 20)})


def test_decode_refuses_covariates_missing_a_formula_name(fitted, no_patsy):
    model, _, predict = fitted
    n_decode = len(predict["time_edges"]) - 1
    predict["discrete_transition_covariate_data"] = {"velocity": np.zeros(n_decode)}
    with pytest.raises(ValidationError, match="not a covariate"):
        model.predict(**predict)


def test_predicting_transitions_refuses_a_tampered_eval_code(monkeypatch):
    """build_design_matrices evaluates the stored ``eval_code``."""
    transition = DiscreteNonStationaryDiagonal(np.full(3, 0.9), "1 + speed")
    _, coefficients, design_matrix = transition.make_state_transition(
        {"speed": np.linspace(0, 3, 20)}
    )
    design_info = design_matrix.design_info
    (factor,) = design_info.factor_infos
    design_info.factor_infos[factor].state["eval_code"] = (
        "(open('formula-allow-list', 'w'), speed)[1]"
    )
    monkeypatch.setattr(dst, "build_design_matrices", _patsy_must_not_run)
    with pytest.raises(ValidationError, match="eval_code"):
        predict_discrete_state_transitions(
            design_matrix, coefficients, {"speed": np.linspace(0, 3, 5)}
        )
