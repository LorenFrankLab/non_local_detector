"""Any patsy formula works in a session.

A transition formula the user writes is their own code, so fitting, decoding,
transition intervals, the transition plot and unpickling evaluate it with
patsy as written, including formulas outside the grammar ``load_model``
applies to formulas read from a saved model file
(``tests/model_format/test_formula_safety.py``).
"""

import pickle

import matplotlib as mpl
import numpy as np
import pytest

from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
)

mpl.use("Agg")
import matplotlib.pyplot as plt

pytestmark = pytest.mark.unit

N_FIT = 61
# The time bins of checkpoint_recording's predict time_edges.
N_DECODE = 100


def clipped(x):
    """A user-defined transform, passed to patsy with the covariates."""
    return np.clip(x, 1.0, 5.0)


def _covariates(n):
    phase = np.linspace(0.0, 2 * np.pi, n)
    return {
        "phase": phase,
        "arm": np.array(["a", "b", "c"])[np.arange(n) % 3],
    }


# (formula, the covariates it uses); the last two are outside the grammar.
FORMULAS = {
    "numpy-sin": ("1 + np.sin(phase) + np.cos(phase)", ["phase"]),
    "contrast": ("1 + C(arm, Treatment('b'))", ["arm"]),
    "outside-grammar": ("1 + np.tanh(phase) + I(phase % 2.0)", ["phase"]),
    "user-function": ("1 + clipped(phase)", ["phase", "clipped"]),
}


def _data(names, n):
    covariates = {**_covariates(n), "clipped": clipped}
    return {name: covariates[name] for name in names}


@pytest.fixture(params=list(FORMULAS))
def fitted(request, checkpoint_recording):
    """A small detector fitted with a covariate transition, and decode inputs."""
    formula, names = FORMULAS[request.param]
    model, fit, predict = checkpoint_recording("sorted")
    model.discrete_transition_type = DiscreteNonStationaryDiagonal(
        np.array([0.75, 0.85, 0.9, 0.95]), formula=formula
    )
    model.fit(**fit, discrete_transition_covariate_data=_data(names, N_FIT))
    predict["discrete_transition_covariate_data"] = _data(names, N_DECODE)
    return model, predict, names


def test_formula_fits_and_decodes(fitted):
    model, predict, _ = fitted
    n_columns = model.discrete_transition_design_matrix_.shape[1]

    results = model.predict(**predict)

    assert model.discrete_transition_design_matrix_.shape[0] == N_FIT
    assert n_columns > 1
    probabilities = results.acausal_state_probabilities.to_numpy()
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)


def test_formula_gives_intervals_and_plots(fitted):
    model, _, names = fitted
    covariates = _data(names, 5)
    n_states = len(model.state_names)
    n_parameters = model.discrete_transition_coefficients_[:, 0, :].size
    # A zero covariance gives zero-width intervals around the transitions.
    model.discrete_transition_coefficient_covariance_ = np.zeros(
        (n_states, n_parameters, n_parameters)
    )

    transitions, lower, upper = model.discrete_transition_intervals(covariates)
    # The plot draws the transitions against every entry of covariate_data,
    # so it cannot take a function among them.
    if "clipped" not in names:
        model.plot_discrete_state_transition(covariate_data=covariates)
        plt.close("all")

    assert transitions.shape == (5, n_states, n_states)
    np.testing.assert_allclose(transitions.sum(axis=-1), 1.0, rtol=1e-10)
    np.testing.assert_allclose(lower, transitions, rtol=1e-10)
    np.testing.assert_allclose(upper, transitions, rtol=1e-10)


def test_unpickled_detector_with_any_formula_rebuilds_its_design_and_decodes(
    fitted,
):
    model, predict, names = fitted
    if "clipped" in names:
        pytest.skip("covariates holding a test-module function do not pickle")
    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(
        np.asarray(restored.discrete_transition_design_matrix_),
        np.asarray(model.discrete_transition_design_matrix_),
    )
    np.testing.assert_array_equal(
        restored.predict(**predict).acausal_posterior.to_numpy(),
        model.predict(**predict).acausal_posterior.to_numpy(),
    )


@pytest.mark.parametrize(
    "transition",
    [
        lambda formula: DiscreteNonStationaryDiagonal(np.full(4, 0.9), formula),
        lambda formula: DiscreteNonStationaryCustom(np.full((4, 4), 0.25), formula),
    ],
    ids=["diagonal", "custom"],
)
@pytest.mark.parametrize("formula", [formula for formula, _ in FORMULAS.values()])
def test_any_formula_constructs(transition, formula):
    assert transition(formula).formula == formula


def test_plot_without_covariate_data_says_it_is_required(checkpoint_recording):
    model, fit, _ = checkpoint_recording("sorted")
    model.discrete_transition_type = DiscreteNonStationaryDiagonal(
        np.array([0.75, 0.85, 0.9, 0.95]), formula="1 + np.sin(phase)"
    )
    model.fit(**fit, discrete_transition_covariate_data=_data(["phase"], N_FIT))
    with pytest.raises(ValueError, match="covariate_data is required"):
        model.plot_discrete_state_transition()
    plt.close("all")
