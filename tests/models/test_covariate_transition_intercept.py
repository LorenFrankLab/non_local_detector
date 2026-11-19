"""A detector refuses to fit a covariate transition formula without an intercept.

The initial coefficients put the configured matrix's logits in the first
coefficient row, which only reproduces that matrix when the first design
column is the intercept. Fitting refuses other formulas; a detector fitted
with one before the check existed still restores and decodes as it did.
"""

import pickle
import warnings

import numpy as np
import pytest

import non_local_detector.models.base as detector_base
from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
)
from tests.conftest import em_stops_at_max_iter

pytestmark = pytest.mark.unit

DIAGONAL = np.array([0.75, 0.85, 0.9, 0.95])


def _transition(transition_class, formula):
    if transition_class is DiscreteNonStationaryDiagonal:
        return DiscreteNonStationaryDiagonal(DIAGONAL, formula=formula)
    off_diagonal = (1.0 - DIAGONAL) / 3.0
    values = np.tile(off_diagonal[:, np.newaxis], (1, 4))
    np.fill_diagonal(values, DIAGONAL)
    return DiscreteNonStationaryCustom(values, formula=formula)


def _speed(n):
    return {"speed": np.linspace(0.0, 3.0, n)}


@pytest.mark.parametrize(
    "transition_class", [DiscreteNonStationaryDiagonal, DiscreteNonStationaryCustom]
)
@pytest.mark.parametrize("formula", ["0 + speed", "speed - 1"])
@pytest.mark.parametrize("method", ["fit", "estimate_parameters"])
@em_stops_at_max_iter  # a refusal that did not happen would run max_iter=1
def test_fitting_a_formula_without_an_intercept_is_refused(
    checkpoint_recording, transition_class, formula, method
):
    model, fit, predict = checkpoint_recording("sorted")
    model.discrete_transition_type = _transition(transition_class, formula)
    if method == "fit":
        arguments = {**fit, "discrete_transition_covariate_data": _speed(61)}
    else:
        n_decode = len(predict["time_edges"]) - 1
        arguments = {
            **predict,
            "discrete_transition_covariate_data": _speed(n_decode),
            "max_iter": 1,
        }
    before = dict(vars(model))

    with pytest.raises(ValueError, match="intercept") as error:
        getattr(model, method)(**arguments)

    assert "'1 + speed'" in str(error.value)
    assert "['speed']" in str(error.value)
    if method == "estimate_parameters":
        # EM is atomic, so a refused estimate leaves the detector as it was.
        assert vars(model).keys() == before.keys()


@pytest.mark.parametrize("route", ["saved model", "pickle"])
def test_a_detector_fitted_without_an_intercept_still_restores_and_decodes(
    checkpoint_recording, monkeypatch, tmp_path, route
):
    """Restoring rebuilds the design but does not refit, so it is not refused."""
    model, fit, predict = checkpoint_recording("sorted")
    model.discrete_transition_type = _transition(
        DiscreteNonStationaryDiagonal, "0 + speed"
    )
    predict["discrete_transition_covariate_data"] = _speed(
        len(predict["time_edges"]) - 1
    )
    path = tmp_path / "model.nld"
    # A detector fitted and stored by a release without the check.
    with monkeypatch.context() as patch:
        patch.setattr(
            detector_base, "_require_transition_intercept", lambda *args: None
        )
        model.fit(**fit, discrete_transition_covariate_data=_speed(61))
        # Decoding before saving warns about nothing, so a warning below
        # comes from restoring the detector.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expected = model.predict(**predict).acausal_posterior.to_numpy()
        model.save_model(path)
        pickled = pickle.dumps(model)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = (
            type(model).load_model(path)
            if route == "saved model"
            else pickle.loads(pickled)
        )
        actual = restored.predict(**predict).acausal_posterior.to_numpy()

    np.testing.assert_array_equal(actual, expected)
