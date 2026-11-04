"""An untrusted formula stays unevaluated in every copy of a loaded model.

A saved model's transition formula outside the formula grammar, loaded
without ``trust_formulas``, is held as an inert ``UntrustedFormula`` (not a
string, so patsy cannot evaluate it). Copies, pickles and ``sklearn.clone``
carry it, so none of them, nor fitting, decoding or saving any of them, ever
evaluates it. The formulas call a sentinel that records any evaluation.
"""

import builtins
import copy
import json
import pickle
import warnings
import zipfile

import numpy as np
import pytest
from sklearn.base import clone

import non_local_detector as nld
from non_local_detector import time_edges_from_centers
from non_local_detector._formula_grammar import LoadedFormula, UntrustedFormula
from non_local_detector.exceptions import ValidationError
from tests.model_format.conftest import algorithm_params, decode

pytestmark = pytest.mark.integration

SENTINELS = [
    "1 + speed + nld_formula_sentinel(speed)",
    "1 + speed + __import__('builtins').nld_formula_sentinel(speed)",
]


@pytest.fixture
def calls(monkeypatch):
    """Record every evaluation of the sentinel the formulas call."""
    recorded = []

    def sentinel(x):
        recorded.append(1)
        return x

    monkeypatch.setattr(builtins, "nld_formula_sentinel", sentinel, raising=False)
    return recorded


def _speed(position, time):
    return np.abs(np.gradient(position, time))


def _detector(formula="1 + speed"):
    return nld.NonLocalSortedSpikesDetector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.full(4, 0.98), formula=formula
        ),
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    )


def _fit(detector, sorted_recording):
    time, position, spike_times = sorted_recording
    return detector.fit(
        time,
        position,
        spike_times,
        discrete_transition_covariate_data={"speed": _speed(position, time)},
    )


def _with_formula(path, formula):
    with zipfile.ZipFile(path) as source:
        members = [(item, source.read(item.filename)) for item in source.infolist()]
    with zipfile.ZipFile(path, "w") as target:
        for item, data in members:
            if item.filename == "config.json":
                config = json.loads(data)
                config["discrete_transition_type"]["state"]["formula"] = formula
                data = json.dumps(config).encode()
            target.writestr(item, data)


@pytest.fixture(params=[True, False], ids=["fitted", "unfitted"])
def loaded(request, sorted_recording, tmp_path, calls):
    """Load a saved model whose formula was edited to each sentinel."""

    def load(formula):
        detector = _detector()
        if request.param:
            _fit(detector, sorted_recording)
        path = tmp_path / "model.nld"
        detector.save_model(path)
        _with_formula(path, formula)
        with pytest.warns(UserWarning, match="trust_formulas=True") as warned:
            model = nld.NonLocalSortedSpikesDetector.load_model(path)
        assert len(warned) == 1
        return model, request.param

    return load


COPIES = {
    "itself": lambda model: model,
    "copy.copy": copy.copy,
    "copy.deepcopy": copy.deepcopy,
    "detector.copy": lambda model: model.copy(),
    "pickle": lambda model: pickle.loads(pickle.dumps(model)),
    "clone": clone,
    "clone-of-pickle": lambda model: clone(pickle.loads(pickle.dumps(model))),
}


@pytest.mark.parametrize("make_copy_id", list(COPIES))
@pytest.mark.parametrize("formula", SENTINELS)
def test_copies_stay_inert_and_refuse_fit_decode_and_save(
    formula, make_copy_id, loaded, sorted_recording, tmp_path, calls
):
    make_copy = COPIES[make_copy_id]
    model, was_fitted = loaded(formula)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        copied = make_copy(model)

    untrusted = copied.discrete_transition_type.formula
    assert isinstance(untrusted, UntrustedFormula)
    assert untrusted.text == formula
    time, position, spike_times = sorted_recording
    # A clone is unfitted; the other copies keep the fitted state.
    assert hasattr(copied, "encoding_model_") == (
        was_fitted and "clone" not in make_copy_id
    )
    if hasattr(copied, "encoding_model_"):
        assert copied.discrete_transition_design_matrix_ is None
        with pytest.raises(ValidationError, match="trust_formulas=True"):
            decode(
                copied,
                time,
                position,
                {"spike_times": spike_times},
                discrete_transition_covariate_data={"speed": np.zeros(300)},
            )
        with pytest.raises(ValidationError, match="trust_formulas=True"):
            copied.discrete_transition_intervals({"speed": np.zeros(5)})
        with pytest.raises(ValidationError, match="trust_formulas=True"):
            copied.plot_discrete_state_transition(covariate_data={"speed": np.zeros(5)})
    with pytest.raises(ValidationError, match="trust_formulas=True"):
        _fit(copied, sorted_recording)
    with pytest.raises(ValidationError, match="trust_formulas=True"):
        copied.estimate_parameters(
            time_edges=time_edges_from_centers(time),
            position_time=time,
            position=position,
            spike_times=spike_times,
            discrete_transition_covariate_data={"speed": _speed(position, time)},
            max_iter=1,
        )
    with pytest.raises(ValidationError, match="trust_formulas=True"):
        copied.save_model(tmp_path / "again.nld")
    assert calls == []


def test_a_refused_fit_keeps_the_loaded_model(loaded, sorted_recording):
    model, was_fitted = loaded(SENTINELS[0])
    environments = getattr(model, "environments_", None)

    with pytest.raises(ValidationError, match="trust_formulas=True"):
        _fit(model, sorted_recording)

    assert getattr(model, "environments_", None) is environments
    assert was_fitted == (environments is not None)


def test_a_formula_of_your_own_clears_the_taint(loaded, sorted_recording, calls):
    model, _ = loaded(SENTINELS[0])
    copied = clone(model)

    copied.discrete_transition_type.formula = "1 + speed"
    _fit(copied, sorted_recording)

    assert copied.discrete_transition_design_matrix_.shape[1] == 2
    assert calls == []


def test_an_in_grammar_file_formula_must_name_covariates_at_fit(
    sorted_recording, tmp_path
):
    """An unfitted file's formula is checked against the covariates when
    first fitted: patsy would otherwise read ``n_states`` from its frame."""
    path = tmp_path / "model.nld"
    _detector().save_model(path)
    _with_formula(path, "1 + speed + I(n_states * 0 + 1.5)")

    model = nld.NonLocalSortedSpikesDetector.load_model(path)

    assert isinstance(model.discrete_transition_type.formula, LoadedFormula)
    for candidate in (model, clone(model), pickle.loads(pickle.dumps(model))):
        with pytest.raises(ValidationError, match="not a covariate"):
            _fit(candidate, sorted_recording)


def test_a_trusted_file_formula_is_a_plain_string(sorted_recording, tmp_path):
    path = tmp_path / "model.nld"
    _fit(_detector("1 + np.tanh(speed)"), sorted_recording).save_model(path)

    model = nld.NonLocalSortedSpikesDetector.load_model(path, trust_formulas=True)

    assert type(model.discrete_transition_type.formula) is str
    _fit(clone(model), sorted_recording)
