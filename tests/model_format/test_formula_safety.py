"""A saved model's transition formula reaches patsy only if it is checked or trusted.

A file whose stored formula is outside the formula grammar (written so, or
edited to an attack) loads for inspection only unless
``load_model(..., trust_formulas=True)``: ``load_model`` warns, never calls
patsy, and the loaded detector refuses to decode; ``read_model_summary``
still reads it. With ``trust_formulas=True`` it loads and decodes as the
original. Covariates are stored as numeric or string ``.npy`` arrays
(categoricals as codes and categories).
"""

import json
import pickle
import zipfile

import numpy as np
import pandas as pd
import patsy
import pytest

import non_local_detector as nld
from non_local_detector import discrete_state_transitions as dst
from non_local_detector._formula_grammar import LoadedFormula, UntrustedFormula
from non_local_detector.exceptions import ParameterWarning, ValidationError
from non_local_detector.likelihoods import sorted_spikes_glm
from non_local_detector.model_format import read_model_summary
from tests.model_format._compare import assert_same_fitted_state
from tests.model_format.conftest import SORTED_RATE, algorithm_params, decode
from tests.model_format.test_refusals import _edit_json

pytestmark = pytest.mark.integration

ATTACKS = ["1 + C(speed, breakpoint)", "1 + __import__('os').system('x')"]
# Every evaluation of SENTINEL_FORMULA appends to SENTINEL_CALLS.
SENTINEL_CALLS: list[int] = []
SENTINEL_FORMULA = (
    "1 + I(__import__('tests.model_format.test_formula_safety', fromlist=['_'])"
    ".SENTINEL_CALLS.append(1) or speed)"
)
# Outside the formula grammar, but a formula a user may fit with.
OUTSIDE_GRAMMAR = "1 + np.tanh(speed) + I(speed % 2.0)"


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


def _fit(sorted_recording, formula):
    time, position, spike_times = sorted_recording
    speed = np.abs(np.gradient(position, time))
    detector = nld.NonLocalSortedSpikesDetector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.full(4, 0.98), formula=formula
        ),
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
        sampling_frequency=SORTED_RATE,
    )
    return detector.fit(
        time, position, spike_times, discrete_transition_covariate_data={"speed": speed}
    )


@pytest.fixture
def fitted(sorted_recording):
    return _fit(sorted_recording, "1 + speed")


@pytest.fixture
def fitted_outside_grammar(sorted_recording):
    return _fit(sorted_recording, OUTSIDE_GRAMMAR)


def _decode(detector, sorted_recording):
    time, position, spike_times = sorted_recording
    return decode(
        detector,
        time,
        position,
        {"spike_times": spike_times},
        discrete_transition_covariate_data={"speed": np.linspace(0.0, 5.0, 300)},
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
    path = tmp_path / "model.nld"
    fitted.save_model(path)
    _with_formula(path, formula)

    with pytest.warns(ParameterWarning, match="trust_formulas=True") as warned:
        loaded = nld.NonLocalSortedSpikesDetector.load_model(path)
    summary = read_model_summary(path)

    assert loaded.discrete_transition_type.formula == UntrustedFormula(formula)
    assert loaded.discrete_transition_design_matrix_ is None
    np.testing.assert_array_equal(
        loaded.discrete_transition_coefficients_,
        fitted.discrete_transition_coefficients_,
    )
    assert summary["config"]["discrete_transition_type"]["formula"] == formula
    assert len(warned) == 1
    assert repr(formula) in str(warned[0].message)
    with pytest.raises(ValidationError, match="trust_formulas=True") as refused:
        _decode(loaded, sorted_recording)
    assert repr(formula) in str(refused.value)
    with pytest.raises(ValidationError, match="outside the formula grammar"):
        loaded.save_model(tmp_path / "again.nld")


def test_a_formula_outside_the_grammar_is_not_evaluated_by_default(
    fitted_outside_grammar, sorted_recording, tmp_path, monkeypatch
):
    path = tmp_path / "model.nld"
    fitted_outside_grammar.save_model(path)
    for module in (patsy, dst):
        monkeypatch.setattr(module, "dmatrix", _patsy_must_not_run)
        monkeypatch.setattr(module, "build_design_matrices", _patsy_must_not_run)

    with pytest.warns(ParameterWarning, match="trust_formulas=True") as warned:
        loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    assert len(warned) == 1
    assert repr(OUTSIDE_GRAMMAR) in str(warned[0].message)
    assert loaded.discrete_transition_design_matrix_ is None
    with pytest.raises(ValidationError, match="trust_formulas=True"):
        _decode(loaded, sorted_recording)


def test_a_trusted_formula_outside_the_grammar_loads_and_decodes_identically(
    fitted_outside_grammar, sorted_recording, tmp_path
):
    path = tmp_path / "model.nld"
    fitted_outside_grammar.save_model(path)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path, trust_formulas=True)

    assert_same_fitted_state(fitted_outside_grammar, loaded)
    for a, b in zip(
        _decode(loaded, sorted_recording),
        _decode(fitted_outside_grammar, sorted_recording),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_a_pickle_ignores_trust_formulas(
    fitted_outside_grammar, sorted_recording, tmp_path
):
    """A pickle can run any code when loaded, so its formula is not checked."""
    path = tmp_path / "model.pkl"
    with path.open("wb") as file:
        pickle.dump(fitted_outside_grammar, file)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    for a, b in zip(
        _decode(loaded, sorted_recording),
        _decode(fitted_outside_grammar, sorted_recording),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_an_allowed_stored_formula_rebuilds_the_design(fitted, tmp_path):
    path = tmp_path / "model.nld"
    fitted.save_model(path)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    np.testing.assert_array_equal(
        loaded.discrete_transition_design_matrix_,
        fitted.discrete_transition_design_matrix_,
    )


def _task(n):
    return np.where(np.arange(n) < n // 2, "run", "rest")


@pytest.mark.parametrize(
    "make_covariates",
    [
        lambda task: {"task": task},
        lambda task: {"task": task.astype(object)},
        lambda task: {"task": pd.Categorical(task, categories=["run", "rest"])},
        lambda task: pd.DataFrame({"task": task}),
        lambda task: pd.DataFrame(
            {"task": pd.Categorical(task, categories=["run", "rest"], ordered=True)}
        ),
        lambda task: {"task": pd.Series(task, name="task")},
        lambda task: {
            "task": pd.Series(pd.Categorical(task, categories=["run", "rest"]))
        },
    ],
    ids=[
        "unicode",
        "object-strings",
        "categorical",
        "frame",
        "frame-categorical",
        "series",
        "series-categorical",
    ],
)
def test_string_and_categorical_covariates_round_trip(
    make_covariates, sorted_recording, tmp_path
):
    """Categories of a ``C(task)`` covariate keep their values, order and
    type, so the reloaded design (whose reference level follows the category
    order) and its decode are identical."""
    time, position, spike_times = sorted_recording
    detector = nld.NonLocalSortedSpikesDetector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.full(4, 0.98), formula="1 + C(task)"
        ),
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
        sampling_frequency=SORTED_RATE,
    ).fit(
        time,
        position,
        spike_times,
        discrete_transition_covariate_data=make_covariates(_task(len(time))),
    )
    path = tmp_path / "model.nld"

    detector.save_model(path)
    loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    assert_same_fitted_state(detector, loaded)
    covariates = make_covariates(_task(300))
    for a, b in zip(
        decode(
            loaded,
            time,
            position,
            {"spike_times": spike_times},
            discrete_transition_covariate_data=covariates,
        ),
        decode(
            detector,
            time,
            position,
            {"spike_times": spike_times},
            discrete_transition_covariate_data=covariates,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_covariates_that_are_not_numbers_or_strings_are_refused(fitted, tmp_path):
    fitted._discrete_transition_covariate_data_ = {
        "speed": np.array([1.0, None], dtype=object)
    }

    with pytest.raises(ValidationError, match="object array"):
        fitted.save_model(tmp_path / "object.nld")


def _refit(detector, sorted_recording):
    time, position, spike_times = sorted_recording
    speed = np.abs(np.gradient(position, time))
    return detector.fit(
        time, position, spike_times, discrete_transition_covariate_data={"speed": speed}
    )


def test_a_formula_given_to_a_fitted_model_needs_a_refit_to_save(
    fitted_outside_grammar, tmp_path
):
    path = tmp_path / "model.nld"
    fitted_outside_grammar.save_model(path)
    with pytest.warns(ParameterWarning, match="trust_formulas=True"):
        loaded = nld.NonLocalSortedSpikesDetector.load_model(path)
    loaded.discrete_transition_type.formula = "1 + speed"

    with pytest.raises(ValidationError, match="changed after it was fitted"):
        loaded.save_model(tmp_path / "again.nld")


@pytest.mark.parametrize("formula", ATTACKS)
@pytest.mark.parametrize("fit_before_saving", [True, False], ids=["fitted", "unfitted"])
def test_a_model_loaded_for_inspection_refuses_to_fit(
    formula, fit_before_saving, sorted_recording, tmp_path, monkeypatch
):
    """Fitting would evaluate the file's formula, so it is refused too."""
    detector = _fit(sorted_recording, "1 + speed")
    if not fit_before_saving:
        detector = nld.NonLocalSortedSpikesDetector(**detector.get_params(deep=False))
    path = tmp_path / "model.nld"
    detector.save_model(path)
    _with_formula(path, formula)
    for module in (patsy, dst):
        monkeypatch.setattr(module, "dmatrix", _patsy_must_not_run)
        monkeypatch.setattr(module, "build_design_matrices", _patsy_must_not_run)

    with pytest.warns(ParameterWarning, match="trust_formulas=True"):
        loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    with pytest.raises(ValidationError, match="trust_formulas=True") as refused:
        _refit(loaded, sorted_recording)
    assert repr(formula) in str(refused.value)


def test_a_fitted_file_without_stored_covariates_still_marks_its_formula(
    fitted, sorted_recording, tmp_path
):
    """A fitted file whose covariates are null and whose transitions are
    stored, as a stationary transition's are, cannot carry a formula past the
    check: loading marks it, and refitting refuses it instead of evaluating
    it. patsy is not patched out, so an evaluation would run the payload."""
    path = tmp_path / "model.nld"
    fitted.save_model(path)
    _with_formula(path, SENTINEL_FORMULA)
    n_states = len(fitted.state_names)

    def stationary(state):
        state["discrete_transition_covariate_data"] = None
        state["discrete_state_transitions"] = np.full(
            (n_states, n_states), 1 / n_states
        ).tolist()

    _edit_json(path, "state.json", stationary)
    SENTINEL_CALLS.clear()

    with pytest.warns(ParameterWarning, match="trust_formulas=True"):
        loaded = nld.NonLocalSortedSpikesDetector.load_model(path)
    with pytest.raises(ValidationError, match="trust_formulas=True"):
        _refit(loaded, sorted_recording)

    assert loaded.discrete_transition_type.formula == UntrustedFormula(SENTINEL_FORMULA)
    assert SENTINEL_CALLS == []


def test_the_sentinel_formula_runs_when_evaluated(sorted_recording):
    """The payload of the test above does run if patsy evaluates it."""
    SENTINEL_CALLS.clear()

    _fit(sorted_recording, SENTINEL_FORMULA)

    assert SENTINEL_CALLS


def test_an_unfitted_model_with_a_trusted_formula_loads_without_warning(
    sorted_recording, tmp_path
):
    detector = nld.NonLocalSortedSpikesDetector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.full(4, 0.98), formula=OUTSIDE_GRAMMAR
        ),
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
        sampling_frequency=SORTED_RATE,
    )
    path = tmp_path / "model.nld"
    detector.save_model(path)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path, trust_formulas=True)

    _refit(loaded, sorted_recording)
    assert loaded.discrete_transition_type.formula == OUTSIDE_GRAMMAR


def test_a_stored_spline_size_beyond_the_caps_is_not_evaluated(
    fitted, sorted_recording, tmp_path, no_patsy, monkeypatch
):
    """A file cannot ask for an unbounded design: a spline of ten million
    columns is refused before patsy builds it, or its spline basis."""
    path = tmp_path / "model.nld"
    fitted.save_model(path)
    _with_formula(path, "1 + bs(speed, df=10000000)")
    monkeypatch.setattr(patsy.splines.BS, "memorize_chunk", _patsy_must_not_run)
    monkeypatch.setattr(patsy.splines.BS, "transform", _patsy_must_not_run)

    with pytest.warns(ParameterWarning, match="trust_formulas=True"):
        loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    assert loaded.discrete_transition_type.formula == UntrustedFormula(
        "1 + bs(speed, df=10000000)"
    )
    with pytest.raises(ValidationError, match="trust_formulas=True"):
        _decode(loaded, sorted_recording)


def test_a_stored_spline_within_the_caps_rebuilds(sorted_recording, tmp_path):
    formula = "1 + quantile_bs(speed, n_knots=8)"
    detector = _fit(sorted_recording, formula)
    path = tmp_path / "model.nld"
    detector.save_model(path)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(path)

    assert isinstance(loaded.discrete_transition_type.formula, LoadedFormula)
    np.testing.assert_array_equal(
        loaded.discrete_transition_design_matrix_,
        detector.discrete_transition_design_matrix_,
    )
