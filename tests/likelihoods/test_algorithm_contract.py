"""The data contract between the detectors and every registered likelihood algorithm.

A detector stores the dict an algorithm's fit returns as its encoding model and
unpacks it into the algorithm's predict beside its own data and keyword
arguments (see ``LikelihoodAlgorithm``). Nothing checks the keys when an
algorithm is registered, so a mismatch would surface only as a ``TypeError`` at
decode time, or not at all. These tests fit each registered algorithm through a
detector, so the keyword arguments checked are the ones the detector actually
passes, and check the fit's keys against the ``TypedDict`` its record names.
"""

import copy
import inspect
import pickle
import subprocess
import sys
import typing
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import Environment, time_edges_from_centers
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    _ARRAY_ENCODING_KEYS,
    _CLUSTERLESS_ALGORITHMS,
    _CLUSTERLESS_PREDICT_KEYWORDS,
    _ENCODING_MODEL_KEYS,
    _PREDICT_KEYWORDS,
    _SORTED_SPIKES_ALGORITHMS,
    ClusterlessLikelihoodAlgorithm,
    LikelihoodAlgorithm,
)
from non_local_detector.likelihoods.encoding_model import (
    ClusterlessEncodingModel,
    SortedSpikesEncodingModel,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    SortedSpikesDiffusionEncodingModel,
    fit_sorted_spikes_diffusion_encoding_model,
    predict_sorted_spikes_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    SortedSpikesKDEEncodingModel,
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.model_format import read_model_summary
from non_local_detector.models.base import ClusterlessDetector
from tests.models._helpers import ALGORITHMS, _decoder, type_violations

pytestmark = pytest.mark.unit

# The decode rows requested from the fitted decoder.
_DECODE_SAMPLES = slice(100, 140)


@pytest.fixture(scope="module")
def recording():
    """500 samples on a ramp over (-1.4, 1.4): the first 3 of the 5 bins."""
    rng = np.random.default_rng(0)
    time = np.arange(500) * 0.002
    position = np.linspace(-1.4, 1.4, time.size)[:, None]
    spike_times = [np.sort(rng.uniform(time[0], time[-1], 60)) for _ in range(2)]
    features = [rng.normal(size=(times.size, 2)) for times in spike_times]
    return time, position, spike_times, features


def _registry(algorithm):
    if algorithm in _SORTED_SPIKES_ALGORITHMS:
        return _SORTED_SPIKES_ALGORITHMS, _PREDICT_KEYWORDS
    return _CLUSTERLESS_ALGORITHMS, _CLUSTERLESS_PREDICT_KEYWORDS


def _fit(algorithm, recording):
    """A calibration decoder for ``algorithm`` and its spike data arguments.

    The calibration grid is extended by two unvisited bins and the interior
    inferred, so the non-local shape check tells interior bins from all bins.
    """
    time, position, spike_times, features = recording
    spike_data = (spike_times,)
    if algorithm in _CLUSTERLESS_ALGORITHMS:
        spike_data = (spike_times, features)
    decoder = _decoder(algorithm, calibration=True)
    decoder.environments = [
        Environment(place_bin_size=1, position_range=((-1.5, 3.5),))
    ]
    decoder.infer_track_interior = True
    return decoder.fit(time, position, *spike_data), spike_data


def _data_names(decoder):
    """The predict arguments the detector passes positionally."""
    names = {"position_time", "position", "spike_times"}
    return (
        names | {"spike_waveform_features"}
        if isinstance(decoder, ClusterlessDetector)
        else names
    )


def _declared_keys(encoding_model_type):
    """Every key of an encoding-model ``TypedDict``."""
    return encoding_model_type.__required_keys__ | encoding_model_type.__optional_keys__


def contract_violations(
    record: LikelihoodAlgorithm,
    encoding_model: dict,
    data_names: set[str],
    keywords: frozenset[str],
) -> list[str]:
    """Each way ``encoding_model`` breaks the contract with ``record.predict``."""
    parameters = inspect.signature(record.predict).parameters
    takes_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    named = {
        name: p
        for name, p in parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    required = {n for n, p in named.items() if p.default is inspect.Parameter.empty}
    keys = set(encoding_model)
    supplied = keys | data_names | keywords
    declared = _declared_keys(record.encoding_model_type)
    violations = []
    if undeclared := keys - declared:
        violations.append(f"keys the encoding model type omits: {sorted(undeclared)}")
    if unreturned := declared - keys:
        violations.append(f"declared keys the fit omits: {sorted(unreturned)}")
    if collisions := keys & (data_names | keywords):
        violations.append(f"keys the detector also passes: {sorted(collisions)}")
    if missing := required - supplied:
        violations.append(
            f"required predict parameters not supplied: {sorted(missing)}"
        )
    # A renamed fit key leaves its predict parameter at its default, and a
    # predict taking ``**kwargs`` swallows the new name.
    if defaulted := set(named) - required - supplied:
        violations.append(
            f"optional predict parameters not supplied: {sorted(defaulted)}"
        )
    if not takes_var_keyword and (unaccepted := keys - set(parameters)):
        violations.append(f"keys predict does not accept: {sorted(unaccepted)}")
    if absent := _ENCODING_MODEL_KEYS - keys:
        violations.append(f"common keys missing: {sorted(absent)}")
    return violations


@pytest.mark.parametrize("algorithm", ALGORITHMS)
# The mark GLM initializes its mixture with k-means, which can find fewer
# distinct points than components on this tiny fixture.
@pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
)
# The sorted GLM's fit on this tiny fixture ends near its gradient tolerance,
# above it on some platforms; the test checks the fit/predict contract, not
# the fit's quality.
@pytest.mark.filterwarnings(
    "ignore:GLM Poisson regression may not have converged:UserWarning"
)
def test_registered_algorithm_meets_the_detector_contract(
    monkeypatch, recording, algorithm
):
    """Fit keys, predict parameters and detector arguments fit together.

    The predict call is recorded through the detector, so the keyword
    arguments checked are the ones it passes, and the output is checked for
    both the non-local and the local shape.
    """
    registry, keywords = _registry(algorithm)
    record = registry[algorithm]
    decoder, spike_data = _fit(algorithm, recording)
    (encoding_model,) = decoder.encoding_model_.values()
    data_names = _data_names(decoder)

    assert contract_violations(record, encoding_model, data_names, keywords) == []
    assert encoding_model["algorithm"] == algorithm
    assert type(encoding_model) is dict  # pickles and round-trips as a plain dict

    calls = []

    def spy(*args, **kwargs):
        result = record.predict(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setitem(registry, algorithm, replace(record, predict=spy))
    time, position, *_ = recording
    edges = time_edges_from_centers(time[_DECODE_SAMPLES])
    decoder.predict(
        *spike_data,
        time_edges=edges,
        position_time=time,
        position=position,
        return_outputs="log_likelihood",
    )
    ((args, kwargs, non_local),) = calls  # one state, so one predict call
    assert len(args) == len(data_names)
    assert set(kwargs) == set(encoding_model) | keywords

    n_rows = len(edges) - 1
    is_interior = decoder.environments_[0].is_track_interior_flat_
    n_interior_bins = int(is_interior.sum())
    assert 0 < n_interior_bins < is_interior.size
    local_kwargs = {**kwargs, "is_local": True}
    local = record.predict(*args, **local_kwargs)
    for likelihood, shape in [
        (non_local, (n_rows, n_interior_bins)),
        (local, (n_rows, 1)),
    ]:
        assert isinstance(likelihood, jax.Array)
        assert likelihood.dtype == np.float32
        assert likelihood.shape == shape


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
)
@pytest.mark.filterwarnings(
    "ignore:GLM Poisson regression may not have converged:UserWarning"
)
def test_predict_takes_no_position_when_not_local(monkeypatch, recording, algorithm):
    """A detector that reads no position passes predict None for both position
    arguments; the non-local likelihood needs neither, and the local one asks
    for them by name."""
    registry, _ = _registry(algorithm)
    record = registry[algorithm]
    decoder, spike_data = _fit(algorithm, recording)
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return record.predict(*args, **kwargs)

    monkeypatch.setitem(registry, algorithm, replace(record, predict=spy))
    time, *_ = recording
    edges = time_edges_from_centers(time[_DECODE_SAMPLES])
    decoder.predict(*spike_data, time_edges=edges, return_outputs="log_likelihood")

    ((args, kwargs),) = calls
    assert args[:2] == (None, None)
    assert kwargs["is_local"] is False
    with pytest.raises(ValidationError, match="position_time and position are None"):
        record.predict(*args, **{**kwargs, "is_local": True})


def test_contract_check_reports_a_broken_algorithm(monkeypatch, recording):
    """A fit whose keys break each rule is caught, one message per rule."""

    def broken_fit(*args, **kwargs):
        model = fit_sorted_spikes_kde_encoding_model(*args, **kwargs)
        del model["place_fields"], model["rate_units"]
        return {**model, "row_slice": None, "unread": 0}

    broken = LikelihoodAlgorithm(
        broken_fit,
        predict_sorted_spikes_kde_log_likelihood,
        SortedSpikesKDEEncodingModel,
    )
    monkeypatch.setitem(_SORTED_SPIKES_ALGORITHMS, "sorted_spikes_kde", broken)
    decoder, _ = _fit("sorted_spikes_kde", recording)
    (encoding_model,) = decoder.encoding_model_.values()

    violations = contract_violations(
        broken, encoding_model, _data_names(decoder), _PREDICT_KEYWORDS
    )
    assert violations == [
        "keys the encoding model type omits: ['row_slice', 'unread']",
        "declared keys the fit omits: ['place_fields', 'rate_units']",
        "keys the detector also passes: ['row_slice']",
        "required predict parameters not supplied: ['place_fields']",
        "optional predict parameters not supplied: ['rate_units']",
        "keys predict does not accept: ['unread']",
        "common keys missing: ['rate_units']",
    ]


def test_contract_check_reports_a_renamed_key(monkeypatch, recording):
    """A renamed fit key breaks the type, and predict neither takes nor gets it."""

    def renamed_fit(*args, **kwargs):
        model = fit_sorted_spikes_diffusion_encoding_model(*args, **kwargs)
        model["interior_log_place_field"] = model.pop("interior_log_place_fields")
        return model

    renamed = LikelihoodAlgorithm(
        renamed_fit,
        predict_sorted_spikes_diffusion_log_likelihood,
        SortedSpikesDiffusionEncodingModel,
    )
    monkeypatch.setitem(_SORTED_SPIKES_ALGORITHMS, "sorted_spikes_diffusion", renamed)
    decoder, _ = _fit("sorted_spikes_diffusion", recording)
    (encoding_model,) = decoder.encoding_model_.values()

    violations = contract_violations(
        renamed, encoding_model, _data_names(decoder), _PREDICT_KEYWORDS
    )
    assert violations == [
        "keys the encoding model type omits: ['interior_log_place_field']",
        "declared keys the fit omits: ['interior_log_place_fields']",
        "optional predict parameters not supplied: ['interior_log_place_fields']",
        "keys predict does not accept: ['interior_log_place_field']",
    ]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
)
@pytest.mark.filterwarnings(
    "ignore:GLM Poisson regression may not have converged:UserWarning"
)
def test_encoding_model_values_have_their_declared_types(
    tmp_path, recording, algorithm
):
    """Each value has its declared type, as fitted and as loaded from a pickle
    or a saved model."""
    record = _registry(algorithm)[0][algorithm]
    decoder, _ = _fit(algorithm, recording)
    decoder.save_model(tmp_path / "model.nld")
    for detector in (
        decoder,
        pickle.loads(pickle.dumps(decoder)),
        type(decoder).load_model(tmp_path / "model.nld"),
    ):
        (encoding_model,) = detector.encoding_model_.values()
        assert type_violations(encoding_model, record.encoding_model_type) == []


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
)
@pytest.mark.filterwarnings(
    "ignore:GLM Poisson regression may not have converged:UserWarning"
)
def test_values_saved_with_their_old_types_load_unified(tmp_path, recording, algorithm):
    """Models saved before the shared keys held NumPy arrays (Python floats in
    a list, or a JAX array) load with them as arrays of the same dtype and
    values, and decode as the new fit does."""
    decoder, spike_data = _fit(algorithm, recording)
    time, position, *_ = recording
    edges = time_edges_from_centers(time[_DECODE_SAMPLES])

    def log_likelihood(detector):
        return detector.predict(
            *spike_data,
            time_edges=edges,
            position_time=time,
            position=position,
            return_outputs="log_likelihood",
        ).log_likelihood.values

    expected = log_likelihood(decoder)
    (key,) = decoder.encoding_model_
    fitted = decoder.encoding_model_[key]
    shared = [name for name in _ARRAY_ENCODING_KEYS if name in fitted]
    old = dict(fitted)
    for name in shared:
        value = fitted[name]
        old[name] = value.tolist() if value.dtype == np.float64 else jnp.asarray(value)
    decoder.encoding_model_ = {key: old}
    decoder.save_model(tmp_path / "model.nld")
    for loaded in (
        pickle.loads(pickle.dumps(decoder)),
        type(decoder).load_model(tmp_path / "model.nld"),
    ):
        (model,) = loaded.encoding_model_.values()
        for name in shared:
            assert type(model[name]) is np.ndarray, name
            assert model[name].dtype == fitted[name].dtype, name
            np.testing.assert_array_equal(model[name], fitted[name])
        np.testing.assert_array_equal(log_likelihood(loaded), expected)
    (summary,) = read_model_summary(tmp_path / "model.nld")["encoding_models"]
    for name in shared:
        if name in summary:  # the summary omits large arrays
            assert type(summary[name]) is np.ndarray, name
            np.testing.assert_array_equal(summary[name], fitted[name])


def test_type_check_reports_a_wrong_value():
    """A value of another type than declared is reported, element types too."""

    class Model(typing.TypedDict):
        rates: np.ndarray
        counts: list[int]
        scale: float | None

    model = {"rates": [1.0], "counts": [1, 2.5], "scale": 3}
    assert type_violations(model, Model) == [
        "rates: declared <class 'numpy.ndarray'>, got list",
        "counts: declared list[int], got list",
    ]


# Keys several algorithms store whose values mean different things, so each
# algorithm declares its own type.
_PER_ALGORITHM_KEY_TYPES = {
    # Each algorithm's own fitted density model.
    "gpi_models",
    "occupancy_model",
    # The fit's bandwidth parameters, whose accepted forms differ: the KDE
    # stores position_std as a per-dimension vector, diffusion as a scalar
    # (it rejects per-dimension values).
    "position_std",
    "waveform_std",
}


def test_shared_keys_have_one_declared_type():
    """A key several algorithms store is declared with the same type in each."""
    declared: dict[str, dict[str, object]] = {}
    for algorithm in ALGORITHMS:
        record = _registry(algorithm)[0][algorithm]
        hints = typing.get_type_hints(record.encoding_model_type)
        for key, annotation in hints.items():
            declared.setdefault(key, {})[algorithm] = annotation
    differing = {
        key: by_algorithm
        for key, by_algorithm in declared.items()
        if len(set(map(str, by_algorithm.values()))) > 1
    }
    assert set(differing) == _PER_ALGORITHM_KEY_TYPES, differing


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_record_names_the_type_its_fit_returns(algorithm):
    """The fit is annotated with the record's TypedDict, with its family's keys."""
    record = _registry(algorithm)[0][algorithm]
    assert typing.get_type_hints(record.fit)["return"] is record.encoding_model_type
    family = (
        SortedSpikesEncodingModel
        if algorithm in _SORTED_SPIKES_ALGORITHMS
        else ClusterlessEncodingModel
    )
    # A TypedDict supports no subclass check; its keys include its bases'.
    assert family.__required_keys__ <= record.encoding_model_type.__required_keys__
    assert family.__required_keys__ >= _ENCODING_MODEL_KEYS


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_model_parameters_are_keyword_only(algorithm):
    """After the data arguments, fit and predict take their parameters by keyword.

    A catch-all ``**kwargs`` predict would hide a renamed key, so none takes one.
    """
    record = _registry(algorithm)[0][algorithm]
    spike_data = ["spike_times"]
    if algorithm in _CLUSTERLESS_ALGORITHMS:
        spike_data.append("spike_waveform_features")
    data = ["position_time", "position", *spike_data]
    for function, positional in [
        (record.predict, data),
        (record.fit, [*data, "environment", "weights"]),
    ]:
        parameters = inspect.signature(function).parameters
        assert list(parameters)[: len(positional)] == positional
        kinds = {p.kind for name, p in parameters.items() if name not in positional}
        assert kinds == {inspect.Parameter.KEYWORD_ONLY}, function.__name__


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_execution_settings_are_recorded_fit_parameters_of_predict(algorithm):
    """Each execution setting is a settable fit parameter the fit records and
    predict reads, so a detector can pass predict its current value."""
    record = _registry(algorithm)[0][algorithm]
    assert "disable_progress_bar" in record.execution_settings
    fit_parameters = inspect.signature(record.fit).parameters
    predict_parameters = inspect.signature(record.predict).parameters
    declared = _declared_keys(record.encoding_model_type)
    for name in record.execution_settings:
        assert fit_parameters[name].default is not inspect.Parameter.empty, name
        assert name in declared, name
        assert name in predict_parameters, name


# Execution settings other than the fitted ones, chosen to change each
# algorithm's blocking or tiling on the small recording.
_CHANGED_SETTINGS = {
    "clusterless_kde": {
        "block_size": 7,
        "enc_tile_size": 16,
        "pos_tile_size": 2,
        "use_streaming": True,
    },
    "clusterless_gmm": {"spike_block_size": 7, "bin_tile_size": 2},
    # Far below the dense kernel, so the kernel is built one column per tile.
    "clusterless_diffusion": {"block_size": 7, "memory_budget": 64},
    "clusterless_mark_glm": {"spike_block_size": 7},
}
# A fit parameter that is not an execution setting, changed after fitting:
# prediction must ignore it until the next fit.
_CHANGED_FIT_PARAMETER = {
    "sorted_spikes_glm": {"emission_knot_spacing": 0.5},
    "sorted_spikes_kde": {"position_std": 0.3},
    "sorted_spikes_diffusion": {"position_std": 0.3},
    "sorted_spikes_mrf": {"penalty": 10.0},
    "clusterless_kde": {"position_std": 0.3},
    "clusterless_gmm": {"gmm_components_joint": 2},
    "clusterless_diffusion": {"position_std": 0.3},
    "clusterless_mark_glm": {"n_mark_components": 2},
}
# An invalid value of each algorithm's execution settings.
_INVALID_SETTINGS = {
    "clusterless_kde": [
        {"block_size": 0},
        {"enc_tile_size": -1},
        {"use_streaming": True, "enc_tile_size": None},
    ],
    "clusterless_gmm": [{"spike_block_size": 0}, {"bin_tile_size": 0}],
    "clusterless_diffusion": [{"block_size": 0}, {"memory_budget": 0}],
    "clusterless_mark_glm": [{"spike_block_size": 0}],
}


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
)
@pytest.mark.filterwarnings(
    "ignore:GLM Poisson regression may not have converged:UserWarning"
)
def test_predict_reads_execution_settings_changed_after_fitting(
    monkeypatch, recording, algorithm
):
    """Settings set in the algorithm params after fitting reach predict without a
    refit; settings the params do not name keep their fitted values, and other
    fit parameters wait for the next fit."""
    registry, _ = _registry(algorithm)
    record = registry[algorithm]
    decoder, spike_data = _fit(algorithm, recording)
    (encoding_model,) = decoder.encoding_model_.values()
    fitted = {name: encoding_model[name] for name in record.execution_settings}
    received = []

    def spy(*args, **kwargs):
        received.append({name: kwargs[name] for name in record.execution_settings})
        return record.predict(*args, **kwargs)

    monkeypatch.setitem(registry, algorithm, replace(record, predict=spy))
    time, position, *_ = recording
    edges = time_edges_from_centers(time[_DECODE_SAMPLES])

    def log_likelihood():
        return decoder.predict(
            *spike_data,
            time_edges=edges,
            position_time=time,
            position=position,
            return_outputs="log_likelihood",
        ).log_likelihood

    before = log_likelihood()
    assert received == [fitted]

    changed = _CHANGED_SETTINGS.get(algorithm, {})
    assert set(changed) <= record.execution_settings
    name = (
        "sorted_spikes_algorithm_params"
        if algorithm in _SORTED_SPIKES_ALGORITHMS
        else "clusterless_algorithm_params"
    )
    params = getattr(decoder, name)
    if algorithm == "clusterless_diffusion":
        import non_local_detector.likelihoods.clusterless_diffusion as diffusion

        n_bins = encoding_model["occupancy"].shape[0]
        rank = int(encoding_model["resolved_rank"])
        assert diffusion._dense_kernel_fits(n_bins, encoding_model["memory_budget"])
        assert not diffusion._dense_kernel_fits(n_bins, changed["memory_budget"])
        assert diffusion._source_tile_size(changed["memory_budget"], n_bins, rank) == 1
    setattr(decoder, name, {**params, **changed})
    after = log_likelihood()
    assert received[-1] == {**fitted, **changed}
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-4)

    setattr(decoder, name, {**params, **_CHANGED_FIT_PARAMETER[algorithm]})
    np.testing.assert_array_equal(log_likelihood(), before)
    assert received[-1] == fitted


@pytest.mark.parametrize(
    ("algorithm", "invalid"),
    [
        (algorithm, invalid)
        for algorithm, cases in _INVALID_SETTINGS.items()
        for invalid in cases
    ],
)
def test_invalid_execution_settings_are_rejected_at_fit_and_after_it(
    recording, algorithm, invalid
):
    """The fit's checks also apply to settings changed after fitting, so a bad
    value raises ValidationError instead of failing deep in predict."""
    decoder, spike_data = _fit(algorithm, recording)
    params = decoder.clusterless_algorithm_params
    decoder.clusterless_algorithm_params = {**params, **invalid}
    time, position, *_ = recording
    with pytest.raises(ValidationError):
        decoder.predict(
            *spike_data,
            time_edges=time_edges_from_centers(time[_DECODE_SAMPLES]),
            position_time=time,
            position=position,
        )
    with pytest.raises(ValidationError):
        decoder.fit(time, position, *spike_data)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_records_pickle_and_copy(algorithm):
    """Records survive pickling and deep copies, defaults still read-only."""
    record = _registry(algorithm)[0][algorithm]
    for restored in (pickle.loads(pickle.dumps(record)), copy.deepcopy(record)):
        assert restored == record
        with pytest.raises(TypeError):
            restored.default_params["block_size"] = 5  # type: ignore[index]


def mark_scale_default_mismatches(record: ClusterlessLikelihoodAlgorithm) -> list[str]:
    """Each ``mark_reference_scale`` default that differs from its fit's.

    A detector whose params leave the mark bandwidth unset calls the scale
    without it, while the fit uses its own default, so the two must agree or
    the mark reference is built for a kernel the fit does not use.
    """
    fit_parameters = inspect.signature(record.fit).parameters
    mismatches = []
    for name, parameter in inspect.signature(
        record.mark_reference_scale
    ).parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if name not in fit_parameters:
            mismatches.append(f"{name}: not a fit parameter")
        elif not np.array_equal(parameter.default, fit_parameters[name].default):
            mismatches.append(
                f"{name}: scale default {parameter.default!r}, "
                f"fit default {fit_parameters[name].default!r}"
            )
    return mismatches


@pytest.mark.parametrize("algorithm", sorted(_CLUSTERLESS_ALGORITHMS))
def test_mark_reference_scale_defaults_match_the_fit(algorithm):
    """With no params, the mark reference uses the fit's own mark bandwidth."""
    record = _CLUSTERLESS_ALGORITHMS[algorithm]
    assert mark_scale_default_mismatches(record) == []
    fit_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(record.fit).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    np.testing.assert_array_equal(
        record.mark_reference_scale(), record.mark_reference_scale(**fit_defaults)
    )


def test_mark_scale_check_reports_a_changed_default():
    """A scale default that drifted from the fit's, or that names no fit
    parameter, is caught."""

    def drifted(waveform_std=12.0, **_fit_params):
        return waveform_std

    def misnamed(mark_std=24.0, **_fit_params):
        return mark_std

    record = _CLUSTERLESS_ALGORITHMS["clusterless_kde"]
    assert mark_scale_default_mismatches(
        replace(record, mark_reference_scale=drifted)
    ) == ["waveform_std: scale default 12.0, fit default 24.0"]
    assert mark_scale_default_mismatches(
        replace(record, mark_reference_scale=misnamed)
    ) == ["mark_std: not a fit parameter"]


@pytest.mark.parametrize("algorithm", sorted(_CLUSTERLESS_ALGORITHMS))
def test_model_checking_intensity_names_a_class(algorithm):
    """A record's model-checking intensity names a class model checking has."""
    from non_local_detector.model_checking import _clusterless_models

    name = _CLUSTERLESS_ALGORITHMS[algorithm].electrode_intensity
    if name is not None:
        assert isinstance(getattr(_clusterless_models, name), type)


@pytest.mark.parametrize(
    "document", ["README.md", "docs/getting-started/core-concepts.md"]
)
def test_user_docs_list_every_algorithm(document):
    """The hand-written algorithm lists name every registered algorithm."""
    path = Path(__file__).parents[2] / document
    if not path.is_file():
        pytest.skip(f"{document} is not part of this installation")
    text = path.read_text()
    assert [name for name in ALGORITHMS if f"`{name}`" not in text] == []


def test_likelihoods_do_not_import_model_checking():
    """Model checking imports the registry; the registry must not import it."""
    code = (
        "import sys, non_local_detector.likelihoods; "
        "assert not [m for m in sys.modules if 'model_checking' in m]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
