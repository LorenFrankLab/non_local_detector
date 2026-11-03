"""The data contract between the detectors and every registered likelihood algorithm.

A detector stores the dict an algorithm's fit returns as its encoding model and
unpacks it into the algorithm's predict beside its own data and keyword
arguments (see ``LikelihoodAlgorithm``). Nothing checks the keys when an
algorithm is registered, so a mismatch would surface only as a ``TypeError`` at
decode time. These tests fit each registered algorithm through a detector, so
the keyword arguments checked are the ones the detector actually passes.
"""

import copy
import inspect
import pickle
from dataclasses import replace

import jax
import numpy as np
import pytest

from non_local_detector import time_edges_from_centers
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _CLUSTERLESS_PREDICT_KEYWORDS,
    _ENCODING_MODEL_KEYS,
    _PREDICT_KEYWORDS,
    _SORTED_SPIKES_ALGORITHMS,
    LikelihoodAlgorithm,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)
from tests.models._helpers import ALGORITHMS, _decoder

pytestmark = pytest.mark.unit

# The decode rows requested from the fitted decoder.
_DECODE_SAMPLES = slice(100, 140)


@pytest.fixture(scope="module")
def recording():
    """500 samples on a ramp over the calibration decoder's (-1.5, 1.5) track."""
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
    """A calibration decoder for ``algorithm`` and its spike data arguments."""
    time, position, spike_times, features = recording
    spike_data = (spike_times,)
    if algorithm in _CLUSTERLESS_ALGORITHMS:
        spike_data = (spike_times, features)
    decoder = _decoder(algorithm, calibration=True).fit(time, position, *spike_data)
    return decoder, spike_data


def _data_names(decoder):
    """The predict arguments the detector passes positionally."""
    return {"position_time", "position", *type(decoder)._SPIKE_DATA_NAMES}


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
    required = {
        name
        for name, p in parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    keys = set(encoding_model)
    violations = []
    if collisions := keys & (data_names | keywords):
        violations.append(f"keys the detector also passes: {sorted(collisions)}")
    if missing := required - keys - data_names - keywords:
        violations.append(
            f"required predict parameters not supplied: {sorted(missing)}"
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
    n_interior_bins = int(decoder.environments[0].is_track_interior_flat_.sum())
    local_kwargs = {**kwargs, "is_local": True}
    local = record.predict(*args, **local_kwargs)
    for likelihood, shape in [
        (non_local, (n_rows, n_interior_bins)),
        (local, (n_rows, 1)),
    ]:
        assert isinstance(likelihood, jax.Array)
        assert likelihood.dtype == np.float32
        assert likelihood.shape == shape


def test_contract_check_reports_a_broken_algorithm(monkeypatch, recording):
    """A fit whose keys break each rule is caught, one message per rule."""

    def broken_fit(*args, **kwargs):
        model = fit_sorted_spikes_kde_encoding_model(*args, **kwargs)
        del model["place_fields"], model["rate_units"]
        return {**model, "row_slice": None, "unread": 0}

    broken = LikelihoodAlgorithm(broken_fit, predict_sorted_spikes_kde_log_likelihood)
    monkeypatch.setitem(_SORTED_SPIKES_ALGORITHMS, "sorted_spikes_kde", broken)
    decoder, _ = _fit("sorted_spikes_kde", recording)
    (encoding_model,) = decoder.encoding_model_.values()

    violations = contract_violations(
        broken, encoding_model, _data_names(decoder), _PREDICT_KEYWORDS
    )
    assert violations == [
        "keys the detector also passes: ['row_slice']",
        "required predict parameters not supplied: ['place_fields']",
        "keys predict does not accept: ['unread']",
        "common keys missing: ['rate_units']",
    ]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_records_pickle_and_copy(algorithm):
    """Records survive pickling and deep copies, defaults still read-only."""
    record = _registry(algorithm)[0][algorithm]
    for restored in (pickle.loads(pickle.dumps(record)), copy.deepcopy(record)):
        assert restored == record
        with pytest.raises(TypeError):
            restored.default_params["block_size"] = 5  # type: ignore[index]


def test_clusterless_record_without_a_mark_scale_raises(monkeypatch, recording):
    """The detector names the algorithm instead of calling None."""
    record = _CLUSTERLESS_ALGORITHMS["clusterless_kde"]
    monkeypatch.setitem(
        _CLUSTERLESS_ALGORITHMS,
        "clusterless_kde",
        replace(record, mark_reference_scale=None),
    )
    with pytest.raises(ValidationError, match="without a mark_reference_scale"):
        _fit("clusterless_kde", recording)
