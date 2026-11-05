"""Each likelihood algorithm's default fit parameters, and their validation.

A detector whose ``*_algorithm_params`` is ``None`` passes its algorithm's own
defaults to the encoding fit; a user dict replaces those defaults. Every key is
checked against the registered fit's signature, so a misspelled or foreign
parameter raises instead of being dropped or ignored.
"""

import functools
import inspect
import pickle
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from non_local_detector import time_edges_from_centers
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.models import ClusterlessDecoder, SortedSpikesDecoder
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from tests.models._helpers import CLUSTERLESS_SIM_RATE

pytestmark = pytest.mark.unit

# What reaches each fit when the user passes nothing. Bandwidths are left to
# the fits' own defaults; sorted KDE keeps the detector's larger block size.
EXPECTED_DEFAULTS = {
    "sorted_spikes_glm": {},
    "sorted_spikes_kde": {"block_size": 10_000},
    "sorted_spikes_diffusion": {},
    "sorted_spikes_mrf": {},
    "clusterless_kde": {},
    "clusterless_gmm": {},
    "clusterless_diffusion": {},
    "clusterless_mark_glm": {},
}

# Fit arguments the detector supplies itself, not from the algorithm params.
_DETECTOR_SUPPLIED = {
    "position_time",
    "position",
    "spike_times",
    "spike_waveform_features",
    "environment",
    "weights",
    "_encoding_support",
}


class _StopFit(Exception):
    """Raised by the spy so a test records the fit's arguments without fitting."""


@pytest.fixture(scope="module")
def sim():
    return make_simulated_run_data(
        n_tetrodes=3, place_field_means=np.arange(0, 90, 15), n_runs=1, seed=0
    )


def _environment():
    return Environment(place_bin_size=5.0)


def _sorted_decoder(algorithm, params=None):
    return SortedSpikesDecoder(
        environments=[_environment()],
        sorted_spikes_algorithm=algorithm,
        sorted_spikes_algorithm_params=params,
        sampling_frequency=CLUSTERLESS_SIM_RATE,
    )


def _clusterless_decoder(algorithm, params=None):
    return ClusterlessDecoder(
        environments=[_environment()],
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params=params,
        sampling_frequency=CLUSTERLESS_SIM_RATE,
    )


def _fit(decoder, sim):
    if isinstance(decoder, ClusterlessDecoder):
        return decoder.fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
        )
    return decoder.fit(sim.position_time, sim.position, sim.spike_times)


def _record_fit_params(monkeypatch, registry, algorithm, decoder, sim):
    """The algorithm params the detector passes to the registered fit."""
    fit = registry[algorithm].fit
    calls = []

    @functools.wraps(fit)
    def spy(*args, **kwargs):
        calls.append(kwargs)
        raise _StopFit

    monkeypatch.setitem(registry, algorithm, replace(registry[algorithm], fit=spy))
    with pytest.raises(_StopFit):
        _fit(decoder, sim)
    (kwargs,) = calls
    return {k: v for k, v in kwargs.items() if k not in _DETECTOR_SUPPLIED}


def test_default_table_is_pinned():
    """Every registered algorithm's defaults are pinned."""
    assert {
        name: dict(record.default_params)
        for name, record in {
            **_SORTED_SPIKES_ALGORITHMS,
            **_CLUSTERLESS_ALGORITHMS,
        }.items()
    } == EXPECTED_DEFAULTS


def test_registered_defaults_are_read_only():
    """Neither a record nor its defaults can be changed in place."""
    record = _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"]
    with pytest.raises(TypeError):
        record.default_params["block_size"] = 5  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        record.default_params = {}  # type: ignore[misc]


@pytest.mark.parametrize(
    ("registry", "algorithm"),
    [
        (_SORTED_SPIKES_ALGORITHMS, "sorted_spikes_kde"),
        (_SORTED_SPIKES_ALGORITHMS, "sorted_spikes_diffusion"),
        (_CLUSTERLESS_ALGORITHMS, "clusterless_kde"),
        (_CLUSTERLESS_ALGORITHMS, "clusterless_diffusion"),
    ],
)
def test_default_position_bandwidth(registry, algorithm):
    """A detector's default position kernel is sqrt(12.5), the fits' default."""
    parameters = inspect.signature(registry[algorithm].fit).parameters
    effective = {
        **{name: p.default for name, p in parameters.items()},
        **registry[algorithm].default_params,
    }
    assert effective["position_std"] == pytest.approx(np.sqrt(12.5))


@pytest.mark.parametrize(
    "fit",
    [
        record.fit
        for record in [
            *_SORTED_SPIKES_ALGORITHMS.values(),
            *_CLUSTERLESS_ALGORITHMS.values(),
        ]
    ],
    ids=[*_SORTED_SPIKES_ALGORITHMS, *_CLUSTERLESS_ALGORITHMS],
)
def test_registered_fits_take_no_var_keyword(fit):
    """The parameter check skips a ``**kwargs`` fit, so no registered fit has one."""
    kinds = {p.kind for p in inspect.signature(fit).parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds


@pytest.mark.parametrize("algorithm", sorted(_SORTED_SPIKES_ALGORITHMS))
def test_sorted_defaults_reach_the_fit(monkeypatch, sim, algorithm):
    decoder = _sorted_decoder(algorithm)
    received = _record_fit_params(
        monkeypatch, _SORTED_SPIKES_ALGORITHMS, algorithm, decoder, sim
    )
    assert received == EXPECTED_DEFAULTS[algorithm]


@pytest.mark.parametrize("algorithm", sorted(_CLUSTERLESS_ALGORITHMS))
def test_clusterless_defaults_reach_the_fit(monkeypatch, sim, algorithm):
    decoder = _clusterless_decoder(algorithm)
    received = _record_fit_params(
        monkeypatch, _CLUSTERLESS_ALGORITHMS, algorithm, decoder, sim
    )
    assert received == EXPECTED_DEFAULTS[algorithm]


def test_user_params_replace_the_defaults(monkeypatch, sim):
    """A user dict is passed as given: omitted defaults are not merged back in."""
    decoder = _sorted_decoder("sorted_spikes_kde", {"position_std": 3.0})
    received = _record_fit_params(
        monkeypatch, _SORTED_SPIKES_ALGORITHMS, "sorted_spikes_kde", decoder, sim
    )
    assert received == {"position_std": 3.0}


def test_resolved_params_are_a_fresh_copy():
    """Mutating a resolved dict leaks into neither the table nor the user's dict."""
    decoder = _sorted_decoder("sorted_spikes_kde")
    decoder._resolve_sorted_spikes_algorithm_params()["block_size"] = 5
    assert (
        _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"].default_params["block_size"]
        == 10_000
    )

    params = {"position_std": 5.0}
    decoder = _clusterless_decoder("clusterless_kde", params)
    resolved = decoder._resolve_clusterless_algorithm_params()
    assert resolved == params
    assert resolved is not params


@pytest.mark.parametrize(
    ("algorithm", "typo", "accepted"),
    [
        ("sorted_spikes_kde", "position_sd", "position_std"),
        ("sorted_spikes_glm", "l2_penalyt", "l2_penalty"),
        ("sorted_spikes_mrf", "block_size", "penalty"),
    ],
)
def test_sorted_unknown_param_raises(sim, algorithm, typo, accepted):
    """Unknown sorted-spikes keys raise; they used to be dropped silently."""
    decoder = _sorted_decoder(algorithm, {typo: 1.0})
    with pytest.raises(ValidationError, match=typo) as error:
        _fit(decoder, sim)
    assert accepted in str(error.value)


def test_unknown_sorted_algorithm_raises():
    """An unregistered algorithm name raises ValidationError listing the names.

    The detectors reject one at construction; resolving its params rejects
    one assigned to the attribute directly.
    """
    decoder = _sorted_decoder("sorted_spikes_kde")
    decoder.sorted_spikes_algorithm = "sorted_spikes_kde_typo"
    with pytest.raises(ValidationError, match="sorted_spikes_kde_typo") as error:
        decoder._resolve_sorted_spikes_algorithm_params()
    assert "sorted_spikes_kde" in str(error.value)


def test_sorted_glm_geometry_is_not_a_user_param(sim):
    """The GLM reads its geometry from the environment; it is not a parameter."""
    decoder = _sorted_decoder("sorted_spikes_glm", {"edges": None})
    with pytest.raises(ValidationError, match="edges"):
        _fit(decoder, sim)


@pytest.mark.parametrize(
    ("algorithm", "typo", "accepted"),
    [
        ("clusterless_kde", "waveform_sd", "waveform_std"),
        ("clusterless_diffusion", "heat_kernel_rnk", "heat_kernel_rank"),
        ("clusterless_gmm", "gmm_componets_joint", "bin_tile_size"),
        ("clusterless_mark_glm", "n_mark_component", "n_mark_components"),
    ],
)
def test_clusterless_unknown_param_raises(sim, algorithm, typo, accepted):
    """The message names the bad key and lists the accepted ones."""
    decoder = _clusterless_decoder(algorithm, {typo: 1.0})
    with pytest.raises(ValidationError, match=typo) as error:
        _fit(decoder, sim)
    assert accepted in str(error.value)


@pytest.mark.parametrize(
    "algorithm", sorted(_SORTED_SPIKES_ALGORITHMS) + sorted(_CLUSTERLESS_ALGORITHMS)
)
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("encoding_time_range", (0.0, 0.2)),
        ("valid_position_intervals", [[0.0, 0.2]]),
    ],
)
def test_encoding_support_is_not_an_algorithm_param(sim, algorithm, name, value):
    """The encoding support belongs to the detector's ``fit``, not the params.

    The detector builds the support from its own ``fit`` arguments and passes
    it to the algorithm's fit, which then ignores these parameters, so they
    used to be dropped silently.
    """
    if algorithm in _SORTED_SPIKES_ALGORITHMS:
        decoder = _sorted_decoder(algorithm, {name: value})
    else:
        decoder = _clusterless_decoder(algorithm, {name: value})
    with pytest.raises(ValidationError, match=name) as error:
        _fit(decoder, sim)
    assert f"detector.fit(..., {name}=" in str(error.value)


@pytest.mark.parametrize(
    ("algorithm", "accepted"),
    [
        ("clusterless_gmm", "gmm_components_joint"),
        ("clusterless_mark_glm", "n_mark_components"),
    ],
)
@pytest.mark.parametrize("kde_only", ["waveform_std", "position_std", "block_size"])
def test_kde_only_param_raises_for_mixture_models(sim, algorithm, accepted, kde_only):
    """The mixture models have no kernel bandwidths; these keys used to be ignored."""
    decoder = _clusterless_decoder(algorithm, {kde_only: 1.0})
    with pytest.raises(ValidationError, match=kde_only) as error:
        _fit(decoder, sim)
    assert accepted in str(error.value)


# The default position_std is below the simulated bin spacing; the test is about
# predicting from a fitted model, not the smoothing width.
@pytest.mark.filterwarnings(
    r"ignore:position_std \(.*\) is below the characteristic bin spacing:UserWarning"
)
def test_fitted_detector_with_a_removed_param_still_predicts(sim):
    """A fitted detector predicts from its encoding model, not its params.

    An old pickle may store a key a fit no longer accepts (``block_size`` for
    sorted diffusion); loading and predicting must not re-validate it.
    """
    n_decode = 200
    decoder = _fit(_sorted_decoder("sorted_spikes_diffusion"), sim)
    decoder.sorted_spikes_algorithm_params = {"position_std": 6.0, "block_size": 10}
    restored = pickle.loads(pickle.dumps(decoder))
    results = restored.predict(
        sim.spike_times,
        time_edges=time_edges_from_centers(sim.position_time[:n_decode]),
    )
    assert np.all(np.isfinite(results.acausal_posterior.values))
