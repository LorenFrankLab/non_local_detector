"""Clusterless spike times and waveform features must describe the same electrodes.

The detector used to pair the two collections with a non-strict ``zip`` before any
backend validation, so an extra spike-time or feature array was silently dropped
and the model was fit on fewer electrodes. A per-electrode row mismatch raised a
raw numpy ``IndexError`` from boolean indexing. At predict time the KDE, log-KDE
and diffusion backends raised a bare ``ValueError`` from ``zip(strict=True)``.
Every clusterless backend now raises the package ``ValidationError``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    NonLocalClusterlessDetector,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.likelihoods.mark_reference import (
    mark_reference_log_likelihood,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

ALGORITHMS = [
    "clusterless_kde",
    "clusterless_gmm",
    "clusterless_diffusion",
    "clusterless_mark_glm",
]
N_DECODE = 300


@pytest.fixture(scope="module")
def sim():
    return make_simulated_run_data(
        n_tetrodes=3, place_field_means=np.arange(0, 90, 15), n_runs=1, seed=0
    )


@pytest.fixture(scope="module", params=ALGORITHMS)
def fitted_decoder(request, sim):
    decoder = ClusterlessDecoder(clusterless_algorithm=request.param)
    return decoder.fit(
        sim.position_time, sim.position, sim.spike_times, sim.spike_waveform_features
    )


def _mismatched_populations(sim):
    """(spike_times, features) pairs whose electrode counts differ."""
    times, features = list(sim.spike_times), list(sim.spike_waveform_features)
    return [(times, features[:-1]), (times[:-1], features)]


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "detector_cls", [ClusterlessDecoder, NonLocalClusterlessDetector]
)
def test_fit_rejects_electrode_count_mismatch(sim, algorithm, detector_cls):
    for spike_times, features in _mismatched_populations(sim):
        detector = detector_cls(clusterless_algorithm=algorithm)
        with pytest.raises(ValidationError, match="electrode population lengths"):
            detector.fit(sim.position_time, sim.position, spike_times, features)
        assert not hasattr(detector, "encoding_model_")


@pytest.mark.unit
def test_fit_rejects_removed_log_kde_name_with_a_hint(sim):
    """The removed ``clusterless_kde_log`` name points to ``clusterless_kde``."""
    detector = ClusterlessDecoder(clusterless_algorithm="clusterless_kde_log")
    with pytest.raises(ValidationError, match='Use "clusterless_kde"'):
        detector.fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
        )
    assert not hasattr(detector, "encoding_model_")


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_fit_rejects_per_electrode_row_mismatch(sim, algorithm):
    features = list(sim.spike_waveform_features)
    features[1] = features[1][:-1]
    detector = ClusterlessDecoder(clusterless_algorithm=algorithm)
    with pytest.raises(ValidationError, match="electrode 1"):
        detector.fit(sim.position_time, sim.position, sim.spike_times, features)


@pytest.mark.unit
def test_predict_rejects_electrode_count_mismatch(fitted_decoder, sim):
    """Public prediction and a direct predictor call both raise the package error."""
    time_edges = time_edges_from_centers(sim.position_time[:N_DECODE])
    _, predict = _CLUSTERLESS_ALGORITHMS[fitted_decoder.clusterless_algorithm]
    (encoding_model,) = fitted_decoder.encoding_model_.values()
    for spike_times, features in _mismatched_populations(sim):
        with pytest.raises(ValidationError, match="electrode population lengths"):
            fitted_decoder.predict(spike_times, features, time_edges=time_edges)
        with pytest.raises(ValidationError, match="electrode population lengths"):
            predict(
                None,
                None,
                spike_times,
                features,
                **encoding_model,
                mark_reference=fitted_decoder.mark_reference_,
                is_local=False,
                time_edges=jnp.asarray(time_edges),
            )


@pytest.mark.unit
def test_refit_with_mismatch_leaves_fitted_state_unchanged(sim):
    """Validation runs before ``fit`` (or ``estimate_parameters``) rebuilds the
    environments and transitions of an already-fitted detector."""
    detector = NonLocalClusterlessDetector().fit(
        sim.position_time, sim.position, sim.spike_times, sim.spike_waveform_features
    )
    state_before = dict(vars(detector))
    spike_times, features = _mismatched_populations(sim)[0]

    with pytest.raises(ValidationError, match="electrode population lengths"):
        detector.fit(sim.position_time, sim.position, spike_times, features)
    with pytest.raises(ValidationError, match="electrode population lengths"):
        detector.estimate_parameters(
            sim.position_time,
            sim.position,
            spike_times,
            features,
            time_edges=time_edges_from_centers(sim.position_time),
            max_iter=1,
        )

    assert vars(detector).keys() == state_before.keys()
    for name, value in state_before.items():
        assert vars(detector)[name] is value, name


def _with_nonfinite_mark(sim, before_time, value=np.nan):
    """Copies of the waveform features with one mark of electrode 0 set to ``value``.

    The corrupted spike is the first one (it must lie before ``before_time``) or,
    when ``before_time`` is None, the last spike of the recording.
    """
    features = [np.array(f, copy=True) for f in sim.spike_waveform_features]
    times = np.asarray(sim.spike_times[0])
    spike = 0 if before_time is not None else len(times) - 1
    if before_time is not None:
        assert times[spike] < before_time
    features[0][spike, 0] = value
    return features


def _direct_predict(decoder, sim, features, time_edges, is_local):
    _, predict = _CLUSTERLESS_ALGORITHMS[decoder.clusterless_algorithm]
    (encoding_model,) = decoder.encoding_model_.values()
    return predict(
        sim.position_time,
        sim.position,
        sim.spike_times,
        features,
        **encoding_model,
        mark_reference=decoder.mark_reference_,
        is_local=is_local,
        time_edges=jnp.asarray(time_edges),
    )


@pytest.mark.unit
@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("is_local", [False, True], ids=["non_local", "local"])
def test_predict_rejects_nonfinite_decode_mark(fitted_decoder, sim, is_local, value):
    """One non-finite mark in the decoding window raises instead of returning NaN.

    A NaN mark makes its bin's likelihood NaN, and the filter then carries NaN
    into the posterior of every later bin.
    """
    time_edges = time_edges_from_centers(sim.position_time[:N_DECODE])
    features = _with_nonfinite_mark(sim, before_time=time_edges[-1], value=value)
    with pytest.raises(ValidationError, match="non-finite.*electrode 0"):
        _direct_predict(fitted_decoder, sim, features, time_edges, is_local)
    with pytest.raises(ValidationError, match="non-finite.*electrode 0"):
        fitted_decoder.predict(sim.spike_times, features, time_edges=time_edges)


@pytest.mark.unit
@pytest.mark.parametrize("is_local", [False, True], ids=["non_local", "local"])
def test_predict_ignores_nonfinite_mark_outside_decoding_window(
    fitted_decoder, sim, is_local
):
    """A spike outside the requested bins never enters the likelihood, so its
    mark is not checked and the result equals the uncorrupted prediction."""
    time_edges = time_edges_from_centers(sim.position_time[:N_DECODE])
    features = _with_nonfinite_mark(sim, before_time=None)
    assert sim.spike_times[0][-1] > time_edges[-1]
    corrupted = _direct_predict(fitted_decoder, sim, features, time_edges, is_local)
    clean = _direct_predict(
        fitted_decoder, sim, sim.spike_waveform_features, time_edges, is_local
    )
    assert np.all(np.isfinite(np.asarray(corrupted)))
    np.testing.assert_array_equal(np.asarray(corrupted), np.asarray(clean))


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_fit_rejects_nonfinite_encoding_mark(sim, algorithm):
    features = _with_nonfinite_mark(sim, before_time=sim.position_time[-1])
    detector = ClusterlessDecoder(clusterless_algorithm=algorithm)
    with pytest.raises(ValidationError, match="finite"):
        detector.fit(sim.position_time, sim.position, sim.spike_times, features)


@pytest.mark.unit
def test_mark_reference_likelihood_rejects_nonfinite_decode_mark(fitted_decoder, sim):
    """No-Spike scores marks with the reference density alone; it checks them too."""
    time_edges = time_edges_from_centers(sim.position_time[:N_DECODE])
    features = _with_nonfinite_mark(sim, before_time=time_edges[-1])
    with pytest.raises(ValidationError, match="non-finite.*electrode 0"):
        mark_reference_log_likelihood(
            fitted_decoder.mark_reference_, sim.spike_times, features, time_edges
        )
