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

from non_local_detector import ClusterlessDecoder, NonLocalClusterlessDetector
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

ALGORITHMS = [
    "clusterless_kde",
    "clusterless_kde_log",
    "clusterless_gmm",
    "clusterless_diffusion",
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
    time = sim.position_time[:N_DECODE]
    _, predict = _CLUSTERLESS_ALGORITHMS[fitted_decoder.clusterless_algorithm]
    (encoding_model,) = fitted_decoder.encoding_model_.values()
    for spike_times, features in _mismatched_populations(sim):
        with pytest.raises(ValidationError, match="electrode population lengths"):
            fitted_decoder.predict(spike_times, features, time=time)
        with pytest.raises(ValidationError, match="electrode population lengths"):
            predict(
                jnp.asarray(time),
                None,
                None,
                spike_times,
                features,
                **encoding_model,
                is_local=False,
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
            time=sim.position_time,
            max_iter=1,
        )

    assert vars(detector).keys() == state_before.keys()
    for name, value in state_before.items():
        assert vars(detector)[name] is value, name
