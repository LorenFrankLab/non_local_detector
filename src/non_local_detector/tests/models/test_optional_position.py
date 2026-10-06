"""Position is required at predict time only by states that read it.

Non-local clusterless likelihoods do not depend on the animal's position, so
they must predict with ``position=None`` both through the public decoder and
when a predictor is called directly. Position is required when a Local state
exists or the non-local position penalty is enabled. ``local_position_std``
only changes Local states, so it does not require position by itself.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.models.base import ClusterlessDetector, SortedSpikesDetector
from non_local_detector.models.cont_frag_model import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)

POSITION_FREE_ALGORITHMS = [
    "clusterless_gmm",
    "clusterless_kde",
    "clusterless_kde_log",
]
N_DECODE = 400


def _fit_clusterless(detector, sim):
    return detector.fit(
        sim.position_time, sim.position, sim.spike_times, sim.spike_waveform_features
    )


@pytest.fixture(scope="module", params=POSITION_FREE_ALGORITHMS)
def fitted_decoder(request, clusterless_sim):
    decoder = ClusterlessDecoder(clusterless_algorithm=request.param)
    return _fit_clusterless(decoder, clusterless_sim)


@pytest.mark.unit
def test_decoder_predicts_without_position(fitted_decoder, clusterless_sim):
    """Omitting position gives the same posterior as supplying it."""
    sim = clusterless_sim
    time = sim.position_time[:N_DECODE]
    time_edges = time_edges_from_centers(time)
    without = fitted_decoder.predict(
        sim.spike_times, sim.spike_waveform_features, time_edges=time_edges
    )
    with_position = fitted_decoder.predict(
        sim.spike_times,
        sim.spike_waveform_features,
        time_edges=time_edges,
        position=sim.position[:N_DECODE],
        position_time=time,
    )
    np.testing.assert_array_equal(
        without.acausal_posterior.values, with_position.acausal_posterior.values
    )


@pytest.mark.unit
def test_direct_non_local_predict_without_position(fitted_decoder, clusterless_sim):
    """The registered predictor accepts ``position=None`` when not local."""
    sim = clusterless_sim
    _, predict = _CLUSTERLESS_ALGORITHMS[fitted_decoder.clusterless_algorithm]
    (encoding_model,) = fitted_decoder.encoding_model_.values()
    time_edges = jnp.asarray(time_edges_from_centers(sim.position_time[:N_DECODE]))

    def call(position_time, position):
        return np.asarray(
            predict(
                position_time,
                position,
                sim.spike_times,
                sim.spike_waveform_features,
                time_edges=time_edges,
                **encoding_model,
                is_local=False,
            )
        )

    without = call(None, None)
    assert np.all(np.isfinite(without))
    np.testing.assert_array_equal(without, call(sim.position_time, sim.position))


@pytest.mark.unit
def test_local_state_requires_position(clusterless_sim, sorted_sim):
    """A Local state still rejects a missing position with a package error."""
    sim = clusterless_sim
    clusterless = _fit_clusterless(NonLocalClusterlessDetector(), sim)
    with pytest.raises(ValidationError, match="local observation models"):
        clusterless.predict(
            sim.spike_times,
            sim.spike_waveform_features,
            time_edges=time_edges_from_centers(sim.position_time[:N_DECODE]),
        )

    time, position, spike_times = sorted_sim
    sorted_detector = NonLocalSortedSpikesDetector().fit(time, position, spike_times)
    with pytest.raises(ValidationError, match="local observation models"):
        sorted_detector.predict(
            spike_times, time_edges=time_edges_from_centers(time[:N_DECODE])
        )


@pytest.mark.unit
def test_local_position_std_without_local_state_does_not_need_position(
    clusterless_sim, sorted_sim
):
    """``local_position_std`` affects only Local states, so it alone does not
    make position required."""
    sim = clusterless_sim
    clusterless = ClusterlessDetector(
        **ContFragClusterlessClassifier().get_params(), local_position_std=5.0
    )
    assert not any(obs.is_local for obs in clusterless.observation_models)
    _fit_clusterless(clusterless, sim)
    results = clusterless.predict(
        sim.spike_times,
        sim.spike_waveform_features,
        time_edges=time_edges_from_centers(sim.position_time[:N_DECODE]),
    )
    assert np.all(np.isfinite(results.acausal_posterior.values))

    time, position, spike_times = sorted_sim
    sorted_detector = SortedSpikesDetector(
        **ContFragSortedSpikesClassifier().get_params(), local_position_std=5.0
    )
    sorted_detector.fit(time, position, spike_times)
    results = sorted_detector.predict(
        spike_times, time_edges=time_edges_from_centers(time[:N_DECODE])
    )
    assert np.all(np.isfinite(results.acausal_posterior.values))


@pytest.mark.unit
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_non_local_position_penalty_requires_position(
    family, clusterless_sim, sorted_sim
):
    """The non-local position penalty reads position, so its reason appears in
    the missing-position error."""
    if family == "sorted":
        time, position, spike_times = sorted_sim
        detector = NonLocalSortedSpikesDetector(non_local_position_penalty=1.0).fit(
            time, position, spike_times
        )
        predict_args = (spike_times,)
        decode_time = time[:N_DECODE]
    else:
        sim = clusterless_sim
        detector = _fit_clusterless(
            NonLocalClusterlessDetector(non_local_position_penalty=1.0), sim
        )
        predict_args = (sim.spike_times, sim.spike_waveform_features)
        decode_time = sim.position_time[:N_DECODE]

    with pytest.raises(ValidationError, match="non_local_position_penalty > 0"):
        detector.predict(*predict_args, time_edges=time_edges_from_centers(decode_time))
