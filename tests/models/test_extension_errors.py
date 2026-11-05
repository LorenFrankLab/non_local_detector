"""A broken extension fails at the cause, with an error naming what it broke.

Toy algorithms and transitions break one rule each of "Adding a likelihood
algorithm" and "Adding a transition" in CONTRIBUTING.md.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from tests.models._helpers import SORTED_SIM_RATE

pytestmark = pytest.mark.unit

_ALGORITHM = "sorted_spikes_kde"
_FIT_PARAMS = {"disable_progress_bar": True}


def _detector(**kwargs):
    return NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm=_ALGORITHM,
        sorted_spikes_algorithm_params=_FIT_PARAMS,
        environments=Environment(place_bin_size=10.0),
        sampling_frequency=SORTED_SIM_RATE,
        **kwargs,
    )


def _decode_kwargs(sorted_sim, n_time=40):
    time, position, spike_times = sorted_sim
    return {
        "spike_times": spike_times,
        "time_edges": time_edges_from_centers(time[:n_time]),
        "position": position,
        "position_time": time,
    }


def _patch_predict(monkeypatch, predict):
    record = _SORTED_SPIKES_ALGORITHMS[_ALGORITHM]

    def patched(*args, **kwargs):
        return predict(record.predict(*args, **kwargs), kwargs)

    monkeypatch.setitem(
        _SORTED_SPIKES_ALGORITHMS, _ALGORITHM, replace(record, predict=patched)
    )


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(lambda result: result[:, :1], id="one column for every bin"),
        pytest.param(lambda result: result[:1], id="one row for every time"),
        pytest.param(lambda result: result[:, 0], id="missing the bin axis"),
        pytest.param(lambda result: jnp.float32(0.0), id="a scalar"),
    ],
)
def test_likelihood_of_the_wrong_shape_names_the_algorithm(
    monkeypatch, sorted_sim, broken
):
    """A predict of the wrong shape is refused, not broadcast over the bins."""
    time, position, spike_times = sorted_sim
    detector = _detector().fit(time, position, spike_times)
    _patch_predict(
        monkeypatch, lambda result, k: result if k["is_local"] else broken(result)
    )

    with pytest.raises(ValidationError, match=_ALGORITHM) as error:
        detector.predict(**_decode_kwargs(sorted_sim))
    assert "(40, " in str(error.value)


def test_local_likelihood_of_the_wrong_shape_names_the_state(monkeypatch, sorted_sim):
    time, position, spike_times = sorted_sim
    detector = _detector().fit(time, position, spike_times)
    _patch_predict(
        monkeypatch,
        lambda result, k: jnp.tile(result, (1, 2)) if k["is_local"] else result,
    )

    with pytest.raises(ValidationError, match="Local"):
        detector.predict(**_decode_kwargs(sorted_sim))


@pytest.mark.parametrize("local_position_std", [None, 5.0])
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_builtin_blocks_pass_the_shape_check(sorted_sim, local_position_std, n_chunks):
    """Local, multi-bin Local and No-Spike decode in one or several chunks."""
    time, position, spike_times = sorted_sim
    detector = _detector(local_position_std=local_position_std).fit(
        time, position, spike_times
    )
    results = detector.predict(**_decode_kwargs(sorted_sim), n_chunks=n_chunks)
    assert np.all(np.isfinite(results.acausal_state_probabilities.values))
