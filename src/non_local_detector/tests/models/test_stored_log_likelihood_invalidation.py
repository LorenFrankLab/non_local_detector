"""A stored ``log_likelihood_`` must never be consumed by a later public call.

``estimate_parameters(store_log_likelihood=True)`` keeps the final E-step's log
likelihood on the detector as an OUTPUT. It is tied to the inputs that produced
it, so a later run on different data must not read it back: doing so decodes the
new spikes with the old likelihood and returns a plausible, wrong posterior.

``estimate_parameters`` refits the model on the data it is given (``self.fit``
resets initial conditions, transitions and the encoding model), so a second run
on dataset B must return exactly what a freshly constructed detector returns on
dataset B. Any difference is contamination carried across the call boundary.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

N_CHUNKS = 5
DECODE = slice(10_000, 10_200)
PARITY_KWARGS = {"rtol": 1e-5, "atol": 1e-6}


def make_detector():
    """A small detector; identical construction for every run in this module."""
    return NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": int(2**12),
        },
    )


def make_dataset(seed):
    """One simulated recording; different seeds give different spikes/position.

    The decoding timeline has the same length for every seed, so a stale
    likelihood is shape-compatible and a length check would not catch it.
    """
    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=4, seed=seed)
    return {
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "is_training": ~is_event,
        "time": time[DECODE],
    }


@pytest.fixture(scope="module")
def datasets():
    """Two recordings with identical timeline lengths but different spikes."""
    first, second = make_dataset(0), make_dataset(1)
    assert len(first["time"]) == len(second["time"])
    assert not np.array_equal(first["spike_times"][0], second["spike_times"][0])
    return first, second


def run_em(detector, data, max_iter=1, **kwargs):
    """One EM run on ``data`` (one iteration keeps the test fast)."""
    return detector.estimate_parameters(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        time=data["time"],
        is_training=data["is_training"],
        max_iter=max_iter,
        **kwargs,
    )


def assert_matches_fresh_run(reused, fresh):
    np.testing.assert_allclose(
        reused.acausal_posterior.to_numpy(),
        fresh.acausal_posterior.to_numpy(),
        **PARITY_KWARGS,
    )
    np.testing.assert_allclose(
        reused.attrs["marginal_log_likelihoods"],
        fresh.attrs["marginal_log_likelihoods"],
        **PARITY_KWARGS,
    )


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:EM did not converge")
@pytest.mark.parametrize("n_chunks", [1, N_CHUNKS])
def test_second_estimate_parameters_ignores_the_stored_log_likelihood(
    datasets, n_chunks
):
    """A stored likelihood from run A must not leak into run B.

    Parametrized over ``n_chunks`` because the hazard predates chunking: with
    ``n_chunks == 1`` the stored array came from the cached branch, with
    ``n_chunks > 1`` from the accumulated rows.
    """
    first, second = datasets

    detector = make_detector()
    run_em(detector, first, n_chunks=n_chunks, store_log_likelihood=True)
    assert detector.log_likelihood_ is not None

    reused = run_em(detector, second, n_chunks=n_chunks, cache_likelihood=False)
    fresh = run_em(make_detector(), second, n_chunks=n_chunks, cache_likelihood=False)

    assert_matches_fresh_run(reused, fresh)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:EM did not converge")
def test_cached_likelihood_is_recomputed_after_the_encoding_model_updates(datasets):
    """Within one EM run, the cached likelihood must not outlive the M-step.

    With ``cache_likelihood=True`` and ``n_chunks == 1`` the E-step's likelihood
    is kept for the next iteration. The M-step then refits the encoding model,
    so a second E-step that reused the cached array would decode with the
    first iteration's place fields: plausible, wrong, and invisible from the
    posterior alone. The stored output must equal a fresh computation on the
    FINAL encoding model, and that must differ from the first iteration's.
    """
    first, _ = datasets

    detector = make_detector()
    run_em(
        detector,
        first,
        max_iter=1,
        n_chunks=1,
        cache_likelihood=True,
        estimate_encoding_model=True,
        store_log_likelihood=True,
    )
    after_one_iteration = np.asarray(detector.log_likelihood_)

    detector = make_detector()
    run_em(
        detector,
        first,
        max_iter=2,
        n_chunks=1,
        cache_likelihood=True,
        estimate_encoding_model=True,
        store_log_likelihood=True,
    )
    stored = np.asarray(detector.log_likelihood_)
    fresh = np.asarray(
        detector.compute_log_likelihood(
            first["time"],
            first["position_time"],
            first["position"],
            first["spike_times"],
        )
    )

    # Guard the guard: the M-step must have changed the likelihood, otherwise
    # a stale cache would be indistinguishable from a fresh computation.
    assert not np.allclose(after_one_iteration, fresh, **PARITY_KWARGS)
    np.testing.assert_allclose(stored, fresh, **PARITY_KWARGS)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:EM did not converge")
def test_predict_ignores_the_stored_log_likelihood(datasets):
    """``predict`` must decode the spikes it is given, not a stored likelihood."""
    first, second = datasets

    detector = make_detector()
    run_em(detector, first, n_chunks=N_CHUNKS, store_log_likelihood=True)
    assert detector.log_likelihood_ is not None

    predict_kwargs = {
        "spike_times": second["spike_times"],
        "time": second["time"],
        "position": second["position"][DECODE],
        "position_time": second["time"],
        "cache_likelihood": False,
    }
    reused = detector.predict(**predict_kwargs, n_chunks=N_CHUNKS)

    fresh_detector = make_detector()
    run_em(fresh_detector, first, n_chunks=N_CHUNKS)
    fresh = fresh_detector.predict(**predict_kwargs, n_chunks=N_CHUNKS)

    assert_matches_fresh_run(reused, fresh)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:EM did not converge")
def test_most_likely_sequence_ignores_the_stored_log_likelihood(datasets):
    """The Viterbi path must be decoded from the spikes passed to it."""
    first, second = datasets

    detector = make_detector()
    run_em(detector, first, n_chunks=1, store_log_likelihood=True)
    assert detector.log_likelihood_ is not None

    fresh_detector = make_detector()
    run_em(fresh_detector, first, n_chunks=1)

    args = (
        second["position_time"],
        second["position"],
        second["spike_times"],
        second["time"],
    )
    reused = detector.most_likely_sequence(*args)
    fresh = fresh_detector.most_likely_sequence(*args)

    np.testing.assert_array_equal(reused["state"].to_numpy(), fresh["state"].to_numpy())


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:EM did not converge")
def test_refitting_clears_the_stored_log_likelihood(datasets):
    """Any refit invalidates the stored output it no longer describes."""
    first, second = datasets

    detector = make_detector()
    run_em(detector, first, n_chunks=N_CHUNKS, store_log_likelihood=True)
    assert detector.log_likelihood_ is not None

    detector.fit(
        second["position_time"],
        second["position"],
        second["spike_times"],
        is_training=second["is_training"],
    )
    assert getattr(detector, "log_likelihood_", None) is None

    # A run that does not ask to store it leaves nothing behind either.
    run_em(detector, second, n_chunks=N_CHUNKS, cache_likelihood=False)
    assert getattr(detector, "log_likelihood_", None) is None
