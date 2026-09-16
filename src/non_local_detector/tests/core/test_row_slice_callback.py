"""The chunked drivers' likelihood-callback contract.

The drivers accept arbitrary user callables, so the row information cannot be
injected into every callback. A callback marked with ``row_slice_aware``
receives the FULL ``time`` plus the chunk's global row range and therefore owns
every spike exactly once; an unmarked (legacy) callable keeps the historical
``time[chunk]`` call and is not a global-binning implementation. Both branches
are covered here, together with the row accumulation that makes an explicitly
requested log likelihood available when caching is off.
"""

import functools

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
import pytest

from non_local_detector.core import (
    accepts_row_slice,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    row_slice_aware,
)
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

N_TIME = 12
N_STATES = 3
N_CHUNKS = 4
RATES = np.array([0.5, 4.0, 20.0])
PARITY_KWARGS = {"rtol": 1e-5, "atol": 1e-6}


def poisson_log_likelihood(time, spike_times, row_slice=None):
    """Tiny Poisson backend over ``N_STATES`` bins, binned on the given ``time``."""
    counts = get_spikecount_per_time_bin(spike_times, time, row_slice=row_slice)
    rates = jnp.asarray(RATES)[None, :]
    return jax.scipy.special.xlogy(jnp.asarray(counts)[:, None], rates) - rates


@row_slice_aware
def row_aware_callback(time, spike_times, calls, is_missing=None, row_slice=None):
    """Row-aware callback: bins against the full ``time``, returns the rows asked for."""
    calls.append((len(time), row_slice))
    return poisson_log_likelihood(time, spike_times, row_slice=row_slice)


def legacy_callback(time, spike_times, calls, is_missing=None):
    """Legacy callback: only ever sees the rows' own timestamps (no ``row_slice``)."""
    calls.append(np.asarray(time))
    return poisson_log_likelihood(time, spike_times)


@pytest.fixture
def problem():
    """A decoding timeline whose every bin gap holds a spike."""
    time = np.arange(N_TIME, dtype=float)
    return {
        "time": time,
        # A spike in every gap, so every chunk boundary has one to lose.
        "spike_times": 0.5 * (time[:-1] + time[1:]),
        "state_ind": np.arange(N_STATES),
        "initial_distribution": np.full((N_STATES,), 1.0 / N_STATES),
        "transition_matrix": np.full((N_STATES, N_STATES), 1.0 / N_STATES),
    }


def run_driver(driver, problem, log_likelihood_func, calls, n_chunks, **kwargs):
    """Run either chunked driver with the same small problem."""
    common = {
        "time": problem["time"],
        "state_ind": problem["state_ind"],
        "initial_distribution": problem["initial_distribution"],
        "log_likelihood_func": log_likelihood_func,
        "log_likelihood_args": (problem["spike_times"], calls),
        "n_chunks": n_chunks,
        "cache_log_likelihoods": False,
        **kwargs,
    }
    if driver is chunked_filter_smoother:
        return driver(transition_matrix=problem["transition_matrix"], **common)
    return driver(
        discrete_transition_matrix=np.broadcast_to(
            problem["transition_matrix"], (N_TIME, N_STATES, N_STATES)
        ).copy(),
        continuous_transition_matrix=np.eye(N_STATES),
        **common,
    )


DRIVERS = [chunked_filter_smoother, chunked_filter_smoother_covariate_dependent]
DRIVER_IDS = ["stationary", "covariate_dependent"]


@pytest.mark.unit
def test_marker_is_visible_through_the_bound_method():
    """``accepts_row_slice`` must see the mark through every common wrapper.

    Bound methods forward attribute lookups on their own; ``functools.partial``
    and ``__wrapped__``-recording decorators do not, and a missed mark silently
    demotes the callback to the legacy branch (boundary spikes dropped, numbers
    still plausible).
    """

    class Detector:
        @row_slice_aware
        def compute_log_likelihood(self, time, is_missing=None, row_slice=None):
            return None

        def legacy(self, time, is_missing=None):
            return None

    detector = Detector()

    def wrapping_decorator(*args, **kwargs):
        return row_aware_callback(*args, **kwargs)

    # Only ``__wrapped__`` is set: unlike ``functools.wraps`` this does not copy
    # the wrappee's ``__dict__``, so the mark is reachable only by unwrapping.
    wrapping_decorator.__wrapped__ = row_aware_callback

    assert accepts_row_slice(row_aware_callback)
    assert accepts_row_slice(detector.compute_log_likelihood)
    assert accepts_row_slice(functools.partial(row_aware_callback, np.zeros(2)))
    assert accepts_row_slice(
        functools.partial(functools.partial(detector.compute_log_likelihood))
    )
    assert accepts_row_slice(wrapping_decorator)

    assert not accepts_row_slice(legacy_callback)
    assert not accepts_row_slice(detector.legacy)
    assert not accepts_row_slice(functools.partial(legacy_callback, np.zeros(2)))


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_row_aware_callback_gets_full_time_and_tiling_rows(driver, problem):
    """A marked callback sees the whole timeline and a partition of its rows."""
    calls: list = []
    run_driver(driver, problem, row_aware_callback, calls, N_CHUNKS)

    assert len(calls) == N_CHUNKS
    assert [n_time for n_time, _ in calls] == [N_TIME] * N_CHUNKS
    rows = np.concatenate(
        [np.arange(row_slice.start, row_slice.stop) for _, row_slice in calls]
    )
    np.testing.assert_array_equal(rows, np.arange(N_TIME))


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_legacy_callback_gets_the_sliced_time_and_no_row_slice(driver, problem):
    """An unmarked callback keeps the historical chunk-local call.

    It is called with ``time[chunk]`` and no ``row_slice`` keyword -- injecting
    one would raise ``TypeError`` against this signature. Chunk-local clipping
    then loses the boundary spikes, which is why a legacy callable must not be
    presented as a global-binning implementation.
    """
    calls: list = []
    chunked = run_driver(driver, problem, legacy_callback, calls, N_CHUNKS)
    unchunked = run_driver(driver, problem, legacy_callback, [], 1)

    row_chunks = np.array_split(np.arange(N_TIME), N_CHUNKS)
    assert len(calls) == N_CHUNKS
    for recorded_time, chunk in zip(calls, row_chunks, strict=True):
        np.testing.assert_array_equal(recorded_time, problem["time"][chunk])

    # Documented limitation: the legacy call drops the boundary spikes.
    assert not np.isclose(chunked[2], unchunked[2], **PARITY_KWARGS)


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_row_aware_chunked_matches_unchunked(driver, problem):
    """Chunked prediction through a row-aware callback is exact."""
    chunked = run_driver(driver, problem, row_aware_callback, [], N_CHUNKS)
    unchunked = run_driver(driver, problem, row_aware_callback, [], 1)

    np.testing.assert_allclose(chunked[0], unchunked[0], **PARITY_KWARGS)
    np.testing.assert_allclose(chunked[1], unchunked[1], **PARITY_KWARGS)
    np.testing.assert_allclose(chunked[2], unchunked[2], **PARITY_KWARGS)


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_accumulated_log_likelihoods_cover_every_row(driver, problem):
    """``accumulate_log_likelihoods`` returns all rows in global order."""
    without = run_driver(driver, problem, row_aware_callback, [], N_CHUNKS)
    assert without[5] is None

    accumulated = run_driver(
        driver,
        problem,
        row_aware_callback,
        [],
        N_CHUNKS,
        accumulate_log_likelihoods=True,
    )
    expected = np.asarray(
        poisson_log_likelihood(problem["time"], problem["spike_times"])
    )

    assert accumulated[5].shape == expected.shape
    np.testing.assert_allclose(accumulated[5], expected, **PARITY_KWARGS)


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_positional_dtype_still_selects_the_dtype(driver, problem):
    """A pre-existing positional ``dtype`` must not land on the new flag.

    ``accumulate_log_likelihoods`` was added to drivers whose parameter order
    was already public. A caller passing ``dtype`` positionally in its original
    slot would otherwise set the accumulation flag instead and silently keep the
    computation in float32, so the flag is keyword-only and comes last.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")

    args = [
        problem["time"],
        problem["state_ind"],
        problem["initial_distribution"],
    ]
    if driver is chunked_filter_smoother:
        args.append(problem["transition_matrix"])
    else:
        args.append(
            np.broadcast_to(
                problem["transition_matrix"], (N_TIME, N_STATES, N_STATES)
            ).copy()
        )
        args.append(np.eye(N_STATES))
    args += [
        row_aware_callback,  # log_likelihood_func
        (problem["spike_times"], []),  # log_likelihood_args
        None,  # is_missing
        N_CHUNKS,  # n_chunks
        None,  # log_likelihoods
        False,  # cache_log_likelihoods
        jnp.float64,  # dtype, in its pre-existing positional slot
    ]
    result = driver(*args)

    assert result[0].dtype == np.float64  # acausal_posterior
    assert result[6].dtype == np.float64  # causal_posterior
    assert result[5] is None  # log_likelihoods: accumulation stayed off
