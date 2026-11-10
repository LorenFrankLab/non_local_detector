"""The chunked drivers' likelihood-callback contract.

Every driver calls a callback as ``f(*log_likelihood_args,
time_edges=time_edges, is_missing=...)``. One marked with ``row_slice_aware``
receives the FULL ``time_edges`` plus the chunk's global row range and
therefore owns every spike exactly once. An unmarked callable cannot be asked
for a chunk's rows, so the drivers reject it there. Both are covered here,
together with the row accumulation that makes an explicitly requested log
likelihood available when caching is off.
"""

import functools
import tracemalloc

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
import pytest

from non_local_detector import core
from non_local_detector.core import (
    accepts_row_slice,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    row_slice_aware,
)
from tests.conftest import precision_mode

N_TIME = 12
N_STATES = 3
N_CHUNKS = 4
RATES = np.array([0.5, 4.0, 20.0])
PARITY_KWARGS = {"rtol": 1e-5, "atol": 1e-6}


def count_spikes(spike_times, time_edges, row_slice=None):
    """Toy binning against the full ``time_edges``, returning the requested rows.

    Row ``i`` owns ``[time_edges[i], time_edges[i + 1])``; the last row also
    owns the final edge.
    """
    counts, _ = np.histogram(spike_times, bins=time_edges)
    return counts[row_slice or slice(None)]


def poisson_log_likelihood(spike_times, row_slice=None, *, time_edges):
    """Tiny Poisson backend over ``N_STATES`` bins, binned on ``time_edges``."""
    counts = count_spikes(spike_times, time_edges, row_slice=row_slice)
    rates = jnp.asarray(RATES)[None, :]
    return jax.scipy.special.xlogy(jnp.asarray(counts)[:, None], rates) - rates


@row_slice_aware
def row_aware_callback(
    spike_times, calls, *, time_edges, is_missing=None, row_slice=None
):
    """Row-aware callback: bins against the full edges, returns the rows asked for."""
    calls.append((len(time_edges), row_slice))
    return poisson_log_likelihood(
        spike_times, row_slice=row_slice, time_edges=time_edges
    )


def unmarked_callback(spike_times, calls, *, time_edges, is_missing=None):
    """Unmarked callback: takes no ``row_slice``, so it can only see a whole timeline."""
    calls.append(np.asarray(time_edges))
    return poisson_log_likelihood(spike_times, time_edges=time_edges)


@pytest.fixture
def problem():
    """A decoding timeline with a spike on every interior edge and bin center."""
    time_edges = np.arange(N_TIME + 1, dtype=float)
    return {
        "time_edges": time_edges,
        # A spike on every chunk boundary, owned by the row that starts there.
        "spike_times": np.sort(
            np.concatenate([time_edges[1:-1], time_edges[:-1] + 0.5])
        ),
        "state_ind": np.arange(N_STATES),
        "initial_distribution": np.full((N_STATES,), 1.0 / N_STATES),
        "transition_matrix": np.full((N_STATES, N_STATES), 1.0 / N_STATES),
    }


def run_driver(driver, problem, log_likelihood_func, calls, n_chunks, **kwargs):
    """Run either chunked driver with the same small problem."""
    common = {
        "time_edges": problem["time_edges"],
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
    and ``__wrapped__``-recording decorators do not, and a missed mark would
    reject a valid callback for chunked evaluation.
    """

    class Detector:
        @row_slice_aware
        def compute_log_likelihood(
            self, *, time_edges, is_missing=None, row_slice=None
        ):
            return None

        def unmarked(self, *, time_edges, is_missing=None):
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

    assert not accepts_row_slice(unmarked_callback)
    assert not accepts_row_slice(detector.unmarked)
    assert not accepts_row_slice(functools.partial(unmarked_callback, np.zeros(2)))


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_row_aware_callback_gets_full_time_and_tiling_rows(driver, problem):
    """A marked callback sees the whole timeline and a partition of its rows."""
    calls: list = []
    run_driver(driver, problem, row_aware_callback, calls, N_CHUNKS)

    assert len(calls) == N_CHUNKS
    assert [n_edges for n_edges, _ in calls] == [N_TIME + 1] * N_CHUNKS
    rows = np.concatenate(
        [np.arange(row_slice.start, row_slice.stop) for _, row_slice in calls]
    )
    np.testing.assert_array_equal(rows, np.arange(N_TIME))


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_unmarked_callback_is_rejected_when_evaluated_per_chunk(driver, problem):
    """Chunk-by-chunk evaluation of an unmarked callback raises before any call.

    An unmarked callback cannot be asked for one chunk's rows, so there is no
    chunk-local fallback.
    """
    calls: list = []
    with pytest.raises(ValueError, match="row_slice_aware"):
        run_driver(driver, problem, unmarked_callback, calls, N_CHUNKS)
    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_unmarked_callback_is_called_once_on_the_full_time(driver, problem):
    """One chunk, or cached likelihoods, call an unmarked callback once.

    It receives the whole ``time_edges`` and no ``row_slice``, so it bins every
    spike, and the chunked-cached result equals the single-chunk one.
    """
    single_calls: list = []
    cached_calls: list = []
    single = run_driver(driver, problem, unmarked_callback, single_calls, 1)
    cached = run_driver(
        driver,
        problem,
        unmarked_callback,
        cached_calls,
        N_CHUNKS,
        cache_log_likelihoods=True,
    )
    reference = run_driver(driver, problem, row_aware_callback, [], N_CHUNKS)

    for calls in (single_calls, cached_calls):
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], problem["time_edges"])
    np.testing.assert_allclose(cached[2], single[2], **PARITY_KWARGS)
    np.testing.assert_allclose(reference[2], single[2], **PARITY_KWARGS)


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
        poisson_log_likelihood(problem["spike_times"], time_edges=problem["time_edges"])
    )

    assert accumulated[5].shape == expected.shape
    np.testing.assert_allclose(accumulated[5], expected, **PARITY_KWARGS)


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_requested_likelihood_retains_one_host_array(driver, monkeypatch):
    """Host storage at smoothing must not retain both chunks and their copy.

    Stub the numerical kernels to isolate driver-owned storage. Warm up JAX
    slicing and aggregation before measuring; the budget allows half an output
    array for Python bookkeeping, but not a second recording-sized likelihood.
    Real filtering/donation correctness is covered by the accumulation test.
    """
    n_time, n_bins, n_chunks = 1024, 256, 4
    expected = np.arange(n_time * n_bins, dtype=np.float32).reshape(n_time, n_bins)
    device_values = jnp.asarray(expected)
    retained = []
    measuring = False

    @row_slice_aware
    def callback(*, time_edges, is_missing=None, row_slice=None):
        if measuring and not tracemalloc.is_tracing():
            tracemalloc.start()
        return device_values[row_slice]

    def filter_stub(initial_distribution, log_likelihoods, **kwargs):
        return (0.0, initial_distribution), (
            log_likelihoods,
            log_likelihoods,
            log_likelihoods[:, 0],
        )

    def smoother_stub(filtered_probs, **kwargs):
        if tracemalloc.is_tracing():
            retained.append(tracemalloc.get_traced_memory()[0])
            tracemalloc.stop()
        return filtered_probs

    monkeypatch.setattr(core, "_filter_internal", filter_stub)
    monkeypatch.setattr(core, "_filter_covariate_dependent_internal", filter_stub)
    monkeypatch.setattr(core, "_smoother_internal", smoother_stub)
    monkeypatch.setattr(core, "_smoother_covariate_dependent_internal", smoother_stub)
    kwargs = {
        "time_edges": np.arange(n_time + 1, dtype=float),
        "state_ind": np.zeros(n_bins, dtype=int),
        "initial_distribution": np.full(n_bins, 1 / n_bins),
        "log_likelihood_func": callback,
        "log_likelihood_args": (),
        "cache_log_likelihoods": False,
        "accumulate_log_likelihoods": True,
        "n_chunks": n_chunks,
    }
    if driver is chunked_filter_smoother:
        kwargs["transition_matrix"] = np.eye(n_bins)
    else:
        kwargs["discrete_transition_matrix"] = np.ones((n_time, 1, 1))
        kwargs["continuous_transition_matrix"] = np.eye(n_bins)
    driver(**kwargs)
    measuring = True
    try:
        result = driver(**kwargs)
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(result[5], expected)
    assert len(retained) == 1
    assert retained[0] < 1.5 * expected.nbytes


@pytest.mark.unit
@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_positional_dtype_still_selects_the_dtype(driver, problem, dtype):
    """A pre-existing positional ``dtype`` must not land on the new flag.

    ``accumulate_log_likelihoods`` was added to drivers whose parameter order
    was already public. A caller passing ``dtype`` positionally in its original
    slot would otherwise set the accumulation flag instead (a dtype object is
    truthy) and silently keep the computation in float32, so the flag is
    keyword-only and comes last. The float32 case runs everywhere and guards
    the flag; the float64 case (run with x64 enabled) additionally shows the
    dtype was honoured.
    """
    args = [
        problem["time_edges"],
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
        dtype,  # dtype, in its pre-existing positional slot
    ]
    with precision_mode(dtype == jnp.float64):
        result = driver(*args)

    assert result[5] is None  # log_likelihoods: accumulation stayed off
    assert result[0].dtype == dtype  # acausal_posterior
    assert result[6].dtype == dtype  # causal_posterior


@pytest.mark.unit
def test_checkpointed_driver_calls_callbacks_as_the_dense_drivers_do(problem):
    """``checkpointed_forward_backward`` follows the same callback contract.

    An unmarked callback is rejected before any call when ``chunk_size``
    splits the rows and is called on the full ``time_edges`` without
    ``row_slice`` when one chunk covers them; a marked one receives the full
    edges per chunk. All three decode like the dense driver.
    """
    from non_local_detector.checkpointed_inference import (
        checkpointed_forward_backward,
    )

    def run(callback, calls, chunk_size):
        return checkpointed_forward_backward(
            problem["time_edges"],
            problem["initial_distribution"],
            callback,
            log_likelihood_args=(problem["spike_times"], calls),
            transition_matrix=problem["transition_matrix"],
            state_ind=problem["state_ind"],
            chunk_size=chunk_size,
        )

    rejected_calls: list = []
    with pytest.raises(ValueError, match="row_slice_aware"):
        run(unmarked_callback, rejected_calls, N_TIME // N_CHUNKS)
    assert rejected_calls == []

    single_calls: list = []
    single = run(unmarked_callback, single_calls, N_TIME)
    assert single_calls
    for edges in single_calls:
        np.testing.assert_array_equal(edges, problem["time_edges"])

    chunked_calls: list = []
    chunked = run(row_aware_callback, chunked_calls, N_TIME // N_CHUNKS)
    assert {n_edges for n_edges, _ in chunked_calls} == {N_TIME + 1}
    dense = run_driver(chunked_filter_smoother, problem, row_aware_callback, [], 1)
    for result in (single, chunked):
        np.testing.assert_allclose(
            result.marginal_log_likelihood, dense[2], **PARITY_KWARGS
        )
