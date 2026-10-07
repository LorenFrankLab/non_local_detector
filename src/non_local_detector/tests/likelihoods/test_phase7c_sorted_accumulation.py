"""Batched sorted likelihoods retain an independent per-neuron reference."""

import math
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import xlogy

from non_local_detector.likelihoods import sorted_spikes_glm, sorted_spikes_kde
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.tests.likelihoods.conftest import EXACT

pytestmark = pytest.mark.unit


@contextmanager
def x64_mode(enabled):
    """Portable precision context for both older and current supported JAX."""
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def float32_arithmetic():
    # These references intentionally round every operation to float32. The
    # separate mixed-input and native tests retain their real x64 policies.
    with x64_mode(False):
        yield


def inputs(backend, fields):
    edges = np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.010])
    # Unsorted data, duplicate boundary events and the closed final bin.
    spikes = [
        np.array([0.010, 0.002, 0.002, -0.001, 0.001]),
        np.array([0.003, 0.006, 0.010]),
        np.array([]),
    ]
    fields = jnp.asarray(fields)
    arguments = {
        "position_time": edges[[0, -1]],
        "position": np.zeros((2, 1)),
        "spike_times": spikes,
        "environment": None,
        "place_fields": fields,
        "no_spike_part_log_likelihood": fields.sum(axis=0),
        "is_track_interior": np.array([True, False, True, True]),
        "disable_progress_bar": True,
        "time_edges": edges,
    }
    if backend is sorted_spikes_kde:
        arguments.update(
            marginal_models=[None] * 3,
            occupancy_model=None,
            occupancy=None,
            mean_rates=np.ones(3),
        )
    else:
        arguments.update(coefficients=np.zeros((3, 1)), emission_design_info=None)
    return arguments


def reference(arguments, rows):
    edges = arguments["time_edges"]
    start, stop, _ = (slice(None) if rows is None else rows).indices(len(edges) - 1)
    stop = max(start, stop)
    fields = np.asarray(arguments["place_fields"])
    dtype = fields.dtype
    duration = np.diff(edges)[start:stop].astype(dtype)
    interior = arguments["is_track_interior"]
    likelihood = np.zeros((stop - start, interior.sum()), dtype=dtype)
    for spikes, rates in zip(arguments["spike_times"], fields, strict=True):
        counts = get_spikecount_per_time_bin(spikes, time_edges=edges, row_slice=rows)
        likelihood += xlogy(
            counts.astype(dtype)[:, None], rates[interior][None, :] * duration[:, None]
        )
    likelihood -= (
        duration[:, None]
        * np.asarray(arguments["no_spike_part_log_likelihood"])[interior]
    )
    return likelihood


def float64_emission(counts, rates, durations, summed_rates):
    """Non-local emission from the same inputs, summed exactly over neurons."""
    counts = np.asarray(counts, np.float64)
    rates = np.asarray(rates, np.float64)
    durations = np.asarray(durations, np.float64)
    terms = xlogy(
        counts.T[:, :, None], rates[:, None, :] * durations[None, :, None]
    )  # (n_neurons, n_rows, n_bins)
    total = np.array(
        [[math.fsum(column) for column in row.T] for row in terms.transpose(1, 0, 2)]
    ).reshape(counts.shape[0], rates.shape[1])
    return total - durations[:, None] * np.asarray(summed_rates, np.float64)


def assert_within_ulps(actual, exact, dtype, ulps=16):
    """Each element lies within ``ulps`` units in the last place of ``exact``."""
    actual = np.asarray(actual, np.float64)
    spacing = np.spacing(np.abs(exact).astype(dtype)).astype(np.float64)
    error = np.abs(actual - exact) / spacing
    assert np.all(error <= ulps), f"max error {np.max(error):.1f} ulp > {ulps}"


@pytest.mark.parametrize("backend", [sorted_spikes_kde, sorted_spikes_glm])
@pytest.mark.parametrize("rows", [None, slice(1, 4), slice(4, 5), slice(2, 2)])
@pytest.mark.parametrize("zero", [False, True])
def test_sorted_accumulation_matches_per_neuron_xlogy(
    backend, rows, zero, float32_arithmetic
):
    fields = np.array(
        [[0.8, 7.0, 12.0, 0.2], [4.0, 3.0, 2.0, 8.0], [0.5, 11.0, 0.1, 7.0]],
        dtype=np.float32,
    )
    if zero:
        fields[:, 2] = 0.0
        fields[0, 3] = 0.0
    arguments = inputs(backend, fields)
    function = (
        backend.predict_sorted_spikes_kde_log_likelihood
        if backend is sorted_spikes_kde
        else backend.predict_sorted_spikes_glm_log_likelihood
    )
    actual = function(**arguments, row_slice=rows)
    expected = reference(arguments, rows)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, **EXACT)
    if rows is not None:
        np.testing.assert_allclose(actual, function(**arguments)[rows], **EXACT)


@pytest.mark.parametrize("backend", [sorted_spikes_kde, sorted_spikes_glm])
def test_finite_nonlocal_fields_do_not_dispatch_xlogy_for_each_neuron(
    backend, monkeypatch, float32_arithmetic
):
    """The measured eager neuron-loop bottleneck must actually be removed."""
    fields = np.full((3, 4), 3.0, dtype=np.float32)
    arguments = inputs(backend, fields)
    original = jax.scipy.special.xlogy
    calls = []

    def record(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(jax.scipy.special, "xlogy", record)
    function = (
        backend.predict_sorted_spikes_kde_log_likelihood
        if backend is sorted_spikes_kde
        else backend.predict_sorted_spikes_glm_log_likelihood
    )
    actual = function(**arguments)
    actual.block_until_ready()
    np.testing.assert_allclose(actual, reference(arguments, None), **EXACT)
    assert len(calls) <= 1, "non-local prediction still dispatches per-neuron xlogy"


@pytest.mark.parametrize("backend", [sorted_spikes_kde, sorted_spikes_glm])
def test_many_neurons_preserve_singleton_and_ragged_chunk_values(backend):
    rng = np.random.default_rng(7301)
    edges = np.arange(514) * 0.002
    fields = jnp.asarray(rng.uniform(0.02, 40, (96, 257)).astype(np.float32))
    arguments = inputs(backend, np.ones((3, 4), np.float32))
    arguments.update(
        time_edges=edges,
        spike_times=[
            np.sort(
                np.r_[rng.uniform(0, edges[-1], 10 + index % 29), edges[100], edges[-1]]
            )
            for index in range(96)
        ],
        place_fields=fields,
        no_spike_part_log_likelihood=fields.sum(axis=0),
        is_track_interior=np.ones(257, bool),
    )
    if backend is sorted_spikes_kde:
        arguments.update(marginal_models=[None] * 96, mean_rates=np.ones(96))
    else:
        arguments["coefficients"] = np.zeros((96, 1))
    function = (
        backend.predict_sorted_spikes_kde_log_likelihood
        if backend is sorted_spikes_kde
        else backend.predict_sorted_spikes_glm_log_likelihood
    )
    full = function(**arguments)
    for rows in [
        slice(0, 1),
        slice(1, 57),
        slice(57, 300),
        slice(300, 513),
        slice(512, 513),
    ]:
        np.testing.assert_allclose(
            function(**arguments, row_slice=rows), full[rows], **EXACT
        )


def test_float32_parameters_preserve_enabled_x64_accumulator_dtype():
    from non_local_detector.likelihoods.common import _poisson_nonlocal_log_likelihood

    with x64_mode(True):
        counts = jnp.ones((2, 1), dtype=jnp.int32)
        rates = jnp.ones((1, 3), dtype=jnp.float32)
        durations = jnp.full(2, 0.002, dtype=jnp.float32)
        summed = jnp.ones(3, dtype=jnp.float32)
        expected = (
            jnp.zeros((2, 3))
            + jax.scipy.special.xlogy(counts[:, 0, None], rates * durations[:, None])
            - durations[:, None] * summed
        )
        actual = _poisson_nonlocal_log_likelihood(counts, rates, durations, summed)
        assert actual.dtype == expected.dtype == jnp.float64
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("kind", ["sparse", "coincident", "zero", "nonfinite"])
def test_host_counts_match_float64_oracle_and_keep_xlogy_fallback(x64, kind):
    """Matrix rows stay near the exact sum; unsafe products keep ordered xlogy."""
    from non_local_detector.likelihoods.common import _poisson_nonlocal_log_likelihood

    with x64_mode(x64):
        rng = np.random.default_rng(8173)
        dtype = np.float64 if x64 else np.float32
        # Dyadic rate/exposure products isolate addition order from the
        # backend's scalar-versus-vector elementary-function rounding.
        rates = jnp.asarray(np.exp2(rng.integers(-4, 5, (96, 17))).astype(dtype))
        counts = rng.poisson(0.025, (139, 96)).astype(np.int32)
        counts[0] = 0
        counts[1, :4] = [3, 0, 2, 1]
        if kind == "coincident":
            counts[100] = 1
            counts[-1] = 2
        elif kind == "zero":
            rates = rates.at[0, 0].set(0)
        elif kind == "nonfinite":
            rates = rates.at[0, 0].set(jnp.nan).at[1, 1].set(jnp.inf)
        durations = jnp.full(len(counts), 2**-9, dtype=dtype)
        summed = jnp.sum(rates, axis=0)
        expected = jnp.zeros((len(counts), rates.shape[1]))
        for neuron in range(len(rates)):
            expected += jax.scipy.special.xlogy(
                jnp.asarray(counts[:, neuron, None]),
                rates[neuron][None, :] * durations[:, None],
            )
        expected -= durations[:, None] * summed
        actual = _poisson_nonlocal_log_likelihood(counts, rates, durations, summed)
        if kind in ("zero", "nonfinite"):
            # Unsafe products take the sequential per-neuron xlogy fallback.
            np.testing.assert_array_equal(actual, expected)
        else:
            assert_within_ulps(
                actual, float64_emission(counts, rates, durations, summed), dtype
            )
        for rows in [slice(0, 1), slice(1, 73), slice(73, 139)]:
            np.testing.assert_array_equal(
                _poisson_nonlocal_log_likelihood(
                    counts[rows], rates, durations[rows], summed
                ),
                actual[rows],
            )


@pytest.mark.parametrize("host_counts", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
def test_poisson_accumulation_preserves_rate_and_duration_gradients(
    host_counts,
    uniform,
):
    from non_local_detector.likelihoods.common import _poisson_nonlocal_log_likelihood

    with x64_mode(True):
        counts = np.array([[0, 2, 0], [1, 0, 0], [0, 0, 0], [1, 0, 3]], dtype=np.int32)
        if not host_counts:
            counts = jnp.asarray(counts)
        rates = jnp.array([[1.5, 0.2], [3.0, 2.2], [0.4, 7.0]])
        durations = jnp.array(
            [
                0.002,
                0.002 if uniform else 0.003,
                0.002 if uniform else 0.004,
                0.002 if uniform else 0.005,
            ]
        )

        def reference(rates, durations):
            value = jnp.zeros((len(counts), rates.shape[1]))
            for neuron in range(len(rates)):
                value += jax.scipy.special.xlogy(
                    jnp.asarray(counts[:, neuron, None], dtype=rates.dtype),
                    rates[neuron] * durations[:, None],
                )
            return jnp.sum(value - durations[:, None] * rates.sum(0))

        def actual(rates, durations):
            return jnp.sum(
                _poisson_nonlocal_log_likelihood(counts, rates, durations, rates.sum(0))
            )

        for derivative in [jax.jacfwd, jax.jacrev]:
            for result, expected in zip(
                derivative(actual, argnums=(0, 1))(rates, durations),
                derivative(reference, argnums=(0, 1))(rates, durations),
                strict=True,
            ):
                np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def per_neuron_jax_reference(position_time, position, spike_times, **arguments):
    arguments = arguments | {
        "position_time": position_time,
        "position": position,
        "spike_times": spike_times,
    }
    """Original arithmetic, kept independent of the accumulation helper."""
    if arguments.get("is_local", False):
        return sorted_spikes_kde.predict_sorted_spikes_kde_log_likelihood(**arguments)
    edges = arguments["time_edges"]
    rows = arguments.get("row_slice")
    start, stop, _ = (slice(None) if rows is None else rows).indices(len(edges) - 1)
    stop = max(start, stop)
    durations = jnp.asarray(np.diff(edges)[start:stop])
    interior = arguments["is_track_interior"]
    likelihood = jnp.zeros((stop - start, int(np.sum(interior))))
    for spikes, rates in zip(
        arguments["spike_times"], arguments["place_fields"], strict=True
    ):
        counts = get_spikecount_per_time_bin(spikes, time_edges=edges, row_slice=rows)
        likelihood += jax.scipy.special.xlogy(
            counts[:, None], rates[interior][None, :] * durations[:, None]
        )
    return (
        likelihood
        - durations[:, None] * arguments["no_spike_part_log_likelihood"][interior]
    )


def float64_nonlocal_emission(spike_times, arguments):
    """The per-neuron reference's emission, summed exactly in float64."""
    edges = arguments["time_edges"]
    rows = arguments.get("row_slice")
    start, stop, _ = (slice(None) if rows is None else rows).indices(len(edges) - 1)
    stop = max(start, stop)
    interior = arguments["is_track_interior"]
    fields = np.asarray(arguments["place_fields"])
    durations = np.asarray(np.diff(edges)[start:stop]).astype(fields.dtype)
    counts = np.stack(
        [
            get_spikecount_per_time_bin(spikes, time_edges=edges, row_slice=rows)
            for spikes in spike_times
        ],
        axis=1,
    )
    return float64_emission(
        counts,
        fields[:, interior],
        durations,
        np.asarray(arguments["no_spike_part_log_likelihood"])[interior],
    )


def float64_oracle_reference(position_time, position, spike_times, **arguments):
    """Correctly rounded non-local emission; the local state is unchanged."""
    if arguments.get("is_local", False):
        return per_neuron_jax_reference(
            position_time, position, spike_times, **arguments
        )
    exact = float64_nonlocal_emission(spike_times, arguments)
    return jnp.asarray(exact.astype(np.asarray(arguments["place_fields"]).dtype))


@pytest.mark.parametrize("nonlocal_model", [False, True])
def test_many_neurons_match_native_posterior_and_evidence(nonlocal_model, monkeypatch):
    from non_local_detector import (
        Environment,
        NonLocalSortedSpikesDetector,
        SortedSpikesDecoder,
    )
    from non_local_detector.continuous_state_transitions import Uniform
    from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS

    rng = np.random.default_rng(7303)
    position_time = np.linspace(0, 4, 501)
    position = (10 + 8 * np.sin(position_time * 2))[:, None]
    cls = NonLocalSortedSpikesDetector if nonlocal_model else SortedSpikesDecoder
    model = cls(
        environments=Environment(place_bin_size=2, position_range=((0, 20),)),
        continuous_transition_types=None if nonlocal_model else [[Uniform()]],
        infer_track_interior=False,
        sorted_spikes_algorithm_params={
            "position_std": 3,
            "block_size": 100,
            "disable_progress_bar": True,
        },
    )
    training_spikes = [np.sort(rng.uniform(0, 4, 12)) for _ in range(96)]
    model.fit(position_time, position, training_spikes)
    edges = np.arange(514) * 0.002
    spikes = [
        np.sort(np.r_[rng.uniform(0, edges[-1], 4), edges[100], edges[-1]])
        for _ in range(96)
    ]
    missing = np.zeros(513, bool)
    missing[240:250] = True
    arguments = {
        "time_edges": edges,
        "position_time": position_time,
        "position": position,
        "is_missing": missing,
        "return_outputs": "all",
    }
    fitter, optimized = _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"]
    emissions = []

    def recording(position_time, position, spike_times, **kwargs):
        result = optimized(position_time, position, spike_times, **kwargs)
        if not kwargs.get("is_local", False):
            emissions.append((spike_times, kwargs, np.asarray(result)))
        return result

    results = {}
    for name, likelihood in [
        ("reference", per_neuron_jax_reference),
        ("oracle", float64_oracle_reference),
        ("optimized", recording),
    ]:
        with monkeypatch.context() as context:
            context.setitem(
                _SORTED_SPIKES_ALGORITHMS, "sorted_spikes_kde", (fitter, likelihood)
            )
            results[name] = model.predict(spikes, **arguments)
    assert _SORTED_SPIKES_ALGORITHMS["sorted_spikes_kde"][1] is optimized

    # The float32 emission stays within a few ulp of the exact float64 sum.
    assert emissions
    for spike_times, kwargs, emission in emissions:
        exact = float64_nonlocal_emission(spike_times, kwargs)
        assert_within_ulps(emission, exact, np.float32)

    # Downstream outputs are no further from the correctly rounded emission's
    # outputs than the per-neuron float32 reference is.
    reference, oracle, actual = (
        results["reference"],
        results["oracle"],
        results["optimized"],
    )
    for name in oracle.data_vars:
        bound = np.max(np.abs(np.asarray(reference[name]) - np.asarray(oracle[name])))
        error = np.max(np.abs(np.asarray(actual[name]) - np.asarray(oracle[name])))
        assert error <= 2 * bound + 1e-6, (name, error, bound)
    evidence = [
        np.asarray(result.attrs["marginal_log_likelihoods"])
        for result in (reference, oracle, actual)
    ]
    assert np.max(np.abs(evidence[2] - evidence[1])) <= (
        2 * np.max(np.abs(evidence[0] - evidence[1])) + 1e-6
    )
    np.testing.assert_array_equal(actual.is_missing, missing)
