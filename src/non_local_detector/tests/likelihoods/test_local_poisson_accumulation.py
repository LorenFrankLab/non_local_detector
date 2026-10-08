"""Preserve local/No-Spike Poisson arithmetic while batching dispatches."""

from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods import (
    common,
    no_spike,
    sorted_spikes_glm,
    sorted_spikes_kde,
)
from non_local_detector.likelihoods.common import (
    EPS,
    RATE_EPS_HZ,
    KDEModel,
    get_spikecount_per_time_bin,
)

pytestmark = pytest.mark.unit


@contextmanager
def precision_mode(x64):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def spikes(population):
    rng = np.random.default_rng(7361)
    return [
        rng.permutation(np.r_[rng.uniform(0, 0.034, unit % 9), 0.006, 0.034])
        for unit in range(population)
    ]


def no_spike_reference(events, edges, rate, rows):
    durations = jnp.asarray(np.diff(edges)[rows]) * rate
    result = jnp.zeros((len(np.diff(edges)[rows]),))
    for events_for_unit in events:
        counts = get_spikecount_per_time_bin(
            events_for_unit, time_edges=edges, row_slice=rows
        )
        result += jax.scipy.special.xlogy(counts, durations) - durations
    return result[:, None]


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("population", [0, 3, 64])
@pytest.mark.parametrize("rate", [0.0, 1e-10, 2.0, -2.0, np.inf, np.nan])
def test_no_spike_matches_original_neuron_order(x64, population, rate):
    edges = np.arange(18) * 0.002
    rows = slice(2, 17)
    events = spikes(population)
    with precision_mode(x64):
        expected = no_spike_reference(events, edges, rate, rows)
        actual = no_spike.predict_no_spike_log_likelihood(
            events, no_spike_rate=rate, time_edges=edges, row_slice=rows
        )
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("x64", [False, True])
def test_compiled_local_glm_matches_per_neuron_rates_and_gradient(x64):
    rng = np.random.default_rng(7363)
    with precision_mode(x64):
        design = jnp.asarray(rng.normal(size=(17, 9)))
        coefficients = jnp.asarray(rng.normal(0, 0.2, size=(8, 9)))
        counts = jnp.asarray(rng.integers(0, 3, size=(17, 8)))
        durations = jnp.full(17, 0.002)

        def reference(coefs):
            total = jnp.zeros(17)
            for unit, coef in enumerate(coefs):
                expected_counts = (
                    jnp.clip(jnp.exp(design @ coef), min=RATE_EPS_HZ, max=None)
                    * durations
                )
                total += (
                    jax.scipy.special.xlogy(
                        counts[:, unit].astype(expected_counts.dtype), expected_counts
                    )
                    - expected_counts
                )
            return total

        def candidate(coefs):
            return sorted_spikes_glm._local_glm_log_likelihood(
                design, coefs, counts, durations
            )

        np.testing.assert_allclose(
            candidate(coefficients), reference(coefficients), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            jax.grad(lambda coef: candidate(coef).sum())(coefficients),
            jax.grad(lambda coef: reference(coef).sum())(coefficients),
            rtol=1e-6,
            atol=1e-6,
        )


def test_compiled_local_glm_empty_population_keeps_zero_likelihood():
    result = sorted_spikes_glm._local_glm_log_likelihood(
        jnp.ones((17, 9)), jnp.asarray([]), jnp.zeros((17, 0)), jnp.full(17, 0.002)
    )
    np.testing.assert_array_equal(result, np.zeros(17))


def record_rows(monkeypatch, module, name, argument=0):
    """Wrap a compiled kernel to record the rows of each call's ``argument``."""
    original = getattr(module, name)
    rows = []

    def wrapper(*args, **kwargs):
        rows.append(args[argument].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return rows


def small_count_blocks(monkeypatch, n_neurons, block_rows=64):
    monkeypatch.setattr(common, "COUNT_BLOCK_BYTES", block_rows * n_neurons * 8)
    assert common._count_block_rows(n_neurons) == block_rows


def test_large_no_spike_request_keeps_bounded_neuron_workspace(monkeypatch):
    events = spikes(3)
    small_count_blocks(monkeypatch, len(events))
    rows = record_rows(monkeypatch, no_spike, "_poisson_row_log_likelihood")
    edges = np.arange(3001) * 0.002
    result = no_spike.predict_no_spike_log_likelihood(events, time_edges=edges)
    expected = no_spike_reference(events, edges, 1e-10, slice(None))
    assert max(rows) == 64 and sum(rows) == 3000
    np.testing.assert_array_equal(result, expected)


def local_kde_problem(n_rows, block_size):
    rng = np.random.default_rng(7364)
    models = [
        KDEModel(std=std, block_size=block_size).fit(
            jnp.asarray(rng.uniform(0, 1, (size, 1)))
        )
        for size, std in zip([5, 9, 13], [0.2, 0.3, 0.5], strict=True)
    ]
    edges = np.arange(n_rows + 1) * 0.002
    kwargs = {
        "position_time": np.array([0.0, edges[-1]]),
        "position": np.array([[0.0], [1.0]]),
        "spike_times": [rng.uniform(0, edges[-1], 40) for _ in models],
        "environment": None,
        "marginal_models": models,
        "occupancy_model": models[-1],
        "occupancy": None,
        "mean_rates": [2.0, 5.0, 0.5],
        "place_fields": np.ones((len(models), 1)),
        "no_spike_part_log_likelihood": np.ones(1),
        "is_track_interior": np.ones(1, bool),
        "disable_progress_bar": True,
        "is_local": True,
        "time_edges": edges,
    }
    return models, kwargs


def local_kde_reference(models, kwargs):
    edges = kwargs["time_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    points = jnp.asarray(centers / edges[-1])[:, None]
    occupancy = kwargs["occupancy_model"].predict(points)
    durations = jnp.asarray(np.diff(edges))
    total = jnp.zeros(len(centers))
    for events, model, mean_rate in zip(
        kwargs["spike_times"], models, kwargs["mean_rates"], strict=True
    ):
        counts = get_spikecount_per_time_bin(events, time_edges=edges)
        rate = mean_rate * jnp.where(
            occupancy > 0,
            model.predict(points) / jnp.where(occupancy > 0, occupancy, 1.0),
            EPS,
        )
        expected = jnp.clip(rate, min=RATE_EPS_HZ, max=None) * durations
        total += jax.scipy.special.xlogy(counts, expected) - expected
    return total[:, None]


@pytest.mark.parametrize("block_size", [100, None])
def test_large_local_kde_request_uses_bounded_compiled_blocks(monkeypatch, block_size):
    models, kwargs = local_kde_problem(3000, block_size)
    small_count_blocks(monkeypatch, len(models))
    rows = record_rows(monkeypatch, sorted_spikes_kde, "_local_kde_log_likelihood")
    result = sorted_spikes_kde.predict_sorted_spikes_kde_log_likelihood(**kwargs)
    assert max(rows) == 64 and sum(rows) == 3000
    np.testing.assert_allclose(
        result, local_kde_reference(models, kwargs), rtol=1e-6, atol=1e-6
    )


def test_local_kde_rows_do_not_depend_on_count_blocks(monkeypatch):
    models, kwargs = local_kde_problem(300, None)
    unblocked = sorted_spikes_kde.predict_sorted_spikes_kde_log_likelihood(**kwargs)
    small_count_blocks(monkeypatch, len(models))
    blocked = sorted_spikes_kde.predict_sorted_spikes_kde_log_likelihood(**kwargs)
    np.testing.assert_array_equal(blocked, unblocked)


def test_compiled_local_kde_graph_does_not_grow_with_rows():
    model = KDEModel(std=1.0, block_size=100).fit(jnp.ones((3, 1)))
    leaves = ((model.samples_, model.weights_, model.std),)

    def n_equations(n_rows):
        jaxpr = jax.make_jaxpr(
            partial(
                sorted_spikes_kde._local_kde_log_likelihood.__wrapped__,
                block_sizes=(100,),
            )
        )(
            jnp.zeros((n_rows, 1)),
            jnp.ones(n_rows),
            jnp.zeros((n_rows, 1), dtype=int),
            jnp.ones(1),
            jnp.ones(n_rows),
            leaves,
        )
        return len(jaxpr.jaxpr.eqns)

    assert n_equations(4096) == n_equations(1050)


@pytest.mark.parametrize("n_points", [0, 1, 55, 56, 300])
def test_traced_block_kde_matches_block_kde(n_points):
    rng = np.random.default_rng(7365)
    points = jnp.asarray(rng.normal(size=(n_points, 2)))
    samples = jnp.asarray(rng.normal(size=(9, 2)))
    weights = jnp.asarray(rng.uniform(0.1, 1, 9))
    std = jnp.asarray([0.7, 1.3])
    # Block size 7: up to 56 points are unrolled, 300 points use the loop.
    actual = jax.jit(common._traced_block_kde, static_argnums=3)(
        points, samples, std, 7, weights
    )
    expected = common.block_kde(points, samples, std, 7, weights)
    np.testing.assert_array_equal(actual, expected)


def test_large_local_glm_request_keeps_bounded_neuron_workspace(monkeypatch):
    small_count_blocks(monkeypatch, 1)
    rows = record_rows(monkeypatch, sorted_spikes_glm, "_local_glm_log_likelihood")
    monkeypatch.setattr(
        sorted_spikes_glm,
        "make_spline_predict_matrix",
        lambda info, points: jnp.ones((len(points), 2)),
    )
    result = sorted_spikes_glm.predict_sorted_spikes_glm_log_likelihood(
        position_time=np.array([0.0, 6.0]),
        position=np.array([[0.0], [1.0]]),
        spike_times=[np.array([0.006])],
        environment=None,
        coefficients=jnp.zeros((1, 2)),
        emission_design_info=None,
        place_fields=np.ones((1, 1)),
        no_spike_part_log_likelihood=np.ones(1),
        is_track_interior=np.ones(1, bool),
        disable_progress_bar=True,
        is_local=True,
        time_edges=np.arange(3001) * 0.002,
    )
    assert max(rows) == 64 and sum(rows) == 3000
    assert result.shape == (3000, 1)
    np.testing.assert_allclose(result[3], np.log(0.002) - 0.002, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("x64", [False, True])
def test_compiled_local_kde_accepts_vector_positions(x64):
    with precision_mode(x64):
        points = jnp.linspace(0, 1, 17)
        model = KDEModel(std=1.0, block_size=7).fit(jnp.array([0.0, 0.5, 1.0]))
        occupancy = model.predict(points)
        counts = jnp.ones((17, 1))
        means = jnp.array([2.0])
        durations = jnp.full(17, 0.002)
        actual = sorted_spikes_kde._local_kde_log_likelihood(
            points,
            occupancy,
            counts,
            means,
            durations,
            ((model.samples_, model.weights_, model.std),),
            block_sizes=(7,),
        )
        expected = (
            jax.scipy.special.xlogy(counts[:, 0], 2.0 * durations) - 2.0 * durations
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_no_spike_dispatch_does_not_scale_with_population(monkeypatch):
    original = jax.scipy.special.xlogy
    calls = []

    def record(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(jax.scipy.special, "xlogy", record)
    jax.clear_caches()
    actual = no_spike.predict_no_spike_log_likelihood(
        spikes(64), time_edges=np.arange(18) * 0.002
    )
    jax.block_until_ready(actual)
    assert len(calls) <= 1


@pytest.mark.parametrize("x64", [False, True])
def test_no_spike_rate_gradient_keeps_physical_exposure(x64):
    events = spikes(3)
    edges = np.arange(18) * 0.002
    with precision_mode(x64):
        derivative = jax.grad(
            lambda rate: no_spike.predict_no_spike_log_likelihood(
                events, no_spike_rate=rate, time_edges=edges
            ).sum()
        )(2.0)
        counts = sum(len(unit) for unit in events)
        expected = counts / 2.0 - len(events) * (edges[-1] - edges[0])
        np.testing.assert_allclose(derivative, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("n_rows", [0, 1, 17])
@pytest.mark.parametrize("case", ["ordinary", "missing", "zero_occupancy"])
def test_compiled_local_kde_preserves_each_model_and_addition_order(x64, n_rows, case):
    rng = np.random.default_rng(7362)
    with precision_mode(x64):
        points = jnp.asarray(rng.normal(size=(n_rows, 2)))
        if case == "missing" and n_rows:
            points = points.at[0].set(jnp.nan)
        models = [
            KDEModel(std=std, block_size=7).fit(
                jnp.asarray(rng.normal(size=(size, 2))),
                jnp.asarray(rng.uniform(0, 2, size)),
            )
            for size, std in zip([0, 1, 7, 19], [1.0, 0.75, 1.25, 2.0], strict=True)
        ]
        occupancy = jnp.asarray(rng.uniform(0.01, 0.5, n_rows))
        if case == "zero_occupancy":
            occupancy = jnp.zeros(n_rows)
        counts = jnp.asarray(rng.integers(0, 3, size=(n_rows, 4)))
        means = jnp.asarray([0.0, 1e-10, 2.0, 7.5])
        durations = jnp.full(n_rows, 0.002)
        expected = jnp.zeros(n_rows)
        for unit, model in enumerate(models):
            marginal = model.predict(points)
            marginal = jnp.where(jnp.isnan(marginal), 0.0, marginal)
            rate = means[unit] * jnp.where(
                occupancy > 0,
                marginal / jnp.where(occupancy > 0, occupancy, 1.0),
                EPS,
            )
            rate = jnp.clip(rate, min=RATE_EPS_HZ, max=None) * durations
            expected += jax.scipy.special.xlogy(counts[:, unit], rate) - rate
        actual = sorted_spikes_kde._local_kde_log_likelihood(
            points,
            occupancy,
            counts,
            means,
            durations,
            tuple((model.samples_, model.weights_, model.std) for model in models),
            block_sizes=(7,) * len(models),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
