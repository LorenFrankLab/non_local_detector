"""Preserve local/No-Spike Poisson arithmetic while batching dispatches."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods import (
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


def test_large_no_spike_request_keeps_bounded_neuron_workspace(monkeypatch):
    def oversized(*args):
        raise AssertionError("Full rows times population count matrix is unbounded")

    monkeypatch.setattr(no_spike, "_poisson_row_log_likelihood", oversized)
    edges = np.arange(3001) * 0.002
    events = spikes(3)
    result = no_spike.predict_no_spike_log_likelihood(events, time_edges=edges)
    expected = no_spike_reference(events, edges, 1e-10, slice(None))
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_large_local_kde_request_does_not_trace_recording_sized_graph(monkeypatch):
    def oversized(*args, **kwargs):
        raise AssertionError("Tracing all KDE evaluation blocks at once is unbounded")

    monkeypatch.setattr(sorted_spikes_kde, "_local_kde_log_likelihood", oversized)
    model = KDEModel(std=1.0, block_size=100).fit(jnp.ones((3, 1)))
    edges = np.arange(3001) * 0.002
    result = sorted_spikes_kde.predict_sorted_spikes_kde_log_likelihood(
        position_time=np.array([0.0, 6.0]),
        position=np.array([[0.0], [1.0]]),
        spike_times=[np.array([0.006])],
        environment=None,
        marginal_models=[model],
        occupancy_model=model,
        occupancy=None,
        mean_rates=[2.0],
        place_fields=np.ones((1, 1)),
        no_spike_part_log_likelihood=np.ones(1),
        is_track_interior=np.ones(1, bool),
        disable_progress_bar=True,
        is_local=True,
        time_edges=edges,
    )
    assert result.shape == (3000, 1)
    np.testing.assert_allclose(result[3], np.log(0.004) - 0.004, rtol=1e-6, atol=1e-6)


def test_large_local_glm_request_keeps_bounded_neuron_workspace(monkeypatch):
    def oversized(*args):
        raise AssertionError("Full rows times population rate matrix is unbounded")

    monkeypatch.setattr(sorted_spikes_glm, "_local_glm_log_likelihood", oversized)
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
