"""Tests for the sorted-spikes MRF-GAM likelihood.

``sorted_spikes_mrf`` fits a penalized-Poisson GAM whose reduced-rank basis is the
smoothest eigenmodes of the environment's graph Laplacian (the same spectral engine
as ``sorted_spikes_diffusion``) and whose penalty is those eigenvalues, with
occupancy entering as a log-offset (never a denominator) and the smoothing
parameter chosen by REML. These tests pin the vectorized population fit against an
independent per-neuron reference, the eigenvalue-ridge penalty, occupancy-offset
robustness, REML field recovery, no cross-wall leakage, and the encoding contract.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.linalg

from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    diffusion_eigenbasis,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    predict_sorted_spikes_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_mrf import (
    fit_sorted_spikes_mrf_encoding_model,
    mrf_penalized_poisson_fit,
    predict_sorted_spikes_mrf_log_likelihood,
    select_penalty_by_reml,
)
from non_local_detector.tests.likelihoods.test_sorted_spikes_diffusion import (
    make_2d_env,
    simulate_place_data,
)

ENCODING_DICT_KEYS = {
    "environment",
    "occupancy",
    "mean_rates",
    "place_fields",
    "no_spike_part_log_likelihood",
    "is_track_interior",
    "node_order",
    "bin_sizes",
    "disable_progress_bar",
}


def path_graph(n_nodes, distance=1.0):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for node in range(n_nodes - 1):
        graph.add_edge(node, node + 1, distance=distance)
    return graph


def reference_fit_one_neuron(counts, occupancy, basis, penalty_weights, penalty):
    """Independent per-neuron penalized-Poisson IRLS reference (mgcv `_fit_dense`).

    Fits ``counts ~ Poisson(occupancy * exp(basis @ gamma))`` with penalty
    ``penalty * gamma^T diag(penalty_weights) gamma`` via Newton/IRLS.
    """
    penalty_diag = penalty * penalty_weights
    eta = np.full(
        basis.shape[0],
        np.log(max(counts.sum() / max(occupancy.sum(), 1e-9), 1e-6)),
    )
    gamma = np.linalg.lstsq(basis, eta, rcond=None)[0]
    for _ in range(100):
        eta = basis @ gamma
        mu = occupancy * np.exp(np.clip(eta, -30, 30))
        grad = basis.T @ (counts - mu) - penalty_diag * gamma
        hessian = basis.T @ (basis * mu[:, None])
        hessian[np.diag_indices_from(hessian)] += penalty_diag + 1e-10
        cho = scipy.linalg.cho_factor(hessian, lower=True)
        step = scipy.linalg.cho_solve(cho, grad)
        gamma = gamma + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return gamma


def test_population_fit_matches_per_neuron_reference():
    """The vectorized population fit equals an independent per-neuron IRLS reference.

    Covers both "population fit == per-neuron loop" and "matches a penalized-Poisson
    reference" — the reference is a distinct implementation (Cholesky-based, one
    neuron at a time).
    """
    rng = np.random.default_rng(0)
    n_bins, rank, n_neurons = 40, 6, 5
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=rank
    )
    occupancy = rng.uniform(0.5, 3.0, size=n_bins)
    counts = rng.poisson(1.5, size=(n_bins, n_neurons)).astype(float)
    penalty = 2.0

    coeffs, eta, mu = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty
    )

    assert coeffs.shape == (rank, n_neurons)
    assert eta.shape == (n_bins, n_neurons)
    assert mu.shape == (n_bins, n_neurons)
    for neuron in range(n_neurons):
        expected = reference_fit_one_neuron(
            counts[:, neuron], occupancy, basis, penalty_weights, penalty
        )
        np.testing.assert_allclose(coeffs[:, neuron], expected, atol=1e-6)
    # eta / mu are consistent with the fitted coefficients.
    np.testing.assert_allclose(eta, basis @ coeffs, atol=1e-10)
    np.testing.assert_allclose(mu, occupancy[:, None] * np.exp(eta), rtol=1e-10)


def test_basis_is_smoothest_modes_and_penalty_is_eigenvalue_ridge():
    """The design basis columns are the smoothest Laplacian eigenmodes (ascending
    eigenvalue), and the penalty weights are those eigenvalues (a generalized ridge)."""
    laplacian = build_laplacian(path_graph(50))
    penalty_weights, basis = diffusion_eigenbasis(laplacian, rank=10)

    # Ascending eigenvalues, first is the (constant) null mode.
    assert np.all(np.diff(penalty_weights) >= -1e-12)
    assert penalty_weights[0] < 1e-9
    # basis columns satisfy L v = d v (they are eigenvectors of the penalty).
    dense = laplacian.toarray()
    for mode in range(10):
        np.testing.assert_allclose(
            dense @ basis[:, mode], penalty_weights[mode] * basis[:, mode], atol=1e-8
        )


def test_reml_recovers_smooth_field():
    """REML selects a sensible lambda whose fitted rate map tracks a simulated field."""
    rng = np.random.default_rng(1)
    n_bins = 60
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=25
    )
    x = np.arange(n_bins)
    true_log_rate = 2.5 * np.exp(-((x - 30.0) ** 2) / (2 * 6.0**2)) - 0.5
    occupancy = rng.uniform(8.0, 20.0, size=n_bins)  # enough dwell for a clear field
    counts = rng.poisson(occupancy * np.exp(true_log_rate)).astype(float)[:, None]

    penalty = select_penalty_by_reml(counts, occupancy, basis, penalty_weights)
    assert 0.0 < penalty < np.inf

    _, eta, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty
    )
    corr = np.corrcoef(np.exp(eta[:, 0]), np.exp(true_log_rate))[0, 1]
    assert corr > 0.95


def test_occupancy_offset_gives_finite_rate_at_zero_occupancy():
    """Occupancy enters as an exposure offset, so zero-occupancy bins yield finite
    (unconstrained, smoothly-interpolated) rates — no division blow-up."""
    rng = np.random.default_rng(2)
    n_bins = 40
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=12
    )
    occupancy = rng.uniform(1.0, 2.0, size=n_bins)
    occupancy[18:22] = 0.0  # an unvisited patch (would divide-by-zero in a ratio)
    counts = rng.poisson(1.0, size=(n_bins, 3)).astype(float)
    counts[18:22, :] = 0.0  # no spikes where there is no occupancy

    _, eta, mu = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1.0
    )
    rate = np.exp(eta)
    assert np.all(np.isfinite(rate))
    assert np.all(rate > 0.0)
    # mu is exactly zero where occupancy is zero (offset, not denominator).
    np.testing.assert_array_equal(mu[18:22, :], 0.0)


def test_mrf_penalty_does_not_smooth_across_a_wall():
    """The eigenbasis is component-local, so a cell firing in one component cannot
    raise the fitted rate in a disconnected one (no smoothing across a wall)."""
    graph = nx.disjoint_union(path_graph(20), path_graph(20))
    penalty_weights, basis = diffusion_eigenbasis(build_laplacian(graph), rank=None)
    occupancy = np.ones(40)
    counts = np.zeros((40, 1))
    counts[5, 0] = 30.0  # all spikes in the first component (nodes 0..19)

    _, eta, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1.0
    )
    rate = np.exp(eta[:, 0])
    assert rate[20:].max() < 0.02 * rate[:20].max()


def test_fit_encoding_dict_keys_and_full_grid_shapes():
    """The MRF fit returns the shared encoding contract; place_fields are FULL-GRID."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=4)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=30,
    )
    assert ENCODING_DICT_KEYS <= set(encoding)

    is_interior = env.is_track_interior_.ravel()
    n_total = env.place_bin_centers_.shape[0]
    place_fields = np.asarray(encoding["place_fields"])
    assert place_fields.shape == (4, n_total)
    assert np.all(np.isfinite(place_fields))
    assert np.all(place_fields[:, ~is_interior] == 0.0)
    assert np.all(place_fields[:, is_interior] > 0.0)
    np.testing.assert_allclose(
        np.asarray(encoding["no_spike_part_log_likelihood"]),
        place_fields.sum(axis=0),
        rtol=1e-6,
    )


def test_fit_finite_under_near_zero_occupancy_and_low_penalty():
    """A near-zero-occupancy bin with a coincident spike at a low penalty must not
    overflow to inf: the returned eta/mu are clipped (finite-log-prob invariant)."""
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(40)), rank=None
    )
    occupancy = np.ones(40)
    occupancy[10] = 1e-12  # essentially unvisited
    counts = np.zeros((40, 1))
    counts[10, 0] = 5.0  # spikes where there is ~no occupancy -> huge saturated rate

    _, eta, mu = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1e-4
    )
    assert np.all(np.isfinite(eta))
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(np.exp(eta)))
    assert np.abs(eta).max() <= 30.0 + 1e-9  # clipped to _ETA_CLIP


def test_low_penalty_full_rank_tracks_saturated_ratio():
    """At a low penalty and full rank, exp(eta) approaches the saturated MLE
    counts/occupancy -- substantiating that exp(eta) is a rate (spikes per sample)."""
    rng = np.random.default_rng(4)
    n_bins = 50
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=None
    )
    occupancy = rng.uniform(20.0, 40.0, size=n_bins)  # well sampled
    true_rate = 0.5 + 0.3 * np.sin(np.arange(n_bins) / 5.0)
    counts = rng.poisson(occupancy * true_rate).astype(float)[:, None]

    _, eta, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1e-4
    )
    saturated_ratio = counts[:, 0] / occupancy
    corr = np.corrcoef(np.exp(eta[:, 0]), saturated_ratio)[0, 1]
    assert corr > 0.95


def test_fit_empty_spike_times_returns_empty_place_fields():
    """Zero neurons (spike_times=[]) must not crash and yields empty place fields."""
    env = make_2d_env()
    time, position, _ = simulate_place_data(env, n_neurons=1)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=[],
        environment=env,
        rank=20,
    )
    n_total = env.place_bin_centers_.shape[0]
    assert np.asarray(encoding["place_fields"]).shape == (0, n_total)
    assert np.asarray(encoding["no_spike_part_log_likelihood"]).shape == (n_total,)


def test_penalty_zero_is_respected_not_overridden():
    """An explicit penalty=0.0 (unpenalized) must not be silently replaced by 1.0."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=2)

    def fit(penalty):
        return np.asarray(
            fit_sorted_spikes_mrf_encoding_model(
                position_time=time,
                position=position,
                spike_times=spike_times,
                environment=env,
                rank=20,
                penalty=penalty,
            )["place_fields"]
        )

    assert not np.allclose(fit(0.0), fit(1.0))


@pytest.mark.property
def test_invariants_place_fields_and_likelihood():
    """Place fields >= 0 & finite; likelihoods finite; softmax posterior sums to 1."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=25,
    )
    place_fields = np.asarray(encoding["place_fields"])
    assert np.all(place_fields >= 0.0)
    assert np.all(np.isfinite(place_fields))

    ll = np.asarray(
        predict_sorted_spikes_mrf_log_likelihood(
            time[:100], time, position, spike_times, is_local=False, **encoding
        )
    )
    assert np.all(np.isfinite(ll))
    posterior = np.exp(ll - ll.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    # float32-safe: the likelihood is float32 when JAX x64 is off.
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-6)


def test_registered_and_predict_shared_with_diffusion():
    """sorted_spikes_mrf is registered and reuses the diffusion Poisson prediction."""
    assert "sorted_spikes_mrf" in _SORTED_SPIKES_ALGORITHMS
    fit_fn, predict_fn = _SORTED_SPIKES_ALGORITHMS["sorted_spikes_mrf"]
    assert fit_fn is fit_sorted_spikes_mrf_encoding_model
    assert predict_fn is predict_sorted_spikes_mrf_log_likelihood
    # The prediction is literally the diffusion predict (shared place-field contract).
    assert (
        predict_sorted_spikes_mrf_log_likelihood
        is predict_sorted_spikes_diffusion_log_likelihood
    )


def test_predict_shapes_local_and_nonlocal():
    """Non-local -> (n_time, n_interior); local -> (n_time, 1); both finite."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=30,
    )
    decode_time = time[:150]
    n_interior = int(env.is_track_interior_.ravel().sum())

    nonlocal_ll = predict_sorted_spikes_mrf_log_likelihood(
        decode_time, time, position, spike_times, is_local=False, **encoding
    )
    assert nonlocal_ll.shape == (decode_time.shape[0], n_interior)
    assert np.all(np.isfinite(nonlocal_ll))

    local_ll = predict_sorted_spikes_mrf_log_likelihood(
        decode_time, time, position, spike_times, is_local=True, **encoding
    )
    assert local_ll.shape == (decode_time.shape[0], 1)
    assert np.all(np.isfinite(local_ll))


@pytest.mark.integration
def test_end_to_end_sorted_spikes_mrf_decoder():
    """A full SortedSpikesDecoder run with the MRF algorithm decodes a valid
    posterior (sums to 1, finite) and recovers the trajectory."""
    from non_local_detector.models import SortedSpikesDecoder
    from non_local_detector.simulate.sorted_spikes_simulation import (
        make_simulated_data,
    )

    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=15)

    decoder = SortedSpikesDecoder(
        sorted_spikes_algorithm="sorted_spikes_mrf",
        sorted_spikes_algorithm_params={"rank": 40},
    ).fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=~is_event,
    )
    results = decoder.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        save_log_likelihood_to_results=False,
    )

    posterior = results.acausal_posterior
    assert posterior.shape[0] == len(time)
    assert np.all(np.isfinite(posterior.values))
    np.testing.assert_allclose(posterior.sum("state_bins").values, 1.0, atol=1e-5)

    bin_position = np.asarray(posterior["position"].values)
    decoded_position = bin_position[np.asarray(posterior.values).argmax(axis=1)]
    corr = np.corrcoef(decoded_position, np.asarray(position).ravel())[0, 1]
    assert corr > 0.9
