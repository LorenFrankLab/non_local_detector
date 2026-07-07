"""Tests for the sorted-spikes graph-diffusion likelihood.

``sorted_spikes_diffusion`` is a model-level drop-in for ``sorted_spikes_kde``:
same encoding-dict contract and Poisson log-likelihood, but place fields are
smoothed on the environment's manifold graph (heat kernel) instead of with a
Gaussian KDE. These tests pin the encoding-dict shapes, KDE equivalence on a
wall-less field, weighted-fit parity, predict shapes, and the splat contract.
"""

import inspect

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from non_local_detector.likelihoods.common import EPS, get_position_at_time
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    fit_sorted_spikes_diffusion_encoding_model,
    predict_sorted_spikes_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)

# Encoding-dict keys the fit must return: the KDE parity keys (minus the KDE model
# objects) plus the diffusion additions node_order / bin_sizes (shared contract).
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


def make_2d_env(seed: int = 0) -> Environment:
    """2D open field with an inferred interior (padding bins stay non-interior)."""
    rng = np.random.default_rng(seed)
    position = rng.uniform(1.0, 49.0, size=(4000, 2))
    return Environment(
        environment_name="of",
        place_bin_size=5.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(position, infer_track_interior=True)


def simulate_place_data(env, n_neurons=3, n_time=3000, seed=1, sampling_frequency=100):
    """A folded random-walk trajectory with Gaussian place-tuned Poisson spikes."""
    rng = np.random.default_rng(seed)
    interior = env.place_bin_centers_[env.is_track_interior_.ravel()]
    lo, hi = interior.min(axis=0), interior.max(axis=0)
    span = hi - lo

    time = np.arange(n_time) / sampling_frequency
    walk = np.cumsum(rng.normal(0.0, 2.0, size=(n_time, lo.size)), axis=0)
    # Fold the unbounded walk into [lo, hi] with a triangle wave (reflecting walls).
    position = lo + np.abs((walk % (2 * span)) - span)

    centers = rng.uniform(lo, hi, size=(n_neurons, lo.size))
    dt = 1.0 / sampling_frequency
    spike_times = []
    for center in centers:
        rate = 40.0 * np.exp(-((position - center) ** 2).sum(axis=1) / (2 * 6.0**2))
        fired = rng.random(n_time) < rate * dt
        spike_times.append(time[fired])
    return time, position, spike_times


def place_field_core_median_rel_errors(env, std, diff_pf, kde_pf):
    """Per-neuron median |diffusion - KDE| / KDE at the field's core.

    "Core" = interior bins eroded by ``round(std / bin_size)`` (past the smoother's
    reach, where the reflecting-vs-edge-biased boundary treatments differ) AND above
    0.4 * peak (where the field is well estimated). Shared by the drop-in and
    weighted-parity tests so both use the same honest, boundary-aware methodology.
    """
    erode = int(round(std / env.place_bin_size))
    core = binary_erosion(
        env.is_track_interior_.reshape(env.centers_shape_), iterations=erode
    ).ravel()
    medians = []
    for neuron in range(diff_pf.shape[0]):
        peak = kde_pf[neuron].max()
        substantial = core & (kde_pf[neuron] > 0.4 * peak)
        rel_err = (
            np.abs(diff_pf[neuron, substantial] - kde_pf[neuron, substantial])
            / kde_pf[neuron, substantial]
        )
        medians.append(float(np.median(rel_err)))
    return medians


def test_fit_encoding_dict_keys_and_full_grid_shapes():
    """Fit returns the contract keys; place_fields are FULL-GRID with zeros off-track."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)

    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=6.0,
    )

    assert ENCODING_DICT_KEYS <= set(encoding)

    n_total = env.place_bin_centers_.shape[0]
    is_interior = env.is_track_interior_.ravel()
    place_fields = np.asarray(encoding["place_fields"])
    assert place_fields.shape == (3, n_total)
    assert np.all(np.isfinite(place_fields))
    # Interior bins carry positive rate; non-interior (padding) bins are exactly 0.
    assert np.all(place_fields[:, ~is_interior] == 0.0)
    assert np.all(place_fields[:, is_interior] > 0.0)

    no_spike = np.asarray(encoding["no_spike_part_log_likelihood"])
    assert no_spike.shape == (n_total,)
    np.testing.assert_allclose(no_spike, place_fields.sum(axis=0), rtol=1e-6)

    # node_order / bin_sizes are carried per the shared contract.
    n_interior = int(is_interior.sum())
    np.testing.assert_array_equal(encoding["node_order"], np.where(is_interior)[0])
    assert np.asarray(encoding["bin_sizes"]).shape == (n_interior,)


def test_fit_accepts_shared_sorted_spikes_params():
    """The fit signature accepts the shared sorted-spikes params so the base class's
    signature filter does not silently drop user/default values (e.g. block_size)."""
    params = set(
        inspect.signature(fit_sorted_spikes_diffusion_encoding_model).parameters
    )
    assert {
        "weights",
        "sampling_frequency",
        "position_std",
        "rank",
        "block_size",
    } <= params


def test_predict_signature_matches_encoding_dict():
    """Every encoding-dict key is a predict parameter (the base-class splat contract)."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=6.0,
    )
    params = set(
        inspect.signature(predict_sorted_spikes_diffusion_log_likelihood).parameters
    )
    assert set(encoding) <= params
    assert {"time", "position_time", "position", "spike_times", "is_local"} <= params


def test_predict_shapes_local_and_nonlocal():
    """Non-local -> (n_time, n_interior); local -> (n_time, 1); both finite."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=6.0,
    )
    decode_time = time[:200]
    n_interior = int(env.is_track_interior_.ravel().sum())

    nonlocal_ll = predict_sorted_spikes_diffusion_log_likelihood(
        decode_time, time, position, spike_times, is_local=False, **encoding
    )
    assert nonlocal_ll.shape == (decode_time.shape[0], n_interior)
    assert np.all(np.isfinite(nonlocal_ll))

    local_ll = predict_sorted_spikes_diffusion_log_likelihood(
        decode_time, time, position, spike_times, is_local=True, **encoding
    )
    assert local_ll.shape == (decode_time.shape[0], 1)
    assert np.all(np.isfinite(local_ll))


@pytest.mark.property
def test_invariants_place_fields_and_likelihood():
    """Place fields >= 0 & finite; likelihoods finite; softmax posterior sums to 1."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=6.0,
    )
    place_fields = np.asarray(encoding["place_fields"])
    assert np.all(place_fields >= 0.0)
    assert np.all(np.isfinite(place_fields))

    ll = np.asarray(
        predict_sorted_spikes_diffusion_log_likelihood(
            time[:100], time, position, spike_times, is_local=False, **encoding
        )
    )
    assert np.all(np.isfinite(ll))
    posterior = np.exp(ll - ll.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-10)


def test_kde_dropin_equivalence_on_wall_less_field():
    """On a wall-less open field with matched bandwidth, diffusion place fields track
    KDE place fields at interior bins away from the boundary (pins units/scale)."""
    rng = np.random.default_rng(7)
    # Finer grid keeps pixellation error small relative to the bandwidth.
    env = Environment(
        environment_name="of_fine",
        place_bin_size=2.5,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(rng.uniform(1.0, 49.0, size=(6000, 2)), infer_track_interior=True)
    time, position, spike_times = simulate_place_data(
        env, n_neurons=3, n_time=12000, seed=8
    )
    std = 8.0

    diffusion = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=std,
    )
    kde = fit_sorted_spikes_kde_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=std,
    )
    medians = place_field_core_median_rel_errors(
        env,
        std,
        np.asarray(diffusion["place_fields"]),
        np.asarray(kde["place_fields"]),
    )
    assert max(medians) < 0.05


def test_nonuniform_weights_parity_with_kde():
    """Non-uniform (posterior-like) weights: diffusion mean_rates match KDE's, and the
    weighted place fields track KDE's weighted result (spike fields are weighted by
    the interpolated weights, not unweighted counts)."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3, n_time=6000)
    rng = np.random.default_rng(5)
    weights = rng.uniform(0.1, 1.0, size=time.shape[0])
    std = 8.0

    diffusion = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        position_std=std,
    )
    kde = fit_sorted_spikes_kde_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        position_std=std,
    )
    # mean_rates are smoother-independent (weighted spike sum / weight sum).
    np.testing.assert_allclose(
        np.asarray(diffusion["mean_rates"]), np.asarray(kde["mean_rates"]), rtol=1e-6
    )
    # The weighted place fields must also track KDE's weighted result: this is the
    # load-bearing check that spike fields are weighted by weights_at_spike_times,
    # not unweighted counts (mean_rates alone can't catch a spatial-weighting bug,
    # since it depends only on the weighted spike sum, not the pixellation).
    medians = place_field_core_median_rel_errors(
        env,
        std,
        np.asarray(diffusion["place_fields"]),
        np.asarray(kde["place_fields"]),
    )
    assert max(medians) < 0.05


def make_linear_track_env():
    """Two-edge linearized track sharing a junction (a continuous line 0..10.5)."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(5.0, 0.0))
    graph.add_node(2, pos=(10.5, 0.0))
    graph.add_edge(0, 1, distance=5.0)
    graph.add_edge(1, 2, distance=5.5)
    for eid, edge in enumerate(graph.edges):
        graph.edges[edge]["edge_id"] = eid
    return Environment(
        environment_name="line2",
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
    ).fit_place_grid()


def test_linearized_track_graph_fit_and_predict():
    """Exercise the linearized (track_graph) branch through the likelihood: fields are
    full-grid, place at the right linear location, and predict runs finite."""
    env = make_linear_track_env()
    rng = np.random.default_rng(3)
    n_time, sampling_frequency = 4000, 100
    time = np.arange(n_time) / sampling_frequency
    # 1D back-and-forth trajectory along the track (x in [0.2, 10.3], y = 0).
    x = 0.2 + np.abs((np.cumsum(rng.normal(0.0, 0.3, n_time)) % (2 * 10.1)) - 10.1)
    position = np.column_stack([x, np.zeros_like(x)])

    place_centers = [2.0, 8.0]  # place-cell x-locations on the track
    dt = 1.0 / sampling_frequency
    spike_times = [
        time[rng.random(n_time) < 40.0 * np.exp(-((x - c) ** 2) / (2 * 1.0**2)) * dt]
        for c in place_centers
    ]

    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=2.0,
    )
    is_interior = env.is_track_interior_.ravel()
    n_total = env.place_bin_centers_.shape[0]
    place_fields = np.asarray(encoding["place_fields"])
    assert place_fields.shape == (2, n_total)
    assert np.all(np.isfinite(place_fields))
    assert np.all(place_fields[:, ~is_interior] == 0.0)

    # Each field peaks near its place cell's linear position (linear ≈ x here).
    linear_centers = env.place_bin_centers_.ravel()
    for center, field in zip(place_centers, place_fields, strict=True):
        assert abs(linear_centers[np.argmax(field)] - center) < 2.0

    decode_time = time[:100]
    nonlocal_ll = predict_sorted_spikes_diffusion_log_likelihood(
        decode_time, time, position, spike_times, is_local=False, **encoding
    )
    assert nonlocal_ll.shape == (100, int(is_interior.sum()))
    assert np.all(np.isfinite(nonlocal_ll))
    local_ll = predict_sorted_spikes_diffusion_log_likelihood(
        decode_time, time, position, spike_times, is_local=True, **encoding
    )
    assert local_ll.shape == (100, 1)
    assert np.all(np.isfinite(local_ll))


def test_registered_in_sorted_spikes_algorithms():
    """The algorithm is selectable via sorted_spikes_algorithm='sorted_spikes_diffusion'."""
    assert "sorted_spikes_diffusion" in _SORTED_SPIKES_ALGORITHMS
    fit_fn, predict_fn = _SORTED_SPIKES_ALGORITHMS["sorted_spikes_diffusion"]
    assert fit_fn is fit_sorted_spikes_diffusion_encoding_model
    assert predict_fn is predict_sorted_spikes_diffusion_log_likelihood
    # Existing algorithms are untouched.
    assert {"sorted_spikes_kde", "sorted_spikes_glm"} <= set(_SORTED_SPIKES_ALGORITHMS)


def test_nonlocal_no_spikes_equals_negative_no_spike_part():
    """With no decoding spikes, the non-local LL is exactly -no_spike_part (Poisson)."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=6.0,
    )
    decode_time = time[:20]
    empty_spikes = [np.array([]), np.array([])]
    ll = np.asarray(
        predict_sorted_spikes_diffusion_log_likelihood(
            decode_time, time, position, empty_spikes, is_local=False, **encoding
        )
    )
    is_interior = env.is_track_interior_.ravel()
    expected = -np.asarray(encoding["no_spike_part_log_likelihood"])[is_interior]
    np.testing.assert_allclose(
        ll, np.tile(expected, (decode_time.shape[0], 1)), rtol=1e-5, atol=1e-6
    )


def test_local_no_spikes_equals_negative_local_rate_sum():
    """With no decoding spikes, the local LL is -sum_k place_field_k at the animal's bin."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=6.0,
    )
    decode_time = time[:20]
    empty_spikes = [np.array([]), np.array([]), np.array([])]
    ll_local = np.asarray(
        predict_sorted_spikes_diffusion_log_likelihood(
            decode_time, time, position, empty_spikes, is_local=True, **encoding
        )
    )
    interpolated = get_position_at_time(time, position, decode_time, env)
    bin_inds = env.get_bin_ind(interpolated)
    place_fields = np.asarray(encoding["place_fields"])
    expected = -np.clip(place_fields[:, bin_inds], EPS, None).sum(axis=0)
    np.testing.assert_allclose(ll_local, expected[:, None], rtol=1e-5, atol=1e-6)


@pytest.mark.integration
def test_end_to_end_sorted_spikes_diffusion_decoder():
    """A full SortedSpikesDecoder run with the diffusion algorithm decodes a valid
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
        sorted_spikes_algorithm="sorted_spikes_diffusion",
        sorted_spikes_algorithm_params={"position_std": 6.0},
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

    # Trajectory recovery: the MAP-decoded position tracks the true position.
    bin_position = np.asarray(posterior["position"].values)
    decoded_position = bin_position[np.asarray(posterior.values).argmax(axis=1)]
    corr = np.corrcoef(decoded_position, np.asarray(position).ravel())[0, 1]
    assert corr > 0.9
