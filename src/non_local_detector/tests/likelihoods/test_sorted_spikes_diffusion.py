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
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from non_local_detector.likelihoods.common import EPS, get_position_at_time
from non_local_detector.likelihoods.diffusion import (
    connected_component_labels,
    environment_graph,
    heat_kernel_rank,
)
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
    "local_interpolation",
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
    assert encoding["local_interpolation"] == "linear"


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
        "local_interpolation",
    } <= params


def test_fit_rejects_invalid_local_interpolation():
    """The interpolation mode is validated at fit time before it enters the dict."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=1)

    with pytest.raises(ValidationError):
        fit_sorted_spikes_diffusion_encoding_model(
            position_time=time,
            position=position,
            spike_times=spike_times,
            environment=env,
            local_interpolation="cubic",
        )


def test_fit_truncated_rank_full_grid_and_predict():
    """A truncated-rank fit still yields finite full-grid place fields and a valid
    posterior (exercises pixellate -> truncated diffuse -> to_density through the fit)."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        position_std=6.0,
        rank=20,
    )
    is_interior = env.is_track_interior_.ravel()
    n_total = env.place_bin_centers_.shape[0]
    place_fields = np.asarray(encoding["place_fields"])
    assert place_fields.shape == (3, n_total)
    assert np.all(np.isfinite(place_fields))
    assert np.all(place_fields[:, ~is_interior] == 0.0)
    assert np.all(place_fields[:, is_interior] > 0.0)
    ll = np.asarray(
        predict_sorted_spikes_diffusion_log_likelihood(
            time[:100], time, position, spike_times, is_local=False, **encoding
        )
    )
    assert np.all(np.isfinite(ll))


def test_fit_auto_truncation_is_near_lossless_and_reduces_rank():
    """Default ``rank=None`` auto-truncates by heat-kernel decay: on a fine grid it
    resolves a rank well below n_interior, yet the place fields match the exact
    full-rank fit to ~1e-4 (near-lossless)."""
    rng_env = np.random.default_rng(3)
    env = Environment(
        environment_name="fine",
        place_bin_size=2.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(
        rng_env.uniform(1.0, 49.0, size=(6000, 2)), infer_track_interior=True
    )
    time, position, spike_times = simulate_place_data(env, n_neurons=3, seed=2)
    std = 8.0
    n_interior = int(env.is_track_interior_.sum())

    common = {
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "environment": env,
        "position_std": std,
    }
    auto = fit_sorted_spikes_diffusion_encoding_model(**common)  # rank=None -> auto
    full = fit_sorted_spikes_diffusion_encoding_model(**common, rank=n_interior)

    # The default fit truncated, and used exactly the resolved rank.
    resolved = heat_kernel_rank(env._diffusion_laplacian_, std)
    assert resolved is not None and resolved < n_interior
    assert resolved in env._diffusion_eigenbasis_

    # Compare where the field is substantial: at EPS-floor bins the rate is ~0 and the
    # relative error is meaningless (dividing by ~EPS), so restrict to bins above
    # 10% of each neuron's peak -- the region the smoother actually estimates.
    interior = env.is_track_interior_.ravel()
    auto_pf = np.asarray(auto["place_fields"])
    full_pf = np.asarray(full["place_fields"])
    rels = []
    for neuron in range(auto_pf.shape[0]):
        substantial = interior & (full_pf[neuron] > 0.1 * full_pf[neuron].max())
        rels.append(
            np.abs(auto_pf[neuron, substantial] - full_pf[neuron, substantial])
            / full_pf[neuron, substantial]
        )
    rel = np.concatenate(rels)
    assert np.median(rel) < 1e-5
    assert np.max(rel) < 1e-3


def test_fit_silent_cell_and_zero_neurons():
    """A silent cell (empty spike train) yields a finite EPS-floored field; zero
    neurons yields an empty full-grid place_fields without crashing."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    n_total = env.place_bin_centers_.shape[0]
    is_interior = env.is_track_interior_.ravel()

    with_silent = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=[spike_times[0], np.array([])],  # one firing, one silent
        environment=env,
        position_std=6.0,
    )
    place_fields = np.asarray(with_silent["place_fields"])
    assert place_fields.shape == (2, n_total)
    assert np.all(np.isfinite(place_fields))
    # The silent cell has mean_rate 0, so its interior field floors uniformly to EPS.
    np.testing.assert_allclose(place_fields[1, is_interior], EPS)

    empty = fit_sorted_spikes_diffusion_encoding_model(
        position_time=time,
        position=position,
        spike_times=[],
        environment=env,
        position_std=6.0,
    )
    assert np.asarray(empty["place_fields"]).shape == (0, n_total)
    assert np.asarray(empty["no_spike_part_log_likelihood"]).shape == (n_total,)


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
        local_interpolation="nearest",
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


def test_local_linear_interpolation_evaluates_point_rate_in_1d():
    """Linear local mode evaluates the full-grid rate map at the continuous position."""
    position_for_fit = np.linspace(0.1, 3.9, 100)[:, np.newaxis]
    env = Environment(
        environment_name="line",
        place_bin_size=1.0,
        position_range=((0.0, 4.0),),
    ).fit_place_grid(position_for_fit, infer_track_interior=False)
    is_interior = env.is_track_interior_.ravel()
    node_order = np.where(is_interior)[0]
    centers = env.place_bin_centers_.ravel()
    place_fields = centers[np.newaxis, :]
    no_spike = place_fields.sum(axis=0)
    time = np.array([0.0, 1.0])
    position = np.array([[1.75], [1.75]])

    ll_local = np.asarray(
        predict_sorted_spikes_diffusion_log_likelihood(
            time,
            time,
            position,
            [np.array([])],
            environment=env,
            occupancy=np.ones(node_order.shape[0]),
            mean_rates=[0.0],
            place_fields=place_fields,
            no_spike_part_log_likelihood=no_spike,
            is_track_interior=is_interior,
            node_order=node_order,
            bin_sizes=np.ones(node_order.shape[0]),
            local_interpolation="linear",
            disable_progress_bar=True,
            is_local=True,
        )
    )

    np.testing.assert_allclose(ll_local, -1.75, rtol=1e-6, atol=1e-6)


def test_local_linear_interpolation_evaluates_point_rate_in_2d():
    """Linear local mode bilinearly interpolates the full-grid rate map in 2D.

    Pins the N-D value path (grid reshape + per-dim axis ordering), which the 1D
    value test and the 2D fallback test do not cover: a rate map that is a pure
    x-ramp interpolates exactly to the position's x (bilinear is exact for a linear
    field), a value strictly between bin centers -- so it also proves interpolation
    was used, not the nearest-bin fallback.
    """
    rng = np.random.default_rng(0)
    env = Environment(
        environment_name="of_linear",
        place_bin_size=1.0,
        position_range=((0.0, 10.0), (0.0, 10.0)),
    ).fit_place_grid(rng.uniform(1.0, 9.0, size=(6000, 2)), infer_track_interior=True)
    is_interior = env.is_track_interior_.ravel()
    node_order = np.where(is_interior)[0]
    # Rate map = x-coordinate of each bin center (globally linear in x).
    place_fields = env.place_bin_centers_[:, 0][np.newaxis, :].astype(float)
    no_spike = place_fields.sum(axis=0)
    time = np.array([0.0, 1.0])
    # A central interior point strictly between bin centers (nearest center x is 4.5).
    x0 = 4.3
    position = np.array([[x0, 5.1], [x0, 5.1]])

    ll_local = np.asarray(
        predict_sorted_spikes_diffusion_log_likelihood(
            time,
            time,
            position,
            [np.array([])],
            environment=env,
            occupancy=np.ones(node_order.shape[0]),
            mean_rates=[0.0],
            place_fields=place_fields,
            no_spike_part_log_likelihood=no_spike,
            is_track_interior=is_interior,
            node_order=node_order,
            bin_sizes=np.ones(node_order.shape[0]),
            local_interpolation="linear",
            disable_progress_bar=True,
            is_local=True,
        )
    )

    # No spikes -> local LL = -local_rate; the x-ramp interpolates exactly to x0.
    np.testing.assert_allclose(ll_local, -x0, rtol=1e-6, atol=1e-6)


def test_local_linear_interpolation_falls_back_to_nearest_when_unsafe():
    """Unsafe interpolation rows fall back to the nearest-bin local likelihood."""
    env = make_2d_env()
    is_interior = env.is_track_interior_.ravel()
    node_order = np.where(is_interior)[0]
    place_fields = np.arange(env.place_bin_centers_.shape[0], dtype=float)[
        np.newaxis, :
    ]
    no_spike = place_fields.sum(axis=0)
    time = np.array([0.0, 1.0])
    position = np.array([[-10.0, -10.0], [-10.0, -10.0]])
    common_kwargs = {
        "time": time,
        "position_time": time,
        "position": position,
        "spike_times": [np.array([])],
        "environment": env,
        "occupancy": np.ones(node_order.shape[0]),
        "mean_rates": [0.0],
        "place_fields": place_fields,
        "no_spike_part_log_likelihood": no_spike,
        "is_track_interior": is_interior,
        "node_order": node_order,
        "bin_sizes": np.ones(node_order.shape[0]),
        "disable_progress_bar": True,
        "is_local": True,
    }

    linear_ll = predict_sorted_spikes_diffusion_log_likelihood(
        **common_kwargs, local_interpolation="linear"
    )
    nearest_ll = predict_sorted_spikes_diffusion_log_likelihood(
        **common_kwargs, local_interpolation="nearest"
    )

    np.testing.assert_allclose(linear_ll, nearest_ll, rtol=1e-6, atol=1e-6)


def make_two_room_env(seed: int = 0):
    """Two rooms separated by an impassable gap -> a disconnected manifold graph.

    Left room x in [2, 16], right room x in [24, 38], gap x in (16, 24): four bins
    wide, wider than ``get_track_interior``'s ``binary_closing`` can bridge, so
    ``environment_graph`` resolves two connected components (a genuine barrier).
    """
    rng = np.random.default_rng(seed)
    left = rng.uniform([2.0, 2.0], [16.0, 38.0], size=(12000, 2))
    right = rng.uniform([24.0, 2.0], [38.0, 38.0], size=(12000, 2))
    position = np.vstack([left, right])
    rng.shuffle(position)
    env = Environment(
        environment_name="two_rooms",
        place_bin_size=2.0,
        position_range=((0.0, 40.0), (0.0, 40.0)),
    ).fit_place_grid(position, infer_track_interior=True)
    return env, position


def test_no_rate_leak_across_impassable_barrier():
    """A cell that fires only in the left room has ~zero estimated rate in the right
    room: the diffusion smoother cannot cross the disconnected graph, whereas the
    Euclidean KDE leaks the field across the physical gap.

    This is the estimator-level counterpart to the engine's component-isolation tests
    (test_diffusion.py) -- it proves the full fit -> pixellate -> diffuse -> to_density
    path respects an impassable barrier, not just the raw eigenbasis.
    """
    env, position = make_two_room_env()
    graph, _, _ = environment_graph(env)
    assert nx.number_connected_components(graph) == 2  # a genuine barrier

    sampling_frequency = 100
    time = np.arange(position.shape[0]) / sampling_frequency
    # A left-room place cell: Gaussian tuning hard-masked to zero in the right room, so
    # there are exactly zero right-room spikes (seed-robust: no stray tail spikes).
    center = np.array([10.0, 20.0])
    rng = np.random.default_rng(2)
    rate = 50.0 * np.exp(-((position - center) ** 2).sum(axis=1) / (2 * 4.0**2))
    rate[position[:, 0] > 16.0] = 0.0
    spike_times = [time[rng.random(position.shape[0]) < rate / sampling_frequency]]

    common = {
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "environment": env,
        "position_std": 6.0,
    }
    diffusion_pf = np.asarray(
        fit_sorted_spikes_diffusion_encoding_model(**common)["place_fields"]
    )[0]
    kde_pf = np.asarray(fit_sorted_spikes_kde_encoding_model(**common)["place_fields"])[
        0
    ]

    interior = env.is_track_interior_.ravel()
    right_room = interior & (env.place_bin_centers_[:, 0] > 20.0)
    diffusion_leak = diffusion_pf[right_room].max() / diffusion_pf.max()
    kde_leak = kde_pf[right_room].max() / kde_pf.max()

    # Diffusion assigns ~no rate across the barrier (right room has no spikes and the
    # component is isolated, so it floors to EPS); KDE leaks a substantial fraction.
    assert diffusion_leak < 0.01
    assert kde_leak > 0.05
    assert kde_leak > 10 * diffusion_leak


def test_truncated_rank_conserves_per_component_occupancy_mass():
    """A truncated-rank fit on a disconnected environment must not shift occupancy
    mass between components: heat conserves each component's mass at any rank, so the
    per-component occupancy density mass is rank-independent. This exercises the
    estimator's wiring of per-component labels into ``diffuse`` -- with a single
    global renormalization, clipped truncation lobes would bleed mass across rooms.
    """
    rng = np.random.default_rng(0)
    # Two rooms with NON-uniform (clustered) occupancy so the occupancy field has
    # structure -> truncation lobes (a uniform field is the null mode and never lobes).
    left = rng.normal([9.0, 20.0], [2.5, 6.0], size=(9000, 2)).clip([2, 2], [16, 38])
    right = rng.normal([31.0, 20.0], [2.5, 6.0], size=(9000, 2)).clip([24, 2], [38, 38])
    position = np.vstack([left, right])
    rng.shuffle(position)
    env = Environment(
        environment_name="two_rooms_nonuniform",
        place_bin_size=2.0,
        position_range=((0.0, 40.0), (0.0, 40.0)),
    ).fit_place_grid(position, infer_track_interior=True)
    graph, _, _ = environment_graph(env)
    assert nx.number_connected_components(graph) == 2
    labels = connected_component_labels(graph)  # interior / node_order order
    time = np.arange(position.shape[0]) / 100.0

    def per_component_occupancy_mass(rank):
        encoding = fit_sorted_spikes_diffusion_encoding_model(
            position_time=time,
            position=position,
            spike_times=[np.array([])],  # occupancy is spike-independent
            environment=env,
            position_std=6.0,
            rank=rank,
        )
        integrand = np.asarray(encoding["bin_sizes"]) * np.asarray(
            encoding["occupancy"]
        )
        return np.array([integrand[labels == c].sum() for c in (0, 1)])

    full = per_component_occupancy_mass(None)
    truncated = per_component_occupancy_mass(8)
    np.testing.assert_allclose(truncated, full, atol=1e-9)


def test_less_edge_bias_than_kde():
    """The reflecting (Neumann) graph boundary preserves a flat density to the edge,
    whereas the Euclidean KDE loses kernel mass past the wall.

    On full-grid uniform coverage the true occupancy density is flat everywhere, so any
    dip at the boundary is pure estimator edge bias. Diffusion's boundary ring stays
    near the flat truth; KDE's dips far below it. (The place-*field* ratio cancels this
    bias -- spike and occupancy densities drop together -- so the effect is measured on
    the occupancy density directly, where it does not cancel.)
    """
    rng = np.random.default_rng(11)
    # Full-grid uniform coverage: every interior bin (incl. the boundary ring) is fully
    # and uniformly sampled, so the true occupancy density is genuinely flat.
    position = rng.uniform([0.0, 0.0], [50.0, 50.0], size=(60000, 2))
    env = Environment(
        environment_name="of_full",
        place_bin_size=2.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(position, infer_track_interior=True)
    sampling_frequency = 200
    time = np.arange(position.shape[0]) / sampling_frequency
    spike_times = [time[:50]]  # occupancy is spike-independent; a token cell
    std = 6.0

    common = {
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "environment": env,
        "position_std": std,
    }
    diffusion_occ = np.asarray(
        fit_sorted_spikes_diffusion_encoding_model(**common)["occupancy"]
    )
    kde_occ = np.asarray(fit_sorted_spikes_kde_encoding_model(**common)["occupancy"])

    interior = env.is_track_interior_.ravel()
    centers = env.place_bin_centers_[interior]  # occupancy is interior-order
    x, y = centers[:, 0], centers[:, 1]
    xs = np.unique(x)
    lo, hi = xs[0], xs[-1]
    boundary = (
        (np.abs(x - lo) < 1e-6)
        | (np.abs(x - hi) < 1e-6)
        | (np.abs(y - lo) < 1e-6)
        | (np.abs(y - hi) < 1e-6)
    )

    def boundary_bias(occ):
        normalized = occ / occ[~boundary].mean()  # deep interior -> ~1
        return abs(normalized[boundary].mean() - 1.0)

    diffusion_bias = boundary_bias(diffusion_occ)
    kde_bias = boundary_bias(kde_occ)

    # Diffusion's reflecting boundary keeps the boundary ring near the flat truth; KDE's
    # boundary ring dips far below it (~40% mass loss at the wall).
    assert diffusion_bias < 0.1
    assert kde_bias > 0.2
    assert diffusion_bias < 0.3 * kde_bias


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
