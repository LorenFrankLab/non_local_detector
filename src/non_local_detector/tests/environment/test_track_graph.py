import networkx as nx
import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import get_position_at_time
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)


def make_line_graph(length=10.0):
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(float(length), 0.0))
    g.add_edge(0, 1, distance=length)
    # Assign stable edge ids expected by environment utilities
    for eid, e in enumerate(g.edges):
        g.edges[e]["edge_id"] = eid
    edge_order = [(0, 1)]
    edge_spacing = 0.0
    return g, edge_order, edge_spacing


def make_env_graph(place_bin_size=1.0):
    g, edge_order, edge_spacing = make_line_graph(10.0)
    env = Environment(
        environment_name="line-graph",
        place_bin_size=place_bin_size,
        track_graph=g,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )
    env = env.fit_place_grid()
    return env


def test_fit_place_grid_with_track_graph_produces_centers_and_edges():
    env = make_env_graph(place_bin_size=1.0)
    assert env.place_bin_centers_ is not None
    assert env.place_bin_edges_ is not None
    # 1D linearized bins: centers are (n, 1)
    assert env.place_bin_centers_.ndim == 2 and env.place_bin_centers_.shape[1] == 1
    # Centers monotonic in 1D
    centers = env.place_bin_centers_.squeeze()
    assert np.all(np.diff(centers) > 0)


def test_get_position_at_time_linearizes_2d_positions_on_graph():
    env = make_env_graph(place_bin_size=1.0)
    # 2D positions along the x-axis from 0..10
    t = np.linspace(0.0, 10.0, 11)
    pos = np.stack([t, np.zeros_like(t)], axis=1)
    spikes = np.array([0.0, 2.5, 7.5, 10.0])
    lin = get_position_at_time(t, pos, spikes, env)
    # On a straight line with start at x=0, linear position equals x coordinate
    assert lin.shape == (spikes.shape[0], 1)
    assert np.allclose(lin.squeeze(), spikes, rtol=1e-6, atol=1e-6)


def test_sorted_kde_encoding_uses_linearized_positions_when_graph_and_2d():
    env = make_env_graph(place_bin_size=1.0)
    t = np.linspace(0.0, 10.0, 101)
    # Provide 2D positions on the graph so encoding code takes linearization branch
    pos = np.stack([t, np.zeros_like(t)], axis=1)
    spikes = [np.array([2.0, 5.0]), np.array([8.0])]
    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spikes,
        environment=env,
        weights=np.ones_like(t),
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )
    assert enc["occupancy"].ndim == 1
    # Non-local prediction shape sanity
    time_edges = np.linspace(0.0, 10.0, 6)
    ll = predict_sorted_spikes_kde_log_likelihood(
        time=time_edges,
        position_time=t,
        position=pos,
        spike_times=spikes,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=np.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=False,
    )
    assert ll.shape[0] == time_edges.shape[0] and ll.ndim == 2


def test_clusterless_kde_encoding_uses_linearized_positions_when_graph_and_2d():
    env = make_env_graph(place_bin_size=1.0)
    t = np.linspace(0.0, 10.0, 101)
    pos = np.stack([t, np.zeros_like(t)], axis=1)
    enc_spike_times = [np.array([2.0, 5.0])]
    enc_feats = [np.array([[0.0, 0.0], [1.0, -1.0]], dtype=float)]
    encoding = fit_clusterless_kde_encoding_model(
        position_time=t,
        position=pos,
        spike_times=enc_spike_times,
        spike_waveform_features=enc_feats,
        environment=env,
        weights=np.ones_like(t),
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )
    assert encoding["occupancy"].ndim == 1
    # Non-local predictor shape sanity
    time_edges = np.linspace(0.0, 10.0, 6)
    ll = predict_clusterless_kde_log_likelihood(
        time=time_edges,
        position_time=t,
        position=pos,
        spike_times=[np.array([2.1])],
        spike_waveform_features=[np.array([[0.1, 0.0]], dtype=float)],
        occupancy=encoding["occupancy"],
        occupancy_model=encoding["occupancy_model"],
        gpi_models=encoding["gpi_models"],
        encoding_spike_waveform_features=encoding["encoding_spike_waveform_features"],
        encoding_positions=encoding["encoding_positions"],
        environment=env,
        mean_rates=np.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=encoding["position_std"],
        waveform_std=encoding["waveform_std"],
        is_local=False,
        block_size=8,
        disable_progress_bar=True,
    )
    assert ll.shape[0] == time_edges.shape[0] and ll.ndim == 2


# =============================================================================
# get_distances_to_interior_bins
# =============================================================================


def test_get_distances_to_interior_bins_shape_1d_graph():
    """1D track graph path returns (n_positions, n_interior_bins)."""
    env = make_env_graph(place_bin_size=1.0)
    positions = np.array([[2.5], [5.0], [7.5]])

    distances = env.get_distances_to_interior_bins(positions)

    n_interior = int(env.is_track_interior_.sum())
    assert distances.shape == (3, n_interior)
    assert np.all(distances >= 0) or np.any(np.isnan(distances))


def test_get_distances_to_interior_bins_1d_graph_values():
    """1D track graph distances are correct for a known linear track.

    On a straight line with `place_bin_size=1.0` and track length 10,
    interior bins are centered at 0.5, 1.5, ..., 9.5. Distance from
    position 2.5 (≈ bin at 2.5) to bin 5.5 should be ~3.0.
    """
    env = make_env_graph(place_bin_size=1.0)
    positions = np.array([[2.5]])

    distances = env.get_distances_to_interior_bins(positions)

    # Distance from animal at 2.5 to every interior bin center
    interior_centers = env.place_bin_centers_[env.is_track_interior_.ravel()].ravel()
    expected = np.abs(interior_centers - 2.5)
    # Allow small tolerance for graph linearization approximations
    np.testing.assert_allclose(distances[0], expected, atol=0.6)


def test_get_distances_to_interior_bins_caches_matrix():
    """First call on a 1D track graph populates _bin_distance_matrix_."""
    env = make_env_graph(place_bin_size=1.0)
    assert not hasattr(env, "_bin_distance_matrix_")

    env.get_distances_to_interior_bins(np.array([[5.0]]))

    assert hasattr(env, "_bin_distance_matrix_")
    n_total_bins = env.place_bin_centers_.shape[0]
    assert env._bin_distance_matrix_.shape == (n_total_bins, n_total_bins)


def test_fit_place_grid_invalidates_distance_cache():
    """Re-calling fit_place_grid removes the cached distance matrix."""
    env = make_env_graph(place_bin_size=1.0)
    env.get_distances_to_interior_bins(np.array([[5.0]]))
    assert hasattr(env, "_bin_distance_matrix_")

    env.fit_place_grid()

    assert not hasattr(env, "_bin_distance_matrix_"), (
        "fit_place_grid should invalidate the distance matrix cache"
    )


def test_get_distances_to_interior_bins_nd_array_path():
    """1D open-field (no track_graph) uses the N-D array distance path.

    When no ``track_graph`` is provided but position data is given,
    ``fit_place_grid`` builds an internal N-D track graph and computes
    ``distance_between_nodes_`` as a dense numpy array. This is the
    most common path for user-facing decoders without a track graph.
    """
    env = Environment(place_bin_size=1.0)
    position_1d = np.linspace(0, 10, 20)[:, np.newaxis]
    env = env.fit_place_grid(position_1d)

    # Sanity: this is the N-D array path, not 1D dict or Euclidean
    assert env.track_graph is None
    assert isinstance(env.distance_between_nodes_, np.ndarray)

    # Query at a known interior position
    distances = env.get_distances_to_interior_bins(np.array([[5.0]]))
    n_interior = int(env.is_track_interior_.sum())
    assert distances.shape == (1, n_interior)
    assert np.all(distances >= 0)

    # Distance from animal to its own bin should be 0
    assert distances.min() == 0.0


def test_get_distances_to_interior_bins_euclidean_fallback():
    """Defensive Euclidean fallback runs when no distance matrix is available.

    This path is reached if ``distance_between_nodes_`` was not computed
    (e.g. a partially-constructed environment). We simulate this by
    clearing the precomputed matrix after fitting.
    """
    env = Environment(place_bin_size=1.0)
    position_1d = np.linspace(0, 10, 20)[:, np.newaxis]
    env = env.fit_place_grid(position_1d)

    # Force Euclidean fallback by clearing the precomputed matrix
    env.distance_between_nodes_ = None

    query = np.array([[5.0]])
    distances = env.get_distances_to_interior_bins(query)

    n_interior = int(env.is_track_interior_.sum())
    assert distances.shape == (1, n_interior)

    # Verify values match Euclidean distances to each interior bin center
    interior_centers = env.place_bin_centers_[env.is_track_interior_.ravel()]
    expected = np.sqrt(((interior_centers - query[0]) ** 2).sum(axis=-1))
    np.testing.assert_allclose(distances[0], expected, atol=1e-10)


def test_get_distances_to_interior_bins_off_track_returns_nan():
    """Off-track positions (gap bins, node_id == -1) return NaN rows.

    A two-segment track with ``edge_spacing > 0`` creates gap bins
    between the segments in the linearized grid. Animal positions that
    linearize into those gap bins yield NaN distance rows so callers
    can apply a uniform fallback.
    """
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(50.0, 0.0))
    g.add_node(2, pos=(60.0, 0.0))
    g.add_node(3, pos=(110.0, 0.0))
    g.add_edge(0, 1, distance=50.0, edge_id=0)
    g.add_edge(2, 3, distance=50.0, edge_id=1)

    env = Environment(
        environment_name="two-segment",
        place_bin_size=5.0,
        track_graph=g,
        edge_order=[(0, 1), (2, 3)],
        edge_spacing=10.0,
    )
    position_1d = np.concatenate(
        [np.linspace(0.0, 50.0, 25), np.linspace(60.0, 110.0, 25)]
    )
    env = env.fit_place_grid(position_1d, infer_track_interior=True)

    # Find a bin that is off-track (gap) and query its center position
    is_interior = env.is_track_interior_.ravel()
    gap_bins = np.where(~is_interior)[0]
    assert len(gap_bins) > 0, "Environment should have at least one gap bin"
    gap_center = env.place_bin_centers_[gap_bins[0]]

    distances = env.get_distances_to_interior_bins(gap_center[np.newaxis])
    assert np.all(np.isnan(distances[0])), (
        "Off-track animal position should produce NaN distance row"
    )
