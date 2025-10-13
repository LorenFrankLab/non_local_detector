import numpy as np
import networkx as nx

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import get_position_at_time
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
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
        encoding_spike_weights=encoding["encoding_spike_weights"],
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
