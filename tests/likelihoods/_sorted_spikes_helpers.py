"""Simulation and tracing helpers shared by the sorted-spikes likelihood tests.

The diffusion and MRF test modules build the same small environments and spike
trains, and several modules check the precision of traced matmuls. Keeping these
helpers here lets the test modules share them without importing one another.
"""

import networkx as nx
import numpy as np

from non_local_detector.environment import Environment


def path_graph(n_nodes: int, distance: float = 1.0) -> nx.Graph:
    """1D chain 0-1-...-(n-1) with uniform ``distance`` edge attribute."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for node in range(n_nodes - 1):
        graph.add_edge(node, node + 1, distance=distance)
    return graph


def make_2d_env(seed: int = 0) -> Environment:
    """2D open field with an inferred interior (padding bins stay non-interior)."""
    rng = np.random.default_rng(seed)
    position = rng.uniform(1.0, 49.0, size=(4000, 2))
    return Environment(
        environment_name="of",
        place_bin_size=5.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    ).fit_place_grid(position, infer_track_interior=True)


def make_linear_track_env(edge_spacing: float) -> Environment:
    """Two-edge linearized track sharing a junction node (a continuous line 0..10.5)."""
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
        edge_spacing=edge_spacing,
    ).fit_place_grid()


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


def simulate_sparse_uniform_tracking(
    sampling_frequency: float = 30.0,
    rate: float = 10.0,
    duration: float = 600.0,
    seed: int = 0,
):
    """A uniform-rate neuron tracked at 150 units/s on a [0, 100] track.

    The animal bounces between 1.25 and 96.25 at constant speed. At 30 Hz it moves
    5 units per sample, so with 2.5-unit bins every sample sits on an even bin's
    center (1.25 + 5k) and the odd bins between them get no tracking sample.

    Returns
    -------
    time : np.ndarray, shape (n_time,)
    position : np.ndarray, shape (n_time, 1)
    spike_times : list[np.ndarray]
        One neuron firing as a homogeneous Poisson process at ``rate`` Hz over
        the encoding support (sample centers plus endpoint half-cells).
    environment : Environment
        2.5-unit bins on [0, 100], every bin interior.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / sampling_frequency
    time = np.arange(0.0, duration, dt)
    span = 95.0
    phase = (150.0 * time) % (2 * span)
    position = (1.25 + np.where(phase < span, phase, 2 * span - phase))[:, None]
    n_spikes = rng.poisson(rate * (time[-1] - time[0] + dt))
    spike_times = np.sort(rng.uniform(time[0] - dt / 2, time[-1] + dt / 2, n_spikes))
    environment = Environment(
        place_bin_size=2.5, position_range=((0.0, 100.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    return time, position, [spike_times], environment


def poisson_log_likelihood_reference(place_fields, is_interior, spike_times, edges):
    """Independent float64 non-local Poisson log likelihood (up to ``-log n!``).

    ``LL[t, j] = sum_k n_tk log(rate_kj dt_t) - dt_t sum_k rate_kj`` with ``rate``
    the fitted Hz place fields on interior bins. ``edges`` must not coincide with
    any spike time so ``np.histogram``'s closed last bin cannot differ from the
    package's half-open bins.

    Parameters
    ----------
    place_fields : array, shape (n_neurons, n_total_bins)
        Fitted full-grid rates in Hz.
    is_interior : np.ndarray, shape (n_total_bins,)
        Boolean interior mask; output columns follow its interior order.
    spike_times : list[np.ndarray]
        One spike-time array per neuron, in seconds.
    edges : np.ndarray, shape (n_time_bins + 1,)
        Decoding time-bin edges in seconds.

    Returns
    -------
    log_likelihood : np.ndarray, shape (n_time_bins, n_interior_bins)
    """
    rates = np.asarray(place_fields, dtype=np.float64)[:, is_interior]
    durations = np.diff(edges)
    counts = np.stack([np.histogram(spikes, edges)[0] for spikes in spike_times])
    return (
        counts.T @ np.log(rates)
        + counts.sum(axis=0)[:, None] * np.log(durations)[:, None]
        - durations[:, None] * rates.sum(axis=0)
    )


def dot_general_precisions(jaxpr):
    """Precision of every ``dot_general`` in ``jaxpr`` and its nested jaxprs."""
    precisions = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            precisions.append(eqn.params["precision"])
        for value in eqn.params.values():
            for sub in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    precisions.extend(dot_general_precisions(inner))
    return precisions
