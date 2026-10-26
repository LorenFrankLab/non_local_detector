"""Simulation and tracing helpers shared by the sorted-spikes likelihood tests.

The diffusion and MRF test modules build the same small environments and spike
trains, and several modules check the precision of traced matmuls. Keeping these
helpers here lets the test modules share them without importing one another.
"""

import numpy as np

from non_local_detector.environment import Environment


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
