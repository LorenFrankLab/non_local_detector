"""Benchmark the sorted-spikes graph-diffusion engine.

Records the one-time eigendecomposition cost (dense vs truncated), the per-fit
batched-diffuse cost, and the full encoding-fit cost across a range of grid sizes,
to inform the dense-vs-truncated crossover and a default ``rank``.

Run with::

    uv run python scripts/benchmark_sorted_spikes_diffusion.py

The eigenbasis is cached on the ``Environment`` and reused across neurons and EM
refits, so the dense ``eigh`` cost is amortized; a truncated ``eigsh`` is only
worthwhile when a single dense decomposition dominates the fit at large grids.
"""

import time

import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    cached_eigenbasis,
    diffuse,
    diffusion_eigenbasis,
    environment_graph,
    to_density,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    fit_sorted_spikes_diffusion_encoding_model,
)


def make_env(bin_size: float) -> Environment:
    rng = np.random.default_rng(0)
    position = rng.uniform(1.0, 99.0, size=(8000, 2))
    return Environment(
        environment_name="bench",
        place_bin_size=bin_size,
        position_range=((0.0, 100.0), (0.0, 100.0)),
    ).fit_place_grid(position, infer_track_interior=True)


def simulate(env, n_neurons, seed=1, n_time=8000, sampling_frequency=100):
    rng = np.random.default_rng(seed)
    interior = env.place_bin_centers_[env.is_track_interior_.ravel()]
    lo, hi = interior.min(0), interior.max(0)
    span = hi - lo
    time_ = np.arange(n_time) / sampling_frequency
    walk = np.cumsum(rng.normal(0.0, 3.0, size=(n_time, 2)), axis=0)
    position = lo + np.abs((walk % (2 * span)) - span)
    centers = rng.uniform(lo, hi, size=(n_neurons, 2))
    dt = 1.0 / sampling_frequency
    spike_times = []
    for c in centers:
        rate = 40.0 * np.exp(-((position - c) ** 2).sum(1) / (2 * 8.0**2))
        spike_times.append(time_[rng.random(n_time) < rate * dt])
    return time_, position, spike_times


def timeit(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    n_neurons = 50
    print(
        f"{'n_int':>7} {'dense_eig':>10} {'trunc64':>9} {'diffuse':>9} {'fit_full':>9}"
    )
    for bin_size in (5.0, 2.5, 2.0, 1.5, 1.0):
        env = make_env(bin_size)
        graph, node_order, bin_sizes = environment_graph(env)
        n_interior = node_order.shape[0]
        laplacian = build_laplacian(graph)

        # Default-arg binding pins the current-iteration values into each timed
        # closure (the loop variable would otherwise late-bind).
        dense = timeit(lambda L=laplacian: diffusion_eigenbasis(L, None))
        rank = min(64, n_interior - 1)
        trunc = timeit(lambda L=laplacian, r=rank: diffusion_eigenbasis(L, r))

        eigvals, eigvecs = cached_eigenbasis(env, None)
        fields = np.random.default_rng(0).uniform(
            0, 1, size=(n_interior, n_neurons + 1)
        )
        diff_t = timeit(
            lambda ev=eigvals, evec=eigvecs, f=fields, bs=bin_sizes: to_density(
                diffuse(ev, evec, 6.0, f), bs
            )
        )

        time_, position, spike_times = simulate(env, n_neurons)
        # Warm the cache so fit_full measures diffuse + pixellate, not the first eig.
        cached_eigenbasis(env, None)
        fit_t = timeit(
            lambda t=time_, p=position, st=spike_times, e=env: (
                fit_sorted_spikes_diffusion_encoding_model(
                    position_time=t,
                    position=p,
                    spike_times=st,
                    environment=e,
                    position_std=8.0,
                    disable_progress_bar=True,
                )
            ),
            repeat=2,
        )
        print(
            f"{n_interior:>7} {dense:>10.4f} {trunc:>9.4f} {diff_t:>9.4f} {fit_t:>9.4f}"
        )


if __name__ == "__main__":
    main()
