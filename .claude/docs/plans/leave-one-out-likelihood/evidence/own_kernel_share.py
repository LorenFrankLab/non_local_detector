"""How much of an encoding spike's in-sample KDE intensity is its own kernel?

For each encoding spike j on an electrode, at the animal's position and the
spike's own mark, the clusterless KDE numerator is

    S_j = sum_i K_x(x_j - x_i) K_m(m_j - m_i).

The spike's own term is K(0) = 1 (unnormalized; the normalizers cancel in the
share). ``share(window)`` is the fraction of S_j that comes from encoding
spikes within ``window`` seconds of spike j; window 0 is the spike itself.
Leaving those spikes out changes the Local log intensity by ``log(1 - share)``
(for window 0 exactly; for a wider window the matching occupancy would have to
be removed too, so those rows describe how the kernel sum is made up, not a
bias).

Simulated data only. ``noise`` adds independent Gaussian noise to every mark
feature, because the simulator's marks are much cleaner than recorded ones.

Run from the repository root:

    JAX_PLATFORMS=cpu uv run python \
        .claude/docs/plans/leave-one-out-likelihood/evidence/own_kernel_share.py
"""

import numpy as np

from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

POSITION_STD = np.sqrt(12.5)  # package default, cm
WAVEFORM_STD = 24.0  # package default, feature units
WINDOWS = (0.0, 0.05, 0.5, 2.0)  # seconds; 0 = the spike itself
# (mark dimensions, laps, added mark noise SD per feature)
CONFIGURATIONS = (
    (4, 10, 0.0),
    (4, 40, 0.0),
    (4, 10, 20.0),
    (8, 10, 20.0),
    (32, 10, 10.0),
    (32, 10, 20.0),
    (32, 40, 20.0),
)
N_EVALUATED_SPIKES = 3000  # per electrode


def kernel_sum_shares(
    times: np.ndarray,
    positions: np.ndarray,
    marks: np.ndarray,
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """Share of each evaluated spike's kernel sum from spikes within each window.

    Parameters
    ----------
    times : np.ndarray, shape (n_spikes,)
    positions : np.ndarray, shape (n_spikes,)
        Position of the animal at each spike time.
    marks : np.ndarray, shape (n_spikes, n_features)
    rng : np.random.Generator

    Returns
    -------
    shares : dict of np.ndarray, shape (n_evaluated,)
        Keyed by window half-width in seconds.
    """
    evaluated = rng.choice(
        times.size, size=min(N_EVALUATED_SPIKES, times.size), replace=False
    )
    position_distance = (positions[evaluated, None] - positions[None, :]) / POSITION_STD
    mark_distance = (marks[evaluated, None, :] - marks[None, :, :]) / WAVEFORM_STD
    kernel = np.exp(-0.5 * (position_distance**2 + (mark_distance**2).sum(-1)))
    total = kernel.sum(axis=1)
    time_distance = np.abs(times[evaluated, None] - times[None, :])
    return {
        window: (kernel * (time_distance <= window)).sum(axis=1) / total
        for window in WINDOWS
    }


def main() -> None:
    rng = np.random.default_rng(0)
    for n_features, n_laps, noise in CONFIGURATIONS:
        sim = make_simulated_run_data(n_features=n_features, n_runs=n_laps, seed=0)
        position = np.asarray(sim.position)
        position = position[:, 0] if position.ndim > 1 else position
        pooled: dict[float, list[np.ndarray]] = {window: [] for window in WINDOWS}
        n_spikes = []
        for spike_times, features in zip(
            sim.spike_times, sim.spike_waveform_features, strict=True
        ):
            spike_times = np.asarray(spike_times)
            marks = np.asarray(features, dtype=float)
            marks = marks + rng.normal(0.0, noise, size=marks.shape)
            n_spikes.append(spike_times.size)
            spike_position = np.interp(spike_times, sim.position_time, position)
            for window, share in kernel_sum_shares(
                spike_times, spike_position, marks, rng
            ).items():
                pooled[window].append(share)

        duration = sim.position_time[-1] - sim.position_time[0]
        print(
            f"\nmark dims={n_features}, added mark noise SD={noise}, laps={n_laps}, "
            f"duration={duration:.0f} s, spikes/electrode={int(np.mean(n_spikes))}, "
            f"electrodes={len(n_spikes)}"
        )
        print(
            f"  {'window (s)':>10} {'mean share':>11} {'median':>8} {'p90':>6} "
            f"{'share>0.5':>10} {'mean -log(1-share), nats/spike':>31}"
        )
        for window, shares in pooled.items():
            share = np.concatenate(shares)
            log_change = -np.log1p(-np.minimum(share, 1 - 1e-12))
            print(
                f"  {window:>10.2f} {share.mean():>11.3f} {np.median(share):>8.3f} "
                f"{np.quantile(share, 0.9):>6.3f} {(share > 0.5).mean():>10.3f} "
                f"{log_change.mean():>31.3f}"
            )


if __name__ == "__main__":
    main()
