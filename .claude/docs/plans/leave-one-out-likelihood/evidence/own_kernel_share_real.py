"""In-sample versus held-out clusterless KDE intensity on real recordings.

For a random sample of spikes on each electrode, evaluates the clusterless KDE
marked intensity at the animal's position and the spike's own mark (the Local
term) four ways, every one from the same encoding spikes:

- ``in``: every encoding spike, the spike itself included (decoding data that
  were also used for encoding, as the package does today);
- ``own``: the spike's own kernel left out;
- ``win``: every encoding spike within ``WINDOW`` seconds of the spike left
  out, with the occupancy from the same window removed;
- ``fold``: every encoding spike and all occupancy in the spike's time block
  left out (one of ``N_FOLDS`` contiguous blocks), i.e. blocked cross-validation.

Intensities are in Hz per (feature unit)^D:

    lambda(x, m) = sum_i K_x(x - x_i) N(m - m_i; waveform_std) / occupancy(x),
    occupancy(x) = sum_t K_x(x - x(t)) dt,

with the same unnormalized position kernel in both sums. Each is combined with
the package's background, ``logaddexp(log lambda, log(RATE_EPS_HZ) + log q(m))``,
``q`` a product of Student-t densities (3 degrees of freedom) with scale
``sqrt(training variance + waveform_std**2)``.

Encoding uses the whole epoch, like the held-out harnesses in
``/cumulus/edeno/lowocc``. Read-only; NumPy on the CPU.

Usage (on breeze):

    OMP_NUM_THREADS=8 nice -n 10 <python> own_kernel_share_real.py rs58|j16
"""

import sys

import numpy as np
from scipy.special import gammaln, logsumexp

POSITION_STD = np.sqrt(12.5)  # cm, package default
WAVEFORM_STDS = (12.0, 16.0, 24.0)  # 24 is the package default
RATE_EPS_HZ = 1e-15 / 0.002
WINDOW = 2.0  # seconds
N_FOLDS = 5
N_EVALUATED = 1500  # spikes per electrode
CHUNK = 200  # evaluated spikes per block
RUNNING_SPEED = 4.0  # cm/s


def load_rs58():
    from track_linearization import get_linearized_position, make_track_graph

    cache = np.load("/cumulus/edeno/lowocc/rs58/rs58_ep6.npz", allow_pickle=True)
    graph = make_track_graph(cache["node_positions"], cache["edges"])
    edge_order = [tuple(int(node) for node in edge) for edge in cache["linear_edge_order"]]
    linear = get_linearized_position(
        cache["xy"],
        graph,
        edge_order=edge_order,
        edge_spacing=float(cache["linear_edge_spacing"]),
    )
    electrodes = [
        (cache[f"times_{k}"], np.asarray(cache[f"marks_{k}"], dtype=np.float64))
        for k in range(len(cache["sort_groups"]))
    ]
    return (
        cache["time"],
        linear.linear_position.to_numpy(),
        cache["speed"],
        electrodes,
    )


def load_j16():
    from state_space_playground.session import load_session
    from track_linearization import get_linearized_position

    data = load_session(
        nwb_file_name="j1620210710_.nwb", epoch_name="02_r1", use_sorted_hpc=False
    )
    position_info = data["position_info"]
    time = position_info.index.to_numpy()
    xy = position_info[["head_position_x", "head_position_y"]].to_numpy()
    linear = get_linearized_position(
        xy,
        data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )
    speed_columns = [c for c in position_info.columns if "speed" in c]
    if speed_columns:
        speed = position_info[speed_columns[0]].to_numpy()
    else:
        speed = np.linalg.norm(np.gradient(xy, time, axis=0), axis=1)
    electrodes = []
    for spike_times, marks in zip(
        data["spike_times"]["HPC"], data["spike_waveform_features"]["HPC"], strict=True
    ):
        spike_times = np.asarray(spike_times)
        in_epoch = (spike_times >= time[0]) & (spike_times <= time[-1])
        electrodes.append(
            (spike_times[in_epoch], np.asarray(marks, dtype=np.float64)[in_epoch])
        )
    return time, linear.linear_position.to_numpy(), speed, electrodes


def student_t_log_density(marks, location, scale, dof=3.0):
    """Log density of independent Student-t features, shape (n_spikes,)."""
    z = (marks - location) / scale
    constant = gammaln(0.5 * (dof + 1)) - gammaln(0.5 * dof) - 0.5 * np.log(dof * np.pi)
    return (
        marks.shape[1] * constant
        - np.log(scale).sum()
        - 0.5 * (dof + 1) * np.log1p(z * z / dof).sum(axis=1)
    )


def evaluate_electrode(spike_times, marks, time, position, fold_edges, rng):
    """Per-spike log intensities for one electrode.

    Returns
    -------
    results : dict
        ``results[waveform_std][name]`` is an array over evaluated spikes for
        ``name`` in in, own, win, fold (log intensity with background) and
        ``share`` (own kernel's share of the kernel sum).
    evaluated : np.ndarray
        Indices of the evaluated spikes.
    """
    n_spikes, n_features = marks.shape
    dt = np.gradient(time)
    spike_position = np.interp(spike_times, time, position)
    spike_fold = np.clip(np.searchsorted(fold_edges, spike_times, side="right") - 1, 0, N_FOLDS - 1)
    sample_fold = np.clip(np.searchsorted(fold_edges, time, side="right") - 1, 0, N_FOLDS - 1)
    evaluated = np.sort(rng.choice(n_spikes, size=min(N_EVALUATED, n_spikes), replace=False))
    squared_norms = (marks**2).sum(axis=1)
    location = marks.mean(axis=0)
    variance = marks.var(axis=0)

    results = {
        waveform_std: {name: [] for name in ("in", "own", "win", "fold", "share")}
        for waveform_std in WAVEFORM_STDS
    }
    for start in range(0, evaluated.size, CHUNK):
        rows = evaluated[start : start + CHUNK]
        # (n_rows, n_spikes) squared mark distance and position kernel exponent
        mark_distance = np.maximum(
            squared_norms[rows, None] + squared_norms[None, :] - 2.0 * marks[rows] @ marks.T,
            0.0,
        )
        mark_distance[np.arange(rows.size), rows] = 0.0
        position_exponent = -0.5 * ((spike_position[rows, None] - spike_position[None, :]) / POSITION_STD) ** 2
        is_own = np.zeros((rows.size, n_spikes), dtype=bool)
        is_own[np.arange(rows.size), rows] = True
        in_window = np.abs(spike_times[rows, None] - spike_times[None, :]) <= WINDOW
        in_fold = spike_fold[rows, None] == spike_fold[None, :]

        # (n_rows, n_samples) occupancy kernel, in seconds
        occupancy_kernel = (
            np.exp(-0.5 * ((spike_position[rows, None] - position[None, :]) / POSITION_STD) ** 2) * dt[None, :]
        )
        occupancy = occupancy_kernel.sum(axis=1)
        occupancy_window = (
            occupancy_kernel * (np.abs(spike_times[rows, None] - time[None, :]) <= WINDOW)
        ).sum(axis=1)
        occupancy_fold = (occupancy_kernel * (spike_fold[rows, None] == sample_fold[None, :])).sum(axis=1)

        for waveform_std in WAVEFORM_STDS:
            log_kernel = position_exponent - 0.5 * mark_distance / waveform_std**2
            log_normalizer = -0.5 * n_features * np.log(2 * np.pi) - n_features * np.log(waveform_std)
            background = np.log(RATE_EPS_HZ) + student_t_log_density(
                marks[rows], location, np.sqrt(variance + waveform_std**2)
            )

            def log_intensity(excluded, occupancy_left):
                log_sum = logsumexp(np.where(excluded, -np.inf, log_kernel), axis=1)
                with np.errstate(divide="ignore"):
                    log_rate = log_sum + log_normalizer - np.log(occupancy_left)
                return np.logaddexp(log_rate, background)

            nothing = np.zeros_like(is_own)
            out = results[waveform_std]
            out["in"].append(log_intensity(nothing, occupancy))
            out["own"].append(log_intensity(is_own, occupancy))
            out["win"].append(log_intensity(in_window, occupancy - occupancy_window))
            out["fold"].append(log_intensity(in_fold, occupancy - occupancy_fold))
            # own kernel is exp(0) = 1
            out["share"].append(np.exp(-logsumexp(log_kernel, axis=1)))
    return (
        {ws: {name: np.concatenate(v) for name, v in out.items()} for ws, out in results.items()},
        evaluated,
    )


def main():
    recording = sys.argv[1]
    time, position, speed, electrodes = {"rs58": load_rs58, "j16": load_j16}[recording]()
    keep = np.isfinite(position)
    time, position, speed = time[keep], position[keep], speed[keep]
    duration = time[-1] - time[0]
    fold_edges = np.linspace(time[0], time[-1], N_FOLDS + 1)
    rng = np.random.default_rng(0)
    is_running_sample = speed > RUNNING_SPEED
    running_seconds = np.gradient(time)[is_running_sample].sum()
    print(
        f"{recording}: {duration:.0f} s, running {running_seconds:.0f} s, "
        f"{len(electrodes)} electrodes, mark dims "
        f"{sorted({m.shape[1] for _, m in electrodes})}, "
        f"spikes/electrode {[t.size for t, _ in electrodes]}"
    )

    pooled = {ws: [] for ws in WAVEFORM_STDS}
    for spike_times, marks in electrodes:
        inside = (spike_times >= time[0]) & (spike_times <= time[-1])
        spike_times, marks = spike_times[inside], marks[inside]
        if spike_times.size < 50:
            continue
        results, evaluated = evaluate_electrode(spike_times, marks, time, position, fold_edges, rng)
        running = np.interp(spike_times[evaluated], time, speed) > RUNNING_SPEED
        n_running_spikes = (np.interp(spike_times, time, speed) > RUNNING_SPEED).sum()
        for ws, out in results.items():
            pooled[ws].append((out, running, spike_times.size, n_running_spikes))

    for ws in WAVEFORM_STDS:
        print(f"\nwaveform_std = {ws:g}, position_std = {POSITION_STD:.2f}")
        for label, select in (("all spikes", None), ("running spikes", True)):
            share, rows = [], {"in-own": 0.0, "own-win": 0.0, "own-fold": 0.0, "in-fold": 0.0}
            per_spike = {name: [] for name in rows}
            for out, running, n_all, n_running in pooled[ws]:
                mask = np.ones_like(running) if select is None else running
                if mask.sum() == 0:
                    continue
                n_spikes = n_all if select is None else n_running
                seconds = duration if select is None else running_seconds
                share.append(out["share"][mask])
                for name in rows:
                    first, second = name.split("-")
                    difference = out[first][mask] - out[second][mask]
                    per_spike[name].append(difference)
                    rows[name] += difference.mean() * n_spikes / seconds
            share = np.concatenate(share)
            print(
                f"  {label}: own-kernel share mean {share.mean():.3f}, median "
                f"{np.median(share):.3f}, p90 {np.quantile(share, 0.9):.3f}, "
                f"p99 {np.quantile(share, 0.99):.3f}, share>0.5: {(share > 0.5).mean():.3f}"
            )
            for name, total in rows.items():
                values = np.concatenate(per_spike[name])
                print(
                    f"    log intensity {name:>8}: {values.mean():+.3f} nats/spike "
                    f"(median {np.median(values):+.3f}, p90 {np.quantile(values, 0.9):+.3f}); "
                    f"{total:+.1f} nats/s over electrodes"
                )


if __name__ == "__main__":
    main()
