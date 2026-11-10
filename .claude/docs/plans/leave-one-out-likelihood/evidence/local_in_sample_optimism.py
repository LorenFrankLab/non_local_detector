"""Is the Local likelihood of a session higher under the model fitted to it?

For every likelihood algorithm, fit a non-local detector on each of several
simulated sessions (same cells, same trajectory, independent spikes). Score
every session's Local state under every fitted model. For one session, the
excess is

    Local log likelihood under the model fitted to it
    - mean Local log likelihood under the models fitted to the other sessions.

A statistically proper Local state gives zero excess on average: a model has
no advantage on the data it was fitted to. A positive excess is in-sample
optimism. Comparing models on the same session cancels that session's spike
count noise, which swamps the comparison of one model across sessions.

Uses the package's own likelihood code through the detector, so it measures
what a user gets today.

Run from the repository root (optionally name algorithms to restrict it):

    JAX_PLATFORMS=cpu uv run python \
        .claude/docs/plans/leave-one-out-likelihood/evidence/local_in_sample_optimism.py
"""

import sys
import time
import warnings

import numpy as np

from non_local_detector import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.simulate.clusterless_simulation import (
    MARK_SPACING,
    make_simulated_run_data,
)

N_LAPS = 10
SAMPLING_FREQUENCY = 1000  # the simulator's bin rate, Hz
SEEDS = (0, 1, 2)
# (mark dimensions, added mark noise SD per feature)
MARK_SETTINGS = ((4, 10.0), (16, 15.0))
CLUSTERLESS = (
    "clusterless_kde",
    "clusterless_diffusion",
    "clusterless_gmm",
    "clusterless_mark_glm",
)
SORTED = (
    "sorted_spikes_kde",
    "sorted_spikes_diffusion",
    "sorted_spikes_glm",
    "sorted_spikes_mrf",
)


def simulate(seed: int, n_features: int, noise: float) -> dict:
    """One session: noisy marks per electrode and the cells' spike trains."""
    sim = make_simulated_run_data(n_features=n_features, n_runs=N_LAPS, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    marks, units = [], []
    for spike_times, features in zip(
        sim.spike_times, sim.spike_waveform_features, strict=True
    ):
        spike_times = np.asarray(spike_times)
        features = np.asarray(features, dtype=np.float64)
        # The simulator puts cell k's marks at k * MARK_SPACING on every feature.
        cell = np.rint(features.mean(axis=1) / MARK_SPACING).astype(int)
        units.extend(spike_times[cell == k] for k in np.unique(cell))
        marks.append(
            (features + rng.normal(0.0, noise, size=features.shape)).astype(np.float32)
        )
    return {
        "position_time": np.asarray(sim.position_time),
        "position": np.asarray(sim.position),
        "time_edges": np.asarray(sim.edges),
        "spike_times": [np.asarray(t) for t in sim.spike_times],
        "marks": marks,
        "units": units,
    }


def local_log_likelihood(detector, session: dict, clusterless: bool) -> float:
    """Summed Local-state log likelihood of a session, in nats."""
    arguments = (
        (session["spike_times"], session["marks"])
        if clusterless
        else (session["units"],)
    )
    log_likelihood = np.asarray(
        detector.compute_log_likelihood(
            session["position_time"],
            session["position"],
            *arguments,
            time_edges=session["time_edges"],
        )
    )
    local = np.flatnonzero(np.asarray(detector.state_ind_) == 0)
    assert local.size == 1, "expected one Local state bin"
    return float(log_likelihood[:, local[0]].sum())


def in_sample_excess(
    algorithm: str, sessions: list[dict], clusterless: bool
) -> np.ndarray:
    """Excess Local log likelihood of each session under its own model, nats.

    Returns
    -------
    excess : np.ndarray, shape (n_sessions,)
    """
    n_sessions = len(sessions)
    # scores[model, data]
    scores = np.empty((n_sessions, n_sessions))
    for model_index, fitted in enumerate(sessions):
        if clusterless:
            detector = NonLocalClusterlessDetector(
                clusterless_algorithm=algorithm, sampling_frequency=SAMPLING_FREQUENCY
            )
            detector.fit(
                fitted["position_time"],
                fitted["position"],
                fitted["spike_times"],
                fitted["marks"],
            )
        else:
            detector = NonLocalSortedSpikesDetector(
                sorted_spikes_algorithm=algorithm, sampling_frequency=SAMPLING_FREQUENCY
            )
            detector.fit(fitted["position_time"], fitted["position"], fitted["units"])
        for data_index, session in enumerate(sessions):
            scores[model_index, data_index] = local_log_likelihood(
                detector, session, clusterless
            )
    own = np.diag(scores)
    others = (scores.sum(axis=0) - own) / (n_sessions - 1)
    return own - others


def report(
    algorithm: str, excess: np.ndarray, duration: float, n_spikes: float, seconds: float
) -> None:
    mean = excess.mean()
    standard_error = excess.std(ddof=1) / np.sqrt(excess.size)
    print(
        f"  {algorithm:<24} {mean / duration:>+9.2f} +/- {standard_error / duration:<6.2f}"
        f" nats/s   {mean / n_spikes:>+8.4f} nats/spike   ({seconds:.0f} s)",
        flush=True,
    )


def main() -> None:
    warnings.simplefilter("ignore")
    selected = set(sys.argv[1:])
    for n_features, noise in MARK_SETTINGS:
        sessions = [simulate(seed, n_features, noise) for seed in SEEDS]
        duration = sessions[0]["time_edges"][-1] - sessions[0]["time_edges"][0]
        n_spikes = float(
            np.mean([sum(t.size for t in s["spike_times"]) for s in sessions])
        )
        print(
            f"\nmark dims={n_features}, mark noise SD={noise}, laps={N_LAPS}, "
            f"{duration:.0f} s, {n_spikes:.0f} spikes, {len(sessions[0]['units'])} cells, "
            f"{len(SEEDS)} sessions; excess = own model minus the other models, "
            f"mean +/- SE over sessions",
            flush=True,
        )
        for algorithm in CLUSTERLESS:
            if selected and algorithm not in selected:
                continue
            start = time.perf_counter()
            excess = in_sample_excess(algorithm, sessions, clusterless=True)
            report(algorithm, excess, duration, n_spikes, time.perf_counter() - start)
        if n_features != MARK_SETTINGS[0][0]:
            continue  # the cells' spike trains do not depend on the marks
        for algorithm in SORTED:
            if selected and algorithm not in selected:
                continue
            start = time.perf_counter()
            excess = in_sample_excess(algorithm, sessions, clusterless=False)
            report(algorithm, excess, duration, n_spikes, time.perf_counter() - start)


if __name__ == "__main__":
    main()
