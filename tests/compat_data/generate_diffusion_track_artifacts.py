"""Write the diffusion detectors saved before the finite-volume diffusion operator.

Run with the package of main 11d9b3de, the last main whose graph diffusion
coupled every pair of arm end bins at a junction with ``1 / distance**2`` and
ignored bin widths, from the repository's own ``uv sync`` environment at that
commit::

    uv run --no-sync python tests/compat_data/generate_diffusion_track_artifacts.py \\
        tests/compat_data/main-11d9b3de 11d9b3de76fd1e60b86d5e96ed24fc4a3b8c5685

It fits three ``NonLocal*Detector``s on a simulated T-maze whose junction is
joined by a 3 cm segment, in 3 cm bins, so its bins are 2.82 to 3.0 cm wide:
``sorted_spikes_diffusion``, ``sorted_spikes_mrf`` and
``clusterless_diffusion``. Each is written as a plain pickle (``*_pickle.pkl``),
the way ``save_model`` wrote models up to 0.6.9, and as that main's
``save_model`` file (``*_model.pkl``). Sorted-spikes detectors decode from their
stored place fields, so ``diffusion_tmaze_reference.pkl`` holds their own
decode of the last 200 samples, which a later version must reproduce. A
clusterless diffusion detector's stored occupancy and intensities come from
the earlier operator, so decoding it must ask for a refit; it has no
reference. ``diffusion_tmaze_inputs.pkl`` holds the recording.
``VERSION.json`` records the environment and the commit, as
``generate_compat_artifacts.py`` does.
"""

import json
import pickle
import sys
from itertools import pairwise
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import xarray as xr
from track_linearization import make_track_graph

import non_local_detector as nld

SAMPLING_FREQUENCY = 100.0
N_SAMPLES = 4_000
N_DECODE = 200
NODES = [(0.0, 0.0), (0.0, 50.0), (0.0, 53.0), (-31.0, 53.0), (37.0, 53.0)]
EDGES = [(0, 1), (1, 2), (2, 3), (2, 4)]
LAP = [0, 1, 2, 3, 2, 1, 0, 1, 2, 4, 2, 1, 0]
ALGORITHMS = ("sorted_spikes_diffusion", "sorted_spikes_mrf", "clusterless_diffusion")


def environment() -> nld.Environment:
    return nld.Environment(
        environment_name="",
        track_graph=make_track_graph(NODES, EDGES),
        edge_order=EDGES,
        edge_spacing=0.0,
        place_bin_size=3.0,
    )


def simulate() -> dict:
    """Laps of the T-maze at 25 cm/s, seven place cells on two tetrodes."""
    rng = np.random.default_rng(0)
    nodes = np.asarray(NODES)
    segments = []
    for a, b in pairwise(LAP):
        n = max(int(np.linalg.norm(nodes[b] - nodes[a]) / 25.0 * SAMPLING_FREQUENCY), 2)
        fraction = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        segments.append(nodes[a] + fraction * (nodes[b] - nodes[a]))
    lap = np.concatenate(segments)
    position = np.concatenate([lap] * (N_SAMPLES // len(lap) + 1))[:N_SAMPLES]
    time = np.arange(N_SAMPLES) / SAMPLING_FREQUENCY
    centers = np.array(
        [[0, 10], [0, 45], [0, 52], [-6, 53], [-25, 53], [8, 53], [30, 53]], float
    )
    distance = ((position[:, None] - centers[None]) ** 2).sum(-1)
    rates = 15.0 * np.exp(-distance / (2 * 5.0**2)) + 0.5
    spikes = [
        time[rng.random(N_SAMPLES) < rate / SAMPLING_FREQUENCY] for rate in rates.T
    ]
    amplitudes = rng.normal(60.0, 30.0, (len(centers), 4))
    electrode_times, electrode_features = [], []
    for cells in ([0, 1, 2, 3], [3, 4, 5, 6]):
        times = np.concatenate([spikes[k] for k in cells])
        features = np.concatenate(
            [amplitudes[k] + rng.normal(0.0, 3.0, (spikes[k].size, 4)) for k in cells]
        )
        order = np.argsort(times)
        electrode_times.append(times[order])
        electrode_features.append(features[order])
    return {
        "time": time,
        "position": position,
        "spike_times": spikes,
        "electrode_spike_times": electrode_times,
        "electrode_features": electrode_features,
        "n_train": N_SAMPLES - N_DECODE,
    }


def window(times: list, low: float, high: float) -> list:
    return [t[(t >= low) & (t < high)] for t in times]


def window_features(times: list, features: list, low: float, high: float) -> list:
    return [f[(t >= low) & (t < high)] for t, f in zip(times, features, strict=True)]


def fit(algorithm: str, data: dict):
    n_train = data["n_train"]
    time, position = data["time"], data["position"]
    end = time[n_train]
    if algorithm.startswith("clusterless"):
        detector = nld.NonLocalClusterlessDetector(
            environments=environment(),
            clusterless_algorithm=algorithm,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        return detector.fit(
            time[:n_train],
            position[:n_train],
            window(data["electrode_spike_times"], 0.0, end),
            window_features(
                data["electrode_spike_times"], data["electrode_features"], 0.0, end
            ),
        )
    detector = nld.NonLocalSortedSpikesDetector(
        environments=environment(),
        sorted_spikes_algorithm=algorithm,
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    return detector.fit(
        time[:n_train], position[:n_train], window(data["spike_times"], 0.0, end)
    )


def decode(detector, data: dict) -> np.ndarray:
    """Decode the samples after the training ones, one bin per sample."""
    n_train = data["n_train"]
    time = data["time"]
    decode_time = time[n_train:]
    time_edges = nld.time_edges_from_centers(decode_time)
    return detector.predict(
        window(data["spike_times"], decode_time[0], time_edges[-1] + 1e-9),
        time_edges=time_edges,
        position_time=decode_time,
        position=data["position"][n_train:],
    ).acausal_posterior.values


def main(out_dir: Path, commit: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = simulate()
    (out_dir / "diffusion_tmaze_inputs.pkl").write_bytes(
        pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    )
    reference = {}
    for algorithm in ALGORITHMS:
        name = f"nonlocal_{algorithm}_tmaze"
        detector = fit(algorithm, data)
        (out_dir / f"{name}_pickle.pkl").write_bytes(
            pickle.dumps(detector, protocol=pickle.HIGHEST_PROTOCOL)
        )
        detector.save_model(out_dir / f"{name}_model.pkl")
        if not algorithm.startswith("clusterless"):
            reference[name] = decode(detector, data)
    (out_dir / "diffusion_tmaze_reference.pkl").write_bytes(
        pickle.dumps(reference, protocol=pickle.HIGHEST_PROTOCOL)
    )
    version_path = out_dir / "VERSION.json"
    if not version_path.exists():
        version_path.write_text(
            json.dumps(
                {
                    "non_local_detector": nld.__version__,
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "xarray": xr.__version__,
                    "jax": jax.__version__,
                    "datajoint_round_trip": False,
                    "commit": commit,
                },
                indent=1,
            )
            + "\n"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
