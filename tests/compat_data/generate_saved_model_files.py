"""Generate the saved-model files of format version 1 under ``tests/compat_data``.

Every later release must load the ``.nld`` files of each format version and
decode them as the release that wrote them did, so this writes one small file
per detector family, ``format_v1/nonlocal_sorted_1d.nld`` (a 1-D track graph)
and ``format_v1/nonlocal_clusterless_2d.nld`` (a 2-D open field), fitted on
``inputs.pkl`` as the stored Spyglass artifacts are (see
``generate_compat_artifacts.py``), and their decodes of the whole recording in
``format_v1/predict_reference.npz``. The sorted-spikes model is fitted with
``estimate_parameters``, so its file holds EM's records too.

Run it once per format version, with the release that freezes that version
installed::

    uv run python tests/compat_data/generate_saved_model_files.py tests/compat_data

The files in ``format_v1`` were written by non_local_detector 0.6.10.dev
(main before 0.7.0, which freezes format version 1) on 2026-11-05 (macOS,
arm64); ``format_v1/VERSION.json`` lists the main package versions.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import xarray as xr
from track_linearization import make_track_graph

import non_local_detector as nld
from non_local_detector.environment import Environment
from non_local_detector.model_format import FORMAT_VERSION

SAMPLING_FREQUENCY = 50.0
NODE_POSITIONS = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0)]
EDGES = [(0, 1), (1, 2)]


def detectors() -> dict:
    """The unfitted detectors, as ``generate_compat_artifacts.main`` builds them."""
    track = Environment(
        environment_name="",
        place_bin_size=10.0,
        track_graph=make_track_graph(NODE_POSITIONS, EDGES),
        edge_order=EDGES,
        edge_spacing=0.0,
    )
    field = Environment(environment_name="", place_bin_size=15.0)
    return {
        "nonlocal_sorted_1d": nld.NonLocalSortedSpikesDetector(
            environments=[track], sampling_frequency=SAMPLING_FREQUENCY
        ),
        "nonlocal_clusterless_2d": nld.NonLocalClusterlessDetector(
            environments=[field], sampling_frequency=SAMPLING_FREQUENCY
        ),
    }


def recording(inputs: dict, clusterless: bool) -> tuple[np.ndarray, dict]:
    """Position and spikes of a family, from ``inputs.pkl``."""
    if clusterless:
        return inputs["field_position"], {
            "spike_times": inputs["clusterless_spike_times"],
            "spike_waveform_features": inputs["clusterless_features"],
        }
    return inputs["track_position"], {"spike_times": inputs["sorted_spike_times"]}


def decode(detector, inputs: dict, position: np.ndarray, spikes: dict) -> np.ndarray:
    """Acausal posterior of the whole recording, shape (n_time_bins, n_state_bins)."""
    time = inputs["time"]
    edges = detector.calculate_time_edges(np.array([time[0], time[-1]]), trim=True)
    keep = [(t >= edges[0]) & (t < edges[-1]) for t in spikes["spike_times"]]
    in_range = {
        key: [v[k] for v, k in zip(value, keep, strict=True)]
        for key, value in spikes.items()
    }
    results = detector.predict(
        **in_range, time_edges=edges, position_time=time, position=position
    )
    return np.asarray(results.acausal_posterior.values)


def main(compat_data: Path) -> None:
    out_dir = compat_data / f"format_v{FORMAT_VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = pickle.loads((compat_data / "inputs.pkl").read_bytes())
    time = inputs["time"]
    reference = {}
    for name, detector in detectors().items():
        position, spikes = recording(inputs, "clusterless" in name)
        if "clusterless" in name:
            detector.fit(time, position, **spikes)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "EM did not converge")
                detector.estimate_parameters(
                    position_time=time,
                    position=position,
                    **spikes,
                    time_edges=detector.calculate_time_edges(
                        np.array([time[0], time[-1]]), trim=True
                    ),
                    max_iter=1,
                )
        detector.save_model(out_dir / f"{name}.nld")
        reference[name] = decode(detector, inputs, position, spikes)
    np.savez(out_dir / "predict_reference.npz", **reference)
    versions = {
        "non_local_detector": nld.__version__,
        "format_version": FORMAT_VERSION,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xarray": xr.__version__,
        "jax": jax.__version__,
    }
    (out_dir / "VERSION.json").write_text(json.dumps(versions, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
