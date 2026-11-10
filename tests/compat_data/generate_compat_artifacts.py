"""Generate the stored-artifact fixtures under ``tests/compat_data``.

Each fixture directory holds what a Spyglass decode stores with one released
or recent ``non_local_detector``: the fitted model pickle (``save_model``), the
results netCDF (``save_results``) and the ``DecodingParameters`` rows. Run this
script with the interpreter of an isolated environment that has that version
installed, from a directory outside any source tree, so that the old package
is imported::

    uv venv -p 3.10 v060 && VIRTUAL_ENV=v060 uv pip install \
        --exclude-newer 2024-06-10 non_local_detector==0.6.0 \
        datajoint setuptools netCDF4
    v060/bin/python -I generate_compat_artifacts.py OUT_DIR

A second argument, the commit the package was installed from, is recorded
in ``VERSION.json`` as ``commit``, for an install whose version string does
not name it (an editable install keeps the version of its first build).

The fixtures in this directory were generated from 2026-11-03 to 2026-11-04
(macOS, arm64), each with ``datajoint setuptools netCDF4`` added:

==============  ======  ===========================================  ======
directory       Python  install                                      JAX
==============  ======  ===========================================  ======
v0.5.0          3.10    ``non_local_detector==0.5.0 "numba>=0.57"``  0.4.18
                        ``--exclude-newer 2023-08-30``
v0.5.12         3.10    ``non_local_detector==0.5.12 "numba>=0.58"`` 0.4.18
                        ``--exclude-newer 2023-10-10``
v0.5.13         3.10    ``non_local_detector==0.5.13 "numba>=0.58"`` 0.4.18
                        ``--exclude-newer 2023-10-13``
v0.6.0          3.10    ``non_local_detector==0.6.0``                0.4.29
                        ``--exclude-newer 2024-06-10``
v0.6.9          3.12    ``non_local_detector==0.6.9``                0.4.35
                        ``--exclude-newer 2024-11-30``
main-84259d49   3.13    ``non_local_detector @ git+...@84259d49``    0.11.2
                        ``--exclude-newer 2026-09-23 "pandas<3"``
main-66047901   3.13    ``non_local_detector @ git+...@66047901``    0.12.0
                        ``--exclude-newer 2026-11-03 "pandas<3"``
main-d5f85823   3.13    ``non_local_detector @ git+...@d5f85823``    0.12.0
                        ``--exclude-newer 2026-11-03 "pandas<3"``
main-11d9b3de   3.13    the repository's own ``uv sync`` environment 0.9.0
                        at ``11d9b3de`` (pandas 3.0, no datajoint)
==============  ======  ===========================================  ======

Spyglass's v1 decoding (2024-01, unpinned) needs ``save_results``, added in
0.5.0, so 0.5.0 is the oldest release it can use; 0.5.13 was current when it
was merged. 0.5.12 is the last release whose open-field environments store
no ``distance_between_nodes_`` or ``track_graphDD`` and whose ``RandomWalk``
stores ``use_diffusion``. For the 0.5 releases ``jax`` and ``jaxlib`` are
taken up to 2023-10-08 (``--exclude-newer-package``), because PyPI holds no
jaxlib older than 0.4.17, and ``numba`` is bounded below, because without
it the resolver backtracks to a numba that does not build.

The main environments pin ``pandas<3`` (2.3.3): pandas 2.3, the supported
floor, cannot read the track-graph DataFrames pandas 3 pickles. Each
directory's ``environment.txt`` lists every installed distribution and
``VERSION.json`` the main ones. Results are saved as Spyglass saves them,
with netCDF4 installed (netCDF4/HDF5 files), and once more with the netCDF
writers hidden (``*_results_netcdf3.nc``), as an environment without them
saves them; one netCDF3 file per version is kept. 84259d49 is the last main
before ``predict`` took ``time_edges``, and the only one here with the
since-removed ``clusterless_kde_log``; only its ``clusterless_kde_log``
artifacts are kept. ``inputs.pkl`` is the same for every version and is kept
once, at the top level. ``predict_reference.pkl`` holds each main version's
own decode of its saved models, for 66047901, d5f85823 and 11d9b3de.

11d9b3de is the last main whose ``RandomWalk`` truncated its Gaussian at the
track's edge and normalized each row. Versions whose ``estimate_parameters``
takes ``transition_representation`` also fit
``contfrag_clusterless_2d_structured``: a ``ContFragClusterlessClassifier``
on the clusterless open field in 2 cm bins with
``transition_representation="structured"`` (in a NonLocal detector this
short recording leaves all of the mass in Local, so the walk would not show
in its decode). From 2026-11-04 main's ``save_model`` writes a zip format
that stores fingerprints of the transitions and rebuilds them on load, so
the structured fit is also written as a plain pickle (``*_pickle.pkl``),
the way ``save_model`` wrote models up to 0.6.9, which keeps the fitted
``GaussianGridBlock`` operator of the walk; 0.6.9 itself could not fit a
structured transition. Only that artifact is kept for 11d9b3de (its rows,
results, both model files and the reference). It was generated in the
repository's own environment, so its rows did not go through DataJoint;
the detector holds no pandas objects, so the pandas 3.0 it was written
with does not matter to readers.

Rows are written the way the Spyglass of each era wrote them: as
``vars(model)`` (Spyglass ``master``, da36473c8) and, for versions with
``time_edges``, also as ``get_params()`` plus ``class_name`` (Spyglass PR #1618,
334d1a6b8; before b879fc30e for an ``Environment`` without ``get_params``).
The rows go through DataJoint's blob ``pack``/``unpack``, as a
``DecodingParameters`` insert and fetch do. The models are built and fitted
the way Spyglass ``make()`` does: from the restored row (the base detector for
a ``vars`` row, the NonLocal class when the base detector rejects its keys,
the stored class for a ``class_name`` row), through Spyglass's default
``estimate_decoding_params=1`` branch (``estimate_parameters``; ``predict``
fails for sorted spikes before 0.6.7 and for clusterless before 0.6.9). Results
use the Spyglass layout of the era: unlabelled ``initial_conditions`` and
``("states", "states")`` transitions for 0.6.0 (Spyglass 736d47b3c, 2024),
``interval_labels`` and ``state_bins``/``states_from``/``states_to`` labels
afterwards.

The fits are tiny: 2 s of simulated tracking at 50 Hz, a 1-D track graph
(sorted spikes, 10 cm bins) and a 2-D open field (clusterless, 15 cm bins).
"""

from __future__ import annotations

import copy
import importlib.metadata
import inspect
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import scipy.ndimage
import xarray as xr
from track_linearization import make_track_graph

import non_local_detector as nld
from non_local_detector.environment import Environment
from non_local_detector.models import base as base_module
from non_local_detector.models.base import ClusterlessDetector, SortedSpikesDetector

SAMPLING_FREQUENCY = 50.0
DURATION = 2.0
ENCODING_INTERVAL = (0.0, 1.6)
DECODING_INTERVALS = ((0.4, 0.6), (1.4, 1.6))
MAX_ITER = 2
NODE_POSITIONS = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0)]
EDGES = [(0, 1), (1, 2)]

try:
    from datajoint.blob import pack, unpack
except ImportError:  # the round trip only checks the row is storable
    pack = unpack = None

NEW_API = "time_edges" in inspect.signature(SortedSpikesDetector.predict).parameters
# Versions that can fit a structured (separable Gaussian) transition operator.
STRUCTURED = (
    "transition_representation"
    in inspect.signature(SortedSpikesDetector.estimate_parameters).parameters
)


def simulate() -> dict:
    """Simulate tracking, sorted spikes and clusterless marks (seed 0)."""
    rng = np.random.default_rng(0)
    time = np.arange(0.0, DURATION, 1 / SAMPLING_FREQUENCY)
    # 1-D: back and forth along the L-shaped track, linear distance in [0, 60].
    linear = 30.0 * (1 - np.cos(2 * np.pi * time / 1.5))
    track = np.where(
        (linear <= 30.0)[:, None],
        np.stack([linear, np.zeros_like(linear)], axis=1),
        np.stack([np.full_like(linear, 30.0), linear - 30.0], axis=1),
    )
    field = np.stack(
        [
            15.0 + 12.0 * np.sin(2 * np.pi * time / 2.0),
            15.0 + 12.0 * np.sin(2 * np.pi * time / 1.3 + 0.5),
        ],
        axis=1,
    )

    def poisson_times(rate: np.ndarray) -> np.ndarray:
        n = rng.poisson(rate / SAMPLING_FREQUENCY)
        sample = np.repeat(np.arange(len(time)), n)
        jitter = rng.uniform(-0.5, 0.5, sample.size) / SAMPLING_FREQUENCY
        return np.clip(time[sample] + jitter, time[0], time[-1])

    sorted_spike_times = [
        np.sort(poisson_times(5.0 + 60.0 * np.exp(-((linear - c) ** 2) / 50.0)))
        for c in np.linspace(5.0, 55.0, 5)
    ]
    marks_times, marks = [], []
    for electrode in range(2):
        times, features = [], []
        for unit, center in enumerate([(5.0, 5.0), (25.0, 10.0), (15.0, 25.0)]):
            distance2 = np.sum((field - center) ** 2, axis=1)
            t = poisson_times(5.0 + 40.0 * np.exp(-distance2 / 50.0))
            times.append(t)
            mean = 60.0 + 40.0 * unit + 10.0 * electrode
            features.append(rng.normal(mean, 8.0, size=(t.size, 4)))
        times = np.concatenate(times)
        order = np.argsort(times)
        marks_times.append(times[order])
        marks.append(np.concatenate(features)[order])
    return {
        "time": time,
        "track_position": track,
        "field_position": field,
        "sorted_spike_times": sorted_spike_times,
        "clusterless_spike_times": marks_times,
        "clusterless_features": marks,
    }


# --- Spyglass DecodingParameters rows -------------------------------------
# Copies of the Spyglass serialization helpers that wrote each row format.


def _graph_to_dict(graph) -> dict:
    return {
        "node_positions": [v["pos"] for v in dict(graph.nodes).values()],
        "edges": list(graph.edges),
    }


def _transition_to_dict(transition) -> dict:
    parameters = dict(vars(transition))
    parameters["class_name"] = type(transition).__name__
    return parameters


def _convert_shared(params: dict) -> dict:
    params["continuous_transition_types"] = [
        [_transition_to_dict(t) for t in row]
        for row in params["continuous_transition_types"]
    ]
    params["discrete_transition_type"] = _transition_to_dict(
        params["discrete_transition_type"]
    )
    params["continuous_initial_conditions_types"] = [
        _transition_to_dict(c) for c in params["continuous_initial_conditions_types"]
    ]
    if params["observation_models"] is not None:
        params["observation_models"] = [
            dict(vars(obs)) for obs in params["observation_models"]
        ]
    return params


def _vars_environment(env) -> dict:
    """Environment as Spyglass master stored it: ``vars(env)``."""
    if env.track_graph is not None:
        env.track_graph = _graph_to_dict(env.track_graph)
    if getattr(env, "track_graphDD", None) is not None:
        env.track_graphDD = _graph_to_dict(env.track_graphDD)
    return vars(env)


def vars_row(model) -> dict:
    """Row as Spyglass master inserted it: ``vars(model)`` (da36473c8)."""
    params = copy.deepcopy(vars(model))
    params["environments"] = [_vars_environment(e) for e in params["environments"]]
    return _convert_shared(params)


def get_params_row(model) -> dict:
    """Row as Spyglass PR #1618 inserts it: ``get_params`` + class name."""
    params = model.get_params(deep=False)
    params["class_name"] = type(model).__name__
    params = copy.deepcopy(params)
    environments = []
    for env in params["environments"]:
        if not hasattr(env, "get_params"):
            # PR #1618 before b879fc30e stored vars(env), as master does.
            environments.append(_vars_environment(env))
            continue
        env_params = env.get_params()
        if env_params.get("track_graph") is not None:
            env_params["track_graph"] = _graph_to_dict(env_params["track_graph"])
        environments.append(env_params)
    params["environments"] = environments
    return _convert_shared(params)


def store(row: dict) -> dict:
    """What a DataJoint insert followed by a fetch returns."""
    return row if pack is None else unpack(pack(row))


def restore(row: dict):
    """Rebuild the decoding parameters as that era's Spyglass ``make()`` did."""
    from non_local_detector import continuous_state_transitions as cst
    from non_local_detector import discrete_state_transitions as dst
    from non_local_detector import initial_conditions as ic
    from non_local_detector.observation_models import ObservationModel

    row = copy.deepcopy(row)

    def build(d, module):
        d = dict(d)
        return getattr(module, d.pop("class_name"))(**d)

    config = set(inspect.signature(Environment).parameters)
    environments = []
    for env in row["environments"]:
        env = {k: v for k, v in env.items() if k in config and not k.endswith("_")}
        env.pop("track_graphDD", None)
        env.pop("_is_fitted", None)
        if env.get("track_graph") is not None:
            env["track_graph"] = make_track_graph(**env["track_graph"])
        environments.append(Environment(**env))
    row["environments"] = environments
    row["continuous_transition_types"] = [
        [build(t, cst) for t in r] for r in row["continuous_transition_types"]
    ]
    row["discrete_transition_type"] = build(row["discrete_transition_type"], dst)
    row["continuous_initial_conditions_types"] = [
        build(c, ic) for c in row["continuous_initial_conditions_types"]
    ]
    if row["observation_models"] is not None:
        row["observation_models"] = [
            ObservationModel(**obs) for obs in row["observation_models"]
        ]
    class_name = row.pop("class_name", None)
    if class_name is not None:
        return getattr(nld, class_name)(**row)
    return row


# --- fitting and decoding as Spyglass make() does --------------------------


def decode(
    detector,
    data: dict,
    position: np.ndarray,
    spikes: dict,
    *,
    legacy_layout: bool,
    fit_options: dict | None = None,
) -> xr.Dataset:
    """Spyglass's default ``estimate_decoding_params=1`` branch of ``make()``.

    ``fit_options`` are further ``estimate_parameters`` arguments.
    """
    time = data["time"]
    is_training = (time >= ENCODING_INTERVAL[0]) & (time <= ENCODING_INTERVAL[1])
    kwargs = {"is_training": is_training, "max_iter": MAX_ITER, **(fit_options or {})}
    if NEW_API:
        # Spyglass PR #1618: uniform bins over the tracking range, missing
        # outside the decoding intervals.
        edges = detector.calculate_time_edges(np.array([time[0], time[-1]]), trim=True)
        decode_time = (edges[:-1] + edges[1:]) / 2
        kwargs["time_edges"] = edges
        kwargs["valid_position_intervals"] = np.array([[time[0], time[-1]]])
    else:
        decode_time = time
        kwargs["time"] = time
    is_missing = np.ones(len(decode_time), dtype=bool)
    for start, stop in DECODING_INTERVALS:
        is_missing[(decode_time >= start) & (decode_time <= stop)] = False
    results = detector.estimate_parameters(
        position_time=time,
        position=position,
        **spikes,
        is_missing=is_missing,
        **kwargs,
    )
    if legacy_layout:
        # Spyglass 2024 (736d47b3c): no interval labels, unlabelled arrays.
        results["initial_conditions"] = xr.DataArray(
            detector.initial_conditions_, name="initial_conditions"
        )
        results["discrete_state_transitions"] = xr.DataArray(
            detector.discrete_state_transitions_,
            dims=("states", "states"),
            name="discrete_state_transitions",
        )
        return results
    # Spyglass master (da36473c8) and PR #1618 (334d1a6b8).
    # Spyglass create_interval_labels: -1 outside, else the interval's index.
    labels = scipy.ndimage.label(~is_missing)[0] - 1
    results = results.assign_coords(interval_labels=("time", labels))
    state_names = np.atleast_1d(results.coords["states"].values)
    results["initial_conditions"] = xr.DataArray(
        detector.initial_conditions_,
        dims=("state_bins",),
        coords={"state_bins": results.coords["state_bins"]},
        name="initial_conditions",
    )
    results["discrete_state_transitions"] = xr.DataArray(
        detector.discrete_state_transitions_,
        dims=("states_from", "states_to"),
        coords={"states_from": state_names, "states_to": state_names},
        name="discrete_state_transitions",
    )
    return results


def predict_whole_recording(
    detector, data: dict, position: np.ndarray, spikes: dict
) -> np.ndarray:
    """Acausal posterior of ``predict`` over the whole recording.

    Returns
    -------
    posterior : np.ndarray, shape (n_time_bins, n_state_bins)
    """
    time = data["time"]
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


def save_netcdf3(detector, results: xr.Dataset, path: Path) -> None:
    """``save_results`` as without netCDF4 and h5netcdf: a netCDF3 file."""
    hidden = {name: sys.modules.get(name) for name in ("netCDF4", "h5netcdf")}
    sys.modules.update(dict.fromkeys(hidden))
    try:
        detector.save_results(results, path)
    finally:
        for name, module in hidden.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert path.read_bytes().startswith(b"CDF"), path


def main(out_dir: Path, commit: str | None = None) -> None:
    """Write every artifact to ``out_dir``; ``commit`` is recorded in VERSION.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    version = nld.__version__
    print(f"non_local_detector {version} from {nld.__file__}")  # noqa: T201
    data = simulate()
    # Spyglass master moved to labelled results after 0.6.0 (see decode).
    legacy_layout = version.startswith("0.5.") or version == "0.6.0"

    track_env = Environment(
        environment_name="",
        place_bin_size=10.0,
        track_graph=make_track_graph(NODE_POSITIONS, EDGES),
        edge_order=EDGES,
        edge_spacing=0.0,
    )
    field_env = Environment(environment_name="", place_bin_size=15.0)
    common = {"sampling_frequency": SAMPLING_FREQUENCY}

    presets = {
        "nonlocal_sorted_1d": nld.NonLocalSortedSpikesDetector(
            environments=[track_env], **common
        ),
        "nonlocal_clusterless_2d": nld.NonLocalClusterlessDetector(
            environments=[field_env], **common
        ),
        "contfrag_sorted_default": nld.ContFragSortedSpikesClassifier(),
        "contfrag_clusterless_default": nld.ContFragClusterlessClassifier(),
        "nonlocal_sorted_default": nld.NonLocalSortedSpikesDetector(),
        "nonlocal_clusterless_default": nld.NonLocalClusterlessDetector(),
    }
    # Registered on main from 2025-10 to 2026-10, never released.
    kde_log = "clusterless_kde_log" in getattr(
        base_module, "_CLUSTERLESS_ALGORITHMS", {}
    )
    if kde_log:
        presets["nonlocal_clusterless_kde_log_2d"] = nld.NonLocalClusterlessDetector(
            environments=[field_env],
            clusterless_algorithm="clusterless_kde_log",
            **common,
        )
    if STRUCTURED:
        # Bins fine enough (2 cm, sd 2.45 cm steps) that the walk's boundary
        # treatment visibly changes its transition.
        presets["contfrag_clusterless_2d_structured"] = (
            nld.ContFragClusterlessClassifier(
                environments=[Environment(environment_name="", place_bin_size=2.0)],
                **common,
            )
        )
    rows = {}
    for name, preset in presets.items():
        rows[name] = {"vars": store(vars_row(preset))}
        if NEW_API:
            rows[name]["get_params"] = store(get_params_row(preset))
    with (out_dir / "decoding_parameters.pkl").open("wb") as f:
        pickle.dump(rows, f, protocol=4)

    fits = {
        "nonlocal_sorted_1d": (
            data["track_position"],
            {"spike_times": data["sorted_spike_times"]},
        ),
        "nonlocal_clusterless_2d": (
            data["field_position"],
            {
                "spike_times": data["clusterless_spike_times"],
                "spike_waveform_features": data["clusterless_features"],
            },
        ),
    }
    if kde_log:
        fits["nonlocal_clusterless_kde_log_2d"] = fits["nonlocal_clusterless_2d"]
    fit_options = {}
    if STRUCTURED:
        fits["contfrag_clusterless_2d_structured"] = fits["nonlocal_clusterless_2d"]
        fit_options["contfrag_clusterless_2d_structured"] = {
            "transition_representation": "structured"
        }
    reference = {}
    for name, (position, spikes) in fits.items():
        # Old Spyglass built the base detector from the restored dict;
        # PR #1618 rebuilds the stored class.
        row = rows[name]["get_params" if NEW_API else "vars"]
        restored = restore(row)
        if isinstance(restored, dict):
            clusterless = "clusterless_algorithm" in restored
            base = ClusterlessDetector if clusterless else SortedSpikesDetector
            if not set(restored) <= set(inspect.signature(base).parameters):
                # A NonLocal row the base detector rejects: Spyglass master
                # cannot fit it; PR #1618 (1ad8791a5) rebuilds the NonLocal
                # class, as here.
                base = getattr(
                    nld,
                    "NonLocalClusterlessDetector"
                    if clusterless
                    else "NonLocalSortedSpikesDetector",
                )
            restored = base(**restored)
        results = decode(
            restored,
            data,
            position,
            spikes,
            legacy_layout=legacy_layout,
            fit_options=fit_options.get(name),
        )
        restored.save_results(results, out_dir / f"{name}_results.nc")
        save_netcdf3(restored, results, out_dir / f"{name}_results_netcdf3.nc")
        restored.save_model(out_dir / f"{name}_model.pkl")
        if name in fit_options:
            # A plain pickle, as save_model wrote up to 0.6.9: it keeps the
            # fitted transition operator, which the zip format rebuilds.
            (out_dir / f"{name}_pickle.pkl").write_bytes(
                pickle.dumps(restored, protocol=pickle.HIGHEST_PROTOCOL)
            )
        if NEW_API:
            # This version's own decode of the whole recording with the saved
            # model, for checking that a later release decodes it the same.
            reference[name] = predict_whole_recording(restored, data, position, spikes)

    if reference:
        with (out_dir / "predict_reference.pkl").open("wb") as f:
            pickle.dump(reference, f, protocol=4)
    with (out_dir / "inputs.pkl").open("wb") as f:
        pickle.dump(data, f, protocol=4)
    (out_dir / "environment.txt").write_text(
        "".join(
            sorted(
                {
                    f"{dist.metadata['Name']}=={dist.version}\n"
                    for dist in importlib.metadata.distributions()
                },
                key=str.lower,
            )
        )
    )
    (out_dir / "VERSION.json").write_text(
        json.dumps(
            {
                "non_local_detector": version,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "xarray": xr.__version__,
                "jax": jax.__version__,
                "datajoint_round_trip": pack is not None,
                **({"commit": commit} if commit else {}),
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
