"""Attribute EM parameter-estimation time and memory to its stages.

Runs ``estimate_parameters`` on a real recording (``--recording``) or on the
``compare_prediction_modes.py`` simulated session, three ways in one process:

1. ``first``: instrumented, including compilation. XLA compilations are
   attributed to the stage that triggered them.
2. ``warm``: uninstrumented, for the total without added synchronization.
3. ``instrumented``: warm and instrumented. Each stage is wrapped in a
   ``jax.profiler.TraceAnnotation`` and synchronized with ``block_until_ready``
   at entry and exit, so stage times are attributable. The extra syncs remove
   asynchronous overlap, so stages sum to slightly more than the ``warm`` total.

Stages, nested as they run (``estimate_parameters`` exclusive time is the host
glue between them):

- ``fit``: the internal refit, split into environment setup (grid and
  all-pairs graph distances), state index, discrete and continuous
  transitions, and the encoding fit;
- ``e_step``: likelihood (``compute_log_likelihood``) and the
  filter/smoother (dense, or the checkpointed driver with
  ``--transition-representation``), whose exclusive time includes the host
  copies of its posteriors;
- ``encoding_m_step`` and ``discrete_m_step``;
- ``results``: building the returned Dataset, with ``masked_posterior``
  for each spatial array expanded to every state bin.

Each stage records inclusive and exclusive seconds, compile count and
seconds, and how much it raised the device and host peaks. Optional
``--xprof-dir`` traces one warm iteration for kernel-level analysis
(TensorBoard or xprof), and ``--cprofile`` lists the host functions with the
most own time in an uninstrumented warm run.

A recording is an ``.npz`` with ``position_time`` (n,), ``position`` (n, 2),
``spike_times_flat`` and ``spike_offsets`` (per unit or electrode), optional
``features_flat`` (clusterless marks, aligned with the spikes), and a
linearized track graph: ``track_node_ids``, ``track_node_pos``,
``track_edges``, ``track_edge_distance``, ``track_edge_id``, ``edge_order``
and ``edge_spacing``. Recordings with ``features_flat`` use the clusterless
detector. ``--environment 2d`` ignores the track graph and bins the head
position on a 2-D grid with an inferred track interior instead.

Example::

    uv run python benchmarks/profile_em.py --recording session.npz \\
        --duration 120 --bin-size 2 --max-iter 3 --output /tmp/em-profile
"""

import argparse
import cProfile
import gc
import inspect
import io
import json
import logging
import os
import platform
import pstats
import re
import sys
import threading
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
import psutil

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_prediction_modes import (  # noqa: E402
    provenance,
    set_population_defaults,
    workload,
)

COMPILE_MESSAGE = re.compile(r"Finished XLA compilation of (\S+) in ([0-9.e-]+) sec")


def uniform_edges(start, stop, sampling_frequency):
    """Edges of whole bins at ``1 / sampling_frequency`` from ``start``.

    Matches ``calculate_time_edges(..., trim=True)``, which older checkouts lack.
    """
    try:
        from non_local_detector import calculate_time_edges
    except ImportError:
        n_bins = int(np.floor((stop - start) * sampling_frequency + 1e-9))
        return start + np.arange(n_bins + 1) / sampling_frequency
    return calculate_time_edges((start, stop), sampling_frequency, trim=True)


def load_recording(path, start, duration, sampling_frequency):
    """EM inputs and environment arguments for a window of a recording."""
    import networkx as nx

    data = np.load(path)
    position_time = data["position_time"]
    t0 = position_time[0] + start
    t1 = (
        position_time[-1] if duration is None else min(t0 + duration, position_time[-1])
    )
    keep = (position_time >= t0) & (position_time <= t1)
    offsets = data["spike_offsets"]
    flat_times = data["spike_times_flat"]
    spike_times, features = [], []
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        times = flat_times[begin:end]
        in_window = (times >= t0) & (times < t1)
        spike_times.append(times[in_window])
        if "features_flat" in data:
            features.append(data["features_flat"][begin:end][in_window])
    graph = nx.Graph()
    for node, pos in zip(data["track_node_ids"], data["track_node_pos"], strict=True):
        graph.add_node(int(node), pos=tuple(pos))
    for (u, v), distance, edge_id in zip(
        data["track_edges"],
        data["track_edge_distance"],
        data["track_edge_id"],
        strict=True,
    ):
        graph.add_edge(int(u), int(v), distance=float(distance), edge_id=int(edge_id))
    em_kwargs = {
        "position_time": position_time[keep],
        "position": data["position"][keep],
        "spike_times": spike_times,
        "time_edges": uniform_edges(
            position_time[keep][0], position_time[keep][-1], sampling_frequency
        ),
    }
    if features:
        em_kwargs["spike_waveform_features"] = features
    environment_kwargs = {
        "track_graph": graph,
        "edge_order": [tuple(int(n) for n in edge) for edge in data["edge_order"]],
        "edge_spacing": data["edge_spacing"].tolist(),
    }
    description = {
        "recording": str(path),
        "recording_source": str(data["source"]) if "source" in data else None,
        "exporter_sha256": (
            str(data["exporter_sha256"]) if "exporter_sha256" in data else None
        ),
        "window_s": [float(t0 - position_time[0]), float(t1 - position_time[0])],
        "n_units": len(spike_times),
        "n_spikes": int(sum(len(times) for times in spike_times)),
        "mark_dimensions": int(features[0].shape[1]) if features else None,
    }
    family = "clusterless" if features else "sorted"
    return family, em_kwargs, environment_kwargs, description


def sync(value=None):
    """Wait for ``value``'s arrays, then for the device queue.

    Stage outputs can be NumPy, JAX arrays, or detector state updated in
    place, so a trivial computation queued after the stage also waits for work
    the stage dispatched but did not return.
    """
    import jax
    import jax.numpy as jnp

    for leaf in jax.tree_util.tree_leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    (jnp.zeros(()) + 1).block_until_ready()


class DevicePeak:
    """Device ``bytes_in_use`` and ``peak_bytes_in_use`` (None on CPU)."""

    def __init__(self, jax):
        self.device = jax.local_devices()[0]

    def stats(self):
        stats = self.device.memory_stats() or {}
        return stats.get("bytes_in_use"), stats.get("peak_bytes_in_use")


class HostPeak:
    """Peak resident set size, sampled every 20 ms in a daemon thread."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def reset(self):
        """Start a new peak from the current resident set size."""
        self.peak = self.process.memory_info().rss

    def stop(self):
        self._stop.set()
        self._thread.join()


class StageProfiler(logging.Handler):
    """Times nested stages; also a log handler that attributes XLA compiles."""

    def __init__(self, jax, synchronize, host_peak):
        super().__init__(level=logging.DEBUG)
        self.jax = jax
        self.synchronize = synchronize
        self.host_peak = host_peak
        self.device = DevicePeak(jax)
        self.records = []
        self.missing_stages = []
        self.stack = []
        self.e_steps = 0

    def emit(self, record):
        match = COMPILE_MESSAGE.search(record.getMessage())
        if match and self.stack:
            frame = self.stack[-1]
            frame["compiles"] += 1
            frame["compile_seconds"] += float(match.group(2))
            frame["compiled"][match.group(1)] += 1

    def wrap(self, name, function):
        @wraps(function)
        def timed(*args, **kwargs):
            label = name
            if name == "e_step":
                self.e_steps += 1
                label = f"e_step[{self.e_steps}]"
            if self.synchronize:
                sync()
            frame = {
                "path": "/".join([f["label"] for f in self.stack] + [label]),
                "label": label,
                "children": 0.0,
                "compiles": 0,
                "compile_seconds": 0.0,
                "compiled": defaultdict(int),
                "device_peak_before": self.device.stats()[1],
                "host_peak_before": self.host_peak.peak if self.host_peak else None,
            }
            self.stack.append(frame)
            start = time.perf_counter()
            try:
                with self.jax.profiler.TraceAnnotation(label):
                    result = function(*args, **kwargs)
                    if self.synchronize:
                        sync(result)
            finally:
                seconds = time.perf_counter() - start
                self.stack.pop()
                if self.stack:
                    self.stack[-1]["children"] += seconds
                in_use, peak = self.device.stats()
                self.records.append(
                    {
                        "path": frame["path"],
                        "seconds": seconds,
                        "exclusive_seconds": seconds - frame["children"],
                        "compiles": frame["compiles"],
                        "compile_seconds": frame["compile_seconds"],
                        "compiled": dict(frame["compiled"]),
                        "device_bytes_in_use_after": in_use,
                        "device_peak_raised_bytes": (
                            None
                            if peak is None
                            else peak - (frame["device_peak_before"] or 0)
                        ),
                        "host_peak_raised_bytes": (
                            None
                            if self.host_peak is None
                            else self.host_peak.peak - frame["host_peak_before"]
                        ),
                    }
                )
            return result

        return timed

    def summary(self):
        """Totals per stage path, with ``e_step[k]`` also pooled as ``e_step[*]``."""
        totals = {}
        for record in self.records:
            for path in {record["path"], re.sub(r"\[\d+\]", "[*]", record["path"])}:
                entry = totals.setdefault(
                    path,
                    {
                        "calls": 0,
                        "seconds": 0.0,
                        "exclusive_seconds": 0.0,
                        "compiles": 0,
                        "compile_seconds": 0.0,
                    },
                )
                entry["calls"] += 1
                for key in ("seconds", "exclusive_seconds", "compile_seconds"):
                    entry[key] += record[key]
                entry["compiles"] += record["compiles"]
        return dict(sorted(totals.items()))


@contextmanager
def instrumented(profiler, detector_class):
    """Patch the EM stages of ``detector_class`` and ``models.base`` in place."""
    import non_local_detector.checkpointed_inference as checkpointed
    import non_local_detector.environment as environment_module
    import non_local_detector.models.base as base

    def defining_class(cls, attribute):
        return next(
            (owner for owner in cls.__mro__ if attribute in owner.__dict__), None
        )

    targets = [
        (detector_class, "estimate_parameters", "estimate_parameters"),
        (detector_class, "fit", "fit"),
        (detector_class, "initialize_environments", "environments"),
        (environment_module.Environment, "fit_place_grid", "fit_place_grid"),
        (detector_class, "initialize_state_index", "state_index"),
        (
            detector_class,
            "initialize_discrete_state_transition",
            "discrete_transition",
        ),
        (
            detector_class,
            "initialize_continuous_state_transition",
            "continuous_transition",
        ),
        (detector_class, "fit_encoding_model", "encoding_fit"),
        (detector_class, "_predict", "e_step"),
        (detector_class, "_checkpointed_e_step", "e_step"),
        (detector_class, "compute_log_likelihood", "likelihood"),
        (base, "chunked_filter_smoother", "filter_smoother"),
        (base, "chunked_filter_smoother_covariate_dependent", "filter_smoother"),
        (base, "_estimate_discrete_transition", "discrete_m_step"),
        (base, "_discrete_transition_from_statistics", "discrete_m_step"),
        (checkpointed, "checkpointed_forward_backward", "filter_smoother"),
        (detector_class, "_convert_results_to_xarray", "results"),
        (detector_class, "_create_masked_posterior", "masked_posterior"),
    ]
    saved = []
    try:
        for source, attribute, name in targets:
            # Methods are patched on the class that defines them, which the
            # detector inherits from; module functions where EM looks them up.
            owner = (
                defining_class(source, attribute)
                if isinstance(source, type)
                else source
            )
            if owner is None or attribute not in owner.__dict__:
                profiler.missing_stages.append(attribute)  # older checkouts
                continue
            original = owner.__dict__[attribute]
            saved.append((owner, attribute, original))
            if isinstance(original, staticmethod):
                wrapped = staticmethod(profiler.wrap(name, original.__func__))
            else:
                wrapped = profiler.wrap(name, original)
            setattr(owner, attribute, wrapped)
        yield
    finally:
        for owner, attribute, original in reversed(saved):
            setattr(owner, attribute, original)


def relabel_m_step_fits(records):
    """Name encoding fits by caller: the internal fit or EM's M-step."""
    for record in records:
        parts = record["path"].split("/")
        if parts[-1] == "encoding_fit" and "fit" not in parts[:-1]:
            parts[-1] = "encoding_m_step"
            record["path"] = "/".join(parts)
    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recording", type=Path, help="recording .npz (see above)")
    source.add_argument(
        "--simulated",
        choices=["sorted", "clusterless"],
        help="use the compare_prediction_modes.py simulated session instead",
    )
    parser.add_argument(
        "--environment",
        choices=["track", "2d"],
        default="track",
        help="recordings: linearized track graph or 2-D grid of the head position",
    )
    parser.add_argument("--start", type=float, default=0.0, help="s into recording")
    parser.add_argument(
        "--duration", type=float, default=None, help="s (default: whole recording)"
    )
    parser.add_argument("--sampling-frequency", type=float, default=500.0, help="Hz")
    parser.add_argument("--bin-size", type=float, default=2.0, help="cm")
    parser.add_argument("--arena", type=float, default=180.0, help="simulated, cm")
    parser.add_argument("--population", type=int, default=None, help="simulated")
    parser.add_argument("--spike-rate", type=float, default=None, help="simulated")
    parser.add_argument("--rate-spread", type=float, default=0.0, help="simulated")
    parser.add_argument("--mark-dimensions", type=int, default=4, help="simulated")
    parser.add_argument("--algorithm", default=None, help="likelihood algorithm")
    parser.add_argument(
        "--detector-kwargs",
        type=json.loads,
        default={},
        help="JSON of extra detector constructor arguments",
    )
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument(
        "--transition-representation",
        choices=["dense", "structured", "auto"],
        default=None,
        help="EM transition representation (structured runs checkpointed EM)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="checkpointed EM chunk rows"
    )
    parser.add_argument(
        "--replay-cache-bytes",
        type=int,
        default=None,
        help="checkpointed EM host replay cache",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=None,
        help="checkpointed EM: store final outputs here (one directory per run)",
    )
    parser.add_argument(
        "--runs",
        default="first,warm,instrumented",
        help="comma-separated subset of first, warm, instrumented",
    )
    parser.add_argument("--xprof-dir", type=Path, default=None)
    parser.add_argument(
        "--xprof-python-tracer-level",
        type=int,
        default=0,
        help="jax.profiler python_tracer_level for the trace (1 records every call)",
    )
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--require-backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax

    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    backend = jax.default_backend()
    if args.require_backend and backend != args.require_backend:
        parser.error(f"default backend is {backend}, not {args.require_backend}")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.recording is not None:
        family, em_kwargs, environment_kwargs, description = load_recording(
            args.recording, args.start, args.duration, args.sampling_frequency
        )
        if args.environment == "2d":
            environment_kwargs = {}
        environment_kwargs["place_bin_size"] = args.bin_size
        description["environment"] = args.environment
    else:
        family = args.simulated
        args.family = family
        args.duration = 60.0 if args.duration is None else args.duration
        args.encoding_duration = args.duration
        set_population_defaults(args)
        fit_kwargs, predict_kwargs = workload(args)
        em_kwargs = {
            key: fit_kwargs[key]
            for key in (
                "position_time",
                "position",
                "spike_times",
                "spike_waveform_features",
            )
            if key in fit_kwargs
        }
        em_kwargs["time_edges"] = predict_kwargs["time_edges"]
        environment_kwargs = {
            "place_bin_size": args.bin_size,
            "position_range": ((0, args.arena), (0, args.arena)),
        }
        description = {
            "simulated": family,
            "duration_s": args.duration,
            "population": args.population,
            "spike_rate_hz": args.spike_rate,
        }
    detector_class = (
        NonLocalSortedSpikesDetector
        if family == "sorted"
        else NonLocalClusterlessDetector
    )
    detector_kwargs = dict(args.detector_kwargs)
    if args.algorithm is not None:
        key = (
            "sorted_spikes_algorithm" if family == "sorted" else "clusterless_algorithm"
        )
        detector_kwargs[key] = args.algorithm
    if args.recording is None:
        detector_kwargs.setdefault("infer_track_interior", False)

    def run_em(max_iter=args.max_iter):
        model = detector_class(
            environments=Environment(**environment_kwargs), **detector_kwargs
        )
        kwargs = dict(em_kwargs)
        if "time_edges" not in inspect.signature(model.estimate_parameters).parameters:
            # Checkouts before the time-edge contract decode at bin centers.
            edges = kwargs.pop("time_edges")
            kwargs["time"] = (edges[:-1] + edges[1:]) / 2
        if args.transition_representation is not None:
            kwargs["transition_representation"] = args.transition_representation
        if args.chunk_size is not None:
            kwargs["chunk_size"] = args.chunk_size
        if args.replay_cache_bytes is not None:
            kwargs["replay_cache_bytes"] = args.replay_cache_bytes
        if args.result_path is not None:
            run_em.count += 1
            kwargs["result_path"] = args.result_path / f"run{run_em.count}"
        results = model.estimate_parameters(
            **kwargs, max_iter=max_iter, n_chunks=args.n_chunks
        )
        states = np.asarray(results["acausal_state_probabilities"])
        sync()
        return model, results, states

    run_em.count = 0
    host_peak = HostPeak()
    compile_loggers = [
        logging.getLogger(name)
        for name in (
            "jax._src.dispatch",
            "jax._src.interpreters.pxla",
            "jax._src.compiler",
        )
    ]
    jax.config.update("jax_log_compiles", True)
    report = {
        "family": family,
        **description,
        "rows": len(em_kwargs["time_edges"]) - 1,
        "bin_size_cm": args.bin_size,
        "algorithm": args.algorithm,
        "detector_kwargs": args.detector_kwargs,
        "max_iter": args.max_iter,
        "n_chunks": args.n_chunks,
        "transition_representation": args.transition_representation,
        "chunk_size": args.chunk_size,
        "replay_cache_bytes": args.replay_cache_bytes,
        "result_path": None if args.result_path is None else str(args.result_path),
        "backend": backend,
        "device": str(jax.local_devices()[0]),
        "jax": jax.__version__,
        **provenance(__file__),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "runs": {},
        "status": "started",
    }

    def profiled_run(label, synchronize, max_iter=args.max_iter):
        profiler = StageProfiler(jax, synchronize, host_peak)
        with ExitStack() as stack:
            for logger in compile_loggers:
                logger.addHandler(profiler)
                stack.callback(setattr, logger, "propagate", logger.propagate)
                stack.callback(logger.setLevel, logger.level)
                logger.setLevel(logging.DEBUG)
                logger.propagate = False
                stack.callback(logger.removeHandler, profiler)
            stack.enter_context(instrumented(profiler, detector_class))
            start = time.perf_counter()
            model, results, states = run_em(max_iter)
            seconds = time.perf_counter() - start
        records = relabel_m_step_fits(profiler.records)
        return (
            model,
            results,
            states,
            {
                "seconds": seconds,
                "stages": profiler.summary(),
                "records": records,
                "missing_stages": profiler.missing_stages,
            },
        )

    def write_report():
        # Written after each step, so a crash in a later optional step (the
        # profiler tracer, for example) keeps the measurements already taken.
        report["peak_host_rss_bytes"] = host_peak.peak
        report["peak_device_bytes"] = DevicePeak(jax).stats()[1]
        (args.output / "report.json").write_text(json.dumps(report, indent=1))

    try:
        runs = [run.strip() for run in args.runs.split(",") if run.strip()]
        model = results = states = None
        overall_host_peak = host_peak.peak
        for run in runs:
            # Release the previous run's model and outputs (a full posterior
            # can be tens of GB) so each run's peaks are its own.
            model = results = None
            gc.collect()
            overall_host_peak = max(overall_host_peak, host_peak.peak)
            host_peak.reset()
            if run == "warm":
                start = time.perf_counter()
                model, results, states = run_em()
                report["runs"]["warm"] = {"seconds": time.perf_counter() - start}
            elif run in ("first", "instrumented"):
                model, results, states, report["runs"][run] = profiled_run(run, True)
            else:
                parser.error(f"unknown run {run!r}")
            report["runs"][run]["peak_host_rss_bytes"] = host_peak.peak
            report["runs"][run]["n_iter"] = int(model.n_iter_)
            report["runs"][run]["marginal_log_likelihoods"] = [
                float(value)
                for value in np.atleast_1d(
                    results.attrs.get("marginal_log_likelihoods", [])
                )
            ]
            print(f"{run}: {report['runs'][run]['seconds']:.1f} s", flush=True)
            write_report()
        if model is not None:
            report["state_bins"] = int(model.state_ind_.shape[0])
            report["interior_state_bins"] = int(
                model.is_track_interior_state_bins_.sum()
            )
            np.save(args.output / "acausal_state_probabilities.npy", states)
        model = results = None
        gc.collect()
        host_peak.peak = max(overall_host_peak, host_peak.peak)
        if args.xprof_dir is not None:
            report["xprof"] = "started"
            write_report()
            # One iteration with stage annotations but no added syncs, so the
            # trace keeps the production overlap of host and device work. The
            # Python tracer records every call; over a whole EM iteration it
            # adds enough events to overflow the tracer, so it is off by default.
            options = jax.profiler.ProfileOptions()
            options.python_tracer_level = args.xprof_python_tracer_level
            profiler = StageProfiler(jax, False, None)
            with instrumented(profiler, detector_class):
                with jax.profiler.trace(str(args.xprof_dir), profiler_options=options):
                    run_em(max_iter=1)
            report["xprof"] = str(args.xprof_dir)
            write_report()
        if args.cprofile:
            report["cprofile"] = "started"
            write_report()
            host_peak.stop()  # keep the sampling thread out of the profile
            profile = cProfile.Profile()
            profile.enable()
            run_em()
            profile.disable()
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats("tottime").print_stats(40)
            (args.output / "cprofile.txt").write_text(stream.getvalue())
            report["cprofile"] = "cprofile.txt"
        report["status"] = "ok"
    except Exception as error:  # record the failure mode (e.g. out of memory)
        report["status"] = "failed"
        message = (str(error).splitlines() or [""])[0][:400]
        report["error"] = f"{type(error).__name__}: {message}"
        raise
    finally:
        host_peak.stop()
        write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
