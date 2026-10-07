"""Synchronized, bounded reference/likelihood kernel benchmarks.

Use --baseline-dir with frozen likelihood source files, --output-dir for JSON
and array artifacts, and --repeats >= 5. CPU RSS is sampled at 2 ms; this is a
sampled process peak, including compiled executables and resident arrays, not a
GPU allocation measurement or a full-session throughput claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import platform
import sys
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from non_local_detector.likelihoods import (
    clusterless_kde,
    common,
    sorted_spikes_glm,
    sorted_spikes_kde,
)


class SampledRSS:
    def __enter__(self):
        self.process = psutil.Process()
        self.initial = self.peak = self.process.memory_info().rss
        self.finished = threading.Event()

        def sample():
            while not self.finished.wait(0.002):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.finished.set()
        self.thread.join()


def load_baseline(directory):
    """Load frozen modules and bind their imports to frozen common helpers."""
    loaded = {}
    for name in ["common", "sorted_spikes_kde", "sorted_spikes_glm", "clusterless_kde"]:
        path = directory / "likelihoods" / f"{name}.py"
        module_name = f"_baseline_{name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if name != "common":
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.ImportFrom) and node.module == (
                    "non_local_detector.likelihoods.common"
                ):
                    for imported in node.names:
                        setattr(
                            module,
                            imported.asname or imported.name,
                            getattr(loaded["common"], imported.name),
                        )
        loaded[name] = module
    return loaded


def fixture(dtype):
    """Fixed physical events/weights, with boundaries and ragged dimensions."""
    rng = np.random.default_rng(7301)
    n_rows, n_cells, n_bins = 513, 96, 257
    edges = np.arange(n_rows + 1, dtype=np.float64) * 0.002
    fields = rng.uniform(0.02, 40, (n_cells, n_bins)).astype(dtype)
    spikes = [
        np.sort(np.r_[rng.uniform(0, edges[-1], 10 + cell % 29), edges[100], edges[-1]])
        for cell in range(n_cells)
    ]
    params = {
        "position_time": np.array([0.0, edges[-1]]),
        "position": np.zeros((2, 1)),
        "spike_times": spikes,
        "environment": None,
        "place_fields": jnp.asarray(fields),
        "no_spike_part_log_likelihood": jnp.asarray(fields.sum(axis=0)),
        "is_track_interior": np.ones(n_bins, bool),
        "disable_progress_bar": True,
        "time_edges": edges,
    }
    kde_args = params | {
        "marginal_models": [None] * n_cells,
        "occupancy_model": None,
        "occupancy": None,
        "mean_rates": np.ones(n_cells),
    }
    glm_args = params | {
        "coefficients": np.zeros((n_cells, 1)),
        "emission_design_info": None,
    }
    kde_inputs = []
    for count in [61, 63, 65, 127, 129, 131]:
        kde_inputs.append(
            (
                jnp.asarray(rng.normal(size=(201, 2)).astype(dtype)),
                jnp.asarray(rng.normal(size=(count, 2)).astype(dtype)),
                jnp.ones(2, dtype=dtype),
                jnp.asarray(rng.uniform(0, 2, count).astype(dtype)),
            )
        )
    n_enc, n_dec, n_pos, n_marks = 1025, 513, 1024, 4
    clusterless = {
        "decoding_spike_waveform_features": jnp.asarray(
            rng.normal(size=(n_dec, n_marks)).astype(dtype)
        ),
        "encoding_spike_waveform_features": jnp.asarray(
            rng.normal(size=(n_enc, n_marks)).astype(dtype)
        ),
        "waveform_stds": jnp.ones(n_marks, dtype=dtype),
        "occupancy": jnp.asarray(rng.uniform(0.02, 0.5, n_pos).astype(dtype)),
        "mean_rate": 0.2,
        "position_distance": jnp.asarray(
            rng.uniform(0.0, 0.4, (n_enc, n_pos)).astype(dtype)
        ),
        "block_size": 64,
        "encoding_weights": jnp.asarray(rng.uniform(0, 2, n_enc).astype(dtype)),
    }
    return kde_args, glm_args, kde_inputs, clusterless


def synchronized(call):
    result = call()
    jax.block_until_ready(result)
    return result


def compare(name, calls, repeats, output_dir, rng):
    report, values = {}, {}
    # Clear compiled caches independently to avoid crediting the second variant
    # with a primitive executable compiled by the first.
    for variant, call in calls.items():
        jax.clear_caches()
        with SampledRSS() as memory:
            start = time.perf_counter()
            result = synchronized(call)
            elapsed = time.perf_counter() - start
        values[variant] = np.asarray(result)
        report[variant] = {
            "compile_plus_first_seconds": elapsed,
            "cold_initial_rss_bytes": memory.initial,
            "cold_peak_rss_bytes": memory.peak,
        }
    # Both paths must be warm after the final global cache clear.
    for call in calls.values():
        synchronized(call)
        synchronized(call)
    timings = {variant: [] for variant in calls}
    peaks = {variant: [] for variant in calls}
    for _ in range(repeats):
        for variant in rng.permutation(list(calls)):
            with SampledRSS() as memory:
                start = time.perf_counter()
                synchronized(calls[variant])
                timings[variant].append(time.perf_counter() - start)
            peaks[variant].append(memory.peak)
    for variant in calls:
        times = timings[variant]
        report[variant].update(
            warm_seconds=times,
            warm_median_seconds=float(np.median(times)),
            warm_min_seconds=min(times),
            warm_max_seconds=max(times),
            warm_peak_rss_bytes=max(peaks[variant]),
            output_dtype=str(values[variant].dtype),
            output_shape=list(values[variant].shape),
        )
    baseline = values["baseline"]
    for variant, value in values.items():
        finite = np.isfinite(baseline) & np.isfinite(value)
        delta = np.abs(value[finite] - baseline[finite])
        report[variant]["max_abs_difference"] = float(delta.max(initial=0))
        report[variant]["finite_pattern_equal"] = bool(
            np.array_equal(np.isfinite(value), np.isfinite(baseline))
        )
        report[variant]["paired_warm_ratios"] = (
            np.asarray(timings["baseline"]) / timings[variant]
        ).tolist()
        if (
            max(
                report[variant]["cold_peak_rss_bytes"],
                report[variant]["warm_peak_rss_bytes"],
            )
            >= 2 * 2**30
        ):
            raise RuntimeError("Phase7c agent process exceeded its 2 GiB budget")
    np.savez(output_dir / f"{name}_outputs.npz", **values)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--x64", action="store_true")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("At least five synchronized interleaved repeats are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", args.x64)
    dtype = np.float64 if args.x64 else np.float32
    baseline = load_baseline(args.baseline_dir)
    kde_args, glm_args, kde_inputs, clusterless_args = fixture(dtype)
    np.savez(
        args.output_dir / "inputs.npz",
        edges=kde_args["time_edges"],
        fields=np.asarray(kde_args["place_fields"]),
        summed_rates=np.asarray(kde_args["no_spike_part_log_likelihood"]),
        **{
            f"clusterless_{name}": np.asarray(value)
            for name, value in clusterless_args.items()
        },
        **{
            f"kde_{i}_{name}": np.asarray(value)
            for i, inputs in enumerate(kde_inputs)
            for name, value in zip(
                ("eval", "samples", "std", "weights"), inputs, strict=True
            )
        },
        **{f"spikes_{i}": values for i, values in enumerate(kde_args["spike_times"])},
    )
    rng = np.random.default_rng(7302)
    metadata = {
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "total_host_bytes": psutil.virtual_memory().total,
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "x64_enabled": jax.config.x64_enabled,
        "seed": 7301,
        "repetitions": args.repeats,
        "rss_sampling_seconds": 0.002,
        "baseline_revision": (args.baseline_dir / "revision.txt").read_text().strip(),
        "baseline_source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.baseline_dir / "likelihoods").glob("*.py")
        },
        "source_hashes": {
            Path(module.__file__).name: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (
                common,
                sorted_spikes_kde,
                sorted_spikes_glm,
                clusterless_kde,
            )
        },
        "benchmark_source_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "input_artifact_hash": hashlib.sha256(
            (args.output_dir / "inputs.npz").read_bytes()
        ).hexdigest(),
        "dimensions": {
            "sorted_rows": 513,
            "neurons": 96,
            "spatial_bins": 257,
            "kde_eval": 201,
            "kde_samples": [61, 63, 65, 127, 129, 131],
            "encoding_marks": 1025,
            "decoding_marks": 513,
            "marked_spatial_bins": 1024,
            "mark_features": 4,
            "block_size": 64,
        },
    }
    results = {"metadata": metadata}
    for name, module, module_baseline, kwargs in [
        ("sorted_kde", sorted_spikes_kde, baseline["sorted_spikes_kde"], kde_args),
        ("sorted_glm", sorted_spikes_glm, baseline["sorted_spikes_glm"], glm_args),
    ]:
        function = f"predict_{name.replace('sorted_', 'sorted_spikes_')}_log_likelihood"
        results[name] = compare(
            name,
            {
                "baseline": lambda m=module_baseline, k=kwargs, f=function: getattr(
                    m, f
                )(**k),
                "candidate": lambda m=module, k=kwargs, f=function: getattr(m, f)(**k),
            },
            args.repeats,
            args.output_dir,
            rng,
        )

    def kde_family(module):
        return jnp.concatenate(
            [
                module.block_kde(points, samples, std, 100, weights)
                for points, samples, std, weights in kde_inputs
            ]
        )

    results["kde_shapes"] = compare(
        "kde_shapes",
        {
            "baseline": lambda: kde_family(baseline["common"]),
            "candidate": lambda: kde_family(common),
        },
        args.repeats,
        args.output_dir,
        rng,
    )
    results["clusterless_blocks"] = compare(
        "clusterless_blocks",
        {
            "baseline": lambda: baseline[
                "clusterless_kde"
            ].block_estimate_log_joint_mark_intensity(**clusterless_args),
            "candidate": lambda: clusterless_kde.block_estimate_log_joint_mark_intensity(
                **clusterless_args
            ),
        },
        args.repeats,
        args.output_dir,
        rng,
    )
    path = args.output_dir / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(path)
    for name in results:
        if name != "metadata":
            print(
                name,
                {
                    variant: round(value["warm_median_seconds"], 6)
                    for variant, value in results[name].items()
                },
            )


if __name__ == "__main__":
    main()
