"""Experimental weighted KDE buckets; this does not change fitted models.

Compare exact shapes with three masked sample/evaluation policies on the same
finite inputs. Results establish kernel trade-offs, not native full-hour speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from non_local_detector.likelihoods import clusterless_kde, common


def bucket_size(size, policy):
    if size == 0 or policy == "exact":
        return size
    if policy == "power2":
        return 2 ** math.ceil(math.log2(size))
    if policy == "multiple256":
        return 256 * math.ceil(size / 256)
    return max(size, math.ceil(2 ** (math.ceil(2 * math.log2(size)) / 2)))


def padded(array, size):
    return jnp.asarray(
        np.pad(array, [(0, size - len(array)), *[(0, 0)] * (array.ndim - 1)])
    )


def synchronized(calls):
    return [np.asarray(call()) for call in calls]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--x64", action="store_true")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("at least five interleaved repetitions are required")
    jax.config.update("jax_enable_x64", args.x64)
    dtype = np.float64 if args.x64 else np.float32
    rng = np.random.default_rng(7351)
    fixtures = []
    # Ragged sample and evaluation sizes span neighbouring bucket boundaries.
    for samples, evaluations, marks in zip(
        [0, 61, 63, 65, 127, 129, 131],
        [0, 97, 100, 103, 199, 201, 203],
        [0, 1, 2, 3, 5, 11, 19],
        strict=True,
    ):
        fixtures.append(
            {
                "samples": rng.normal(size=(samples, 2)).astype(dtype),
                "evaluations": rng.normal(size=(evaluations, 2)).astype(dtype),
                "weights": rng.uniform(0, 2, samples).astype(dtype),
                "encoding_marks": rng.normal(size=(samples, 4)).astype(dtype),
                "decoding_marks": rng.normal(size=(marks, 4)).astype(dtype),
                "position_kernel": rng.uniform(0, 0.4, (samples, 1024)).astype(dtype),
                "occupancy": rng.uniform(0.02, 0.5, 1024).astype(dtype),
            }
        )
    policies = ["exact", "power2", "multiple256", "sqrt2"]
    calls, shapes, input_bytes = {}, {}, {}
    for policy in policies:
        calls[policy], shapes[policy], input_bytes[policy] = [], [], 0
        for f in fixtures:
            n_samples, n_eval, n_marks = map(
                len, [f["samples"], f["evaluations"], f["decoding_marks"]]
            )
            ns, ne, nm = [
                bucket_size(size, policy) for size in [n_samples, n_eval, n_marks]
            ]
            s, e, w, em, dm, pk, occ = (
                padded(f["samples"], ns),
                padded(f["evaluations"], ne),
                padded(f["weights"], ns),
                padded(f["encoding_marks"], ns),
                padded(f["decoding_marks"], nm),
                padded(f["position_kernel"], ns),
                jnp.asarray(f["occupancy"]),
            )
            input_bytes[policy] += sum(x.nbytes for x in [s, e, w, em, dm, pk, occ])
            shapes[policy].append([ns, ne, nm])
            calls[policy].extend(
                [
                    lambda e=e, s=s, w=w, n=n_eval: common.kde(
                        e, s, jnp.ones(2, dtype=dtype), w
                    )[:n],
                    lambda dm=dm, em=em, occ=occ, pk=pk, w=w, n=n_marks: (
                        clusterless_kde.block_estimate_log_joint_mark_intensity(
                            dm,
                            em,
                            jnp.ones(4, dtype=dtype),
                            occ,
                            5.0,
                            pk,
                            block_size=10000,
                            encoding_weights=w,
                        )[:n]
                    ),
                ]
            )
    report, outputs = {}, {}
    for policy in policies:
        jax.clear_caches()
        start = time.perf_counter()
        outputs[policy] = synchronized(calls[policy])
        report[policy] = {
            "compile_plus_first_seconds": time.perf_counter() - start,
            "kde_executable_shapes": common.kde._cache_size(),
            "marked_executable_shapes": clusterless_kde._blocked_joint_mark_intensity._cache_size(),
            "shapes": shapes[policy],
            "resident_input_bytes": input_bytes[policy],
            "warm_seconds": [],
        }
    for policy in policies:
        synchronized(calls[policy])
    for _ in range(args.repeats):
        for policy in rng.permutation(policies):
            start = time.perf_counter()
            synchronized(calls[policy])
            report[policy]["warm_seconds"].append(time.perf_counter() - start)
    for policy in policies:
        deltas = []
        for exact, candidate in zip(outputs["exact"], outputs[policy], strict=True):
            np.testing.assert_allclose(candidate, exact, rtol=1e-6, atol=1e-6)
            deltas.append(float(np.max(np.abs(exact - candidate), initial=0)))
        report[policy].update(
            max_abs_difference=max(deltas),
            warm_median_seconds=float(np.median(report[policy]["warm_seconds"])),
        )
    source_hashes = {
        Path(module.__file__).name: hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in [common, clusterless_kde]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "inputs.npz",
        **{
            f"{index}_{name}": value
            for index, f in enumerate(fixtures)
            for name, value in f.items()
        },
    )
    result = {
        "metadata": {
            "seed": 7351,
            "x64_enabled": jax.config.x64_enabled,
            "devices": [str(device) for device in jax.devices()],
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_hashes": source_hashes,
            "input_sha256": hashlib.sha256(
                (args.output / "inputs.npz").read_bytes()
            ).hexdigest(),
            "final_rss_bytes": psutil.Process().memory_info().rss,
            "measurement": "synchronized kernel calls; padding prepared before timing; all policies' inputs resident; final RSS is not a transient peak",
        },
        "policies": report,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
