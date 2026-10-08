"""Compare exact likelihood replay with an external bounded disk-cache prototype.

This measures the existing checkpoint engine on cheap analytic likelihoods,
not neural likelihoods or a proposed production cache mode. Every invocation
uses fresh checkpoint/cache directories, caches at most one likelihood chunk,
and leaves the engine's replay checksum check enabled. Filesystem reads are
of just-written files; no cold-disk, fsync, or storage-durability claim is made.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import tempfile
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psutil

import non_local_detector.checkpointed_inference as engine


class DeviceMemory:
    """Read total device-wide NVML memory without changing allocator policy."""

    class Info(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    def __init__(self, enabled):
        self.library = None
        if not enabled:
            return
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not uuid.startswith("GPU-") or "," in uuid:
            raise ValueError("GPU measurement requires one CUDA_VISIBLE_DEVICES UUID")
        library = ctypes.CDLL("libnvidia-ml.so.1")
        self.check(library.nvmlInit_v2())
        library.nvmlDeviceGetHandleByUUID.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self.Info),
        ]
        self.handle = ctypes.c_void_p()
        self.check(
            library.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.handle))
        )
        self.library = library

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML failed with status {code}")

    def used(self):
        if self.library is None:
            return None
        info = self.Info()
        self.check(
            self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info))
        )
        return int(info.used)


class MemorySampler:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.host = self.process.memory_info().rss
        self.gpu = device.used()
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample_once(self):
        self.host = max(self.host, self.process.memory_info().rss)
        used = self.device.used()
        if used is not None:
            self.gpu = max(self.gpu, used)

    def sample(self):
        try:
            while not self.stopped.wait(0.02):
                self.sample_once()
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample_once()
        self.stopped.set()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("Memory sampler failed") from self.error


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(engine.__file__).parent
    files = {
        str(path.relative_to(root)): file_hash(path)
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
        and not path.name.startswith("._")
    }
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--checkpoint-sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.rows <= 4096
        or not 4 <= args.hidden <= 512
        or args.hidden % 4
        or args.repeats != 5
        or any(not 1 <= length <= 256 for length in args.checkpoint_sizes)
    ):
        parser.error("bounded prototype requires T<=4096, N<=512, L<=256, five pairs")
    # Includes a conservative interpreter/compiler/context reserve; this is
    # a preflight estimate, not a measured allocator guarantee.
    chunk_bytes = max(args.checkpoint_sizes) * args.hidden * 4
    preflight = 1536 * 1024**2 + 12 * chunk_bytes + args.hidden**2 * 8
    if preflight >= 2 * 1024**3:
        raise MemoryError("Prototype preflight exceeds 2 GiB")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    devices = jax.devices(args.platform)
    if args.platform == "gpu" and not all(
        "NVIDIA" in device.device_kind for device in devices
    ):
        raise RuntimeError("GPU memory qualification requires NVIDIA/NVML")
    with jax.default_device(devices[0]):
        run(args, devices, preflight)


def run(args, devices, preflight):
    args.output.mkdir(parents=True, exist_ok=False)
    source_before = source_hashes()
    engine_before = file_hash(engine.__file__)
    script_before = file_hash(__file__)
    edges = np.arange(args.rows + 1, dtype=np.float64) * 0.002
    initial = np.arange(1, args.hidden + 1, dtype=np.float32)
    initial /= initial.sum()
    matrix = np.eye(args.hidden, dtype=np.float32) * 0.97 + 0.03 / args.hidden
    state_ind = np.arange(args.hidden) % 4
    missing = np.arange(args.rows) % 97 == 0
    columns = jnp.arange(args.hidden, dtype=jnp.float32)[None, :]

    @jax.jit
    def analytic(rows):
        phase = rows[:, None] * np.float32(0.013) + columns * np.float32(0.007)
        return -jnp.square(jnp.sin(phase)) - np.float32(0.1) * jnp.cos(phase * 0.7)

    device_memory = DeviceMemory(args.platform == "gpu")
    records = []
    references = {}
    outputs = (
        "acausal_state_probabilities",
        "causal_state_probabilities",
        "predictive_state_probabilities",
    )

    def invoke(length, strategy, phase, repetition):
        generated = reads = writes = cache_bytes = allocated_bytes = max_host_chunk = 0
        seen = set()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix=f"{strategy}-{length}-", dir=args.output
        ) as directory:
            folder = Path(directory)
            cache = folder / "likelihoods"
            cache.mkdir()

            def callback(full_edges, *, row_slice, is_missing):
                nonlocal generated, reads, writes, max_host_chunk
                assert full_edges is edges
                start, stop = row_slice.start, row_slice.stop
                assert stop - start <= length
                filename = cache / f"{start}-{stop}.npy"
                if strategy == "disk_cache" and start in seen:
                    values = np.load(filename, allow_pickle=False)
                    reads += 1
                    max_host_chunk = max(max_host_chunk, values.nbytes)
                    return values
                generated += 1
                values = analytic(jnp.arange(start, stop, dtype=jnp.float32))
                if strategy == "disk_cache":
                    host = np.asarray(values)
                    max_host_chunk = max(max_host_chunk, host.nbytes)
                    np.save(filename, host, allow_pickle=False)
                    writes += 1
                    seen.add(start)
                return values

            with MemorySampler(device_memory) as memory:
                result = engine.checkpointed_forward_backward(
                    edges,
                    initial,
                    callback,
                    transition_matrix=matrix,
                    state_ind=state_ind,
                    n_states=4,
                    is_missing=missing,
                    chunk_size=length,
                    checkpoint_dir=folder / "checkpoints",
                    return_outputs=outputs,
                    evidence_accumulation="stable",
                )
                values = {
                    name: np.asarray(result.dataset[name]).copy() for name in outputs
                }
                evidence = np.asarray(result.marginal_log_likelihood)
                jax.block_until_ready((values, evidence))
                elapsed = time.perf_counter() - started
                cache_bytes = sum(path.stat().st_size for path in cache.glob("*.npy"))
                allocated_bytes = sum(
                    path.stat().st_blocks * 512 for path in cache.glob("*.npy")
                )
            chunks = int(np.ceil(args.rows / length))
            assert result.diagnostics["likelihood_evaluations"] == 2 * chunks
            assert generated == (chunks if strategy == "disk_cache" else 2 * chunks)
            assert reads == writes == (chunks if strategy == "disk_cache" else 0)
            assert result.diagnostics["checkpoint_cache_entries"] == 1
            assert not list((folder / "checkpoints").glob("checkpoints-*"))
            if length not in references:
                references[length] = (values, evidence)
            else:
                expected, expected_evidence = references[length]
                for name in outputs:
                    np.testing.assert_array_equal(values[name], expected[name])
                np.testing.assert_array_equal(evidence, expected_evidence)
            stats = devices[0].memory_stats() or {}
            record = {
                "checkpoint_size": length,
                "strategy": strategy,
                "phase": phase,
                "repetition": repetition,
                "prediction_seconds": elapsed,
                "generated_chunks": generated,
                "disk_cache_reads": reads,
                "disk_cache_writes": writes,
                "likelihood_cache_logical_bytes": cache_bytes,
                "likelihood_cache_allocated_bytes": allocated_bytes,
                "max_likelihood_chunk_bytes": length * args.hidden * 4,
                "max_callback_host_chunk_bytes": max_host_chunk,
                "checkpoint_bytes": result.diagnostics["checkpoint_bytes"],
                "peak_process_rss_bytes": memory.host,
                "peak_total_nvml_bytes": memory.gpu,
                "jax_allocator_stats_process_lifetime": stats,
                "state_probability_hashes": {
                    name: hashlib.sha256(array.tobytes()).hexdigest()
                    for name, array in values.items()
                },
                "stable_evidence": float(evidence),
                "stable_evidence_sha256": hashlib.sha256(
                    evidence.tobytes()
                ).hexdigest(),
                "pair_outputs_bitwise_equal": True,
            }
        assert not folder.exists()
        record["fresh_directory_removed"] = True
        record["end_to_end_seconds_including_checks_cleanup"] = (
            time.perf_counter() - started
        )
        if memory.host >= 2 * 1024**3 or (memory.gpu or 0) >= 2 * 1024**3:
            raise MemoryError("Measured prototype memory exceeds 2 GiB")
        records.append(record)
        return values, evidence

    for length in args.checkpoint_sizes:
        for strategy in ("recompute", "disk_cache"):
            invoke(length, strategy, "first_invocation_shared_process", -1)
        for repetition in range(5):
            strategies = ("recompute", "disk_cache")
            if repetition % 2:
                strategies = tuple(reversed(strategies))
            for strategy in strategies:
                invoke(length, strategy, "warm_interleaved", repetition)
    # Checkpoint lengths may change floating-point grouping. Existing core
    # parity tolerance is retained; within each matched cache pair is bitwise.
    first, first_evidence = references[args.checkpoint_sizes[0]]
    for values, evidence in references.values():
        for name in outputs:
            np.testing.assert_allclose(values[name], first[name], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(evidence, first_evidence, rtol=1e-5, atol=1e-6)
    assert source_hashes() == source_before
    assert file_hash(engine.__file__) == engine_before
    assert file_hash(__file__) == script_before
    medians = {}
    for length in args.checkpoint_sizes:
        medians[str(length)] = {
            strategy: float(
                np.median(
                    [
                        record["prediction_seconds"]
                        for record in records
                        if record["checkpoint_size"] == length
                        and record["strategy"] == strategy
                        and record["phase"] == "warm_interleaved"
                    ]
                )
            )
            for strategy in ("recompute", "disk_cache")
        }
    report = {
        "scope": "External benchmark-only cache prototype with cheap analytic LL; no neural/full-hour caching decision or production cache API.",
        "configuration": vars(args) | {"output": str(args.output)},
        "preflight_bytes": preflight,
        "hardware": platform.platform(),
        "devices": [str(device) for device in devices],
        "jax_version": jax.__version__,
        "jax_enable_x64": jax.config.x64_enabled,
        "global_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "runtime_aggregate_sha256": source_before,
        "checkpoint_engine_sha256": engine_before,
        "script_sha256": script_before,
        "source_engine_script_stable": True,
        "strict_engine_likelihood_sha_guard": "unchanged and active on every replay chunk",
        "cache_policy": "Fresh directory every invocation; forward saves one .npy chunk; backward reads it. No all-LL RAM allocation. No fsync/cache dropping; readback of just-written files can use filesystem cache.",
        "cold_warm_scope": "First invocation per path in one shared process; the first recompute includes JIT compilation and disk-cache first invocation shares warmed kernels. Five alternating paired repetitions follow; these are warm-engine timings, not cold-disk timings.",
        "memory_scope": "Host process RSS and total device-wide NVML sampled20ms; excludes filesystem cache from RSS. JAX allocator stats are cumulative process-lifetime counters shared across modes.",
        "timing_scope": "prediction_seconds includes engine and callback disk I/O, excludes prototype parity checks/directory cleanup; separate end_to_end_seconds includes those.",
        "bitwise_within_each_checkpoint_size": True,
        "cross_checkpoint_size_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "medians_seconds": medians,
        "records": records,
        "process_highwater_rss_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "medians_seconds": medians,
                "source": source_before,
                "script": script_before,
            }
        )
    )


if __name__ == "__main__":
    main()
