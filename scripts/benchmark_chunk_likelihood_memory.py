"""Measure how one chunk's likelihood scales with chunk rows and selected spikes.

The chunk-boundary fix (``row_slice``) hands every backend the FULL decoding
``time`` but asks for only a contiguous range of rows. The claim this script
tests is narrow and mechanical:

    likelihood output and density-evaluation workspace for ONE requested chunk
    scale with chunk rows and selected spikes, rather than the full recording.

This is not a constant-runtime guarantee: spike-order validation still scans
each unit's spike times on every call. For repeated, synchronized runtime
measurements, use ``benchmark_chunk_likelihood_runtime.py``.

It is *not* a claim about total HMM memory: the driver still holds the full
``time`` array, the full per-unit spike-time (and waveform-feature) arrays and
the posterior it is building. Those are recording-sized by construction and are
reported here **separately** so the two are not conflated.

Two sweeps
----------
A. ``--sweep duration``: the fitted model and the requested chunk length are
   held fixed while the recording duration grows 16x. Density workspace should
   stay flat; recording-sized inputs and spike-order checks are separate costs.
B. ``--sweep chunk``: the recording duration is held at its largest while the
   requested chunk length grows 16x. Density workspace should grow with it.

Decoding spikes are placed on a regular grid at a fixed rate, so a chunk of a
given length selects exactly the same number of spikes at every duration: the
shapes fed to the jitted kernels are identical across a duration sweep and no
recompilation confounds the comparison.

Measurements
------------
* ``tracemalloc`` peak around the single chunk call (host / NumPy allocations).
* Peak process RSS around the same call, sampled by a background thread at 1 ms
  (this is what catches XLA's CPU buffers, which ``tracemalloc`` cannot see).
* ``jax.live_arrays()`` byte total before/after (buffers that survive the call).
* ``jax.jit(...).lower(...).compile().memory_analysis()`` for the one jitted
  clusterless kernel reachable with concrete arguments
  (``estimate_log_joint_mark_intensity``), as a function of the selected
  decoding-spike count.

Usage
-----
    uv run python scripts/benchmark_chunk_likelihood_memory.py --sweep all
    uv run python scripts/benchmark_chunk_likelihood_memory.py --quick
    uv run python scripts/benchmark_chunk_likelihood_memory.py \
        --backends clusterless_kde sorted_spikes_kde --json out.json

Limitations
-----------
* CPU backend only unless a GPU is present; ``memory_stats()`` is ``None`` on
  CPU, so device-side peaks come from the compiled-module analysis and from RSS,
  not from an allocator counter.
* Scaled experiments only (small arena, minutes of simulated data). The large
  production allocation is NOT attempted; extrapolate from the reported
  per-chunk bytes-per-row and bytes-per-spike slopes and treat that
  extrapolation as linear-fit evidence, not a measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import threading
import time as timer
import tracemalloc
from dataclasses import asdict, dataclass, field


def _forced_device_count() -> int:
    """Honour ``--devices N`` before JAX is imported.

    ``XLA_FLAGS`` is read when jaxlib initializes its platforms, so the flag has
    to be in the environment before ``import jax`` -- argparse runs far too late.
    """
    count = 1
    argv = sys.argv[1:]
    for index, token in enumerate(argv):
        if token == "--devices" and index + 1 < len(argv):
            count = int(argv[index + 1])
        elif token.startswith("--devices="):
            count = int(token.split("=", 1)[1])
    if count > 1:
        os.environ["XLA_FLAGS"] = (
            f"{os.environ.get('XLA_FLAGS', '')} "
            f"--xla_force_host_platform_device_count={count}"
        ).strip()
    return count


N_FORCED_DEVICES = _forced_device_count()

# Imported after the XLA flag above is in place, so --devices takes effect.
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

from non_local_detector.environment import Environment  # noqa: E402
from non_local_detector.likelihoods import (  # noqa: E402
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.likelihoods.common import (  # noqa: E402
    resolve_row_slice,
    select_spikes_in_rows,
)

DEFAULT_BACKENDS = [
    "sorted_spikes_kde",
    "sorted_spikes_glm",
    "clusterless_kde",
    "clusterless_kde_log",
    "clusterless_gmm",
    "clusterless_diffusion",
]
# ``clusterless_gmm``'s defaults (64 joint / 32 GPI / 32 occupancy components)
# over-parameterize this deliberately small synthetic fixture and collapse a
# component's covariance. The reduced mixture below fits cleanly and changes
# nothing the benchmark measures: the row-aware call still evaluates the fitted
# mixture at the SELECTED decoding spikes only, which is the scaling under test.
BACKEND_FIT_PARAMS: dict[str, dict] = {
    "clusterless_gmm": {
        "gmm_components_joint": 16,
        "gmm_components_gpi": 8,
        "gmm_components_occupancy": 8,
        "gmm_reg_covar": 1e-4,
    }
}
SAMPLING_FREQUENCY = 100.0  # Hz, decoding time bins
DECODE_RATE_HZ = 10.0  # per unit, regular grid so chunk spike counts are exact
N_UNITS = 8
N_FEATURES = 4
TRACK_LENGTH_CM = 100.0
PLACE_BIN_SIZE_CM = 2.0  # -> 50 position bins ("small arena")
N_ENCODING_SPIKES = 1200  # per unit; enough samples for the GMM joint mixture
ENCODING_DURATION_S = 120.0

DURATIONS_S = (60.0, 120.0, 240.0, 480.0, 960.0)  # 16x span
QUICK_DURATIONS_S = (60.0, 240.0, 960.0)
CHUNK_ROWS = (125, 250, 500, 1000, 2000)  # 16x span
QUICK_CHUNK_ROWS = (125, 500, 2000)
REFERENCE_CHUNK_ROWS = 500

# Spike sweep: total decoding spikes x64 while the REQUEST is held fixed (same
# rows, same selected spikes). Extra spikes land only outside the requested
# rows, so anything that grows here is paid per chunk for spikes the call never
# evaluates -- i.e. a full-array conversion that precedes the selection.
SPIKE_SWEEP_EXTRA_PER_UNIT = (250, 1_000, 4_000, 16_000)
QUICK_SPIKE_SWEEP_EXTRA_PER_UNIT = (250, 16_000)
SPIKE_SWEEP_DURATION_S = 960.0
SPIKE_SWEEP_ROWS = 500


# =============================================================================
# Measurement helpers
# =============================================================================


class RSSPeakSampler:
    """Sample this process's RSS in a background thread to catch transient peaks.

    ``tracemalloc`` only sees allocations made through Python's allocator, so it
    is blind to the buffers XLA's CPU backend allocates. Polling RSS is coarse
    (it also sees the allocator's own high-water behaviour) but it is the only
    measurement here that covers both.
    """

    def __init__(self, interval_s: float = 0.001) -> None:
        self.interval_s = interval_s
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_bytes = 0
        self.peak_bytes = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> RSSPeakSampler:
        gc.collect()
        self.baseline_bytes = self._process.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

    @property
    def peak_increase_bytes(self) -> int:
        return max(self.peak_bytes - self.baseline_bytes, 0)


def live_array_bytes() -> int:
    """Total bytes of JAX buffers currently alive on device."""
    return int(sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in jax.live_arrays()))


def array_bytes(obj, _seen: set[int] | None = None) -> int:
    """Recursively total the bytes of every array reachable from ``obj``.

    Used to size a fitted encoding model (place fields, encoding spike features,
    kernels, KDE sample sets) without knowing each backend's field names.
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, jnp.ndarray):
        return int(np.prod(obj.shape)) * obj.dtype.itemsize
    if isinstance(obj, dict):
        return sum(array_bytes(v, _seen) for v in obj.values())
    if isinstance(obj, list | tuple | set):
        return sum(array_bytes(v, _seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return sum(array_bytes(v, _seen) for v in vars(obj).values())
    return 0


def fmt_bytes(n: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} TiB"


# =============================================================================
# Synthetic recording
# =============================================================================


@dataclass
class Recording:
    """One synthetic decoding recording (encoding data is generated separately)."""

    duration_s: float
    time: np.ndarray
    position_time: np.ndarray
    position: np.ndarray
    spike_times: list[np.ndarray]
    spike_waveform_features: list[np.ndarray]

    @property
    def n_time(self) -> int:
        return len(self.time)

    @property
    def n_spikes(self) -> int:
        return int(sum(len(s) for s in self.spike_times))

    def input_bytes(self) -> dict[str, int]:
        """Recording-sized input and index metadata held by the caller."""
        return {
            "time": int(self.time.nbytes),
            "position_time": int(self.position_time.nbytes),
            "position": int(self.position.nbytes),
            "spike_times": int(sum(s.nbytes for s in self.spike_times)),
            "spike_waveform_features": int(
                sum(f.nbytes for f in self.spike_waveform_features)
            ),
            "is_missing": self.n_time,  # bool array, if supplied
        }


N_MARK_CLUSTERS = 3


def make_marks(n_spikes: int, rng: np.random.Generator) -> np.ndarray:
    """Waveform features drawn from a few well-separated Gaussian clusters."""
    if n_spikes == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    centers = np.array(
        [[20.0, 40.0, 60.0, 80.0], [60.0, 20.0, 80.0, 40.0], [80.0, 80.0, 20.0, 20.0]]
    )[:N_MARK_CLUSTERS]
    assignment = rng.integers(0, len(centers), n_spikes)
    return (
        centers[assignment] + rng.standard_normal((n_spikes, N_FEATURES)) * 5.0
    ).astype(np.float32)


def make_position(duration_s: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A back-and-forth run on a 1-D track, sampled at the decoding rate.

    Seeded tracking jitter is added because the noiseless trajectory is exactly
    periodic: at 100 Hz with a 20 s round trip the position values repeat
    bit-for-bit, and ``clusterless_gmm``'s occupancy mixture then hits a singular
    covariance and refuses to fit. The jitter is small compared with the 2 cm
    place-bin size, so it does not change what the benchmark measures.
    """
    rng = np.random.default_rng(seed + 977)
    n_samples = int(round(duration_s * SAMPLING_FREQUENCY)) + 1
    position_time = np.linspace(0.0, duration_s, n_samples)
    # 20 s round trip, independent of duration, so the encoding model is
    # representative at every duration.
    phase = 2 * np.pi * position_time / 20.0
    position = 0.5 * TRACK_LENGTH_CM * (1.0 - np.cos(phase))
    position = position + rng.normal(0.0, 0.25, n_samples)
    position = np.clip(position, 0.0, TRACK_LENGTH_CM)[:, None]
    return position_time, position


def make_decoding_spikes(
    duration_s: float, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Regularly spaced decoding spikes: a fixed row range selects a fixed count.

    A Poisson train would make the selected-spike count (and therefore the
    jitted kernels' shapes) vary between durations, which would confound the
    duration sweep with recompilation.
    """
    spacing = 1.0 / DECODE_RATE_HZ
    spike_times = []
    features = []
    for unit in range(N_UNITS):
        offset = spacing * unit / N_UNITS
        unit_times = np.arange(offset, duration_s, spacing)
        spike_times.append(unit_times)
        features.append(make_marks(len(unit_times), rng))
    return spike_times, features


def make_recording(duration_s: float, seed: int = 0) -> Recording:
    """A decoding recording of the requested duration."""
    rng = np.random.default_rng(seed)
    position_time, position = make_position(duration_s, seed)
    spike_times, features = make_decoding_spikes(duration_s, rng)
    n_time = int(round(duration_s * SAMPLING_FREQUENCY)) + 1
    return Recording(
        duration_s=duration_s,
        time=np.linspace(0.0, duration_s, n_time),
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=features,
    )


def make_spike_sweep_recording(
    n_extra_per_unit: int, duration_s: float = SPIKE_SWEEP_DURATION_S, seed: int = 0
) -> Recording:
    """A recording whose FIRST ``SPIKE_SWEEP_ROWS`` rows hold a fixed spike set.

    Every extra spike is placed after those rows, so the row request at the
    start of the timeline selects the same spikes at every sweep point while the
    recording's total grows. Anything that then grows with the total is being
    paid for spikes the call never evaluates.
    """
    rng = np.random.default_rng(seed)
    position_time, position = make_position(duration_s, seed)
    n_time = int(round(duration_s * SAMPLING_FREQUENCY)) + 1
    time = np.linspace(0.0, duration_s, n_time)

    requested_end = time[SPIKE_SWEEP_ROWS]
    spike_times = []
    features = []
    for unit in range(N_UNITS):
        # Fixed content inside the request: one spike per 10 rows, offset per unit.
        inside = time[5 + unit : SPIKE_SWEEP_ROWS - 1 : 10] + 1e-4
        outside = np.sort(rng.uniform(requested_end, time[-1], n_extra_per_unit))
        unit_times = np.concatenate([inside, outside])
        spike_times.append(unit_times)
        # float64 features on purpose: numpy's default, what most feature
        # pipelines hand in -- and the case where converting the array actually
        # costs something. On the CPU backend ``jnp.asarray`` of a float32 array
        # is zero-copy, so a full-array conversion of float32 features allocates
        # nothing measurable and would hide the scaling this sweep is for.
        features.append(make_marks(len(unit_times), rng).astype(np.float64))

    return Recording(
        duration_s=duration_s,
        time=time,
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        spike_waveform_features=features,
    )


def as_device_inputs(recording: Recording, n_devices: int) -> Recording:
    """Return ``recording`` with decoding spikes/features as ``jax.Array``.

    With more than one device the arrays are sharded over their leading (spike)
    axis under an ``Auto``-typed mesh, which is the configuration a user hits
    when they hand in device arrays: ``np.asarray`` of such an array gathers it
    AND caches the host copy on the array object, so a per-chunk full-array
    conversion is retained for the lifetime of the input.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = None
    if n_devices > 1:
        mesh = jax.make_mesh(
            (n_devices,), ("spike",), axis_types=(jax.sharding.AxisType.Auto,)
        )

    def convert(array: np.ndarray) -> jnp.ndarray:
        if mesh is None:
            return jnp.asarray(array)
        # An even sharding needs the spike axis divisible by the device count.
        # Repeat the LAST spike as padding: it sits at the end of the recording,
        # far outside the requested rows, so the selection is untouched and the
        # times stay sorted.
        remainder = array.shape[0] % n_devices
        if remainder:
            pad = np.repeat(array[-1:], n_devices - remainder, axis=0)
            array = np.concatenate([array, pad], axis=0)
        device_array = jnp.asarray(array)
        spec = P("spike") if device_array.ndim == 1 else P("spike", None)
        return jax.device_put(device_array, NamedSharding(mesh, spec))

    return Recording(
        duration_s=recording.duration_s,
        time=recording.time,
        position_time=recording.position_time,
        position=recording.position,
        spike_times=[convert(t) for t in recording.spike_times],
        spike_waveform_features=[convert(f) for f in recording.spike_waveform_features],
    )


def host_value_cached(recording: Recording) -> bool:
    """Whether any decoding FEATURE array has had its host value materialized.

    ``jax.Array._npy_value`` is set the first time anything calls ``np.asarray``
    on the array and kept for its lifetime, so it is an exact proxy for "a chunk
    call copied the whole recording to the host". Spike times are exempt:
    ``select_spikes_in_rows`` reads all of them by construction.
    """
    return any(
        getattr(features, "_npy_value", None) is not None
        for features in recording.spike_waveform_features
    )


def make_encoding_data(seed: int = 1):
    """Fit-time data: fixed for every sweep point so the model never changes."""
    rng = np.random.default_rng(seed)
    position_time, position = make_position(ENCODING_DURATION_S, seed)
    environment = Environment(
        environment_name="line",
        place_bin_size=PLACE_BIN_SIZE_CM,
        position_range=((0.0, TRACK_LENGTH_CM),),
    ).fit_place_grid(position=position, infer_track_interior=False)

    spike_times = [
        np.sort(rng.uniform(0.0, ENCODING_DURATION_S, N_ENCODING_SPIKES))
        for _ in range(N_UNITS)
    ]
    # Cluster-structured marks (a few "units" per electrode), as a real tetrode
    # produces. A single isotropic blob leaves ``clusterless_gmm``'s 64-component
    # joint mixture with collapsed components and it refuses to fit.
    features = [make_marks(N_ENCODING_SPIKES, rng) for _ in range(N_UNITS)]
    return position_time, position, environment, spike_times, features


# =============================================================================
# Backend harness
# =============================================================================


@dataclass
class FittedBackend:
    """A fitted backend plus everything needed to call it on one chunk."""

    name: str
    predict_func: object
    encoding_model: dict
    is_clusterless: bool
    environment: Environment
    encoding_bytes: int = 0
    n_position_bins: int = 0

    def call(
        self, recording: Recording, row_slice: slice | None, is_local: bool = False
    ):
        args = [recording.position_time, recording.position, recording.spike_times]
        if self.is_clusterless:
            args.append(recording.spike_waveform_features)
        return self.predict_func(
            recording.time,
            *args,
            **self.encoding_model,
            is_local=is_local,
            row_slice=row_slice,
        )


def fit_backend(name: str, encoding) -> FittedBackend:
    """Fit one registered backend on the shared encoding data."""
    position_time, position, environment, spike_times, features = encoding

    if name in _SORTED_SPIKES_ALGORITHMS:
        fit_func, predict_func = _SORTED_SPIKES_ALGORITHMS[name]
        geometry = {}
        if name == "sorted_spikes_glm":
            geometry = {
                "place_bin_edges": environment.place_bin_edges_,
                "edges": environment.edges_,
                "is_track_interior": environment.is_track_interior_,
                "is_track_boundary": environment.is_track_boundary_,
            }
        encoding_model = fit_func(
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            environment=environment,
            **geometry,
            **BACKEND_FIT_PARAMS.get(name, {}),
        )
        is_clusterless = False
    elif name in _CLUSTERLESS_ALGORITHMS:
        fit_func, predict_func = _CLUSTERLESS_ALGORITHMS[name]
        encoding_model = fit_func(
            position_time=position_time,
            position=position,
            spike_times=spike_times,
            spike_waveform_features=features,
            environment=environment,
            **BACKEND_FIT_PARAMS.get(name, {}),
        )
        is_clusterless = True
    else:
        raise SystemExit(f"unknown backend {name!r}")

    return FittedBackend(
        name=name,
        predict_func=predict_func,
        encoding_model=encoding_model,
        is_clusterless=is_clusterless,
        environment=environment,
        encoding_bytes=array_bytes(encoding_model),
        n_position_bins=int(environment.is_track_interior_.sum()),
    )


def selection_sizes(
    backend: FittedBackend, recording: Recording, row_slice: slice
) -> dict[str, int]:
    """Recompute, from the public helper, what the call selects for this chunk.

    This is the allocation/shape check: the same helper the backends use decides
    which spikes a row range owns, so calling it here reports exactly the sizes
    the backend will feed to its kernels -- without editing production code.
    """
    row_start, row_stop = resolve_row_slice(row_slice, recording.n_time)
    n_rows = row_stop - row_start
    n_selected = 0
    feature_bytes = 0
    for unit_times, unit_features in zip(
        recording.spike_times, recording.spike_waveform_features, strict=True
    ):
        indexer, bin_ind = select_spikes_in_rows(
            unit_times, recording.time, row_start, row_stop
        )
        assert len(bin_ind) == len(unit_times[indexer])
        assert len(bin_ind) == 0 or (bin_ind.min() >= 0 and bin_ind.max() < n_rows)
        n_selected += len(bin_ind)
        feature_bytes += int(unit_features[indexer].nbytes)
    return {
        "n_rows": n_rows,
        "num_segments": n_rows,
        "n_selected_spikes": n_selected,
        "selected_feature_bytes": feature_bytes,
    }


@dataclass
class ChunkMeasurement:
    """One measured chunk call."""

    backend: str
    is_local: bool
    duration_s: float
    n_time: int
    n_total_spikes: int
    n_rows: int
    num_segments: int
    n_selected_spikes: int
    selected_feature_bytes: int
    output_shape: tuple[int, ...]
    output_bytes: int
    full_time_output_bytes: int
    tracemalloc_peak_bytes: int
    rss_peak_increase_bytes: int
    live_array_delta_bytes: int
    encoding_bytes: int
    recording_input_bytes: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def measure_chunk(
    backend: FittedBackend,
    recording: Recording,
    n_rows: int,
    row_start: int | None = None,
    is_local: bool = False,
) -> ChunkMeasurement:
    """Measure one row-aware call for one chunk of the recording.

    The chunk sits in the middle unless ``row_start`` says otherwise (the spike
    sweep pins it to the start, where its selected spikes are held fixed).
    """
    if row_start is None:
        row_start = (recording.n_time - n_rows) // 2
    row_slice = slice(row_start, row_start + n_rows)
    sizes = selection_sizes(backend, recording, row_slice)

    # Warm up: compile for these shapes and let any one-off caches fill, so the
    # measured call is steady-state rather than a compilation.
    warm = backend.call(recording, row_slice, is_local=is_local)
    warm = np.asarray(jax.block_until_ready(warm))
    del warm
    gc.collect()

    live_before = live_array_bytes()
    tracemalloc.start()
    with RSSPeakSampler() as rss:
        start = timer.perf_counter()
        result = jax.block_until_ready(
            backend.call(recording, row_slice, is_local=is_local)
        )
        elapsed = timer.perf_counter() - start
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ``np.asarray`` on a CPU JAX array is zero-copy, so ``output`` keeps the
    # device buffer alive after ``del result``. The live-array delta below is
    # therefore the returned chunk (not zero) -- which is exactly the quantity
    # of interest: the buffer this call leaves behind for its caller.
    output = np.asarray(result)
    assert output.shape[0] == n_rows, (output.shape, n_rows)
    del result
    gc.collect()
    live_after = live_array_bytes()

    n_bins = output.shape[1]
    return ChunkMeasurement(
        backend=backend.name,
        is_local=is_local,
        duration_s=recording.duration_s,
        n_time=recording.n_time,
        n_total_spikes=recording.n_spikes,
        n_rows=sizes["n_rows"],
        num_segments=sizes["num_segments"],
        n_selected_spikes=sizes["n_selected_spikes"],
        selected_feature_bytes=sizes["selected_feature_bytes"],
        output_shape=tuple(int(s) for s in output.shape),
        output_bytes=int(output.nbytes),
        full_time_output_bytes=int(recording.n_time * n_bins * output.dtype.itemsize),
        tracemalloc_peak_bytes=int(tm_peak),
        rss_peak_increase_bytes=int(rss.peak_increase_bytes),
        live_array_delta_bytes=int(live_after - live_before),
        encoding_bytes=backend.encoding_bytes,
        recording_input_bytes=recording.input_bytes(),
        seconds=elapsed,
    )


# =============================================================================
# XLA compiled-module memory analysis
# =============================================================================


def xla_memory_analysis(n_decoding_spikes_list, n_encoding_spikes, n_position_bins):
    """``memory_analysis()`` of the jitted clusterless mark-intensity kernel.

    ``estimate_log_joint_mark_intensity`` is the one jitted kernel in the
    likelihood path reachable with concrete arguments; its leading dimension is
    the number of decoding spikes the row request SELECTED, so its temp/argument
    sizes are the device-side statement of the same scaling claim.
    """
    from non_local_detector.likelihoods.clusterless_kde_log import (
        estimate_log_joint_mark_intensity,
    )

    rng = np.random.default_rng(5)
    encoding_features = jnp.asarray(
        rng.standard_normal((n_encoding_spikes, N_FEATURES)), dtype=jnp.float32
    )
    waveform_stds = jnp.full((N_FEATURES,), 24.0, dtype=jnp.float32)
    occupancy = jnp.full((n_position_bins,), 0.5, dtype=jnp.float32)
    log_position_distance = jnp.asarray(
        rng.standard_normal((n_encoding_spikes, n_position_bins)), dtype=jnp.float32
    )

    rows = []
    for n_decoding_spikes in n_decoding_spikes_list:
        decoding_features = jnp.asarray(
            rng.standard_normal((n_decoding_spikes, N_FEATURES)), dtype=jnp.float32
        )
        compiled = estimate_log_joint_mark_intensity.lower(
            decoding_features,
            encoding_features,
            waveform_stds,
            occupancy,
            1.0,
            log_position_distance,
        ).compile()
        analysis = compiled.memory_analysis()
        rows.append(
            {
                "n_decoding_spikes": int(n_decoding_spikes),
                "n_encoding_spikes": int(n_encoding_spikes),
                "n_position_bins": int(n_position_bins),
                "argument_size_bytes": int(analysis.argument_size_in_bytes),
                "output_size_bytes": int(analysis.output_size_in_bytes),
                "temp_size_bytes": int(analysis.temp_size_in_bytes),
                "generated_code_size_bytes": int(analysis.generated_code_size_in_bytes),
            }
        )
    return rows


# =============================================================================
# Reporting
# =============================================================================


def print_table(title: str, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    """Print a markdown table."""
    print(f"\n### {title}\n")
    print("| " + " | ".join(header for header, _ in columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[key]) for _, key in columns) + " |")


def duration_sweep_rows(measurements: list[ChunkMeasurement]) -> list[dict]:
    return [
        {
            "backend": m.backend,
            "duration": f"{m.duration_s:.0f} s",
            "n_time": f"{m.n_time:,}",
            "total_spikes": f"{m.n_total_spikes:,}",
            "rows": m.n_rows,
            "num_segments": m.num_segments,
            "selected": m.n_selected_spikes,
            "feat_sel": fmt_bytes(m.selected_feature_bytes),
            "out": f"{m.output_shape} = {fmt_bytes(m.output_bytes)}",
            "full_out": fmt_bytes(m.full_time_output_bytes),
            "tm_peak": fmt_bytes(m.tracemalloc_peak_bytes),
            "rss_peak": fmt_bytes(m.rss_peak_increase_bytes),
            "live_delta": fmt_bytes(m.live_array_delta_bytes),
            "secs": f"{m.seconds:.3f}",
        }
        for m in measurements
    ]


DURATION_COLUMNS = [
    ("backend", "backend"),
    ("duration", "duration"),
    ("n_time", "n_time"),
    ("total spikes", "total_spikes"),
    ("chunk rows", "rows"),
    ("num_segments", "num_segments"),
    ("selected spikes", "selected"),
    ("selected features", "feat_sel"),
    ("output", "out"),
    ("T x N would be", "full_out"),
    ("tracemalloc peak", "tm_peak"),
    ("RSS peak delta", "rss_peak"),
    ("live-array delta", "live_delta"),
    ("s", "secs"),
]


def spike_sweep_rows(measurements: list[ChunkMeasurement]) -> list[dict]:
    return [
        {
            "backend": m.backend,
            "is_local": str(m.is_local),
            "total_spikes": f"{m.n_total_spikes:,}",
            "rows": m.n_rows,
            "selected": m.n_selected_spikes,
            "feat_sel": fmt_bytes(m.selected_feature_bytes),
            "feat_all": fmt_bytes(m.recording_input_bytes["spike_waveform_features"]),
            "out": f"{m.output_shape} = {fmt_bytes(m.output_bytes)}",
            "tm_peak": fmt_bytes(m.tracemalloc_peak_bytes),
            "live_delta": fmt_bytes(m.live_array_delta_bytes),
            "secs": f"{m.seconds:.3f}",
        }
        for m in measurements
    ]


SPIKE_COLUMNS = [
    ("backend", "backend"),
    ("is_local", "is_local"),
    ("total spikes", "total_spikes"),
    ("chunk rows", "rows"),
    ("selected spikes", "selected"),
    ("selected features", "feat_sel"),
    ("all features (input)", "feat_all"),
    ("output", "out"),
    ("tracemalloc peak", "tm_peak"),
    ("live-array delta", "live_delta"),
    ("s", "secs"),
]


def report_memory_budget(backends: list[FittedBackend], recording: Recording) -> None:
    """The three budgets the brief asks to keep apart."""
    print("\n### Memory budget, kept separate\n")
    print("| category | what it holds | bytes |")
    print("| --- | --- | --- |")
    for backend in backends:
        print(
            f"| resident encoding (fit-time) | {backend.name}: place fields, encoding "
            f"spike features, kernels | {fmt_bytes(backend.encoding_bytes)} |"
        )
    for name, value in recording.input_bytes().items():
        print(
            f"| recording-sized input/index | {name} "
            f"(duration {recording.duration_s:.0f} s) | {fmt_bytes(value)} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument(
        "--sweep",
        choices=("duration", "chunk", "spikes", "xla", "all"),
        default="all",
    )
    parser.add_argument(
        "--quick", action="store_true", help="fewer sweep points (smoke test)"
    )
    parser.add_argument(
        "--features-as",
        choices=("numpy", "jax"),
        default="numpy",
        help="type of the decoding spike/feature arrays handed to the backends",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help=(
            "force this many CPU devices and shard jax inputs over them "
            "(applies to --features-as jax; read before jax is imported)"
        ),
    )
    parser.add_argument("--json", type=str, default=None, help="write raw results here")
    args = parser.parse_args(argv)

    durations = QUICK_DURATIONS_S if args.quick else DURATIONS_S
    chunk_rows = QUICK_CHUNK_ROWS if args.quick else CHUNK_ROWS

    print("## Environment\n")
    print(
        f"- jax {jax.__version__}, backend `{jax.default_backend()}`, "
        f"devices {jax.devices()}"
    )
    print(f"- numpy {np.__version__}, python {sys.version.split()[0]}")
    print(f"- platform {platform.platform()}")
    print(
        f"- decoding bins at {SAMPLING_FREQUENCY:.0f} Hz, {N_UNITS} units at "
        f"{DECODE_RATE_HZ:.0f} Hz (regular grid), {N_FEATURES} waveform features"
    )

    encoding = make_encoding_data()
    backends = [fit_backend(name, encoding) for name in args.backends]
    print(
        f"- environment: {TRACK_LENGTH_CM:.0f} cm track, "
        f"{backends[0].n_position_bins} interior position bins"
    )

    results: dict[str, list] = {}

    if args.sweep in ("duration", "all"):
        recordings = {d: make_recording(d) for d in durations}
        measurements = [
            measure_chunk(backend, recordings[duration], REFERENCE_CHUNK_ROWS)
            for backend in backends
            for duration in durations
        ]
        results["duration_sweep"] = [asdict(m) for m in measurements]
        print_table(
            f"A. Recording duration x{durations[-1] / durations[0]:.0f}, "
            f"chunk fixed at {REFERENCE_CHUNK_ROWS} rows",
            duration_sweep_rows(measurements),
            DURATION_COLUMNS,
        )
        report_memory_budget(backends, recordings[durations[-1]])

    if args.sweep in ("chunk", "all"):
        recording = make_recording(durations[-1])
        measurements = [
            measure_chunk(backend, recording, n_rows)
            for backend in backends
            for n_rows in chunk_rows
        ]
        results["chunk_sweep"] = [asdict(m) for m in measurements]
        print_table(
            f"B. Chunk rows x{chunk_rows[-1] / chunk_rows[0]:.0f}, "
            f"recording fixed at {durations[-1]:.0f} s",
            duration_sweep_rows(measurements),
            DURATION_COLUMNS,
        )

    if args.sweep in ("spikes", "all"):
        extras = (
            QUICK_SPIKE_SWEEP_EXTRA_PER_UNIT
            if args.quick
            else SPIKE_SWEEP_EXTRA_PER_UNIT
        )
        recordings = {n: make_spike_sweep_recording(n) for n in extras}
        measurements = []
        selected_counts = set()
        cached_after = {}
        for backend in backends:
            for is_local in (False, True):
                for n_extra in extras:
                    recording = recordings[n_extra]
                    if args.features_as == "jax":
                        # Fresh device arrays per measurement: the cached host
                        # value is per-array and per-lifetime.
                        recording = as_device_inputs(recording, args.devices)
                    measurement = measure_chunk(
                        backend,
                        recording,
                        SPIKE_SWEEP_ROWS,
                        row_start=0,
                        is_local=is_local,
                    )
                    if args.features_as == "jax":
                        cached_after[(backend.name, is_local, n_extra)] = (
                            host_value_cached(recording)
                        )
                    selected_counts.add(measurement.n_selected_spikes)
                    measurements.append(measurement)
        # The premise of this sweep: the request selects the same spikes at
        # every point, so only the recording's total changed.
        assert len(selected_counts) == 1, selected_counts
        results["spike_sweep"] = [asdict(m) for m in measurements]
        print_table(
            f"D. Total decoding spikes x{extras[-1] / extras[0]:.0f}, request fixed "
            f"at rows [0:{SPIKE_SWEEP_ROWS}] ({selected_counts.pop()} selected "
            f"spikes), inputs as {args.features_as} on {args.devices} device(s)",
            spike_sweep_rows(measurements),
            SPIKE_COLUMNS,
        )
        if cached_after:
            results["host_value_cached"] = {
                "|".join(map(str, key)): value for key, value in cached_after.items()
            }
            print(
                "\nDecoding-feature host value cached after the call "
                "(True = the whole recording was materialized on the host and "
                "retained on the input array):\n"
            )
            print("| backend | is_local | extra spikes/unit | features_host_cached |")
            print("| --- | --- | --- | --- |")
            for (name, is_local, n_extra), value in cached_after.items():
                print(f"| {name} | {is_local} | {n_extra:,} | {value} |")

    if args.sweep in ("xla", "all"):
        rows = xla_memory_analysis(
            [125, 250, 500, 1000, 2000] if not args.quick else [125, 500, 2000],
            N_ENCODING_SPIKES,
            backends[0].n_position_bins,
        )
        results["xla_memory_analysis"] = rows
        print_table(
            "C. XLA `memory_analysis()` of `estimate_log_joint_mark_intensity`",
            [
                {
                    "dec": r["n_decoding_spikes"],
                    "enc": r["n_encoding_spikes"],
                    "pos": r["n_position_bins"],
                    "arg": fmt_bytes(r["argument_size_bytes"]),
                    "out": fmt_bytes(r["output_size_bytes"]),
                    "temp": fmt_bytes(r["temp_size_bytes"]),
                    "code": fmt_bytes(r["generated_code_size_bytes"]),
                }
                for r in rows
            ],
            [
                ("selected decoding spikes", "dec"),
                ("encoding spikes", "enc"),
                ("position bins", "pos"),
                ("argument", "arg"),
                ("output", "out"),
                ("temp (XLA workspace)", "temp"),
                ("code", "code"),
            ],
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nRaw results written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
