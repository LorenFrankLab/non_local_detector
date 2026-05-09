"""Direct ``zarr.Array`` data source — bypasses xarray on the hot path.

The canonical ``InMemoryDecoderDataSource`` reads window slices through
``xr.DataArray.isel(time=sl).values``: xarray indexing → dask graph
materialization → chunk fetch. For long sessions backed by a
``results.zarr/`` cache, that indirection dominates per-window latency.
``ZarrDirectDecoderDataSource`` opens the cache once via
``zarr.open_consolidated``, holds raw ``zarr.Array`` handles for the
five hot-path arrays (``acausal_posterior``, ``log_likelihood``,
``predictive_posterior``, ``acausal_state_probabilities``, ``time``),
and reads ``arr[sl, :]`` directly — one contiguous chunk fetch, no
xarray indexing.

Sidecars (model, spikes, position, place fields, fitted detector,
event index) load eagerly at construction because they're small and
sit on the per-tick hot path. The ``active_run.results`` xarray
Dataset is the same lazy zarr-backed wrapper the in-memory path uses
for compatibility with code that walks ``results.data_vars`` (overlay
availability, missing-output gating).

Multi-run support via ``for_directories({name: bundle_dir})`` mirrors
``InMemoryDecoderDataSource.from_bundles``. Cross-bundle invariants
(time-grid alignment, overlay-schema alignment) are validated the
same way as the in-memory source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from non_local_detector.visualization.interactive.data_source import SliceWhich
from non_local_detector.visualization.interactive.data_source_zarr import (
    load_zarr_cache_or_fall_back,
)
from non_local_detector.visualization.interactive.view_models.base import (
    RunBundle,
    SpikeEvent,
    SpikeEventIndex,
    bin_edges_array,
)
from non_local_detector.visualization.interactive.view_models.events import (
    find_duplicate_overlay_names,
)


@dataclass
class _ZarrRunHandles:
    """Per-run state: a ``RunBundle`` (sidecars + lazy results) plus
    direct ``zarr.Array`` handles for the hot-path arrays."""

    bundle: RunBundle
    posterior: object  # zarr.Array — required
    state_probabilities: object  # zarr.Array — required
    likelihood: object | None  # zarr.Array | None — optional
    predictive: object | None  # zarr.Array | None — optional
    time: np.ndarray
    time_edges: np.ndarray
    event_index: SpikeEventIndex


def _open_zarr_group(zarr_path: Path):
    """Open ``zarr_path`` as a ``zarr.Group`` (consolidated when possible).

    Falls back to ``zarr.open`` when the cache predates the
    consolidated-metadata format. ``mode='r'`` is read-only because the
    viewer never writes; concurrent viewers reading the same cache is
    fine but writers from another process would need a separate path.
    """
    import zarr  # local import — viewer-cache extra optional

    try:
        return zarr.open_consolidated(str(zarr_path), mode="r")
    except (KeyError, ValueError):
        return zarr.open(str(zarr_path), mode="r")


def _load_run_bundle_from_dir(
    bundle_dir: Path, run_name: str
) -> tuple[RunBundle, Path]:
    """Load one bundle (sidecars + lazy results via the zarr cache).

    Returns the eager ``RunBundle`` plus the absolute ``zarr_path`` so
    the caller can also open it directly for hot-path handles. Mirrors
    ``app._load_run`` but is centralised here so the zarr-direct source
    doesn't depend on the CLI module.
    """
    import pandas as pd

    from non_local_detector.models.base import _DetectorBase

    nc_path = bundle_dir / "results.nc"
    zarr_path = bundle_dir / "results.zarr"
    if not zarr_path.is_dir():
        raise FileNotFoundError(
            f"ZarrDirectDecoderDataSource needs a results.zarr/ at {zarr_path!s}; "
            "build the cache with `python -m "
            "non_local_detector.visualization.interactive.devtools "
            "build-viewer-cache --run-dir <dir>` or use "
            "InMemoryDecoderDataSource for a NetCDF-only bundle."
        )
    if not nc_path.exists():
        raise FileNotFoundError(
            f"ZarrDirectDecoderDataSource needs results.nc next to "
            f"results.zarr/ at {bundle_dir!s} so cross-cache invariants "
            "can be validated."
        )
    # Validate the cache against the canonical NetCDF (same staleness
    # check the eager path uses) before trusting either the lazy results
    # Dataset or the direct zarr handles below.
    canonical_results = _DetectorBase.load_results(str(nc_path))
    results = load_zarr_cache_or_fall_back(
        zarr_path=zarr_path,
        canonical_path=nc_path,
        canonical_results=canonical_results,
    )

    detector = _DetectorBase.load_model(str(bundle_dir / "model.pkl"))
    spike_times_npz = np.load(str(bundle_dir / "spikes.npz"), allow_pickle=True)
    spike_times = list(spike_times_npz["spike_times"])
    position_df = pd.read_parquet(str(bundle_dir / "position.parquet"))
    if "position" in position_df.columns:
        position = position_df["position"].to_numpy()
    elif {"x_position", "y_position"}.issubset(position_df.columns):
        position = position_df[["x_position", "y_position"]].to_numpy()
    else:
        raise ValueError(
            f"position.parquet at {bundle_dir!s} must contain a "
            "'position' column (1D) or 'x_position' + 'y_position' (2D)."
        )
    position_time = position_df.index.to_numpy()
    speed = position_df["speed"].to_numpy() if "speed" in position_df.columns else None
    bundle = RunBundle(
        results=results,
        detector=detector,
        spike_times=spike_times,
        position_time=position_time,
        position=position,
        speed=speed,
    )
    # Tag the bundle with the run name so multi-run callers can read it
    # back without threading an extra arg through the construction path.
    bundle.__dict__["_zarr_run_name"] = run_name
    return bundle, zarr_path


def _build_handles(bundle_dir: Path, run_name: str) -> _ZarrRunHandles:
    """Eagerly load sidecars + open the zarr group + cache handles."""
    bundle, zarr_path = _load_run_bundle_from_dir(bundle_dir, run_name)
    group = _open_zarr_group(zarr_path)

    # ``acausal_posterior`` and ``acausal_state_probabilities`` are
    # required at ``RunBundle.__post_init__`` so the cache validator
    # already enforced them; raise here too for symmetry with the
    # InMemoryDecoderDataSource invariant.
    if "acausal_posterior" not in group:
        raise KeyError(f"results.zarr at {zarr_path!s} is missing 'acausal_posterior'.")
    if "acausal_state_probabilities" not in group:
        raise KeyError(
            f"results.zarr at {zarr_path!s} is missing 'acausal_state_probabilities'."
        )

    time = np.asarray(group["time"][:])
    time_edges = bin_edges_array(time)
    event_index = SpikeEventIndex.from_spike_times(bundle.spike_times, time)

    return _ZarrRunHandles(
        bundle=bundle,
        posterior=group["acausal_posterior"],
        state_probabilities=group["acausal_state_probabilities"],
        likelihood=(group["log_likelihood"] if "log_likelihood" in group else None),
        predictive=(
            group["predictive_posterior"] if "predictive_posterior" in group else None
        ),
        time=time,
        time_edges=time_edges,
        event_index=event_index,
    )


_SLICE_VAR_MAP: dict[SliceWhich, str] = {
    "posterior": "acausal_posterior",
    "likelihood": "log_likelihood",
    "predictive": "predictive_posterior",
    "acausal": "acausal_posterior",
}


class ZarrDirectDecoderDataSource:
    """Direct ``zarr.Array`` data source. Implements ``DecoderDataSource``.

    Construction validates:

    1. All runs share the same time grid (``np.array_equal``). Mismatch
       raises immediately with the offending pair named.
    2. Overlay-schema set matches across runs (same as in-memory source).

    Hot-path reads (``load_posterior`` / ``load_likelihood`` / ...) go
    through cached ``zarr.Array`` handles via
    ``np.asarray(handle[sl, :])`` — one contiguous read per call.
    """

    def __init__(self, runs: dict[str, _ZarrRunHandles]):
        if not runs:
            raise ValueError("ZarrDirectDecoderDataSource requires at least one run.")
        self._runs = dict(runs)
        self._validate_time_grid_alignment()
        self._validate_overlay_alignment()
        self._active_run_name = next(iter(self._runs))
        self._position_cache: dict[str, np.ndarray | None] = {}

    @classmethod
    def for_directories(cls, dirs: dict[str, Path]) -> ZarrDirectDecoderDataSource:
        """Open every bundle in ``dirs`` directly through its zarr cache."""
        if not dirs:
            raise ValueError(
                "ZarrDirectDecoderDataSource.for_directories needs at "
                "least one (name, dir) pair."
            )
        runs = {name: _build_handles(Path(d), name) for name, d in dirs.items()}
        return cls(runs)

    @classmethod
    def for_directory(cls, name: str, bundle_dir: Path) -> ZarrDirectDecoderDataSource:
        """Single-run convenience."""
        return cls.for_directories({name: bundle_dir})

    # ------------------------------------------------------------------
    # Construction validators (mirror InMemoryDecoderDataSource)
    # ------------------------------------------------------------------

    def _validate_time_grid_alignment(self) -> None:
        if len(self._runs) <= 1:
            return
        names = list(self._runs)
        reference_name = names[0]
        reference_time = self._runs[reference_name].time
        for other in names[1:]:
            other_time = self._runs[other].time
            if other_time.shape != reference_time.shape or not np.array_equal(
                other_time, reference_time
            ):
                raise ValueError(
                    "ZarrDirectDecoderDataSource requires all runs to share "
                    f"the same time grid. Runs {reference_name!r} (length "
                    f"{reference_time.shape[0]}) and {other!r} (length "
                    f"{other_time.shape[0]}) disagree."
                )

    def _validate_overlay_alignment(self) -> None:
        for run_name, handles in self._runs.items():
            duplicates = find_duplicate_overlay_names(handles.bundle.event_overlays)
            if duplicates:
                raise ValueError(
                    f"Run {run_name!r} has duplicate overlay names: "
                    f"{duplicates}. Each overlay name must be unique within "
                    "a bundle."
                )
        if len(self._runs) <= 1:
            return
        schemas = {
            name: frozenset((ovl.name, ovl.kind) for ovl in h.bundle.event_overlays)
            for name, h in self._runs.items()
        }
        names = list(schemas)
        reference = schemas[names[0]]
        mismatched = {n: s for n, s in schemas.items() if s != reference}
        if mismatched:
            raise ValueError(
                "All RunBundles loaded together must declare the same "
                "overlay (name, kind) schema. Got: "
                f"{ {n: sorted(s) for n, s in schemas.items()}!r}."
            )

    # ------------------------------------------------------------------
    # Run state (DecoderDataSource Protocol surface)
    # ------------------------------------------------------------------

    @property
    def run_names(self) -> list[str]:
        return list(self._runs)

    @property
    def active_run_name(self) -> str:
        return self._active_run_name

    @property
    def active_run(self) -> RunBundle:
        return self._runs[self._active_run_name].bundle

    @property
    def available_outputs(self) -> set[str]:
        return self.active_run.available_outputs

    def set_active_run(self, name: str) -> None:
        if name not in self._runs:
            raise ValueError(f"No run named {name!r}. Available: {self.run_names!r}")
        # RunBundles are mutable; revalidate so a post-construction
        # overlay drift can't leave the navigator inconsistent.
        self._validate_time_grid_alignment()
        self._validate_overlay_alignment()
        self._active_run_name = name

    @property
    def time(self) -> np.ndarray:
        return self._runs[self._active_run_name].time

    @property
    def time_edges(self) -> np.ndarray:
        return self._runs[self._active_run_name].time_edges

    @property
    def n_time(self) -> int:
        return int(self.time.size)

    @property
    def event_index(self) -> SpikeEventIndex:
        return self._runs[self._active_run_name].event_index

    def spike_event_at(self, event_id: int) -> SpikeEvent:
        return self.event_index.event_at(event_id)

    def event_ids_at_bin(self, t_idx: int) -> np.ndarray:
        return self.event_index.event_ids_at_bin(t_idx)

    def event_ids_for_window(self, sl: slice) -> np.ndarray:
        return self.event_index.event_ids_for_window(sl)

    # ------------------------------------------------------------------
    # Hot-path readers — direct zarr.Array indexing
    # ------------------------------------------------------------------

    def window_indices(self, t_center: float, t_width: float) -> slice:
        """Mirror ``InMemoryDecoderDataSource.window_indices`` exactly.

        Both data sources must compute the same slice for the same
        ``(t_center, t_width)`` so panel rendering is bit-identical
        between paths.
        """
        if t_width <= 0:
            raise ValueError(f"t_width must be positive. Got {t_width}.")
        time = self.time
        n_time = time.size
        if n_time == 0:
            return slice(0, 0)
        if t_center > time[-1]:
            i_start = int(np.searchsorted(time, time[-1] - t_width, side="left"))
            i_start = max(0, min(i_start, n_time - 1))
            return slice(i_start, n_time)
        if t_center < time[0]:
            i_stop = int(np.searchsorted(time, time[0] + t_width, side="right"))
            i_stop = max(1, min(i_stop, n_time))
            return slice(0, i_stop)
        half = t_width / 2.0
        i_start = int(np.searchsorted(time, t_center - half, side="left"))
        i_stop = int(np.searchsorted(time, t_center + half, side="right"))
        i_start = max(0, i_start)
        i_stop = max(i_start + 1, min(i_stop, n_time))
        return slice(i_start, i_stop)

    def load_posterior(self, sl: slice) -> np.ndarray:
        """Window slice of ``acausal_posterior``: ``(n_visible, n_state_bins)``."""
        return np.asarray(
            self._runs[self._active_run_name].posterior[sl], dtype=np.float32
        )

    def load_likelihood(self, sl: slice) -> np.ndarray:
        """Window slice of ``log_likelihood``.

        Raises a ``KeyError`` with the same rebuild-instruction message
        as ``InMemoryDecoderDataSource.load_likelihood`` when the cache
        was built without ``log_likelihood``.
        """
        handles = self._runs[self._active_run_name]
        if handles.likelihood is None:
            raise KeyError(
                "Active run has no `log_likelihood` in results. Re-run "
                "`predict(return_outputs=['log_likelihood'])` and rebuild "
                "the RunBundle."
            )
        return np.asarray(handles.likelihood[sl])

    def load_predictive(self, sl: slice) -> np.ndarray | None:
        """Window slice of ``predictive_posterior``, or ``None`` if absent."""
        handles = self._runs[self._active_run_name]
        if handles.predictive is None:
            return None
        return np.asarray(handles.predictive[sl])

    def load_state_probabilities(self, sl: slice) -> np.ndarray:
        """Window slice of ``acausal_state_probabilities``."""
        return np.asarray(self._runs[self._active_run_name].state_probabilities[sl])

    def load_position(self, sl: slice) -> np.ndarray | None:
        """1D position interpolated onto the decoder time grid, sliced.

        Returns ``None`` for 2D position bundles (heatmap overlay is
        1D-only per the v1 contract). Cache is per-run-name, retained
        across ``set_active_run``.
        """
        position_at_decoder_time = self._position_at_decoder_time()
        if position_at_decoder_time is None:
            return None
        return position_at_decoder_time[sl]

    def _position_at_decoder_time(self) -> np.ndarray | None:
        cached = self._position_cache.get(self._active_run_name, ...)
        if cached is not ...:
            return cached
        run = self.active_run
        position = np.asarray(run.position) if run.position is not None else None
        position_time = (
            np.asarray(run.position_time) if run.position_time is not None else None
        )
        if position is None or position_time is None or position.ndim != 1:
            self._position_cache[self._active_run_name] = None
            return None
        decoder_time = self.time
        interp = np.interp(
            decoder_time, position_time.astype(float), position.astype(float)
        )
        self._position_cache[self._active_run_name] = interp
        return interp

    def slice_at_index(
        self, t_idx: int, which: SliceWhich = "posterior"
    ) -> np.ndarray | None:
        """Single-row slice for the SlicePanel's per-bin readout."""
        handles = self._runs[self._active_run_name]
        which_to_handle = {
            "posterior": handles.posterior,
            "acausal": handles.posterior,
            "likelihood": handles.likelihood,
            "predictive": handles.predictive,
        }
        handle = which_to_handle[which]
        if handle is None:
            if which == "likelihood":
                raise KeyError("Active run has no `log_likelihood` in results.")
            return None
        return np.asarray(handle[t_idx])
