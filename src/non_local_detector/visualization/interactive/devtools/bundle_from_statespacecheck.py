"""Build a viewer bundle directory from a ``statespacecheck-paper-viewer`` cache.

Reads upstream's split layout (intermediates dir holds the source NetCDF
+ pickled fitted model; cache dir holds the per-recording sidecars) and
emits a CLI-compatible bundle directory containing the four files the
viewer's ``--run`` flag consumes.

Upstream filename conventions are encoded directly here rather than
imported from ``statespacecheck_paper.interactive.cache`` so the devtool
runs without statespacecheck installed (the convention is a stable path
table; if upstream ever changes it both repos need updates regardless).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr

from non_local_detector.analysis.place_fields import (
    extract_state_aligned_place_fields,
)
from non_local_detector.models.base import _DetectorBase

ModelName = Literal["continuous", "contfrag"]
MODEL_NAMES: tuple[ModelName, ...] = ("continuous", "contfrag")

# Required by the viewer's ``RunBundle`` contract — both must be present
# in the source results dataset regardless of which substitution flag the
# user passed.
_REQUIRED_RESULTS_VARS: tuple[str, ...] = (
    "acausal_posterior",
    "acausal_state_probabilities",
)

# Upstream's place_fields.npz stores interior-only float32 fields. Match
# the upstream dtype exactly when comparing; the cross-check is a
# version-mismatch sanity guard, not a bit-identity assertion.
_PLACE_FIELDS_RTOL = 1e-5
_PLACE_FIELDS_ATOL = 1e-6


@dataclass(frozen=True)
class _ModelPaths:
    results_nc: Path
    model_pkl: Path


def _model_paths(intermediates_dir: Path, model: ModelName) -> _ModelPaths:
    """Mirror of upstream ``cache.model_paths`` — see module docstring."""
    if model == "continuous":
        return _ModelPaths(
            results_nc=intermediates_dir / "cont_results.nc",
            model_pkl=intermediates_dir / "cont_model.pkl",
        )
    if model == "contfrag":
        return _ModelPaths(
            results_nc=intermediates_dir / "cont_frag_results.nc",
            model_pkl=intermediates_dir / "cont_frag_model.pkl",
        )
    raise ValueError(
        f"Unknown model {model!r}. Expected one of: {list(MODEL_NAMES)!r}"
    )


def _zarr_path(cache_dir: Path, model: ModelName) -> Path:
    return cache_dir / f"figure04_{model}.zarr"


def _place_fields_path(cache_dir: Path, model: ModelName) -> Path:
    return cache_dir / f"figure04_{model}_place_fields.npz"


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "figure04_meta.npz"


def _spike_times_path(cache_dir: Path) -> Path:
    return cache_dir / "figure04_spike_times.npy"


def _load_results_nc(results_nc: Path) -> xr.Dataset:
    """Load + materialize the source NetCDF, restoring the state_bins MultiIndex."""
    return _DetectorBase.load_results(str(results_nc))


def _load_results_zarr(zarr_path: Path) -> xr.Dataset:
    """Load the per-model Zarr fallback. Caller validates required vars.

    Upstream's Zarr stores ``acausal_posterior`` only when the source
    dataset had it (per ``data_source.py``), so callers must validate
    presence; we don't pre-filter.
    """
    try:
        import zarr  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "--results-from-zarr requires the optional 'zarr' package. "
            "Install it (e.g. `uv pip install zarr`) or drop the flag to "
            "consume the source NetCDF instead."
        ) from exc
    return xr.open_zarr(str(zarr_path), consolidated=True).load()


def _validate_required_results_vars(results: xr.Dataset, source: Path) -> None:
    missing = [v for v in _REQUIRED_RESULTS_VARS if v not in results.data_vars]
    if missing:
        raise ValueError(
            f"Source results at {source} is missing required variables "
            f"{missing!r}. RunBundle requires "
            f"{list(_REQUIRED_RESULTS_VARS)!r} to be present. "
            f"Got data_vars: {sorted(results.data_vars)!r}."
        )


def _cross_check_place_fields(
    detector: _DetectorBase,
    cache_pf_path: Path,
    model: ModelName,
) -> None:
    """Compare the cache's interior-only place fields against the detector's.

    Mismatches indicate the cache + intermediates were built from
    different upstream runs (or the underlying detector pickle has
    drifted from the cache snapshot). Raises with shape + magnitude
    diagnostics so the user knows whether to rebuild the cache or
    re-fetch the intermediates.
    """
    if not cache_pf_path.exists():
        raise FileNotFoundError(
            f"Cache place-fields sidecar not found at {cache_pf_path}. "
            f"Expected upstream's figure04_{model}_place_fields.npz."
        )
    with np.load(cache_pf_path) as npz:
        if "place_fields" not in npz:
            raise ValueError(
                f"{cache_pf_path} has no 'place_fields' key. "
                f"Got keys: {sorted(npz.keys())!r}."
            )
        cache_fields = np.asarray(npz["place_fields"], dtype=np.float32)
    detector_fields_full = extract_state_aligned_place_fields(detector)
    interior_mask = np.asarray(detector.is_track_interior_state_bins_, dtype=bool)
    detector_fields = detector_fields_full[:, interior_mask].astype(np.float32)
    if detector_fields.shape != cache_fields.shape:
        raise ValueError(
            f"Place-field shape mismatch between cache and detector. "
            f"Cache shape: {cache_fields.shape}; detector "
            f"(interior-masked): {detector_fields.shape}. "
            "Cache + intermediates likely came from different upstream "
            "runs; rebuild whichever is older."
        )
    if not np.allclose(
        detector_fields,
        cache_fields,
        rtol=_PLACE_FIELDS_RTOL,
        atol=_PLACE_FIELDS_ATOL,
    ):
        max_abs = float(np.nanmax(np.abs(detector_fields - cache_fields)))
        raise ValueError(
            f"Place-field values disagree between cache ({cache_pf_path}) "
            f"and detector pickle. Max abs diff: {max_abs:g} "
            f"(rtol={_PLACE_FIELDS_RTOL}, atol={_PLACE_FIELDS_ATOL}). "
            "Cache + intermediates likely came from different upstream "
            "runs; rebuild whichever is older."
        )


def _read_meta(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time, linear_position)`` from the shared meta sidecar."""
    meta_p = _meta_path(cache_dir)
    if not meta_p.exists():
        raise FileNotFoundError(
            f"Meta sidecar not found at {meta_p}. Expected upstream's "
            "figure04_meta.npz with keys 'time' + 'linear_position'."
        )
    with np.load(meta_p) as npz:
        for key in ("time", "linear_position"):
            if key not in npz:
                raise ValueError(
                    f"{meta_p} has no {key!r} key. "
                    f"Got keys: {sorted(npz.keys())!r}."
                )
        time = np.asarray(npz["time"], dtype=np.float64)
        linear_position = np.asarray(npz["linear_position"], dtype=np.float64)
    return time, linear_position


def _read_spike_times(cache_dir: Path) -> list[np.ndarray]:
    """Return per-cell spike-time arrays from the shared spike-times sidecar."""
    sp_p = _spike_times_path(cache_dir)
    if not sp_p.exists():
        raise FileNotFoundError(
            f"Spike-times sidecar not found at {sp_p}. Expected "
            "upstream's figure04_spike_times.npy (object-dtype)."
        )
    arr = np.load(sp_p, allow_pickle=True)
    if arr.dtype != object:
        raise ValueError(
            f"{sp_p} expected object-dtype array, got {arr.dtype}. "
            "Upstream's spike-times sidecar should be one object-dtype "
            "ndarray of per-cell float64 spike-time arrays."
        )
    return [np.asarray(st, dtype=np.float64) for st in arr]


def _write_bundle_dir(
    out_dir: Path,
    results: xr.Dataset,
    detector: _DetectorBase,
    spike_times: list[np.ndarray],
    time: np.ndarray,
    linear_position: np.ndarray,
) -> None:
    """Emit the four CLI-compatible bundle files under ``out_dir``.

    Round-trips the detector through ``_DetectorBase.save_model`` (plain
    pickle) so the canonical ``app._load_run`` loader can read it back
    regardless of whether the source pkl was joblib- or pickle-flavoured.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _DetectorBase.save_results(results, str(out_dir / "results.nc"))
    detector.save_model(str(out_dir / "model.pkl"))
    np.savez(
        out_dir / "spikes.npz",
        spike_times=np.asarray(spike_times, dtype=object),
    )
    pd.DataFrame({"position": linear_position}, index=pd.Index(time, name="time")).to_parquet(
        out_dir / "position.parquet"
    )


def bundle_from_statespacecheck_cache(
    cache_dir: Path | str,
    intermediates_dir: Path | str,
    model: ModelName,
    out: Path | str,
    *,
    results_nc: Path | str | None = None,
    model_pkl: Path | str | None = None,
    results_from_zarr: bool = False,
) -> Path:
    """Bundle a statespacecheck cache + intermediates into a viewer bundle dir.

    Parameters
    ----------
    cache_dir
        Directory containing upstream's figure-4 sidecars
        (``figure04_meta.npz``, ``figure04_spike_times.npy``,
        ``figure04_<model>_place_fields.npz``,
        ``figure04_<model>.zarr``).
    intermediates_dir
        Directory containing the source NetCDF + pickled fitted detector
        (``cont_results.nc`` / ``cont_model.pkl`` for ``model="continuous"``;
        ``cont_frag_results.nc`` / ``cont_frag_model.pkl`` for ``"contfrag"``).
    model
        Upstream model key — ``"continuous"`` or ``"contfrag"``. Mirrors
        the upstream literal exactly so users don't have to remember the
        ``cont`` / ``cont_frag`` filename quirk.
    out
        Output bundle directory. Created if missing. Will be populated
        with ``results.nc``, ``model.pkl``, ``spikes.npz``, ``position.parquet``.
    results_nc
        Override for the auto-resolved ``model_paths.results_nc``.
    model_pkl
        Override for the auto-resolved ``model_paths.model_pkl``.
    results_from_zarr
        Substitute the per-model Zarr in ``cache_dir`` for the source
        NetCDF. Upstream's Zarr stores ``acausal_posterior`` only when
        the source had it; this devtool revalidates both required vars
        in either case and refuses to run if any are missing.

    Returns
    -------
    Path
        The output bundle directory (same as ``out``).
    """
    cache_dir = Path(cache_dir)
    intermediates_dir = Path(intermediates_dir)
    out = Path(out)
    if model not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model {model!r}. Expected one of: {list(MODEL_NAMES)!r}"
        )

    paths = _model_paths(intermediates_dir, model)
    resolved_model_pkl = Path(model_pkl) if model_pkl is not None else paths.model_pkl

    if results_from_zarr:
        if results_nc is not None:
            raise ValueError(
                "--results-nc and --results-from-zarr are mutually exclusive."
            )
        results_source = _zarr_path(cache_dir, model)
        results = _load_results_zarr(results_source)
    else:
        results_source = (
            Path(results_nc) if results_nc is not None else paths.results_nc
        )
        results = _load_results_nc(results_source)
    _validate_required_results_vars(results, results_source)

    if not resolved_model_pkl.exists():
        raise FileNotFoundError(
            f"Detector pickle not found at {resolved_model_pkl}."
        )
    # Upstream pickles are joblib-serialized. joblib's numpy codec
    # produces files that ``pickle.load`` *cannot* read (verified
    # against the real ``cont_model.pkl`` — fails with
    # ``UnpicklingError: invalid load key``); use ``joblib.load`` for
    # the source. The viewer bundle's ``model.pkl`` is then re-emitted
    # via ``detector.save_model`` so ``app._load_run`` keeps using the
    # canonical pickle-based loader.
    detector = joblib.load(str(resolved_model_pkl))

    time, linear_position = _read_meta(cache_dir)
    spike_times = _read_spike_times(cache_dir)
    _cross_check_place_fields(
        detector, _place_fields_path(cache_dir, model), model
    )

    _write_bundle_dir(out, results, detector, spike_times, time, linear_position)
    return out
