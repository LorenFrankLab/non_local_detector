"""Build a viewer bundle directory from in-memory ``predict()`` outputs.

Mirror image of ``bundle_from_statespacecheck_cache``: that one
adapts an upstream layout; this one packages a fitted detector + its
``predict()`` outputs + the session sidecars (spike_times, position)
into a directory ``--run-from-dir`` understands.

Use case: a user already has the four pieces in RAM (notebook
workflow). Pickling them to a directory lets them
share / re-open the session via the CLI without redoing the fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from non_local_detector.models.base import _DetectorBase


def bundle_from_detector(
    detector: _DetectorBase,
    results: xr.Dataset,
    spike_times: list[np.ndarray],
    position: np.ndarray,
    position_time: np.ndarray,
    out: Path,
    speed: np.ndarray | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a ``--run-from-dir``-compatible bundle directory.

    Parameters
    ----------
    detector
        Fitted detector instance — pickled to ``out/model.pkl`` via
        ``_DetectorBase.save_model``.
    results
        Output of ``detector.predict(...)`` — written to
        ``out/results.nc`` via ``_DetectorBase.save_results`` (which
        preserves the ``state_bins`` MultiIndex).
    spike_times
        List of per-neuron spike-time arrays — saved as object-dtype
        ``np.savez`` to ``out/spikes.npz``.
    position
        1D or 2D position array. 1D goes into a ``position`` column;
        2D goes into ``x_position`` + ``y_position`` columns.
    position_time
        Time index aligned to ``position``.
    out
        Output directory; created if missing.
    speed
        Optional speed array; goes into a ``speed`` column when
        present.
    overwrite
        If ``True``, replace existing files. Default ``False`` raises
        ``FileExistsError`` rather than clobber a directory the user
        may already be using.

    Returns
    -------
    Path
        The output directory path.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    nc_path = out / "results.nc"
    model_path = out / "model.pkl"
    spikes_path = out / "spikes.npz"
    position_path = out / "position.parquet"
    if not overwrite:
        for p in (nc_path, model_path, spikes_path, position_path):
            if p.exists():
                raise FileExistsError(
                    f"bundle-from-detector: {p!s} already exists. Pass "
                    "--overwrite to replace it."
                )

    _DetectorBase.save_results(results, str(nc_path))
    detector.save_model(str(model_path))

    spikes_obj = np.empty(len(spike_times), dtype=object)
    for i, st in enumerate(spike_times):
        spikes_obj[i] = np.asarray(st, dtype=float)
    np.savez(str(spikes_path), spike_times=spikes_obj)

    position = np.asarray(position)
    if position.ndim == 1:
        pos_df = pd.DataFrame(
            {"position": position}, index=pd.Index(position_time, name="time")
        )
    elif position.ndim == 2 and position.shape[1] == 2:
        pos_df = pd.DataFrame(
            {
                "x_position": position[:, 0],
                "y_position": position[:, 1],
            },
            index=pd.Index(position_time, name="time"),
        )
    else:
        raise ValueError(
            "bundle-from-detector: position must be shape (n_pos_time,) "
            f"or (n_pos_time, 2); got {position.shape}."
        )
    if speed is not None:
        pos_df["speed"] = np.asarray(speed)
    pos_df.to_parquet(str(position_path))

    return out


def _load_pickled_detector(path: Path) -> _DetectorBase:
    """Load a pickled detector saved via ``_DetectorBase.save_model``."""
    return _DetectorBase.load_model(str(path))


def bundle_from_detector_cli(
    detector_path: Path,
    results_nc: Path,
    spikes_npz: Path,
    position_parquet: Path,
    out: Path,
    overwrite: bool = False,
) -> Path:
    """CLI wrapper: load each piece from disk, then call
    ``bundle_from_detector``.

    The CLI form takes paths because that's what argparse hands in;
    the in-process function above takes the raw objects directly so
    the notebook + tests can call it without round-tripping through
    files.
    """
    detector = _load_pickled_detector(detector_path)
    results = _DetectorBase.load_results(str(results_nc))
    spikes_npz_data = np.load(str(spikes_npz), allow_pickle=True)
    spike_times = list(spikes_npz_data["spike_times"])
    pos_df = pd.read_parquet(str(position_parquet))
    if "position" in pos_df.columns:
        position = pos_df["position"].to_numpy()
    elif {"x_position", "y_position"}.issubset(pos_df.columns):
        position = pos_df[["x_position", "y_position"]].to_numpy()
    else:
        raise ValueError(
            f"bundle-from-detector --position {position_parquet!s}: "
            "must have either a 'position' column (1D) or "
            "'x_position' + 'y_position' columns (2D)."
        )
    position_time = pos_df.index.to_numpy()
    speed = pos_df["speed"].to_numpy() if "speed" in pos_df else None
    return bundle_from_detector(
        detector=detector,
        results=results,
        spike_times=spike_times,
        position=position,
        position_time=position_time,
        out=out,
        speed=speed,
        overwrite=overwrite,
    )
