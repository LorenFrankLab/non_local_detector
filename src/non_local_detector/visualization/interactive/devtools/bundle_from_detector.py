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

import shutil
import tempfile
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
    position_2d: np.ndarray | None = None,
    position_2d_time: np.ndarray | None = None,
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
    position_2d
        Optional raw 2D animal position for graph-linearized 1D
        decoders. When ``position`` is 1D, this writes
        ``x_position`` + ``y_position`` columns alongside the
        linearized ``position`` column for the opt-in projected-2D
        viewer panel.
    position_2d_time
        Time index aligned to ``position_2d``. May differ from
        ``position_time`` (e.g. a 2D camera sampled on its own
        clock); a different grid causes the 2D track to be written
        to a separate ``position_2d.parquet`` sidecar, while
        equal-grid inputs are co-muxed into ``position.parquet``.
    overwrite
        Controls how an existing ``out`` directory is handled.

        - ``False`` (default): refuse to write if any of the four
          bundle sidecars already exist; **preserve unrelated files**
          in ``out`` (e.g. a ``README.md`` or ``.gitignore``). Each of
          the four new sidecars is written via per-file atomic
          rename out of a sibling staging directory. Aggregate
          atomicity across the four files is not guaranteed (a crash
          between renames could leave 1–3 new sidecars in place);
          this trade is intentional — unrelated files in ``out`` are
          considered the user's, not ours.
        - ``True``: replace the entire ``out`` directory. The four
          sidecars are written to a sibling staging directory, then
          the whole directory is atomically swapped into place
          (existing ``out`` moved to a backup, removed after the
          swap commits). **All non-bundle files in ``out`` are
          discarded** under this mode — only opt in when the
          directory is owned by the bundle.

    Returns
    -------
    Path
        The output directory path.
    """
    out = Path(out)
    out_parent = out.parent
    out_parent.mkdir(parents=True, exist_ok=True)

    sidecars = ["results.nc", "model.pkl", "spikes.npz", "position.parquet"]
    write_position_2d_sidecar = (
        position_2d is not None
        and position_2d_time is not None
        and not np.array_equal(np.asarray(position_2d_time), np.asarray(position_time))
    )
    if write_position_2d_sidecar:
        sidecars.append("position_2d.parquet")

    if not overwrite:
        for fname in sidecars:
            if (out / fname).exists():
                raise FileExistsError(
                    f"bundle-from-detector: {(out / fname)!s} already exists. "
                    "Pass --overwrite to replace it."
                )

    # Validate + build every sidecar in memory BEFORE writing anything,
    # so a bad-shape position or length-mismatched speed raises before
    # any file is touched. Otherwise the function could leave a partial
    # bundle on disk (e.g. results.nc + model.pkl + spikes.npz written,
    # position.parquet missing or stale from a previous run when
    # ``overwrite=True`` mixed new + old files).
    position = np.asarray(position)
    position_time_arr = np.asarray(position_time)
    if position.ndim == 1:
        if position.shape[0] != position_time_arr.shape[0]:
            raise ValueError(
                "bundle-from-detector: position length "
                f"{position.shape[0]} != position_time length "
                f"{position_time_arr.shape[0]}."
            )
        pos_df = pd.DataFrame(
            {"position": position},
            index=pd.Index(position_time_arr, name="time"),
        )
    elif position.ndim == 2 and position.shape[1] == 2:
        if position.shape[0] != position_time_arr.shape[0]:
            raise ValueError(
                "bundle-from-detector: position length "
                f"{position.shape[0]} != position_time length "
                f"{position_time_arr.shape[0]}."
            )
        pos_df = pd.DataFrame(
            {
                "x_position": position[:, 0],
                "y_position": position[:, 1],
            },
            index=pd.Index(position_time_arr, name="time"),
        )
    else:
        raise ValueError(
            "bundle-from-detector: position must be shape (n_pos_time,) "
            f"or (n_pos_time, 2); got {position.shape}."
        )
    if speed is not None:
        speed_arr = np.asarray(speed)
        if speed_arr.shape[0] != position_time_arr.shape[0]:
            raise ValueError(
                "bundle-from-detector: speed length "
                f"{speed_arr.shape[0]} != position_time length "
                f"{position_time_arr.shape[0]}."
            )
        pos_df["speed"] = speed_arr
    pos_2d_df: pd.DataFrame | None = None
    if position_2d is not None:
        position_2d_arr = np.asarray(position_2d)
        position_2d_time_arr = (
            np.asarray(position_2d_time)
            if position_2d_time is not None
            else position_time_arr
        )
        if position.ndim != 1:
            raise ValueError(
                "bundle-from-detector: position_2d is only supported when "
                "position is the 1D decoder coordinate."
            )
        if position_2d_arr.ndim != 2 or position_2d_arr.shape[1] != 2:
            raise ValueError(
                "bundle-from-detector: position_2d must be shape "
                f"(n_position_2d_time, 2); got {position_2d_arr.shape}."
            )
        if position_2d_arr.shape[0] != position_2d_time_arr.shape[0]:
            raise ValueError(
                "bundle-from-detector: position_2d length "
                f"{position_2d_arr.shape[0]} != position_2d_time length "
                f"{position_2d_time_arr.shape[0]}."
            )
        if write_position_2d_sidecar:
            # Different time grid: write to a separate sidecar so the
            # reader can keep the two grids distinct on load.
            pos_2d_df = pd.DataFrame(
                {
                    "x_position": position_2d_arr[:, 0],
                    "y_position": position_2d_arr[:, 1],
                },
                index=pd.Index(position_2d_time_arr, name="time"),
            )
        else:
            # Same time grid: co-mux into position.parquet alongside
            # the linearized ``position`` column. Cheaper than a second
            # sidecar and keeps single-grid bundles trivially portable.
            pos_df["x_position"] = position_2d_arr[:, 0]
            pos_df["y_position"] = position_2d_arr[:, 1]

    spikes_obj = np.empty(len(spike_times), dtype=object)
    for i, st in enumerate(spike_times):
        spikes_obj[i] = np.asarray(st, dtype=float)

    # All sidecars validated. Write to a sibling staging directory so
    # a write-time failure (disk-full mid-``to_parquet``, permission
    # error, kill -9) leaves the user's existing ``out`` directory
    # untouched. On success, atomically swap staging → out (a single
    # ``Path.rename`` is atomic on POSIX when source + dest live on
    # the same filesystem; mkdtemp's sibling placement guarantees
    # that). Any pre-existing ``out`` is moved aside under a backup
    # name first and removed after the swap commits.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.staging.", dir=str(out_parent))
    )
    try:
        _DetectorBase.save_results(results, str(staging / "results.nc"))
        detector.save_model(str(staging / "model.pkl"))
        np.savez(str(staging / "spikes.npz"), spike_times=spikes_obj)
        pos_df.to_parquet(str(staging / "position.parquet"))
        if pos_2d_df is not None:
            pos_2d_df.to_parquet(str(staging / "position_2d.parquet"))
    except BaseException:
        # Any write failure → drop staging and propagate so the user's
        # existing ``out`` (if any) is bit-identical to before the
        # call. ``BaseException`` covers KeyboardInterrupt too.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if overwrite:
        # Whole-directory swap: user explicitly opted in to replace
        # ``out``. Atomic via two renames (``out → backup``, then
        # ``staging → out``); the backup is removed after the swap
        # commits so the only observable end states are "fully old"
        # or "fully new". Non-bundle files in ``out`` are lost under
        # this mode, by design.
        if out.exists():
            backup = out_parent / (f".{out.name}.bak.{staging.name.rsplit('.', 1)[-1]}")
            try:
                out.rename(backup)
            except OSError:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            try:
                staging.rename(out)
            except OSError:
                # Restore the original; staging keeps the new
                # artefacts for inspection.
                backup.rename(out)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            staging.rename(out)
    else:
        # Per-file rename: preserve any unrelated files in ``out``.
        # Each individual rename is atomic on POSIX; the four
        # together are NOT aggregate-atomic, but the per-file
        # existence check above already rejected if any sidecar
        # collided, so the renames only land in empty positions.
        out.mkdir(parents=True, exist_ok=True)
        try:
            for fname in sidecars:
                (staging / fname).rename(out / fname)
        except OSError:
            # Mid-rename failure is rare (renames within the same
            # filesystem don't normally fail post-staging). Surface
            # it; staging may retain unmoved sidecars for the user
            # to inspect / move manually.
            raise
        shutil.rmtree(staging, ignore_errors=True)

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
    sibling_2d = Path(position_parquet).with_name("position_2d.parquet")
    if "position" in pos_df.columns:
        position = pos_df["position"].to_numpy()
        if sibling_2d.exists():
            pos_2d_df = pd.read_parquet(str(sibling_2d))
            position_2d = pos_2d_df[["x_position", "y_position"]].to_numpy()
            position_2d_time = pos_2d_df.index.to_numpy()
        elif {"x_position", "y_position"}.issubset(pos_df.columns):
            position_2d = pos_df[["x_position", "y_position"]].to_numpy()
            position_2d_time = pos_df.index.to_numpy()
        else:
            position_2d = None
            position_2d_time = None
    elif {"x_position", "y_position"}.issubset(pos_df.columns):
        position = pos_df[["x_position", "y_position"]].to_numpy()
        position_2d = None
        position_2d_time = None
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
        position_2d=position_2d,
        position_2d_time=position_2d_time,
        overwrite=overwrite,
    )
