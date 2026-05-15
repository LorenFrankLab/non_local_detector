"""Fit a Non-Local model on the statespacecheck-paper session + view it.

The ``statespacecheck-paper`` cache ships pre-fit ``continuous`` and
``contfrag`` models (the ``viewer_bundles/{continuous,contfrag}/``
directories are built from them via ``bundle_from_statespacecheck``).
There is no pre-fit Non-Local model. This script fits one on the
*same* session — same spike times, same 1D linearized position — and
launches the interactive viewer.

The fit result is cached as a ``viewer_bundles/non_local/`` bundle
directory so re-runs skip the (slow) EM fit and just reload + launch.

Run::

    SSC_CACHE_DIR=/path/to/statespacecheck-paper/data/cache \\
    NLD_REAL_DATA_BUNDLE_DIR=/path/to/non_local_detector/viewer_bundles \\
    uv run python scripts/fit_ssc_non_local.py

Both env vars have defaults matching a side-by-side checkout layout,
so on that layout a bare ``uv run python scripts/fit_ssc_non_local.py``
also works.

Caveat: place fields are estimated from the *whole* session — the
SSC cache carries no ripple / speed mask, so putative replay periods
are not excluded from the encoding fit. Fine for a viewer smoke;
for analysis pass an ``is_training`` mask to ``estimate_parameters``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.models.base import _DetectorBase
from non_local_detector.visualization.interactive import launch_qt
from non_local_detector.visualization.interactive.devtools.bundle_from_detector import (
    bundle_from_detector,
)
from non_local_detector.visualization.interactive.view_models.base import RunBundle

# Matches ``viewer_bundles/continuous``'s environment (256 bins over
# the ~500 cm linearized track).
_PLACE_BIN_SIZE = 2.0
_BUNDLE_NAME = "non_local"
_REQUIRED_SIDECARS = ("results.nc", "model.pkl", "spikes.npz", "position.parquet")

_DEFAULT_SSC_CACHE = Path(
    "/Users/edeno/Documents/GitHub/statespacecheck-paper/data/cache"
)
_DEFAULT_BUNDLE_DIR = Path(
    "/Users/edeno/Documents/GitHub/non_local_detector/viewer_bundles"
)


def _ssc_cache_dir() -> Path:
    return Path(os.environ.get("SSC_CACHE_DIR", _DEFAULT_SSC_CACHE))


def _bundle_root() -> Path:
    return Path(os.environ.get("NLD_REAL_DATA_BUNDLE_DIR", _DEFAULT_BUNDLE_DIR))


def _load_ssc_session(
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Load ``(time, linear_position, spike_times)`` from the SSC cache."""
    meta = np.load(cache_dir / "figure04_meta.npz")
    time = np.asarray(meta["time"], dtype=np.float64)
    position = np.asarray(meta["linear_position"], dtype=np.float64)
    spike_times = list(
        np.load(cache_dir / "figure04_spike_times.npy", allow_pickle=True)
    )
    return time, position, spike_times


def _fit_and_bundle(cache_dir: Path, bundle_dir: Path) -> RunBundle:
    """Fit the Non-Local model, predict, write the bundle, return it."""
    time, position, spike_times = _load_ssc_session(cache_dir)
    print(
        f"SSC session: {time[-1] - time[0]:.1f} s, {time.size} bins, "
        f"{len(spike_times)} cells, "
        f"{sum(len(st) for st in spike_times)} spikes"
    )

    env = Environment(place_bin_size=_PLACE_BIN_SIZE)
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        non_local_position_penalty=1.0,
        non_local_penalty_std=5.0,
        local_position_std=1.0,
        environments=[env],
    )
    print("fitting NonLocalSortedSpikesDetector (EM) — this is the slow step...")
    detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time=time,
        return_outputs=None,
    )
    print("predicting over the full session...")
    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        return_outputs="all",
    )

    print(f"writing bundle → {bundle_dir}")
    bundle_from_detector(
        detector=detector,
        results=results,
        spike_times=spike_times,
        position=position,
        position_time=time,
        out=bundle_dir,
        overwrite=True,
    )
    return RunBundle(
        results=results,
        detector=detector,
        spike_times=spike_times,
        position_time=time,
        position=position,
    )


def _load_cached_bundle(bundle_dir: Path) -> RunBundle:
    """Reload a previously-written ``non_local`` bundle directory."""
    print(f"loading cached bundle ← {bundle_dir}")
    results = _DetectorBase.load_results(str(bundle_dir / "results.nc"))
    detector = _DetectorBase.load_model(str(bundle_dir / "model.pkl"))
    import pandas as pd

    spikes_npz = np.load(str(bundle_dir / "spikes.npz"), allow_pickle=True)
    spike_times = list(spikes_npz["spike_times"])
    position_df = pd.read_parquet(str(bundle_dir / "position.parquet"))
    return RunBundle(
        results=results,
        detector=detector,
        spike_times=spike_times,
        position_time=position_df.index.to_numpy(),
        position=position_df["position"].to_numpy(),
    )


def main() -> int:
    cache_dir = _ssc_cache_dir()
    bundle_dir = _bundle_root() / _BUNDLE_NAME

    cached = bundle_dir.is_dir() and all(
        (bundle_dir / name).exists() for name in _REQUIRED_SIDECARS
    )
    if cached:
        bundle = _load_cached_bundle(bundle_dir)
    else:
        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"SSC cache directory not found: {cache_dir}. Set "
                "SSC_CACHE_DIR to the statespacecheck-paper data/cache path."
            )
        bundle = _fit_and_bundle(cache_dir, bundle_dir)

    return launch_qt(bundle, t_width=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
