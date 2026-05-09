"""Benchmark ``InMemoryDecoderDataSource`` vs ``ZarrDirectDecoderDataSource``.

Phase 5.5 hands out-of-CI: the parity tests (in
``test_data_source_zarr_direct.py``) gate correctness; this script
reports the speedup numbers for the PR description without flaking
on machine-dependent disk speed or cold-cache variance. Run it on a
representative real-data bundle (or the simulated session by default
for a smoke check).

Usage::

    uv run python scripts/bench_data_source.py [--bundle-dir DIR]
                                                [--n-loads N]
                                                [--seed SEED]

The default ``--bundle-dir`` builds a transient simulated bundle,
which is fine for smoke-checking the script itself but doesn't
exercise long-session chunked reads. For the PR-description numbers,
point at a real bundle whose ``results.zarr/`` is already built via
``build-viewer-cache``.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np


def _bench_one(label: str, fn, n_loads: int, sl_factory) -> dict[str, float]:
    """Run ``fn(sl)`` ``n_loads`` times with fresh slices; report stats."""
    durations: list[float] = []
    for _ in range(n_loads):
        sl = sl_factory()
        t0 = time.perf_counter()
        fn(sl)
        durations.append(time.perf_counter() - t0)
    return {
        "label": label,
        "n": n_loads,
        "median_ms": statistics.median(durations) * 1e3,
        "p95_ms": statistics.quantiles(durations, n=20)[-1] * 1e3
        if n_loads >= 20
        else max(durations) * 1e3,
        "min_ms": min(durations) * 1e3,
    }


def _print_table(rows: list[dict[str, float]]) -> None:
    print(f"{'label':<48} {'n':>6} {'median':>10} {'p95':>10} {'min':>10}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['label']:<48} {r['n']:>6} "
            f"{r['median_ms']:>9.3f}ms {r['p95_ms']:>9.3f}ms {r['min_ms']:>9.3f}ms"
        )


def _materialise_simulated_bundle(out_dir: Path) -> Path:
    """Build a transient simulated bundle with a zarr cache (smoke mode)."""
    import pandas as pd

    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.tests._simulated_detectors import (
        fit_nl_detector,
        make_session,
        predict_variants,
    )
    from non_local_detector.visualization.interactive.devtools.build_viewer_cache import (
        build_viewer_cache,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    fitted = fit_nl_detector(session)
    results = predict_variants(fitted, session)["all"]
    _DetectorBase.save_results(results, str(out_dir / "results.nc"))
    _DetectorBase.save_model(fitted.detector, str(out_dir / "model.pkl"))
    spike_obj = np.empty(len(session.spike_times), dtype=object)
    for i, st in enumerate(session.spike_times):
        spike_obj[i] = np.asarray(st, dtype=float)
    np.savez(str(out_dir / "spikes.npz"), spike_times=spike_obj)
    pos_df = pd.DataFrame({"position": np.asarray(session.position).squeeze()})
    pos_df.index = pd.Index(session.time, name="time")
    if session.speed is not None:
        pos_df["speed"] = np.asarray(session.speed)
    pos_df.to_parquet(str(out_dir / "position.parquet"))
    build_viewer_cache(out_dir, overwrite=True)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench_data_source.py")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Bundle directory (results.nc + results.zarr/ + sidecars). "
            "If omitted, builds a transient simulated bundle (smoke mode)."
        ),
    )
    parser.add_argument("--n-loads", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.bundle_dir is None:
        # Smoke mode: build a transient bundle so the script can be run
        # without setup. Real numbers should come from a real bundle.
        import tempfile

        tmp = tempfile.mkdtemp(prefix="bench_data_source_")
        os.environ.setdefault(
            "BENCH_BUNDLE_NOTE",
            "Smoke mode: numbers reflect a tiny simulated session, NOT a "
            "production-scale long session. Pass --bundle-dir to bench "
            "real data.",
        )
        bundle_dir = _materialise_simulated_bundle(Path(tmp))
    else:
        bundle_dir = args.bundle_dir

    print(f"Bundle: {bundle_dir}")
    if "BENCH_BUNDLE_NOTE" in os.environ:
        print(f"NOTE: {os.environ['BENCH_BUNDLE_NOTE']}")
    print()

    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.visualization.interactive.data_source import (
        InMemoryDecoderDataSource,
    )
    from non_local_detector.visualization.interactive.data_source_zarr_direct import (
        ZarrDirectDecoderDataSource,
    )
    from non_local_detector.visualization.interactive.view_models.base import (
        RunBundle,
    )

    # Build both data sources from the same bundle.
    results = _DetectorBase.load_results(str(bundle_dir / "results.nc"))
    detector = _DetectorBase.load_model(str(bundle_dir / "model.pkl"))
    spikes_npz = np.load(str(bundle_dir / "spikes.npz"), allow_pickle=True)
    spike_times = list(spikes_npz["spike_times"])
    import pandas as pd

    pos_df = pd.read_parquet(str(bundle_dir / "position.parquet"))
    bundle = RunBundle(
        results=results,
        detector=detector,
        spike_times=spike_times,
        position_time=pos_df.index.to_numpy(),
        position=pos_df["position"].to_numpy(),
        speed=pos_df["speed"].to_numpy() if "speed" in pos_df else None,
    )
    in_mem = InMemoryDecoderDataSource.from_single(bundle)
    direct = ZarrDirectDecoderDataSource.for_directory("default", bundle_dir)

    rng = np.random.default_rng(args.seed)
    time_grid = in_mem.time

    def _slice_for(width: float):
        def _factory():
            t_center = float(rng.uniform(time_grid[0], time_grid[-1]))
            return in_mem.window_indices(t_center, width)

        return _factory

    rows: list[dict[str, float]] = []
    for width in (1.0, 10.0, 30.0):
        sl_factory = _slice_for(width)
        rows.append(
            _bench_one(
                f"in_memory  load_posterior  ({width:>4.0f}s)",
                in_mem.load_posterior,
                args.n_loads,
                sl_factory,
            )
        )
        rows.append(
            _bench_one(
                f"zarr_direct load_posterior  ({width:>4.0f}s)",
                direct.load_posterior,
                args.n_loads,
                sl_factory,
            )
        )
        rows.append(
            _bench_one(
                f"in_memory  load_likelihood ({width:>4.0f}s)",
                in_mem.load_likelihood,
                args.n_loads,
                sl_factory,
            )
        )
        rows.append(
            _bench_one(
                f"zarr_direct load_likelihood ({width:>4.0f}s)",
                direct.load_likelihood,
                args.n_loads,
                sl_factory,
            )
        )
        rows.append(
            _bench_one(
                f"in_memory  load_state_prob ({width:>4.0f}s)",
                in_mem.load_state_probabilities,
                args.n_loads,
                sl_factory,
            )
        )
        rows.append(
            _bench_one(
                f"zarr_direct load_state_prob ({width:>4.0f}s)",
                direct.load_state_probabilities,
                args.n_loads,
                sl_factory,
            )
        )

    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
