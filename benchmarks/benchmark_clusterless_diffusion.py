"""Compare warm clusterless diffusion and KDE predictions on a large 2D grid.

Defaults preserve the former slow test's workload. Reduce the grid and spike
counts for a smoke run. Timings are measurements, with no speed threshold.
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_kde_distance import measure
from compare_prediction_modes import provenance

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_diffusion, clusterless_kde


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-side", type=int, default=160)
    parser.add_argument("--position-samples", type=int, default=20_000)
    parser.add_argument("--electrodes", type=int, default=8)
    parser.add_argument("--encoding-spikes", type=int, default=8_000)
    parser.add_argument("--decoding-spikes", type=int, default=300)
    parser.add_argument("--time-bins", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(vars(args).values()) < 1:
        parser.error("workload sizes must be positive")
    if args.position_samples < 2:
        parser.error("--position-samples must be at least 2")
    if args.repeats < 5:
        parser.error("--repeats must be at least 5")

    rng = np.random.default_rng(0)
    edges = np.arange(args.time_bins + 1) * 0.02
    position_time = np.linspace(0.0, edges[-1], args.position_samples)
    position = np.column_stack(
        [
            rng.uniform(0, 10, args.position_samples),
            rng.uniform(-3, 3, args.position_samples),
        ]
    )
    environment = Environment(
        environment_name="bench_grid",
        position_range=[(0, 10), (-3, 3)],
        place_bin_size=20.0 / args.grid_side,
    ).fit_place_grid(position=position, infer_track_interior=True)
    encoding_times = [
        np.sort(rng.uniform(0.0, edges[-1], args.encoding_spikes))
        for _ in range(args.electrodes)
    ]
    encoding_features = [
        rng.standard_normal((args.encoding_spikes, 2)).astype(np.float32)
        for _ in range(args.electrodes)
    ]
    decoding_times = [
        jnp.asarray(np.sort(rng.uniform(0.0, edges[-1], args.decoding_spikes)))
        for _ in range(args.electrodes)
    ]
    decoding_features = [
        jnp.asarray(rng.standard_normal((args.decoding_spikes, 2)).astype(np.float32))
        for _ in range(args.electrodes)
    ]
    interior = environment.is_track_interior_.ravel()
    report = {
        **provenance(__file__),
        "jax": jax.__version__,
        "workload": vars(args),
        "interior_bins": int(interior.sum()),
        "models": {},
    }
    for name, model in (("kde", clusterless_kde), ("diffusion", clusterless_diffusion)):
        fit = getattr(model, f"fit_clusterless_{name}_encoding_model")
        predict = getattr(model, f"predict_clusterless_{name}_log_likelihood")
        encoding = fit(
            position_time,
            position,
            encoding_times,
            encoding_features,
            environment,
            position_std=2.0,
            waveform_std=24.0,
            block_size=10_000,
            disable_progress_bar=True,
        )

        def prediction(_predict=predict, _encoding=encoding):
            return _predict(
                position_time,
                position,
                decoding_times,
                decoding_features,
                time_edges=edges,
                is_local=False,
                **_encoding,
            )

        likelihood, timing = measure(prediction, args.repeats)
        likelihood = np.asarray(likelihood)
        assert likelihood.shape == (args.time_bins, interior.sum())
        assert np.all(np.isfinite(likelihood))
        # Check one explicit silent bin, even when the timed workload has no silence.
        silent = np.asarray(
            predict(
                position_time,
                position,
                [np.array([]) for _ in range(args.electrodes)],
                [np.empty((0, 2)) for _ in range(args.electrodes)],
                time_edges=edges,
                is_local=False,
                row_slice=slice(0, 1),
                **encoding,
            )
        )
        rates = np.asarray(encoding["summed_ground_process_intensity"])
        if rates.size != interior.sum():
            rates = rates[interior]
        expected = -(edges[1] - edges[0]) * rates[None, :]
        assert silent.shape == expected.shape
        assert np.all(np.isfinite(silent))
        np.testing.assert_allclose(silent, expected, rtol=2e-6, atol=1e-6)
        report["models"][name] = {
            **timing,
            "dtype": str(likelihood.dtype),
            "silent_reference_rows": 1,
        }
    report["speedup"] = (
        report["models"]["kde"]["warm_ms"]["median"]
        / report["models"]["diffusion"]["warm_ms"]["median"]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
