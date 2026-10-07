"""Bounded long-encoding comparison against independent float64 linear KDEs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.stats import norm

import non_local_detector
import non_local_detector.likelihoods.streamed_kde as streamed
from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
    kde_distance,
)
from non_local_detector.likelihoods.common import EPS, LOG_EPS, kde, safe_log


def sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def inputs():
    rng = np.random.default_rng(7431)
    track_time = np.linspace(0.0, 3600.0, 108_001)
    position = np.column_stack(
        (
            90 + 60 * np.sin(track_time / 31.0),
            90 + 60 * np.cos(track_time / 47.0),
        )
    )
    encoding_time = np.sort(rng.uniform(0.0, 3600.0, 72_000))
    encoding_position = np.column_stack(
        [np.interp(encoding_time, track_time, position[:, dim]) for dim in range(2)]
    )
    marks = rng.normal(80.0, 20.0, (72_000, 4))
    decoded_marks = rng.normal(80.0, 20.0, (7, 4))
    tracking_weights = rng.uniform(0.2, 1.7, len(track_time))
    tracking_weights[::19] = 0.0
    encoding_weights = rng.uniform(0.3, 1.9, len(encoding_time))
    encoding_weights[::13] = 0.0
    centers = np.array([[90.0, 90.0], [135.0, 100.0], [20.0, 20.0]])
    local_time = np.array([21.0, 227.0, 512.0, 1075.0, 1741.0, 2429.0, 3553.0])
    local_position = np.column_stack(
        [np.interp(local_time, track_time, position[:, dim]) for dim in range(2)]
    )
    return {
        "tracking_time": track_time,
        "tracking_position": position,
        "encoding_time": encoding_time,
        "encoding_position": encoding_position,
        "encoding_marks": marks,
        "decoded_marks": decoded_marks,
        "tracking_weights": tracking_weights,
        "encoding_weights": encoding_weights,
        "centers": centers,
        "local_time": local_time,
        "local_position": local_position,
        "position_std": np.array([6.0, 6.0]),
        "waveform_std": np.full(4, 24.0),
    }


def oracle_kernel(points, samples, std):
    """One E×query matrix; never construct an E×query×dimension tensor."""
    points, samples, std = [np.asarray(x, np.float64) for x in (points, samples, std)]
    kernel = np.ones((len(samples), len(points)), np.float64)
    for dim in range(samples.shape[1]):
        kernel *= norm.pdf(
            points[None, :, dim], loc=samples[:, None, dim], scale=std[dim]
        )
    return kernel


def oracle_density(points, samples, std, weights):
    weights = np.asarray(weights, np.float64)
    return weights @ oracle_kernel(points, samples, std) / weights.sum()


def comparison(actual, reference, baseline=None):
    actual = np.asarray(actual)
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-5)
    nonzero = reference != 0
    relative = np.abs(actual[nonzero] - reference[nonzero]) / np.abs(reference[nonzero])
    # Density values can be much smaller than the absolute likelihood tolerance.
    # Also compare their ratio using the same controls to make that check useful.
    diagnostic_pass = bool(
        np.allclose(actual[nonzero] / reference[nonzero], 1.0, rtol=1e-6, atol=1e-5)
    )
    result = {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_absolute_error": float(np.max(abs(actual - reference))),
        "max_relative_error": float(relative.max(initial=0)),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "actual_sha256": sha(actual),
        "reference_sha256": sha(reference),
        "applicable_controls": {"rtol": 1e-6, "atol": 1e-5},
        "ratio_diagnostic_same_controls_pass": diagnostic_pass,
    }
    if baseline is not None:
        baseline = np.asarray(baseline)
        np.testing.assert_allclose(actual, baseline, rtol=1e-6, atol=1e-5)
        result.update(
            {
                "baseline_max_absolute_error": float(np.max(abs(baseline - reference))),
                "baseline_max_relative_error": float(
                    np.max(abs(baseline[nonzero] / reference[nonzero] - 1))
                ),
                "streamed_vs_baseline_max_absolute_error": float(
                    np.max(abs(actual - baseline))
                ),
                "baseline_sha256": sha(baseline),
            }
        )
        if np.all(actual > 0) and np.all(baseline > 0):
            result["streamed_vs_baseline_log_density_max_absolute_error"] = float(
                np.max(abs(np.log(actual) - np.log(baseline)))
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    expected_backend = "gpu" if args.platform == "gpu" else "cpu"
    if jax.default_backend() != expected_backend:
        raise RuntimeError(f"Expected {expected_backend}, got {jax.default_backend()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = inputs()
    np.savez(args.output_dir / "inputs.npz", **data)
    package = Path(non_local_detector.__file__).parent
    sources = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
        if "tests" not in path.relative_to(package).parts
    }
    report = {
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "matmul_precision": jax.config.jax_default_matmul_precision,
        },
        "source_root": str(package),
        "source_sha256": sources,
        "runtime_aggregate_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {name: sha(array) for name, array in data.items()},
        "input_artifact_sha256": hashlib.sha256(
            (args.output_dir / "inputs.npz").read_bytes()
        ).hexdigest(),
        "dimensions": {
            "tracking_samples": 108_001,
            "encoding_events": 72_000,
            "position_dims": 2,
            "mark_dims": 4,
            "centers": 3,
            "decoded_marks": 7,
        },
        "tiles": {"encoding": 1024, "position": 2, "decoding": 3},
        "encoding_tile_counts": {"tracking": 106, "events": 71},
        "largest_oracle_matrix_bytes": 108_001 * 7 * 8,
        "modes": {},
    }
    previous_x64 = jax.config.x64_enabled
    try:
        for enabled, dtype in [(False, np.float32), (True, np.float64)]:
            jax.config.update("jax_enable_x64", enabled)
            values = {name: array.astype(dtype) for name, array in data.items()}
            mode = {
                "global_x64": enabled,
                "input_dtype": str(np.dtype(dtype)),
                "typed_input_sha256": {
                    name: sha(array) for name, array in values.items()
                },
            }
            occupancy_reference = oracle_density(
                values["centers"],
                values["tracking_position"],
                values["position_std"],
                values["tracking_weights"],
            )
            occupancy = streamed._sample_tiled_density(
                values["centers"],
                values["tracking_position"],
                values["position_std"],
                values["tracking_weights"],
                sample_tile_size=1024,
                eval_tile_size=2,
            )
            occupancy_baseline = kde(
                jnp.asarray(values["centers"]),
                jnp.asarray(values["tracking_position"]),
                jnp.asarray(values["position_std"]),
                jnp.asarray(values["tracking_weights"]),
            )
            mode["occupancy_density"] = comparison(
                occupancy, occupancy_reference, occupancy_baseline
            )
            plain_reference = oracle_density(
                values["centers"],
                values["encoding_position"],
                values["position_std"],
                values["encoding_weights"],
            )
            plain = streamed._sample_tiled_density(
                values["centers"],
                values["encoding_position"],
                values["position_std"],
                values["encoding_weights"],
                sample_tile_size=1024,
                eval_tile_size=2,
            )
            plain_baseline = kde(
                jnp.asarray(values["centers"]),
                jnp.asarray(values["encoding_position"]),
                jnp.asarray(values["position_std"]),
                jnp.asarray(values["encoding_weights"]),
            )
            mode["encoding_position_density"] = comparison(
                plain, plain_reference, plain_baseline
            )
            samples = np.concatenate(
                (values["encoding_position"], values["encoding_marks"]), axis=1
            )
            points = np.concatenate(
                (values["local_position"], values["decoded_marks"]), axis=1
            )
            std = np.concatenate((values["position_std"], values["waveform_std"]))
            local_reference = oracle_density(
                points, samples, std, values["encoding_weights"]
            )
            local = streamed._sample_tiled_density(
                points,
                samples,
                std,
                values["encoding_weights"],
                sample_tile_size=1024,
                eval_tile_size=2,
            )
            local_baseline = kde(
                jnp.asarray(points),
                jnp.asarray(samples),
                jnp.asarray(std),
                jnp.asarray(values["encoding_weights"]),
            )
            mode["local_combined_density"] = comparison(
                local, local_reference, local_baseline
            )
            local_occupancy_reference = oracle_density(
                values["local_position"],
                values["tracking_position"],
                values["position_std"],
                values["tracking_weights"],
            )
            local_occupancy = streamed._sample_tiled_density(
                values["local_position"],
                values["tracking_position"],
                values["position_std"],
                values["tracking_weights"],
                sample_tile_size=1024,
                eval_tile_size=2,
            )
            local_occupancy_baseline = kde(
                jnp.asarray(values["local_position"]),
                jnp.asarray(values["tracking_position"]),
                jnp.asarray(values["position_std"]),
                jnp.asarray(values["tracking_weights"]),
            )
            scale = np.asarray(0.04, dtype=dtype)
            local_log_reference = np.log(
                np.maximum(
                    scale.item() * local_reference / local_occupancy_reference, EPS
                )
            )
            local_log = safe_log(scale * local / local_occupancy)
            local_log_baseline = safe_log(
                scale * local_baseline / local_occupancy_baseline
            )
            mode["finished_local_log_intensity"] = comparison(
                local_log, local_log_reference, local_log_baseline
            )

            # Use the same rounded occupancy inputs for both implementations;
            # the zero column independently exercises the final intensity floor.
            intensity_occupancy = occupancy_reference.astype(dtype)
            intensity_occupancy[1] = 0.0
            position_kernel = oracle_kernel(
                values["centers"], values["encoding_position"], values["position_std"]
            )
            mark_kernel = oracle_kernel(
                values["decoded_marks"],
                values["encoding_marks"],
                values["waveform_std"],
            )
            weights64 = values["encoding_weights"].astype(np.float64)
            density = (
                mark_kernel.T @ (weights64[:, None] * position_kernel) / weights64.sum()
            )
            intensity = np.zeros_like(density)
            np.divide(
                density,
                intensity_occupancy,
                out=intensity,
                where=intensity_occupancy > 0,
            )
            intensity *= np.asarray(0.04, dtype=dtype).item()
            reference = np.log(np.maximum(intensity, EPS))
            joint = streamed._streamed_joint_mark_log_intensity(
                values["decoded_marks"],
                values["encoding_marks"],
                values["encoding_position"],
                values["centers"],
                values["waveform_std"],
                values["position_std"],
                intensity_occupancy,
                np.asarray(0.04, dtype=dtype),
                values["encoding_weights"],
                encoding_tile_size=1024,
                position_tile_size=2,
                decoding_tile_size=3,
            )
            joint_baseline = jnp.clip(
                estimate_log_joint_mark_intensity(
                    jnp.asarray(values["decoded_marks"]),
                    jnp.asarray(values["encoding_marks"]),
                    jnp.asarray(values["waveform_std"]),
                    jnp.asarray(intensity_occupancy),
                    scale,
                    kde_distance(
                        jnp.asarray(values["centers"]),
                        jnp.asarray(values["encoding_position"]),
                        jnp.asarray(values["position_std"]),
                    ),
                    jnp.asarray(values["encoding_weights"]),
                ),
                min=LOG_EPS,
            )
            mode["finished_joint_log_intensity"] = comparison(
                joint, reference, joint_baseline
            )
            np.savez(
                args.output_dir / f"outputs-{np.dtype(dtype)}.npz",
                occupancy=np.asarray(occupancy),
                occupancy_reference=occupancy_reference,
                plain=np.asarray(plain),
                plain_reference=plain_reference,
                local=np.asarray(local),
                local_reference=local_reference,
                joint=np.asarray(joint),
                joint_reference=reference,
                occupancy_baseline=np.asarray(occupancy_baseline),
                plain_baseline=np.asarray(plain_baseline),
                local_baseline=np.asarray(local_baseline),
                joint_baseline=np.asarray(joint_baseline),
                local_log=np.asarray(local_log),
                local_log_reference=local_log_reference,
                local_log_baseline=np.asarray(local_log_baseline),
            )
            report["modes"][str(np.dtype(dtype))] = mode
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runtime_aggregate_sha256": report["runtime_aggregate_sha256"],
                "runtime": report["runtime"],
                "modes": {
                    dtype: {
                        name: result
                        for name, result in mode.items()
                        if isinstance(result, dict) and "max_absolute_error" in result
                    }
                    for dtype, mode in report["modes"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
