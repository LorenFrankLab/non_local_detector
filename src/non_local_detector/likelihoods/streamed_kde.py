"""Opt-in prototypes for the existing linear KDE, with tiled kernel workspace.

No production registry uses these helpers by default. Samples and weights remain
the original model leaves. Normalization uses the full weight sum; intensity
floors apply only after all sample contributions have been accumulated. Tile
sizes bound forward kernel workspace, not resident inputs, returned arrays, or
JAX reverse-mode storage. Local callers concatenate position and marks before
calling ``_sample_tiled_density`` so their kernel exponentiation stays combined.
"""

from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.common import (
    LOG_EPS,
    _log_kernel_matrix,
    as_std_array,
    deterministic_row_add,
    safe_log,
)


def _positive_tile(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _points(value, name):
    value = jnp.asarray(value)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"{name} must be a sample vector or matrix")
    return value


def _weights(value, n_samples):
    value = jnp.ones(n_samples) if value is None else jnp.asarray(value)
    if value.shape != (n_samples,):
        raise ValueError("weights must contain one value per encoding sample")
    return value


def _std(value, n_dims, name):
    value = as_std_array(value, n_dims)
    if value.shape != (n_dims,):
        raise ValueError(f"{name} must be scalar or contain one value per dimension")
    return value


@partial(
    jax.jit, static_argnames=("sample_tile_size", "eval_tile_size", "pad_evaluation")
)
def _sample_density_core(
    points,
    samples,
    std,
    weights,
    sample_tile_size,
    eval_tile_size,
    pad_evaluation=False,
):
    n_points, n_samples = points.shape[0], samples.shape[0]
    dtype = jnp.result_type(points, samples, std, weights, jnp.zeros(()))
    if not n_points:
        return jnp.zeros(0, dtype=dtype)
    # Pad only evaluation queries, never encoding support or its denominator.
    # Using the same tile shape for the final queries avoids a float32 nested-
    # JIT transpose losing the last row's gradient on some CPU runtimes.
    padded_points = (
        n_points + (-n_points) % eval_tile_size if pad_evaluation else n_points
    )
    points = jnp.pad(points, ((0, padded_points - n_points), (0, 0)), mode="edge")
    output = jnp.zeros(padded_points, dtype=dtype)
    weight_total = jnp.sum(weights)
    safe_total = jnp.where(weight_total > 0, weight_total, 1.0)
    full_samples, sample_tail = divmod(n_samples, sample_tile_size)

    def evaluate(point_tile):
        def add_sample_tile(number, numerator):
            first = number * sample_tile_size
            sample_tile = jax.lax.dynamic_slice(
                samples, (first, 0), (sample_tile_size, samples.shape[1])
            )
            weight_tile = jax.lax.dynamic_slice(weights, (first,), (sample_tile_size,))
            kernel = jnp.exp(_log_kernel_matrix(point_tile, sample_tile, std))
            contribution = jnp.matmul(
                weight_tile, kernel, precision=jax.lax.Precision.HIGHEST
            )
            return numerator + contribution

        numerator = (
            jax.lax.fori_loop(
                0,
                full_samples,
                add_sample_tile,
                jnp.zeros(point_tile.shape[0], dtype=dtype),
            )
            if full_samples
            else jnp.zeros(point_tile.shape[0], dtype=dtype)
        )
        if sample_tail:
            first = full_samples * sample_tile_size
            kernel = jnp.exp(_log_kernel_matrix(point_tile, samples[first:], std))
            numerator = numerator + jnp.matmul(
                weights[first:], kernel, precision=jax.lax.Precision.HIGHEST
            )
        return jnp.where(weight_total > 0, numerator / safe_total, 0.0)

    full_points, point_tail = divmod(padded_points, eval_tile_size)
    if full_points:
        full_point_inputs = points[: full_points * eval_tile_size]

        def update_points(number, result):
            first = number * eval_tile_size
            point_tile = jax.lax.dynamic_slice(
                full_point_inputs, (first, 0), (eval_tile_size, points.shape[1])
            )
            return jax.lax.dynamic_update_slice(result, evaluate(point_tile), (first,))

        output = jax.lax.fori_loop(0, full_points, update_points, output)
    if point_tail:
        first = full_points * eval_tile_size
        output = jnp.concatenate((output[:first], evaluate(points[first:])))
    return output[:n_points]


def _sample_tiled_density(
    eval_points, samples, std, weights=None, *, sample_tile_size, eval_tile_size
):
    """Plain linear density with exact sample support and no floors.

    Use combined position/mark columns for a local joint density. Scalars expand
    bandwidths and sample vectors expand to columns as in ``KDEModel.predict``.
    Nonfinite caller values retain the existing linear KDE's IEEE behavior;
    only shape and positive-integer tile configuration are validated here.
    """
    points, samples = _points(eval_points, "eval_points"), _points(samples, "samples")
    if points.shape[1] != samples.shape[1]:
        raise ValueError(
            "evaluation and encoding samples must have matching dimensions"
        )
    std = _std(std, samples.shape[1], "std")
    weights = _weights(weights, samples.shape[0])
    sample_tile_size = _positive_tile(sample_tile_size, "sample_tile_size")
    eval_tile_size = _positive_tile(eval_tile_size, "eval_tile_size")
    if any(
        isinstance(value, jax.core.Tracer) for value in (points, samples, std, weights)
    ):
        return _sample_density_core(
            points,
            samples,
            std,
            weights,
            sample_tile_size,
            eval_tile_size,
            pad_evaluation=True,
        )
    first = points.shape[0] // eval_tile_size * eval_tile_size
    if 0 < first < points.shape[0]:
        # Preserve native kernel shapes and rounding. Transformed calls above
        # use fixed query tiles to avoid the nested-JIT tail transpose defect.
        prefix = _sample_density_core(
            points[:first], samples, std, weights, sample_tile_size, eval_tile_size
        )
        tail = jax.checkpoint(_sample_density_core, static_argnums=(4, 5))(
            points[first:], samples, std, weights, sample_tile_size, eval_tile_size
        )
        return jnp.concatenate((prefix, tail))
    return _sample_density_core(
        points, samples, std, weights, sample_tile_size, eval_tile_size
    )


def _joint_inputs(
    decoding_features,
    encoding_features,
    encoding_positions,
    place_bin_centers,
    waveform_stds,
    position_std,
    occupancy,
    mean_rate,
    encoding_weights,
):
    decoded = _points(decoding_features, "decoding_features")
    encoded = _points(encoding_features, "encoding_features")
    positions = _points(encoding_positions, "encoding_positions")
    centers = _points(place_bin_centers, "place_bin_centers")
    if decoded.shape[1] != encoded.shape[1]:
        raise ValueError("encoding and decoding mark dimensions must match")
    if positions.shape[0] != encoded.shape[0]:
        raise ValueError(
            "encoding positions and marks must have matching sample counts"
        )
    if centers.shape[1] != positions.shape[1]:
        raise ValueError("encoding positions and centers must have matching dimensions")
    occupancy = jnp.asarray(occupancy)
    if occupancy.shape != (centers.shape[0],):
        raise ValueError("occupancy must contain one value per spatial center")
    return (
        decoded,
        encoded,
        positions,
        centers,
        _std(waveform_stds, encoded.shape[1], "waveform_stds"),
        _std(position_std, positions.shape[1], "position_std"),
        occupancy,
        jnp.asarray(mean_rate),
        _weights(encoding_weights, encoded.shape[0]),
    )


def _joint_numerator(
    decoded,
    encoded,
    positions,
    centers,
    waveform_std,
    position_std,
    weights,
    encoding_tile_size,
    position_tile_size,
):
    """Compute one decoded block; retain only its numerator and kernel tiles."""
    dtype = jnp.result_type(
        decoded,
        encoded,
        positions,
        centers,
        waveform_std,
        position_std,
        weights,
        jnp.zeros(()),
    )
    numerator = jnp.zeros((decoded.shape[0], centers.shape[0]), dtype=dtype)
    full_positions, position_tail = divmod(centers.shape[0], position_tile_size)

    def add_encoding(result, marks, position_samples, sample_weights):
        # One mark tile per encoding tile, reused across the spatial tiles in
        # `centers`. `_joint_core` passes one spatial tile per call, so it is
        # recomputed for each spatial tile.
        mark_kernel = jnp.exp(_log_kernel_matrix(decoded, marks, waveform_std))

        def contribution(point_tile):
            position_kernel = jnp.exp(
                _log_kernel_matrix(point_tile, position_samples, position_std)
            )
            return jnp.matmul(
                mark_kernel.T,
                sample_weights[:, None] * position_kernel,
                precision=jax.lax.Precision.HIGHEST,
            )

        if full_positions:

            def add_positions(number, sums):
                first = number * position_tile_size
                point_tile = jax.lax.dynamic_slice(
                    centers, (first, 0), (position_tile_size, centers.shape[1])
                )
                old = jax.lax.dynamic_slice(
                    sums, (0, first), (decoded.shape[0], position_tile_size)
                )
                return jax.lax.dynamic_update_slice(
                    sums, old + contribution(point_tile), (0, first)
                )

            result = jax.lax.fori_loop(0, full_positions, add_positions, result)
        if position_tail:
            first = full_positions * position_tile_size
            result = jax.lax.dynamic_update_slice(
                result, result[:, first:] + contribution(centers[first:]), (0, first)
            )
        return result

    full_encoding, encoding_tail = divmod(encoded.shape[0], encoding_tile_size)
    if full_encoding:

        def add_encoding_block(number, result):
            first = number * encoding_tile_size
            return add_encoding(
                result,
                jax.lax.dynamic_slice(
                    encoded, (first, 0), (encoding_tile_size, encoded.shape[1])
                ),
                jax.lax.dynamic_slice(
                    positions, (first, 0), (encoding_tile_size, positions.shape[1])
                ),
                jax.lax.dynamic_slice(weights, (first,), (encoding_tile_size,)),
            )

        numerator = jax.lax.fori_loop(0, full_encoding, add_encoding_block, numerator)
    if encoding_tail:
        first = full_encoding * encoding_tile_size
        numerator = add_encoding(
            numerator, encoded[first:], positions[first:], weights[first:]
        )
    return numerator


@partial(
    jax.jit,
    static_argnames=(
        "encoding_tile_size",
        "position_tile_size",
        "decoding_tile_size",
        "n_rows",
    ),
)
def _joint_core(
    decoded,
    encoded,
    positions,
    centers,
    waveform_std,
    position_std,
    occupancy,
    mean_rate,
    weights,
    encoding_tile_size,
    position_tile_size,
    decoding_tile_size,
    row_indices=None,
    n_rows=0,
):
    n_decoded, n_positions = decoded.shape[0], centers.shape[0]
    dtype = jnp.result_type(
        decoded,
        encoded,
        positions,
        centers,
        waveform_std,
        position_std,
        occupancy,
        mean_rate,
        weights,
        jnp.zeros(()),
    )
    output = jnp.zeros(
        (n_decoded if row_indices is None else n_rows, n_positions), dtype=dtype
    )
    weight_total = jnp.sum(weights)
    safe_total = jnp.where(weight_total > 0, weight_total, 1.0)

    def score_positions(result, center_tile, occupancy_tile, column_start):
        def finish(block):
            # Only this spatial tile enters the numerator. Row-returning calls
            # never retain a decoded-block by FULL-spatial-grid intermediate.
            numerator = _joint_numerator(
                block,
                encoded,
                positions,
                center_tile,
                waveform_std,
                position_std,
                weights,
                encoding_tile_size,
                position_tile_size,
            )
            density = jnp.where(weight_total > 0, numerator / safe_total, 0.0)
            intensity = safe_log(
                mean_rate
                * jnp.where(
                    occupancy_tile > 0.0,
                    density / jnp.where(occupancy_tile > 0.0, occupancy_tile, 1.0),
                    0.0,
                )
            )
            return jnp.clip(intensity, min=LOG_EPS, max=None)

        full_decoded, decoded_tail = divmod(n_decoded, decoding_tile_size)
        if full_decoded:

            def update_decoded(number, sums):
                first = number * decoding_tile_size
                block = jax.lax.dynamic_slice(
                    decoded, (first, 0), (decoding_tile_size, decoded.shape[1])
                )
                finished = finish(block)
                if row_indices is None:
                    return jax.lax.dynamic_update_slice(
                        sums, finished, (first, column_start)
                    )
                rows = jax.lax.dynamic_slice(
                    row_indices, (first,), (decoding_tile_size,)
                )
                return deterministic_row_add(sums, finished, rows, column_start)

            result = jax.lax.fori_loop(0, full_decoded, update_decoded, result)
        if decoded_tail:
            first = full_decoded * decoding_tile_size
            finished = finish(decoded[first:])
            result = (
                jax.lax.dynamic_update_slice(result, finished, (first, column_start))
                if row_indices is None
                else deterministic_row_add(
                    result, finished, row_indices[first:], column_start
                )
            )
        return result

    full_positions, position_tail = divmod(n_positions, position_tile_size)
    if full_positions:

        def update_positions(number, result):
            first = number * position_tile_size
            return score_positions(
                result,
                jax.lax.dynamic_slice(
                    centers, (first, 0), (position_tile_size, centers.shape[1])
                ),
                jax.lax.dynamic_slice(occupancy, (first,), (position_tile_size,)),
                first,
            )

        output = jax.lax.fori_loop(0, full_positions, update_positions, output)
    if position_tail:
        first = full_positions * position_tile_size
        output = score_positions(output, centers[first:], occupancy[first:], first)
    return output


def _streamed_joint_mark_log_intensity(
    decoding_features,
    encoding_features,
    encoding_positions,
    place_bin_centers,
    waveform_stds,
    position_std,
    occupancy,
    mean_rate,
    encoding_weights=None,
    *,
    encoding_tile_size,
    position_tile_size,
    decoding_tile_size,
):
    """Finished nonlocal linear mark intensity, shape (selected marks, positions).

    ``mean_rate`` has the same units as the existing low-level marked estimator:
    native callers pass Hz multiplied by ``RATE_REFERENCE_SECONDS``. Geometry
    is the caller's existing Euclidean/projected coordinates, not graph distance.
    The returned matrix remains mark-count sized; native row callers should use
    ``_streamed_joint_mark_row_sums`` instead.
    """
    return _joint_core(
        *_joint_inputs(
            decoding_features,
            encoding_features,
            encoding_positions,
            place_bin_centers,
            waveform_stds,
            position_std,
            occupancy,
            mean_rate,
            encoding_weights,
        ),
        _positive_tile(encoding_tile_size, "encoding_tile_size"),
        _positive_tile(position_tile_size, "position_tile_size"),
        _positive_tile(decoding_tile_size, "decoding_tile_size"),
    )


def _streamed_joint_mark_row_sums(
    decoding_features,
    encoding_features,
    encoding_positions,
    place_bin_centers,
    waveform_stds,
    position_std,
    occupancy,
    mean_rate,
    encoding_weights=None,
    *,
    row_indices,
    n_rows,
    encoding_tile_size,
    position_tile_size,
    decoding_tile_size,
):
    """Return ordered event log totals per row, with no ground-process term.

    ``row_indices`` is original-order host metadata, one integer per selected
    mark. Invalid IDs are dropped without negative wrapping. Only one decoded
    block's finished intensities are retained before row accumulation.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, Integral) or n_rows < 0:
        raise ValueError("n_rows must be a nonnegative integer")
    n_rows = int(n_rows)
    inputs = _joint_inputs(
        decoding_features,
        encoding_features,
        encoding_positions,
        place_bin_centers,
        waveform_stds,
        position_std,
        occupancy,
        mean_rate,
        encoding_weights,
    )
    indices = np.asarray(row_indices)
    if indices.shape != (inputs[0].shape[0],) or indices.dtype.kind not in "iu":
        raise ValueError("row_indices must contain one integer per decoding mark")
    valid = (indices >= 0) & (indices < n_rows)
    normalized = np.full(indices.shape, -1, dtype=np.intp)
    normalized[indices >= n_rows] = n_rows
    normalized[valid] = indices[valid]
    return _joint_core(
        *inputs,
        _positive_tile(encoding_tile_size, "encoding_tile_size"),
        _positive_tile(position_tile_size, "position_tile_size"),
        _positive_tile(decoding_tile_size, "decoding_tile_size"),
        jnp.asarray(normalized),
        n_rows,
    )
