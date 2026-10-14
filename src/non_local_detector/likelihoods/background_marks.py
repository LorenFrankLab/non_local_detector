"""Normalized shared mark references and coherent low-rate contamination.

The reference is fitted once per detector from permitted encoding spikes. A
small *rate* multiplies this normalized density; its compensator is that rate,
irrespective of waveform units. No density receives a dimensionful flat floor.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from numpy.typing import ArrayLike

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    RATE_EPS_HZ,
    _log_kernel_matrix,
    _pad_rows,
    _padded_spike_count,
    _SpikeTimeOrder,
    as_std_array,
    decode_bin_centers,
    deterministic_row_sum,
    get_position_at_time,
    log_bin_duration_evidence,
    resolve_row_slice,
    select_spike_rows,
    select_spikes_in_rows,
    spike_row_ids,
    validate_population_lengths,
    validate_spike_feature_pair,
)
from non_local_detector.time_edges import _resolve_time_grid


@dataclass(frozen=True)
class BackgroundMarkModel:
    """A normalized diagonal Gaussian with training-derived location and scale."""

    location: np.ndarray
    scale: np.ndarray

    def standardize(
        self, marks: ArrayLike, *, pad_to: int | None = None
    ) -> jnp.ndarray:
        # Fitted statistics live on the host. Center/scale before casting to the
        # device dtype so an affine change of units does not lose additional
        # digits to subtraction in float32.
        standardized = (np.asarray(marks) - self.location) / self.scale
        if pad_to is not None:
            standardized = _pad_rows(standardized, pad_to)
        return jnp.asarray(standardized)

    def log_density(self, marks: ArrayLike) -> jnp.ndarray:
        z = self.standardize(marks)
        return (
            -0.5 * jnp.sum(z * z, axis=1)
            - jnp.sum(jnp.log(jnp.asarray(self.scale)))
            - 0.5 * z.shape[1] * np.log(2 * np.pi)
        )


def fit_background_mark_model(
    marks: ArrayLike, weights: ArrayLike, fallback_std: ArrayLike
) -> BackgroundMarkModel | None:
    """Fit a weighted mark reference without using excluded or decoding spikes.

    A zero empirical variance uses the configured mark-kernel scale, since a
    Dirac mass cannot serve as a density with respect to continuous marks.
    An electrode with no permitted marks has no identifiable location; retain
    None and reject any subsequently observed spike on that electrode.
    """
    marks, weights = np.asarray(marks), np.asarray(weights)
    keep = weights > 0
    if not np.any(keep):
        return None
    marks, weights = marks[keep], weights[keep]
    if not np.all(np.isfinite(marks)):
        raise ValidationError("Permitted background waveform features must be finite")
    weights = weights / weights.max()
    weights = weights / weights.sum()
    location = np.sum(weights[:, None] * marks, axis=0)
    variance = np.sum(weights[:, None] * (marks - location) ** 2, axis=0)
    fallback = np.broadcast_to(np.asarray(fallback_std), (marks.shape[1],))
    scale = np.where(variance > 0, np.sqrt(variance), fallback)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValidationError("Background mark scales must be finite and positive")
    return BackgroundMarkModel(location, scale)


def background_mark_log_density(
    model: BackgroundMarkModel | None, marks: ArrayLike, electrode: int
) -> jnp.ndarray:
    if model is None:
        if np.shape(marks)[0]:
            raise ValidationError(
                f"Electrode {electrode} has decoding spikes but no permitted training marks",
                hint="Fit with permitted training spikes on this electrode, or remove it consistently from encoding and decoding. A background mark location cannot be inferred from an empty encoding population.",
            )
        return jnp.zeros((0,))
    return model.log_density(marks)


def background_mark_row_terms(
    models: list[BackgroundMarkModel | None],
    spike_times: list[ArrayLike],
    marks: list[ArrayLike],
    time_edges: ArrayLike,
    row_slice: slice | None = None,
    spike_time_order: _SpikeTimeOrder | None = None,
) -> jnp.ndarray:
    """State-independent reference density contribution, preserving row ownership."""
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=marks,
        background_mark_models=models,
    )
    start, stop = resolve_row_slice(row_slice, len(time_edges) - 1)
    total = jnp.zeros(stop - start)
    order = spike_time_order or _SpikeTimeOrder()
    for electrode, (model, times, features) in enumerate(
        zip(models, spike_times, marks, strict=True)
    ):
        events = _prepare_background_events(
            model, times, features, time_edges, start, stop, order, electrode
        )
        total = total + order.memo(
            f"background_reference_rows:{start}:{stop}",
            (model, times, features, time_edges),
            lambda events=events: _background_reference_rows(
                events["log_density"],
                events["row_ids"],
                stop - start,
                indices_are_sorted=events["selection"].indices_are_sorted,
            ),
        )
    return total[:, None]


@jax.jit
def _reference_log_density(standardized, scale):
    # Retain the original rounding boundaries while compiling the mark scorer.
    quadratic = jax.lax.optimization_barrier(
        -0.5 * jnp.sum(standardized * standardized, axis=1)
    )
    density = jax.lax.optimization_barrier(quadratic - jnp.sum(jnp.log(scale)))
    return density - 0.5 * standardized.shape[1] * np.log(2 * np.pi)


@partial(jax.jit, static_argnames=("n_rows", "indices_are_sorted"))
def _background_reference_rows(log_density, row_ids, n_rows, *, indices_are_sorted):
    return deterministic_row_sum(
        jnp.where(row_ids < n_rows, log_density, 0),
        row_ids,
        n_rows,
        indices_are_sorted=indices_are_sorted,
    )


def _prepare_background_events(
    reference, times, features, edges, start, stop, order, electrode
):
    """Host selection and padded mark reference shared by every observation state.

    Cache identity includes the original paired arrays, fitted reference, and
    grid; the row range is part of the name. A prediction owns the cache, so
    in-place changes between predictions cannot retain stale preparation.
    """

    def build():
        selection = select_spikes_in_rows(
            times, start, stop, time_edges=edges, _spike_time_order=order
        )
        selected = select_spike_rows(features, selection)
        n_events = np.shape(selected)[0]
        if reference is None:
            background_mark_log_density(reference, selected, electrode)
            standardized = jnp.zeros((0, np.shape(features)[1]))
            log_density = jnp.zeros(0)
            padded_count = 0
        else:
            bucket = 1 << max(n_events - 1, 15).bit_length()
            padded_count = _padded_spike_count(n_events, bucket)
            standardized = reference.standardize(selected, pad_to=padded_count)
            log_density = _reference_log_density(
                standardized, jnp.asarray(reference.scale)
            )
        return {
            "selection": selection,
            "selected_features": selected,
            "features": jnp.asarray(_pad_rows(selected, padded_count)),
            "selected_times": select_spike_rows(times, selection),
            "standardized": standardized,
            "log_density": log_density,
            "row_ids": jnp.asarray(spike_row_ids(selection, padded_count)),
            "n_events": n_events,
        }

    return order.memo(
        f"background_events:{start}:{stop}", (reference, times, features, edges), build
    )


@partial(jax.jit, static_argnames=("sample_tile", "n_position_dims"))
def _kde_density_parts(
    points: ArrayLike,
    samples: ArrayLike,
    std: ArrayLike,
    weights: ArrayLike | None = None,
    sample_tile: int | None = None,
    *,
    occupancy_parts: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    n_position_dims: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Keep the kernel offset separate from normalized mass until cancellation."""
    points, samples = jnp.asarray(points), jnp.asarray(samples)
    n_samples = samples.shape[0]
    if not n_samples:
        return jnp.full(points.shape[0], -jnp.inf), jnp.zeros(points.shape[0])
    weights = jnp.ones(n_samples) if weights is None else jnp.asarray(weights)
    invalid = (
        jnp.any(jnp.isnan(samples))
        | jnp.any(jnp.isnan(weights))
        | jnp.any(jnp.isnan(jnp.asarray(std)))
    )
    weight_max = jnp.max(weights)
    weights = jnp.where(weights > 0, weights, 0) / jnp.where(
        weight_max > 0, weight_max, 1
    )
    total = jnp.sum(weights)
    std = as_std_array(std, points.shape[1])
    tile = min(sample_tile or n_samples, n_samples)
    n_tiles = (n_samples + tile - 1) // tile
    samples = jnp.pad(samples, ((0, n_tiles * tile - n_samples), (0, 0)))
    weights = jnp.pad(weights, ((0, n_tiles * tile - n_samples),))

    def add_tile(
        number: int, carry: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        first = number * tile
        sample_block = jax.lax.dynamic_slice_in_dim(samples, first, tile)
        weight_block = jax.lax.dynamic_slice_in_dim(weights, first, tile)
        if occupancy_parts is None:
            log_kernel = _log_kernel_matrix(points, sample_block, std)
        else:
            # Subtract spatial log density before adding mark normalizers.
            # Otherwise a joint log density near -20000 rounds a mark's
            # order-one normalizer away before the occupancy cancellation.
            log_position = (
                _log_kernel_matrix(
                    points[:, :n_position_dims],
                    sample_block[:, :n_position_dims],
                    std[:n_position_dims],
                )
                - occupancy_parts[0][None, :]
                - occupancy_parts[1][None, :]
            )
            log_mark = _log_kernel_matrix(
                points[:, n_position_dims:],
                sample_block[:, n_position_dims:],
                std[n_position_dims:],
            )
            log_kernel = log_position + log_mark
        log_kernel = jnp.where(weight_block[:, None] > 0, log_kernel, -jnp.inf)
        previous_max, previous_sum = carry
        next_max = jnp.maximum(previous_max, jnp.max(log_kernel, axis=0))
        safe_max = jnp.where(jnp.isneginf(next_max), 0, next_max)
        next_sum = previous_sum * jnp.exp(previous_max - safe_max) + jnp.matmul(
            weight_block,
            jnp.exp(log_kernel - safe_max),
            precision=jax.lax.Precision.HIGHEST,
        )
        return next_max, next_sum

    maximum, numerator = jax.lax.fori_loop(
        0,
        n_tiles,
        add_tile,
        (jnp.full(points.shape[0], -jnp.inf), jnp.zeros(points.shape[0])),
    )
    safe_numerator = jnp.where(numerator > 0, numerator, 1)
    safe_total = jnp.where(total > 0, total, 1)
    normalized_mass = jnp.where(
        numerator > 0, jnp.log(safe_numerator / safe_total), -jnp.inf
    )
    invalid = invalid | jnp.any(jnp.isnan(points), axis=1)
    return jnp.where(invalid, jnp.nan, maximum), jnp.where(
        invalid, jnp.nan, normalized_mass
    )


def _kde_log_density(
    points: ArrayLike,
    samples: ArrayLike,
    std: ArrayLike,
    weights: ArrayLike | None = None,
    sample_tile: int | None = None,
    *,
    log_occupancy: ArrayLike | None = None,
    occupancy_parts: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    n_position_dims: int = 0,
) -> jnp.ndarray:
    """Unfloored density; ratios should retain its two parts until division."""
    if occupancy_parts is None and log_occupancy is not None:
        occupancy_parts = (jnp.asarray(log_occupancy), jnp.zeros_like(log_occupancy))
    maximum, mass = _kde_density_parts(
        points,
        samples,
        std,
        weights,
        sample_tile,
        occupancy_parts=occupancy_parts,
        n_position_dims=n_position_dims,
    )
    return maximum + mass


def _kde_model_density_parts(
    model: Any,
    points: ArrayLike,
    sample_tile: int | None = None,
    eval_tile: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    points = jnp.asarray(points)
    if not points.shape[0]:
        return jnp.zeros(0), jnp.zeros(0)
    tile = min(eval_tile or model.block_size or points.shape[0], points.shape[0])
    parts = [
        _kde_density_parts(
            points[first : first + tile],
            model.samples_,
            model.std,
            model.weights_,
            sample_tile,
        )
        for first in range(0, points.shape[0], tile)
    ]
    return jnp.concatenate([part[0] for part in parts]), jnp.concatenate(
        [part[1] for part in parts]
    )


def _kde_model_log_density(
    model: Any,
    points: ArrayLike,
    sample_tile: int | None = None,
    eval_tile: int | None = None,
) -> jnp.ndarray:
    maximum, mass = _kde_model_density_parts(model, points, sample_tile, eval_tile)
    return maximum + mass


@partial(jax.jit, static_argnames=("sample_tile",))
def _kde_log_joint_nonlocal(
    positions: ArrayLike,
    features: ArrayLike,
    encoded_positions: ArrayLike,
    encoded_marks: ArrayLike,
    position_std: ArrayLike,
    waveform_std: ArrayLike,
    weights: ArrayLike,
    sample_tile: int | None,
    *,
    log_occupancy: ArrayLike | None = None,
    occupancy_parts: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    position_log_ratio: ArrayLike | None = None,
) -> jnp.ndarray:
    """Per-output stabilized GEMM with a logsumexp underflow fallback.

    Scaling independently per observed mark and evaluation position preserves
    batch invariance. The fallback covers a lack of overlapping mass between
    their respective maxima; summing tiles uses logaddexp without density floors.
    """
    invalid_source = (
        jnp.any(jnp.isnan(encoded_positions))
        | jnp.any(jnp.isnan(encoded_marks))
        | jnp.any(jnp.isnan(weights))
        | jnp.any(jnp.isnan(jnp.asarray(position_std)))
        | jnp.any(jnp.isnan(jnp.asarray(waveform_std)))
    )
    invalid = (
        invalid_source
        | jnp.any(jnp.isnan(features), axis=1)[:, None]
        | jnp.any(jnp.isnan(positions), axis=1)[None, :]
    )
    n_samples = encoded_positions.shape[0]
    if not n_samples:
        return jnp.where(
            invalid,
            jnp.nan,
            jnp.full((features.shape[0], positions.shape[0]), -jnp.inf),
        )
    tile = min(sample_tile or n_samples, n_samples)
    output = jnp.full((features.shape[0], positions.shape[0]), -jnp.inf)
    maximum_weight = jnp.max(weights)
    weights = jnp.where(weights > 0, weights, 0) / jnp.where(
        maximum_weight > 0, maximum_weight, 1
    )
    total = jnp.sum(weights)
    n_tiles = (n_samples + tile - 1) // tile
    encoded_positions = jnp.pad(
        encoded_positions, ((0, n_tiles * tile - n_samples), (0, 0))
    )
    encoded_marks = jnp.pad(encoded_marks, ((0, n_tiles * tile - n_samples), (0, 0)))
    weights = jnp.pad(weights, ((0, n_tiles * tile - n_samples),))
    if position_log_ratio is not None:
        position_log_ratio = jnp.pad(
            position_log_ratio, ((0, n_tiles * tile - n_samples), (0, 0))
        )

    def add_supported_tile(operand: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        first, output, weight_block = operand
        mark_block = jax.lax.dynamic_slice_in_dim(encoded_marks, first, tile)
        position_block = jax.lax.dynamic_slice_in_dim(encoded_positions, first, tile)
        log_mark = _log_kernel_matrix(features, mark_block, waveform_std)
        if position_log_ratio is None:
            log_position = _log_kernel_matrix(positions, position_block, position_std)
            if occupancy_parts is not None:
                log_position = (
                    log_position
                    - occupancy_parts[0][None, :]
                    - occupancy_parts[1][None, :]
                )
            elif log_occupancy is not None:
                log_position = log_position - log_occupancy[None, :]
        else:
            log_position = jax.lax.dynamic_slice_in_dim(position_log_ratio, first, tile)
        log_mark = jnp.where(weight_block[:, None] > 0, log_mark, -jnp.inf)
        log_position = jnp.where(weight_block[:, None] > 0, log_position, -jnp.inf)
        mark_max = jnp.max(log_mark, axis=0)
        position_max = jnp.max(log_position, axis=0)
        mark_max = jnp.where(jnp.isfinite(mark_max), mark_max, 0)
        position_max = jnp.where(jnp.isfinite(position_max), position_max, 0)
        numerator = jnp.matmul(
            jnp.exp(log_mark - mark_max).T,
            weight_block[:, None] * jnp.exp(log_position - position_max),
            precision=jax.lax.Precision.HIGHEST,
        )
        log_sum = jnp.where(
            numerator > 0,
            jnp.log(jnp.where(numerator > 0, numerator, 1)),
            -jnp.inf,
        )
        log_numerator = log_sum + mark_max[:, None] + position_max[None, :]
        # A positive subnormal sum can also omit many flushed-to-zero terms.
        # Normalized weights are at most one. Bound their omitted total by
        # tile_count*tiny, and fall back when
        # that bound exceeds four dtype epsilons of the computed sum. Compare
        # logs so the bound cannot itself be flushed to zero.
        dtype_info = jnp.finfo(numerator.dtype)
        log_suspect = (
            np.log(tile) + np.log(dtype_info.tiny) - np.log(4 * dtype_info.eps)
        )
        log_numerator = jax.lax.cond(
            jnp.any(log_sum <= log_suspect),
            _joint_logsumexp_fallback,
            lambda operand: operand[3],
            operand=(
                log_mark,
                log_position,
                jnp.where(
                    weight_block > 0,
                    jnp.log(jnp.where(weight_block > 0, weight_block, 1)),
                    -jnp.inf,
                ),
                log_numerator,
            ),
        )
        return jnp.logaddexp(output, log_numerator)

    def add_tile(number: int, output: jnp.ndarray) -> jnp.ndarray:
        first = number * tile
        weight_block = jax.lax.dynamic_slice_in_dim(weights, first, tile)
        return jax.lax.cond(
            jnp.any(weight_block > 0),
            add_supported_tile,
            lambda operand: operand[1],
            (first, output, weight_block),
        )

    output = jax.lax.fori_loop(0, n_tiles, add_tile, output)
    result = jnp.where(
        total > 0, output - jnp.log(jnp.where(total > 0, total, 1)), -jnp.inf
    )
    return jnp.where(invalid, jnp.nan, result)


def _joint_logsumexp_fallback(operand: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    log_mark, log_position, log_weights, _ = operand
    return jax.lax.map(
        lambda mark: jsp.special.logsumexp(
            mark[:, None] + log_position + log_weights[:, None], axis=0
        ),
        log_mark.T,
    )


def _raw_kde_marks(
    model: dict[str, Any],
    electrode: int,
    positions: ArrayLike,
    features: ArrayLike,
    local: bool,
    sample_tile: int | None,
    *,
    log_occupancy: ArrayLike | None = None,
    occupancy_parts: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    position_log_ratio: ArrayLike | None = None,
) -> jnp.ndarray:
    encoded_positions = jnp.asarray(model["encoding_positions"][electrode])
    encoded_marks = jnp.asarray(model["encoding_spike_waveform_features"][electrode])
    weights = model.get("encoding_weights")
    weights = (
        jnp.ones(encoded_marks.shape[0])
        if weights is None or weights[electrode] is None
        else jnp.asarray(weights[electrode])
    )
    rate = model["mean_rates"][electrode]
    std = jnp.concatenate(
        (
            as_std_array(model["position_std"], positions.shape[1]),
            as_std_array(model["waveform_std"], features.shape[1]),
        )
    )
    if local:
        occupancy_parts = _kde_model_density_parts(
            model["occupancy_model"], positions, sample_tile
        )
        points = jnp.concatenate((positions, features), axis=1)
        joint = _kde_log_density(
            points,
            jnp.concatenate((encoded_positions, encoded_marks), axis=1),
            std,
            weights,
            sample_tile,
            occupancy_parts=occupancy_parts,
            n_position_dims=positions.shape[1],
        )
        return jnp.log(rate) + joint
    if occupancy_parts is None:
        occupancy_parts = (
            _kde_model_density_parts(model["occupancy_model"], positions, sample_tile)
            if log_occupancy is None
            else (jnp.asarray(log_occupancy), jnp.zeros_like(log_occupancy))
        )
    joint = _kde_log_joint_nonlocal(
        positions,
        features,
        encoded_positions,
        encoded_marks,
        as_std_array(model["position_std"], positions.shape[1]),
        as_std_array(model["waveform_std"], features.shape[1]),
        weights,
        sample_tile,
        occupancy_parts=occupancy_parts,
        position_log_ratio=position_log_ratio,
    )
    return jnp.log(rate) + joint


def _mix_background_rows(
    raw, log_reference, row_ids, n_rows, is_local, indices_are_sorted
):
    reference = log_reference if is_local else log_reference[:, None]
    event_terms = jnp.logaddexp(raw, reference)
    mask = row_ids < n_rows
    event_terms = jnp.where(mask if is_local else mask[:, None], event_terms, 0)
    return deterministic_row_sum(
        event_terms, row_ids, n_rows, indices_are_sorted=indices_are_sorted
    )


@partial(
    jax.jit,
    static_argnames=(
        "is_local",
        "n_rows",
        "indices_are_sorted",
        "sample_tile",
        "has_rate",
    ),
)
def _background_kde_rows(
    features,
    row_ids,
    log_reference,
    positions,
    encoded_positions,
    encoded_marks,
    weights,
    position_std,
    waveform_std,
    log_rate,
    occupancy_samples,
    occupancy_weights,
    occupancy_parts,
    position_log_ratio,
    *,
    is_local,
    n_rows,
    indices_are_sorted,
    sample_tile,
    has_rate,
):
    """Score, mix, and deterministically reduce one padded KDE event block."""
    if not has_rate:
        raw = jnp.full(
            (features.shape[0],)
            if is_local
            else (features.shape[0], positions.shape[0]),
            -jnp.inf,
        )
    elif is_local:
        occupancy_parts = _kde_density_parts(
            positions, occupancy_samples, position_std, occupancy_weights, sample_tile
        )
        joint = _kde_log_density(
            jnp.concatenate((positions, features), axis=1),
            jnp.concatenate((encoded_positions, encoded_marks), axis=1),
            jnp.concatenate((position_std, waveform_std)),
            weights,
            sample_tile,
            occupancy_parts=occupancy_parts,
            n_position_dims=positions.shape[1],
        )
        raw = log_rate + jax.lax.optimization_barrier(joint)
    else:
        joint = _kde_log_joint_nonlocal(
            positions,
            features,
            encoded_positions,
            encoded_marks,
            position_std,
            waveform_std,
            weights,
            sample_tile,
            occupancy_parts=occupancy_parts,
            position_log_ratio=position_log_ratio,
        )
        raw = log_rate + jax.lax.optimization_barrier(joint)
    return _mix_background_rows(
        raw, log_reference, row_ids, n_rows, is_local, indices_are_sorted
    )


@partial(
    jax.jit,
    static_argnames=(
        "is_local",
        "n_rows",
        "indices_are_sorted",
        "joint_covariance_type",
        "gpi_covariance_type",
        "occupancy_covariance_type",
        "has_rate",
    ),
)
def _background_gmm_rows(
    features,
    row_ids,
    log_reference,
    positions,
    joint_parameters,
    occupancy_parameters,
    log_rate,
    conditional=None,
    gpi_parameters=None,
    *,
    is_local,
    n_rows,
    indices_are_sorted,
    joint_covariance_type,
    occupancy_covariance_type,
    gpi_covariance_type="full",
    has_rate=True,
):
    """Traceable parameter tuples support both historical and normalized GMMs."""
    from non_local_detector.likelihoods import clusterless_gmm

    def score_rows(evaluation_positions):
        if not has_rate:
            raw = jnp.full(
                (features.shape[0],)
                if is_local
                else (features.shape[0], evaluation_positions.shape[0]),
                -jnp.inf,
            )
        elif conditional is not None:
            raw = clusterless_gmm._raw_gmm_log_intensity_from_parameters(
                evaluation_positions,
                features,
                conditional,
                gpi_parameters,
                occupancy_parameters,
                log_rate,
                is_local=is_local,
                joint_covariance_type=joint_covariance_type,
                gpi_covariance_type=gpi_covariance_type,
                occupancy_covariance_type=occupancy_covariance_type,
            )
        else:
            points = (
                jnp.concatenate((evaluation_positions, features), axis=1)
                if is_local
                else jnp.concatenate(
                    (
                        jnp.tile(evaluation_positions, (features.shape[0], 1)),
                        jnp.repeat(features, evaluation_positions.shape[0], axis=0),
                    ),
                    axis=1,
                )
            )
            raw = log_rate + (
                clusterless_gmm._gmm_score(
                    points, joint_parameters, joint_covariance_type
                )
                - clusterless_gmm._gmm_score(
                    points[:, : evaluation_positions.shape[1]],
                    occupancy_parameters,
                    occupancy_covariance_type,
                )
            )
            if not is_local:
                raw = raw.reshape(features.shape[0], evaluation_positions.shape[0])
        return _mix_background_rows(
            raw, log_reference, row_ids, n_rows, is_local, indices_are_sorted
        )

    if is_local or positions.shape[0] <= 128:
        return score_rows(positions)
    # Bound the mark x position Cartesian sample axis inside this executable.
    # Fusion otherwise keeps the scorer's full temporary arena alive through
    # mixture/reduction, despite a modest logical live-buffer peak. Position
    # tiling preserves the padded event axis and its deterministic sum tree.
    position_tile = 128
    first_rows = score_rows(positions[:position_tile])
    total = jnp.zeros((n_rows, positions.shape[0]), dtype=first_rows.dtype)
    total = jax.lax.dynamic_update_slice(total, first_rows, (0, 0))
    n_tiles = (positions.shape[0] + position_tile - 1) // position_tile

    def add_position_tile(number, total):
        requested_start = number * position_tile
        column_start = jnp.minimum(requested_start, positions.shape[0] - position_tile)
        block_positions = jax.lax.dynamic_slice_in_dim(
            positions, column_start, position_tile
        )
        rows = score_rows(block_positions)
        # An overlapping final tile avoids artificial spatial points. Preserve
        # its already-written prefix so each real column is stored once.
        current = jax.lax.dynamic_slice_in_dim(
            total, column_start, position_tile, axis=1
        )
        rows = jnp.where(
            (column_start + jnp.arange(position_tile) >= requested_start)[None, :],
            rows,
            current,
        )
        return jax.lax.dynamic_update_slice(total, rows, (0, column_start))

    return jax.lax.fori_loop(1, n_tiles, add_position_tile, total)


@partial(
    jax.jit,
    static_argnames=(
        "is_local",
        "n_rows",
        "indices_are_sorted",
        "position_std",
        "n_components",
        "has_rate",
        "source_tile_size",
    ),
)
def _background_diffusion_rows(
    standardized_features,
    row_ids,
    log_reference,
    event_bins,
    encoding_bins,
    encoding_marks,
    encoding_weights,
    waveform_std,
    eigvals,
    eigvecs,
    labels,
    weight_total,
    bin_sizes,
    occupancy,
    rate,
    reference_scale,
    source_kernel=None,
    source_bins=None,
    *,
    is_local,
    n_rows,
    indices_are_sorted,
    position_std,
    n_components,
    has_rate,
    source_tile_size=256,
):
    from non_local_detector.likelihoods import clusterless_diffusion

    if not has_rate:
        raw = jnp.full(
            (standardized_features.shape[0],)
            if is_local
            else (standardized_features.shape[0], occupancy.shape[0]),
            -jnp.inf,
        )
    else:
        source_args = (
            (source_kernel, source_tile_size)
            if hasattr(clusterless_diffusion, "_prediction_source_operator")
            else ()
        )
        if source_bins is not None:
            source_args = (*source_args, source_bins)
        field = clusterless_diffusion._diffused_mark_field(
            standardized_features,
            encoding_bins,
            encoding_marks,
            encoding_weights,
            waveform_std,
            eigvals,
            eigvecs,
            labels,
            position_std,
            n_components,
            *source_args,
        )
        raw_field = field / (weight_total * bin_sizes[:, None] * occupancy[:, None])
        raw = jnp.log(rate * raw_field) - jnp.sum(jnp.log(reference_scale))
        raw = (
            raw[event_bins, jnp.arange(standardized_features.shape[0])]
            if is_local
            else raw.T
        )
    return _mix_background_rows(
        raw, log_reference, row_ids, n_rows, is_local, indices_are_sorted
    )


def _kde_spatial_density_parts(model, centers, sample_tile, position_tile):
    occupancy_parts = _kde_model_density_parts(
        model["occupancy_model"], centers, sample_tile, position_tile
    )
    ground = jnp.zeros(centers.shape[0])
    for rate, gpi in zip(model["mean_rates"], model["gpi_models"], strict=True):
        ground_parts = _kde_model_density_parts(
            gpi, centers, sample_tile, position_tile
        )
        ground = ground + jnp.where(
            rate > 0,
            jnp.exp(
                jnp.log(rate)
                + (ground_parts[0] - occupancy_parts[0])
                + (ground_parts[1] - occupancy_parts[1])
            ),
            0,
        )
    return occupancy_parts, ground


def _gmm_spatial_ground(model, centers):
    log_occ = model["occupancy_model"].score_samples(centers)
    ground = jnp.zeros(centers.shape[0])
    for rate, gpi in zip(model["mean_rates"], model["gpi_models"], strict=True):
        if gpi is not None:
            ground = ground + jnp.exp(
                jnp.log(rate) + (gpi.score_samples(centers) - log_occ)
            )
    return ground


def predict_clusterless_with_background(
    algorithm: str,
    model: dict[str, Any],
    position_time: ArrayLike | None,
    position: ArrayLike | None,
    spike_times: list[ArrayLike],
    marks: list[ArrayLike],
    time_edges: ArrayLike,
    *,
    background_mark_models: list[BackgroundMarkModel | None],
    background_rate: float = RATE_EPS_HZ,
    is_local: bool = False,
    row_slice: slice | None = None,
    spike_time_order: _SpikeTimeOrder | None = None,
    time_grid: Any = None,
) -> jnp.ndarray:
    """Score raw algorithm intensities plus a normalized reference component.

    This path is selected by detectors with the mark-reference contract. Legacy
    standalone predictors retain their historical signature/outputs without it.
    The event mixture precedes row reduction and every electrode adds its rate
    to the expected count, including an electrode with zero encoding spikes.
    """
    grid = _resolve_time_grid(time_edges, time_grid)
    time_edges = grid.edges
    start, stop = resolve_row_slice(row_slice, len(time_edges) - 1)
    order = spike_time_order or _SpikeTimeOrder()
    validate_population_lengths(
        "electrode",
        spike_times=spike_times,
        spike_waveform_features=marks,
        mean_rates=model["mean_rates"],
        background_mark_models=background_mark_models,
    )
    for electrode, (times, features, reference) in enumerate(
        zip(spike_times, marks, background_mark_models, strict=True)
    ):
        validate_spike_feature_pair(times, features, electrode)
        if reference is not None and np.shape(features)[1] != len(reference.location):
            raise ValidationError(
                f"Waveform-feature dimensionality for electrode {electrode} does not match the fitted background mark model"
            )
    if background_rate < 0 or not np.isfinite(background_rate):
        raise ValidationError("background_rate must be finite and nonnegative")
    environment = model["environment"]
    centers = order.memo(
        "background_bin_centers",
        (environment.place_bin_centers_, environment.is_track_interior_),
        lambda: jnp.asarray(
            environment.place_bin_centers_[environment.is_track_interior_.ravel()]
        ),
    )
    sample_tile = model.get("encoding_block_size") or model.get("enc_tile_size")
    position_tile = (
        model.get("position_block_size")
        or model.get("pos_tile_size")
        or model.get("bin_tile_size")
        or centers.shape[0]
    )
    event_block = model.get("spike_block_size") or model.get("block_size") or 100
    if is_local:
        centers = order.memo(
            f"background_local_centers:{start}:{stop}",
            (position_time, position, time_edges, environment),
            lambda: get_position_at_time(
                position_time,
                position,
                decode_bin_centers(time_edges, start, stop),
                environment,
            ),
        )
    spatial_sources = (
        model.get("occupancy_model"),
        *model.get("gpi_models", ()),
        model["mean_rates"],
        environment.place_bin_centers_,
        environment.is_track_interior_,
        *((position_time, position, time_edges) if is_local else ()),
    )
    spatial_range = f"local:{start}:{stop}" if is_local else "nonlocal"
    # Compute the original process's unfloored compensator, then add the reference.
    if algorithm.startswith("clusterless_kde"):
        occupancy_parts, ground = order.memo(
            f"background_kde_spatial:{spatial_range}:{sample_tile}:{position_tile}",
            spatial_sources,
            partial(
                _kde_spatial_density_parts, model, centers, sample_tile, position_tile
            ),
        )
    elif algorithm == "clusterless_gmm":
        ground = order.memo(
            f"background_gmm_spatial:{spatial_range}",
            spatial_sources,
            partial(_gmm_spatial_ground, model, centers),
        )
    else:
        from non_local_detector.likelihoods.clusterless_diffusion import (
            _full_to_local,
            _interior_bin_indices,
            environment_graph,
            get_device_basis,
        )

        if not np.array_equal(model["node_order"], environment_graph(environment)[1]):
            raise ValidationError(
                "encoding model is stale: environment grid was refit; refit before decoding"
            )
        full_to_local = _full_to_local(
            model["node_order"], environment.is_track_interior_.size
        )
        basis = get_device_basis(environment, model["resolved_rank"])
        from non_local_detector.likelihoods import clusterless_diffusion

        source_backend = getattr(
            clusterless_diffusion, "_prediction_source_operator", None
        )
        source_args = () if source_backend is None else source_backend(model, basis[1])
        ground = jnp.asarray(
            model.get(
                "unfloored_ground_process_intensity",
                model["summed_ground_process_intensity"],
            )
        )
        if is_local:
            ground = ground[_interior_bin_indices(environment, centers, full_to_local)]
    ground = ground + len(spike_times) * background_rate
    rates = order.memo(
        "background_mean_rates",
        (model["mean_rates"],),
        lambda: np.asarray(model["mean_rates"]),
    )
    durations = jnp.asarray(grid.durations(start, stop))
    total = -durations[:, None] * (ground[:, None] if is_local else ground[None, :])
    for electrode, (times, features, reference) in enumerate(
        zip(spike_times, marks, background_mark_models, strict=True)
    ):
        events = _prepare_background_events(
            reference, times, features, time_edges, start, stop, order, electrode
        )
        selection = events["selection"]
        n_events = events["n_events"]
        if not n_events:
            continue
        real_event_positions = None
        if is_local:
            real_event_positions = order.memo(
                f"background_spike_positions:{start}:{stop}",
                (position_time, position, times, features, environment, time_edges),
                lambda events=events: get_position_at_time(
                    position_time, position, events["selected_times"], environment
                ),
            )
        rate = rates[electrode]
        log_rate = jnp.log(rate) if rate > 0 else jnp.asarray(-jnp.inf)
        # Mark standardization has happened on the host before device casting.
        # The shared reference is evaluated at bucketed counts even for NoSpike.
        log_reference = events["log_density"] + (
            np.log(background_rate) if background_rate else -np.inf
        )
        electrode_event_block = event_block
        cached_position_log_ratio = None
        if algorithm.startswith("clusterless_kde"):
            encoded_positions = model["encoding_positions"][electrode]
            encoded_marks = model["encoding_spike_waveform_features"][electrode]
            encoding_weights = model.get("encoding_weights")
            encoding_weights = (
                None if encoding_weights is None else encoding_weights[electrode]
            )
            encoded_positions, encoded_marks, encoding_weights = order.memo(
                "background_kde_encoding",
                (encoded_positions, encoded_marks, encoding_weights),
                lambda encoded_positions=encoded_positions,
                encoded_marks=encoded_marks,
                encoding_weights=encoding_weights: (
                    jnp.asarray(encoded_positions),
                    jnp.asarray(encoded_marks),
                    jnp.ones(np.shape(encoded_marks)[0])
                    if encoding_weights is None
                    else jnp.asarray(encoding_weights),
                ),
            )
            if (
                not is_local
                and sample_tile is None
                and position_tile >= centers.shape[0]
            ):
                # Only one electrode's matrix is retained at a time, preserving
                # tiled paths' workspace bounds across electrodes and chunks.
                cached_position_log_ratio = (
                    _log_kernel_matrix(
                        centers,
                        encoded_positions,
                        as_std_array(model["position_std"], centers.shape[1]),
                    )
                    - occupancy_parts[0][None, :]
                    - occupancy_parts[1][None, :]
                )
        elif algorithm == "clusterless_gmm":
            from non_local_detector.likelihoods import clusterless_gmm

            occupancy_model = model["occupancy_model"]
            joint = model["joint_models"][electrode]
            gpi = model["gpi_models"][electrode]
            joint_parameters = (
                None if joint is None else clusterless_gmm._gmm_parameters(joint)
            )
            occupancy_parameters = clusterless_gmm._gmm_parameters(occupancy_model)
            normalized_backend = getattr(
                clusterless_gmm, "_raw_gmm_log_intensity_from_parameters", None
            )
            conditional = (
                None
                if normalized_backend is None or joint is None
                else clusterless_gmm._conditional_parameters(
                    joint, centers.shape[1], order
                )
            )
            gpi_parameters = (
                None if conditional is None else clusterless_gmm._gmm_parameters(gpi)
            )
        else:
            from non_local_detector.likelihoods.clusterless_diffusion import (
                _effective_block,
                _padded_encoding,
            )

            electrode_event_block = _effective_block(
                model["memory_budget"],
                np.shape(model["encoding_marks"][electrode])[0],
                model["resolved_rank"],
                np.shape(model["occupancy"])[0],
                event_block,
            )
            reference_scale = jnp.asarray(reference.scale)
            diffusion_encoding_marks = model["encoding_marks"][electrode]
            physical_bins = model["encoding_bin_indices"][electrode]
            diffusion_weights = model["encoding_weights"][electrode]
            source_bins = model.get("source_bin_indices")

            def prepare_diffusion_encoding(
                physical_bins=physical_bins,
                diffusion_encoding_marks=diffusion_encoding_marks,
                diffusion_weights=diffusion_weights,
                reference=reference,
                source_bins=source_bins,
            ):
                standardized = reference.standardize(diffusion_encoding_marks)
                if source_bins is None:
                    return _padded_encoding(
                        physical_bins,
                        standardized,
                        diffusion_weights,
                        basis[1].shape[0],
                    )
                return _padded_encoding(
                    physical_bins,
                    standardized,
                    diffusion_weights,
                    len(source_bins),
                    source_bins=source_bins,
                )

            encoding_bins, standardized_encoding, diffusion_weights = order.memo(
                "background_standardized_diffusion_encoding",
                (
                    physical_bins,
                    diffusion_encoding_marks,
                    diffusion_weights,
                    reference,
                    source_bins,
                ),
                prepare_diffusion_encoding,
            )
            standardized_bandwidth = jnp.asarray(
                np.broadcast_to(
                    np.asarray(model["waveform_std"]), (np.shape(features)[1],)
                )
                / reference.scale
            )

        for first in range(0, n_events, electrode_event_block):
            last = min(first + electrode_event_block, n_events)
            padded_count = _padded_spike_count(last - first, electrode_event_block)
            pad_after = max(first + padded_count - events["features"].shape[0], 0)
            block = jnp.pad(
                events["features"][first : first + padded_count],
                ((0, pad_after), (0, 0)),
            )
            # Sentinel ids are dropped, and an explicit mask prevents the finite
            # fake marks from contributing to any real row or compensator.
            row_ids = np.full(padded_count, stop - start, dtype=np.intp)
            row_ids[: last - first] = selection.bin_ind[first:last]
            row_ids = jnp.asarray(row_ids)
            pad_after = max(first + padded_count - log_reference.shape[0], 0)
            reference_terms = jnp.pad(
                log_reference[first : first + padded_count], (0, pad_after)
            )
            if is_local:
                real_positions = real_event_positions[first:last]
                # Repeat a finite real position rather than introducing an
                # extrapolated or invalid point in a masked event.
                event_positions = jnp.asarray(
                    np.concatenate(
                        (
                            real_positions,
                            np.repeat(
                                real_positions[:1],
                                padded_count - real_positions.shape[0],
                                axis=0,
                            ),
                        )
                    )
                )
            if algorithm == "clusterless_diffusion":
                standardized_block = jnp.pad(
                    events["standardized"][first : first + padded_count],
                    ((0, pad_after), (0, 0)),
                )
                event_bins = (
                    jnp.asarray(
                        _interior_bin_indices(
                            environment, event_positions, full_to_local
                        )
                    )
                    if is_local
                    else jnp.zeros(padded_count, dtype=int)
                )
                rows = _background_diffusion_rows(
                    standardized_block,
                    row_ids,
                    reference_terms,
                    event_bins,
                    jnp.asarray(encoding_bins),
                    standardized_encoding,
                    diffusion_weights,
                    standardized_bandwidth,
                    *basis[:3],
                    model["weight_total"][electrode],
                    jnp.asarray(model["bin_sizes"]),
                    jnp.asarray(model["occupancy"]),
                    rate,
                    reference_scale,
                    source_args[0] if source_args else None,
                    jnp.asarray(model["source_bin_indices"])
                    if "source_bin_indices" in model
                    else None,
                    is_local=is_local,
                    n_rows=stop - start,
                    indices_are_sorted=selection.indices_are_sorted,
                    position_std=model["position_std"],
                    n_components=basis[3],
                    has_rate=bool(rate > 0),
                    source_tile_size=source_args[1] if source_args else 256,
                )
                total = total + (rows[:, None] if is_local else rows)
                continue
            for col in range(0, 1 if is_local else centers.shape[0], position_tile):
                positions = (
                    event_positions if is_local else centers[col : col + position_tile]
                )
                if algorithm.startswith("clusterless_kde"):
                    occupancy_model = model["occupancy_model"]
                    rows = _background_kde_rows(
                        block,
                        row_ids,
                        reference_terms,
                        positions,
                        encoded_positions,
                        encoded_marks,
                        encoding_weights,
                        as_std_array(model["position_std"], positions.shape[1]),
                        as_std_array(model["waveform_std"], block.shape[1]),
                        log_rate,
                        occupancy_model.samples_,
                        occupancy_model.weights_,
                        None
                        if is_local
                        else tuple(
                            part[col : col + positions.shape[0]]
                            for part in occupancy_parts
                        ),
                        cached_position_log_ratio,
                        is_local=is_local,
                        n_rows=stop - start,
                        indices_are_sorted=selection.indices_are_sorted,
                        sample_tile=sample_tile,
                        has_rate=bool(rate > 0),
                    )
                else:
                    rows = _background_gmm_rows(
                        block,
                        row_ids,
                        reference_terms,
                        positions,
                        joint_parameters,
                        occupancy_parameters,
                        log_rate,
                        conditional,
                        gpi_parameters,
                        is_local=is_local,
                        n_rows=stop - start,
                        indices_are_sorted=selection.indices_are_sorted,
                        joint_covariance_type="full"
                        if joint is None
                        else joint.covariance_type,
                        occupancy_covariance_type=occupancy_model.covariance_type,
                        gpi_covariance_type="full"
                        if gpi is None
                        else gpi.covariance_type,
                        has_rate=bool(rate > 0),
                    )
                if is_local:
                    total = total + rows[:, None]
                else:
                    total = total.at[:, col : col + positions.shape[0]].add(rows)
    return (
        total
        + log_bin_duration_evidence(
            spike_times, time_edges, row_slice, order, _time_grid=grid
        )[:, None]
    )
