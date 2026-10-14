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
    _SpikeTimeOrder,
    as_std_array,
    decode_bin_centers,
    get_position_at_time,
    log_bin_duration_evidence,
    resolve_row_slice,
    select_spike_rows,
    select_spikes_in_rows,
    sum_spikes_into_rows,
    validate_population_lengths,
    validate_spike_feature_pair,
)
from non_local_detector.time_edges import _resolve_time_grid


@dataclass(frozen=True)
class BackgroundMarkModel:
    """A normalized diagonal Gaussian with training-derived location and scale."""

    location: np.ndarray
    scale: np.ndarray

    def standardize(self, marks: ArrayLike) -> jnp.ndarray:
        # Fitted statistics live on the host. Center/scale before casting to the
        # device dtype so an affine change of units does not lose additional
        # digits to subtraction in float32.
        return jnp.asarray((np.asarray(marks) - self.location) / self.scale)

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
        selection = select_spikes_in_rows(
            times, start, stop, time_edges=time_edges, _spike_time_order=order
        )
        selected = select_spike_rows(features, selection)
        total = total + sum_spikes_into_rows(
            background_mark_log_density(model, selected, electrode), selection
        )
    return total[:, None]


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
    centers = jnp.asarray(
        environment.place_bin_centers_[environment.is_track_interior_.ravel()]
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
        centers = get_position_at_time(
            position_time,
            position,
            decode_bin_centers(time_edges, start, stop),
            environment,
        )
    # Compute the original process's unfloored compensator, then add the reference.
    if algorithm.startswith("clusterless_kde"):
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
    elif algorithm == "clusterless_gmm":
        log_occ = model["occupancy_model"].score_samples(centers)
        ground = jnp.zeros(centers.shape[0])
        for rate, gpi in zip(model["mean_rates"], model["gpi_models"], strict=True):
            if gpi is not None:
                ground = ground + jnp.exp(
                    jnp.log(rate) + (gpi.score_samples(centers) - log_occ)
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
    durations = jnp.asarray(grid.durations(start, stop))
    total = -durations[:, None] * (ground[:, None] if is_local else ground[None, :])
    for electrode, (times, features, reference) in enumerate(
        zip(spike_times, marks, background_mark_models, strict=True)
    ):
        selection = select_spikes_in_rows(
            times, start, stop, time_edges=time_edges, _spike_time_order=order
        )
        selected_features = select_spike_rows(features, selection)
        features = jnp.asarray(selected_features)
        event_times = select_spike_rows(times, selection)
        n_events = features.shape[0]
        if not n_events:
            continue
        log_reference = background_mark_log_density(
            reference, selected_features, electrode
        ) + np.log(background_rate)
        rate = model["mean_rates"][electrode]
        cached_position_log_ratio = None
        if (
            algorithm.startswith("clusterless_kde")
            and not is_local
            and sample_tile is None
            and position_tile >= centers.shape[0]
        ):
            # Hoist only the current electrode's default dense matrix. Tiled
            # paths retain their encoding/position workspace bounds, and no
            # matrix remains cached for every electrode or inference chunk.
            cached_position_log_ratio = (
                _log_kernel_matrix(
                    centers,
                    jnp.asarray(model["encoding_positions"][electrode]),
                    as_std_array(model["position_std"], centers.shape[1]),
                )
                - occupancy_parts[0][None, :]
                - occupancy_parts[1][None, :]
            )
        electrode_event_block = event_block
        if algorithm == "clusterless_diffusion":
            from non_local_detector.likelihoods.clusterless_diffusion import (
                _effective_block,
            )

            electrode_event_block = _effective_block(
                model["memory_budget"],
                np.shape(model["encoding_marks"][electrode])[0],
                model["resolved_rank"],
                np.shape(model["occupancy"])[0],
                event_block,
            )
            reference_scale = jnp.asarray(reference.scale)
            standardized_decode = reference.standardize(selected_features)
            encoded_marks = model["encoding_marks"][electrode]
            standardized_encoding = order.memo(
                "background_standardized_diffusion_marks",
                (encoded_marks, reference),
                partial(reference.standardize, encoded_marks),
            )
            standardized_bandwidth = jnp.asarray(
                np.broadcast_to(np.asarray(model["waveform_std"]), (features.shape[1],))
                / reference.scale
            )
        for first in range(0, n_events, electrode_event_block):
            last = min(first + electrode_event_block, n_events)
            block = features[first:last]
            event_positions = (
                get_position_at_time(
                    position_time, position, event_times[first:last], environment
                )
                if is_local
                else None
            )
            for col in range(0, 1 if is_local else centers.shape[0], position_tile):
                positions = (
                    event_positions if is_local else centers[col : col + position_tile]
                )
                if rate == 0:
                    raw = jnp.full(
                        (last - first,)
                        if is_local
                        else (last - first, positions.shape[0]),
                        -jnp.inf,
                    )
                elif algorithm.startswith("clusterless_kde"):
                    raw = _raw_kde_marks(
                        model,
                        electrode,
                        positions,
                        block,
                        is_local,
                        sample_tile,
                        occupancy_parts=None
                        if is_local
                        else tuple(
                            part[col : col + positions.shape[0]]
                            for part in occupancy_parts
                        ),
                        position_log_ratio=cached_position_log_ratio,
                    )
                elif algorithm == "clusterless_gmm":
                    from non_local_detector.likelihoods import clusterless_gmm

                    raw_backend = getattr(
                        clusterless_gmm, "_raw_gmm_log_intensity", None
                    )
                    if raw_backend is not None:
                        raw = raw_backend(
                            model,
                            electrode,
                            positions,
                            block,
                            is_local,
                            preparation=order,
                        )
                    else:
                        joint = model["joint_models"][electrode]
                        if is_local:
                            points = jnp.concatenate((positions, block), axis=1)
                        else:
                            points = jnp.concatenate(
                                (
                                    jnp.tile(positions, (block.shape[0], 1)),
                                    jnp.repeat(block, positions.shape[0], axis=0),
                                ),
                                axis=1,
                            )
                        raw = jnp.log(rate) + (
                            joint.score_samples(points)
                            - model["occupancy_model"].score_samples(
                                points[:, : positions.shape[1]]
                            )
                        )
                        if not is_local:
                            raw = raw.reshape(block.shape[0], positions.shape[0])
                else:
                    eigvals, eigvecs, labels, n_components = basis
                    # Apply the heat operator to dimensionless mark kernels.
                    # Its clipping/rescale roundoff should not depend on the
                    # units' arbitrary Gaussian density normalizer.
                    field = clusterless_diffusion._diffused_mark_field(
                        standardized_decode[first:last],
                        model["encoding_bin_indices"][electrode],
                        standardized_encoding,
                        model["encoding_weights"][electrode],
                        standardized_bandwidth,
                        eigvals,
                        eigvecs,
                        labels,
                        model["position_std"],
                        n_components,
                        *source_args,
                    )
                    raw_field = field / (
                        model["weight_total"][electrode]
                        * jnp.asarray(model["bin_sizes"])[:, None]
                        * jnp.asarray(model["occupancy"])[:, None]
                    )
                    raw = jnp.log(rate * raw_field) - jnp.sum(jnp.log(reference_scale))
                    if is_local:
                        event_bins = _interior_bin_indices(
                            environment, positions, full_to_local
                        )
                        raw = raw[event_bins, jnp.arange(block.shape[0])]
                    else:
                        raw = raw[col : col + position_tile].T
                reference_terms = log_reference[first:last]
                event_terms = jnp.logaddexp(
                    raw, reference_terms if is_local else reference_terms[:, None]
                )
                # Select the same event subrange while preserving original row IDs.
                from non_local_detector.likelihoods.common import deterministic_row_sum

                rows = deterministic_row_sum(
                    event_terms,
                    selection.bin_ind[first:last],
                    stop - start,
                    indices_are_sorted=selection.indices_are_sorted,
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
