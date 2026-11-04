"""Shared building blocks for the likelihood algorithms.

Submodules by responsibility, each depending only on those listed before it:

- ``numerics``: numerical floors, the rate reference and safe logs.
- ``validation``: input validation and the fit/predict encoding contract.
- ``weights``: encoding weights at spike times and weighted mean rates.
- ``padding``: padded sizes shared by electrodes and chunks.
- ``position``: position at spike times, off-track snapping, interior bins.
- ``reductions``: deterministic segment and row sums.
- ``spike_rows``: spikes of a decode-row range, per-row sums and counts.
- ``marks``: mark kernels and masked log-space sums.
- ``poisson``: Poisson emission terms of sorted spike counts.
- ``kernel_density``: weighted Gaussian kernel density estimation.

This package re-exports the names of the former ``likelihoods.common`` module,
so imports and pickles that name ``non_local_detector.likelihoods.common``
(for example ``KDEModel``) keep working.
"""

from non_local_detector._validation import validate_weights
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common.kernel_density import (
    _KDE_BLOCK_KERNEL_BYTES,
    _UNROLLED_KDE_BLOCKS,
    KDEModel,
    _batched_kde_rows,
    _compiled_batched_scaled_kde_log_density,
    _compiled_block_kde,
    _compiled_block_log_kde,
    _compiled_log_space_kde_log_density,
    _compiled_scaled_kde_log_density,
    _kde_block_rows,
    _kde_log_scale,
    _log_kernel_matrix,
    _nearest_sample_log_kernel,
    _relative_weights,
    _repair_in_log_space,
    _scaled_kde_log_density,
    _sorted_sample_keys,
    _traced_block_log_kde,
    _traced_blocks,
    block_kde,
    block_log_kde,
    drop_zero_weight_samples,
    gaussian_pdf,
    kde,
    kde_log_densities,
    kde_log_density,
    local_kde_log_ratio,
    log_density_intensity,
    log_gaussian_pdf,
    log_kde,
    log_space_kde_log_density,
    padded_samples,
    prepare_kde_local_inputs,
    prepared_kde_log_density,
    traced_block_kde,
    true_kde_log_density,
)
from non_local_detector.likelihoods.common.marks import (
    EXACT_REPAIR_SLOTS,
    compute_log_mark_kernel_gemm,
    masked_logaddexp,
    masked_logsumexp,
    repair_flagged,
    zero_if_neginf,
)
from non_local_detector.likelihoods.common.numerics import (
    DEFAULT_WAVEFORM_STD,
    EPS,
    LOG_EPS,
    LOG_RATE_EPS_HZ,
    RATE_EPS_HZ,
    RATE_REFERENCE_SECONDS,
    log_nonnegative,
)
from non_local_detector.likelihoods.common.padding import (
    _octave_size,
    _quarter_octave,
    pad_rows,
    padded_index,
    padded_occupancy_count,
    padded_sample_count,
    padded_spike_count,
)
from non_local_detector.likelihoods.common.poisson import (
    _EMISSION_BLOCK_ROWS,
    COUNT_BLOCK_BYTES,
    _count_block_rows,
    _matrix_emission_is_safe,
    _poisson_nonlocal_log_likelihood,
    blocked_nonlocal_poisson_log_likelihood,
    concatenate_row_blocks,
    log_bin_duration_evidence,
    poisson_row_log_likelihood,
)
from non_local_detector.likelihoods.common.position import (
    full_to_local_index,
    get_position_at_time,
    interior_bin_indices,
    snap_off_track_positions,
)
from non_local_detector.likelihoods.common.reductions import (
    _bucketed_size,
    _reduces_sequentially,
    _scatter_add_segment_totals,
    _segment_totals,
    _segmented_inclusive_sum,
    deterministic_row_add,
    deterministic_row_sum,
    deterministic_segment_sum,
)
from non_local_detector.likelihoods.common.spike_rows import (
    _JAX_SHARDING_ERRORS,
    _KDE_PREPARATION_BYTES,
    _KDE_PREPARATION_ENTRIES,
    DecodeRows,
    SpikeSelection,
    SpikeTimeOrder,
    _normalized_row_ids,
    _spikes_are_ascending,
    get_spikecount_per_time_bin,
    prepare_decode_rows,
    resolve_row_slice,
    select_decode_marks,
    select_spike_rows,
    select_spikes_in_rows,
    spike_counts_matrix,
    spike_row_ids,
    sum_spikes_into_rows,
)
from non_local_detector.likelihoods.common.validation import (
    as_std_array,
    environment_grid_fingerprint,
    validate_bandwidth,
    validate_encoding_contract,
    validate_finite,
    validate_population_lengths,
    validate_spike_feature_pair,
    validate_spike_feature_population,
)
from non_local_detector.likelihoods.common.weights import (
    interpolate_weights_at_spike_times,
    weighted_mean_rate,
)
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    decode_bin_centers,
    requires_time_edges,
    validate_time_edges,
)

# The former module's private names of the helpers above, which the
# likelihood modules now import by their public names.
_SpikeTimeOrder = SpikeTimeOrder
_blocked_nonlocal_poisson_log_likelihood = blocked_nonlocal_poisson_log_likelihood
_concatenate_row_blocks = concatenate_row_blocks
_kde_log_densities = kde_log_densities
_log_space_kde_log_density = log_space_kde_log_density
_pad_rows = pad_rows
_padded_index = padded_index
_padded_occupancy_count = padded_occupancy_count
_padded_sample_count = padded_sample_count
_padded_samples = padded_samples
_padded_spike_count = padded_spike_count
_prepare_kde_local_inputs = prepare_kde_local_inputs
_prepared_kde_log_density = prepared_kde_log_density
_spike_counts_matrix = spike_counts_matrix
_traced_block_kde = traced_block_kde
_true_kde_log_density = true_kde_log_density

__all__ = [
    "COUNT_BLOCK_BYTES",
    "DEFAULT_WAVEFORM_STD",
    "EPS",
    "EXACT_REPAIR_SLOTS",
    "LOG_EPS",
    "LOG_RATE_EPS_HZ",
    "RATE_EPS_HZ",
    "RATE_REFERENCE_SECONDS",
    "_EMISSION_BLOCK_ROWS",
    "_JAX_SHARDING_ERRORS",
    "_KDE_BLOCK_KERNEL_BYTES",
    "_KDE_PREPARATION_BYTES",
    "_KDE_PREPARATION_ENTRIES",
    "_UNROLLED_KDE_BLOCKS",
    "DecodeRows",
    "Environment",
    "KDEModel",
    "SpikeSelection",
    "SpikeTimeOrder",
    "ValidationError",
    "_DecodeTimeGrid",
    "_SpikeTimeOrder",
    "_batched_kde_rows",
    "_blocked_nonlocal_poisson_log_likelihood",
    "_bucketed_size",
    "_compiled_batched_scaled_kde_log_density",
    "_compiled_block_kde",
    "_compiled_block_log_kde",
    "_compiled_log_space_kde_log_density",
    "_compiled_scaled_kde_log_density",
    "_concatenate_row_blocks",
    "_count_block_rows",
    "_kde_block_rows",
    "_kde_log_densities",
    "_kde_log_scale",
    "_log_kernel_matrix",
    "_log_space_kde_log_density",
    "_matrix_emission_is_safe",
    "_nearest_sample_log_kernel",
    "_normalized_row_ids",
    "_octave_size",
    "_pad_rows",
    "_padded_index",
    "_padded_occupancy_count",
    "_padded_sample_count",
    "_padded_samples",
    "_padded_spike_count",
    "_poisson_nonlocal_log_likelihood",
    "_prepare_kde_local_inputs",
    "_prepared_kde_log_density",
    "_quarter_octave",
    "_reduces_sequentially",
    "_relative_weights",
    "_repair_in_log_space",
    "_resolve_time_grid",
    "_scaled_kde_log_density",
    "_scatter_add_segment_totals",
    "_segment_totals",
    "_segmented_inclusive_sum",
    "_sorted_sample_keys",
    "_spike_counts_matrix",
    "_spikes_are_ascending",
    "_traced_block_kde",
    "_traced_block_log_kde",
    "_traced_blocks",
    "_true_kde_log_density",
    "as_std_array",
    "block_kde",
    "block_log_kde",
    "blocked_nonlocal_poisson_log_likelihood",
    "compute_log_mark_kernel_gemm",
    "concatenate_row_blocks",
    "decode_bin_centers",
    "deterministic_row_add",
    "deterministic_row_sum",
    "deterministic_segment_sum",
    "drop_zero_weight_samples",
    "environment_grid_fingerprint",
    "full_to_local_index",
    "gaussian_pdf",
    "get_position_at_time",
    "get_spikecount_per_time_bin",
    "interior_bin_indices",
    "interpolate_weights_at_spike_times",
    "kde",
    "kde_log_densities",
    "kde_log_density",
    "local_kde_log_ratio",
    "log_bin_duration_evidence",
    "log_density_intensity",
    "log_gaussian_pdf",
    "log_kde",
    "log_nonnegative",
    "log_space_kde_log_density",
    "masked_logaddexp",
    "masked_logsumexp",
    "pad_rows",
    "padded_index",
    "padded_occupancy_count",
    "padded_sample_count",
    "padded_samples",
    "padded_spike_count",
    "poisson_row_log_likelihood",
    "prepare_decode_rows",
    "prepare_kde_local_inputs",
    "prepared_kde_log_density",
    "repair_flagged",
    "requires_time_edges",
    "resolve_row_slice",
    "select_decode_marks",
    "select_spike_rows",
    "select_spikes_in_rows",
    "snap_off_track_positions",
    "spike_counts_matrix",
    "spike_row_ids",
    "sum_spikes_into_rows",
    "traced_block_kde",
    "true_kde_log_density",
    "validate_bandwidth",
    "validate_encoding_contract",
    "validate_finite",
    "validate_population_lengths",
    "validate_spike_feature_pair",
    "validate_spike_feature_population",
    "validate_time_edges",
    "validate_weights",
    "weighted_mean_rate",
    "zero_if_neginf",
]
