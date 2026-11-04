"""The likelihoods.common package keeps the former module's names and layering."""

import ast
import importlib
import pickle
from pathlib import Path

import pytest

import non_local_detector.likelihoods.common as common
from non_local_detector.models.base import SortedSpikesDetector

pytestmark = pytest.mark.unit

COMPAT_DATA = Path(__file__).parents[1] / "compat_data"

# Every function, class and constant the single-module ``likelihoods.common``
# defined, and the project names it imported. Code and pickles written
# against that module name them.
FORMER_NAMES = (
    "as_std_array",
    "_batched_kde_rows",
    "block_kde",
    "block_log_kde",
    "_blocked_nonlocal_poisson_log_likelihood",
    "_bucketed_size",
    "_compiled_batched_scaled_kde_log_density",
    "_compiled_block_kde",
    "_compiled_block_log_kde",
    "_compiled_log_space_kde_log_density",
    "_compiled_scaled_kde_log_density",
    "_concatenate_row_blocks",
    "COUNT_BLOCK_BYTES",
    "_count_block_rows",
    "decode_bin_centers",
    "_DecodeTimeGrid",
    "DEFAULT_WAVEFORM_STD",
    "deterministic_row_add",
    "deterministic_row_sum",
    "deterministic_segment_sum",
    "drop_zero_weight_samples",
    "_EMISSION_BLOCK_ROWS",
    "Environment",
    "environment_grid_fingerprint",
    "EPS",
    "gaussian_pdf",
    "get_position_at_time",
    "get_spikecount_per_time_bin",
    "interpolate_weights_at_spike_times",
    "_JAX_SHARDING_ERRORS",
    "kde",
    "_KDE_BLOCK_KERNEL_BYTES",
    "_kde_block_rows",
    "_kde_log_densities",
    "kde_log_density",
    "_kde_log_scale",
    "_KDE_PREPARATION_BYTES",
    "_KDE_PREPARATION_ENTRIES",
    "KDEModel",
    "local_kde_log_ratio",
    "log_bin_duration_evidence",
    "log_density_intensity",
    "LOG_EPS",
    "log_gaussian_pdf",
    "log_kde",
    "_log_kernel_matrix",
    "log_nonnegative",
    "LOG_RATE_EPS_HZ",
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
    "RATE_EPS_HZ",
    "RATE_REFERENCE_SECONDS",
    "_reduces_sequentially",
    "_relative_weights",
    "_repair_in_log_space",
    "requires_time_edges",
    "resolve_row_slice",
    "_resolve_time_grid",
    "_scaled_kde_log_density",
    "_scatter_add_segment_totals",
    "_segment_totals",
    "_segmented_inclusive_sum",
    "select_decode_marks",
    "select_spike_rows",
    "select_spikes_in_rows",
    "snap_off_track_positions",
    "_sorted_sample_keys",
    "_spike_counts_matrix",
    "spike_row_ids",
    "_spikes_are_ascending",
    "SpikeSelection",
    "_SpikeTimeOrder",
    "sum_spikes_into_rows",
    "_traced_block_kde",
    "_traced_block_log_kde",
    "_traced_blocks",
    "_true_kde_log_density",
    "_UNROLLED_KDE_BLOCKS",
    "validate_bandwidth",
    "validate_encoding_contract",
    "validate_finite",
    "validate_population_lengths",
    "validate_spike_feature_pair",
    "validate_spike_feature_population",
    "validate_time_edges",
    "validate_weights",
    "ValidationError",
    "weighted_mean_rate",
)

# Each submodule may import only from submodules before it.
LAYERS = (
    "numerics",
    "validation",
    "weights",
    "padding",
    "position",
    "reductions",
    "spike_rows",
    "marks",
    "poisson",
    "kernel_density",
)


@pytest.mark.parametrize("name", FORMER_NAMES)
def test_former_module_names_remain_importable(name):
    assert hasattr(common, name)
    assert name in common.__all__


def test_former_private_names_alias_their_public_helpers():
    assert common._SpikeTimeOrder is common.SpikeTimeOrder
    assert common._pad_rows is common.pad_rows
    assert common._padded_samples is common.padded_samples


def test_kde_model_pickles_under_the_former_module_path():
    """KDEModel pickles name ``non_local_detector.likelihoods.common.KDEModel``,
    as detectors saved before the split do, so each version loads the other's."""
    model = common.KDEModel(std=2.0)
    assert common.KDEModel.__module__ == "non_local_detector.likelihoods.common"
    data = pickle.dumps(model, protocol=0)  # protocol 0 names classes as text
    assert b"non_local_detector.likelihoods.common\nKDEModel" in data
    assert b"kernel_density" not in data
    loaded = pickle.loads(data)
    assert type(loaded) is common.KDEModel
    assert float(loaded.std) == 2.0


def test_detector_pickled_before_the_split_loads_its_kde_models():
    """A detector saved by main before the split names the former module path."""
    path = COMPAT_DATA / "main-d5f85823" / "nonlocal_sorted_1d_model.pkl"
    data = path.read_bytes()
    assert b"non_local_detector.likelihoods.common" in data
    assert b"kernel_density" not in data
    detector = SortedSpikesDetector.load_model(path)
    models = [
        model
        for encoding_model in detector.encoding_model_.values()
        for model in encoding_model["marginal_models"]
    ]
    assert models
    assert all(type(model) is common.KDEModel for model in models)


@pytest.mark.parametrize("layer", LAYERS)
def test_submodules_import_only_lower_layers(layer):
    module = importlib.import_module(f"non_local_detector.likelihoods.common.{layer}")
    tree = ast.parse(Path(module.__file__).read_text())
    prefix = "non_local_detector.likelihoods"
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(prefix)
    }
    lower = {f"{prefix}.common.{name}" for name in LAYERS[: LAYERS.index(layer)]}
    assert imported <= lower, imported - lower
