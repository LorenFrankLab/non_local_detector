"""A saved encoding model holds exactly the keys of its algorithm's type.

The keys are part of the saved-model format: ``load_model`` refuses an
encoding model with a key its algorithm's type does not declare, without one
it requires, or naming an algorithm that is not registered. The key sets are
pinned below, so renaming or adding a key fails here and needs a format
version (see CONTRIBUTING.md, "Saved-model format").
"""

import shutil

import pytest

import non_local_detector as nld
from non_local_detector.exceptions import DataError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.model_format import read_model_summary
from tests.model_format.conftest import SORTED_RATE, algorithm_params
from tests.model_format.test_refusals import _edit_json

pytestmark = pytest.mark.integration

_COMMON = {
    "algorithm",
    "disable_progress_bar",
    "encoding_exposure_seconds",
    "environment",
    "environment_grid",
    "rate_units",
}
# The keys each algorithm's encoding model stores in format version 1.
_STORED_KEYS = {
    "sorted_spikes_glm": _COMMON
    | {
        "coefficients",
        "emission_design_info",
        "is_track_interior",
        "no_spike_part_log_likelihood",
        "place_fields",
    },
    "sorted_spikes_kde": _COMMON
    | {
        "is_track_interior",
        "log_occupancy",
        "marginal_models",
        "mean_rates",
        "no_spike_part_log_likelihood",
        "occupancy",
        "occupancy_model",
        "place_fields",
    },
    "sorted_spikes_diffusion": _COMMON
    | {
        "bin_sizes",
        "interior_log_place_fields",
        "is_track_interior",
        "local_interpolation",
        "mean_rates",
        "no_spike_part_log_likelihood",
        "node_order",
        "occupancy",
        "place_fields",
    },
    "sorted_spikes_mrf": _COMMON
    | {
        "bin_sizes",
        "interior_log_place_fields",
        "is_track_interior",
        "local_interpolation",
        "mean_rates",
        "mrf_coefficients",
        "mrf_converged",
        "mrf_log_penalty_bounds",
        "mrf_max_step",
        "mrf_n_iter",
        "mrf_penalty",
        "mrf_penalty_selected_by_reml",
        "mrf_penalty_weights",
        "mrf_rank",
        "mrf_reml_objective",
        "no_spike_part_log_likelihood",
        "node_order",
        "occupancy",
        "place_fields",
    },
    "clusterless_kde": _COMMON
    | {
        "block_size",
        "enc_tile_size",
        "encoding_positions",
        "encoding_spike_waveform_features",
        "encoding_weights",
        "gpi_models",
        "log_occupancy",
        "mark_dimensions",
        "mean_rates",
        "occupancy",
        "occupancy_model",
        "pos_tile_size",
        "position_std",
        "summed_ground_process_intensity",
        "use_streaming",
        "waveform_std",
    },
    "clusterless_gmm": _COMMON
    | {
        "bin_tile_size",
        "gmm_effective_components",
        "gmm_requested_components",
        "gpi_models",
        "interior_place_bin_centers",
        "joint_models",
        "log_occupancy",
        "mark_dimensions",
        "mean_rates",
        "occupancy_model",
        "spike_block_size",
        "summed_ground_process_intensity",
    },
    "clusterless_diffusion": _COMMON
    | {
        "bin_sizes",
        "block_size",
        "encoding_bin_indices",
        "encoding_marks",
        "encoding_weights",
        "mark_dimensions",
        "mean_rates",
        "memory_budget",
        "node_order",
        "occupancy",
        "position_std",
        "resolved_rank",
        "summed_ground_process_intensity",
        "waveform_std",
        "weight_total",
    },
    "clusterless_mark_glm": _COMMON
    | {
        "emission_design_info",
        "log_rate_maps",
        "mark_dimensions",
        "mark_mixtures",
        "mean_rates",
        "rate_coefficients",
        "spike_block_size",
        "summed_ground_process_intensity",
    },
}
_REGISTRY = {**_SORTED_SPIKES_ALGORITHMS, **_CLUSTERLESS_ALGORITHMS}


def test_every_registered_algorithm_has_pinned_keys():
    assert set(_REGISTRY) == set(_STORED_KEYS)


@pytest.mark.parametrize("algorithm", sorted(_STORED_KEYS))
def test_encoding_model_keys_are_the_stored_format(algorithm):
    model_type = _REGISTRY[algorithm].encoding_model_type

    assert set(model_type.__optional_keys__) == set()
    assert set(model_type.__required_keys__) == _STORED_KEYS[algorithm], (
        "The encoding model's keys are stored in saved models: changing them "
        "needs a format version and a migration, then update this list."
    )


@pytest.fixture(scope="module")
def sorted_file(sorted_recording, tmp_path_factory):
    time, position, spike_times = sorted_recording
    path = tmp_path_factory.mktemp("encoding") / "model.nld"
    nld.SortedSpikesDecoder(
        sampling_frequency=SORTED_RATE,
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    ).fit(time, position, spike_times).save_model(path)
    return path


@pytest.fixture
def saved(sorted_file, tmp_path):
    return shutil.copy(sorted_file, tmp_path / "model.nld")


def _edit_model(path, edit):
    _edit_json(
        path, "state.json", lambda state: edit(state["encoding_models"][0]["model"])
    )


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda model: model.update(future_key=1), "future_key"),
        (lambda model: model.pop("place_fields"), "'place_fields'"),
        (lambda model: model.update(algorithm="mylab_kde"), "'mylab_kde'"),
        (lambda model: model.pop("algorithm"), "algorithm"),
    ],
    ids=["unknown_key", "missing_key", "unknown_algorithm", "no_algorithm"],
)
def test_an_encoding_model_off_its_type_is_refused(saved, edit, match):
    _edit_model(saved, edit)

    with pytest.raises(DataError, match=match):
        nld.SortedSpikesDecoder.load_model(saved)
    with pytest.raises(DataError, match=match):
        read_model_summary(saved)
