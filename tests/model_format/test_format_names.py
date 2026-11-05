"""The names a saved model stores are pinned per format version.

Reading is strict, so a release refuses a name it does not know. A name added
to, removed from or renamed in what ``save_model`` writes therefore needs a
new ``FORMAT_VERSION`` (so earlier releases ask for an upgrade) and a
``MIGRATIONS`` entry (so this release reads the earlier version). This test
fails when the names change without that: add the new version's names under
its number and keep earlier versions' entries as they are.
"""

import inspect

import pytest

from non_local_detector.model_format import FORMAT_VERSION
from non_local_detector.model_format._builtins import builtin_types
from non_local_detector.model_format._detector import (
    _ENTRY_FIELDS,
    _FINGERPRINT_FIELDS,
    _FITTED_STATE,
    _MANIFEST_NAMES,
    _OPTIONAL_STORED,
    _STORED,
)
from non_local_detector.models.base import _DetectorBase

pytestmark = pytest.mark.unit

_DETECTOR_PARAMS = (
    "discrete_initial_conditions",
    "continuous_initial_conditions_types",
    "discrete_transition_type",
    "discrete_transition_concentration",
    "discrete_transition_stickiness",
    "discrete_transition_regularization",
    "continuous_transition_types",
    "observation_models",
    "environments",
    "infer_track_interior",
    "state_names",
    "sampling_frequency",
    "no_spike_rate",
    "discrete_transition_prior_weight",
    "frozen_discrete_transition_rows",
)
_SORTED = ("sorted_spikes_algorithm", "sorted_spikes_algorithm_params")
_CLUSTERLESS = ("clusterless_algorithm", "clusterless_algorithm_params")
_LOCAL_STD = ("local_position_std",)
_NON_LOCAL = ("non_local_position_penalty", "non_local_penalty_std", *_LOCAL_STD)

# Format version -> the names it stores: manifest.json, state.json, the
# fields of state.json's list entries and fingerprints, and the fields of
# every built-in type (a detector's are its constructor parameters, which
# config.json stores).
_PINNED = {
    1: {
        "manifest": {
            "arrays",
            "detector_type",
            "detector_type_owner",
            "format",
            "format_version",
            "package_version",
        },
        "state": {
            "continuous_initial_conditions",
            "converged",
            "degenerate_timesteps",
            "discrete_initial_conditions",
            "discrete_state_transitions",
            "discrete_transition_coefficient_covariance",
            "discrete_transition_coefficients",
            "discrete_transition_covariate_data",
            "discrete_transition_smoothness",
            "em_monotonicity_violations",
            "encoding_models",
            "environments",
            "fingerprints",
            "fitted",
            "frozen_discrete_transition_rows_baseline",
            "initial_conditions",
            "log_likelihood",
            "mark_reference",
            "max_dense_transition_bytes",
            "n_iter",
            "stored_transition_blocks",
            "time_contract",
            "transition_representation",
            "transition_time_bin_width",
            "transition_time_bin_width_tolerance",
        },
        "entries": {
            "environments": {
                "all_pairs_distances",
                "config",
                "edges",
                "infer_track_interior",
                "is_track_interior",
                "max_dense_distance_bytes",
                "place_bin_centers",
                "place_bin_edges",
            },
            "encoding_models": {"encoding_group", "environment_name", "model"},
            "stored_transition_blocks": {"block", "from_state", "to_state"},
            "fingerprints": {
                "continuous_transitions",
                "discrete_transition_design",
                "environments",
            },
        },
        "types": {
            "nld.Environment": {
                "environment_name",
                "place_bin_size",
                "track_graph",
                "edge_order",
                "edge_spacing",
                "is_track_interior",
                "position_range",
                "infer_track_interior",
                "fill_holes",
                "dilate",
                "bin_count_threshold",
            },
            "nld.ObservationModel": {
                "environment_name",
                "encoding_group",
                "is_local",
                "is_no_spike",
            },
            "nld.UniformInitialConditions": set(),
            "nld.RandomWalk": {
                "environment_name",
                "movement_var",
                "movement_mean",
                "use_manifold_distance",
                "direction",
            },
            "nld.Uniform": {"environment_name", "environment2_name"},
            "nld.Identity": {"environment_name"},
            "nld.EmpiricalMovement": {
                "environment_name",
                "encoding_group",
                "speedup",
                "is_time_reversed",
            },
            "nld.RandomWalkDirection1": {"environment_name", "movement_var"},
            "nld.RandomWalkDirection2": {"environment_name", "movement_var"},
            "nld.Discrete": set(),
            "nld.DiscreteStationaryDiagonal": {"diagonal_values"},
            "nld.DiscreteStationaryCustom": {"values"},
            "nld.DiscreteNonStationaryDiagonal": {
                "diagonal_values",
                "formula",
                "smoothness",
            },
            "nld.DiscreteNonStationaryCustom": {"values", "formula", "smoothness"},
            "nld.KDEModel": {"std", "block_size", "samples_", "weights_"},
            "nld.GaussianMixtureModel": {
                "n_components",
                "covariance_type",
                "tol",
                "reg_covar",
                "max_iter",
                "n_init",
                "init_params",
                "kmeans_init",
                "kmeans_n_init",
                "random_state",
                "weights_init",
                "means_init",
                "covariances_init",
                "weights_",
                "means_",
                "covariances_",
                "precisions_chol_",
                "converged_",
                "n_iter_",
                "lower_bound_",
            },
            "nld.MarkMixture": {"weights", "means", "precisions_cholesky"},
            "nld.MarkReference": {"location", "scale", "degrees_of_freedom"},
            "nld.TrackGraphBasis": {
                "edge_starts",
                "edge_lengths",
                "edge_branches",
                "edge_branch_starts",
                "edge_forward",
                "branch_knots",
                "branch_columns",
                "branch_periodic",
                "n_columns",
            },
            "nld.TensorSplineBasis": {
                "basis_positions",
                "place_bin_edges",
                "knot_spacing",
                "columns",
                "design_fingerprint",
            },
            "nld.SortedSpikesDetector": {*_DETECTOR_PARAMS, *_SORTED, *_LOCAL_STD},
            "nld.ClusterlessDetector": {
                *_DETECTOR_PARAMS,
                *_CLUSTERLESS,
                *_LOCAL_STD,
            },
            "nld.NonLocalSortedSpikesDetector": {
                *_DETECTOR_PARAMS,
                *_SORTED,
                *_NON_LOCAL,
            },
            "nld.NonLocalClusterlessDetector": {
                *_DETECTOR_PARAMS,
                *_CLUSTERLESS,
                *_NON_LOCAL,
            },
            "nld.ContFragSortedSpikesClassifier": {*_DETECTOR_PARAMS, *_SORTED},
            "nld.ContFragClusterlessClassifier": {*_DETECTOR_PARAMS, *_CLUSTERLESS},
            "nld.NoSpikeContFragSortedSpikesClassifier": {
                *_DETECTOR_PARAMS,
                *_SORTED,
            },
            "nld.NoSpikeContFragClusterlessClassifier": {
                *_DETECTOR_PARAMS,
                *_CLUSTERLESS,
            },
            "nld.MultiEnvironmentSortedSpikesClassifier": {
                *_DETECTOR_PARAMS,
                *_SORTED,
            },
            "nld.MultiEnvironmentClusterlessClassifier": {
                *_DETECTOR_PARAMS,
                *_CLUSTERLESS,
            },
            "nld.SortedSpikesDecoder": {*_DETECTOR_PARAMS, *_SORTED},
            "nld.ClusterlessDecoder": {*_DETECTOR_PARAMS, *_CLUSTERLESS},
        },
    },
}


def _stored_names():
    """The names this release writes, in the layout of ``_PINNED``."""
    types = {}
    for name, cls, fields, _ in builtin_types():
        if issubclass(cls, _DetectorBase):
            fields = tuple(inspect.signature(cls).parameters)
        types[name] = set(fields or ())
    return {
        "manifest": set(_MANIFEST_NAMES),
        "state": {
            *_FITTED_STATE,
            *(name for name, _ in _STORED),
            *_OPTIONAL_STORED,
            "discrete_state_transitions",
        },
        "entries": {
            **{name: set(fields) for name, fields in _ENTRY_FIELDS.items()},
            "fingerprints": set(_FINGERPRINT_FIELDS),
        },
        "types": types,
    }


def test_the_current_format_version_has_pinned_names():
    assert FORMAT_VERSION in _PINNED, (
        f"FORMAT_VERSION is {FORMAT_VERSION}: pin its stored names in _PINNED "
        "under that number, keeping the entries of earlier versions."
    )


def test_the_stored_names_are_those_of_the_format_version():
    pinned = _PINNED[FORMAT_VERSION]
    current = _stored_names()

    for part in ("manifest", "state", "entries", "types"):
        assert current[part] == pinned[part], (
            f"The stored {part} names differ from format version "
            f"{FORMAT_VERSION}'s. A release refuses names it does not know, "
            "so bump FORMAT_VERSION, add a MIGRATIONS entry that rewrites "
            f"version {FORMAT_VERSION} files, and pin the new names under the "
            "new version here (do not edit an earlier version's names)."
        )
