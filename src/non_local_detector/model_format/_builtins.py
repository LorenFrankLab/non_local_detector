"""The package's own saved types and the fields each one stores.

The field names are part of the file format: renaming a field of one of these
classes needs a format migration, and ``tests/model_format`` checks that every
dataclass listed here still has exactly these fields.
"""

from collections.abc import Callable, Mapping
from typing import Any

from non_local_detector.model_format._registry import read_fields

# (stored name, class, stored fields, options); fields None means "not a
# dataclass". Options may give "defaults" for optional fields (see
# ``register_type``); a field added to one of these classes needs one, so
# files written before it still load.
BuiltinType = tuple[str, type, tuple[str, ...] | None, dict[str, Any]]

ENVIRONMENT_FIELDS = (
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
)


def _with_fitted_fields(
    cls: type,
    init: tuple[str, ...],
    fitted: tuple[str, ...],
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Options of a dataclass built from ``init`` with ``fitted`` set after."""

    def to_state(obj: Any) -> dict[str, Any]:
        return {name: getattr(obj, name) for name in (*init, *fitted)}

    def from_state(state: dict[str, Any]) -> Any:
        values = read_fields(state, (*init, *fitted), defaults or {}, cls)
        obj = cls(**{name: values[name] for name in init})
        for name in fitted:
            object.__setattr__(obj, name, values[name])
        return obj

    return {"to_state": to_state, "from_state": from_state}


def _restored(
    cls: type, names: tuple[str, ...], defaults: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Options of a class restored through its ``__setstate__``."""

    def to_state(obj: Any) -> dict[str, Any]:
        return {name: getattr(obj, name) for name in names}

    def from_state(state: dict[str, Any]) -> Any:
        obj: Any = object.__new__(cls)
        obj.__setstate__(read_fields(state, names, defaults or {}, cls))
        return obj

    return {"to_state": to_state, "from_state": from_state}


def _stateless(cls: type) -> dict[str, Callable[..., Any]]:
    def from_state(state: dict[str, Any]) -> Any:
        read_fields(state, (), {}, cls)  # warns of any field
        return cls()

    return {"to_state": lambda obj: {}, "from_state": from_state}  # noqa: ARG005 -- stateless


def builtin_types() -> list[BuiltinType]:
    """Every type the package saves, with its stored fields."""
    from non_local_detector import models
    from non_local_detector.continuous_state_transitions import (
        Discrete,
        EmpiricalMovement,
        Identity,
        RandomWalk,
        RandomWalkDirection1,
        RandomWalkDirection2,
        Uniform,
    )
    from non_local_detector.discrete_state_transitions import (
        DiscreteNonStationaryCustom,
        DiscreteNonStationaryDiagonal,
        DiscreteStationaryCustom,
        DiscreteStationaryDiagonal,
    )
    from non_local_detector.environment import Environment
    from non_local_detector.initial_conditions import UniformInitialConditions
    from non_local_detector.likelihoods.clusterless_mark_glm import MarkMixture
    from non_local_detector.likelihoods.common.kernel_density import KDEModel
    from non_local_detector.likelihoods.gmm import GaussianMixtureModel
    from non_local_detector.likelihoods.mark_reference import MarkReference
    from non_local_detector.likelihoods.sorted_spikes_glm import (
        TensorSplineBasis,
        TrackGraphBasis,
    )
    from non_local_detector.models.base import (
        ClusterlessDetector,
        SortedSpikesDetector,
        _DetectorBase,
    )
    from non_local_detector.observation_models import ObservationModel

    gmm_init = (
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
    )
    gmm_fitted = (
        "weights_",
        "means_",
        "covariances_",
        "precisions_chol_",
        "converged_",
        "n_iter_",
        "lower_bound_",
    )
    # The stored type name of each class: part of the format, so written out
    # rather than derived from the class name, which a rename would change.
    names: dict[type, str] = {
        Environment: "nld.Environment",
        ObservationModel: "nld.ObservationModel",
        UniformInitialConditions: "nld.UniformInitialConditions",
        RandomWalk: "nld.RandomWalk",
        Uniform: "nld.Uniform",
        Identity: "nld.Identity",
        EmpiricalMovement: "nld.EmpiricalMovement",
        RandomWalkDirection1: "nld.RandomWalkDirection1",
        RandomWalkDirection2: "nld.RandomWalkDirection2",
        Discrete: "nld.Discrete",
        DiscreteStationaryDiagonal: "nld.DiscreteStationaryDiagonal",
        DiscreteStationaryCustom: "nld.DiscreteStationaryCustom",
        DiscreteNonStationaryDiagonal: "nld.DiscreteNonStationaryDiagonal",
        DiscreteNonStationaryCustom: "nld.DiscreteNonStationaryCustom",
        KDEModel: "nld.KDEModel",
        GaussianMixtureModel: "nld.GaussianMixtureModel",
        MarkMixture: "nld.MarkMixture",
        MarkReference: "nld.MarkReference",
        TrackGraphBasis: "nld.TrackGraphBasis",
        TensorSplineBasis: "nld.TensorSplineBasis",
        SortedSpikesDetector: "nld.SortedSpikesDetector",
        ClusterlessDetector: "nld.ClusterlessDetector",
        models.NonLocalSortedSpikesDetector: "nld.NonLocalSortedSpikesDetector",
        models.NonLocalClusterlessDetector: "nld.NonLocalClusterlessDetector",
        models.ContFragSortedSpikesClassifier: "nld.ContFragSortedSpikesClassifier",
        models.ContFragClusterlessClassifier: "nld.ContFragClusterlessClassifier",
        models.NoSpikeContFragSortedSpikesClassifier: (
            "nld.NoSpikeContFragSortedSpikesClassifier"
        ),
        models.NoSpikeContFragClusterlessClassifier: (
            "nld.NoSpikeContFragClusterlessClassifier"
        ),
        models.MultiEnvironmentSortedSpikesClassifier: (
            "nld.MultiEnvironmentSortedSpikesClassifier"
        ),
        models.MultiEnvironmentClusterlessClassifier: (
            "nld.MultiEnvironmentClusterlessClassifier"
        ),
        models.SortedSpikesDecoder: "nld.SortedSpikesDecoder",
        models.ClusterlessDecoder: "nld.ClusterlessDecoder",
    }
    detectors: list[tuple[type, None, dict[str, Any]]] = [
        (cls, None, {}) for cls in names if issubclass(cls, _DetectorBase)
    ]
    types: list[tuple[type, tuple[str, ...] | None, dict[str, Any]]] = [
        (Environment, ENVIRONMENT_FIELDS, {}),
        (
            ObservationModel,
            ("environment_name", "encoding_group", "is_local", "is_no_spike"),
            {},
        ),
        (UniformInitialConditions, (), {}),
        (
            RandomWalk,
            (
                "environment_name",
                "movement_var",
                "movement_mean",
                "use_manifold_distance",
                "direction",
            ),
            {},
        ),
        (Uniform, ("environment_name", "environment2_name"), {}),
        (Identity, ("environment_name",), {}),
        (
            EmpiricalMovement,
            ("environment_name", "encoding_group", "speedup", "is_time_reversed"),
            {"block_needs_fit_data": True},
        ),
        (RandomWalkDirection1, ("environment_name", "movement_var"), {}),
        (RandomWalkDirection2, ("environment_name", "movement_var"), {}),
        (Discrete, None, _stateless(Discrete)),
        (DiscreteStationaryDiagonal, ("diagonal_values",), {}),
        (DiscreteStationaryCustom, ("values",), {}),
        (
            DiscreteNonStationaryDiagonal,
            ("diagonal_values", "formula", "smoothness"),
            {},
        ),
        (DiscreteNonStationaryCustom, ("values", "formula", "smoothness"), {}),
        (
            KDEModel,
            ("std", "block_size", "samples_", "weights_"),
            _with_fitted_fields(
                KDEModel, ("std", "block_size"), ("samples_", "weights_")
            ),
        ),
        (
            GaussianMixtureModel,
            (*gmm_init, *gmm_fitted),
            _with_fitted_fields(GaussianMixtureModel, gmm_init, gmm_fitted),
        ),
        (MarkMixture, ("weights", "means", "precisions_cholesky"), {}),
        (MarkReference, ("location", "scale", "degrees_of_freedom"), {}),
        (
            TrackGraphBasis,
            (
                "edge_starts",
                "edge_lengths",
                "edge_branches",
                "edge_branch_starts",
                "edge_forward",
                "branch_knots",
                "branch_columns",
                "branch_periodic",
                "n_columns",
            ),
            {},
        ),
        # Restored without rebuilding its patsy design, which is rebuilt and
        # checked against ``design_fingerprint`` on first use.
        (
            TensorSplineBasis,
            (
                "basis_positions",
                "place_bin_edges",
                "knot_spacing",
                "columns",
                "design_fingerprint",
            ),
            _restored(
                TensorSplineBasis,
                (
                    "basis_positions",
                    "place_bin_edges",
                    "knot_spacing",
                    "columns",
                    "design_fingerprint",
                ),
            ),
        ),
        *detectors,
    ]
    return [(names[cls], cls, fields, options) for cls, fields, options in types]
