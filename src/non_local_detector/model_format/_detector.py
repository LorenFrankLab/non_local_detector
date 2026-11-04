"""The saved state of a detector: what is stored, what is rebuilt, and checks.

``config.json`` holds the detector's ``get_params()``; loading calls the
detector class with it. ``state.json`` holds the fitted state under the names
in ``_STORED`` (file name, attribute), plus:

``environments``
    Each fitted environment (``environments_``): its configuration, and for an
    environment without a track graph the bin ``edges`` and interior mask it
    was fitted with. The grid is rebuilt from these (a track graph's from its
    configuration alone). The bin centers and mask are stored too, for
    reading the grid without non_local_detector.
``encoding_models``
    ``encoding_model_`` as ``{environment_name, encoding_group, model}``
    entries; each model's ``environment`` refers to a fitted environment by
    name.
``stored_transition_blocks``
    Continuous transition blocks built from the fit data (registered with
    ``block_needs_fit_data``, as ``EmpiricalMovement``), which the file does not
    keep: ``{from_state, to_state, block}``.
``fingerprints``
    Small float64 summaries of the structures rebuilt on load.

Rebuilt on load and not stored: the environments' grids and distances, the
state-bin index, the continuous transition (dense matrix or structured
operator), and, for covariate-dependent discrete transitions, the patsy design,
its smoothness penalties and the training-time transition matrices. Before a
loaded detector first decodes, the rebuilt structures are compared with their
fingerprints; a mismatch refuses to decode and asks for a refit.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from non_local_detector._warnings import user_stacklevel
from non_local_detector.continuous_state_transitions import (
    _declared_destination,
    _no_structured_block,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import NotFittedError, ValidationError
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.model_format._codec import (
    JSON,
    ArrayWriter,
    Decoder,
    Encoder,
)
from non_local_detector.model_format._container import (
    FORMAT_NAME,
    FORMAT_VERSION,
    StoredModel,
    new_model_file,
    open_model,
    write_document,
)
from non_local_detector.model_format._registry import (
    TypeSpec,
    spec_for_class,
    spec_for_name,
    spec_for_object,
    unregistered_type_error,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase

# (file name, detector attribute) of the fitted state stored as is. An
# attribute the detector does not have is left out of the file.
_STORED = (
    ("initial_conditions", "initial_conditions_"),
    ("continuous_initial_conditions", "continuous_initial_conditions_"),
    ("discrete_initial_conditions", "discrete_initial_conditions_"),
    ("discrete_transition_coefficients", "discrete_transition_coefficients_"),
    (
        "discrete_transition_coefficient_covariance",
        "discrete_transition_coefficient_covariance_",
    ),
    ("discrete_transition_smoothness", "discrete_transition_smoothness_"),
    ("discrete_transition_covariate_data", "_discrete_transition_covariate_data_"),
    (
        "frozen_discrete_transition_rows_baseline",
        "_frozen_discrete_transition_rows_baseline_",
    ),
    ("transition_time_bin_width", "transition_time_bin_width_"),
    ("transition_time_bin_width_tolerance", "_transition_time_bin_width_tolerance"),
    ("transition_representation", "_transition_representation_"),
    ("max_dense_transition_bytes", "_max_dense_transition_bytes_"),
    ("time_contract", "time_contract_"),
    ("mark_reference", "mark_reference_"),
    ("converged", "converged_"),
    ("n_iter", "n_iter_"),
    ("em_monotonicity_violations", "em_monotonicity_violations_"),
    ("degenerate_timesteps", "degenerate_timesteps_"),
    ("log_likelihood", "log_likelihood_"),
)
# Fitted attributes rebuilt on load (or written separately, as
# ``environments_`` and ``encoding_model_``).
_REBUILT = frozenset(
    {
        "environments_",
        "encoding_model_",
        "n_discrete_states_",
        "state_ind_",
        "n_state_bins_",
        "bin_sizes_",
        "is_track_interior_state_bins_",
        "continuous_state_transitions_",
        "_continuous_transition_operator_",
        "discrete_state_transitions_",
        "discrete_transition_design_matrix_",
        "discrete_transition_smoothness_matrix_",
    }
)
# Not part of a model: the pending rebuild check of a loaded detector.
_NOT_SAVED = frozenset({"_rebuild_check_"})
# Every fitted detector has these.
_REQUIRED = ("environments_", "encoding_model_", "continuous_state_transitions_")

# A rebuilt structure matches its fingerprint to this relative tolerance, plus
# this absolute tolerance per summed point (as TensorSplineBasis checks its
# spline design): a rebuild through exp/log differs in the last bits between
# platforms, but a changed formula does not pass.
_REBUILD_RTOL = 1e-10
_REBUILD_ATOL_PER_POINT = 1e-12


# --------------------------------------------------------------- fingerprints
def _column_summary(values: Any) -> np.ndarray:
    """Ramp-weighted column sums and column sums of squares, shape (2, n_columns)."""
    values = np.asarray(values, dtype=np.float64)
    # (n_points, n_columns): a vector is one column; n_points may be 0 (a
    # one-bin grid has no bin-graph edges), which sums to zeros.
    if values.ndim == 1:
        values = values[:, np.newaxis]
    values = values.reshape(values.shape[0], int(np.prod(values.shape[1:])))
    ramp = np.linspace(1.0, 2.0, len(values))
    return np.stack([ramp @ values, np.sum(values**2, axis=0)])


def _environment_fingerprint(environment: Environment) -> np.ndarray:
    """Summary of a fitted grid: bin centers and edges, interior count and
    the bin graph's edge distances (from which the distances follow)."""
    graph = (
        environment.track_graphDD
        if environment.track_graph is None
        else environment.track_graph_with_bin_centers_edges_
    )
    assert graph is not None  # a fitted grid has a bin graph
    distances = np.array([d for *_, d in graph.edges(data="distance")], dtype=float)
    return np.concatenate(
        [
            _column_summary(environment.place_bin_centers_).ravel(),
            _column_summary(environment.place_bin_edges_).ravel(),
            [float(np.sum(environment.is_track_interior_))],
            _column_summary(distances[:, np.newaxis]).ravel(),
        ]
    )


def _fingerprint_weights(n: int) -> np.ndarray:
    """Two fixed weight patterns over ``n`` bins, shape (n, 2): a ramp from 1
    to 2 and the fractional parts of multiples of the golden ratio, plus 1.

    The second is not a function of the first, so a change that keeps one
    weighted sum cannot keep both.
    """
    index = np.arange(n, dtype=np.float64)
    ramp = np.linspace(1.0, 2.0, n)
    golden = 1.0 + np.mod(index * 0.6180339887498949, 1.0)
    return np.stack([ramp, golden], axis=1)


def _block_summary(block: Any, n_rows: int, n_columns: int) -> np.ndarray:
    """Weighted sums over every row and every column of a transition block.

    ``block @ W_columns`` (each row's two weighted sums) followed by
    ``W_rows.T @ block`` (each column's), raveled. ``block`` is a dense array,
    or a structured operator block whose ``forward``/``backward`` products
    are evaluated in float64, so nothing larger than the block's own
    operator is built.
    """
    row_weights = _fingerprint_weights(n_rows)
    column_weights = _fingerprint_weights(n_columns)
    if isinstance(block, np.ndarray):
        values = np.asarray(block, dtype=np.float64)
        by_row = values @ column_weights
        by_column = row_weights.T @ values
    else:
        import jax
        import jax.numpy as jnp

        from non_local_detector.discrete_state_transitions import _float64_jax

        with _float64_jax(), jax.default_matmul_precision("highest"):
            by_row = np.stack(
                [
                    np.asarray(block.backward(jnp.asarray(column_weights[:, k])))
                    for k in range(2)
                ],
                axis=1,
            )
            by_column = np.stack(
                [
                    np.asarray(block.forward(jnp.asarray(row_weights[:, k])))
                    for k in range(2)
                ]
            )
    return np.concatenate([by_row.ravel(), by_column.ravel()]).astype(np.float64)


def _transition_fingerprints(
    detector: "_DetectorBase", stored_blocks: Mapping[tuple[int, int], Any]
) -> list[Any]:
    """``[from_state, to_state, summary]`` of each rebuilt continuous block.

    The summary (``_block_summary``) weights every row and every column, so
    a change to any entry shows. It reads the dense matrix's block, or the
    structured operator's block through its products.
    """
    transitions = detector.continuous_state_transitions_
    operator = getattr(transitions, "operator", None)
    state_ind = np.asarray(detector.state_ind_)
    n_states = int(detector.n_discrete_states_)
    fingerprints = []
    for source in range(n_states):
        rows = np.flatnonzero(state_ind == source)
        for target in range(n_states):
            if (source, target) in stored_blocks:
                continue
            columns = np.flatnonzero(state_ind == target)
            block = (
                np.asarray(transitions[np.ix_(rows, columns)])
                if operator is None
                else operator.blocks[source][target]
            )
            summary = _block_summary(block, len(rows), len(columns))
            fingerprints.append([source, target, summary])
    return fingerprints


def _structure_fingerprints(
    detector: "_DetectorBase", stored_blocks: Mapping[tuple[int, int], Any]
) -> dict[str, Any]:
    design = getattr(detector, "discrete_transition_design_matrix_", None)
    return {
        "environments": [
            _environment_fingerprint(env) for env in detector.environments_
        ],
        "continuous_transitions": _transition_fingerprints(detector, stored_blocks),
        "discrete_transition_design": None
        if design is None
        else _column_summary(design),
    }


def _mismatch(saved: Any, rebuilt: Any, n_points: int) -> str | None:
    """Why ``rebuilt`` does not match ``saved``, or None if it does."""
    saved, rebuilt = np.asarray(saved), np.asarray(rebuilt)
    if saved.shape != rebuilt.shape:
        return f"shape {rebuilt.shape} instead of {saved.shape}"
    if np.allclose(
        rebuilt,
        saved,
        rtol=_REBUILD_RTOL,
        atol=_REBUILD_ATOL_PER_POINT * max(n_points, 1),
    ):
        return None
    return f"largest fingerprint difference {np.max(np.abs(rebuilt - saved)):.3g}"


class _RebuildCheck:
    """The fingerprints a loaded detector's rebuilt structures must match.

    ``verify`` runs before the first decode; once it passes it is not repeated.
    """

    def __init__(
        self,
        fingerprints: dict[str, Any],
        stored_blocks: Sequence[tuple[int, int]],
        package_version: str,
        refusals: Sequence[str] = (),
    ) -> None:
        self.fingerprints = fingerprints
        self.stored_blocks = tuple(stored_blocks)
        self.package_version = package_version
        self.refusals = tuple(refusals)
        self.verified = False

    def mismatches(self, detector: "_DetectorBase") -> list[str]:
        """Each rebuilt structure that differs from its saved fingerprint."""
        saved = self.fingerprints
        rebuilt = _structure_fingerprints(detector, dict.fromkeys(self.stored_blocks))
        problems = []
        for env, before, after in zip(
            detector.environments_,
            saved["environments"],
            rebuilt["environments"],
            strict=True,
        ):
            why = _mismatch(before, after, env.place_bin_centers_.shape[0])
            if why is not None:
                problems.append(f"environment {env.environment_name!r} ({why})")
        for (source, target, before), (_, _, after) in zip(
            saved["continuous_transitions"],
            rebuilt["continuous_transitions"],
            strict=True,
        ):
            why = _mismatch(before, after, max(detector.bin_sizes_[[source, target]]))
            if why is not None:
                problems.append(
                    f"continuous transition from state {source} to {target} ({why})"
                )
        before, after = (
            saved["discrete_transition_design"],
            rebuilt["discrete_transition_design"],
        )
        if (before is None) != (after is None):
            problems.append("discrete transition design (presence)")
        elif before is not None:
            design = detector.discrete_transition_design_matrix_
            assert design is not None  # rebuilt with the fingerprint's presence
            n_rows = len(design)
            why = _mismatch(before, after, n_rows)
            if why is not None:
                problems.append(f"discrete transition covariate design ({why})")
        return problems

    def verify(self, detector: "_DetectorBase") -> None:
        """Raise if a rebuilt structure differs from the saved one.

        Raises
        ------
        ValidationError
            Naming each structure that differs.
        """
        if self.verified:
            return
        if self.refusals:
            msg = (
                "The saved model cannot decode: "
                f"{'; '.join(self.refusals)}. It loaded for inspection only."
            )
            raise ValidationError(
                msg, hint="Refit the model with a formula the grammar allows."
            )
        problems = self.mismatches(detector)
        if problems:
            msg = (
                "The model's structures, rebuilt from the saved model, differ "
                f"from those it was fitted with: {'; '.join(problems)}. The "
                "rebuilding code or a dependency changed since non_local_"
                f"detector {self.package_version} saved it."
            )
            raise ValidationError(
                msg,
                hint=(
                    "Inspection still works. To decode, refit the model with "
                    "the installed versions, or decode with the version that "
                    "saved it."
                ),
            )
        self.verified = True


# ----------------------------------------------------------- stored blocks
@dataclass(frozen=True)
class _FittedBlock:
    """A continuous transition whose saved block replaces its constructor.

    Used only while rebuilding a loaded detector's continuous transition,
    for blocks built from fit data the file does not keep.
    """

    environment_name: str | None
    environment2_name: str | None
    block: np.ndarray

    def make_state_transition(
        self,
        environments: Sequence[Environment],  # noqa: ARG002 -- ContinuousTransition signature
        context: Any = None,  # noqa: ARG002 -- ContinuousTransition signature
    ) -> np.ndarray:
        return self.block

    def structured_block(self, environments: Sequence[Environment]) -> Any:  # noqa: ARG002 -- SupportsStructuredBlock signature
        return _no_structured_block(self)

    def dense_block_shape(self, environments: Sequence[Environment]) -> tuple[int, int]:  # noqa: ARG002 -- SupportsStructuredBlock signature
        return cast("tuple[int, int]", self.block.shape)

    def graph_distance_environment(self, environments: Sequence[Environment]) -> None:  # noqa: ARG002 -- SupportsStructuredBlock signature
        return None


def _data_blocks(detector: "_DetectorBase") -> dict[tuple[int, int], np.ndarray]:
    """The continuous blocks built from fit data, by (from_state, to_state)."""
    state_ind = np.asarray(detector.state_ind_)
    blocks = {}
    for source, row in enumerate(detector.continuous_transition_types):
        for target, transition in enumerate(row):
            spec = spec_for_object(transition)
            if spec is None or not spec.block_needs_fit_data:
                continue
            rows = np.flatnonzero(state_ind == source)
            columns = np.flatnonzero(state_ind == target)
            if len(rows) == 1 or len(columns) == 1:
                continue  # built from the one-bin rule, not the transition
            blocks[(source, target)] = np.array(
                detector.continuous_state_transitions_[np.ix_(rows, columns)]
            )
    return blocks


# ------------------------------------------------------------- environments
def _environment_state(environment: Environment, encoder: Encoder, where: str) -> JSON:
    grid = environment.place_grid_
    state: dict[str, Any] = {
        "config": encoder.encode_object(environment, f"{where}/config"),
        "place_bin_centers": encoder.encode(
            grid.place_bin_centers, f"{where}/place_bin_centers"
        ),
        "is_track_interior": encoder.encode(
            np.asarray(grid.is_track_interior), f"{where}/is_track_interior"
        ),
    }
    state["edges"] = [
        encoder.encode(np.asarray(edge), f"{where}/edges_{i}")
        for i, edge in enumerate(grid.edges)
    ]
    if environment.track_graph is not None:
        state["place_bin_edges"] = encoder.encode(
            np.asarray(grid.place_bin_edges), f"{where}/place_bin_edges"
        )
    else:
        distances = grid.distance_between_nodes
        state.update(
            infer_track_interior=grid.infer_track_interior,
            all_pairs_distances=not isinstance(distances, LazyGraphDistances),
            max_dense_distance_bytes=distances.max_dense_bytes
            if isinstance(distances, LazyGraphDistances)
            else None,
        )
    return state


def _environment_from_state(state: dict[str, Any], decoder: Decoder) -> Environment:
    environment = cast(Environment, decoder.decode(state["config"]))
    edges = [np.asarray(decoder.decode(edge)) for edge in state["edges"]]
    if environment.track_graph is not None:
        environment.fit_place_grid()
    else:
        options: dict[str, Any] = {
            "compute_all_pairs_distances": state["all_pairs_distances"]
        }
        if state["max_dense_distance_bytes"] is not None:
            options["max_dense_distance_bytes"] = state["max_dense_distance_bytes"]
        environment.fit_place_grid_from_edges(
            edges,
            decoder.decode(state["is_track_interior"]),
            infer_track_interior=state["infer_track_interior"],
            **options,
        )
    stored = {
        "edges": tuple(edges),
        "place_bin_centers": np.asarray(decoder.decode(state["place_bin_centers"])),
        "is_track_interior": np.asarray(decoder.decode(state["is_track_interior"])),
    }
    if "place_bin_edges" in state:
        stored["place_bin_edges"] = np.asarray(decoder.decode(state["place_bin_edges"]))
    return _with_stored_grid(environment, stored)


def _same_within_tolerance(rebuilt: Any, saved: np.ndarray) -> bool:
    rebuilt = np.asarray(rebuilt)
    if rebuilt.shape != saved.shape or rebuilt.dtype != saved.dtype:
        return False
    if saved.dtype.kind != "f":
        return bool(np.array_equal(rebuilt, saved))
    return bool(
        np.allclose(
            rebuilt,
            saved,
            rtol=_REBUILD_RTOL,
            atol=_REBUILD_ATOL_PER_POINT,
            equal_nan=True,
        )
    )


def _with_stored_grid(environment: Environment, stored: dict[str, Any]) -> Environment:
    """``environment`` with the grid arrays the file stores, exactly.

    The bin edges, centers and interior mask are stored, and encoding models
    record an exact hash of them (``environment_grid_fingerprint``). A rebuild
    on another platform can differ in the last bits (``np.interp`` is
    FMA-fused on some), so when every rebuilt array matches its stored value
    within the rebuild tolerance (booleans exactly), the grid takes the stored
    values. Otherwise the rebuilt grid is kept, and the environment's
    fingerprint check refuses to decode.
    """
    grid = environment.place_grid_
    rebuilt = {
        "edges": grid.edges,
        "place_bin_centers": grid.place_bin_centers,
        "is_track_interior": grid.is_track_interior,
        "place_bin_edges": grid.place_bin_edges,
    }
    if len(stored["edges"]) != len(grid.edges) or not all(
        _same_within_tolerance(a, b)
        for a, b in zip(grid.edges, stored["edges"], strict=True)
    ):
        return environment
    if not all(
        _same_within_tolerance(rebuilt[name], value)
        for name, value in stored.items()
        if name != "edges"
    ):
        return environment
    return environment._replace_place_grid(**stored)


# ------------------------------------------------------------------ detector
def _package_version() -> str:
    import non_local_detector

    # Unset only in a source tree that was never installed.
    return str(getattr(non_local_detector, "__version__", "unknown"))


def _detector_spec(detector: "_DetectorBase") -> TypeSpec:
    spec = spec_for_class(type(detector))
    if spec is None:
        raise unregistered_type_error(type(detector), "the detector")
    return spec


def _fitted_attributes(
    detector: "_DetectorBase", params: Mapping[str, Any]
) -> set[str]:
    """The detector's attributes that its constructor does not set.

    Attributes the constructor sets from the parameters (``params`` and any it
    derives, such as a model's fixed ``local_position_std``) are rebuilt by
    calling it on load, so they must still hold the constructor's values.

    Raises
    ------
    ValidationError
        If a derived constructor attribute was changed after construction.
    """
    constructed = vars(type(detector)(**params))
    for name in set(constructed) - set(params):
        value, expected = vars(detector).get(name), constructed[name]
        same = value is expected
        if not same:
            try:
                same = bool(np.all(value == expected))
            except (TypeError, ValueError):
                same = False
        if not same:
            msg = f"Cannot save the detector: its {name} was changed after construction"
            raise ValidationError(
                msg,
                expected=repr(expected),
                got=repr(value),
                hint="Set it through a constructor parameter, which is saved.",
            )
    return set(vars(detector)) - set(constructed) - _NOT_SAVED


def detector_documents(
    detector: "_DetectorBase", writer: ArrayWriter
) -> tuple[dict[str, Any], JSON, dict[str, Any]]:
    """The manifest, config and state documents and the arrays of a detector.

    Raises
    ------
    ValidationError
        If the detector or an object it holds is of an unregistered type, it
        holds an attribute the format does not know, or it is a loaded
        detector whose rebuilt structures no longer match.
    NotFittedError
        If it is partly fitted.
    """
    spec = _detector_spec(detector)
    params = detector.get_params(deep=False)
    config = Encoder(writer).encode(params, "config")
    fitted = _fitted_attributes(detector, params)
    stored_names = {attribute for _, attribute in _STORED}
    unknown = sorted(fitted - stored_names - _REBUILT)
    if unknown:
        msg = f"Cannot save the detector's attributes {unknown}, which the saved format does not hold"
        raise ValidationError(
            msg,
            hint="Remove them before save_model, or refit the detector.",
        )
    state: dict[str, Any] = {"fitted": bool(fitted)}
    if not fitted:
        return _manifest(spec, writer), config, state
    missing = [name for name in _REQUIRED if name not in fitted]
    if missing:
        msg = f"Cannot save a partly fitted detector (it has no {missing})"
        raise NotFittedError(msg, hint="Call fit(...) before save_model.")
    check = vars(detector).get("_rebuild_check_")
    if check is not None:
        check.verify(detector)

    environments = detector.environments_
    names = {id(env): env.environment_name for env in environments}
    encoder = Encoder(writer, names)
    state["environments"] = [
        _environment_state(env, encoder, f"environments/{i}")
        for i, env in enumerate(environments)
    ]
    state["encoding_models"] = [
        {
            "environment_name": key[0],
            "encoding_group": encoder.encode(key[1], f"encoding_models/{i}/group"),
            "model": encoder.encode(dict(model), f"encoding_models/{i}/model"),
        }
        for i, (key, model) in enumerate(detector.encoding_model_.items())
    ]
    for name, attribute in _STORED:
        if attribute in fitted:
            state[name] = encoder.encode(getattr(detector, attribute), name)
    if getattr(detector, "_discrete_transition_covariate_data_", None) is None:
        state["discrete_state_transitions"] = encoder.encode(
            detector.discrete_state_transitions_, "discrete_state_transitions"
        )
    blocks = _data_blocks(detector)
    state["stored_transition_blocks"] = [
        {
            "from_state": source,
            "to_state": target,
            "block": encoder.encode(
                block, f"stored_transition_blocks/{source}_{target}"
            ),
        }
        for (source, target), block in blocks.items()
    ]
    state["fingerprints"] = encoder.encode(
        _structure_fingerprints(detector, blocks), "fingerprints"
    )
    return _manifest(spec, writer), config, state


def _manifest(spec: TypeSpec, writer: ArrayWriter) -> dict[str, Any]:
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "package_version": _package_version(),
        "detector_type": spec.name,
        "detector_type_owner": spec.owner,
        "arrays": writer.index,
    }


def save_detector(detector: "_DetectorBase", path: str | Path) -> None:
    """Write ``detector`` to ``path`` in the saved-model format."""
    with new_model_file(path) as archive:
        manifest, config, state = detector_documents(detector, ArrayWriter(archive))
        for name, document in (
            ("config.json", config),
            ("state.json", state),
            ("manifest.json", manifest),
        ):
            write_document(archive, name, document)


def _rebuild_discrete(detector: "_DetectorBase", stored: dict[str, Any]) -> str | None:
    """Rebuild the discrete transition's design, penalties and transitions.

    Returns
    -------
    refusal : str or None
        Why a covariate design was not rebuilt (its formula is outside the
        formula grammar), or None. The detector then loads for inspection
        only, without a design or penalties, and without the training-time
        ``discrete_state_transitions_``.
    """
    from non_local_detector.discrete_state_transitions import (
        _smoothness_matrix,
        _transitions_on_design,
    )

    covariate_data = detector._discrete_transition_covariate_data_
    if covariate_data is None:
        detector.discrete_transition_design_matrix_ = None
        detector.discrete_transition_smoothness_matrix_ = None
        detector.discrete_state_transitions_ = stored["discrete_state_transitions"]
        return None
    try:
        # Checks the formula against the formula grammar before patsy sees it.
        initial = detector.discrete_transition_type.make_state_transition(
            covariate_data
        )
    except ValidationError as error:
        detector.discrete_transition_design_matrix_ = None
        detector.discrete_transition_smoothness_matrix_ = None
        # The training-time transitions follow from the design: left unset.
        warnings.warn(
            "The detector's transition formula is outside the allowed formula "
            "grammar, so its design matrix was not rebuilt. It can be "
            f"inspected, but decoding is refused.\n{error}",
            UserWarning,
            stacklevel=user_stacklevel(),
        )
        return f"its discrete transition formula was refused ({error.args[0]})"
    design = initial.design_matrix
    coefficients = detector.discrete_transition_coefficients_
    # A covariate-dependent transition has a design and coefficients.
    assert design is not None
    assert coefficients is not None
    assert initial.coefficients is not None
    detector.discrete_transition_design_matrix_ = design
    detector.discrete_transition_smoothness_matrix_ = _smoothness_matrix(
        design.design_info
    )
    # Until EM updates the coefficients, the fit's transitions are the
    # configured matrix at every time, not its softmax round trip.
    detector.discrete_state_transitions_ = (
        initial.transition_matrix
        if np.array_equal(coefficients, initial.coefficients)
        else _transitions_on_design(design, coefficients)
    )
    return None


def _rebuild_continuous(
    detector: "_DetectorBase", blocks: Mapping[tuple[int, int], np.ndarray]
) -> None:
    configured = detector.continuous_transition_types
    building = [
        [
            _FittedBlock(
                transition.environment_name,
                _declared_destination(transition),
                blocks[(source, target)],
            )
            if (source, target) in blocks
            else transition
            for target, transition in enumerate(row)
        ]
        for source, row in enumerate(configured)
    ]
    representation = detector._transition_representation_
    # Dense EM after an "auto" fit converts the fitted operator (whose
    # environments keep deferred distances) to a dense matrix: rebuild that.
    from_operator = representation == "dense" and any(
        isinstance(env.distance_between_nodes_, LazyGraphDistances)
        for env in detector.environments_
    )
    detector._initialize_continuous_state_transition(
        cast(Any, building),
        transition_representation="auto" if from_operator else representation,
        max_dense_transition_bytes=detector._max_dense_transition_bytes_,
    )
    detector.continuous_transition_types = configured
    if from_operator:
        detector.continuous_state_transitions_ = np.asarray(
            detector.continuous_state_transitions_
        )
        detector.__dict__.pop("_continuous_transition_operator_", None)
        detector._transition_representation_ = "dense"


def detector_from_stored(stored: StoredModel) -> "_DetectorBase":
    """Build the detector a saved model describes.

    Raises
    ------
    ValidationError
        If the file names a type that is not registered.
    """
    manifest = stored.manifest
    spec = spec_for_name(manifest["detector_type"], manifest.get("detector_type_owner"))
    params = Decoder(stored.read_array).decode(stored.config)
    detector = cast("_DetectorBase", spec.cls(**params))
    state = stored.state
    if not state["fitted"]:
        return detector

    decoder = Decoder(stored.read_array)
    environments = tuple(
        _environment_from_state(entry, decoder) for entry in state["environments"]
    )
    detector.environments_ = environments
    decoder.environments = {env.environment_name: env for env in environments}
    detector.encoding_model_ = {
        (
            entry["environment_name"],
            decoder.decode(entry["encoding_group"]),
        ): decoder.decode(entry["model"])
        for entry in state["encoding_models"]
    }
    values = {
        name: decoder.decode(node)
        for name, node in state.items()
        if name not in {"fitted", "environments", "encoding_models"}
    }
    for name, attribute in _STORED:
        if name in values:
            setattr(detector, attribute, values[name])
    detector._initialize_state_index()
    refusal = _rebuild_discrete(detector, values)
    blocks = {
        (entry["from_state"], entry["to_state"]): np.asarray(entry["block"])
        for entry in values["stored_transition_blocks"]
    }
    _rebuild_continuous(detector, blocks)
    detector._rebuild_check_ = _RebuildCheck(
        values["fingerprints"],
        list(blocks),
        manifest.get("package_version", "unknown"),
        refusals=[] if refusal is None else [refusal],
    )
    return detector


def load_detector(path: str | Path) -> "_DetectorBase":
    """Load the detector saved at ``path`` (the saved-model format)."""
    # Built while the file is open, each array read straight from its member;
    # the file is closed on return.
    with open_model(path) as stored:
        return detector_from_stored(stored)
