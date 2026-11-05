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

patsy evaluates a covariate transition's formula as Python, and the file may
be someone else's, so unless ``trust_formulas`` is set the stored formula is
marked as read from a file: a ``LoadedFormula`` within the formula grammar,
checked again whenever a design is built from it, or an inert
``UntrustedFormula`` outside it, which is never evaluated, so the detector and
every copy of it are for inspection only.
"""

import contextlib
import inspect
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from non_local_detector._formula_grammar import (
    TRUST_HINT,
    LoadedFormula,
    UntrustedFormula,
    check_formula,
    refuse_untrusted,
)
from non_local_detector._warnings import user_stacklevel
from non_local_detector.continuous_state_transitions import (
    TransitionContext,
    _declared_destination,
    _no_structured_block,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import (
    DataError,
    NonLocalDetectorError,
    NotFittedError,
    ParameterWarning,
    ValidationError,
)
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.model_format._codec import (
    JSON,
    ArrayWriter,
    Decoder,
    Encoder,
    refuse_object_array,
)
from non_local_detector.model_format._container import (
    FORMAT_NAME,
    FORMAT_VERSION,
    StoredModel,
    array_digest,
    new_model_file,
    open_model,
    write_document,
)
from non_local_detector.model_format._registry import (
    TypeSpec,
    ignore_unread_names,
    ignoring_unread_names,
    spec_for_class,
    spec_for_name,
    spec_for_object,
    unregistered_type_error,
)
from non_local_detector.models._state_space import (
    StateSpace,
    continuous_state_transitions,
    layout_parameters,
    state_layout,
)

if TYPE_CHECKING:
    import pandas as pd

    from non_local_detector.models.base import _DetectorBase
    from non_local_detector.types import TransitionRepresentation

# (file name, detector attribute) of the fitted state stored as is. An
# attribute the detector does not have is left out of the file. The names in
# ``_STATE_SPACE_STORED`` are fields of the detector's ``state_space_``
# instead, which is rebuilt from them on load.
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
    ("transition_representation", "transition_representation"),
    ("max_dense_transition_bytes", "max_dense_transition_bytes"),
    ("time_contract", "time_contract_"),
    ("mark_reference", "mark_reference_"),
    ("converged", "converged_"),
    ("n_iter", "n_iter_"),
    ("em_monotonicity_violations", "em_monotonicity_violations_"),
    ("degenerate_timesteps", "degenerate_timesteps_"),
    ("log_likelihood", "log_likelihood_"),
)
_STATE_SPACE_STORED = frozenset(
    {"transition_representation", "max_dense_transition_bytes"}
)
# Fitted attributes rebuilt on load (or written separately, as
# ``environments_`` and ``encoding_model_``).
_REBUILT = frozenset(
    {
        "environments_",
        "encoding_model_",
        "state_space_",
        "discrete_state_transitions_",
        "discrete_transition_design_matrix_",
        "discrete_transition_smoothness_matrix_",
    }
)
# Not part of a model: the pending rebuild check of a loaded detector.
_NOT_SAVED = frozenset({"_rebuild_check_"})
# Every fitted detector has these.
_REQUIRED = ("environments_", "encoding_model_", "state_space_")

# A rebuilt structure (continuous transition blocks, environment grids and
# their graph distances, a covariate design) matches its fingerprint to this
# relative tolerance, float32-rounding level: rebuilds differ in the last bits
# between platforms, and rounding-level changes to the rebuilding code (about
# 1e-7) are accepted, but a changed model (the 0.02 shift of one bin's random
# walk) is not. The absolute tolerance, per summed point, only covers
# summaries that are zero or nearly so: entries are probabilities or
# coordinates, so a saved summary of order 1 cannot match an all-zero rebuild.
_REBUILD_RTOL = 1e-6
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
    # See _block_summary: spurious BLAS floating-point flags.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.stack([ramp @ values, np.sum(values**2, axis=0)])


def _environment_fingerprint(environment: Environment) -> np.ndarray:
    """Summarize a fitted grid for comparison.

    The summary holds bin centers and edges, interior count and the bin
    graph's edge distances (from which the distances follow).
    """
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
    """Return two fixed weight patterns over ``n`` bins, shape (n, 2).

    They are a ramp from 1 to 2 and the fractional parts of multiples of the
    golden ratio, plus 1.

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
        # Finite probabilities times weights in [1, 2] cannot divide by zero
        # or overflow; some BLAS builds (NumPy 2.2 on Accelerate) still raise
        # spurious floating-point flags from matmul.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
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
    state_space = detector.state_space_
    transitions = state_space.continuous_transitions
    operator = getattr(transitions, "operator", None)
    state_ind = np.asarray(state_space.state_ind)
    n_states = state_space.n_states
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
    ) -> None:
        self.fingerprints = fingerprints
        self.stored_blocks = tuple(stored_blocks)
        self.package_version = package_version
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
            why = _mismatch(
                before, after, max(detector.state_space_.bin_sizes[[source, target]])
            )
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
    """Return the continuous blocks built from fit data, by (from_state, to_state)."""
    state_space = detector.state_space_
    state_ind = np.asarray(state_space.state_ind)
    blocks = {}
    for source, row in enumerate(state_space.continuous_transition_types):
        for target, transition in enumerate(row):
            spec = spec_for_object(transition)
            if spec is None or not spec.block_needs_fit_data:
                continue
            rows = np.flatnonzero(state_ind == source)
            columns = np.flatnonzero(state_ind == target)
            if len(rows) == 1 or len(columns) == 1:
                continue  # built from the one-bin rule, not the transition
            blocks[(source, target)] = np.array(
                state_space.continuous_transitions[np.ix_(rows, columns)]
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
    """Return the detector's attributes that its constructor does not set.

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


class _DigestWriter:
    """Stands in for ``ArrayWriter``: names each array by its ``.npy`` digest."""

    def put(self, array: np.ndarray, where: str) -> str:
        refuse_object_array(array, where)
        return array_digest(np.require(array, requirements="C"))


def _stored_form(value: Any, name: str) -> JSON:
    """Return parameter ``name`` as a saved model stores it, arrays named by content.

    Two values with the same stored form save to the same file, whatever
    their objects' ``==`` says (``Discrete`` compares by identity, and a
    dataclass holding an array cannot reduce ``==`` to one bool).
    """
    return Encoder(cast("ArrayWriter", _DigestWriter())).encode(value, name)


def changed_layout_parameters(detector: "_DetectorBase") -> list[str]:
    """Return the recorded parameters that changed since the detector was fitted.

    A parameter changed when its saved form differs from that of the value
    its fit recorded in ``state_space_`` (see ``layout_parameters``).

    Parameters
    ----------
    detector : _DetectorBase
        Fitted detector.

    Returns
    -------
    changed : list of str
        Names of the changed parameters, in ``layout_parameters`` order.

    Raises
    ------
    ValidationError
        If a parameter holds what a saved model cannot: an object of a type
        not registered for saving, or an object array.
    """
    return [
        name
        for name, value in layout_parameters(detector).items()
        if _stored_form(value, name)
        != _stored_form(getattr(detector.state_space_, name), name)
    ]


def detector_documents(
    detector: "_DetectorBase", writer: ArrayWriter
) -> tuple[dict[str, Any], JSON, dict[str, Any]]:
    """Return the manifest, config and state documents and the arrays of a detector.

    Raises
    ------
    ValidationError
        If the detector or an object it holds is of an unregistered type, it
        holds an attribute the format does not know, it is a loaded
        detector whose rebuilt structures no longer match, or a parameter its
        ``state_space_`` records changed after it was fitted.
    NotFittedError
        If it is partly fitted.
    """
    spec = _detector_spec(detector)
    params = detector.get_params(deep=False)
    config = Encoder(writer).encode(params, "config")
    fitted = _fitted_attributes(detector, params)
    stored_names = {
        attribute for name, attribute in _STORED if name not in _STATE_SPACE_STORED
    }
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
    # The file holds the current parameters, and loading rebuilds the state
    # space from them.
    changed = changed_layout_parameters(detector)
    if changed:
        msg = (
            f"Cannot save the detector: its {changed} changed after it was "
            "fitted, so they no longer describe the fit"
        )
        raise ValidationError(
            msg,
            hint="Set them back with set_params, or refit before save_model.",
        )
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
        if name in _STATE_SPACE_STORED:
            state[name] = encoder.encode(
                getattr(detector.state_space_, attribute), name
            )
        elif attribute in fitted:
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
    """Write ``detector`` to ``path`` in the saved-model format.

    Refused for a transition formula a saved model file was not trusted with
    (an ``UntrustedFormula``): saving it would turn it back into plain text.
    """
    formula = getattr(detector.discrete_transition_type, "formula", None)
    if isinstance(formula, UntrustedFormula):
        refuse_untrusted(formula, "save the model")
    with new_model_file(path) as archive:
        manifest, config, state = detector_documents(detector, ArrayWriter(archive))
        for name, document in (
            ("config.json", config),
            ("state.json", state),
            ("manifest.json", manifest),
        ):
            write_document(archive, name, document)


def _mark_file_formula(
    detector: "_DetectorBase",
    covariate_data: "pd.DataFrame | Mapping[str, Any] | None",
) -> None:
    """Mark the detector's transition formula, read from a file, as such.

    A formula within the formula grammar becomes a ``LoadedFormula``, checked
    again with the covariates whenever a design is built from it; one outside
    it becomes an inert ``UntrustedFormula``, with a ``ParameterWarning``, so
    neither the detector nor any copy of it evaluates it.

    Parameters
    ----------
    detector : _DetectorBase
        The detector being loaded.
    covariate_data : pd.DataFrame or dict, optional
        The stored covariates of a fitted detector, whose keys the formula's
        names must be; None for an unfitted detector.
    """
    # A covariate transition of the package, or a registered one with a
    # formula; one without a formula is left alone.
    transition: Any = detector.discrete_transition_type
    formula = getattr(transition, "formula", None)
    if formula is None:
        return
    try:
        check_formula(
            formula, None if covariate_data is None else list(covariate_data.keys())
        )
    except ValidationError as error:
        # The first line names what is outside the grammar.
        reason = str(error).splitlines()[0]
        warnings.warn(
            f"The saved model's transition formula {formula!r} is outside the "
            "formula grammar for formulas read from a file, so it was not "
            f"evaluated (patsy evaluates formulas as Python): {reason} The "
            "model loaded for inspection only: decoding, fitting and saving it, "
            f"or any copy of it, are refused. {TRUST_HINT}",
            ParameterWarning,
            stacklevel=user_stacklevel(),
        )
        transition.formula = UntrustedFormula(formula)
        return
    transition.formula = LoadedFormula(formula)


def _rebuild_discrete(
    detector: "_DetectorBase", stored: dict[str, Any], *, trust_formulas: bool
) -> None:
    """Rebuild the discrete transition's design, penalties and transitions.

    Parameters
    ----------
    detector : _DetectorBase
        The detector being loaded, with its stored fitted state set.
    stored : dict
        The decoded stored state.
    trust_formulas : bool
        Evaluate the transition formula even if it is outside the formula
        grammar. Otherwise one outside it is not evaluated: the detector
        loads for inspection only, without a design or penalties, and
        without the training-time ``discrete_state_transitions_``.
    """
    from non_local_detector.discrete_state_transitions import (
        _smoothness_matrix,
        _transitions_on_design,
    )

    covariate_data = detector._discrete_transition_covariate_data_
    # Marked whatever the file stores beside it: a formula the file stores
    # without covariates is evaluated by the next fit.
    if not trust_formulas:
        _mark_file_formula(detector, covariate_data)
    if covariate_data is None:
        detector.discrete_transition_design_matrix_ = None
        detector.discrete_transition_smoothness_matrix_ = None
        detector.discrete_state_transitions_ = stored["discrete_state_transitions"]
        return
    if isinstance(
        getattr(detector.discrete_transition_type, "formula", None), UntrustedFormula
    ):
        detector.discrete_transition_design_matrix_ = None
        detector.discrete_transition_smoothness_matrix_ = None
        # The training-time transitions follow from the design: left unset.
        return
    initial = detector.discrete_transition_type.make_state_transition(covariate_data)
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


def _rebuild_state_space(
    detector: "_DetectorBase",
    blocks: Mapping[tuple[int, int], np.ndarray],
    layout: tuple[np.ndarray, np.ndarray, np.ndarray],
    stored: Mapping[str, Any],
) -> None:
    """Rebuild the detector's ``state_space_`` from its stored inputs.

    Parameters
    ----------
    detector : _DetectorBase
        The detector being loaded, with its environments set.
    blocks : mapping
        The stored continuous transition blocks, by (from_state, to_state).
    layout : tuple of np.ndarray
        ``state_layout`` of the detector's parameters and environments.
    stored : mapping
        The decoded stored state, holding the ``_STATE_SPACE_STORED`` names.
    """
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
        for source, row in enumerate(detector.continuous_transition_types)
    ]
    representation = stored["transition_representation"]
    max_dense_transition_bytes = stored["max_dense_transition_bytes"]
    # Dense EM after an "auto" fit converts the fitted operator (whose
    # environments keep deferred distances) to a dense matrix: rebuild that.
    from_operator = representation == "dense" and any(
        isinstance(env.distance_between_nodes_, LazyGraphDistances)
        for env in detector.environments_
    )
    built_as = cast(
        "TransitionRepresentation", "auto" if from_operator else representation
    )
    state_ind, bin_sizes, is_track_interior = layout
    transitions, operator = continuous_state_transitions(
        cast(Any, building),
        detector.environments_,
        detector.observation_models,
        detector.local_position_std,
        state_ind,
        bin_sizes,
        context=TransitionContext(),
        transition_representation=built_as,
        max_dense_transition_bytes=max_dense_transition_bytes,
    )
    state_space = StateSpace(
        **layout_parameters(detector),
        state_ind=state_ind,
        bin_sizes=bin_sizes,
        is_track_interior=is_track_interior,
        continuous_transitions=transitions,
        transition_operator=operator,
        transition_representation=built_as,
        max_dense_transition_bytes=int(max_dense_transition_bytes),
    )
    detector.state_space_ = state_space.densified() if from_operator else state_space


# The schema of the documents ``detector_documents`` writes. A reader ignores,
# with a warning, a name it does not know, which a newer version wrote.
_MANIFEST_NAMES = frozenset(
    {
        "format",
        "format_version",
        "package_version",
        "detector_type",
        "detector_type_owner",
        "arrays",
    }
)
# state.json of an unfitted detector holds only "fitted". A fitted one holds
# these, every ``_STORED`` name but ``_OPTIONAL_STORED``,
# "discrete_state_transitions" if it stores no covariates (then the
# transitions are stored, not rebuilt) and "mark_reference" if it is
# clusterless.
_FITTED_STATE = (
    "fitted",
    "environments",
    "encoding_models",
    "stored_transition_blocks",
    "fingerprints",
)
# Stored names a fitted detector may lack: EM's records, which only
# ``estimate_parameters`` sets, and a clusterless detector's mark reference.
_OPTIONAL_STORED = frozenset(
    {
        "discrete_initial_conditions",
        "mark_reference",
        "converged",
        "n_iter",
        "em_monotonicity_violations",
        "degenerate_timesteps",
        "log_likelihood",
    }
)
# The fields of each entry of these lists of state.json; an environment holds
# "place_bin_edges" with a track graph and the distance options without.
_ENTRY_FIELDS = {
    "environments": frozenset(
        {
            "config",
            "place_bin_centers",
            "is_track_interior",
            "edges",
            "place_bin_edges",
            "infer_track_interior",
            "all_pairs_distances",
            "max_dense_distance_bytes",
        }
    ),
    "encoding_models": frozenset({"environment_name", "encoding_group", "model"}),
    "stored_transition_blocks": frozenset({"from_state", "to_state", "block"}),
}
_FINGERPRINT_FIELDS = frozenset(
    {"environments", "continuous_transitions", "discrete_transition_design"}
)
_CORRUPT_HINT = "The file is corrupt or was edited."


def check_documents(stored: StoredModel) -> None:
    """Raise ``DataError`` unless a saved model's documents hold what it needs.

    state.json must hold every name a model of its kind always has, and each
    encoding model every key of its algorithm's type. A name this version
    does not read (outside ``_MANIFEST_NAMES``, ``_FITTED_STATE``,
    ``_STORED``, ``_ENTRY_FIELDS``, ``_FINGERPRINT_FIELDS`` or the encoding
    model's type), which a newer version wrote, is ignored with a warning
    (``ignore_unread_names``). The values are checked as they are decoded.

    Parameters
    ----------
    stored : StoredModel
        The open saved model, migrated to ``FORMAT_VERSION``.

    Raises
    ------
    DataError
        If a document lacks a name it needs.
    """
    manifest, config = stored.manifest, stored.config
    state: Any = stored.state  # parsed JSON: a file may hold any value
    ignore_unread_names("manifest.json", set(manifest) - _MANIFEST_NAMES)
    if "detector_type" not in manifest:
        msg = "The saved model's manifest has no 'detector_type'"
        raise DataError(msg, hint=_CORRUPT_HINT)
    if not isinstance(state, dict) or "fitted" not in state:
        msg = "The saved model's state has no 'fitted' entry"
        raise DataError(msg, hint=_CORRUPT_HINT)
    if not state["fitted"]:
        required = allowed = {"fitted"}
    else:
        required = {*_FITTED_STATE} | {
            name for name, _ in _STORED if name not in _OPTIONAL_STORED
        }
        if state.get("discrete_transition_covariate_data") is None:
            required.add("discrete_state_transitions")
        if isinstance(config, dict) and "clusterless_algorithm" in config:
            required.add("mark_reference")
        allowed = required | _OPTIONAL_STORED | {"discrete_state_transitions"}
    ignore_unread_names("state.json", set(state) - allowed)
    missing = sorted(required - set(state))
    if missing:
        msg = f"The saved model is marked fitted but its state has no {missing}"
        raise DataError(msg, hint=_CORRUPT_HINT)
    for name, fields in _ENTRY_FIELDS.items():
        entries = [entry for entry in state.get(name, []) if isinstance(entry, dict)]
        ignore_unread_names(
            f"state.json {name} entries", set().union(*entries) - fields
        )
    fingerprints = state.get("fingerprints")
    if isinstance(fingerprints, dict):
        ignore_unread_names(
            "state.json fingerprints", set(fingerprints) - _FINGERPRINT_FIELDS
        )
    for entry in state.get("encoding_models", []):
        model = entry["model"]
        keys = encoding_model_keys(model)
        ignore_unread_names(f"{model['algorithm']} encoding model", set(model) - keys)


def encoding_model_keys(model: Any) -> frozenset[str]:
    """Return the keys of a stored encoding model's type.

    Those of the ``encoding_model_type`` of the registered algorithm the
    model names; a reader ignores any other key.

    Parameters
    ----------
    model : dict
        The stored encoding model (its JSON node, or decoded).

    Returns
    -------
    keys : frozenset of str

    Raises
    ------
    DataError
        If the model names no registered algorithm, or lacks a required key.
    """
    from non_local_detector.likelihoods import (
        _CLUSTERLESS_ALGORITHMS,
        _SORTED_SPIKES_ALGORITHMS,
        LikelihoodAlgorithm,
    )

    registry: dict[str, LikelihoodAlgorithm[Any, Any]] = {
        **_SORTED_SPIKES_ALGORITHMS,
        **_CLUSTERLESS_ALGORITHMS,
    }
    algorithm = model.get("algorithm") if isinstance(model, dict) else None
    if algorithm not in registry:
        msg = f"The saved model has an encoding model of algorithm {algorithm!r}, which is not registered"
        raise DataError(
            msg,
            expected=f"one of {sorted(registry)}",
            hint="The file was written by a newer or different version of "
            "non_local_detector, or was edited.",
        )
    # A TypedDict, which mypy types as a plain type.
    model_type: Any = registry[algorithm].encoding_model_type
    missing = sorted(model_type.__required_keys__ - set(model))
    if missing:
        msg = f"The saved {algorithm} encoding model has no {missing}"
        raise DataError(msg, hint=_CORRUPT_HINT)
    return frozenset(model_type.__required_keys__ | model_type.__optional_keys__)


@contextlib.contextmanager
def data_errors(action: str) -> Iterator[None]:
    """Raise the built-in errors of reading a saved model as ``DataError``.

    ``KeyError``, ``IndexError``, ``TypeError`` and ``ValueError`` from a
    missing or malformed entry, and ``RecursionError`` from one nested too
    deeply to decode, become ``DataError``, chained to the original; package
    errors pass through unchanged.

    Parameters
    ----------
    action : str
        What failed, for the message: "rebuild" or "read".
    """
    try:
        yield
    except NonLocalDetectorError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, RecursionError) as error:
        msg = f"Cannot {action} the saved model ({type(error).__name__}: {error})"
        raise DataError(msg, hint=_CORRUPT_HINT) from error


def detector_from_stored(
    stored: StoredModel, *, trust_formulas: bool = False
) -> "_DetectorBase":
    """Build the detector a saved model describes.

    Parameters
    ----------
    stored : StoredModel
        The open saved model.
    trust_formulas : bool, optional
        Evaluate a covariate transition formula outside the formula grammar
        (see ``load_model``).

    Raises
    ------
    ValidationError
        If the file names a type that is not registered.
    DataError
        If the file is corrupt or was edited: missing fitted state
        (``check_documents``) or a malformed entry.
        Built-in errors from reading the file become this (``data_errors``),
        chained to the original; package errors pass through unchanged.
    """
    with (
        ignoring_unread_names(stored.manifest.get("package_version")),
        data_errors("rebuild"),
    ):
        check_documents(stored)
        return _build_detector(stored, trust_formulas=trust_formulas)


def _check_fingerprints(fingerprints: Any, n_environments: int) -> None:
    """Raise ``DataError`` unless ``fingerprints`` has the saved structure."""
    ok = (
        isinstance(fingerprints, dict)
        and isinstance(fingerprints.get("environments"), list)
        and len(fingerprints["environments"]) == n_environments
        and isinstance(fingerprints.get("continuous_transitions"), list)
    )
    if not ok:
        msg = "The saved model's fingerprints are malformed"
        raise DataError(
            msg,
            expected=f"environments (one per environment, {n_environments}) "
            "and continuous_transitions lists",
            hint=_CORRUPT_HINT,
        )


def _build_detector(stored: StoredModel, *, trust_formulas: bool) -> "_DetectorBase":
    manifest = stored.manifest
    spec = spec_for_name(manifest["detector_type"], manifest.get("detector_type_owner"))
    params = Decoder(stored.read_array).decode(stored.config)
    accepted = inspect.signature(spec.cls).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
        ignore_unread_names(
            f"config.json ({spec.cls.__name__} parameters)",
            set(params) - set(accepted),
        )
        params = {name: value for name, value in params.items() if name in accepted}
    detector = cast("_DetectorBase", spec.cls(**params))
    state = stored.state
    if not state["fitted"]:
        if not trust_formulas:
            _mark_file_formula(detector, None)
        return detector

    decoder = Decoder(stored.read_array)
    environments = tuple(
        _environment_from_state(entry, decoder) for entry in state["environments"]
    )
    detector.environments_ = environments
    decoder.environments = {env.environment_name: env for env in environments}
    detector.encoding_model_ = {}
    for entry in state["encoding_models"]:
        keys = encoding_model_keys(entry["model"])
        detector.encoding_model_[
            (entry["environment_name"], decoder.decode(entry["encoding_group"]))
        ] = {
            key: decoder.decode(node)
            for key, node in entry["model"].items()
            if key in keys
        }
    # Files saved before the shared keys' values were NumPy arrays.
    from non_local_detector.likelihoods import _unify_stored_encoding_values

    for model in detector.encoding_model_.values():
        _unify_stored_encoding_values(model)
    values = {
        name: decoder.decode(state[name])
        for name in (
            *(name for name, _ in _STORED),
            "discrete_state_transitions",
            "stored_transition_blocks",
            "fingerprints",
        )
        if name in state
    }
    for name, attribute in _STORED:
        if name in values and name not in _STATE_SPACE_STORED:
            setattr(detector, attribute, values[name])
    _check_fingerprints(values["fingerprints"], len(environments))
    layout = state_layout(
        detector.observation_models, environments, detector.local_position_std
    )
    _rebuild_discrete(detector, values, trust_formulas=trust_formulas)
    blocks = {
        (entry["from_state"], entry["to_state"]): np.asarray(entry["block"])
        for entry in values["stored_transition_blocks"]
    }
    _rebuild_state_space(detector, blocks, layout, values)
    detector._rebuild_check_ = _RebuildCheck(
        values["fingerprints"],
        list(blocks),
        manifest.get("package_version", "unknown"),
    )
    return detector


def load_detector(path: str | Path, *, trust_formulas: bool = False) -> "_DetectorBase":
    """Load the detector saved at ``path`` (the saved-model format).

    ``trust_formulas`` is as for ``load_model``.
    """
    # Built while the file is open, each array read straight from its member;
    # the file is closed on return.
    with open_model(path) as stored:
        return detector_from_stored(stored, trust_formulas=trust_formulas)
