"""Exact conditional state-block transitions without a combined dense matrix.

Forward products are ``p @ T``; backward products are ``T @ v``. Discrete
state weights remain an S-by-S array and never expand to N-by-N. Structured
Gaussian factors retain every grid-to-grid term, including tails and holes.
The host builder is outside JAX tracing; runtime products are pure JAX.
"""

import itertools
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field, fields, replace
from typing import Any, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal, norm

from non_local_detector.continuous_state_transitions import (
    ContinuousTransition,
    RandomWalk,
    TransitionContext,
    _check_continuous_transition,
    _make_transition_block,
    _multi_bin_local_transition,
    _scalar_to_spatial_destination,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import NonLocalDetectorError
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.observation_models import ObservationModel

DEFAULT_MAX_DENSE_BYTES = 256 * 1024**2

_T = TypeVar("_T")

# Host or device array.
_Array: TypeAlias = np.ndarray | jax.Array

# Any one conditional spatial block of a ``BlockTransitionOperator``.
Block: TypeAlias = (
    "UniformBlock | IdentityBlock | GaussianGridBlock | DenseBlock | RestrictedBlock"
)


def _register_operator(cls: type[_T]) -> type[_T]:
    """Register explicit fields on JAX versions without inferred dataclasses."""
    dataclass_cls = cast(Any, cls)
    data = tuple(
        item.name
        for item in fields(dataclass_cls)
        if not item.metadata.get("static", False)
    )
    static = tuple(
        item.name
        for item in fields(dataclass_cls)
        if item.metadata.get("static", False)
    )

    def flatten(value: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return (
            tuple(getattr(value, name) for name in data),
            tuple(getattr(value, name) for name in static),
        )

    def unflatten(metadata: tuple[Any, ...], children: Sequence[Any]) -> _T:
        return cls(
            **dict(zip(data, children, strict=True)),
            **dict(zip(static, metadata, strict=True)),
        )

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


# Appended to messages that refuse a structured block.
_DENSE_BLOCK_HINT = (
    'use transition_representation="auto" (dense block within '
    'max_dense_transition_bytes) or "dense"'
)


class UnsupportedTransitionError(NonLocalDetectorError, ValueError):
    """A configuration has no proven exact structured implementation."""


class DenseTransitionBudgetError(NonLocalDetectorError, MemoryError):
    """A requested dense allocation exceeds its explicit byte budget."""


def _budget(nbytes: int, maximum: Any, description: str) -> None:
    if (
        not isinstance(maximum, (int, np.integer))
        or isinstance(maximum, bool)
        or maximum < 0
    ):
        msg = "Dense transition byte budget must be a nonnegative integer"
        raise ValueError(msg)
    if nbytes > maximum:
        msg = (
            f"{description} requires {nbytes:,} bytes, exceeding the "
            f"{maximum:,}-byte dense transition budget. Raise "
            "max_dense_transition_bytes, use transition_representation="
            '"dense", or request a smaller dense selection. EM, Viterbi and '
            "plotting may need the dense transition."
        )
        raise DenseTransitionBudgetError(msg)


def _nan_result(result: jax.Array, source: jax.Array) -> jax.Array:
    # Dense products propagate any NaN through even zero matrix entries.
    return jnp.where(jnp.any(jnp.isnan(source)), jnp.full_like(result, jnp.nan), result)


def _axes_product(
    values: Any,
    factors: Sequence[_Array],
    *,
    backward: bool = False,
    numpy: bool = False,
) -> Any:
    xp = np if numpy else jnp
    for axis, factor in enumerate(factors):
        factor = xp.asarray(factor, dtype=values.dtype)
        if backward:
            factor = factor.T
        options = {} if numpy else {"precision": jax.lax.Precision.HIGHEST}
        values = xp.moveaxis(
            xp.tensordot(values, factor, axes=((axis,), (0,)), **options), -1, axis
        )
    return values


def _ratio(numerator: np.ndarray, denominator: ArrayLike) -> np.ndarray:
    denominator = np.asarray(denominator)
    return np.where(
        denominator > 0, numerator / np.where(denominator > 0, denominator, 1), 0
    )


@_register_operator
@dataclass(frozen=True)
class UniformBlock:
    """A masked rank-one block, including legacy scalar/spatial overrides.

    Attributes
    ----------
    source_mask : np.ndarray or jax.Array, shape (n_source_bins,)
        Weight of each source bin (1 where the bin can be left, else 0).
    destination_mask : np.ndarray or jax.Array, shape (n_destination_bins,)
        Weight of each destination bin (1 where the bin can be entered, else 0).
    normalization : np.ndarray or jax.Array
        Scalar multiplying ``source_mask[i] * destination_mask[j]``; typically
        the reciprocal of the number of enabled destination bins, so each
        enabled row sums to one.
    """

    source_mask: _Array
    destination_mask: _Array
    normalization: _Array

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(n_source_bins, n_destination_bins)`` of the block."""
        return (len(self.source_mask), len(self.destination_mask))

    def forward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the left, ``values @ T``.

        Parameters
        ----------
        values : jax.Array, shape (n_source_bins,)
            Probabilities (or other weights) over the source bins.

        Returns
        -------
        result : jax.Array, shape (n_destination_bins,)
            ``values @ T``, over the destination bins.
        """
        source = jnp.asarray(self.source_mask, dtype=values.dtype)
        destination = jnp.asarray(self.destination_mask, dtype=values.dtype)
        divisor = jnp.asarray(self.normalization, dtype=values.dtype)
        result = jnp.sum(values * source * divisor) * destination
        return _nan_result(result, values)

    def backward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the right, ``T @ values``.

        Parameters
        ----------
        values : jax.Array, shape (n_destination_bins,)
            Values over the destination bins.

        Returns
        -------
        result : jax.Array, shape (n_source_bins,)
            ``T @ values``, over the source bins.
        """
        source = jnp.asarray(self.source_mask, dtype=values.dtype)
        destination = jnp.asarray(self.destination_mask, dtype=values.dtype)
        divisor = jnp.asarray(self.normalization, dtype=values.dtype)
        return _nan_result(source * jnp.sum(values * destination * divisor), values)

    def entries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return selected entries of the block.

        Parameters
        ----------
        rows : np.ndarray of int
            Source-bin indices, broadcast against ``columns``.
        columns : np.ndarray of int
            Destination-bin indices, broadcast against ``rows``.

        Returns
        -------
        entries : np.ndarray, shape of the broadcast ``rows`` and ``columns``
            Transition probabilities ``T[rows, columns]``.
        """
        return cast(
            np.ndarray,
            (
                np.asarray(self.source_mask)[rows]
                * np.asarray(self.destination_mask)[columns]
            )
            * np.asarray(self.normalization),
        )


@_register_operator
@dataclass(frozen=True)
class IdentityBlock:
    """A masked identity block: probability stays in the same bin.

    Attributes
    ----------
    mask : np.ndarray or jax.Array, shape (n_bins,)
        Weight of each bin (1 where the bin is on the track, else 0), applied on
        the diagonal.
    """

    mask: _Array

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(n_bins, n_bins)`` of the block."""
        return (len(self.mask), len(self.mask))

    def forward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the left, ``values @ T``.

        Parameters
        ----------
        values : jax.Array, shape (n_bins,)
            Probabilities (or other weights) over the source bins.

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            ``values @ T``, over the destination bins.
        """
        return _nan_result(values * jnp.asarray(self.mask, dtype=values.dtype), values)

    def backward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the right, ``T @ values``.

        Parameters
        ----------
        values : jax.Array, shape (n_bins,)
            Values over the destination bins.

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            ``T @ values``, over the source bins.
        """
        return self.forward(values)

    def entries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return selected entries of the block.

        Parameters
        ----------
        rows : np.ndarray of int
            Source-bin indices, broadcast against ``columns``.
        columns : np.ndarray of int
            Destination-bin indices, broadcast against ``rows``.

        Returns
        -------
        entries : np.ndarray, shape of the broadcast ``rows`` and ``columns``
            Transition probabilities ``T[rows, columns]``.
        """
        return cast(np.ndarray, np.asarray(self.mask)[rows] * (rows == columns))


@_register_operator
@dataclass(frozen=True)
class GaussianGridBlock:
    """Masked, row-normalized tensor product of untruncated Gaussian axes.

    Attributes
    ----------
    factors : tuple of np.ndarray or jax.Array
        One square matrix per grid axis, shape ``(n_axis_bins, n_axis_bins)``,
        holding the Gaussian transition density between that axis's bins.
    mask : np.ndarray or jax.Array, shape (n_bins,)
        Weight of each flattened grid bin (1 where on the track, else 0).
    row_normalization : np.ndarray or jax.Array, shape (n_bins,)
        Sum over destination bins of each masked row of the unnormalized
        product, used to normalize the rows.
    """

    factors: tuple[_Array, ...]
    mask: _Array
    row_normalization: _Array

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Number of bins along each grid axis."""
        return tuple(factor.shape[0] for factor in self.factors)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape ``(n_bins, n_bins)`` of the block, with ``n_bins`` the grid size."""
        return (int(np.prod(self.grid_shape)),) * 2

    def forward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the left, ``values @ T``.

        Parameters
        ----------
        values : jax.Array, shape (n_bins,)
            Probabilities (or other weights) over the source bins.

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            ``values @ T``, over the destination bins.
        """
        mask = jnp.asarray(self.mask, dtype=values.dtype)
        normalizer = jnp.asarray(self.row_normalization, dtype=values.dtype)
        safe = jnp.where(normalizer > 0, normalizer, 1)
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1)
        # Keep the original arithmetic for ordinary values. Cap finite scales
        # so a compiler-hoisted reciprocal remains normal instead of flushing
        # to zero; row-normalized products still undo exactly this scale.
        limit = jnp.asarray(
            1 / jnp.finfo(values.dtype).tiny,  # type: ignore[no-untyped-call]  # jax.numpy.finfo is untyped
            dtype=values.dtype,
        )
        scale = jnp.where(jnp.isfinite(scale), jnp.minimum(scale, limit), scale)
        weighted = jnp.where(normalizer > 0, (values / scale) * mask / safe, 0)
        result = (
            _axes_product(weighted.reshape(self.grid_shape), self.factors).ravel()
            * mask
        )
        return _nan_result(result * scale, values)

    def backward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the right, ``T @ values``.

        Parameters
        ----------
        values : jax.Array, shape (n_bins,)
            Values over the destination bins.

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            ``T @ values``, over the source bins.
        """
        mask = jnp.asarray(self.mask, dtype=values.dtype)
        normalizer = jnp.asarray(self.row_normalization, dtype=values.dtype)
        safe = jnp.where(normalizer > 0, normalizer, 1)
        maximum = jnp.max(jnp.abs(values))
        scale = jnp.where(maximum > 0, maximum, 1)
        limits = jnp.finfo(values.dtype)  # type: ignore[no-untyped-call]  # jax.numpy.finfo is untyped
        scale = jnp.where(
            jnp.isfinite(scale),
            jnp.clip(
                scale,
                jnp.asarray(limits.tiny, dtype=values.dtype),
                jnp.asarray(1 / limits.tiny, dtype=values.dtype),
            ),
            scale,
        )
        result = _axes_product(
            (values * mask / scale).reshape(self.grid_shape),
            self.factors,
            backward=True,
        ).ravel()
        result = jnp.where(normalizer > 0, result * mask / safe, 0)
        return _nan_result(result * scale, values)

    def entries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return selected entries of the block.

        Parameters
        ----------
        rows : np.ndarray of int
            Source-bin indices, broadcast against ``columns``.
        columns : np.ndarray of int
            Destination-bin indices, broadcast against ``rows``.

        Returns
        -------
        entries : np.ndarray, shape of the broadcast ``rows`` and ``columns``
            Transition probabilities ``T[rows, columns]``.
        """
        rows, columns = np.broadcast_arrays(rows, columns)
        coordinates_from = np.unravel_index(rows, self.grid_shape)
        coordinates_to = np.unravel_index(columns, self.grid_shape)
        result = np.ones(rows.shape, dtype=float)
        for factor, source, destination in zip(
            self.factors, coordinates_from, coordinates_to, strict=True
        ):
            result *= np.asarray(factor)[source, destination]
        result *= np.asarray(self.mask)[rows] * np.asarray(self.mask)[columns]
        return _ratio(result, np.asarray(self.row_normalization)[rows])


@_register_operator
@dataclass(frozen=True)
class DenseBlock:
    """Explicit fallback; its retained matrix counts against the dense budget.

    Attributes
    ----------
    matrix : np.ndarray or jax.Array, shape (n_source_bins, n_destination_bins)
        Dense transition probabilities.
    """

    matrix: _Array

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape ``(n_source_bins, n_destination_bins)`` of the block."""
        return self.matrix.shape

    def forward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the left, ``values @ T``.

        Parameters
        ----------
        values : jax.Array, shape (n_source_bins,)
            Probabilities (or other weights) over the source bins.

        Returns
        -------
        result : jax.Array, shape (n_destination_bins,)
            ``values @ T``, over the destination bins.
        """
        return jnp.matmul(
            values,
            jnp.asarray(self.matrix, dtype=values.dtype),
            precision=jax.lax.Precision.HIGHEST,
        )

    def backward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the right, ``T @ values``.

        Parameters
        ----------
        values : jax.Array, shape (n_destination_bins,)
            Values over the destination bins.

        Returns
        -------
        result : jax.Array, shape (n_source_bins,)
            ``T @ values``, over the source bins.
        """
        return jnp.matmul(
            jnp.asarray(self.matrix, dtype=values.dtype),
            values,
            precision=jax.lax.Precision.HIGHEST,
        )

    def entries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return selected entries of the block.

        Parameters
        ----------
        rows : np.ndarray of int
            Source-bin indices, broadcast against ``columns``.
        columns : np.ndarray of int
            Destination-bin indices, broadcast against ``rows``.

        Returns
        -------
        entries : np.ndarray, shape of the broadcast ``rows`` and ``columns``
            Transition probabilities ``T[rows, columns]``.
        """
        return cast(np.ndarray, np.asarray(self.matrix)[rows, columns])


@_register_operator
@dataclass(frozen=True)
class RestrictedBlock:
    """An exact indexed submatrix; selection never renormalizes its rows.

    Attributes
    ----------
    original : Block
        The block being restricted.
    rows : np.ndarray or jax.Array, shape (n_rows,)
        Indices of the original block's source bins that are kept.
    columns : np.ndarray or jax.Array, shape (n_columns,)
        Indices of the original block's destination bins that are kept.
    """

    original: "Block"
    rows: _Array
    columns: _Array

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(n_rows, n_columns)`` of the restricted block."""
        return (len(self.rows), len(self.columns))

    def forward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the left, ``values @ T``.

        Parameters
        ----------
        values : jax.Array, shape (n_rows,)
            Probabilities (or other weights) over the source bins.

        Returns
        -------
        result : jax.Array, shape (n_columns,)
            ``values @ T``, over the destination bins.
        """
        if not len(self.rows) or not len(self.columns):
            return jnp.zeros(len(self.columns), dtype=values.dtype)
        full = (
            jnp.zeros(self.original.shape[0], dtype=values.dtype)
            .at[jnp.asarray(self.rows)]
            .set(values)
        )
        return self.original.forward(full)[jnp.asarray(self.columns)]

    def backward(self, values: jax.Array) -> jax.Array:
        """Apply the block to a vector from the right, ``T @ values``.

        Parameters
        ----------
        values : jax.Array, shape (n_columns,)
            Values over the destination bins.

        Returns
        -------
        result : jax.Array, shape (n_rows,)
            ``T @ values``, over the source bins.
        """
        if not len(self.rows) or not len(self.columns):
            return jnp.zeros(len(self.rows), dtype=values.dtype)
        full = (
            jnp.zeros(self.original.shape[1], dtype=values.dtype)
            .at[jnp.asarray(self.columns)]
            .set(values)
        )
        return self.original.backward(full)[jnp.asarray(self.rows)]

    def entries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return selected entries of the block.

        Parameters
        ----------
        rows : np.ndarray of int
            Indices into the kept source bins, broadcast against ``columns``.
        columns : np.ndarray of int
            Indices into the kept destination bins, broadcast against ``rows``.

        Returns
        -------
        entries : np.ndarray, shape of the broadcast ``rows`` and ``columns``
            Transition probabilities ``T[rows, columns]``.
        """
        return self.original.entries(
            np.asarray(self.rows)[rows], np.asarray(self.columns)[columns]
        )


@_register_operator
@dataclass(frozen=True)
class BlockTransitionOperator:
    """Conditional spatial blocks with separate stationary or per-step weights.

    ``bind_discrete`` sets stationary defaults. A driver supplies a single S-by-S
    matrix to each forward/backward call for covariate transitions. All floating
    runtime factors cast to the input vector dtype; host dense views remain float64.

    Attributes
    ----------
    blocks : tuple of tuple of Block
        ``blocks[i][j]`` is the conditional spatial transition from the bins of
        discrete state ``i`` to the bins of discrete state ``j``.
    state_sizes : tuple of int
        Number of spatial bins of each discrete state.
    discrete_weights : np.ndarray or jax.Array, shape (n_states, n_states), optional
        Stationary discrete transition probabilities multiplying each block, by
        default None (all ones).
    """

    blocks: tuple[tuple[Block, ...], ...]
    state_sizes: tuple[int, ...] = field(metadata={"static": True})
    discrete_weights: _Array | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape ``(n_bins, n_bins)`` of the combined transition matrix."""
        return (sum(self.state_sizes),) * 2

    @property
    def n_bins(self) -> int:
        """Total number of bins over all discrete states."""
        return self.shape[0]

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Data type of dense views of the operator, always float64."""
        return np.dtype(np.float64)

    @property
    def storage_nbytes(self) -> int:
        """Memory held by the operator's array leaves, in bytes."""
        return sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(self))

    @property
    def dense_fallback_nbytes(self) -> int:
        """Memory held by the dense matrices of its ``DenseBlock`` blocks, in bytes."""

        def size(block: Block) -> int:
            if isinstance(block, RestrictedBlock):
                return size(block.original)
            return (
                np.asarray(block.matrix).nbytes if isinstance(block, DenseBlock) else 0
            )

        return sum(size(block) for row in self.blocks for block in row)

    def bind_discrete(self, weights: _Array) -> "BlockTransitionOperator":
        """Return a copy with stationary discrete transition weights.

        Parameters
        ----------
        weights : np.ndarray or jax.Array, shape (n_states, n_states)
            Discrete state transition probabilities multiplying each block.

        Returns
        -------
        operator : BlockTransitionOperator
            Copy of this operator with ``discrete_weights`` set.

        Raises
        ------
        ValueError
            If ``weights`` is not ``(n_states, n_states)``.
        """
        if np.shape(weights) != (len(self.state_sizes),) * 2:
            msg = "discrete_weights must have shape (n_states, n_states)"
            raise ValueError(msg)
        return replace(self, discrete_weights=weights)

    def restricted(self, mask: ArrayLike) -> "BlockTransitionOperator":
        """Return the operator restricted to a subset of the combined bins.

        Rows are not renormalized after the restriction.

        Parameters
        ----------
        mask : array_like of bool, shape (n_bins,)
            True for the bins to keep, over the combined bins of all states.

        Returns
        -------
        operator : BlockTransitionOperator
            Operator over the kept bins, with ``state_sizes`` updated.

        Raises
        ------
        ValueError
            If ``mask`` does not have one entry per combined bin.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_bins,):
            msg = "Restriction mask must have one entry per combined state bin"
            raise ValueError(msg)
        boundaries = np.cumsum((0, *self.state_sizes))
        indices = tuple(
            np.flatnonzero(mask[start:stop])
            for start, stop in itertools.pairwise(boundaries)
        )
        blocks = tuple(
            tuple(
                RestrictedBlock(block, indices[i], indices[j])
                for j, block in enumerate(row)
            )
            for i, row in enumerate(self.blocks)
        )
        return BlockTransitionOperator(
            blocks, tuple(len(index) for index in indices), self.discrete_weights
        )

    def _weights(self, values: jax.Array, override: _Array | None) -> jax.Array:
        weights = self.discrete_weights if override is None else override
        if weights is None:
            return jnp.ones((len(self.state_sizes),) * 2, dtype=values.dtype)
        if weights.shape != (len(self.state_sizes),) * 2:
            msg = "Pass one S-by-S discrete transition matrix per step"
            raise ValueError(msg)
        return jnp.asarray(weights, dtype=values.dtype)

    def forward(
        self, probabilities: ArrayLike, discrete_weights: _Array | None = None
    ) -> jax.Array:
        """Apply the transition from the left, ``probabilities @ T``.

        Parameters
        ----------
        probabilities : array_like, shape (n_bins,)
            Probability over the combined bins of all states.
        discrete_weights : np.ndarray or jax.Array, shape (n_states, n_states), optional
            Discrete state transition matrix for this step, overriding the bound
            stationary weights. By default None, which uses the bound weights (or
            ones if none were bound).

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            The one-step-ahead prediction ``probabilities @ T``.
        """
        probabilities = jnp.asarray(probabilities)
        weights = self._weights(probabilities, discrete_weights)
        boundaries = tuple(np.cumsum((0, *self.state_sizes)))
        output = []
        for target, size in enumerate(self.state_sizes):
            result = jnp.zeros(size, dtype=probabilities.dtype)
            for source, row in enumerate(self.blocks):
                values = probabilities[boundaries[source] : boundaries[source + 1]]
                result = result + weights[source, target] * row[target].forward(values)
            output.append(result)
        return jnp.concatenate(output)

    def backward(
        self, values: ArrayLike, discrete_weights: _Array | None = None
    ) -> jax.Array:
        """Apply the transition from the right, ``T @ values``.

        Parameters
        ----------
        values : array_like, shape (n_bins,)
            Values over the combined bins of all states.
        discrete_weights : np.ndarray or jax.Array, shape (n_states, n_states), optional
            Discrete state transition matrix for this step, overriding the bound
            stationary weights. By default None, which uses the bound weights (or
            ones if none were bound).

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            The product ``T @ values``.
        """
        values = jnp.asarray(values)
        weights = self._weights(values, discrete_weights)
        boundaries = tuple(np.cumsum((0, *self.state_sizes)))
        output = []
        for source, (row, size) in enumerate(
            zip(self.blocks, self.state_sizes, strict=True)
        ):
            result = jnp.zeros(size, dtype=values.dtype)
            for target, block in enumerate(row):
                result = result + weights[source, target] * block.backward(
                    values[boundaries[target] : boundaries[target + 1]]
                )
            output.append(result)
        return jnp.concatenate(output)

    def pair_products(
        self,
        filtered: ArrayLike,
        ratio: ArrayLike,
        discrete_weights: _Array | None = None,
    ) -> jax.Array:
        """Per-state-pair terms of ``filtered @ T * ratio``, before summing pairs.

        ``result[i, j] = w[i, j] * filtered[bins of i] . (C_ij @ ratio[bins of
        j])``. With a filtered row and the smoother ratio
        ``smoothed_{t+1} / predicted_{t+1}`` these are one step's expected
        discrete transition counts.

        Parameters
        ----------
        filtered : jax.Array, shape (n_bins,)
            Filtered state probabilities at time ``t``.
        ratio : jax.Array, shape (n_bins,)
            Elementwise ratio of smoothed to predicted probabilities at time
            ``t + 1``.
        discrete_weights : jax.Array, shape (n_states, n_states), optional
            Overrides the bound stationary weights.

        Returns
        -------
        pairs : jax.Array, shape (n_states, n_states)
            Expected transition mass between each pair of discrete states.
        """
        filtered, ratio = jnp.asarray(filtered), jnp.asarray(ratio)
        weights = self._weights(filtered, discrete_weights)
        boundaries = tuple(np.cumsum((0, *self.state_sizes)))
        pairs = jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.dot(
                            filtered[boundaries[source] : boundaries[source + 1]],
                            block.backward(
                                ratio[boundaries[target] : boundaries[target + 1]]
                            ),
                            precision=jax.lax.Precision.HIGHEST,
                        )
                        for target, block in enumerate(row)
                    ]
                )
                for source, row in enumerate(self.blocks)
            ]
        )
        return weights * pairs

    def fused(
        self, max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES
    ) -> "FusedBlockTransition":
        """Return a product-only operator with rank-one blocks folded together.

        See ``FusedBlockTransition``. Gaussian blocks whose float32 dense form
        fits ``_DENSE_GAUSSIAN_BLOCK_BYTES`` are stored dense while all retained
        dense matrices (as ``dense_fallback_nbytes`` counts them, float64 on the
        host) fit ``max_dense_bytes``; the rest keep separable products.
        Products agree with this operator to rounding; ``entries`` and dense
        views stay on this operator.

        Parameters
        ----------
        max_dense_bytes : int, optional
            Budget in bytes for all retained dense matrices, by default 256 MiB.

        Returns
        -------
        operator : FusedBlockTransition
            Operator with the same forward and backward products.
        """
        boundaries = np.cumsum((0, *self.state_sizes))
        n_states = len(self.state_sizes)
        sources: list[np.ndarray] = []
        destinations: list[np.ndarray] = []
        weight_index: list[int] = []
        blocks: list[Block] = []
        positions: list[tuple[int, int]] = []
        dense_budget = int(max_dense_bytes) - self.dense_fallback_nbytes
        for source, row in enumerate(self.blocks):
            for target, block in enumerate(row):
                factors = _rank_one_factors(block)
                if factors is _EMPTY_BLOCK:
                    continue
                if factors is None:
                    block = _small_gaussian_as_dense(block, dense_budget)
                    if isinstance(block, DenseBlock):
                        dense_budget -= np.asarray(block.matrix).nbytes
                    blocks.append(block)
                    positions.append((source, target))
                    continue
                column = np.zeros(boundaries[-1])
                column[boundaries[source] : boundaries[source + 1]] = factors[0]
                destination = np.zeros(boundaries[-1])
                destination[boundaries[target] : boundaries[target + 1]] = factors[1]
                sources.append(column)
                destinations.append(destination)
                weight_index.append(source * n_states + target)
        return FusedBlockTransition(
            np.stack(sources, axis=1) if sources else np.zeros((boundaries[-1], 0)),
            np.stack(destinations) if destinations else np.zeros((0, boundaries[-1])),
            np.asarray(weight_index, dtype=np.int32),
            tuple(blocks),
            tuple(positions),
            self.state_sizes,
            self.discrete_weights,
        )

    def entries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return selected entries of the combined transition matrix.

        Parameters
        ----------
        rows : np.ndarray of int
            Source-bin indices over the combined bins, broadcast against ``columns``.
        columns : np.ndarray of int
            Destination-bin indices over the combined bins, broadcast against
            ``rows``.

        Returns
        -------
        entries : np.ndarray, shape of the broadcast ``rows`` and ``columns``
            Transition probabilities ``T[rows, columns]``, including the bound
            discrete weights.
        """
        rows, columns = np.broadcast_arrays(rows, columns)
        shape = rows.shape
        rows, columns = rows.ravel(), columns.ravel()
        output = np.zeros(len(rows), dtype=float)
        boundaries = np.cumsum((0, *self.state_sizes))
        for source, row in enumerate(self.blocks):
            for target, block in enumerate(row):
                selected = (
                    (rows >= boundaries[source])
                    & (rows < boundaries[source + 1])
                    & (columns >= boundaries[target])
                    & (columns < boundaries[target + 1])
                )
                if np.any(selected):
                    weight = (
                        1.0
                        if self.discrete_weights is None
                        else np.asarray(self.discrete_weights)[source, target]
                    )
                    output[selected] = weight * block.entries(
                        rows[selected] - boundaries[source],
                        columns[selected] - boundaries[target],
                    )
        return output.reshape(shape)

    def to_dense(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
        dtype: DTypeLike | None = None,
    ) -> np.ndarray:
        """Convert to a dense matrix within a byte budget.

        Parameters
        ----------
        max_bytes : int, optional
            Largest allowed size of the dense matrix in bytes, by default 256 MiB.
        dtype : dtype, optional
            Data type of the result, by default float64.

        Returns
        -------
        matrix : np.ndarray, shape (n_bins, n_bins)
            The combined transition matrix.

        Raises
        ------
        DenseTransitionBudgetError
            If the dense matrix would exceed ``max_bytes``.
        """
        dtype = np.dtype(self.dtype if dtype is None else dtype)
        _budget(self.n_bins**2 * dtype.itemsize, max_bytes, "Dense combined transition")
        result = np.empty(self.shape, dtype=dtype)
        columns = np.arange(self.n_bins)[None, :]
        for start in range(0, self.n_bins, 64):
            stop = min(start + 64, self.n_bins)
            result[start:stop] = self.entries(np.arange(start, stop)[:, None], columns)
        return result


# Sentinel: annotated ``Any`` so ``is`` checks against it narrow the factor tuple.
_EMPTY_BLOCK: Any = object()

# Gaussian blocks up to this many bytes of float32 dense matrix are stored
# dense in the fused operator. A separable product costs several small
# kernels per step (scatter to the grid, scaling, per-axis products, gather);
# a dense block costs one matrix-vector product. On an A100, a filter step
# plus a smoothing step with transition statistics took 52 against 82 us for
# a 594-bin block (1.4 MB), 65 against 101 us at 1,648 bins (10.9 MB) and
# 122 against 105 us at 2,616 bins (27.4 MB).
_DENSE_GAUSSIAN_BLOCK_BYTES = 16 * 2**20


def _small_gaussian_as_dense(block: Block, dense_budget: int) -> Block:
    """A Gaussian (or restricted Gaussian) block as a ``DenseBlock`` when its
    float32 matrix fits ``_DENSE_GAUSSIAN_BLOCK_BYTES`` and its float64 host
    matrix fits ``dense_budget`` bytes; any other block as is."""
    original = block.original if isinstance(block, RestrictedBlock) else block
    n_rows, n_columns = block.shape
    if (
        not isinstance(original, GaussianGridBlock)
        or n_rows * n_columns * 4 > _DENSE_GAUSSIAN_BLOCK_BYTES
        or n_rows * n_columns * 8 > dense_budget
    ):
        return block
    return DenseBlock(
        block.entries(np.arange(n_rows)[:, None], np.arange(n_columns)[None, :])
    )


def _rank_one_factors(block: Block) -> tuple[np.ndarray, np.ndarray] | None:
    """Host ``(source, destination)`` vectors of a rank-one block, else None.

    A uniform block adds ``(source * normalization) . p`` times its destination
    mask; a scalar identity block is the same with unit vectors. Restricted
    blocks index their original's vectors. Returns ``_EMPTY_BLOCK`` for a
    restriction with no rows or columns, which contributes nothing.
    """
    if isinstance(block, RestrictedBlock):
        if not len(block.rows) or not len(block.columns):
            return _EMPTY_BLOCK  # type: ignore[no-any-return]  # sentinel, not a factor pair
        factors = _rank_one_factors(block.original)
        if factors is None or factors is _EMPTY_BLOCK:
            return None
        return cast(
            tuple[np.ndarray, np.ndarray],
            (factors[0][np.asarray(block.rows)], factors[1][np.asarray(block.columns)]),
        )
    if isinstance(block, UniformBlock):
        normalization = np.asarray(block.normalization, dtype=float)
        # A non-finite normalization (no valid destination) keeps its own product.
        if normalization.ndim or not np.isfinite(normalization):
            return None
        return (
            np.asarray(block.source_mask, dtype=float) * normalization,
            np.asarray(block.destination_mask, dtype=float),
        )
    if isinstance(block, IdentityBlock) and block.shape == (1, 1):
        return np.asarray(block.mask, dtype=float), np.ones(1)
    return None


@_register_operator
@dataclass(frozen=True)
class FusedBlockTransition:
    """Block-operator products with every rank-one block in two matrices.

    Uniform and scalar identity blocks are rank one. Their source vectors,
    scaled by their normalizations, form the columns of ``sources`` and their
    destination masks the rows of ``destinations``, so all of them cost one
    product with each matrix per call instead of a reduction, scatter and
    gather per block. Gaussian, dense and spatial identity blocks keep their
    own products, except small Gaussian blocks, which are stored dense (see
    ``_DENSE_GAUSSIAN_BLOCK_BYTES``). Any NaN in the input makes the whole
    output NaN, as in ``BlockTransitionOperator``. Products agree with that
    operator to rounding, not bitwise, because the rank-one sums are taken in
    a different order and small Gaussian blocks are products with a dense
    matrix.

    Built by ``BlockTransitionOperator.fused``. Forward products are ``p @ T``
    and backward products ``T @ v``.

    Attributes
    ----------
    sources : np.ndarray or jax.Array, shape (n_bins, n_rank_one)
        Source vectors of the rank-one blocks, scaled by their normalizations.
    destinations : np.ndarray or jax.Array, shape (n_rank_one, n_bins)
        Destination masks of the rank-one blocks.
    weight_index : np.ndarray or jax.Array, shape (n_rank_one,)
        Flat index into the ``(n_states, n_states)`` discrete weights for each
        rank-one block.
    blocks : tuple of Block
        Blocks that keep their own products.
    block_positions : tuple of tuple of int
        ``(source_state, target_state)`` of each entry of ``blocks``.
    state_sizes : tuple of int
        Number of spatial bins of each discrete state.
    discrete_weights : np.ndarray or jax.Array, shape (n_states, n_states), optional
        Stationary discrete transition probabilities, by default None (all ones).
    """

    sources: _Array  # (n_bins, n_rank_one)
    destinations: _Array  # (n_rank_one, n_bins)
    weight_index: _Array  # (n_rank_one,) flat index into the S-by-S weights
    blocks: tuple[Block, ...]
    block_positions: tuple[tuple[int, int], ...] = field(metadata={"static": True})
    state_sizes: tuple[int, ...] = field(metadata={"static": True})
    discrete_weights: _Array | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape ``(n_bins, n_bins)`` of the combined transition matrix."""
        return (sum(self.state_sizes),) * 2

    @property
    def n_bins(self) -> int:
        """Total number of bins over all discrete states."""
        return self.shape[0]

    def _flat_weights(self, values: jax.Array, override: _Array | None) -> jax.Array:
        weights = self.discrete_weights if override is None else override
        if weights is None:
            return jnp.ones(len(self.state_sizes) ** 2, dtype=values.dtype)
        if weights.shape != (len(self.state_sizes),) * 2:
            msg = "Pass one S-by-S discrete transition matrix per step"
            raise ValueError(msg)
        return jnp.asarray(weights, dtype=values.dtype).ravel()

    def forward(
        self, probabilities: ArrayLike, discrete_weights: _Array | None = None
    ) -> jax.Array:
        """Apply the transition from the left, ``probabilities @ T``.

        Parameters
        ----------
        probabilities : array_like, shape (n_bins,)
            Probability over the combined bins of all states.
        discrete_weights : np.ndarray or jax.Array, shape (n_states, n_states), optional
            Discrete state transition matrix for this step, overriding the bound
            stationary weights. By default None.

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            The one-step-ahead prediction ``probabilities @ T``.
        """
        probabilities = jnp.asarray(probabilities)
        weights = self._flat_weights(probabilities, discrete_weights)
        boundaries = tuple(np.cumsum((0, *self.state_sizes)))
        precision = jax.lax.Precision.HIGHEST
        source_sums = jnp.matmul(
            probabilities,
            jnp.asarray(self.sources, dtype=probabilities.dtype),
            precision=precision,
        )
        result = jnp.matmul(
            source_sums * weights[self.weight_index],
            jnp.asarray(self.destinations, dtype=probabilities.dtype),
            precision=precision,
        )
        for (source, target), block in zip(
            self.block_positions, self.blocks, strict=True
        ):
            values = probabilities[boundaries[source] : boundaries[source + 1]]
            result = result.at[boundaries[target] : boundaries[target + 1]].add(
                weights[source * len(self.state_sizes) + target] * block.forward(values)
            )
        return _nan_result(result, probabilities)

    def backward(
        self, values: ArrayLike, discrete_weights: _Array | None = None
    ) -> jax.Array:
        """Apply the transition from the right, ``T @ values``.

        Parameters
        ----------
        values : array_like, shape (n_bins,)
            Values over the combined bins of all states.
        discrete_weights : np.ndarray or jax.Array, shape (n_states, n_states), optional
            Discrete state transition matrix for this step, overriding the bound
            stationary weights. By default None.

        Returns
        -------
        result : jax.Array, shape (n_bins,)
            The product ``T @ values``.
        """
        values = jnp.asarray(values)
        weights = self._flat_weights(values, discrete_weights)
        boundaries = tuple(np.cumsum((0, *self.state_sizes)))
        precision = jax.lax.Precision.HIGHEST
        destination_sums = jnp.matmul(
            jnp.asarray(self.destinations, dtype=values.dtype),
            values,
            precision=precision,
        )
        result = jnp.matmul(
            jnp.asarray(self.sources, dtype=values.dtype),
            destination_sums * weights[self.weight_index],
            precision=precision,
        )
        for (source, target), block in zip(
            self.block_positions, self.blocks, strict=True
        ):
            part = values[boundaries[target] : boundaries[target + 1]]
            result = result.at[boundaries[source] : boundaries[source + 1]].add(
                weights[source * len(self.state_sizes) + target] * block.backward(part)
            )
        return _nan_result(result, values)

    def pair_products(
        self,
        filtered: ArrayLike,
        ratio: ArrayLike,
        discrete_weights: _Array | None = None,
    ) -> jax.Array:
        """Per-state-pair terms; see ``BlockTransitionOperator.pair_products``.

        A rank-one block contributes ``(filtered . source) * (destination .
        ratio)``, so all of them cost one product with each factor matrix.

        Parameters
        ----------
        filtered : jax.Array, shape (n_bins,)
            Filtered state probabilities at time ``t``.
        ratio : jax.Array, shape (n_bins,)
            Elementwise ratio of smoothed to predicted probabilities at time
            ``t + 1``.
        discrete_weights : jax.Array, shape (n_states, n_states), optional
            Overrides the bound stationary weights.

        Returns
        -------
        pairs : jax.Array, shape (n_states, n_states)
            Expected transition mass between each pair of discrete states.
        """
        filtered, ratio = jnp.asarray(filtered), jnp.asarray(ratio)
        weights = self._flat_weights(filtered, discrete_weights)
        n_states = len(self.state_sizes)
        boundaries = tuple(np.cumsum((0, *self.state_sizes)))
        precision = jax.lax.Precision.HIGHEST
        rank_one = jnp.matmul(
            filtered,
            jnp.asarray(self.sources, dtype=filtered.dtype),
            precision=precision,
        ) * jnp.matmul(
            jnp.asarray(self.destinations, dtype=filtered.dtype),
            ratio,
            precision=precision,
        )
        pairs = (
            jnp.zeros(n_states * n_states, dtype=filtered.dtype)
            .at[jnp.asarray(self.weight_index)]
            .add(rank_one)
        )
        for (source, target), block in zip(
            self.block_positions, self.blocks, strict=True
        ):
            pairs = pairs.at[source * n_states + target].add(
                jnp.dot(
                    filtered[boundaries[source] : boundaries[source + 1]],
                    block.backward(ratio[boundaries[target] : boundaries[target + 1]]),
                    precision=precision,
                )
            )
        return (weights * pairs).reshape(n_states, n_states)

    def backward_and_pairs(
        self,
        values: ArrayLike,
        filtered: ArrayLike,
        discrete_weights: _Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute ``backward(values)`` and ``pair_products(filtered, values)`` together.

        Each block's backward product, and the rank-one destination sums, are
        computed once and serve both, so the smoother's transition statistics
        do not repeat the separable Gaussian products.

        Parameters
        ----------
        values : jax.Array, shape (n_bins,)
            Values over the combined bins, used both as the vector to multiply
            and as the ratio in the pair products.
        filtered : jax.Array, shape (n_bins,)
            Filtered state probabilities for the pair products.
        discrete_weights : jax.Array, shape (n_states, n_states), optional
            Overrides the bound stationary weights.

        Returns
        -------
        backward : jax.Array, shape (n_bins,)
            The product ``T @ values``.
        pairs : jax.Array, shape (n_states, n_states)
            Per-state-pair terms of ``pair_products(filtered, values)``.
        """
        values, filtered = jnp.asarray(values), jnp.asarray(filtered)
        weights = self._flat_weights(values, discrete_weights)
        n_states = len(self.state_sizes)
        boundaries = tuple(np.cumsum((0, *self.state_sizes)))
        precision = jax.lax.Precision.HIGHEST
        sources = jnp.asarray(self.sources, dtype=values.dtype)
        destination_sums = jnp.matmul(
            jnp.asarray(self.destinations, dtype=values.dtype),
            values,
            precision=precision,
        )
        result = jnp.matmul(
            sources, destination_sums * weights[self.weight_index], precision=precision
        )
        pairs = (
            jnp.zeros(n_states * n_states, dtype=values.dtype)
            .at[jnp.asarray(self.weight_index)]
            .add(jnp.matmul(filtered, sources, precision=precision) * destination_sums)
        )
        for (source, target), block in zip(
            self.block_positions, self.blocks, strict=True
        ):
            product = block.backward(
                values[boundaries[target] : boundaries[target + 1]]
            )
            flat = source * n_states + target
            result = result.at[boundaries[source] : boundaries[source + 1]].add(
                weights[flat] * product
            )
            pairs = pairs.at[flat].add(
                jnp.dot(
                    filtered[boundaries[source] : boundaries[source + 1]],
                    product,
                    precision=precision,
                )
            )
        return _nan_result(result, values), (weights * pairs).reshape(
            n_states, n_states
        )


class LazyDenseTransition:
    """Read-only budgeted NumPy-compatible view; pickling stores only the operator.

    Basic and orthogonal/paired integer indexing materialize only the requested
    entries. Other NumPy indexing falls back to a budgeted full dense view.
    Dense results are not cached and cannot mutate the structured operator.

    Parameters
    ----------
    operator : BlockTransitionOperator
        The structured transition operator to view as a dense matrix.
    max_bytes : int, optional
        Largest allowed size of any materialized selection in bytes, by default
        256 MiB.
    """

    def __init__(
        self,
        operator: BlockTransitionOperator,
        max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    ) -> None:
        _budget(0, max_bytes, "Dense transition budget")
        self.operator = operator
        self.max_bytes = int(max_bytes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape ``(n_bins, n_bins)`` of the transition matrix."""
        return self.operator.shape

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Data type of the transition matrix, float64."""
        return self.operator.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions, always 2."""
        return 2

    @property
    def size(self) -> int:
        """Total number of entries in the transition matrix."""
        return int(np.prod(self.shape))

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        return self.operator.to_dense(max_bytes=self.max_bytes, dtype=dtype)

    def __getitem__(self, key: Any) -> Any:
        if key is Ellipsis:
            key = (slice(None), slice(None))
        if not isinstance(key, tuple):
            if np.ndim(key) > 1 and np.asarray(key).dtype == bool:
                # A full-shape mask selects elements, not rows.
                return np.asarray(self)[key]
            key = (key, slice(None))
        if len(key) != 2 or any(item is None or item is Ellipsis for item in key):
            return np.asarray(self)[key]
        indices = []
        basic = []
        for item, size in zip(key, self.shape, strict=True):
            if isinstance(item, slice):
                indices.append(np.arange(size)[item])
                basic.append(True)
            elif isinstance(item, (int, np.integer)):
                if not -size <= item < size:
                    msg = "Transition index out of range"
                    raise IndexError(msg)
                indices.append(np.asarray(item % size))
                basic.append(True)
            else:
                index = np.asarray(item)
                if index.dtype == bool and index.ndim == 1 and index.size == size:
                    index = np.flatnonzero(index)
                elif not np.issubdtype(index.dtype, np.integer):
                    return np.asarray(self)[key]
                if np.any((index < -size) | (index >= size)):
                    msg = "Transition index out of range"
                    raise IndexError(msg)
                indices.append(np.where(index < 0, index + size, index))
                basic.append(False)
        rows, columns = indices
        if rows.ndim and columns.ndim and any(basic):
            rows, columns = (
                rows[..., None],
                columns.reshape((1,) * rows.ndim + columns.shape),
            )
        requested_shape = np.broadcast_shapes(rows.shape, columns.shape)
        _budget(
            int(np.prod(requested_shape)) * self.dtype.itemsize,
            self.max_bytes,
            "Dense transition selection",
        )
        if not requested_shape:
            return self.operator.entries(rows, columns)
        # Evaluate leading-axis chunks of about to_dense's size so index
        # temporaries stay small; entries are elementwise, so values match.
        rows = np.broadcast_to(rows, requested_shape)
        columns = np.broadcast_to(columns, requested_shape)
        chunk = max(1, 64 * self.shape[1] // max(1, int(np.prod(requested_shape[1:]))))
        result = np.empty(requested_shape, dtype=float)
        for start in range(0, requested_shape[0], chunk):
            result[start : start + chunk] = self.operator.entries(
                rows[start : start + chunk], columns[start : start + chunk]
            )
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        msg = "Structured transitions are read-only; choose the dense representation to edit a matrix"
        raise TypeError(msg)

    def astype(self, dtype: DTypeLike, **kwargs: Any) -> np.ndarray:  # noqa: ARG002 -- ndarray.astype signature
        """Convert to a dense array within the byte budget.

        Parameters
        ----------
        dtype : dtype
            Data type of the result.
        **kwargs
            Ignored; accepted for compatibility with ``np.ndarray.astype``.

        Returns
        -------
        matrix : np.ndarray, shape (n_bins, n_bins)
            The dense transition matrix.
        """
        return self.operator.to_dense(max_bytes=self.max_bytes, dtype=dtype)


def _mask(environment: Environment) -> np.ndarray:
    return environment.is_track_interior_flat_


def _gaussian_block(
    transition: RandomWalk, environment: Environment
) -> GaussianGridBlock:
    if (
        environment.track_graph is not None
        or transition.use_manifold_distance
        or transition.direction is not None
    ):
        msg = (
            "Graph, manifold-distance and directional RandomWalk have no "
            f"structured operator; {_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    centers = np.asarray(environment.place_bin_centers_, dtype=float)
    axes = tuple(np.unique(centers[:, axis]) for axis in range(centers.shape[1]))
    shape = tuple(len(axis) for axis in axes)
    if int(np.prod(shape)) != len(centers):
        msg = (
            "Structured RandomWalk needs a complete Cartesian grid before "
            f"interior masking; {_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    expected = np.stack(
        [grid.ravel() for grid in np.meshgrid(*axes, indexing="ij")], axis=1
    )
    if not np.array_equal(centers, expected):
        msg = (
            "Structured RandomWalk needs canonical Cartesian bin ordering; "
            f"{_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    mean = np.asarray(transition.movement_mean)
    if mean.ndim != 0 or not np.isfinite(mean):
        msg = (
            "Structured RandomWalk supports only a finite scalar mean shift; "
            f"{_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    covariance = np.asarray(transition.movement_var, dtype=float)
    dimensions = centers.shape[1]
    if covariance.ndim == 0:
        variance = np.full(dimensions, covariance)
    elif covariance.shape == (dimensions,):
        variance = covariance
    elif covariance.shape == (dimensions, dimensions) and np.array_equal(
        covariance, np.diag(np.diag(covariance))
    ):
        variance = np.diag(covariance)
    else:
        msg = (
            "Structured RandomWalk needs a separable (diagonal) covariance; "
            f"{_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        msg = (
            "Structured RandomWalk needs a finite, positive variance; "
            f"{_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    mask = _mask(environment)
    raw_factors = tuple(
        norm.pdf(axis[None, :], loc=axis[:, None] + float(mean), scale=np.sqrt(var))
        for axis, var in zip(axes, variance, strict=True)
    )
    if not mask.any():
        return GaussianGridBlock(raw_factors, mask, np.zeros(len(mask)))

    # Query the closest VALID destination in covariance-scaled coordinates.
    # This O(N) storage check uses scipy's original density evaluation to keep
    # its genuinely all-zero float64 rows and reject subnormal normalization.
    valid = centers[mask]
    scaled_destinations = valid / np.sqrt(variance)
    scaled_sources = (valid + float(mean)) / np.sqrt(variance)
    if not np.all(np.isfinite(scaled_destinations)) or not np.all(
        np.isfinite(scaled_sources)
    ):
        msg = f"Structured RandomWalk coordinate scaling overflows; {_DENSE_BLOCK_HINT}"
        raise UnsupportedTransitionError(msg)
    _, closest = cKDTree(scaled_destinations).query(scaled_sources)
    distribution = multivariate_normal(mean=np.zeros(dimensions), cov=variance)
    deviations = valid[closest] - (valid + float(mean))
    largest_pdf = np.atleast_1d(distribution.pdf(deviations))
    largest_log_pdf = np.atleast_1d(distribution.logpdf(deviations))
    zero_cutoff = np.log(np.nextafter(0.0, 1.0)) - np.log(2.0)
    uncertainty = 32 * np.finfo(float).eps * np.maximum(1, np.abs(largest_log_pdf))
    ambiguous_zero = (largest_pdf == 0) & (largest_log_pdf + uncertainty >= zero_cutoff)
    subnormal = (largest_pdf > 0) & (largest_pdf < np.finfo(float).tiny)
    if not np.all(np.isfinite(largest_pdf)) or np.any(ambiguous_zero | subnormal):
        msg = (
            "Subnormal/overflowed original Gaussian PDF normalization has no "
            f"structured operator; {_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)

    # Multiplying each axis row by a positive constant cancels EXACTLY in the
    # final row normalization. Scale before casting to float32 so a normal
    # float64 density does not disappear before that normalization.
    mask_grid = mask.reshape(shape)
    factors: list[np.ndarray] | tuple[np.ndarray, ...] = []
    source_coordinates = np.unravel_index(np.flatnonzero(mask), shape)
    log_row_scale = np.zeros(len(valid))
    for axis, factor in enumerate(raw_factors):
        others = tuple(index for index in range(dimensions) if index != axis)
        eligible = np.any(mask_grid, axis=others)
        maximum = np.max(factor[:, eligible], axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            log_row_scale += np.log(maximum[source_coordinates[axis], 0])
        # Only columns outside `eligible` can exceed the row maximum and
        # overflow; they are zeroed next.
        with np.errstate(over="ignore"):
            scaled = np.where(
                maximum > 0, factor / np.where(maximum > 0, maximum, 1), 0
            )
        # Columns absent from every valid destination contribute exact zero.
        # Remove them before a tiny eligible maximum can produce inf * 0.
        scaled[:, ~eligible] = 0
        cast(list[np.ndarray], factors).append(scaled)
    # A normal SUM is insufficient when each contributing float32 product is
    # subnormal: accelerator contractions may flush every term before summing.
    largest_scaled_log_pdf = largest_log_pdf - log_row_scale
    if np.any(
        (largest_pdf > 0)
        & (largest_scaled_log_pdf <= np.log(np.finfo(np.float32).tiny))
    ):
        msg = (
            "Largest valid scaled Gaussian term is not safely normal in "
            f"float32; {_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    factors = tuple(factors)
    normalizer = _axes_product(
        mask.astype(float).reshape(shape), factors, backward=True, numpy=True
    ).ravel()
    original_positive = np.zeros(len(mask), dtype=bool)
    original_positive[mask] = largest_pdf > 0
    normalizer = np.where(original_positive, normalizer, 0)
    if np.any(original_positive & (normalizer < np.finfo(np.float32).tiny)):
        msg = (
            "Masked Gaussian normalization cannot be represented safely in "
            f"float32; {_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    return GaussianGridBlock(factors, mask, normalizer)


@runtime_checkable
class SupportsStructuredBlock(Protocol):
    """A continuous transition the operator builder can plan without
    calling ``make_state_transition``.

    A transition opts into structured operators by implementing these
    methods alongside its ``make_state_transition``, as the package's
    transitions do. A subclass that overrides ``make_state_transition``
    without also overriding ``structured_block`` is treated as a transition
    without them. Transitions without them fall back to their dense block
    with ``transition_representation="auto"``.
    """

    def structured_block(self, environments: Sequence[Environment]) -> Block:
        """The exact block between two spatial (or two one-bin) states.

        Parameters
        ----------
        environments : sequence of Environment
            The model's fitted environments.

        Returns
        -------
        block : Block
            Equal to ``make_state_transition(environments)``.

        Raises
        ------
        UnsupportedTransitionError
            If this configuration has no exact structured block; its dense
            block is then planned from ``dense_block_shape`` and
            ``graph_distance_environment``.
        """
        ...

    def dense_block_shape(self, environments: Sequence[Environment]) -> tuple[int, int]:
        """Shape of ``make_state_transition``'s block, without building it."""
        ...

    def graph_distance_environment(
        self, environments: Sequence[Environment]
    ) -> Environment | None:
        """The environment whose graph distances ``make_state_transition``
        reads, or None. Deferred distances are densified for it within the
        dense budget."""
        ...


def _defining_class_index(cls: type, name: str) -> int:
    """Position in ``cls.__mro__`` of the class that defines ``name``, or
    ``len(cls.__mro__)`` if only an instance does."""
    return next(
        (index for index, base in enumerate(cls.__mro__) if name in vars(base)),
        len(cls.__mro__),
    )


def _planned(transition: object) -> SupportsStructuredBlock | None:
    """The transition if its ``SupportsStructuredBlock`` methods describe its
    ``make_state_transition``, else None."""
    if not isinstance(transition, SupportsStructuredBlock):
        return None
    cls = type(transition)
    if _defining_class_index(cls, "structured_block") > _defining_class_index(
        cls, "make_state_transition"
    ):
        return None
    return transition


def _one_bin_block(
    transition: ContinuousTransition,
    environments: Sequence[Environment],
    source_size: int,
    target_size: int,
) -> UniformBlock | None:
    """The block between a one-bin state and a spatial state, or None.

    From a one-bin state (No-Spike, or Local without ``local_position_std``)
    the block is uniform over the interior bins of the environment named by
    the transition's ``environment_name``; into one, every bin moves there
    with probability 1. Both the dense and the structured transition use
    this block, without calling ``make_state_transition``.

    Parameters
    ----------
    transition : ContinuousTransition
    environments : sequence of Environment
    source_size, target_size : int
        Bins of the source and destination states.

    Returns
    -------
    block : UniformBlock or None
        None unless exactly one of the two states has one bin.
    """
    if source_size == 1 and target_size > 1:
        destination = _scalar_to_spatial_destination(transition, environments)
        normalization = np.array(
            1.0 / destination.sum() if destination.any() else np.nan
        )
        return UniformBlock(np.ones(1, dtype=bool), destination, normalization)
    if source_size > 1 and target_size == 1:
        return UniformBlock(
            np.ones(source_size, dtype=bool), np.ones(1, dtype=bool), np.array(1.0)
        )
    return None


def _structured_block(
    transition: ContinuousTransition,
    environments: Sequence[Environment],
    source_size: int,
    target_size: int,
) -> Block:
    block = _one_bin_block(transition, environments, source_size, target_size)
    if block is not None:
        return block
    planned = _planned(transition)
    if planned is None:
        msg = (
            f"No proven exact structured operator for {type(transition).__name__}; "
            f"{_DENSE_BLOCK_HINT}"
        )
        raise UnsupportedTransitionError(msg)
    return planned.structured_block(environments)


# Every block type, to tell planned blocks from transitions awaiting a dense
# constructor.
_BLOCK_TYPES = (
    UniformBlock,
    IdentityBlock,
    GaussianGridBlock,
    DenseBlock,
    RestrictedBlock,
)


def _check_fallback_shape(
    transition: object, environments: Sequence[Environment], expected: tuple[int, int]
) -> None:
    """Fail an invalid planned shape before its dense constructor allocates."""
    planned = _planned(transition)
    if planned is None:
        return
    shape = planned.dense_block_shape(environments)
    if shape != expected:
        msg = (
            f"Dense transition block shape {shape} does not match state bins {expected}"
        )
        raise ValueError(msg)


def build_transition_operator(
    transition_types: Sequence[Sequence[ContinuousTransition]],
    environments: Sequence[Environment],
    state_sizes: Sequence[int],
    observation_models: Sequence[ObservationModel] | None = None,
    local_position_std: float | None = None,
    *,
    allow_dense_fallback: bool = False,
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    context: TransitionContext | None = None,
) -> BlockTransitionOperator:
    """Build exact padded conditional blocks and preflight dense fallback.

    Scalar-to-spatial and spatial-to-scalar blocks, and the blocks of
    transitions implementing ``SupportsStructuredBlock`` (Uniform, Identity,
    scalar ``Discrete``, and Euclidean, nondirectional RandomWalks with a
    scalar mean and separable covariance on complete Cartesian grids) are
    built without dense matrices.
    Scalar/spatial blocks and the multi-bin Local ``Discrete`` upgrade match
    the dense detector constructor. Any other block raises
    ``UnsupportedTransitionError`` unless ``allow_dense_fallback=True``; the
    retained dense blocks plus the temporary dense graph distances that
    ``graph_distance_environment`` names must then fit ``max_dense_bytes``.
    Dense blocks are checked like the dense constructor's (shape, finite,
    non-negative, rows summing to 1 or 0).

    Parameters
    ----------
    transition_types : sequence of sequence of ContinuousTransition
        ``(n_states, n_states)`` grid of continuous transition models.
    environments : list of Environment
        Fitted environments.
    state_sizes : sequence of int
        Padded bins per discrete state.
    observation_models : sequence of ObservationModel, optional
        With ``local_position_std``, turns Local ``Discrete`` blocks into
        Uniform blocks over the Local environment.
    local_position_std : float, optional
        Multi-bin Local width; None keeps a single-bin Local state.
    allow_dense_fallback : bool, optional
        Build unsupported blocks with their dense constructor.
    max_dense_bytes : int, optional
        Dense fallback budget, by default 256 MiB.
    context : TransitionContext, optional
        Fit data passed to every dense fallback constructor; required by
        ``EmpiricalMovement`` fallback blocks.

    Returns
    -------
    operator : BlockTransitionOperator
        Blocks in source-major order and the padded state sizes.
    """
    sizes = tuple(int(size) for size in state_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        msg = "state_sizes must contain positive padded bin counts"
        raise ValueError(msg)
    count = len(sizes)
    if len(transition_types) != count or any(
        len(row) != count for row in transition_types
    ):
        msg = "transition_types must be an n_states by n_states grid"
        raise ValueError(msg)
    _budget(0, max_dense_bytes, "Dense fallback budget")
    plan: list[list[Any]] = []
    fallback_bytes = 0
    for source, row in enumerate(transition_types):
        planned_row: list[Any] = []
        for target, transition in enumerate(row):
            _check_continuous_transition(transition)
            if observation_models is not None:
                transition = _multi_bin_local_transition(
                    transition,
                    observation_models[source],
                    observation_models[target],
                    local_position_std,
                )
            try:
                block = _structured_block(
                    transition, environments, sizes[source], sizes[target]
                )
                if block.shape != (sizes[source], sizes[target]):
                    msg = f"Transition block shape {block.shape} does not match state bins {(sizes[source], sizes[target])}"
                    raise ValueError(msg)
                planned_row.append(block)
            except UnsupportedTransitionError:
                if not allow_dense_fallback:
                    raise
                _check_fallback_shape(
                    transition, environments, (sizes[source], sizes[target])
                )
                fallback_bytes += (
                    sizes[source] * sizes[target] * np.dtype(float).itemsize
                )
                planned_row.append(transition)
        plan.append(planned_row)
    _budget(fallback_bytes, max_dense_bytes, "Retained dense fallback blocks")
    # Legacy manifold constructors require an ndarray and otherwise silently
    # substitute Euclidean distances. Only explicit, budgeted temporary views
    # may bridge that legacy contract. Opaque custom constructors have no
    # verified distance-view contract and cannot claim bounded exact fallback.
    distance_environments = {}
    for row in plan:
        for block in row:
            if isinstance(block, _BLOCK_TYPES):
                continue
            planned = _planned(block)
            if planned is None and any(
                isinstance(env.distance_between_nodes_, LazyGraphDistances)
                for env in environments
            ):
                msg = (
                    "Opaque custom dense fallback with deferred graph "
                    "distances has no verified exact distance contract; use "
                    'transition_representation="dense"'
                )
                raise UnsupportedTransitionError(msg)
            env = (
                None
                if planned is None
                else planned.graph_distance_environment(environments)
            )
            if env is not None and isinstance(
                env.distance_between_nodes_, LazyGraphDistances
            ):
                distance_environments[id(env)] = env
    distance_bytes = sum(
        cast(LazyGraphDistances, env.distance_between_nodes_).shape[0] ** 2 * 8
        for env in distance_environments.values()
    )
    _budget(
        fallback_bytes + distance_bytes,
        max_dense_bytes,
        "Retained dense fallback blocks plus temporary graph distances",
    )
    fallback_environments = []
    for env in environments:
        if id(env) in distance_environments:
            view = copy(env)
            view.distance_between_nodes_ = cast(
                LazyGraphDistances, env.distance_between_nodes_
            ).to_dense(max_bytes=max_dense_bytes - fallback_bytes)
            fallback_environments.append(view)
        else:
            fallback_environments.append(env)
    blocks: list[tuple[Block, ...]] = []
    for source, row in enumerate(plan):
        result: list[Block] = []
        for target, block in enumerate(row):
            if isinstance(block, _BLOCK_TYPES):
                result.append(block)
                continue
            matrix = _make_transition_block(
                block, fallback_environments, context, (sizes[source], sizes[target])
            )
            result.append(DenseBlock(np.array(matrix, dtype=float, copy=True)))
        blocks.append(tuple(result))
    return BlockTransitionOperator(tuple(blocks), sizes)
