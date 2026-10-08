"""Exact conditional state-block transitions without a combined dense matrix.

Forward products are ``p @ T``; backward products are ``T @ v``. Discrete
state weights remain an S-by-S array and never expand to N-by-N. Structured
Gaussian factors retain every grid-to-grid term, including tails and holes.
The host builder is outside JAX tracing; runtime products are pure JAX.
"""

from copy import copy
from dataclasses import dataclass, field, fields, replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal, norm

from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    Uniform,
)
from non_local_detector.graph_distances import LazyGraphDistances

DEFAULT_MAX_DENSE_BYTES = 256 * 1024**2


def _register_operator(cls):
    """Register explicit fields on JAX versions without inferred dataclasses."""
    data = tuple(
        item.name for item in fields(cls) if not item.metadata.get("static", False)
    )
    static = tuple(
        item.name for item in fields(cls) if item.metadata.get("static", False)
    )

    def flatten(value):
        return (
            tuple(getattr(value, name) for name in data),
            tuple(getattr(value, name) for name in static),
        )

    def unflatten(metadata, children):
        return cls(
            **dict(zip(data, children, strict=True)),
            **dict(zip(static, metadata, strict=True)),
        )

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


class UnsupportedTransitionError(ValueError):
    """A configuration has no proven exact structured implementation."""


class DenseTransitionBudgetError(MemoryError):
    """A requested dense allocation exceeds its explicit byte budget."""


def _budget(nbytes, maximum, description):
    if (
        not isinstance(maximum, (int, np.integer))
        or isinstance(maximum, bool)
        or maximum < 0
    ):
        raise ValueError("Dense transition byte budget must be a nonnegative integer")
    if nbytes > maximum:
        raise DenseTransitionBudgetError(
            f"{description} requires {nbytes:,} bytes, exceeding the "
            f"{maximum:,}-byte dense transition budget. Use supported structured "
            "products, request a smaller dense selection, or explicitly choose "
            "a feasible dense budget. EM/Viterbi/plotting may require dense data."
        )


def _nan_result(result, source):
    # Dense products propagate any NaN through even zero matrix entries.
    return jnp.where(jnp.any(jnp.isnan(source)), jnp.full_like(result, jnp.nan), result)


def _axes_product(values, factors, *, backward=False, numpy=False):
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


def _ratio(numerator, denominator):
    denominator = np.asarray(denominator)
    return np.where(
        denominator > 0, numerator / np.where(denominator > 0, denominator, 1), 0
    )


@_register_operator
@dataclass(frozen=True)
class UniformBlock:
    """A masked rank-one block, including legacy scalar/spatial overrides."""

    source_mask: object
    destination_mask: object
    normalization: object

    @property
    def shape(self):
        return (len(self.source_mask), len(self.destination_mask))

    def forward(self, values):
        source = jnp.asarray(self.source_mask, dtype=values.dtype)
        destination = jnp.asarray(self.destination_mask, dtype=values.dtype)
        divisor = jnp.asarray(self.normalization, dtype=values.dtype)
        result = jnp.sum(values * source * divisor) * destination
        return _nan_result(result, values)

    def backward(self, values):
        source = jnp.asarray(self.source_mask, dtype=values.dtype)
        destination = jnp.asarray(self.destination_mask, dtype=values.dtype)
        divisor = jnp.asarray(self.normalization, dtype=values.dtype)
        return _nan_result(source * jnp.sum(values * destination * divisor), values)

    def entries(self, rows, columns):
        return (
            np.asarray(self.source_mask)[rows]
            * np.asarray(self.destination_mask)[columns]
        ) * np.asarray(self.normalization)


@_register_operator
@dataclass(frozen=True)
class IdentityBlock:
    mask: object

    @property
    def shape(self):
        return (len(self.mask), len(self.mask))

    def forward(self, values):
        return _nan_result(values * jnp.asarray(self.mask, dtype=values.dtype), values)

    def backward(self, values):
        return self.forward(values)

    def entries(self, rows, columns):
        return np.asarray(self.mask)[rows] * (rows == columns)


@_register_operator
@dataclass(frozen=True)
class GaussianGridBlock:
    """Masked, row-normalized tensor product of untruncated Gaussian axes."""

    factors: tuple
    mask: object
    row_normalization: object

    @property
    def grid_shape(self):
        return tuple(factor.shape[0] for factor in self.factors)

    @property
    def shape(self):
        return (int(np.prod(self.grid_shape)),) * 2

    def forward(self, values):
        mask = jnp.asarray(self.mask, dtype=values.dtype)
        normalizer = jnp.asarray(self.row_normalization, dtype=values.dtype)
        safe = jnp.where(normalizer > 0, normalizer, 1)
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1)
        # Keep the original arithmetic for ordinary values. Cap finite scales
        # so a compiler-hoisted reciprocal remains normal instead of flushing
        # to zero; row-normalized products still undo exactly this scale.
        limit = jnp.asarray(1 / jnp.finfo(values.dtype).tiny, dtype=values.dtype)
        scale = jnp.where(jnp.isfinite(scale), jnp.minimum(scale, limit), scale)
        weighted = jnp.where(normalizer > 0, (values / scale) * mask / safe, 0)
        result = (
            _axes_product(weighted.reshape(self.grid_shape), self.factors).ravel()
            * mask
        )
        return _nan_result(result * scale, values)

    def backward(self, values):
        mask = jnp.asarray(self.mask, dtype=values.dtype)
        normalizer = jnp.asarray(self.row_normalization, dtype=values.dtype)
        safe = jnp.where(normalizer > 0, normalizer, 1)
        maximum = jnp.max(jnp.abs(values))
        scale = jnp.where(maximum > 0, maximum, 1)
        limits = jnp.finfo(values.dtype)
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

    def entries(self, rows, columns):
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
    """Explicit fallback; its retained matrix counts against the dense budget."""

    matrix: object

    @property
    def shape(self):
        return self.matrix.shape

    def forward(self, values):
        return jnp.matmul(
            values,
            jnp.asarray(self.matrix, dtype=values.dtype),
            precision=jax.lax.Precision.HIGHEST,
        )

    def backward(self, values):
        return jnp.matmul(
            jnp.asarray(self.matrix, dtype=values.dtype),
            values,
            precision=jax.lax.Precision.HIGHEST,
        )

    def entries(self, rows, columns):
        return np.asarray(self.matrix)[rows, columns]


@_register_operator
@dataclass(frozen=True)
class RestrictedBlock:
    """An exact indexed submatrix; selection never renormalizes its rows."""

    original: object
    rows: object
    columns: object

    @property
    def shape(self):
        return (len(self.rows), len(self.columns))

    def forward(self, values):
        if not len(self.rows) or not len(self.columns):
            return jnp.zeros(len(self.columns), dtype=values.dtype)
        full = (
            jnp.zeros(self.original.shape[0], dtype=values.dtype)
            .at[jnp.asarray(self.rows)]
            .set(values)
        )
        return self.original.forward(full)[jnp.asarray(self.columns)]

    def backward(self, values):
        if not len(self.rows) or not len(self.columns):
            return jnp.zeros(len(self.rows), dtype=values.dtype)
        full = (
            jnp.zeros(self.original.shape[1], dtype=values.dtype)
            .at[jnp.asarray(self.columns)]
            .set(values)
        )
        return self.original.backward(full)[jnp.asarray(self.rows)]

    def entries(self, rows, columns):
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
    """

    blocks: tuple
    state_sizes: tuple = field(metadata={"static": True})
    discrete_weights: object = None

    @property
    def shape(self):
        return (sum(self.state_sizes),) * 2

    @property
    def n_bins(self):
        return self.shape[0]

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @property
    def storage_nbytes(self):
        return sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(self))

    @property
    def dense_fallback_nbytes(self):
        def size(block):
            if isinstance(block, RestrictedBlock):
                return size(block.original)
            return (
                np.asarray(block.matrix).nbytes if isinstance(block, DenseBlock) else 0
            )

        return sum(size(block) for row in self.blocks for block in row)

    def bind_discrete(self, weights):
        if np.shape(weights) != (len(self.state_sizes),) * 2:
            raise ValueError("discrete_weights must have shape (n_states, n_states)")
        return replace(self, discrete_weights=weights)

    def restricted(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_bins,):
            raise ValueError(
                "Restriction mask must have one entry per combined state bin"
            )
        boundaries = np.cumsum((0, *self.state_sizes))
        indices = tuple(
            np.flatnonzero(mask[start:stop])
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
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

    def _weights(self, values, override):
        weights = self.discrete_weights if override is None else override
        if weights is None:
            return jnp.ones((len(self.state_sizes),) * 2, dtype=values.dtype)
        if weights.shape != (len(self.state_sizes),) * 2:
            raise ValueError("Pass one S-by-S discrete transition matrix per step")
        return jnp.asarray(weights, dtype=values.dtype)

    def forward(self, probabilities, discrete_weights=None):
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

    def backward(self, values, discrete_weights=None):
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

    def fused(self):
        """Return a product-only operator with rank-one blocks folded together.

        See ``FusedBlockTransition``. Products agree with this operator to
        rounding; ``entries`` and dense views stay on this operator.
        """
        boundaries = np.cumsum((0, *self.state_sizes))
        n_states = len(self.state_sizes)
        sources, destinations, weight_index = [], [], []
        blocks, positions = [], []
        for source, row in enumerate(self.blocks):
            for target, block in enumerate(row):
                factors = _rank_one_factors(block)
                if factors is _EMPTY_BLOCK:
                    continue
                if factors is None:
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

    def entries(self, rows, columns):
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

    def to_dense(self, *, max_bytes=DEFAULT_MAX_DENSE_BYTES, dtype=None):
        dtype = np.dtype(self.dtype if dtype is None else dtype)
        _budget(self.n_bins**2 * dtype.itemsize, max_bytes, "Dense combined transition")
        result = np.empty(self.shape, dtype=dtype)
        columns = np.arange(self.n_bins)[None, :]
        for start in range(0, self.n_bins, 64):
            stop = min(start + 64, self.n_bins)
            result[start:stop] = self.entries(np.arange(start, stop)[:, None], columns)
        return result


_EMPTY_BLOCK = object()


def _rank_one_factors(block):
    """Host ``(source, destination)`` vectors of a rank-one block, else None.

    A uniform block adds ``(source * normalization) . p`` times its destination
    mask; a scalar identity block is the same with unit vectors. Restricted
    blocks index their original's vectors. Returns ``_EMPTY_BLOCK`` for a
    restriction with no rows or columns, which contributes nothing.
    """
    if isinstance(block, RestrictedBlock):
        if not len(block.rows) or not len(block.columns):
            return _EMPTY_BLOCK
        factors = _rank_one_factors(block.original)
        if factors is None or factors is _EMPTY_BLOCK:
            return None
        return factors[0][np.asarray(block.rows)], factors[1][np.asarray(block.columns)]
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
    own products. Any NaN in the input makes the whole output NaN, as in
    ``BlockTransitionOperator``. Products agree with that operator to rounding,
    not bitwise, because the rank-one sums are taken in a different order.

    Built by ``BlockTransitionOperator.fused``. Forward products are ``p @ T``
    and backward products ``T @ v``.
    """

    sources: object  # (n_bins, n_rank_one)
    destinations: object  # (n_rank_one, n_bins)
    weight_index: object  # (n_rank_one,) flat index into the S-by-S weights
    blocks: tuple
    block_positions: tuple = field(metadata={"static": True})
    state_sizes: tuple = field(metadata={"static": True})
    discrete_weights: object = None

    @property
    def shape(self):
        return (sum(self.state_sizes),) * 2

    @property
    def n_bins(self):
        return self.shape[0]

    def _flat_weights(self, values, override):
        weights = self.discrete_weights if override is None else override
        if weights is None:
            return jnp.ones(len(self.state_sizes) ** 2, dtype=values.dtype)
        if weights.shape != (len(self.state_sizes),) * 2:
            raise ValueError("Pass one S-by-S discrete transition matrix per step")
        return jnp.asarray(weights, dtype=values.dtype).ravel()

    def forward(self, probabilities, discrete_weights=None):
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

    def backward(self, values, discrete_weights=None):
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


class LazyDenseTransition:
    """Read-only budgeted NumPy-compatible view; pickling stores only the operator.

    Basic and orthogonal/paired integer indexing materialize only the requested
    entries. Other NumPy indexing falls back to a budgeted full dense view.
    Dense results are not cached and cannot mutate the structured operator.
    """

    def __init__(self, operator, max_bytes=DEFAULT_MAX_DENSE_BYTES):
        _budget(0, max_bytes, "Dense transition budget")
        self.operator = operator
        self.max_bytes = int(max_bytes)

    @property
    def shape(self):
        return self.operator.shape

    @property
    def dtype(self):
        return self.operator.dtype

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __array__(self, dtype=None, copy=None):
        return self.operator.to_dense(max_bytes=self.max_bytes, dtype=dtype)

    def __getitem__(self, key):
        if key is Ellipsis:
            key = (slice(None), slice(None))
        if not isinstance(key, tuple):
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
                    raise IndexError("Transition index out of range")
                indices.append(np.asarray(item % size))
                basic.append(True)
            else:
                index = np.asarray(item)
                if index.dtype == bool and index.ndim == 1 and index.size == size:
                    index = np.flatnonzero(index)
                elif not np.issubdtype(index.dtype, np.integer):
                    return np.asarray(self)[key]
                if np.any((index < -size) | (index >= size)):
                    raise IndexError("Transition index out of range")
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

    def __setitem__(self, key, value):
        raise TypeError(
            "Structured transitions are read-only; choose the dense representation to edit a matrix"
        )

    def astype(self, dtype, **kwargs):
        return self.operator.to_dense(max_bytes=self.max_bytes, dtype=dtype)


def _find_environment(environments, name):
    for environment in environments:
        if environment.environment_name == name:
            return environment
    raise ValueError(f"Environment {name!r} not found")


def _mask(environment):
    count = len(environment.place_bin_centers_)
    return (
        np.ones(count, dtype=bool)
        if environment.is_track_interior_ is None
        else np.asarray(environment.is_track_interior_, dtype=bool).ravel()
    )


def _gaussian_block(transition, environment):
    if (
        environment.track_graph is not None
        or transition.use_manifold_distance
        or transition.direction is not None
    ):
        raise UnsupportedTransitionError(
            "Graph, manifold-distance and directional RandomWalk require dense fallback"
        )
    centers = np.asarray(environment.place_bin_centers_, dtype=float)
    axes = tuple(np.unique(centers[:, axis]) for axis in range(centers.shape[1]))
    shape = tuple(len(axis) for axis in axes)
    if int(np.prod(shape)) != len(centers):
        raise UnsupportedTransitionError(
            "RandomWalk needs a complete Cartesian grid before interior masking"
        )
    expected = np.stack(
        [grid.ravel() for grid in np.meshgrid(*axes, indexing="ij")], axis=1
    )
    if not np.array_equal(centers, expected):
        raise UnsupportedTransitionError(
            "RandomWalk grid must retain canonical Cartesian ordering"
        )
    mean = np.asarray(transition.movement_mean)
    if mean.ndim != 0 or not np.isfinite(mean):
        raise UnsupportedTransitionError(
            "Structured RandomWalk supports only a finite scalar mean shift"
        )
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
        raise UnsupportedTransitionError(
            "Nonseparable covariance requires dense RandomWalk fallback"
        )
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise UnsupportedTransitionError(
            "Structured Gaussian variance must be finite and positive"
        )
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
        raise UnsupportedTransitionError(
            "Gaussian coordinate scaling needs dense fallback"
        )
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
        raise UnsupportedTransitionError(
            "Subnormal/overflowed original Gaussian PDF normalization requires dense fallback"
        )

    # Multiplying each axis row by a positive constant cancels EXACTLY in the
    # final row normalization. Scale before casting to float32 so a normal
    # float64 density does not disappear before that normalization.
    mask_grid = mask.reshape(shape)
    factors = []
    source_coordinates = np.unravel_index(np.flatnonzero(mask), shape)
    log_row_scale = np.zeros(len(valid))
    for axis, factor in enumerate(raw_factors):
        others = tuple(index for index in range(dimensions) if index != axis)
        eligible = np.any(mask_grid, axis=others)
        maximum = np.max(factor[:, eligible], axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            log_row_scale += np.log(maximum[source_coordinates[axis], 0])
        scaled = np.where(maximum > 0, factor / np.where(maximum > 0, maximum, 1), 0)
        # Columns absent from every valid destination contribute exact zero.
        # Remove them before a tiny eligible maximum can produce inf * 0.
        scaled[:, ~eligible] = 0
        factors.append(scaled)
    # A normal SUM is insufficient when each contributing float32 product is
    # subnormal: accelerator contractions may flush every term before summing.
    largest_scaled_log_pdf = largest_log_pdf - log_row_scale
    if np.any(
        (largest_pdf > 0)
        & (largest_scaled_log_pdf <= np.log(np.finfo(np.float32).tiny))
    ):
        raise UnsupportedTransitionError(
            "Largest valid scaled Gaussian term is not safely normal in float32; use dense fallback"
        )
    factors = tuple(factors)
    normalizer = _axes_product(
        mask.astype(float).reshape(shape), factors, backward=True, numpy=True
    ).ravel()
    original_positive = np.zeros(len(mask), dtype=bool)
    original_positive[mask] = largest_pdf > 0
    normalizer = np.where(original_positive, normalizer, 0)
    if np.any(original_positive & (normalizer < np.finfo(np.float32).tiny)):
        raise UnsupportedTransitionError(
            "Masked Gaussian normalization cannot be represented safely in float32; use dense fallback"
        )
    return GaussianGridBlock(factors, mask, normalizer)


def _structured_block(transition, environments, source_size, target_size):
    if isinstance(transition, EmpiricalMovement):
        raise UnsupportedTransitionError(
            "EmpiricalMovement requires its original fit-context dense fallback"
        )
    if source_size == 1 and target_size > 1:
        if not hasattr(transition, "environment_name"):
            raise UnsupportedTransitionError(
                "Scalar-to-spatial transition needs the original environment_name"
            )
        destination = _mask(
            _find_environment(environments, transition.environment_name)
        )
        normalization = np.array(
            1.0 / destination.sum() if destination.any() else np.nan
        )
        return UniformBlock(np.ones(1, dtype=bool), destination, normalization)
    if source_size > 1 and target_size == 1:
        return UniformBlock(
            np.ones(source_size, dtype=bool), np.ones(1, dtype=bool), np.array(1.0)
        )
    if type(transition) is Discrete and source_size == target_size == 1:
        return IdentityBlock(np.ones(1, dtype=bool))
    if type(transition) is Uniform:
        source = _mask(_find_environment(environments, transition.environment_name))
        destination = (
            source.copy()
            if transition.environment2_name is None
            else _mask(_find_environment(environments, transition.environment2_name))
        )
        return UniformBlock(
            source,
            destination,
            np.array(1.0 / destination.sum() if destination.any() else 0.0),
        )
    if type(transition) is Identity:
        return IdentityBlock(
            _mask(_find_environment(environments, transition.environment_name))
        )
    if type(transition) is RandomWalk:
        return _gaussian_block(
            transition, _find_environment(environments, transition.environment_name)
        )
    raise UnsupportedTransitionError(
        f"No proven exact structured operator for {type(transition).__name__}"
    )


def _check_known_fallback_shape(transition, environments, expected):
    """Fail invalid primitive shapes before their dense constructor allocates."""
    if type(transition) is Discrete:
        shape = (1, 1)
    elif type(transition) in (
        Identity,
        RandomWalk,
        RandomWalkDirection1,
        RandomWalkDirection2,
        EmpiricalMovement,
        Uniform,
    ):
        env = _find_environment(environments, transition.environment_name)
        count = len(env.place_bin_centers_)
        other = count
        if type(transition) is Uniform and transition.environment2_name is not None:
            other = len(
                _find_environment(
                    environments, transition.environment2_name
                ).place_bin_centers_
            )
        shape = (count, other)
    else:
        return
    if np.broadcast_shapes(shape, expected) != expected:
        raise ValueError(
            f"Dense transition block shape {shape} does not match state bins {expected}"
        )


def build_transition_operator(
    transition_types,
    environments,
    state_sizes,
    observation_models=None,
    local_position_std=None,
    *,
    allow_dense_fallback=False,
    max_dense_bytes=DEFAULT_MAX_DENSE_BYTES,
    position=None,
    is_training=None,
    encoding_group_labels=None,
    environment_labels=None,
):
    """Build exact padded conditional blocks and preflight cumulative fallback.

    Scalar/spatial overrides and the multi-bin Local ``Discrete`` upgrade match
    the existing detector constructor. Only exact Cartesian separable Euclidean
    Gaussians bypass its dense constructor. Unsupported configurations require
    ``allow_dense_fallback=True`` and a feasible SUM of retained block bytes.
    """
    sizes = tuple(int(size) for size in state_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("state_sizes must contain positive padded bin counts")
    count = len(sizes)
    if len(transition_types) != count or any(
        len(row) != count for row in transition_types
    ):
        raise ValueError("transition_types must be an n_states by n_states grid")
    _budget(0, max_dense_bytes, "Dense fallback budget")
    plan = []
    fallback_bytes = 0
    for source, row in enumerate(transition_types):
        planned_row = []
        for target, transition in enumerate(row):
            if (
                isinstance(transition, Discrete)
                and local_position_std is not None
                and observation_models is not None
            ):
                from_obs, to_obs = (
                    observation_models[source],
                    observation_models[target],
                )
                if from_obs.is_local or to_obs.is_local:
                    if from_obs.environment_name != to_obs.environment_name:
                        raise ValueError(
                            "Cross-environment Discrete with multi-bin Local is unsupported; use explicit Uniform"
                        )
                    transition = Uniform(
                        environment_name=(
                            from_obs if from_obs.is_local else to_obs
                        ).environment_name
                    )
            try:
                block = _structured_block(
                    transition, environments, sizes[source], sizes[target]
                )
                if block.shape != (sizes[source], sizes[target]):
                    raise ValueError(
                        f"Transition block shape {block.shape} does not match state bins {(sizes[source], sizes[target])}"
                    )
                planned_row.append(block)
            except UnsupportedTransitionError:
                if not allow_dense_fallback:
                    raise
                _check_known_fallback_shape(
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
    known = (
        Discrete,
        Uniform,
        Identity,
        RandomWalk,
        RandomWalkDirection1,
        RandomWalkDirection2,
        EmpiricalMovement,
    )
    for row in plan:
        for block in row:
            if isinstance(block, (UniformBlock, IdentityBlock, GaussianGridBlock)):
                continue
            if type(block) not in known and any(
                isinstance(env.distance_between_nodes_, LazyGraphDistances)
                for env in environments
            ):
                raise UnsupportedTransitionError(
                    "Opaque custom dense fallback with deferred graph distances has no verified exact distance contract; use the legacy dense representation"
                )
            if type(block) is RandomWalk and block.use_manifold_distance:
                env = _find_environment(environments, block.environment_name)
                if isinstance(env.distance_between_nodes_, LazyGraphDistances):
                    distance_environments[id(env)] = env
    distance_bytes = sum(
        env.distance_between_nodes_.shape[0] ** 2 * 8
        for env in distance_environments.values()
    )
    _budget(
        fallback_bytes + distance_bytes,
        max_dense_bytes,
        "Retained dense fallback blocks plus temporary graph distances",
    )
    fallback_environments = []
    originals_by_view = {}
    for env in environments:
        if id(env) in distance_environments:
            view = copy(env)
            view.distance_between_nodes_ = env.distance_between_nodes_.to_dense(
                max_bytes=max_dense_bytes - fallback_bytes
            )
            fallback_environments.append(view)
            originals_by_view[id(view)] = env
        else:
            fallback_environments.append(env)
    blocks = []
    for source, row in enumerate(plan):
        result = []
        for target, block in enumerate(row):
            if isinstance(block, (UniformBlock, IdentityBlock, GaussianGridBlock)):
                result.append(block)
                continue
            if isinstance(block, EmpiricalMovement):
                if position is None:
                    raise UnsupportedTransitionError(
                        "EmpiricalMovement dense fallback requires position fit context"
                    )
                training = (
                    np.ones(len(position), dtype=bool)
                    if is_training is None
                    else np.asarray(is_training).squeeze()
                )
                labels = (
                    np.zeros(len(position), dtype=np.int32)
                    if encoding_group_labels is None
                    else encoding_group_labels
                )
                matrix = block.make_state_transition(
                    fallback_environments,
                    position,
                    training,
                    labels,
                    environment_labels,
                )
            else:
                matrix = block.make_state_transition(fallback_environments)
            # Primitive descriptors retain their environment. Restore the
            # original sparse views so pickling never retains temporary N²
            # distance matrices after the fitted dense block is constructed.
            for name, value in getattr(block, "__dict__", {}).items():
                if id(value) in originals_by_view:
                    setattr(block, name, originals_by_view[id(value)])
            matrix = np.array(
                np.broadcast_to(matrix, (sizes[source], sizes[target])),
                dtype=float,
                copy=True,
            )
            result.append(DenseBlock(matrix))
        blocks.append(tuple(result))
    return BlockTransitionOperator(tuple(blocks), sizes)
