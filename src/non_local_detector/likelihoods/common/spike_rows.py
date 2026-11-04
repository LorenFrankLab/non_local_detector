"""Spikes of a decode-row range: selection, per-row sums and counts."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from tqdm.autonotebook import tqdm

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common.padding import (
    pad_rows,
    padded_sample_count,
)
from non_local_detector.likelihoods.common.reductions import deterministic_row_sum
from non_local_detector.time_edges import (
    _DecodeTimeGrid,
    _resolve_time_grid,
    requires_time_edges,
    validate_time_edges,
)

# JAX exposes this exception only privately. Older supported releases predate
# explicit sharding and may lack the type; an empty tuple catches nothing.
try:
    from jax._src.core import ShardingTypeError as _JaxShardingTypeError
except ImportError:
    _JAX_SHARDING_ERRORS: tuple[type[Exception], ...] = ()
else:
    _JAX_SHARDING_ERRORS = (_JaxShardingTypeError,)


_T = TypeVar("_T")


def resolve_row_slice(row_slice: slice | None, n_bins: int) -> tuple[int, int]:
    """Normalize a likelihood row request against the full decoding timeline.

    A backend's ``row_slice`` selects which rows of the FULL-time likelihood to
    return; ``time_edges`` always stay the full decoding edges so that
    spike-to-row ownership is independent of how the rows were chunked.

    Parameters
    ----------
    row_slice : slice | None
        Contiguous (unit-step) range of output rows, or None for all rows.
    n_bins : int
        Number of decode bins, ``len(time_edges) - 1``.

    Returns
    -------
    row_start : int
        First requested row (inclusive).
    row_stop : int
        End of the requested rows (exclusive), with
        ``0 <= row_start <= row_stop <= n_bins``.
    """
    if row_slice is None:
        return 0, n_bins
    if row_slice.step not in (None, 1):
        msg = "row_slice must select a contiguous range of rows"
        raise ValidationError(
            msg,
            expected="a slice with step None or 1",
            got=f"step={row_slice.step}",
            hint="Chunked prediction only requests contiguous row ranges.",
            example="    predict_..._log_likelihood(time_edges, ..., row_slice=slice(0, 100))",
        )
    row_start, row_stop, _ = row_slice.indices(n_bins)
    return row_start, max(row_start, row_stop)


def _spikes_are_ascending(spike_times: np.ndarray) -> bool:
    """Verify non-decreasing times without sorting or changing precision."""
    return spike_times.size < 2 or bool(np.all(spike_times[1:] >= spike_times[:-1]))


_KDE_PREPARATION_BYTES = 32 * 1024**2


_KDE_PREPARATION_ENTRIES = 256


class SpikeTimeOrder:
    """Host spike times, verified ordering and other preparation for one prediction.

    Each original input is converted and checked on first use, then reused by
    every observation state and chunk. Strong references prevent identity reuse.
    Inputs must remain unchanged during the prediction; a fresh instance on the
    next call rechecks even arrays modified in place. Never store this object on
    a detector or pass it into a JIT kernel.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[object, np.ndarray, bool]] = {}
        self._prepared: dict[tuple[Any, ...], tuple[tuple[Any, ...], Any]] = {}
        self._kde_prepared: OrderedDict[
            tuple[Any, ...], tuple[tuple[Any, ...], Any, int]
        ] = OrderedDict()
        self._kde_prepared_bytes = 0

    def memo(self, name: str, sources: tuple[Any, ...], build: Callable[[], _T]) -> _T:
        """Return ``build()``, computed once per ``name`` and ``sources`` identities.

        ``sources`` are kept alive so their ids cannot be reused during the
        prediction.
        """
        key = (name, *map(id, sources))
        if key not in self._prepared:
            self._prepared[key] = (sources, build())
        return cast(_T, self._prepared[key][1])

    def memo_kde(
        self, name: str, sources: tuple[Any, ...], build: Callable[[], _T]
    ) -> _T:
        """Prediction-scoped preparation, bounding values and retained source arrays.

        Counting source arrays conservatively includes already resident inputs.
        It also bounds old fitted leaves retained after a model is refitted.
        """
        key = (name, *map(id, sources))
        if key in self._kde_prepared:
            self._kde_prepared.move_to_end(key)
            return cast(_T, self._kde_prepared[key][1])
        value = build()
        size = sum(
            int(getattr(leaf, "nbytes", 0))
            for leaf in jax.tree_util.tree_leaves((sources, value))
        )
        if size <= _KDE_PREPARATION_BYTES:
            while self._kde_prepared and (
                self._kde_prepared_bytes + size > _KDE_PREPARATION_BYTES
                or len(self._kde_prepared) >= _KDE_PREPARATION_ENTRIES
            ):
                _, (_, _, old_size) = self._kde_prepared.popitem(last=False)
                self._kde_prepared_bytes -= old_size
            self._kde_prepared[key] = (sources, value, size)
            self._kde_prepared_bytes += size
        return value

    def get(self, spike_times: ArrayLike) -> tuple[np.ndarray, bool]:
        """Return the original input's host values and established ordering."""
        key = id(spike_times)
        if key not in self._entries:
            values = np.asarray(spike_times)
            self._entries[key] = (spike_times, values, _spikes_are_ascending(values))
        _, values, ascending = self._entries[key]
        return values, ascending


@dataclass(frozen=True)
class DecodeRows:
    """The decode grid, requested rows and spike preparation of one predict call.

    Attributes
    ----------
    time_grid : _DecodeTimeGrid
        The FULL decoding grid.
    spike_time_order : SpikeTimeOrder
        The prediction's spike-time preparation, shared across states and chunks.
    start, stop : int
        The requested rows ``[start, stop)`` of the full grid.
    """

    time_grid: _DecodeTimeGrid
    spike_time_order: SpikeTimeOrder
    start: int
    stop: int

    @property
    def time_edges(self) -> np.ndarray:
        """The FULL decoding bin edges, shape (n_bins + 1,), in seconds."""
        return self.time_grid.edges

    def durations(self) -> np.ndarray:
        """Durations of the requested rows, shape (stop - start,), in seconds."""
        return self.time_grid.durations(self.start, self.stop)


def prepare_decode_rows(
    time_edges: ArrayLike,
    row_slice: slice | None,
    time_grid: _DecodeTimeGrid | None = None,
    spike_time_order: SpikeTimeOrder | None = None,
) -> DecodeRows:
    """Resolve a predict call's decode grid, row range and spike preparation.

    Parameters
    ----------
    time_edges : array_like, shape (n_bins + 1,)
        FULL decoding bin edges, in seconds.
    row_slice : slice | None
        Contiguous range of decode bins to evaluate, or None for all bins.
    time_grid : _DecodeTimeGrid, optional
        A detector's prepared grid for these edges, reused when it matches;
        otherwise the edges are validated here.
    spike_time_order : SpikeTimeOrder, optional
        A detector's spike preparation shared across states and chunks; None
        starts a fresh one, which verifies spike ordering on this call.

    Returns
    -------
    rows : DecodeRows
    """
    time_grid = _resolve_time_grid(time_edges, time_grid)
    if spike_time_order is None:
        spike_time_order = SpikeTimeOrder()
    start, stop = resolve_row_slice(row_slice, time_grid.edges.shape[0] - 1)
    return DecodeRows(time_grid, spike_time_order, start, stop)


@dataclass(frozen=True)
class SpikeSelection:
    """Host metadata for spikes belonging to one likelihood row range.

    ``n_spikes`` is the original train length, used to validate paired arrays
    before slicing. ``n_rows`` is the requested row count, the segment count
    that pairs with the LOCAL rows in ``bin_ind``; ``sum_spikes_into_rows``
    reads it from here so a reduction cannot be paired with another request's
    row count. ``indices_are_sorted`` records the verified ordering of
    ``bin_ind`` independently of how ``indexer`` represents the selection.
    This object stays outside JIT kernels.
    """

    indexer: slice | np.ndarray
    bin_ind: np.ndarray
    indices_are_sorted: bool
    n_spikes: int
    n_rows: int


@requires_time_edges
def select_spikes_in_rows(
    spike_times: np.ndarray | jax.Array,
    row_start: int,
    row_stop: int,
    *,
    time_edges: np.ndarray,
    _spike_time_order: SpikeTimeOrder | None = None,
) -> SpikeSelection:
    """Select the spikes owned by the decode bins ``[row_start, row_stop)``.

    Bin ``i`` owns ``time_edges[i] <= t < time_edges[i + 1]``; the final bin
    also owns ``t == time_edges[-1]``. Spikes outside
    ``[time_edges[0], time_edges[-1]]`` belong to no bin.

    Owning a bin ``>= row_start`` is exactly ``t >= time_edges[row_start]`` and
    owning a bin ``< row_stop`` is exactly ``t < time_edges[row_stop]``
    (inclusive when ``row_stop == n_bins``, i.e. the range reaches the final
    bin), so the selection is a
    contiguous range in time, found with ``np.searchsorted`` when the spike
    times are ascending. Only the selected subset is binned, so no spike is
    re-binned for every chunk.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Decoding spike times for one neuron/electrode.
    row_start : int
        First requested bin (inclusive), as returned by ``resolve_row_slice``.
    row_stop : int
        End of the requested bins (exclusive), as returned by ``resolve_row_slice``.
    time_edges : np.ndarray, shape (n_bins + 1,)
        FULL decoding bin edges (not the chunk's).
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal prediction-local preparation shared across states and chunks.
        Direct callers omit it and verify ordering on each call.

    Returns
    -------
    selection : SpikeSelection
        ``indexer`` selects owned spikes in their original order and is applied
        identically to waveform features by ``select_spike_rows``. ``bin_ind``
        contains LOCAL row indices (``global_row - row_start``) and ``n_rows``
        their segment count, ``row_stop - row_start``. ``indices_are_sorted`` is the
        verified JAX reduction hint; callers must use it rather than infer
        ordering from the indexer's type. ``n_spikes`` records the original
        length so paired arrays can be validated even for empty requests.
    """
    time_edges = (
        validate_time_edges(time_edges)
        if _spike_time_order is None
        else np.asarray(time_edges)
    )
    n_bins = time_edges.shape[0] - 1
    n_spikes = len(spike_times)
    n_rows = row_stop - row_start
    if n_spikes == 0 or n_rows <= 0:
        return SpikeSelection(
            slice(0, 0), np.zeros((0,), dtype=int), True, n_spikes, n_rows
        )

    # Direct callers get a fresh preparation: converted and checked once here.
    if _spike_time_order is None:
        _spike_time_order = SpikeTimeOrder()
    spike_times, is_ascending = _spike_time_order.get(spike_times)

    lower = time_edges[row_start]
    upper = time_edges[row_stop]
    # The final bin is closed so a spike at the last edge is counted.
    upper_is_inclusive = row_stop == n_bins

    # The ordering the range lookup needs was established, never assumed.
    if is_ascending:
        start = int(np.searchsorted(spike_times, lower, side="left"))
        stop = int(
            np.searchsorted(
                spike_times, upper, side="right" if upper_is_inclusive else "left"
            )
        )
        indexer: slice | np.ndarray = slice(start, max(start, stop))
        selected = spike_times[indexer]
    else:
        in_rows = spike_times >= lower
        in_rows &= spike_times <= upper if upper_is_inclusive else spike_times < upper
        indexer = in_rows
        selected = spike_times[in_rows]

    # Selection above establishes global ownership. Only the interior edges of
    # the requested bins are needed for the local bin index; the full edge
    # array would make digitize's monotonicity check scan the whole recording
    # for every unit and every chunk. A spike at the closing edge of the final
    # bin lands past the last interior edge, in the final local bin.
    interior_edges = time_edges[row_start + 1 : row_stop]
    return SpikeSelection(
        indexer, np.digitize(selected, interior_edges), is_ascending, n_spikes, n_rows
    )


def sum_spikes_into_rows(
    values: np.ndarray | jax.Array, selection: SpikeSelection
) -> jax.Array:
    """Sum one value per selected spike into the local row that owns it.

    Parameters
    ----------
    values : np.ndarray or jax.Array, shape (n_selected, ...)
        One entry per spike in ``selection``, in selection order. Host values
        are padded to a few sizes first, so chunks with nearby spike counts
        share one compiled reduction.
    selection : SpikeSelection
        The selection from ``select_spikes_in_rows``; supplies the local row of
        each spike, the row count, and the verified sorted-index hint.

    Returns
    -------
    row_sums : jax.Array, shape (selection.n_rows, ...)
        Rows owning no selected spike sum to zero.

    Notes
    -----
    Uses ``deterministic_row_sum``: CPU keeps the sequential segment reduction
    and its verified ordering hint; other backends use a segmented scan with a
    fixed reduction tree, so repeated evaluations are bitwise identical.
    Invalid row indices are dropped.
    """
    on_host = isinstance(values, np.ndarray)
    values = values if on_host else jnp.asarray(values)
    if values.dtype == np.bool_:
        msg = "Spike row sums require numeric values; Boolean sums are unsupported"
        raise TypeError(msg)
    indices = np.asarray(selection.bin_ind)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        msg = "Spike row indices must be a one-dimensional integer array"
        raise ValueError(msg)
    if values.ndim == 0 or values.shape[0] != len(indices):
        msg = "Values must contain one entry per selected spike"
        raise ValueError(msg)
    # Padding rows are dropped from the sum.
    n_padded = padded_sample_count(len(indices)) if on_host else None
    return deterministic_row_sum(
        jnp.asarray(pad_rows(values, cast(int, n_padded))) if on_host else values,
        spike_row_ids(selection, n_padded),
        selection.n_rows,
        indices_are_sorted=selection.indices_are_sorted,
    )


def _normalized_row_ids(
    indices: np.ndarray, n_rows: int, n_padded: int | None = None
) -> np.ndarray:
    """Host row ids for ``deterministic_row_sum``, optionally padded.

    Parameters
    ----------
    indices : array-like, shape (n_selected,), integer
    n_rows : int
    n_padded : int, optional
        Length to pad to. Padded entries use the dropped id ``n_rows``, which
        keeps sorted ids sorted.

    Returns
    -------
    row_ids : np.ndarray of intp, shape (n_padded or n_selected,)
        Negative ids become -1 and ids at or above ``n_rows`` become ``n_rows``
        before JAX dtype canonicalization, so a huge unsigned or out-of-range
        index cannot truncate into a valid row.
    """
    indices = np.asarray(indices)
    n_ids = len(indices) if n_padded is None else n_padded
    safe_indices = np.full(n_ids, n_rows, dtype=np.intp)
    valid = (indices >= 0) & (indices < n_rows)
    safe_indices[: len(indices)][indices < 0] = -1
    safe_indices[: len(indices)][valid] = indices[valid]
    return safe_indices


def spike_row_ids(selection: SpikeSelection, n_padded: int | None = None) -> np.ndarray:
    """Normalize the row ids of a selection's spikes for ``deterministic_row_sum``.

    Parameters
    ----------
    selection : SpikeSelection
        The selection from ``select_spikes_in_rows``; supplies the local row of
        each spike and the row count.
    n_padded : int, optional
        Length to pad to, by default None (no padding). Padded entries use the
        dropped id ``selection.n_rows``, which keeps sorted ids sorted.

    Returns
    -------
    row_ids : np.ndarray of intp, shape (n_padded or n_selected,)
        Local row id of each selected spike, with invalid ids mapped to dropped
        ids (see ``deterministic_row_sum``).
    """
    return _normalized_row_ids(selection.bin_ind, selection.n_rows, n_padded)


def select_spike_rows(
    array: "np.ndarray | jax.Array", selection: SpikeSelection
) -> np.ndarray:
    """Take the rows an indexer names without materializing the whole array.

    ``select_spikes_in_rows`` returns one selection per unit/electrode, applied to
    the spike times and to the per-spike waveform features. Those arrays are
    recording-length; the selection is one chunk's worth. Converting the array
    before selecting therefore pays for the whole recording on every chunk call.
    For a ``jax.Array`` it is worse than a copy: ``np.asarray`` gathers the array
    to the host **and caches that host copy on the array object**
    (``jax.Array._npy_value``), so the memory is retained for as long as the
    caller holds its inputs.

    Indexing on device avoids both. It is not always available: under a mesh with
    ``AxisType.Explicit`` axes JAX refuses a gather whose output sharding it
    cannot infer, so that case falls back to gather-then-slice — the behaviour
    every backend had before, correct but paying the host copy.

    Parameters
    ----------
    array : np.ndarray or jax.Array, shape (n_spikes, ...)
        Per-spike array for one unit/electrode.
    selection : SpikeSelection
        The selection from ``select_spikes_in_rows``, including the original
        spike count. Every paired array must have exactly that many rows,
        whether the requested selection uses a slice or a mask.

    Returns
    -------
    selected : np.ndarray, shape (n_selected, ...)
        The selected rows, always on the host. A selection that kept a sharded
        input's sharding would carry it into the jitted kernels downstream, which
        also take single-device encoding-model arrays -- JAX rejects that mix.
        The selection is request-sized, so this costs the chunk, not the
        recording; consumers that want a device array convert it themselves.

    Raises
    ------
    ValidationError
        If the array's row count differs from the original spike count. Checking
        before selection catches both extra and missing feature rows, including
        mismatches outside the requested chunk, without reading array values.
    """
    if array.shape[0] != selection.n_spikes:
        msg = "spike selection does not match the length of a per-spike array"
        raise ValidationError(
            msg,
            expected=f"{selection.n_spikes} rows (one per spike time)",
            got=f"{array.shape[0]} rows",
            hint="Spike times and waveform features must be paired row for row.",
        )

    indexer = selection.indexer
    if not isinstance(array, jax.Array):
        return array[indexer]

    try:
        if isinstance(indexer, slice):
            selected = array[indexer]
        else:
            # A boolean mask becomes a gather of explicit integer positions,
            # which is the form JAX can shard; the mask itself is a host array,
            # so its positions are known here. The length check above is what
            # keeps ``take``'s default ``mode='fill'`` from turning a mismatch
            # into NaN rows.
            selected = jnp.take(array, jnp.asarray(np.flatnonzero(indexer)), axis=0)
    except _JAX_SHARDING_ERRORS:
        # Explicitly sharded indexing may have ambiguous output sharding.
        # Preserve its host fallback, but let unrelated indexing, device and
        # memory errors propagate without copying the full recording.
        return np.asarray(array)[indexer]

    return np.asarray(selected)


def select_decode_marks(
    spike_waveform_features: "np.ndarray | jax.Array",
    selection: SpikeSelection,
    electrode: int,
) -> np.ndarray:
    """Select one electrode's decode-spike marks and require them to be finite.

    Every clusterless likelihood (and the No-Spike mark density) reads the
    waveform features of the spikes in the requested bins through this
    function. One non-finite mark gives its bin a NaN likelihood, and filtering
    carries NaN into the posterior of every later bin, so it is rejected here.
    Only the selected spikes are checked: a spike outside the requested bins
    never enters the likelihood, matching the encoding fits, which check only
    the spikes inside the encoding support.

    Parameters
    ----------
    spike_waveform_features : np.ndarray or jax.Array, shape (n_spikes, n_features)
        One electrode's waveform features, row-paired with its spike times.
    selection : SpikeSelection
        The selection from ``select_spikes_in_rows`` for the same electrode.
    electrode : int
        Electrode index, used in the error message.

    Returns
    -------
    marks : np.ndarray, shape (n_selected, n_features)
        The selected rows, on the host.

    Raises
    ------
    ValidationError
        If any selected mark is NaN or infinite, or if the feature rows do not
        match the spike count (see ``select_spike_rows``).
    """
    marks = select_spike_rows(spike_waveform_features, selection)
    is_finite = np.isfinite(marks)
    if not is_finite.all():
        n_bad = int(np.count_nonzero(~is_finite.all(axis=1)))
        msg = f"non-finite waveform features for electrode {electrode}"
        raise ValidationError(
            msg,
            expected="finite (non-NaN, non-inf) waveform features for every "
            "spike in the decoding bins",
            got=f"{n_bad} of {marks.shape[0]} spikes with NaN or inf features",
            hint="Drop these spikes from both spike_times and "
            "spike_waveform_features, or replace their features, before "
            "predicting. A non-finite mark makes the likelihood of its bin NaN, "
            "and the posterior of every later bin with it.",
        )
    return marks


@requires_time_edges
def get_spikecount_per_time_bin(
    spike_times: np.ndarray | jax.Array,
    row_slice: slice | None = None,
    *,
    time_edges: np.ndarray,
    _spike_time_order: SpikeTimeOrder | None = None,
) -> np.ndarray:
    """Get the number of spikes in each requested decode bin.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Decoding spike times, in seconds.
    row_slice : slice | None, optional
        Contiguous range of bins to count, by default None (all bins).
        Counting bins ``[a, b)`` of the full timeline gives the same values as
        counting all bins and slicing ``[a:b]``, so concatenating the chunks of
        a partition reproduces the full-time counts exactly.
    time_edges : np.ndarray, shape (n_bins + 1,)
        FULL decoding bin edges, which define the bin a spike belongs to (see
        ``select_spikes_in_rows``). Direct calls validate these edges. Internal
        calls with prepared spike ordering reuse the detector's validated grid.
    _spike_time_order : SpikeTimeOrder | None, optional
        Internal ordering preparation that a detector prediction shares across
        observation states and chunks. Direct callers omit it; the spike-time
        ordering is then verified on this call.

    Returns
    -------
    count : np.ndarray, shape (n_rows,)
        ``n_rows`` is ``n_bins`` by default, else the length of ``row_slice``.
    """
    time_edges = (
        validate_time_edges(time_edges)
        if _spike_time_order is None
        else np.asarray(time_edges)
    )
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    selection = select_spikes_in_rows(
        spike_times,
        row_start,
        row_stop,
        time_edges=time_edges,
        _spike_time_order=_spike_time_order,
    )
    return np.bincount(selection.bin_ind, minlength=selection.n_rows)


def spike_counts_matrix(
    spike_times: list[np.ndarray],
    time_edges: np.ndarray,
    desc: str,
    disable_progress_bar: bool,
    row_slice: slice | None = None,
    *,
    _spike_time_order: SpikeTimeOrder | None = None,
) -> np.ndarray:
    """Per-neuron spike counts as a ``(n_rows, n_neurons)`` matrix.

    Spikes are binned against the full ``time_edges`` and selected by
    ``row_slice`` as in ``get_spikecount_per_time_bin``, so the values equal its
    per-neuron counts. One ``bincount`` over ``row * n_neurons + neuron`` fills
    the row-major matrix in a single pass instead of stacking columns. ``n_rows``
    is ``n_bins`` unless ``row_slice`` is given.
    """
    if _spike_time_order is None:
        time_edges = validate_time_edges(time_edges)
    row_start, row_stop = resolve_row_slice(row_slice, time_edges.shape[0] - 1)
    n_rows, n_neurons = row_stop - row_start, len(spike_times)
    if not n_neurons:
        return np.zeros((n_rows, 0))
    flat_index = [
        select_spikes_in_rows(
            neuron_spike_times,
            row_start,
            row_stop,
            time_edges=time_edges,
            _spike_time_order=_spike_time_order,
        ).bin_ind.astype(np.int64)
        * n_neurons
        + neuron
        for neuron, neuron_spike_times in enumerate(
            tqdm(spike_times, unit="cell", desc=desc, disable=disable_progress_bar)
        )
    ]
    return np.bincount(
        np.concatenate(flat_index), minlength=n_rows * n_neurons
    ).reshape(n_rows, n_neurons)
