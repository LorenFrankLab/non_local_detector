"""Atomic chunk files and a bounded, lazy xarray reader for decoder outputs.

A completed directory is immutable. Only explicit caller reads materialize arrays;
large reads fail before allocation at ``max_read_bytes``. Readers map one chunk
at a time and close it immediately. OS filesystem page cache is outside this
working-memory bound and must be measured separately in production benchmarks.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing


class IncrementalResultWriter:
    """Write nonoverlapping contiguous chunks, then publish atomically.

    ``variables`` maps names to ``(dimensions, shape, dtype)``. The leading
    dimension must be time. ``coordinates`` maps names to ``(dimensions, values)``.
    Call ``complete`` only after every variable row is written. An exception or
    an uncompleted context removes only the writer's own temporary directory.
    Existing destinations are refused and never overwritten.
    """

    def __init__(self, path, variables, coordinates):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}-", dir=self.path.parent)
        )
        self._published = False
        self._variables = {}
        self._intervals = {}
        self._coordinates = {}
        try:
            for name, (dims, shape, dtype) in variables.items():
                self._check_name(name)
                dims, shape = tuple(dims), tuple(shape)
                if (
                    not dims
                    or dims[0] != "time"
                    or len(dims) != len(shape)
                    or any(s < 0 for s in shape)
                ):
                    raise ValueError(
                        "Variables require time-leading dimensions and nonnegative shapes"
                    )
                self._variables[name] = {
                    "dims": dims,
                    "shape": shape,
                    "dtype": np.dtype(dtype).str,
                    "chunks": [],
                }
                self._intervals[name] = []
            for name, (dims, values) in coordinates.items():
                self._check_name(name)
                values = np.asarray(values)
                if values.dtype.kind == "O":
                    if not all(isinstance(x, str) for x in values.flat):
                        raise ValueError(
                            "Object coordinates must contain strings; encode MultiIndex levels separately"
                        )
                    values = values.astype(str)
                filename = f"coordinate-{name}.npy"
                np.save(self._temporary / filename, values, allow_pickle=False)
                self._coordinates[name] = {"dims": tuple(dims), "file": filename}
        except BaseException:
            self.abort()
            raise

    @staticmethod
    def _check_name(name):
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_]+", name) is None:
            raise ValueError("Store names must use letters, numbers, and underscores")

    def write(self, name, start, values):
        """Write one bounded chunk at its result-row offset, in any chunk order."""
        if self._published:
            raise RuntimeError("Result already completed")
        spec = self._variables[name]
        values = np.asarray(values, dtype=np.dtype(spec["dtype"]))
        stop = start + len(values)
        if (
            values.shape[1:] != tuple(spec["shape"][1:])
            or not 0 <= start <= stop <= spec["shape"][0]
        ):
            raise ValueError(
                "Output chunk shape or row range does not match the variable"
            )
        if start == stop:
            return
        # Sorted written (start, stop) intervals: only neighbours can overlap.
        intervals = self._intervals[name]
        position = bisect.bisect_left(intervals, (start, stop))
        before = intervals[position - 1] if position else None
        after = intervals[position] if position < len(intervals) else None
        if (before is not None and before[1] > start) or (
            after is not None and after[0] < stop
        ):
            raise ValueError("Output chunks overlap")
        filename = f"{name}-{start}-{stop}.npy"
        np.save(self._temporary / filename, values, allow_pickle=False)
        spec["chunks"].append(
            {"start": int(start), "stop": int(stop), "file": filename}
        )
        bisect.insort(intervals, (int(start), int(stop)))

    def complete(self, attrs=None):
        """Validate coverage and atomically expose a completed immutable result."""
        if self._published:
            raise RuntimeError("Result already completed")
        for spec in self._variables.values():
            spec["chunks"].sort(key=lambda chunk: chunk["start"])
            previous = 0
            for chunk in spec["chunks"]:
                if chunk["start"] != previous:
                    raise ValueError("Incomplete output row coverage")
                previous = chunk["stop"]
            if previous != spec["shape"][0]:
                raise ValueError("Incomplete output row coverage")
        manifest = {
            "format_version": 1,
            "complete": True,
            "variables": self._variables,
            "coordinates": self._coordinates,
            "attrs": attrs or {},
        }
        (self._temporary / "manifest.json").write_text(
            json.dumps(manifest, default=_json_value)
        )
        if self.path.exists():
            raise FileExistsError(self.path)
        # Rename within one parent filesystem: readers see either no result or
        # every data/coordinate file plus a complete manifest, never partial data.
        os.rename(self._temporary, self.path)
        self._published = True
        return self.path

    def abort(self):
        """Discard this writer's unpublished files without touching a destination."""
        if not self._published:
            shutil.rmtree(self._temporary, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.abort()


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported result attribute: {type(value).__name__}")


class _ChunkedArray(BackendArray):
    def __init__(self, path, spec, max_read_bytes):
        self.path, self.spec = path, spec
        self.shape, self.dtype = tuple(spec["shape"]), np.dtype(spec["dtype"])
        self.max_read_bytes = max_read_bytes
        # Opening validated contiguous row coverage in start order.
        self._starts = np.array(
            [chunk["start"] for chunk in spec["chunks"]], dtype=np.int64
        )

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _getitem(self, key):
        indices = [
            _index_values(size, item)
            for size, item in zip(self.shape, key, strict=True)
        ]
        read_shape = tuple(len(index) for index in indices)
        size = int(np.prod(read_shape, dtype=np.int64)) * self.dtype.itemsize
        if size > self.max_read_bytes:
            raise MemoryError(
                f"Explicit read needs {size} bytes, exceeding max_read_bytes={self.max_read_bytes}; select fewer rows/bins"
            )
        out = np.empty(read_shape, dtype=self.dtype)
        # Group requested rows by owning chunk without scanning every chunk.
        owner = np.searchsorted(self._starts, indices[0], side="right") - 1
        order = np.argsort(owner, kind="stable")
        touched, first = np.unique(owner[order], return_index=True)
        bounds = np.append(first, len(order))
        for number, lo, hi in zip(touched, bounds[:-1], bounds[1:], strict=True):
            chunk = self.spec["chunks"][number]
            selected = order[lo:hi]
            array = np.load(
                self.path / chunk["file"], mmap_mode="r", allow_pickle=False
            )
            try:
                if (
                    array.shape != (chunk["stop"] - chunk["start"],) + self.shape[1:]
                    or array.dtype != self.dtype
                ):
                    raise ValueError(
                        "Result chunk shape/dtype does not match its manifest"
                    )
                source = [indices[0][selected] - chunk["start"], *indices[1:]]
                destination = [
                    selected,
                    *[np.arange(len(index)) for index in indices[1:]],
                ]
                out[np.ix_(*destination)] = array[np.ix_(*source)]
            finally:
                array._mmap.close()
        scalar_axes = tuple(
            i for i, item in enumerate(key) if isinstance(item, (int, np.integer))
        )
        return np.squeeze(out, axis=scalar_axes) if scalar_axes else out


def _validate_read_budget(max_read_bytes):
    """Normalize an explicit byte limit without accepting booleans or truncation."""
    if (
        isinstance(max_read_bytes, (bool, np.bool_))
        or not isinstance(max_read_bytes, (int, np.integer))
        or max_read_bytes <= 0
    ):
        raise ValueError("max_read_bytes must be a positive integer byte limit")
    return int(max_read_bytes)


def _validate_variable_chunks(path, spec):
    """Check complete row coverage and mapped file headers without reading data."""
    shape, dims = tuple(spec["shape"]), tuple(spec["dims"])
    if (
        not shape
        or len(shape) != len(dims)
        or dims[0] != "time"
        or any(type(size) is not int or size < 0 for size in shape)
    ):
        raise ValueError("Invalid variable shape/dimensions in result manifest")
    dtype = np.dtype(spec["dtype"])
    previous = 0
    for chunk in spec["chunks"]:
        start, stop = chunk["start"], chunk["stop"]
        if (
            type(start) is not int
            or type(stop) is not int
            or start != previous
            or not start < stop <= shape[0]
        ):
            raise ValueError("Incomplete or overlapping result chunk row coverage")
        try:
            array = np.load(path / chunk["file"], mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError("Result chunk file is missing or invalid") from exc
        try:
            if array.shape != (stop - start,) + shape[1:] or array.dtype != dtype:
                raise ValueError("Result chunk shape/dtype does not match its manifest")
        finally:
            array._mmap.close()
        previous = stop
    if previous != shape[0]:
        raise ValueError("Incomplete result chunk row coverage")


def open_result_store(path, *, max_read_bytes=512 * 1024**2):
    """Open a completed chunk store without materializing spatial data.

    Coordinates are recording-sized metadata and loaded eagerly; each spatial
    variable remains a lazy xarray backend. Completion validation briefly maps
    and closes each chunk to inspect its header; no mapping is retained.
    The explicit read budget bounds the
    materialized returned array, not user arrays, filesystem cache, or copies
    made by downstream libraries. This loader does not enable global gradients.
    """
    max_read_bytes = _validate_read_budget(max_read_bytes)
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Result is incomplete or uses an unsupported store version")
    for spec in manifest["variables"].values():
        _validate_variable_chunks(path, spec)
    variables = {
        name: xr.Variable(
            tuple(spec["dims"]),
            indexing.LazilyIndexedArray(_ChunkedArray(path, spec, max_read_bytes)),
        )
        for name, spec in manifest["variables"].items()
    }
    coordinates = {
        name: (tuple(spec["dims"]), np.load(path / spec["file"], allow_pickle=False))
        for name, spec in manifest["coordinates"].items()
    }
    return xr.Dataset(variables, coords=coordinates, attrs=manifest["attrs"])


def _index_values(size, item):
    if isinstance(item, slice):
        return np.arange(*item.indices(size))
    values = np.atleast_1d(np.asarray(item, dtype=np.intp))
    return np.where(values < 0, values + size, values)


class PaddedBackendArray(BackendArray):
    """Lazily scatter selected interior columns into the original padded bins.

    Unsupported columns contain NaN. No eager global posterior or full-file
    mapping is introduced. ``source`` is a two-dimensional xarray variable or
    DataArray whose backend remains lazy until selected rows/bins are read.
    """

    def __init__(self, source, interior_mask, *, max_read_bytes=512 * 1024**2):
        self.source = source.variable if isinstance(source, xr.DataArray) else source
        self.mask = np.asarray(interior_mask, dtype=bool)
        if (
            self.source.ndim != 2
            or self.mask.ndim != 1
            or self.mask.sum() != self.source.shape[1]
        ):
            raise ValueError("Interior mask must match the source spatial columns")
        if np.dtype(self.source.dtype).kind != "f":
            raise ValueError("NaN-padded spatial variables require floating-point data")
        max_read_bytes = _validate_read_budget(max_read_bytes)
        self.shape = (self.source.shape[0], len(self.mask))
        self.dtype = np.dtype(self.source.dtype)
        self.max_read_bytes = max_read_bytes
        self.mapping = np.cumsum(self.mask) - 1

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _getitem(self, key):
        rows, columns = [
            _index_values(size, item)
            for size, item in zip(self.shape, key, strict=True)
        ]
        read_size = len(rows) * len(columns) * self.dtype.itemsize
        if read_size > self.max_read_bytes:
            raise MemoryError(
                f"Explicit padded read needs {read_size} bytes, exceeding max_read_bytes={self.max_read_bytes}; select fewer rows/bins"
            )
        output = np.full((len(rows), len(columns)), np.nan, dtype=self.dtype)
        inside = self.mask[columns]
        if inside.any() and len(rows):
            selected = self.source.isel(
                {
                    self.source.dims[0]: rows,
                    self.source.dims[1]: self.mapping[columns[inside]],
                }
            )
            output[:, inside] = selected.values
        scalar_axes = tuple(
            i for i, item in enumerate(key) if isinstance(item, (int, np.integer))
        )
        return np.squeeze(output, axis=scalar_axes) if scalar_axes else output


def pad_state_bins(variable, interior_mask, *, max_read_bytes=512 * 1024**2):
    """Return a lazy padded variable; caller supplies original bin coordinates."""
    source = variable.variable if isinstance(variable, xr.DataArray) else variable
    backend = PaddedBackendArray(source, interior_mask, max_read_bytes=max_read_bytes)
    return xr.Variable(
        source.dims, indexing.LazilyIndexedArray(backend), attrs=source.attrs
    )
