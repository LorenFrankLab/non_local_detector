"""Values to JSON documents plus ``.npy`` arrays, and back.

A value is written as JSON when it is a JSON scalar, list or string-keyed
dict, and as a tagged object otherwise. Every tag is the single ``$`` key of a
JSON object:

``{"$array": member, "ref": n, "jax": bool}``
    An array, stored as the ``.npy`` member ``member``. Values that were the
    same object when saved share ``ref`` and load as one object; equal arrays
    that were different objects share the member but not the object.
``{"$fill": value, "dtype": str, "shape": [...], "ref": n, "jax": bool}``
    A floating array whose every element is 1 (sample weights, by default).
    No member holds it, so the archive's member-size checks do not bound
    its size.
``{"$ragged": {"data": node, "offsets": node}, "ref": n, "jax": bool}``
    A list of arrays with one dtype and trailing shape (one per neuron or
    electrode), concatenated along the first axis; item ``i`` is
    ``data[offsets[i]:offsets[i + 1]]``.
``{"$item": n, "index": i}``
    Item ``i`` of the ragged list ``ref`` ``n``: the same object.
``{"$tuple": [...]}``, ``{"$dict": [[key, value], ...]}``
    A tuple; a dict whose keys are not all strings (or start with ``$``).
``{"$float": "nan" | "inf" | "-inf"}``, ``{"$scalar": value, "dtype": str}``
    A non-finite float; a NumPy scalar.
``{"$type": name, "owner": package, "state": {...}}``
    An object of a registered type (see ``register_type``), and the package
    that registered it, named if the type is not registered when loading.
``{"$environment": name}``
    The model's fitted environment of that name.
``{"$graph": {...}}``, ``{"$dataframe": {...}}``, ``{"$series": {...}}``,
``{"$categorical": {...}}``
    A networkx graph; a pandas DataFrame or Series, whose columns or values
    are arrays or categoricals; a pandas Categorical, as integer codes, its categories and
    whether they are ordered (patsy takes a categorical's reference level
    from the category order).

An object array of strings is stored as a fixed-width unicode array with
``"object": true``; any other object array is refused.

Nothing in a file names a module or a class except through ``$type`` names,
which only the registry resolves.
"""

import itertools
import math
import re
import zipfile
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd

from non_local_detector.environment import Environment
from non_local_detector.exceptions import (
    DataError,
    NonLocalDetectorError,
    ValidationError,
)
from non_local_detector.model_format._container import (
    array_digest,
    write_array_member,
)
from non_local_detector.model_format._registry import (
    spec_for_name,
    spec_for_object,
    unregistered_type_error,
)

JSON = Any

# Floating arrays of ones of at least this many elements are stored as a flag.
_MIN_FILL_SIZE = 64

# Dimension names recorded in the manifest for arrays with these names.
_DIMS = {
    "place_fields": ("neuron", "position_bin"),
    "interior_log_place_fields": ("neuron", "position_bin"),
    "place_bin_centers": ("position_bin", "position_dim"),
    "occupancy": ("position_bin",),
    "log_occupancy": ("position_bin",),
    "no_spike_part_log_likelihood": ("position_bin",),
    "summed_ground_process_intensity": ("position_bin",),
    "initial_conditions": ("state_bin",),
    "continuous_initial_conditions": ("state_bin",),
    "discrete_initial_conditions": ("state",),
    "log_likelihood": ("time", "state_bin"),
}


def _member_name(where: str) -> str:
    """Return the ``.npy`` member path for an array found at ``where``."""
    parts = [re.sub(r"[^A-Za-z0-9_.-]", "_", part) or "_" for part in where.split("/")]
    return "arrays/" + "/".join(parts)


def refuse_object_array(array: np.ndarray, where: str) -> None:
    """Raise if a saved model cannot hold ``array``.

    Parameters
    ----------
    array : np.ndarray
        The array to store.
    where : str
        Where it was found, for the message.

    Raises
    ------
    ValidationError
        If the array has an object dtype, which ``.npy`` stores only through
        pickle.
    """
    if array.dtype.hasobject:
        msg = f"Cannot save the object array at {where}"
        raise ValidationError(
            msg,
            expected="a numeric, boolean or string array",
            got=f"dtype {array.dtype}",
        )


class ArrayWriter:
    """Writes a saved model's ``.npy`` members and keeps their manifest index.

    Each array is streamed into its member of ``archive`` as it is encoded,
    so saving holds no second copy of it. Arrays with the same dtype, shape
    and bytes are stored once.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open for writing.
    """

    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.index: dict[str, dict[str, Any]] = {}
        self._by_digest: dict[str, str] = {}

    def put(self, array: np.ndarray, where: str) -> str:
        """Store ``array`` and return its member path.

        Raises
        ------
        ValidationError
            If the array has an object dtype, which ``.npy`` stores only
            through pickle.
        """
        refuse_object_array(array, where)
        # np.require keeps a 0-d array 0-d (np.ascontiguousarray makes it 1-d).
        array = np.require(array, requirements="C")
        digest = array_digest(array)
        if digest in self._by_digest:
            return self._by_digest[digest]
        base = _member_name(where)
        path, suffix = f"{base}.npy", 2
        while path in self.index:
            path, suffix = f"{base}~{suffix}.npy", suffix + 1
        write_array_member(self.archive, path, array)
        name = where.rsplit("/", 1)[-1]
        dims = _DIMS.get(name)
        self.index[path] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "dims": list(dims)
            if dims is not None and len(dims) == array.ndim
            else None,
            "sha256": digest,
        }
        self._by_digest[digest] = path
        return path


def _encode_float(value: float) -> JSON:
    if math.isfinite(value):
        return value
    return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}


def _decode_float(node: JSON) -> float:
    return float(node["$float"])


def _is_array(value: Any) -> bool:
    return isinstance(value, (np.ndarray, jax.Array))


class Encoder:
    """Turns values into JSON nodes, handing arrays to an ``ArrayWriter``.

    Parameters
    ----------
    writer : ArrayWriter
    environments : Mapping[int, str], optional
        ``id`` of each fitted environment to its name. Given, environments are
        written as ``$environment`` references and any other environment is
        refused; by default None, which writes environments whole (their
        configuration).
    """

    def __init__(
        self, writer: ArrayWriter, environments: Mapping[int, str] | None = None
    ) -> None:
        self.writer = writer
        self.environments = None if environments is None else dict(environments)
        self._refs: dict[int, JSON] = {}
        # Keep every encoded array alive, so no id is reused while encoding.
        self._alive: list[Any] = []
        self._next_ref = 0

    def _new_ref(self, value: Any) -> int:
        ref = self._next_ref
        self._next_ref += 1
        self._alive.append(value)
        return ref

    def encode(self, value: Any, where: str) -> JSON:
        """Return the JSON node of ``value``, found at ``where`` (used to name arrays)."""
        if value is None or isinstance(value, (bool, str)):
            return value
        # NumPy scalars first: np.float64 is a float.
        if isinstance(value, np.generic):
            return self._scalar(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return _encode_float(value)
        if _is_array(value):
            return self._array(value, where)
        if isinstance(value, tuple):
            return {
                "$tuple": [self.encode(v, f"{where}/{i}") for i, v in enumerate(value)]
            }
        if isinstance(value, list):
            return self._list(value, where)
        if isinstance(value, Mapping) and not isinstance(value, pd.DataFrame):
            return self._mapping(value, where)
        if isinstance(value, Environment) and self.environments is not None:
            if id(value) not in self.environments:
                msg = (
                    f"The environment at {where} is not one of the fitted environments"
                )
                raise ValidationError(
                    msg,
                    hint="Encoding models must refer to the detector's environments_.",
                )
            return {"$environment": self.environments[id(value)]}
        if isinstance(value, nx.Graph):
            return self._graph(value, where)
        if isinstance(value, pd.DataFrame):
            return self._dataframe(value, where)
        if isinstance(value, pd.Series):
            return self._series(value, where)
        if isinstance(value, pd.Categorical):
            return self._categorical(value, where)
        return self.encode_object(value, where)

    def encode_object(self, value: Any, where: str) -> JSON:
        """Return the ``$type`` node of ``value``, an object of a registered type.

        Raises
        ------
        ValidationError
            If its type is not registered.
        """
        spec = spec_for_object(value)
        if spec is None or spec.to_state is None:
            raise unregistered_type_error(type(value), where)
        state = spec.to_state(value)
        return {
            "$type": spec.name,
            "owner": spec.owner,
            "state": self._mapping(dict(state), where, plain_keys_only=True),
        }

    def _scalar(self, value: np.generic) -> JSON:
        if value.dtype.kind not in "biuf":
            msg = f"Cannot save the NumPy scalar {value!r}"
            raise ValidationError(msg, expected="a boolean, integer or float scalar")
        item = value.item()
        encoded = _encode_float(item) if isinstance(item, float) else item
        return {"$scalar": encoded, "dtype": value.dtype.str}

    def _array(self, value: Any, where: str) -> JSON:
        if id(value) in self._refs:
            return self._refs[id(value)]
        array = np.asarray(value)
        is_jax = isinstance(value, jax.Array)
        ref = self._new_ref(value)
        node: JSON
        if (
            array.dtype.kind == "f"
            and array.size >= _MIN_FILL_SIZE
            and bool(np.all(array == 1))
        ):
            node = {
                "$fill": 1.0,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "ref": ref,
                "jax": is_jax,
            }
        elif (
            array.dtype.hasobject
            and array.size
            and all(isinstance(item, str) for item in array.flat)
        ):
            node = {
                "$array": self.writer.put(array.astype(str), where),
                "ref": ref,
                "jax": False,
                "object": True,
            }
        else:
            node = {"$array": self.writer.put(array, where), "ref": ref, "jax": is_jax}
        self._refs[id(value)] = node
        return node

    def _list(self, value: list[Any], where: str) -> JSON:
        if (
            len(value) > 1
            and all(_is_array(v) and v.ndim >= 1 for v in value)
            and len(
                {
                    (v.shape[1:], np.dtype(v.dtype), isinstance(v, jax.Array))
                    for v in value
                }
            )
            == 1
            and len({id(v) for v in value}) == len(value)
            and not any(id(v) in self._refs for v in value)
        ):
            arrays = [np.asarray(v) for v in value]
            offsets = np.cumsum([0, *(len(a) for a in arrays)]).astype(np.int64)
            ref = self._new_ref(value)
            node = {
                "$ragged": {
                    "data": self._array(np.concatenate(arrays), f"{where}/data"),
                    "offsets": self._array(offsets, f"{where}/offsets"),
                },
                "ref": ref,
                "jax": isinstance(value[0], jax.Array),
            }
            for index, item in enumerate(value):
                self._alive.append(item)
                self._refs[id(item)] = {"$item": ref, "index": index}
            return node
        return [self.encode(v, f"{where}/{i}") for i, v in enumerate(value)]

    def _mapping(
        self, value: Mapping[Any, Any], where: str, *, plain_keys_only: bool = False
    ) -> JSON:
        if all(isinstance(k, str) and not k.startswith("$") for k in value):
            return {k: self.encode(v, f"{where}/{k}") for k, v in value.items()}
        if plain_keys_only:
            msg = f"The state saved at {where} must have string keys not starting with '$'"
            raise ValidationError(msg, got=f"keys {list(value)!r}")
        return {
            "$dict": [
                [self.encode(k, f"{where}/key{i}"), self.encode(v, f"{where}/{i}")]
                for i, (k, v) in enumerate(value.items())
            ]
        }

    def _graph(self, graph: nx.Graph, where: str) -> JSON:
        if graph.is_multigraph():
            msg = f"Cannot save the multigraph at {where}"
            raise ValidationError(msg, expected="a networkx Graph or DiGraph")
        return {
            "$graph": {
                "directed": graph.is_directed(),
                "attributes": self.encode(dict(graph.graph), f"{where}/attributes"),
                "nodes": [
                    [
                        self.encode(node, f"{where}/node"),
                        self.encode(dict(attrs), f"{where}/node"),
                    ]
                    for node, attrs in graph.nodes(data=True)
                ],
                "edges": [
                    [
                        self.encode(u, f"{where}/edge"),
                        self.encode(v, f"{where}/edge"),
                        self.encode(dict(attrs), f"{where}/edge"),
                    ]
                    for u, v, attrs in graph.edges(data=True)
                ],
            }
        }

    def _categorical(self, values: pd.Categorical, where: str) -> JSON:
        return {
            "$categorical": {
                "codes": self._array(np.asarray(values.codes), f"{where}/codes"),
                "categories": self._array(
                    values.categories.to_numpy(), f"{where}/categories"
                ),
                "ordered": bool(values.ordered),
            }
        }

    def _index(self, index: pd.Index, where: str) -> JSON:
        if isinstance(index, pd.RangeIndex):
            return {"range": [index.start, index.stop, index.step]}
        return {"values": self._array(index.to_numpy(), f"{where}/index")}

    def _column(self, column: pd.Series, where: str) -> JSON:
        """Return the values of a Series or DataFrame column.

        These are an array, or a categorical keeping its categories and their
        order.
        """
        if isinstance(column.dtype, pd.CategoricalDtype):
            return self._categorical(column.array, where)
        return self._array(column.to_numpy(), where)

    def _dataframe(self, frame: pd.DataFrame, where: str) -> JSON:
        return {
            "$dataframe": {
                "columns": [
                    self.encode(name, f"{where}/columns") for name in frame.columns
                ],
                "values": [
                    self._column(frame[name], f"{where}/{i}")
                    for i, name in enumerate(frame.columns)
                ],
                "index": self._index(frame.index, where),
            }
        }

    def _series(self, series: pd.Series, where: str) -> JSON:
        return {
            "$series": {
                "name": self.encode(series.name, f"{where}/name"),
                "values": self._column(series, where),
                "index": self._index(series.index, where),
            }
        }


class Decoder:
    """Turns JSON nodes back into values.

    Parameters
    ----------
    read_array : callable
        ``read_array(member) -> np.ndarray``.
    environments : Mapping[str, Environment], optional
        The fitted environments ``$environment`` references resolve to.
    plain : bool, optional
        Build no registered objects: ``$type`` nodes become
        ``{"type": name, **state}``, environments their names, graphs and
        data frames dicts. By default False.
    """

    def __init__(
        self,
        read_array: Callable[[str], np.ndarray],
        environments: Mapping[str, Environment] | None = None,
        *,
        plain: bool = False,
    ) -> None:
        self.read_array = read_array
        self.environments = dict(environments or {})
        self.plain = plain
        self._refs: dict[int, Any] = {}

    def decode(self, node: JSON) -> Any:
        """Return the value of ``node``."""
        if node is None or isinstance(node, (bool, int, float, str)):
            return node
        if isinstance(node, list):
            return [self.decode(v) for v in node]
        if not isinstance(node, dict):
            msg = f"Unexpected JSON value {node!r} in the saved model"
            raise ValidationError(msg)
        tags = [key for key in node if key.startswith("$")]
        if not tags:
            return {k: self.decode(v) for k, v in node.items()}
        tag = tags[0]
        handler = self._HANDLERS.get(tag)
        if handler is None:
            msg = f"Unknown tag {tag!r} in the saved model"
            raise ValidationError(msg, hint="Upgrade non_local_detector to read it.")
        try:
            return handler(self, node)
        except NonLocalDetectorError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as error:
            # A node missing a field, pointing at no stored array, or holding
            # the wrong kind of value, or a registered type refusing its
            # state. Package errors (ValidationError, DataError, ...) are
            # ValueErrors too and pass through unchanged.
            if tag == "$type":
                msg = (
                    f"The registered type {node[tag]!r} could not be rebuilt "
                    f"from its stored state ({type(error).__name__}: {error})"
                )
                hint = (
                    "The file was edited, or the type's from_state does not "
                    "read the state this file holds (for example it was saved "
                    "with a different version of that type)."
                )
            else:
                label = f"{tag} {node[tag]!r}" if isinstance(node[tag], str) else tag
                msg = (
                    f"Cannot rebuild the saved model's {label} entry "
                    f"({type(error).__name__}: {error})"
                )
                hint = "The file is corrupt or was edited."
            raise DataError(msg, hint=hint) from error

    def _as(self, array: np.ndarray, *, is_jax: bool) -> Any:
        """``array`` as the JAX array it was saved as, or as is.

        A 64-bit array saved from JAX with ``JAX_ENABLE_X64`` stays a NumPy
        array when x64 is off: JAX would narrow it to 32 bits, losing
        precision (and wrapping large integers), while NumPy keeps the
        stored values, which JAX code accepts as input.
        """
        if not is_jax or jax.dtypes.canonicalize_dtype(array.dtype) != array.dtype:
            return array
        return jnp.asarray(array)

    def _array(self, node: JSON) -> Any:
        ref = node["ref"]
        if ref not in self._refs:
            array = self.read_array(node["$array"])
            if node.get("object", False):
                array = array.astype(object)
            self._refs[ref] = self._as(array, is_jax=node["jax"])
        return self._refs[ref]

    def _fill(self, node: JSON) -> Any:
        ref = node["ref"]
        if ref not in self._refs:
            array = np.full(tuple(node["shape"]), node["$fill"], dtype=node["dtype"])
            self._refs[ref] = self._as(array, is_jax=node["jax"])
        return self._refs[ref]

    def _ragged(self, node: JSON) -> Any:
        ref = node["ref"]
        if ref not in self._refs:
            data = np.asarray(self.decode(node["$ragged"]["data"]))
            offsets = np.asarray(self.decode(node["$ragged"]["offsets"]))
            self._refs[ref] = [
                self._as(data[start:stop], is_jax=node["jax"])
                for start, stop in itertools.pairwise(offsets)
            ]
        return self._refs[ref]

    def _item(self, node: JSON) -> Any:
        return self._refs[node["$item"]][node["index"]]

    def _tuple(self, node: JSON) -> Any:
        return tuple(self.decode(v) for v in node["$tuple"])

    def _dict(self, node: JSON) -> Any:
        return {self.decode(k): self.decode(v) for k, v in node["$dict"]}

    def _float(self, node: JSON) -> Any:
        return _decode_float(node)

    def _scalar(self, node: JSON) -> Any:
        value = node["$scalar"]
        if isinstance(value, dict):
            value = _decode_float(value)
        return np.dtype(node["dtype"]).type(value)

    def _type(self, node: JSON) -> Any:
        state = self.decode(node["state"])
        if self.plain:
            return {"type": node["$type"], **state}
        spec = spec_for_name(node["$type"], node.get("owner"))
        if spec.from_state is None:
            msg = f"The saved model holds a detector ({spec.name!r}) inside its state"
            raise ValidationError(msg, hint="The file is corrupt or was edited.")
        return spec.from_state(state)

    def _environment(self, node: JSON) -> Any:
        name = node["$environment"]
        if self.plain:
            return name
        return self.environments[name]

    def _graph(self, node: JSON) -> Any:
        content = node["$graph"]
        nodes = [(self.decode(n), self.decode(a)) for n, a in content["nodes"]]
        edges = [
            (self.decode(u), self.decode(v), self.decode(a))
            for u, v, a in content["edges"]
        ]
        attributes = self.decode(content["attributes"])
        if self.plain:
            return {"directed": content["directed"], "nodes": nodes, "edges": edges}
        graph = nx.DiGraph() if content["directed"] else nx.Graph()
        graph.graph.update(attributes)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _categorical(self, node: JSON) -> Any:
        content = node["$categorical"]
        codes = np.asarray(self.decode(content["codes"]))
        categories = self.decode(content["categories"])
        if self.plain:
            return {
                "codes": codes,
                "categories": categories,
                "ordered": content["ordered"],
            }
        return pd.Categorical.from_codes(
            codes, categories=categories, ordered=content["ordered"]
        )

    def _dataframe(self, node: JSON) -> Any:
        content = node["$dataframe"]
        columns = [self.decode(name) for name in content["columns"]]
        values = [self.decode(value) for value in content["values"]]
        if self.plain:
            return dict(zip(columns, values, strict=True))
        frame = pd.DataFrame(index=self._index(content["index"]))
        for name, value in zip(columns, values, strict=True):
            frame[name] = value
        return frame

    def _index(self, encoded: JSON) -> pd.Index:
        if "range" in encoded:
            return pd.RangeIndex(*encoded["range"])
        return pd.Index(self.decode(encoded["values"]))

    def _series(self, node: JSON) -> Any:
        # Restored as a Series, with its index and name, as it was fitted.
        content = node["$series"]
        values = self.decode(content["values"])
        if self.plain:
            return values
        return pd.Series(
            values,
            index=self._index(content["index"]),
            name=self.decode(content["name"]),
        )

    # The tags a file may use, each to its handler; nothing else is called.
    _HANDLERS: Mapping[str, Callable[["Decoder", JSON], Any]] = MappingProxyType(
        {
            "$array": _array,
            "$fill": _fill,
            "$ragged": _ragged,
            "$item": _item,
            "$tuple": _tuple,
            "$dict": _dict,
            "$float": _float,
            "$scalar": _scalar,
            "$type": _type,
            "$environment": _environment,
            "$graph": _graph,
            "$categorical": _categorical,
            "$dataframe": _dataframe,
            "$series": _series,
        }
    )
