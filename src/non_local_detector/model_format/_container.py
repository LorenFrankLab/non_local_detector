"""The zip container of a saved model and its versions.

A saved model is one zip file holding ``manifest.json`` (format name and
version, the writing package version, the detector type and an index of the
arrays with their sha256), ``config.json`` (the detector's ``get_params()``),
``state.json`` (the fitted state, referring to arrays) and ``arrays/**.npy``.
Arrays over ``DEFLATE_THRESHOLD`` bytes are compressed with deflate level
``DEFLATE_LEVEL``; smaller members are stored. Arrays load with
``allow_pickle=False``.
"""

import contextlib
import hashlib
import io
import json
import zipfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from non_local_detector.exceptions import DataError, ValidationError

FORMAT_NAME = "non_local_detector.model"
FORMAT_VERSION = 1
DEFLATE_THRESHOLD = 4096
DEFLATE_LEVEL = 1
_ZIP_MAGIC = b"PK\x03\x04"
_DOCUMENTS = ("manifest.json", "config.json", "state.json")
# A JSON document may declare at most this many uncompressed bytes, and an
# array member its data's size plus this much for the .npy header.
_MAX_DOCUMENT_BYTES = 256 * 2**20
_NPY_HEADER_BYTES = 64 * 2**10
# Fixed member timestamps, so saving the same model twice writes the same bytes.
_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass
class StoredModel:
    """The documents and arrays of a saved model, before anything is built.

    Migrations (``MIGRATIONS``) edit these in place.

    Attributes
    ----------
    manifest, config, state : dict
        The parsed JSON documents.
    arrays : dict[str, np.ndarray]
        Arrays a migration replaced or added, by member path; they take
        precedence over the file's members and are not checked against the
        manifest.
    """

    manifest: dict[str, Any]
    config: Any
    state: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    _read_member: Callable[[str], bytes] | None = field(default=None, repr=False)

    def read_array(self, member: str) -> np.ndarray:
        """The array stored as ``member``, after checking its sha256.

        Raises
        ------
        DataError
            If the member is missing, not in the manifest, or its bytes do not
            match the manifest's sha256.
        """
        if member in self.arrays:
            return self.arrays[member]
        entry = self.manifest.get("arrays", {}).get(member)
        if entry is None:
            msg = f"The saved model refers to {member!r}, which its manifest does not list"
            raise DataError(msg, hint="The file is corrupt or was edited.")
        data = None
        if self._read_member is not None:
            with contextlib.suppress(KeyError):
                data = self._read_member(member)
        if data is None:
            msg = f"The saved model is missing its array member {member!r}"
            raise DataError(msg, hint="The file is corrupt or truncated.")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            msg = f"The array {member!r} of the saved model does not match its checksum"
            raise DataError(
                msg,
                hint="The file is corrupt or was edited; save the model again.",
            )
        return cast(np.ndarray, np.load(io.BytesIO(data), allow_pickle=False))


# MIGRATIONS[v] upgrades a StoredModel of format_version v to v + 1, in place.
MIGRATIONS: dict[int, Callable[[StoredModel], None]] = {}


def is_saved_model_format(path: str | Path) -> bool:
    """Whether the file at ``path`` starts like a zip, the saved-model format.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    bool
        True for the zip format; False for anything else (a pickle).
    """
    with Path(path).open("rb") as file:
        return file.read(len(_ZIP_MAGIC)) == _ZIP_MAGIC


def write_model(
    path: str | Path,
    manifest: dict[str, Any],
    config: Any,
    state: dict[str, Any],
    members: dict[str, bytes],
) -> None:
    """Write a saved model to ``path``, replacing any file there.

    The file is written beside ``path`` and moved into place, so a failed save
    leaves an existing file unchanged.
    """
    path = Path(path)
    documents = {
        "manifest.json": manifest,
        "config.json": config,
        "state.json": state,
    }
    partial = path.with_name(path.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w") as archive:
            for name, document in documents.items():
                text = json.dumps(document, allow_nan=False, indent=1)
                _write_member(archive, name, text.encode())
            for name, data in members.items():
                _write_member(archive, name, data)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_TIMESTAMP)
    if len(data) > DEFLATE_THRESHOLD:
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, data, compresslevel=DEFLATE_LEVEL)
    else:
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, data)


class _Archive:
    """The members of a saved model's zip, checked before any is decompressed.

    Every member must have a plain relative name, appear once, be a document
    or an array the manifest lists, and declare an uncompressed size within
    what its document limit or array (dtype and shape) allows, so a member
    added to the file, or one inflating far beyond its array, is refused
    before it is read.

    Raises
    ------
    DataError
        If the file is not a zip, or a member breaks these rules.
    """

    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.infos: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            name = info.filename
            if (
                name.startswith("/")
                or "\\" in name
                or ".." in name.split("/")
                or ":" in name
            ):
                msg = f"The saved model has a member with an unsafe name {name!r}"
                raise DataError(msg, hint="The file is corrupt or was edited.")
            if name in self.infos:
                msg = f"The saved model lists the member {name!r} more than once"
                raise DataError(msg, hint="The file is corrupt or was edited.")
            self.infos[name] = info
        self.limits: dict[str, int] = dict.fromkeys(_DOCUMENTS, _MAX_DOCUMENT_BYTES)

    def document(self, name: str) -> Any:
        """The parsed JSON document ``name``."""
        if name not in self.infos:
            msg = f"The file has no {name}; it is not a saved non_local_detector model"
            raise DataError(msg)
        try:
            return json.loads(self.read(name))
        except ValueError:
            msg = f"The {name} member of the saved model is not valid JSON"
            raise DataError(msg, hint="The file is corrupt or was edited.") from None

    def list_arrays(self, manifest: dict[str, Any]) -> None:
        """Allow the members the manifest lists, and refuse any other."""
        arrays = manifest.get("arrays", {})
        for member, entry in arrays.items():
            try:
                expected = int(np.dtype(entry["dtype"]).itemsize) * int(
                    np.prod(entry["shape"], dtype=np.int64)
                )
            except (KeyError, TypeError, ValueError):
                msg = f"The manifest entry of the array member {member!r} is invalid"
                raise DataError(msg) from None
            self.limits[member] = expected + _NPY_HEADER_BYTES
        unlisted = sorted(set(self.infos) - set(self.limits))
        if unlisted:
            msg = f"The saved model has members its manifest does not list: {unlisted}"
            raise DataError(msg, hint="The file is corrupt or was edited.")

    def read(self, name: str) -> bytes:
        """The uncompressed bytes of member ``name``.

        Raises
        ------
        KeyError
            If there is no such member.
        DataError
            If it declares more bytes than its limit, or fails to decompress
            or its CRC check.
        """
        info = self.infos[name]
        limit = self.limits.get(name)
        if limit is None or info.file_size > limit:
            msg = (
                f"The member {name!r} of the saved model is larger than its "
                f"contents can be ({info.file_size} bytes)"
            )
            raise DataError(msg, hint="The file is corrupt or was edited.")
        try:
            return self.archive.read(info)
        except (zipfile.BadZipFile, zlib.error, EOFError) as error:
            msg = f"The member {name!r} of the saved model is corrupt ({error})"
            raise DataError(msg, hint="Save the model again.") from None


def _zip(source: str | Path | io.BytesIO) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(source)
    except zipfile.BadZipFile as error:
        msg = f"The file is not a readable saved model ({error})"
        raise DataError(msg, hint="The file is corrupt or truncated.") from None


def _open(archive: zipfile.ZipFile) -> tuple[StoredModel, int]:
    """The documents of an opened archive and their format version."""
    checked = _Archive(archive)
    manifest = checked.document("manifest.json")
    version = _check_version(manifest)
    checked.list_arrays(manifest)
    config, state = checked.document("config.json"), checked.document("state.json")
    return StoredModel(manifest, config, state, _read_member=checked.read), version


def _check_version(manifest: dict[str, Any]) -> int:
    if manifest.get("format") != FORMAT_NAME:
        msg = "The file is not a saved non_local_detector model"
        raise DataError(msg, got=f"format {manifest.get('format')!r}")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version < 1:
        msg = f"The saved model has an invalid format_version {version!r}"
        raise DataError(msg)
    if version > FORMAT_VERSION:
        writer = manifest.get("package_version")
        newer = (
            f"non_local_detector {writer}"
            if isinstance(writer, str) and writer and writer != "unknown"
            else "an unknown, newer non_local_detector"
        )
        msg = (
            f"The model was saved in format version {version}, newer than the "
            f"{FORMAT_VERSION} this non_local_detector reads: it was written by "
            f"{newer}"
        )
        raise ValidationError(
            msg,
            hint=f"Upgrade to non_local_detector >= {writer} to read it."
            if newer.startswith("non_local_detector ")
            else "Upgrade non_local_detector to a release that reads format "
            f"version {version}.",
        )
    return version


def _migrated(stored: StoredModel, version: int) -> StoredModel:
    while version < FORMAT_VERSION:
        MIGRATIONS[version](stored)
        version += 1
        stored.manifest["format_version"] = version
    return stored


def read_model(path: str | Path) -> StoredModel:
    """Read a whole saved model into memory, migrated to ``FORMAT_VERSION``.

    The file is closed before this returns.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    StoredModel

    Raises
    ------
    DataError
        If the file is not a saved model.
    ValidationError
        If it was written in a newer format version.
    """
    data = Path(path).read_bytes()
    with _zip(io.BytesIO(data)) as archive:
        stored, version = _open(archive)
        members = {
            name: stored._read_member(name)  # type: ignore[misc]  # set by _open
            for name in stored.manifest.get("arrays", {})
            if name in archive.namelist()
        }
    stored._read_member = members.__getitem__
    return _migrated(stored, version)


@contextlib.contextmanager
def open_model(path: str | Path) -> Iterator[StoredModel]:
    """Open a saved model, reading arrays only on request.

    Like :func:`read_model`, but the arrays are read from the file, which
    stays open for the ``with`` block.

    Yields
    ------
    StoredModel
    """
    with _zip(path) as archive:
        stored, version = _open(archive)
        yield _migrated(stored, version)
