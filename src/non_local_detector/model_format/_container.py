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
from typing import IO, Any

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
# Arrays are hashed and written in chunks of this many bytes.
_CHUNK_BYTES = 2**20
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
    _open_member: Callable[[str], IO[bytes]] | None = field(default=None, repr=False)

    def read_array(self, member: str) -> np.ndarray:
        """Return the array stored as ``member``, after checking its sha256.

        The member is decompressed straight into the array, hashing as it
        reads, so reading takes about the array's own memory.

        Raises
        ------
        DataError
            If the member is missing, not in the manifest, corrupt, or its
            bytes, dtype or shape do not match the manifest.
        """
        if member in self.arrays:
            return self.arrays[member]
        entry = self.manifest.get("arrays", {}).get(member)
        if entry is None:
            msg = f"The saved model refers to {member!r}, which its manifest does not list"
            raise DataError(msg, hint="The file is corrupt or was edited.")
        if self._open_member is None:
            msg = f"The saved model is closed; cannot read its array {member!r}"
            raise DataError(msg)
        try:
            stream = self._open_member(member)
        except KeyError:
            msg = f"The saved model is missing its array member {member!r}"
            raise DataError(msg, hint="The file is corrupt or truncated.") from None
        reader = _HashingReader(stream)
        try:
            with stream:
                array = np.lib.format.read_array(reader, allow_pickle=False)
                # Read to the end: the zip checks a member's CRC there.
                trailing = reader.read()
        except (zipfile.BadZipFile, zlib.error, EOFError, ValueError) as error:
            msg = f"The array member {member!r} of the saved model is corrupt ({error})"
            raise DataError(msg, hint="Save the model again.") from None
        if trailing or reader.hexdigest() != entry["sha256"]:
            msg = f"The array {member!r} of the saved model does not match its checksum"
            raise DataError(
                msg,
                hint="The file is corrupt or was edited; save the model again.",
            )
        _check_against_manifest(array, entry, member)
        return array


class _HashingReader:
    """A read-only stream that hashes what it reads (SHA-256)."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self._digest.update(data)
        return data

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _check_against_manifest(
    array: np.ndarray, entry: dict[str, Any], member: str
) -> None:
    """Refuse an array whose dtype or shape is not the manifest's."""
    if array.dtype.str != entry["dtype"] or list(array.shape) != list(entry["shape"]):
        msg = (
            f"The array {member!r} of the saved model has dtype {array.dtype.str} "
            f"and shape {list(array.shape)}, not the manifest's {entry['dtype']} "
            f"and {list(entry['shape'])}"
        )
        raise DataError(msg, hint="The file is corrupt or was edited.")


# MIGRATIONS[v] upgrades a StoredModel of format_version v to v + 1, in place.
MIGRATIONS: dict[int, Callable[[StoredModel], None]] = {}


def is_saved_model_format(path: str | Path) -> bool:
    """Whether the file at ``path`` starts like a zip, the saved-model format.

    Parameters
    ----------
    path : str or Path
        File to check.

    Returns
    -------
    bool
        True for the zip format; False for anything else (a pickle).
    """
    with Path(path).open("rb") as file:
        return file.read(len(_ZIP_MAGIC)) == _ZIP_MAGIC


def _npy_header(array: np.ndarray) -> bytes:
    """Return the ``.npy`` header of ``array``: version 1.0, or 2.0 if it needs it."""
    header = io.BytesIO()
    fields = np.lib.format.header_data_from_array_1_0(array)
    try:
        np.lib.format.write_array_header_1_0(header, fields)
    except ValueError:
        header = io.BytesIO()
        np.lib.format.write_array_header_2_0(header, fields)
    return header.getvalue()


def _npy_chunks(array: np.ndarray) -> Iterator[memoryview]:
    """Yield the bytes of a C-contiguous ``array``'s ``.npy`` member.

    The chunks view the array's memory rather than copy it.
    """
    yield memoryview(_npy_header(array))
    data = array.reshape(-1).view(np.uint8)
    for start in range(0, data.size, _CHUNK_BYTES):
        yield memoryview(data[start : start + _CHUNK_BYTES])


def array_digest(array: np.ndarray) -> str:
    """Return the sha256 of a C-contiguous ``array``'s ``.npy`` member.

    The member is not built.
    """
    digest = hashlib.sha256()
    for chunk in _npy_chunks(array):
        digest.update(chunk)
    return digest.hexdigest()


def _member_info(name: str, size: int) -> zipfile.ZipInfo:
    """Return a member info of fixed timestamp, deflated if over ``DEFLATE_THRESHOLD``."""
    info = zipfile.ZipInfo(name, date_time=_TIMESTAMP)
    if size > DEFLATE_THRESHOLD:
        info.compress_type = zipfile.ZIP_DEFLATED
        # Called compress_level from Python 3.13, _compresslevel before.
        name_of_level = (
            "compress_level" if hasattr(info, "compress_level") else "_compresslevel"
        )
        setattr(info, name_of_level, DEFLATE_LEVEL)
    else:
        info.compress_type = zipfile.ZIP_STORED
    return info


def write_array_member(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    """Write a C-contiguous ``array`` as the ``.npy`` member ``name``.

    The array is streamed in, so no copy of it is built.
    """
    size = array.nbytes + _NPY_HEADER_BYTES
    with archive.open(
        _member_info(name, size), "w", force_zip64=size > 2**31 - 1
    ) as member:
        for chunk in _npy_chunks(array):
            member.write(chunk)


def write_document(archive: zipfile.ZipFile, name: str, document: Any) -> None:
    """Write ``document`` as the JSON member ``name``."""
    data = json.dumps(document, allow_nan=False, indent=1).encode()
    info = _member_info(name, len(data))
    if info.compress_type == zipfile.ZIP_DEFLATED:
        archive.writestr(info, data, compresslevel=DEFLATE_LEVEL)
    else:
        archive.writestr(info, data)


@contextlib.contextmanager
def new_model_file(path: str | Path) -> Iterator[zipfile.ZipFile]:
    """Open a zip to write a saved model into, moved to ``path`` once complete.

    The file is written beside ``path`` and replaces it only if the ``with``
    block completes, so a failed save leaves an existing file unchanged.

    Yields
    ------
    zipfile.ZipFile
    """
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w") as archive:
            yield archive
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


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
        """Return the parsed JSON document ``name``."""
        if name not in self.infos:
            msg = f"The file has no {name}; it is not a saved non_local_detector model"
            raise DataError(msg)
        try:
            return json.loads(self.read(name))
        except (ValueError, RecursionError):  # nested too deeply to parse
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

    def open(self, name: str) -> IO[bytes]:
        """Open a stream of the uncompressed bytes of member ``name``.

        Raises
        ------
        KeyError
            If there is no such member.
        DataError
            If it declares more bytes than its limit, or cannot be opened.
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
            return self.archive.open(info)
        except (zipfile.BadZipFile, zlib.error, EOFError) as error:
            msg = f"The member {name!r} of the saved model is corrupt ({error})"
            raise DataError(msg, hint="Save the model again.") from None

    def read(self, name: str) -> bytes:
        """Return the uncompressed bytes of member ``name`` (see :meth:`open`)."""
        try:
            with self.open(name) as stream:
                return stream.read()
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
    """Return the documents of an opened archive and their format version."""
    checked = _Archive(archive)
    manifest = checked.document("manifest.json")
    version = _check_version(manifest)
    checked.list_arrays(manifest)
    config, state = checked.document("config.json"), checked.document("state.json")
    return StoredModel(manifest, config, state, _open_member=checked.open), version


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


@contextlib.contextmanager
def open_model(path: str | Path) -> Iterator[StoredModel]:
    """Open a saved model, migrated to ``FORMAT_VERSION``, for the ``with`` block.

    Arrays are read from the file on request (``StoredModel.read_array``);
    the file is closed when the block ends, after which no array can be read.

    Yields
    ------
    StoredModel

    Raises
    ------
    DataError
        If the file is not a saved model, or a member breaks the rules of
        ``_Archive``.
    ValidationError
        If it was written in a newer format version.
    """
    with _zip(path) as archive:
        stored, version = _open(archive)
        try:
            yield _migrated(stored, version)
        finally:
            stored._open_member = None
