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
    _members: dict[str, bytes] = field(default_factory=dict, repr=False)
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
        data = self._members.get(member)
        if data is None and self._read_member is not None:
            with contextlib.suppress(KeyError):
                data = self._read_member(member)
        if data is None:
            msg = f"The saved model is missing its array {member!r}"
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


def _parse(archive: zipfile.ZipFile, name: str) -> Any:
    try:
        return json.loads(archive.read(name))
    except KeyError:
        msg = f"The file has no {name}; it is not a saved non_local_detector model"
        raise DataError(msg) from None


def _check_version(manifest: dict[str, Any]) -> int:
    if manifest.get("format") != FORMAT_NAME:
        msg = "The file is not a saved non_local_detector model"
        raise DataError(msg, got=f"format {manifest.get('format')!r}")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version < 1:
        msg = f"The saved model has an invalid format_version {version!r}"
        raise DataError(msg)
    if version > FORMAT_VERSION:
        writer = manifest.get("package_version", "a newer version")
        msg = (
            f"The model was saved in format version {version}, newer than the "
            f"{FORMAT_VERSION} this non_local_detector reads: it was written by "
            f"non_local_detector {writer}"
        )
        raise ValidationError(
            msg, hint=f"Upgrade to non_local_detector >= {writer} to read it."
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
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest, config, state = (_parse(archive, name) for name in _DOCUMENTS)
        version = _check_version(manifest)
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if name not in _DOCUMENTS
        }
    return _migrated(StoredModel(manifest, config, state, _members=members), version)


@contextlib.contextmanager
def open_model(path: str | Path) -> Iterator[StoredModel]:
    """Open a saved model, reading arrays only on request.

    Like :func:`read_model`, but the arrays are read from the file, which
    stays open for the ``with`` block.

    Yields
    ------
    StoredModel
    """
    with zipfile.ZipFile(path) as archive:
        manifest, config, state = (_parse(archive, name) for name in _DOCUMENTS)
        version = _check_version(manifest)
        yield _migrated(
            StoredModel(manifest, config, state, _read_member=archive.read), version
        )
