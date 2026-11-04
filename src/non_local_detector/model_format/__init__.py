"""The file format of saved detectors.

``detector.save_model(path)`` writes a fitted (or unfitted) detector as one
zip file of JSON documents and ``.npy`` arrays, and ``load_model(path)`` reads
it, or a pickle saved by an earlier release. Loading a saved model never runs
code from the file: arrays load without pickle, and every object is built
through a registry of known types (see :func:`register_type`).

What a file stores, and what loading rebuilds and checks, is described in the
"Saving and loading models" documentation page.

Functions
---------
register_type
    Let models holding a custom transition or detector subclass be saved.
is_saved_model_format
    Whether a file is in the saved-model format (as opposed to a pickle).

Attributes
----------
FORMAT_VERSION : int
    The format version this release writes and the newest it reads.
MIGRATIONS : dict[int, callable]
    ``MIGRATIONS[v]`` upgrades a stored model of format version ``v`` to
    ``v + 1`` before it is read.
"""

from non_local_detector.model_format._container import (
    FORMAT_VERSION,
    MIGRATIONS,
    StoredModel,
    is_saved_model_format,
)
from non_local_detector.model_format._registry import register_type, registered_types

__all__ = [
    "FORMAT_VERSION",
    "MIGRATIONS",
    "StoredModel",
    "is_saved_model_format",
    "register_type",
    "registered_types",
]
