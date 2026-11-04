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
read_model_summary
    Read a saved model's configuration, grids and place fields without
    building it.
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

from pathlib import Path
from typing import Any

from non_local_detector.exceptions import ValidationError
from non_local_detector.model_format._codec import Decoder
from non_local_detector.model_format._container import (
    FORMAT_VERSION,
    MIGRATIONS,
    StoredModel,
    is_saved_model_format,
    open_model,
)
from non_local_detector.model_format._registry import register_type, registered_types

__all__ = [
    "FORMAT_VERSION",
    "MIGRATIONS",
    "StoredModel",
    "is_saved_model_format",
    "read_model_summary",
    "register_type",
    "registered_types",
]

# Encoding-model fields summarized by ``read_model_summary``: JSON values and
# single arrays. Lists of arrays and fitted sub-models (KDE samples, mixtures)
# are left out.
_SUMMARY_TAGS = ("$array", "$fill", "$scalar", "$float", "$tuple")


def _is_small(node: Any) -> bool:
    if node is None or isinstance(node, (bool, int, float, str)):
        return True
    if isinstance(node, list):
        return all(_is_small(item) and not isinstance(item, dict) for item in node)
    if isinstance(node, dict):
        if any(tag in node for tag in _SUMMARY_TAGS):
            return "$tuple" not in node or all(
                _is_small(item) and not isinstance(item, dict)
                for item in node["$tuple"]
            )
        if any(key.startswith("$") for key in node):
            return False
        return all(_is_small(value) for value in node.values())
    return False


def read_model_summary(path: str | Path) -> dict[str, Any]:
    """Read what a saved model learned, without building the detector.

    Reads the configuration, each environment's grid and each encoding
    model's per-bin summaries (place fields, mean rates, occupancy, ...),
    but not encoding spike samples or fitted mixtures. Nothing is built, so
    this works whether or not the rebuilt structures of the model would still
    match (see ``load_model``), and is fast on large clusterless models.

    Parameters
    ----------
    path : str or Path
        A file written by ``save_model`` (not a pickle).

    Returns
    -------
    summary : dict
        ``detector_type`` (str), ``format_version`` (int),
        ``package_version`` (str, the version that saved it), ``config``
        (the detector's ``get_params()``, with registered objects as
        ``{"type": name, **fields}`` dicts and environments' track graphs as
        ``{"directed", "nodes", "edges"}`` dicts), ``fitted`` (bool) and, for a
        fitted model:

        ``environments``
            list of dicts with ``environment_name``, ``place_bin_centers``
            (n_bins, n_dims), ``is_track_interior`` (bins per dimension) and,
            without a track graph, ``edges`` (bin edges of each dimension).
        ``encoding_models``
            list of dicts with ``environment_name``, ``encoding_group`` and
            the model's JSON and single-array fields, e.g. ``algorithm``,
            ``place_fields`` (n_neurons, n_bins) in Hz and ``mean_rates``.
        ``discrete_state_transitions``
            (n_states, n_states) for a stationary discrete transition, else
            None.
        ``initial_conditions``
            (n_state_bins,).

    Raises
    ------
    ValidationError
        If the file is a pickle; read it with ``load_model``.
    """
    if not is_saved_model_format(path):
        msg = f"{path} is not in the saved-model format (it is probably a pickle)"
        raise ValidationError(msg, hint="Read it with load_model.")
    with open_model(path) as stored:
        decoder = Decoder(stored.read_array, plain=True)
        state = stored.state
        summary: dict[str, Any] = {
            "detector_type": stored.manifest["detector_type"],
            "format_version": stored.manifest["format_version"],
            "package_version": stored.manifest.get("package_version"),
            "config": decoder.decode(stored.config),
            "fitted": state["fitted"],
        }
        if not state["fitted"]:
            return summary
        decoder = Decoder(stored.read_array, plain=True)
        environments = []
        for entry in state["environments"]:
            environment = {
                "environment_name": entry["config"]["state"]["environment_name"],
                "place_bin_centers": decoder.decode(entry["place_bin_centers"]),
                "is_track_interior": decoder.decode(entry["is_track_interior"]),
            }
            if "edges" in entry:
                environment["edges"] = [decoder.decode(edge) for edge in entry["edges"]]
            environments.append(environment)
        summary["environments"] = environments
        summary["encoding_models"] = [
            {
                "environment_name": entry["environment_name"],
                "encoding_group": decoder.decode(entry["encoding_group"]),
                **{
                    key: decoder.decode(value)
                    for key, value in entry["model"].items()
                    if _is_small(value)
                },
            }
            for entry in state["encoding_models"]
        ]
        summary["discrete_state_transitions"] = (
            decoder.decode(state["discrete_state_transitions"])
            if "discrete_state_transitions" in state
            else None
        )
        summary["initial_conditions"] = decoder.decode(state["initial_conditions"])
        return summary
