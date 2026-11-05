"""The file format of saved detectors.

``detector.save_model(path)`` writes a fitted (or unfitted) detector as one
zip file of JSON documents and ``.npy`` arrays, and ``load_model(path)`` reads
it, or a pickle saved by an earlier release. Loading a saved model never runs
code from the file: arrays load without pickle, and every object is built
through a registry of known types (see :func:`register_type`).

What a file stores, and what loading rebuilds and checks, is described in the
"Saving and loading models" documentation page.

It provides:

- ``register_type``, to let models holding a custom transition or a detector
  subclass be saved;
- ``read_model_summary``, to read a saved model's configuration, grids and
  place fields without building it;
- ``convert_model``, to rewrite a pickled model in the saved-model format;
- ``is_saved_model_format``, whether a file is in the saved-model format (as
  opposed to a pickle);
- ``FORMAT_VERSION``, the format version this release writes and the newest
  it reads, and ``MIGRATIONS``, where ``MIGRATIONS[v]`` upgrades a stored
  model of format version ``v`` to ``v + 1`` before it is read.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from non_local_detector.exceptions import NotFittedError, ValidationError
from non_local_detector.model_format._codec import Decoder
from non_local_detector.model_format._container import (
    FORMAT_VERSION,
    MIGRATIONS,
    StoredModel,
    is_saved_model_format,
    open_model,
)
from non_local_detector.model_format._detector import (
    check_documents,
    data_errors,
    encoding_model_keys,
)
from non_local_detector.model_format._registry import (
    ignoring_unread_names,
    register_type,
    registered_types,
)

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase

__all__ = [
    "FORMAT_VERSION",
    "MIGRATIONS",
    "StoredModel",
    "convert_model",
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
    but not encoding spike samples or fitted mixtures. Nothing is built and
    no transition formula is evaluated, so this works whether or not the
    rebuilt structures of the model would still match (see ``load_model``),
    on files you do not trust, and is fast on large clusterless models.

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
    DataError
        If the file is corrupt or was edited.

    Warns
    -----
    CompatibilityWarning
        Once, if the file holds names this version does not read (a newer
        version wrote it), which are left out.
    """
    if not is_saved_model_format(path):
        msg = f"{path} is not in the saved-model format (it is probably a pickle)"
        raise ValidationError(msg, hint="Read it with load_model.")
    with (
        open_model(path) as stored,
        ignoring_unread_names(stored.manifest.get("package_version")),
        data_errors("read"),
    ):
        check_documents(stored)
        return _summary(stored)


def _summary(stored: StoredModel) -> dict[str, Any]:
    """Return the summary ``read_model_summary`` returns, of checked documents."""
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
            "environment_name": decoder.decode(entry["config"])["environment_name"],
            "place_bin_centers": decoder.decode(entry["place_bin_centers"]),
            "is_track_interior": decoder.decode(entry["is_track_interior"]),
            "edges": [decoder.decode(edge) for edge in entry["edges"]],
        }
        environments.append(environment)
    summary["environments"] = environments
    summary["encoding_models"] = []
    for entry in state["encoding_models"]:
        keys = encoding_model_keys(entry["model"])
        summary["encoding_models"].append(
            {
                "environment_name": entry["environment_name"],
                "encoding_group": decoder.decode(entry["encoding_group"]),
                **{
                    key: decoder.decode(value)
                    for key, value in entry["model"].items()
                    if key in keys and _is_small(value)
                },
            }
        )
    # Files saved before the shared keys' values were NumPy arrays.
    from non_local_detector.likelihoods import _unify_stored_encoding_values

    for model in summary["encoding_models"]:
        _unify_stored_encoding_values(model)
    summary["discrete_state_transitions"] = (
        decoder.decode(state["discrete_state_transitions"])
        if "discrete_state_transitions" in state
        else None
    )
    summary["initial_conditions"] = decoder.decode(state["initial_conditions"])
    return summary


def _refuse_undecodable(detector: "_DetectorBase", path: str | Path) -> None:
    """Raise if a legacy detector cannot decode, so cannot be converted."""
    algorithms = {
        getattr(detector, "clusterless_algorithm", None),
        *(
            model.get("algorithm")
            for model in getattr(detector, "encoding_model_", {}).values()
            if isinstance(model, dict)
        ),
    }
    if "clusterless_kde_log" in algorithms:
        msg = (
            f"Cannot convert {path}: it uses the removed clusterless_kde_log "
            "algorithm, so it cannot decode"
        )
        raise ValidationError(
            msg,
            hint="Keep it as a pickle to inspect it with load_model, or refit "
            'with clusterless_algorithm="clusterless_kde" and save that.',
        )
    try:
        detector._validate_time_contract()
    except (NotFittedError, ValidationError) as error:
        msg = f"Cannot convert {path}: the model cannot decode ({error.args[0]})"
        raise ValidationError(
            msg,
            hint="Keep it as a pickle to inspect it with load_model, or refit "
            "it and save the refit model.",
        ) from error


def convert_model(pickle_path: str | Path, path: str | Path) -> None:
    """Rewrite a pickled detector in the saved-model format.

    Loads ``pickle_path`` as ``load_model`` loads a pickle and saves it to
    ``path`` with ``save_model``. Only convert files you trust: loading a
    pickle can execute arbitrary code. The fingerprints the new file uses to
    check its rebuilt structures are computed now, from the structures the
    pickle holds, not at the original fit. The transition formula is written
    as the pickle has it; if it is outside the formula grammar, loading the
    converted file evaluates it only with ``load_model(path,
    trust_formulas=True)``.

    Parameters
    ----------
    pickle_path : str or Path
        A detector saved with pickle by an earlier release.
    path : str or Path
        Where to write the converted model, by convention with suffix
        ``.nld``.

    Raises
    ------
    ValidationError
        If ``pickle_path`` is already in the saved-model format, or the
        model cannot decode: fitted before rates were in Hz (the 0.6
        releases and earlier) or with the removed ``clusterless_kde_log``
        algorithm. Such files stay pickles, which ``load_model`` still
        reads for inspection.
    """
    from non_local_detector.models.base import _DetectorBase

    if is_saved_model_format(pickle_path):
        msg = f"{pickle_path} is already in the saved-model format"
        raise ValidationError(msg)
    detector = _DetectorBase.load_model(pickle_path)
    _refuse_undecodable(detector, pickle_path)
    detector.save_model(path)
