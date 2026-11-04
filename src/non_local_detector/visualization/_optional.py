"""Detection of the optional ``sortingview`` dependency (``viz`` extra)."""

import importlib

SORTINGVIEW_INSTALL_HINT = (
    'The figurl views require the optional "sortingview" dependency. '
    'Install it with: pip install "non_local_detector[viz]"'
)


def sortingview_available() -> bool:
    """Return whether ``sortingview`` can be imported.

    Only a missing ``sortingview`` itself counts as unavailable; any other
    import failure (a broken installation, a missing transitive dependency)
    propagates so it is not mistaken for a missing extra.

    Returns
    -------
    available : bool
        True if ``import sortingview`` succeeds.
    """
    try:
        importlib.import_module("sortingview")
    except ModuleNotFoundError as e:
        if e.name == "sortingview":
            return False
        raise
    return True
