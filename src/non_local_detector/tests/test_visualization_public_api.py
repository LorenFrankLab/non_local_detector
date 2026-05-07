"""Tests for the public ``non_local_detector.visualization`` API surface.

These run without the ``[viewer]`` extra (lazy import), so they're
plain unit tests rather than ``gui``-marked.
"""

from __future__ import annotations


def test_launch_lazy_exposed_from_visualization():
    """``visualization.launch`` resolves through ``__getattr__`` lazily.

    Importing the parent module must not eagerly pull in the
    ``visualization.interactive`` sub-package. Accessing ``launch``
    triggers the import on demand and returns the function shipped
    by ``visualization.interactive``.
    """
    import sys

    # Drop any previously cached interactive submodule so we can verify
    # the lazy-import really runs on first attribute access.
    for cached in list(sys.modules):
        if cached.startswith("non_local_detector.visualization.interactive"):
            del sys.modules[cached]
    import non_local_detector.visualization as v

    assert "launch" in v.__all__
    assert (
        "non_local_detector.visualization.interactive" not in sys.modules
    ), "Plain import of visualization should NOT load interactive sub-package"

    launch = v.launch
    assert callable(launch)
    assert (
        "non_local_detector.visualization.interactive" in sys.modules
    ), "Accessing visualization.launch should lazy-load the interactive sub-package"

    interactive_launch = sys.modules[
        "non_local_detector.visualization.interactive"
    ].launch
    assert launch is interactive_launch


def test_visualization_unknown_attribute_raises_attribute_error():
    """Module-level ``__getattr__`` must still raise on unknown names."""
    import non_local_detector.visualization as v
    import pytest

    with pytest.raises(AttributeError, match="no attribute 'definitely_not_a_name'"):
        v.definitely_not_a_name  # noqa: B018
