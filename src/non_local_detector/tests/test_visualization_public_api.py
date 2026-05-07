"""Tests for the public ``non_local_detector.visualization`` API surface.

These run without the ``[viewer]`` extra (lazy import), so they're
plain unit tests rather than ``gui``-marked.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_launch_lazy_exposed_from_visualization():
    """``visualization.launch`` resolves through ``__getattr__`` lazily.

    Importing the parent module must not eagerly pull in the
    ``visualization.interactive`` sub-package. Accessing ``launch``
    triggers the import on demand and returns the function shipped
    by ``visualization.interactive``.

    Run in a subprocess so the lazy-import probe doesn't pollute
    ``sys.modules`` for the surrounding test session — earlier tests
    have already imported ``MetricSpec`` etc. from
    ``visualization.interactive``, and a re-import here would replace
    those classes with new identities and break later ``isinstance``
    checks in the Qt viewer suite.
    """
    script = textwrap.dedent(
        """
        import sys

        import non_local_detector.visualization as v

        assert "launch" in v.__all__, f"launch missing from __all__: {v.__all__}"
        assert (
            "non_local_detector.visualization.interactive" not in sys.modules
        ), "Plain import of visualization should NOT load interactive sub-package"

        launch = v.launch
        assert callable(launch), f"launch is not callable: {launch!r}"
        assert (
            "non_local_detector.visualization.interactive" in sys.modules
        ), "Accessing visualization.launch should lazy-load the interactive sub-package"

        interactive_launch = sys.modules[
            "non_local_detector.visualization.interactive"
        ].launch
        assert launch is interactive_launch
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"lazy-import probe failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK")


def test_visualization_unknown_attribute_raises_attribute_error():
    """Module-level ``__getattr__`` must still raise on unknown names."""
    import pytest

    import non_local_detector.visualization as v

    with pytest.raises(AttributeError, match="no attribute 'definitely_not_a_name'"):
        v.definitely_not_a_name  # noqa: B018


def test_visualization_star_import_only_exposes_resolvable_names():
    """``__all__`` must list only names that actually resolve.

    ``from non_local_detector.visualization import *`` is the canonical
    way Python users probe the public API; if ``__all__`` lists a name
    whose lazy import fails (e.g. a figurl helper when ``sortingview``
    isn't installed), the star-import raises ``AttributeError`` and the
    package becomes effectively unimportable.

    Run in a subprocess so the optional deps' import side-effects don't
    leak across tests.
    """
    script = textwrap.dedent(
        """
        ns = {}
        exec("from non_local_detector.visualization import *", ns)
        import non_local_detector.visualization as v

        for name in v.__all__:
            assert name in ns, f"__all__ contains unresolvable name: {name!r}"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"star-import probe failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK")
