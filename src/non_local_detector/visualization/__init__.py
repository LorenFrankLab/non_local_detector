"""Visualization sub-package.

Eagerly exports ``plot_non_local_model`` (no optional deps) and
lazily exposes:

- ``launch`` — Qt interactive viewer (``[viewer]`` extra).
- ``create_interactive_1D_decoding_figurl`` /
  ``create_interactive_2D_decoding_figurl`` — figurl exports
  (require ``sortingview``).

Lazy access is via the module-level ``__getattr__`` (PEP 562). The
optional-dep imports run only when the user accesses the name, and
fail with the underlying ``ImportError`` so the missing-extra hint
points at the right install command.
"""

from non_local_detector.visualization.static import plot_non_local_model

_EAGER_NAMES = ("plot_non_local_model",)
_LAZY_NAMES = (
    "launch",
    "create_interactive_1D_decoding_figurl",
    "create_interactive_2D_decoding_figurl",
)


def __getattr__(name: str):
    """Lazy-load optional-dep names.

    Importing names from ``visualization.interactive`` (Qt viewer) and
    ``visualization.figurl_*`` (sortingview) is deferred until first
    use so the parent package stays importable without the optional
    extras.
    """
    if name == "launch":
        from non_local_detector.visualization.interactive import launch

        return launch
    if name == "create_interactive_1D_decoding_figurl":
        from non_local_detector.visualization.figurl_1D import (
            create_interactive_1D_decoding_figurl,
        )

        return create_interactive_1D_decoding_figurl
    if name == "create_interactive_2D_decoding_figurl":
        from non_local_detector.visualization.figurl_2D import (
            create_interactive_2D_decoding_figurl,
        )

        return create_interactive_2D_decoding_figurl
    raise AttributeError(
        f"module 'non_local_detector.visualization' has no attribute {name!r}"
    )


def __dir__() -> list[str]:
    """Make all eager + lazy names show up in tab-completion."""
    return sorted(set(_EAGER_NAMES) | set(_LAZY_NAMES))


def _build_all() -> list[str]:
    """``__all__`` exposes lazy names only if their optional dep imports.

    ``from non_local_detector.visualization import *`` will fail if any
    name in ``__all__`` doesn't resolve, so we must filter out
    figurl names when ``sortingview`` isn't installed.
    """
    available = list(_EAGER_NAMES) + ["launch"]
    for figurl_name in (
        "create_interactive_1D_decoding_figurl",
        "create_interactive_2D_decoding_figurl",
    ):
        try:
            __getattr__(figurl_name)
        except ImportError:
            continue
        available.append(figurl_name)
    return available


__all__ = _build_all()
