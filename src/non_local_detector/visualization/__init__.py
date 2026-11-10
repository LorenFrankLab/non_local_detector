"""Visualization tools: static plots, movies, and interactive figurl views."""

import importlib
from typing import TYPE_CHECKING, Any

from non_local_detector.visualization._optional import (
    SORTINGVIEW_INSTALL_HINT,
    sortingview_available,
)
from non_local_detector.visualization.static import plot_non_local_model

if TYPE_CHECKING:
    from non_local_detector.visualization.figurl_1D import (
        create_interactive_1D_decoding_figurl,
    )
    from non_local_detector.visualization.figurl_2D import (
        create_interactive_2D_decoding_figurl,
    )

_FIGURL_FUNCTIONS = {
    "create_interactive_1D_decoding_figurl": "figurl_1D",
    "create_interactive_2D_decoding_figurl": "figurl_2D",
}

__all__ = [
    "create_interactive_1D_decoding_figurl",
    "create_interactive_2D_decoding_figurl",
    "plot_non_local_model",
]


def __dir__() -> list[str]:
    """List the lazily imported figurl views with the module's own names."""
    return sorted({*globals(), *__all__})


def __getattr__(name: str) -> Any:
    """Lazily import the figurl views, which need the optional ``sortingview``."""
    if name in _FIGURL_FUNCTIONS:
        if not sortingview_available():
            msg = f"Cannot use {name}. {SORTINGVIEW_INSTALL_HINT}"
            raise ImportError(msg)
        module = importlib.import_module(
            f"non_local_detector.visualization.{_FIGURL_FUNCTIONS[name]}"
        )
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
