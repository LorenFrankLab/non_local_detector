try:
    from non_local_detector.visualization.figurl_1D import (
        create_interactive_1D_decoding_figurl,
    )
    from non_local_detector.visualization.figurl_2D import (
        create_interactive_2D_decoding_figurl,
    )
except ImportError:
    pass
from non_local_detector.visualization.static import plot_non_local_model

__all__ = [
    "create_interactive_1D_decoding_figurl",
    "create_interactive_2D_decoding_figurl",
    "launch",
    "plot_non_local_model",
]


def __getattr__(name: str):
    """Lazy-expose ``launch`` from the interactive sub-package.

    Defers loading ``visualization.interactive`` (and its transitive
    dataclasses) until the user actually accesses
    ``non_local_detector.visualization.launch``. The Qt frontend is
    further lazy-loaded inside ``launch`` itself, so importing this
    name doesn't require the ``[viewer]`` extra; calling it does.
    """
    if name == "launch":
        from non_local_detector.visualization.interactive import launch

        return launch
    raise AttributeError(
        f"module 'non_local_detector.visualization' has no attribute {name!r}"
    )
