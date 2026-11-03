"""Attribution of the library's warnings to the caller's code."""

import os
import sys
from types import FrameType

# Not resolved: frame filenames carry the same, possibly symlinked, path the
# package was imported from.
_LIBRARY_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_TESTS_DIR = os.path.join(_LIBRARY_DIR, "tests")


def _inside(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory + os.sep)


def user_stacklevel() -> int:
    """``warnings.warn`` stacklevel of the first caller outside the library.

    Call it in the function that calls ``warnings.warn``. Frames in this
    package, apart from its tests, are skipped, so the warning points at the
    user's code however deep inside the library it was raised and whichever
    public method led there.

    Returns
    -------
    stacklevel : int
    """
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None:
        path = os.path.abspath(frame.f_code.co_filename)
        if not _inside(path, _LIBRARY_DIR) or _inside(path, _LIBRARY_TESTS_DIR):
            break
        frame = frame.f_back
        level += 1
    return level
