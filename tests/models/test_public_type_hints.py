"""Type hints of the detectors' public methods resolve at runtime.

``typing.get_type_hints`` (used by documentation and introspection tools)
evaluates every annotation, so a name imported only for type checking makes
it raise ``NameError``.
"""

import inspect
import types
import typing

import pytest

from non_local_detector import models

pytestmark = pytest.mark.unit


def test_public_method_type_hints_resolve():
    """``get_type_hints`` evaluates every annotation of every public method of
    the base detectors and the ready-made models, including methods defined on
    the mixins with a ``self: "non_local_detector.models.base._DetectorBase"``
    annotation."""
    detector_classes = {
        cls
        for module in (models, models.base)
        for name, cls in vars(module).items()
        # On Python 3.10 aliases such as list[Environment] pass both
        # inspect.isclass and isinstance(..., type), but issubclass rejects them.
        if isinstance(cls, type)
        and not isinstance(cls, types.GenericAlias)
        and not name.startswith("_")
        and issubclass(cls, models.base._DetectorBase)
    }
    assert {"SortedSpikesDetector", "ClusterlessDetector", "SortedSpikesDecoder"} <= {
        cls.__name__ for cls in detector_classes
    }

    unresolved = []
    for cls in sorted(detector_classes, key=lambda cls: cls.__name__):
        for name, member in inspect.getmembers(cls):
            if name.startswith("_") or not inspect.isroutine(member):
                continue
            try:
                typing.get_type_hints(member)
            except Exception as error:
                unresolved.append(f"{cls.__name__}.{name}: {error!r}")
    assert unresolved == []
