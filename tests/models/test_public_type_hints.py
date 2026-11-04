"""Type hints of the detectors' public methods resolve at runtime.

``typing.get_type_hints`` (used by documentation and introspection tools)
evaluates every annotation, so a name imported only for type checking makes
it raise ``NameError``.
"""

import inspect
import typing

import pytest

from non_local_detector import models

pytestmark = pytest.mark.unit

DETECTOR_CLASSES = sorted(
    {
        cls
        for module in (models, models.base)
        for name, cls in vars(module).items()
        # Not inspect.isclass: on Python 3.10 it is True for aliases such as
        # list[Environment], which issubclass rejects.
        if isinstance(cls, type)
        and not name.startswith("_")
        and issubclass(cls, models.base._DetectorBase)
    },
    key=lambda cls: cls.__name__,
)
PUBLIC_METHODS = [
    (cls, name)
    for cls in DETECTOR_CLASSES
    for name, member in inspect.getmembers(cls)
    if not name.startswith("_") and inspect.isroutine(member)
]


def test_public_detector_classes_are_found():
    """The parametrization covers the base detectors and the ready-made models."""
    assert {"SortedSpikesDetector", "ClusterlessDetector", "SortedSpikesDecoder"} <= {
        cls.__name__ for cls in DETECTOR_CLASSES
    }


@pytest.mark.parametrize(
    ("cls", "name"),
    PUBLIC_METHODS,
    ids=[f"{cls.__name__}.{name}" for cls, name in PUBLIC_METHODS],
)
def test_public_method_type_hints_resolve(cls, name):
    """``get_type_hints`` evaluates every annotation of the method."""
    typing.get_type_hints(getattr(cls, name))
