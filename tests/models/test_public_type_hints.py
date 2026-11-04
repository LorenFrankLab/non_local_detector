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

DETECTOR_CLASSES = sorted(
    {
        cls
        for module in (models, models.base)
        for name, cls in vars(module).items()
        # On Python 3.10 aliases such as list[Environment] pass both
        # inspect.isclass and isinstance(..., type), but issubclass rejects them.
        if isinstance(cls, type)
        and not isinstance(cls, types.GenericAlias)
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


MIXIN_MODULES = (
    models._em,
    models._checkpointed,
    models._input_validation,
    models._persistence,
)
MIXIN_FUNCTIONS = [
    (module, f"{cls.__name__}.{attr}", value)
    for module in MIXIN_MODULES
    for cls in vars(module).values()
    # Mixin classes defined in the module (not imported into it); see above for
    # the Python 3.10 generic-alias guard.
    if isinstance(cls, type)
    and not isinstance(cls, types.GenericAlias)
    and cls.__module__ == module.__name__
    for attr, value in vars(cls).items()
    if inspect.isfunction(value)
]


def test_mixin_methods_are_found():
    """The parametrization covers private methods of all four mixin modules."""
    assert {module.__name__ for module, _, _ in MIXIN_FUNCTIONS} == {
        module.__name__ for module in MIXIN_MODULES
    }
    assert "_EMMixin._run_em" in {name for _, name, _ in MIXIN_FUNCTIONS}


@pytest.mark.parametrize(
    "function",
    [function for _, _, function in MIXIN_FUNCTIONS],
    ids=[
        f"{module.__name__.rsplit('.', 1)[-1]}.{name}"
        for module, name, _ in MIXIN_FUNCTIONS
    ],
)
def test_mixin_method_type_hints_resolve(function):
    """``self: "non_local_detector.models.base._DetectorBase"`` resolves too."""
    typing.get_type_hints(function)
