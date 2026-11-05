"""The registry of types a saved model may contain.

A saved model names each object it holds by a stable type name, never by its
module or class. Loading resolves names only through this registry, so a file
cannot make the reader import or call anything else. The package registers its
own types under fixed ``nld.`` names on first use; other packages register theirs
with :func:`register_type`.
"""

import contextlib
import contextvars
import copy
import dataclasses
import warnings
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from non_local_detector.exceptions import ValidationError

ToState = Callable[[Any], Mapping[str, Any]]
FromState = Callable[[dict[str, Any]], Any]

_RESERVED_PREFIX = "nld."
_PACKAGE = "non_local_detector"


@dataclass(frozen=True)
class TypeSpec:
    """How objects of one registered type are saved and loaded.

    Attributes
    ----------
    name : str
        The stable name a file stores.
    cls : type
    to_state : callable or None
        ``to_state(obj) -> dict``; None for detectors, whose state the package
        writes.
    from_state : callable or None
        ``from_state(state) -> obj``; None for detectors.
    owner : str
        The package that registers the type, named in errors.
    block_needs_fit_data : bool
        For a continuous transition: its ``make_state_transition`` reads the
        fit data (``TransitionContext``), so the saved model stores its block
        instead of rebuilding it.
    fields : tuple of str or None
        The stored fields a reader passes ``from_state``; None for a type
        with its own ``from_state``, which gets all of them.
    """

    name: str
    cls: type
    to_state: ToState | None
    from_state: FromState | None
    owner: str
    block_needs_fit_data: bool = False
    fields: tuple[str, ...] | None = None

    def readable_fields(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the stored fields of ``state`` this version of the type reads.

        A field not in ``fields``, written by a newer version of the type, is
        left undecoded: it may hold a type this version lacks. It is ignored
        with a warning (``ignore_unread_names``).

        Parameters
        ----------
        state : Mapping[str, Any]
            The stored state, as JSON nodes.

        Returns
        -------
        state : Mapping[str, Any]
            ``state`` without the ignored fields.
        """
        if self.fields is None:
            return state
        return readable_names(f"{self.cls.__qualname__} fields", state, self.fields)


# The names of a saved model this version does not read, collected while one
# model loads (see ``ignoring_unread_names``).
_UNREAD: contextvars.ContextVar[list[str]] = contextvars.ContextVar("_UNREAD")


def ignore_unread_names(where: str, names: Iterable[str]) -> None:
    """Note names of a saved model that this version does not read.

    A newer version of the package wrote them; loading ignores them and
    warns once (``ignoring_unread_names``, which every reader runs in).

    Parameters
    ----------
    where : str
        The document or object holding them, for the warning.
    names : iterable of str
        The names; none means nothing to note.
    """
    names = sorted(names)
    if names:
        _UNREAD.get().append(f"{where} {names}")


def readable_names(
    where: str, node: Mapping[str, Any], names: Collection[str]
) -> dict[str, Any]:
    """Return ``node`` with only ``names``, the ones this version reads.

    The others are left undecoded, as they may hold a type this version
    lacks, and ignored with a warning (``ignore_unread_names``).

    Parameters
    ----------
    where : str
        The document or object ``node`` is, for the warning.
    node : Mapping[str, Any]
        The stored names and their JSON nodes.
    names : Collection[str]
        The names this version reads.

    Returns
    -------
    node : dict[str, Any]
        ``node`` without the ignored names.
    """
    ignore_unread_names(where, set(node) - set(names))
    return {name: node[name] for name in node if name in names}


@contextlib.contextmanager
def ignoring_unread_names(package_version: str | None) -> Iterator[None]:
    """Warn once, after loading a saved model, of the names it ignored.

    Parameters
    ----------
    package_version : str or None
        The version that wrote the file, named in the warning.
    """
    collected: list[str] = []
    token = _UNREAD.set(collected)
    try:
        yield
    finally:
        _UNREAD.reset(token)
    if collected:
        _warn_unread(collected, package_version)


def _warn_unread(entries: list[str], package_version: str | None) -> None:
    from non_local_detector._warnings import user_stacklevel
    from non_local_detector.exceptions import DataQualityWarning

    writer = (
        f"non_local_detector {package_version}"
        if package_version
        else "a newer version of non_local_detector"
    )
    warnings.warn(
        f"The saved model holds names this version does not read, which "
        f"were ignored: {'; '.join(entries)}. It was written by {writer}, a "
        "newer version; upgrade to use them.",
        DataQualityWarning,
        stacklevel=user_stacklevel(),
    )


_BY_NAME: dict[str, TypeSpec] = {}
_BY_CLASS: dict[type, TypeSpec] = {}
_BUILTINS_REGISTERED = False


def read_fields(
    state: Mapping[str, Any],
    names: tuple[str, ...],
    defaults: Mapping[str, Any],
    cls: type,
) -> dict[str, Any]:
    """Return the ``names`` fields of a stored state, defaulting optional ones.

    A field added to a type after files were written is absent from those
    files; ``defaults`` gives its value then (a copy).

    Raises
    ------
    DataError
        If a field without a default is missing.
    """
    from non_local_detector.exceptions import DataError

    values = {}
    for name in names:
        if name in state:
            values[name] = state[name]
        elif name in defaults:
            values[name] = copy.deepcopy(defaults[name])
        else:
            msg = f"The saved {cls.__qualname__} has no field {name!r}"
            raise DataError(
                msg,
                hint="The file is corrupt, or the field needs a default "
                "(register_type's defaults) to read files written before it.",
            )
    return values


def _dataclass_state(
    cls: type, names: tuple[str, ...], defaults: Mapping[str, Any]
) -> tuple[ToState, FromState]:
    """``to_state`` and ``from_state`` of a dataclass built from ``names``."""
    unknown = sorted(set(defaults) - set(names))
    if unknown:
        msg = f"defaults names fields {cls.__qualname__} does not store: {unknown}"
        raise ValidationError(msg, expected=f"keys among {list(names)}")

    def to_state(obj: Any) -> dict[str, Any]:
        return {name: getattr(obj, name) for name in names}

    def from_state(state: dict[str, Any]) -> Any:
        return cls(**read_fields(state, names, defaults, cls))

    return to_state, from_state


def _add(spec: TypeSpec) -> None:
    existing = _BY_NAME.get(spec.name)
    if existing is not None and existing.cls is not spec.cls:
        msg = f"The type name {spec.name!r} is already registered"
        raise ValidationError(
            msg,
            got=f"{existing.cls.__module__}.{existing.cls.__qualname__}",
            hint="Choose another name, prefixed with your package's name.",
        )
    by_class = _BY_CLASS.get(spec.cls)
    if by_class is not None and by_class.name != spec.name:
        msg = f"{spec.cls.__qualname__} is already registered as {by_class.name!r}"
        raise ValidationError(msg, hint="A class has one stable name.")
    _BY_NAME[spec.name] = spec
    _BY_CLASS[spec.cls] = spec


def register_type(
    cls: type,
    name: str,
    *,
    to_state: ToState | None = None,
    from_state: FromState | None = None,
    owner: str | None = None,
    block_needs_fit_data: bool = False,
    defaults: Mapping[str, Any] | None = None,
) -> None:
    """Let ``save_model`` save, and ``load_model`` load, objects of ``cls``.

    Register a custom transition, initial condition or observation-model class
    (anything a detector's parameters hold), or a detector subclass, once per
    session, before saving or loading a model that contains it, for example
    when your package is imported. The registration is exact: a subclass of a
    registered class is registered separately.

    Parameters
    ----------
    cls : type
        The class.
    name : str
        A stable name, stored in files instead of the module and class name,
        prefixed with your package's name (``"mylab.SpeedTransition"``).
        Keep it when you rename or move the class. Names starting with
        ``"nld."`` belong to non_local_detector.
    to_state : callable, optional
        ``to_state(obj) -> dict``: the object's state, a dict with string keys
        whose values are None, bool, int, float, str, NumPy or JAX arrays and
        scalars, lists, tuples and dicts of these, or registered objects. By
        default, for a dataclass, its ``__init__`` fields. Not allowed for a
        detector subclass, whose fitted state the package saves.
    from_state : callable, optional
        ``from_state(state) -> obj``, the inverse of ``to_state``. By default,
        for a dataclass, ``cls(**state)``. To change what ``to_state``
        returns, keep ``from_state`` reading the states earlier versions of
        your package wrote.
    owner : str, optional
        The package to import to register ``cls``, named when a file needs
        the type but it is not registered. By default the top-level package
        of ``cls``.
    block_needs_fit_data : bool, optional
        For a continuous transition whose ``make_state_transition`` reads the
        fit data (its ``context``), as ``EmpiricalMovement`` does: the saved
        model then stores its transition block, since the fit data are not
        saved. By default False: the block is rebuilt from the transition and
        the environments on load, and checked against a fingerprint before
        decoding.
    defaults : Mapping[str, Any], optional
        For a dataclass registered without ``to_state``/``from_state``: the
        value of each optional field when a stored state lacks it, so a
        field added later reads files written before it. Other fields
        missing from a state raise ``DataError``; a field the dataclass
        does not store (written by a newer version of it) is ignored with a
        ``DataQualityWarning``.

    Raises
    ------
    ValidationError
        If ``name`` is not a prefixed name, is reserved, or is registered to
        another class; if ``cls`` is registered under another name; if a
        non-dataclass is registered without both functions, or a detector
        with either.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from non_local_detector.model_format import register_type
    >>> @dataclass
    ... class SpeedTransition:
    ...     environment_name: str = ""
    ...     speed_scale: float = 1.0
    >>> register_type(SpeedTransition, "mylab.SpeedTransition")
    """
    _register_builtins()
    if "." not in name.strip(".") or " " in name:
        msg = f"Type name {name!r} must be prefixed with a package name"
        raise ValidationError(msg, example='"mylab.SpeedTransition"')
    if name.startswith(_RESERVED_PREFIX):
        msg = f"Type names starting with {_RESERVED_PREFIX!r} are reserved"
        raise ValidationError(msg, hint="Prefix the name with your package's name.")
    _register(
        cls,
        name,
        to_state=to_state,
        from_state=from_state,
        owner=owner or cls.__module__.split(".")[0],
        block_needs_fit_data=block_needs_fit_data,
        defaults=defaults,
    )


def _register(
    cls: type,
    name: str,
    *,
    to_state: ToState | None,
    from_state: FromState | None,
    owner: str,
    block_needs_fit_data: bool = False,
    fields: tuple[str, ...] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> None:
    from non_local_detector.models.base import _DetectorBase

    if defaults and (
        issubclass(cls, _DetectorBase) or to_state is not None or from_state is not None
    ):
        msg = "defaults apply only to a dataclass registered without to_state or from_state"
        raise ValidationError(msg, hint="Supply missing fields in from_state instead.")
    if issubclass(cls, _DetectorBase):
        if to_state is not None or from_state is not None:
            msg = "A detector class is registered without to_state or from_state"
            raise ValidationError(
                msg, hint="non_local_detector saves a detector's fitted state."
            )
    elif to_state is None and from_state is None:
        if not dataclasses.is_dataclass(cls):
            msg = f"{cls.__qualname__} is not a dataclass"
            raise ValidationError(
                msg, hint="Pass to_state and from_state to register it."
            )
        if fields is None:
            fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
        to_state, from_state = _dataclass_state(cls, fields, defaults or {})
    elif to_state is None or from_state is None:
        msg = "Pass both to_state and from_state, or neither (for a dataclass)"
        raise ValidationError(msg)
    _add(TypeSpec(name, cls, to_state, from_state, owner, block_needs_fit_data, fields))


def unregistered_type_error(cls: type, where: str) -> ValidationError:
    """Return the error ``save_model`` raises for an object of an unregistered type."""
    qualified = f"{cls.__module__}.{cls.__qualname__}"
    msg = f"Cannot save the {qualified} at {where}: its type is not registered"
    return ValidationError(
        msg,
        hint=(
            "Register it once per session with "
            "non_local_detector.model_format.register_type("
            f"{cls.__qualname__}, 'yourpackage.{cls.__qualname__}'), passing "
            "to_state and from_state unless it is a dataclass whose __init__ "
            "fields are its state. A subclass of a registered class is "
            "registered separately."
        ),
    )


def spec_for_object(obj: Any) -> TypeSpec | None:
    """Return the spec of ``type(obj)``, or None if it is not registered."""
    _register_builtins()
    return _BY_CLASS.get(type(obj))


def spec_for_class(cls: type) -> TypeSpec | None:
    """Return the spec of ``cls``, or None if it is not registered."""
    _register_builtins()
    return _BY_CLASS.get(cls)


def spec_for_name(name: str, owner: str | None = None) -> TypeSpec:
    """Return the spec registered as ``name``.

    Parameters
    ----------
    name : str
        Registered name of the type.
    owner : str, optional
        The registering package the file recorded, named in the error.

    Raises
    ------
    ValidationError
        If no type of that name is registered in this session.
    """
    _register_builtins()
    spec = _BY_NAME.get(name)
    if spec is None:
        msg = f"The saved model holds a {name!r}, a type not registered in this session"
        package = owner or name.split(".")[0]
        raise ValidationError(
            msg,
            hint=(
                f"Import the package that registers it ({package!r}) before "
                "load_model; it calls non_local_detector.model_format."
                "register_type."
            )
            if not name.startswith(_RESERVED_PREFIX)
            else "The file was written by a newer non_local_detector; upgrade to read it.",
        )
    return spec


def registered_types() -> dict[str, type]:
    """Every registered type name and its class.

    Returns
    -------
    types : dict[str, type]
    """
    _register_builtins()
    return {name: spec.cls for name, spec in _BY_NAME.items()}


def _register_builtins() -> None:
    """Register the package's own types, once."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
    from non_local_detector.model_format._builtins import builtin_types

    for name, cls, fields, options in builtin_types():
        _register(
            cls,
            name,
            to_state=options.get("to_state"),
            from_state=options.get("from_state"),
            owner=_PACKAGE,
            block_needs_fit_data=options.get("block_needs_fit_data", False),
            fields=fields,
            defaults=options.get("defaults"),
        )
