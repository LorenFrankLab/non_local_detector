from .base import DiscreteTransitionModel

_TRANSITIONS: dict[str, type[DiscreteTransitionModel]] = {}


def register_discrete_transition(name: str):
    """A decorator to register a new discrete transition model."""

    def _decorator(cls: type[DiscreteTransitionModel]) -> type[DiscreteTransitionModel]:
        if name in _TRANSITIONS:
            raise ValueError(f"Discrete transition '{name}' already registered.")

        if not hasattr(cls, "matrix"):
            raise TypeError(f"{cls.__name__} must implement the 'matrix' method.")

        if not hasattr(cls, "update_parameters"):
            raise TypeError(
                f"{cls.__name__} must implement the 'update_parameters' method."
            )

        _TRANSITIONS[name] = cls
        return cls

    return _decorator


def get_discrete_transition(name: str, **kwargs) -> DiscreteTransitionModel:
    try:
        cls = _TRANSITIONS[name]
    except KeyError:
        raise ValueError(f"No discrete transition named '{name}'.")
    return cls(**kwargs)


def list_all_discrete_transitions() -> list[str]:
    return sorted(_TRANSITIONS)
