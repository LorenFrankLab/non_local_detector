from .base import DiscreteTransitionModel

_TRANSITIONS: dict[str, type[DiscreteTransitionModel]] = {}


def register_discrete_transition(name: str):
    def _wrap(cls: type[DiscreteTransitionModel]) -> type[DiscreteTransitionModel]:
        if name in _TRANSITIONS:
            raise ValueError(f"Discrete transition '{name}' already registered.")
        _TRANSITIONS[name] = cls
        return cls

    return _wrap


def get_discrete_transition(name: str, **kwargs) -> DiscreteTransitionModel:
    try:
        cls = _TRANSITIONS[name]
    except KeyError:
        raise ValueError(f"No discrete transition named '{name}'.")
    return cls(**kwargs)


def list_all_discrete_transitions() -> list[str]:
    return sorted(_TRANSITIONS)
