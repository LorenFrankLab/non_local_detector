from typing import Callable, Dict

from .base import Kernel

_KERNELS: Dict[str, Callable] = {}


def register_continuous_transition(name: str) -> Callable:
    def _decorator(cls: type[Kernel]) -> type[Kernel]:
        if not issubclass(cls, Kernel):
            raise TypeError(f"{cls.__name__} must inherit from Kernel")
        if name in _KERNELS:
            raise ValueError(f"Kernel {name!r} is already registered")
        if not hasattr(cls, "block"):
            raise TypeError(f"{cls.__name__} must implement the 'block' method")
        if not callable(cls.block):
            raise TypeError(f"{cls.__name__}.block must be a callable method")

        _KERNELS[name] = cls
        return cls

    return _decorator


def get_continuous_transitions(name: str, **kwargs) -> Kernel:
    try:
        return _KERNELS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown kernel {name!r}") from exc


def list_all_continuous_transitions() -> list[str]:
    return sorted(_KERNELS)
