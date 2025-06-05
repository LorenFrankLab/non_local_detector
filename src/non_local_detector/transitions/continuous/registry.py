from typing import Callable, Dict

_KERNELS: Dict[str, Callable] = {}


def register_continuous_transition(name: str) -> Callable:
    def _decorator(cls) -> Callable:
        _KERNELS[name] = cls
        return cls

    return _decorator


def get(name: str, **kwargs) -> Callable:
    try:
        return _KERNELS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown kernel {name!r}") from exc


def list_all() -> list[str]:
    return sorted(_KERNELS)
