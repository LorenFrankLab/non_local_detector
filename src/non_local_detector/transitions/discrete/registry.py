_TRANSITIONS: dict[str, type] = {}


def register_discrete_transition(name: str):
    def _wrap(cls):
        _TRANSITIONS[name] = cls
        return cls

    return _wrap


def get_discrete_transition(name: str, **kwargs):
    return _TRANSITIONS[name](**kwargs)


def list_all_discrete_transitions() -> list[str]:
    return sorted(_TRANSITIONS)
