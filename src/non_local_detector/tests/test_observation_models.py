"""Tests for the ObservationModel dataclass equality semantics."""

from non_local_detector.observation_models import ObservationModel


def test_observation_model_eq_all_fields():
    """ObservationModel equality compares all four fields, not just (env, group).

    Before the custom ``__eq__`` was removed, two models that differed only in
    ``is_local`` or ``is_no_spike`` were considered equal, silently merging
    distinct decoding states.
    """
    a = ObservationModel("a", 1, is_local=True)
    b = ObservationModel("a", 1, is_local=False)
    assert a != b

    c = ObservationModel("a", 1, is_no_spike=True)
    d = ObservationModel("a", 1, is_no_spike=False)
    assert c != d

    # Same fields still compare equal
    e = ObservationModel("a", 1, is_local=True, is_no_spike=False)
    f = ObservationModel("a", 1, is_local=True, is_no_spike=False)
    assert e == f


def test_observation_model_in_set_preserves_distinctions():
    """Putting ObservationModels in a set keeps them distinct on all fields."""
    models = {
        ObservationModel("a", 1, is_local=True),
        ObservationModel("a", 1, is_local=False),
    }
    assert len(models) == 2
