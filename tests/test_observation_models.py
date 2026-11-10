"""Tests for the ObservationModel dataclass equality semantics."""

import pytest

from non_local_detector.exceptions import ConfigurationError
from non_local_detector.observation_models import ObservationModel

pytestmark = pytest.mark.unit


def test_observation_model_eq_all_fields():
    """ObservationModel equality compares all four fields, not just (env, group).

    Two models that differ only in ``is_local`` or ``is_no_spike`` must not
    compare equal; that would silently merge distinct decoding states.
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


def test_observation_model_is_immutable_and_round_trips():
    """An observation model is a frozen value: it cannot change after it is
    hashed, and pickling and ``type(obj)(**vars(obj))`` rebuild it."""
    import dataclasses
    import pickle

    model = ObservationModel("track", 1, is_local=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.is_local = False
    assert pickle.loads(pickle.dumps(model)) == model
    assert ObservationModel(**vars(model)) == model
    assert hash(ObservationModel(**vars(model))) == hash(model)
    assert min(ObservationModel("b"), ObservationModel("a")).environment_name == "a"


def test_local_no_spike_observation_model_is_rejected():
    """A state is Local (decoded at the animal's position) or No-Spike (no
    position at all), not both; the combination was accepted and decoded as
    No-Spike while the Local kernel and position reads still applied."""
    with pytest.raises(ConfigurationError, match="is_local and is_no_spike"):
        ObservationModel(is_local=True, is_no_spike=True)


@pytest.mark.parametrize(
    ("is_local", "is_no_spike"), [(False, False), (True, False), (False, True)]
)
def test_meaningful_flag_combinations_construct(is_local, is_no_spike):
    model = ObservationModel(is_local=is_local, is_no_spike=is_no_spike)
    assert (model.is_local, model.is_no_spike) == (is_local, is_no_spike)
