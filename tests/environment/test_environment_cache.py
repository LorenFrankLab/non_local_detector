"""The declared derived-data cache on ``Environment``."""

import pickle

import numpy as np
import pytest

from non_local_detector.environment import Environment

pytestmark = pytest.mark.unit


@pytest.fixture
def environment():
    return Environment(place_bin_size=1.0).fit_place_grid(
        np.linspace(0.0, 10.0, 50)[:, None]
    )


def test_set_cached_returns_value_and_get_cached_reads_it(environment):
    value = {"a": 1}
    assert environment.get_cached("key") is None
    assert environment.set_cached("key", value) is value
    assert environment.get_cached("key") is value


def test_fit_place_grid_clears_every_entry(environment):
    environment.set_cached("persistent", 1)
    environment.set_cached("transient", 2, transient=True)
    environment.fit_place_grid(np.linspace(0.0, 20.0, 50)[:, None])
    assert environment.get_cached("persistent") is None
    assert environment.get_cached("transient") is None


def test_pickle_keeps_persistent_entries_and_drops_transient_ones(environment):
    environment.set_cached("persistent", np.arange(3))
    environment.set_cached("transient", object(), transient=True)
    restored = pickle.loads(pickle.dumps(environment))
    np.testing.assert_array_equal(restored.get_cached("persistent"), np.arange(3))
    assert restored.get_cached("transient") is None
    # Pickling does not change the original.
    assert environment.get_cached("transient") is not None


def test_cache_is_not_a_constructor_argument_or_in_repr(environment):
    with pytest.raises(TypeError):
        Environment(_cache={})
    environment.set_cached("key", 1)
    assert "_cache" not in repr(environment)


def test_unpickling_state_saved_without_a_cache_gives_an_empty_cache(environment):
    state = dict(vars(environment))
    del state["_cache"]
    restored = Environment.__new__(Environment)
    restored.__setstate__(state)
    assert restored.get_cached("key") is None
    restored.set_cached("key", 1)
    assert restored.get_cached("key") == 1
