"""``load_environment`` unpickles only when the caller says the file is trusted.

Unpickling can run code from the file, so by default it is refused before the
file is read.
"""

import pickle

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError

pytestmark = pytest.mark.unit

# Every unpickling of a ``_Payload`` appends to this list.
UNPICKLED: list[int] = []


def _record():
    UNPICKLED.append(1)
    return Environment()


class _Payload:
    def __reduce__(self):
        return (_record, ())


@pytest.fixture
def fitted():
    position = np.linspace(0.0, 10.0, 50)[:, np.newaxis]
    environment = Environment(place_bin_size=1.0)
    environment.fit_place_grid(position)
    return environment


def test_a_trusted_file_loads_the_saved_environment(fitted, tmp_path):
    path = str(tmp_path / "environment.pkl")
    fitted.save_environment(path)

    loaded = Environment.load_environment(path, allow_pickle=True)

    np.testing.assert_array_equal(loaded.place_bin_centers_, fitted.place_bin_centers_)
    assert loaded.get_params() == fitted.get_params()


def test_loading_is_refused_without_allow_pickle(tmp_path):
    path = tmp_path / "environment.pkl"
    path.write_bytes(pickle.dumps(_Payload()))
    UNPICKLED.clear()

    with pytest.raises(ValidationError, match="allow_pickle=True"):
        Environment.load_environment(str(path))

    assert UNPICKLED == []
    Environment.load_environment(str(path), allow_pickle=True)
    assert UNPICKLED == [1]


def test_a_file_that_is_not_an_environment_is_refused(tmp_path):
    path = tmp_path / "environment.pkl"
    path.write_bytes(pickle.dumps([1, 2]))

    with pytest.raises(ValidationError, match="holds a list"):
        Environment.load_environment(str(path), allow_pickle=True)
