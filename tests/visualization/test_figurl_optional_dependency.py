"""The figurl views are always importable and fail clearly without ``sortingview``."""

import importlib.util
import sys
import types

import numpy as np
import pytest

import non_local_detector.visualization as viz
from non_local_detector.exceptions import ValidationError

pytestmark = pytest.mark.unit


def _load_figurl_2d_with_stub_sortingview(monkeypatch):
    """Execute a private copy of ``figurl_2D`` against a stub ``sortingview``."""
    for name in ("sortingview", "sortingview.views", "sortingview.views.franklab"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["sortingview.views.franklab"].TrackPositionAnimationV1 = type(
        "TrackPositionAnimationV1", (), {}
    )
    spec = importlib.util.find_spec("non_local_detector.visualization.figurl_2D")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_NAMES = [
    "create_interactive_1D_decoding_figurl",
    "create_interactive_2D_decoding_figurl",
]


@pytest.mark.parametrize("name", _NAMES)
def test_missing_sortingview_raises_install_hint(monkeypatch, name):
    for key in list(sys.modules):
        if key.startswith("sortingview"):
            monkeypatch.delitem(sys.modules, key)
    monkeypatch.setitem(sys.modules, "sortingview", None)  # import -> not found
    with pytest.raises(ImportError, match=r"non_local_detector\[viz\]"):
        getattr(viz, name)


def test_other_missing_module_is_not_masked(monkeypatch, tmp_path):
    """A missing transitive dependency is not reported as a missing extra."""
    package = tmp_path / "sortingview"
    package.mkdir()
    (package / "__init__.py").write_text("import _nld_absent_transitive_dep\n")
    for key in list(sys.modules):
        if key.startswith("sortingview"):
            monkeypatch.delitem(sys.modules, key)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="_nld_absent_transitive_dep"):
        getattr(viz, "create_interactive_2D_decoding_figurl")  # noqa: B009


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        getattr(viz, "not_a_real_name")  # noqa: B009


def test_names_resolve_with_sortingview_present(monkeypatch):
    _load_figurl_2d_with_stub_sortingview(monkeypatch)  # installs the stub
    module_name = "non_local_detector.visualization.figurl_2D"
    # Register the module and the package attribute with monkeypatch first, so
    # the stub-bound copy the lazy import creates is undone after the test.
    monkeypatch.setitem(sys.modules, module_name, None)
    del sys.modules[module_name]
    monkeypatch.setattr(viz, "figurl_2D", None, raising=False)
    delattr(viz, "figurl_2D")

    function = viz.create_interactive_2D_decoding_figurl

    assert function is sys.modules[module_name].create_interactive_2D_decoding_figurl
    assert function.__module__ == module_name


def test_dir_lists_figurl_names():
    assert set(_NAMES) <= set(dir(viz))


def test_length_mismatch_raises_validation_error(monkeypatch):
    figurl_2D = _load_figurl_2d_with_stub_sortingview(monkeypatch)
    position = np.zeros((5, 2))
    with pytest.raises(ValidationError, match="position_time and position"):
        figurl_2D.create_2D_decode_view(
            position_time=np.zeros(4),
            position=position,
            interior_place_bin_centers=np.zeros((3, 2)),
            place_bin_size=(1.0, 1.0),
            posterior=None,
        )

    class _Post:
        shape = (4, 3)

    with pytest.raises(ValidationError, match="posterior and position") as ei:
        figurl_2D.create_2D_decode_view(
            position_time=np.zeros(5),
            position=position,
            interior_place_bin_centers=np.zeros((3, 2)),
            place_bin_size=(1.0, 1.0),
            posterior=_Post(),
        )
    assert "Expected" in str(ei.value)
    assert "Got" in str(ei.value)
