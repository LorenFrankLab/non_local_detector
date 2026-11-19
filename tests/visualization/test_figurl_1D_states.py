"""The 1-D figurl decodes the detector's non-local states, found by role.

``figurl_1D`` needs ``sortingview``; the module is executed against a stub
``sortingview`` so the state selection is checked without the dependency.
"""

import importlib.util
import sys
import types

import numpy as np
import pytest

from non_local_detector import SortedSpikesDecoder
from non_local_detector.models.non_local_model import NonLocalSortedSpikesDetector
from tests.conftest import make_state_bins_results

pytestmark = pytest.mark.unit


class _Recorder:
    """Stand-in for a sortingview view: accepts any call, chains, has a url."""

    def __init__(self, *args, **kwargs):
        pass

    def add_line_series(self, **kwargs):
        return self

    def url(self, label):
        return "stub-url"


def _load_figurl_1d_with_stub_sortingview(monkeypatch):
    """Execute a private copy of ``figurl_1D`` against a stub ``sortingview``."""
    sortingview = types.ModuleType("sortingview")
    views = types.ModuleType("sortingview.views")
    franklab = types.ModuleType("sortingview.views.franklab")
    for name in ("TimeseriesGraph", "LayoutItem", "Box"):
        setattr(views, name, _Recorder)
    franklab.DecodedLinearPositionData = _Recorder
    # Referenced in figurl_2D's return annotations, which run on import.
    franklab.TrackPositionAnimationV1 = _Recorder
    sortingview.views = views
    views.franklab = franklab
    monkeypatch.setitem(sys.modules, "sortingview", sortingview)
    monkeypatch.setitem(sys.modules, "sortingview.views", views)
    monkeypatch.setitem(sys.modules, "sortingview.views.franklab", franklab)

    spec = importlib.util.find_spec("non_local_detector.visualization.figurl_1D")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_panel_uses_non_local_states_whatever_their_names(monkeypatch):
    """With renamed states the decode panel still shows the posterior given
    the two non-local states, not the sum of every state."""
    figurl_1D = _load_figurl_1d_with_stub_sortingview(monkeypatch)
    decoded = {}

    def capture_decode_view(posterior, linear_position):
        decoded["posterior"] = posterior
        return _Recorder()

    monkeypatch.setattr(figurl_1D, "create_1D_decode_view", capture_decode_view)

    names = ["Here", "Quiet", "Replay-Smooth", "Replay-Jump"]
    detector = NonLocalSortedSpikesDetector(state_names=names)
    n_time, n_bins = 10, 5
    results = make_state_bins_results(
        names, {"position": np.arange(n_bins, dtype=float)}, n_time=n_time
    )
    per_state = results.acausal_posterior.values.reshape(n_time, len(names), n_bins)
    results["acausal_state_probabilities"] = (
        ("time", "states"),
        per_state.sum(axis=2),
    )
    results = results.assign_coords(states=names)

    url = figurl_1D.create_interactive_1D_decoding_figurl(
        position=np.zeros(n_time),
        speed=np.zeros(n_time),
        spike_times=[np.array([1.0, 4.0])],
        results=results,
        detector=detector,
    )

    non_local = per_state[:, 2] + per_state[:, 3]
    assert url == "stub-url"
    np.testing.assert_allclose(
        decoded["posterior"].values,
        non_local / non_local.sum(axis=1, keepdims=True),
        rtol=1e-12,
    )


def test_probability_panel_shows_the_one_state_of_a_decoder(monkeypatch):
    """A decoder's ``states`` is a scalar and its probabilities have no
    ``states`` dimension; neither is iterated or selected by name."""
    figurl_1D = _load_figurl_1d_with_stub_sortingview(monkeypatch)
    monkeypatch.setattr(
        figurl_1D, "create_1D_decode_view", lambda **kwargs: _Recorder()
    )
    series = []
    monkeypatch.setattr(
        _Recorder,
        "add_line_series",
        lambda self, **kwargs: series.append(kwargs) or self,
    )
    n_time, n_bins = 10, 5
    results = make_state_bins_results(
        ["Continuous"], {"position": np.arange(n_bins, dtype=float)}, n_time=n_time
    )
    results["acausal_state_probabilities"] = ("time", np.ones(n_time))
    results = results.assign_coords(states="Continuous")

    figurl_1D.create_interactive_1D_decoding_figurl(
        position=np.zeros(n_time),
        speed=np.zeros(n_time),
        spike_times=[np.array([1.0, 4.0])],
        results=results,
        detector=SortedSpikesDecoder(),
    )

    (state,) = [entry for entry in series if entry["name"] == "Continuous"]
    np.testing.assert_array_equal(state["y"], np.ones(n_time))


def test_decode_view_is_the_sparse_encoding_of_the_posterior(monkeypatch):
    """Values, bin indices and per-frame counts equal ``np.nonzero`` of the
    uint8 posterior, on bins with a gap between two arms."""
    import xarray as xr

    figurl_1D = _load_figurl_1d_with_stub_sortingview(monkeypatch)

    class _KeepArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        sys.modules["sortingview.views.franklab"],
        "DecodedLinearPositionData",
        _KeepArguments,
    )
    rng = np.random.default_rng(0)
    n_time, n_position = 6, 5
    probs = np.zeros((n_time, n_position))
    for frame in range(n_time):
        n_bins = rng.integers(0, 4)
        chosen = rng.choice(n_position, size=n_bins, replace=False)
        probs[frame, chosen] = rng.dirichlet(np.ones(n_bins)) if n_bins else []
    probs[2] = 0.0  # a frame with no mass
    posterior = xr.DataArray(
        probs,
        dims=["time", "position"],
        coords={
            "time": 3.0 + np.arange(n_time) / 500.0,
            "position": [1.0, 3.0, 5.0, 20.0, 22.0],
        },
    )

    view = figurl_1D.create_1D_decode_view(posterior)

    discrete = (probs * 255).astype(np.uint8)
    frame, position_bin = np.nonzero(discrete)
    np.testing.assert_array_equal(view.kwargs["values"], discrete[frame, position_bin])
    np.testing.assert_array_equal(
        view.kwargs["positions"], position_bin.astype(np.uint16)
    )
    np.testing.assert_array_equal(
        view.kwargs["frame_bounds"],
        np.bincount(frame, minlength=n_time).astype(np.uint8),
    )
    assert view.kwargs["frame_bounds"][2] == 0
