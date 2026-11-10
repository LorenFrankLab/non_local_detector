"""``plot_non_local_model`` draws the posterior given the non-local states."""

from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.models.base import ClusterlessDetector, SortedSpikesDetector
from non_local_detector.observation_models import ObservationModel
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.visualization.static import _unit_order, plot_non_local_model
from tests.models._helpers import SORTED_SIM_RATE

mpl.use("Agg")
import matplotlib.pyplot as plt

pytestmark = pytest.mark.integration


def test_posterior_panel_uses_non_local_states_whatever_their_names():
    """With renamed states the posterior panel shows the posterior given the
    two non-local states (previously found by "Non-Local" in the name)."""
    speed, position, spike_times, time, *_ = make_simulated_data(seed=0, n_neurons=5)
    n = 2_000
    time, position, speed = time[:n], position[:n], speed[:n]
    spike_times = [st[st <= time[-1]] for st in spike_times]
    detector = NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_SIM_RATE,
        state_names=["Here", "Quiet", "Replay-Smooth", "Replay-Jump"],
    )
    detector.fit(time, position, spike_times)
    results = detector.predict(
        spike_times,
        time_edges=time_edges_from_centers(time),
        position=position,
        position_time=time,
    )
    time_slice = slice(float(time[100]), float(time[1_000]))

    plot_non_local_model(
        time,
        position,
        spike_times,
        speed,
        detector,
        results,
        time_slice=time_slice,
    )
    drawn = np.asarray(plt.gcf().axes[2].collections[0].get_array())
    time_label = plt.gcf().axes[-1].get_xlabel()
    plt.close("all")

    expected = detector.get_conditional_non_local_posterior(
        results.sel(time=time_slice)
    ).values
    np.testing.assert_allclose(drawn.reshape(expected.T.shape), expected.T)
    assert np.isfinite(drawn).all()
    # results.time is in seconds.
    assert time_label == "Time [s]"


def _stub_detector(cls, observation_models, environments, encoding_model=None):
    """A detector carrying only what ``_unit_order`` reads."""
    detector = cls.__new__(cls)
    detector.state_space_ = SimpleNamespace(observation_models=observation_models)
    detector.environments_ = environments
    detector.encoding_model_ = encoding_model or {}
    return detector


def _env(name, centers):
    return SimpleNamespace(
        environment_name=name, place_bin_centers_=np.asarray(centers)[:, None]
    )


@pytest.mark.unit
def test_unit_order_uses_observation_model_key_and_environment():
    """Neurons sort by place-field peak, using the named environment and
    encoding group of the first non-No-Spike observation model."""
    centers_a = [0.0, 1.0, 2.0]
    centers_b = [10.0, 5.0, 0.0]  # reversed ordering of the same bin indices
    # Peak bins of the three neurons: 2, 0, 1
    place_fields = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
    detector = _stub_detector(
        SortedSpikesDetector,
        [ObservationModel(is_no_spike=True), ObservationModel("B", "g1")],
        [_env("A", centers_a), _env("B", centers_b)],
        {("B", "g1"): {"place_fields": place_fields}},
    )

    order, label = _unit_order(detector, 3)

    # In environment B the peak positions are 0, 10, 5, so the order is 0, 2, 1.
    # Environment A (the first one) would give 1, 2, 0.
    np.testing.assert_array_equal(order, [0, 2, 1])
    assert label == "Neuron"


@pytest.mark.unit
def test_unit_order_clusterless_keeps_electrode_order():
    detector = _stub_detector(
        ClusterlessDetector, [ObservationModel()], [_env("", [0.0, 1.0])]
    )
    order, label = _unit_order(detector, 4)
    np.testing.assert_array_equal(order, np.arange(4))
    assert label == "Electrode\nGroup"


@pytest.mark.unit
def test_unit_order_sorted_with_only_no_spike_models_falls_back():
    detector = _stub_detector(
        SortedSpikesDetector, [ObservationModel(is_no_spike=True)], []
    )
    order, label = _unit_order(detector, 3)
    np.testing.assert_array_equal(order, np.arange(3))
    assert label == "Electrode\nGroup"
