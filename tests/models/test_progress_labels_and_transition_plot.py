"""Progress-bar labels of plain decoders and the one-state transition plot."""

import matplotlib as mpl
import numpy as np
import pytest

from non_local_detector import SortedSpikesDecoder, time_edges_from_centers
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from tests.models._helpers import ALGORITHMS, _decoder

mpl.use("Agg")
import matplotlib.pyplot as plt

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def fitted_decoder_and_data():
    _, position, spike_times, time, *_ = make_simulated_data(seed=0, n_neurons=5)
    n = 1_000
    time, position = time[:n], position[:n]
    spike_times = [st[st <= time[-1]] for st in spike_times]
    decoder = SortedSpikesDecoder().fit(time, position, spike_times)
    return decoder, time, position, spike_times


# Diffusion fits with the default position_std on these 5-unit bins; the
# smoothing width is not what this test checks.
@pytest.mark.filterwarnings(
    r"ignore:position_std \(.*\) is below the characteristic bin spacing:UserWarning"
)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_decoder_predict_progress_bar_is_not_labeled_non_local(algorithm, capfd):
    """A decoder's spatial likelihood is labeled by position bin, never Non-Local."""
    family = "sorted_spikes" if algorithm.startswith("sorted") else "clusterless"
    decoder = _decoder(algorithm)
    params = decoder.get_params()[f"{family}_algorithm_params"]
    decoder.set_params(
        **{f"{family}_algorithm_params": {**params, "disable_progress_bar": False}}
    )
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    data = [[np.array([0.021, 0.067, 0.121, 0.167])]]
    if family == "clusterless":
        data.append([np.array([[0.1], [-0.1], [0.2], [-0.2]])])
    decoder.fit(time, position, *data)
    capfd.readouterr()
    decoder.predict(
        *data,
        time_edges=time_edges_from_centers(time),
        position=position,
        position_time=time,
    )
    stderr = capfd.readouterr().err
    assert "Position-Bin Likelihood" in stderr
    assert "Non-Local" not in stderr


def test_single_state_transition_plot_is_readable(fitted_decoder_and_data):
    decoder, *_ = fitted_decoder_and_data
    assert decoder.n_discrete_states_ == 1
    decoder.plot_continuous_state_transition()
    fig = plt.gcf()
    try:
        (ax,) = fig.axes
        width, height = fig.get_size_inches()
        assert width >= 4.0
        assert height >= 3.5
        assert ax.get_xlabel() == "To position bin"
        assert ax.get_ylabel() == "From position bin"
        n_bins = decoder.continuous_state_transitions_.shape[0]
        mesh = ax.collections[0]
        np.testing.assert_array_equal(
            np.asarray(mesh.get_array()).reshape(n_bins, n_bins),
            decoder.continuous_state_transitions_,
        )
    finally:
        plt.close(fig)
