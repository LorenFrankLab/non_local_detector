"""Progress-bar labels of plain decoders and the one-state transition plot."""

import matplotlib as mpl
import numpy as np
import pytest

from non_local_detector import SortedSpikesDecoder, time_edges_from_centers
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

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


def test_decoder_predict_progress_bar_is_not_labeled_non_local(
    fitted_decoder_and_data, capfd
):
    decoder, time, position, spike_times = fitted_decoder_and_data
    capfd.readouterr()
    decoder.predict(
        spike_times,
        time_edges=time_edges_from_centers(time),
        position=position,
        position_time=time,
    )
    stderr = capfd.readouterr().err
    assert "Likelihood" in stderr
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
