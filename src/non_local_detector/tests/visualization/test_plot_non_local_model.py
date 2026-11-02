"""``plot_non_local_model`` draws the posterior given the non-local states."""

import matplotlib as mpl
import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.visualization.static import plot_non_local_model

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
        state_names=["Here", "Quiet", "Replay-Smooth", "Replay-Jump"]
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
    plt.close("all")

    expected = detector.get_conditional_non_local_posterior(
        results.sel(time=time_slice)
    ).values
    np.testing.assert_allclose(drawn.reshape(expected.T.shape), expected.T)
    assert np.isfinite(drawn).all()
