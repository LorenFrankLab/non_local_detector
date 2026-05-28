"""Smoke tests for ``visualization.static``.

These exercise the public plotting/utility functions on small synthetic
inputs and assert the documented return type. They do not check pixel
output (fragile across matplotlib versions); they guard against the
plotting code paths raising on a realistic detector + posterior.
"""

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from non_local_detector.visualization.static import (  # noqa: E402
    get_multiunit_firing_rate,
)


def test_get_multiunit_firing_rate_returns_dataframe(multiunit_inputs):
    """Returns a per-time-bin firing-rate DataFrame with finite values."""
    result = get_multiunit_firing_rate(
        multiunit_inputs["spike_times"], multiunit_inputs["time"]
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["firing_rate"]
    assert len(result) == len(multiunit_inputs["time"])
    assert np.all(np.isfinite(result["firing_rate"].to_numpy()))
    # A nonzero number of spikes must yield a nonzero rate somewhere.
    assert np.any(result["firing_rate"].to_numpy() > 0)


def test_get_multiunit_firing_rate_no_spikes_is_zero(multiunit_inputs):
    """With no spikes the firing rate is identically zero everywhere."""
    empty_spike_times = [np.array([]) for _ in multiunit_inputs["spike_times"]]
    result = get_multiunit_firing_rate(empty_spike_times, multiunit_inputs["time"])

    assert np.allclose(result["firing_rate"].to_numpy(), 0.0)


@pytest.mark.slow
@pytest.mark.integration
def test_plot_non_local_model_runs_without_exception(fitted_nonlocal_1d):
    """``plot_non_local_model`` builds its figure and returns ``None``.

    The function draws into a freshly created figure (4 stacked axes) and
    returns nothing per its docstring; we assert it produced an active
    figure with the expected number of axes.
    """
    plt.close("all")
    ret = get_static_plot(fitted_nonlocal_1d)

    assert ret is None
    fig = plt.gcf()
    assert len(fig.axes) == 4
    plt.close("all")


def get_static_plot(data):
    """Call ``plot_non_local_model`` on the fitted-detector fixture."""
    from non_local_detector.visualization.static import plot_non_local_model

    return plot_non_local_model(
        position_time=data["position_time"],
        position=data["position"],
        spike_times=data["spike_times"],
        speed=data["speed"],
        detector=data["detector"],
        results=data["results"],
    )
