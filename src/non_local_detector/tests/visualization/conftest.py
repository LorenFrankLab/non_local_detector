"""Fixtures for visualization smoke tests.

The visualization functions consume fitted detectors and the ``xr.Dataset``
returned by ``predict``/``estimate_parameters``. These fixtures build the
smallest synthetic inputs that still exercise the real plotting code paths:

- ``multiunit_inputs`` — spike-time list + time grid (no model needed).
- ``fitted_2d_decoder`` — a 2D ``SortedSpikesDecoder`` fit on a tiny diagonal
  trajectory, plus its posterior and a matching ``position_info`` frame; used
  by the movie writer which needs ``x_position``/``y_position`` posteriors.
- ``fitted_nonlocal_1d`` — a 1D ``NonLocalSortedSpikesDetector`` fit on the
  shared simulator; used by ``plot_non_local_model`` which requires the
  Non-Local states and per-state probabilities.

The detector fits are module-scoped so the cost is paid once per test module.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.models import SortedSpikesDecoder
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture
def multiunit_inputs():
    """Spike times and a time grid for multiunit-firing-rate tests.

    Returns
    -------
    dict
        Contains ``spike_times`` (list of arrays) and ``time`` (array).
    """
    time = np.linspace(0.0, 1.0, 50)
    spike_times = [
        np.array([0.1, 0.3, 0.31, 0.7]),
        np.array([0.2, 0.5, 0.85]),
        np.array([0.05, 0.9]),
    ]
    return {"spike_times": spike_times, "time": time}


@pytest.fixture(scope="module")
def fitted_2d_decoder():
    """Fit a tiny 2D decoder and return inputs for the movie writer.

    Returns
    -------
    dict
        Contains the fitted ``classifier``, decoding ``results``,
        ``position_info`` (with head x/y position and orientation columns),
        ``spike_times`` and ``sampling_frequency``.
    """
    rng = np.random.default_rng(0)
    n_time = 30
    sampling_frequency = 100.0
    position_time = np.arange(n_time) / sampling_frequency
    x = np.linspace(2.0, 18.0, n_time)
    y = np.linspace(2.0, 18.0, n_time)
    position = np.column_stack([x, y])
    spike_times = [np.sort(rng.uniform(0.0, position_time[-1], 6)) for _ in range(3)]

    env = Environment(
        environment_name="",
        place_bin_size=5.0,
        position_range=((0.0, 20.0), (0.0, 20.0)),
    )
    classifier = SortedSpikesDecoder(environments=[env])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(position_time, position, spike_times)
        results = classifier.predict(
            spike_times=spike_times,
            time=position_time,
            position=position,
            position_time=position_time,
        )

    position_info = pd.DataFrame(
        {
            "head_position_x": x,
            "head_position_y": y,
            "head_orientation": np.zeros(n_time),
        },
        index=position_time,
    )

    return {
        "classifier": classifier,
        "results": results,
        "position_info": position_info,
        "spike_times": spike_times,
        "sampling_frequency": sampling_frequency,
    }


@pytest.fixture(scope="module")
def fitted_nonlocal_1d():
    """Fit a 1D Non-Local sorted-spikes detector on simulated data.

    Returns
    -------
    dict
        Contains ``detector``, decoding ``results``, ``position_time``,
        ``position``, ``spike_times`` and ``speed``.
    """
    (
        speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _,
    ) = make_simulated_data(seed=42, n_neurons=4)

    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time=time,
            is_training=~is_event,
            max_iter=1,
        )

    return {
        "detector": detector,
        "results": results,
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "speed": speed,
    }
