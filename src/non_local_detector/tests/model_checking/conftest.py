"""Fixtures for the goodness-of-fit (model_checking) tests.

Provides synthetic spike trains and conditional-intensity functions used by
the time-rescaling and clusterless goodness-of-fit tests. A homogeneous
Poisson process (constant conditional intensity) is the canonical "correct
model" case for the Brown-Barbieri-Ventura-Kass-Frank time-rescaling theorem:
when the model matches the data-generating process, the rescaled interspike
intervals are exponential with mean 1 (uniform after the CDF transform).
"""

import numpy as np
import pytest


@pytest.fixture
def homogeneous_poisson_spike_train():
    """Constant-intensity (homogeneous Poisson) spike train.

    The data-generating intensity matches the model intensity, so the
    rescaled ISIs should be approximately uniform on ``[0, 1]``.

    Returns
    -------
    dict
        ``conditional_intensity`` : ndarray, shape (n_time,)
        ``is_spike`` : bool ndarray, shape (n_time,)
        ``rate`` : float, the constant per-bin intensity.
    """
    rng = np.random.default_rng(0)
    n_time = 4000
    rate = 0.05  # spikes per bin
    conditional_intensity = np.full(n_time, rate)
    is_spike = rng.random(n_time) < rate
    return {
        "conditional_intensity": conditional_intensity,
        "is_spike": is_spike,
        "rate": rate,
    }


@pytest.fixture
def regular_spike_train():
    """Deterministic, evenly spaced spikes with constant intensity.

    Useful for hand-computed expectations: with a constant intensity ``c`` and
    spikes every ``k`` bins, each rescaled ISI integrates to ``c * k`` (using
    the trapezoidal integral of a constant function).

    Returns
    -------
    dict
        ``conditional_intensity`` : ndarray, shape (n_time,)
        ``is_spike`` : bool ndarray, shape (n_time,)
        ``rate`` : float
        ``spacing`` : int, number of bins between spikes.
    """
    n_time = 100
    rate = 0.2
    spacing = 10
    conditional_intensity = np.full(n_time, rate)
    is_spike = np.zeros(n_time, dtype=bool)
    is_spike[spacing::spacing] = True
    return {
        "conditional_intensity": conditional_intensity,
        "is_spike": is_spike,
        "rate": rate,
        "spacing": spacing,
    }


@pytest.fixture
def empty_spike_train():
    """Constant intensity with no spikes (zero-spike edge case)."""
    n_time = 50
    return {
        "conditional_intensity": np.full(n_time, 0.1),
        "is_spike": np.zeros(n_time, dtype=bool),
    }
