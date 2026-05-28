"""Smoke tests for ``visualization.figurl_1D``.

``figurl_1D`` wraps its whole body in ``try: ... except ImportError`` so its
public names only exist when the optional ``sortingview`` dependency is
installed. When it is missing these tests skip via ``importorskip``.
"""

import numpy as np
import pytest
import xarray as xr

# figurl_1D defines its functions only when sortingview imports successfully.
pytest.importorskip("sortingview")

from non_local_detector.visualization.figurl_1D import (  # noqa: E402
    create_1D_decode_view,
    discretize_and_trim,
    get_sampling_freq,
)


def _make_1d_posterior(n_time: int = 8, n_position: int = 5) -> xr.DataArray:
    """Build a normalized 1D posterior (one bin per time gets most mass)."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n_position), size=n_time)
    return xr.DataArray(
        probs,
        dims=["time", "position"],
        coords={
            "time": np.arange(n_time, dtype=float) / 100.0,
            "position": np.arange(n_position, dtype=float),
        },
    )


def test_get_sampling_freq_recovers_known_rate():
    """A 100 Hz time grid is recovered as ~100 Hz."""
    times = np.arange(10, dtype=float) / 100.0
    assert get_sampling_freq(times) == pytest.approx(100.0)


def test_discretize_and_trim_drops_zero_mass_bins():
    """Zero-mass bins are removed; only the occupied bins survive.

    Uses a one-hot posterior (each of ``n_time`` rows puts all mass in a single
    position bin) so the surviving count is known exactly: ``n_time`` nonzero
    entries out of the full ``n_time * n_position`` grid. Asserting that the
    output is strictly smaller than the grid verifies the trim actually fired,
    rather than merely restating the ``> 0`` filter.
    """
    n_time, n_position = 8, 5
    probs = np.zeros((n_time, n_position))
    probs[np.arange(n_time), np.arange(n_time) % n_position] = 1.0
    posterior = xr.DataArray(
        probs,
        dims=["time", "position"],
        coords={
            "time": np.arange(n_time, dtype=float) / 100.0,
            "position": np.arange(n_position, dtype=float),
        },
    )

    trimmed = discretize_and_trim(posterior)

    assert trimmed.dtype == np.uint8
    assert np.all(trimmed.values > 0)
    # The (n_position - 1) zero bins per time step were dropped.
    assert trimmed.size == n_time
    assert trimmed.size < n_time * n_position


def test_create_1D_decode_view_returns_decoded_position_data():
    """``create_1D_decode_view`` returns a sortingview DecodedLinearPositionData."""
    import sortingview.views.franklab as vvf

    posterior = _make_1d_posterior()
    linear_position = np.linspace(0.0, 4.0, posterior.sizes["time"])

    view = create_1D_decode_view(posterior=posterior, linear_position=linear_position)

    assert isinstance(view, vvf.DecodedLinearPositionData)
    # The view carries per-frame counts spanning the full time axis.
    assert len(view.frame_bounds) == posterior.sizes["time"]
