"""Independent tail and weight-scale checks for reference-aware KDE scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from non_local_detector.likelihoods.background_marks import (
    BackgroundMarkModel,
    _kde_log_joint_nonlocal,
    _raw_kde_marks,
    predict_clusterless_with_background,
)
from non_local_detector.likelihoods.common import RATE_EPS_HZ, KDEModel


def _repeated_position_model(x, sample_tile):
    """Different empirical weights describe the same one-point position law."""
    model = {
        "environment": SimpleNamespace(
            place_bin_centers_=np.array([[x]]),
            is_track_interior_=np.array([True]),
            track_graph=None,
        ),
        "occupancy_model": KDEModel(std=np.ones(1)).fit(
            np.zeros((3, 1)), weights=np.array([0.1, 0.01, 0.001])
        ),
        "gpi_models": [KDEModel(std=np.ones(1)).fit(np.zeros((1, 1)))],
        "encoding_positions": [np.zeros((1, 1))],
        "encoding_spike_waveform_features": [np.zeros((1, 1))],
        "encoding_weights": [np.ones(1)],
        "mean_rates": np.ones(1),
        "position_std": np.ones(1),
        "waveform_std": np.ones(1),
        "block_size": 4,
    }
    if sample_tile is not None:
        model["encoding_block_size"] = sample_tile
    return model


@pytest.mark.unit
@pytest.mark.parametrize("x", [200.0, 1457.0])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("sample_tile", [None, 2])
def test_unequal_repeated_position_weights_preserve_mark_density(x, local, sample_tile):
    """Position factors cancel exactly, leaving a standard Gaussian mark law."""
    model = _repeated_position_model(x, sample_tile)
    result = _raw_kde_marks(
        model, 0, np.array([[x]]), np.zeros((1, 1)), local, sample_tile
    )
    assert_allclose(result, -0.5 * np.log(2 * np.pi), atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("x", [200.0, 1457.0])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("sample_tile", [None, 2])
def test_unequal_repeated_position_weights_preserve_silent_ground_rate(
    x, local, sample_tile
):
    """Equal position laws give a constant ground rate plus the background rate."""
    model = _repeated_position_model(x, sample_tile)
    reference = BackgroundMarkModel(np.zeros(1), np.ones(1))
    result = predict_clusterless_with_background(
        "clusterless_kde",
        model,
        np.array([0.0, 1.0]),
        np.full((2, 1), x),
        [np.empty(0)],
        [np.empty((0, 1))],
        np.array([0.0, 1.0]),
        background_mark_models=[reference],
        is_local=local,
    )
    assert_allclose(result, -(1.0 + RATE_EPS_HZ), atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("weight_scale", [1.0, 1e-10])
@pytest.mark.parametrize("sample_tile", [None, 64])
def test_joint_kde_weight_scaling_preserves_small_positive_gemm_mass(
    weight_scale, sample_tile
):
    """A common weight scale cancels even when products lose subnormal mass."""
    a = 65.0
    encoded_positions = np.array(
        [0.0, np.sqrt(2 * a), *([np.sqrt(a)] * 256), 0.0], dtype=np.float32
    )[:, None]
    encoded_marks = np.array(
        [np.sqrt(2 * a), 0.0, *([np.sqrt(a)] * 256), np.sqrt(2 * (a - 8))],
        dtype=np.float32,
    )[:, None]
    weights = np.full(encoded_positions.shape[0], weight_scale, dtype=np.float32)
    result = _kde_log_joint_nonlocal(
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        encoded_positions,
        encoded_marks,
        np.ones(1),
        np.ones(1),
        weights,
        sample_tile,
    )
    # Evaluate the actual float32 source coordinates in double precision. The
    # two Gaussian normalizers and the weighted normalization are analytical.
    log_kernel = -0.5 * (
        encoded_positions[:, 0].astype(float) ** 2
        + encoded_marks[:, 0].astype(float) ** 2
    ) - np.log(2 * np.pi)
    log_weights = np.log(weights.astype(float))
    expected = logsumexp(log_kernel + log_weights) - logsumexp(log_weights)
    assert_allclose(result, expected, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("sample_tile", [None, 1])
def test_far_tail_retains_distinct_source_weight_normalization(local, sample_tile):
    """The shared extreme kernel cancels before distinct normalized masses."""
    x = 200.0
    model = _repeated_position_model(x, sample_tile)
    sources = np.array([[0.0], [1.0]])
    occupancy_weights = np.array([0.05, 0.002])
    spike_weights = np.array([0.2, 0.3])
    model.update(
        occupancy_model=KDEModel(std=np.ones(1)).fit(
            sources, weights=occupancy_weights
        ),
        gpi_models=[KDEModel(std=np.ones(1)).fit(sources, weights=spike_weights)],
        encoding_positions=[sources],
        encoding_spike_waveform_features=[np.array([[-1.0], [1.0]])],
        encoding_weights=[spike_weights],
    )
    # At x=200 the x=0 kernel is smaller by exp(-199.5). Its contribution is
    # negligible, so the exact limiting ratio uses each law's normalized mass
    # at x=1. These masses must survive cancellation of the extreme kernel.
    ground = (spike_weights[1] / spike_weights.sum()) / (
        occupancy_weights[1] / occupancy_weights.sum()
    )
    raw = _raw_kde_marks(
        model, 0, np.array([[x]]), np.ones((1, 1)), local, sample_tile
    )
    assert_allclose(raw, np.log(ground) - 0.5 * np.log(2 * np.pi), atol=1e-5)
    silent = predict_clusterless_with_background(
        "clusterless_kde",
        model,
        np.array([0.0, 1.0]),
        np.full((2, 1), x),
        [np.empty(0)],
        [np.empty((0, 1))],
        np.array([0.0, 1.0]),
        background_mark_models=[BackgroundMarkModel(np.zeros(1), np.ones(1))],
        is_local=local,
    )
    assert_allclose(silent, -(ground + RATE_EPS_HZ), atol=1e-5)
