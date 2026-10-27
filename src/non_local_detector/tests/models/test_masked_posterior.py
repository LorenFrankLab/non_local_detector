"""Expanding interior posteriors to every state bin."""

import numpy as np
import pytest

from non_local_detector.models import base

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n_time", [0, 1, 7, 37])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_masked_posterior_fills_exterior_bins_with_nan(n_time, dtype, monkeypatch):
    """Interior columns hold the data, exterior columns NaN, in float32, for
    row counts below, at and across the copy block size."""
    rng = np.random.default_rng(3)
    is_track_interior = rng.random(23) < 0.6
    data = rng.random((n_time, is_track_interior.sum())).astype(dtype)
    # Seven rows per block, so 37 rows span six blocks and a partial one.
    monkeypatch.setattr(base, "_MASKED_POSTERIOR_BLOCK_BYTES", 7 * 23 * 4)
    result = base._DetectorBase._create_masked_posterior(
        data, is_track_interior, len(is_track_interior)
    )
    expected = np.full((n_time, 23), np.nan, dtype=np.float32)
    expected[:, is_track_interior] = data
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
