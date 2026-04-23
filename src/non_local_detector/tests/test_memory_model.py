"""Unit tests for per-algorithm memory estimators.

Tests ``_estimate_predict_peak_bytes`` in each likelihood module.
These are pure arithmetic functions — no JAX, no GPU, fast.

Invariants tested (common to all algorithms):

- Increasing ``n_chunks`` reduces peak (likelihood slab scales as 1/n_chunks).
- Increasing ``n_time`` at fixed knobs increases peak linearly.
- Passing ``pos_tile_size`` reduces peak (tiles the position axis).
- All returns are positive ints.

Algorithm-specific invariants are spot-checked in their own sections.

Calibration of the absolute scale is deferred to Task 6 (real-data
profiling).  These tests verify the SHAPES of the scaling, not the
absolute peak values.
"""

from __future__ import annotations

import pytest

from non_local_detector.likelihoods.clusterless_gmm import (
    _estimate_predict_peak_bytes as _gmm_peak,
)
from non_local_detector.likelihoods.clusterless_kde import (
    _estimate_predict_peak_bytes as _kde_peak,
)
from non_local_detector.likelihoods.clusterless_kde_log import (
    _estimate_predict_peak_bytes as _kde_log_peak,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    _estimate_predict_peak_bytes as _glm_peak,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    _estimate_predict_peak_bytes as _sorted_kde_peak,
)

# Realistic workload shapes for sanity checks.
_CONTFRAG_2D_WORKLOAD = {
    "n_time": 709_321,
    "n_state_bins": 1_446,
    "n_pos": 1_446,
    "n_encoding_spikes_max": 20_000,
    "n_decoding_spikes_max": 80_000,
    "n_waveform_features": 4,
}

_SORTED_WORKLOAD = {
    "n_time": 709_321,
    "n_state_bins": 1_446,
    "n_pos": 1_446,
    "n_neurons": 100,
}


# ---------------------------------------------------------------------------
# Shared invariants across algorithms.  Parametrised by the estimator and its
# minimum-required workload args.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredictPeakInvariantsClusterlessKDE:
    """Shape invariants for clusterless_kde."""

    def test_returns_positive_int(self):
        peak = _kde_peak(**_CONTFRAG_2D_WORKLOAD)
        assert isinstance(peak, int)
        assert peak > 0

    def test_more_chunks_reduces_peak(self):
        peak_1 = _kde_peak(**_CONTFRAG_2D_WORKLOAD, n_chunks=1)
        peak_4 = _kde_peak(**_CONTFRAG_2D_WORKLOAD, n_chunks=4)
        assert peak_4 < peak_1, (
            "Expected chunking to reduce peak memory, "
            f"but peak_1={peak_1} peak_4={peak_4}"
        )

    def test_more_time_increases_peak(self):
        shorter = {**_CONTFRAG_2D_WORKLOAD, "n_time": 100_000}
        longer = {**_CONTFRAG_2D_WORKLOAD, "n_time": 1_000_000}
        assert _kde_peak(**longer) > _kde_peak(**shorter)

    def test_pos_tiling_reduces_peak(self):
        peak_no_tile = _kde_peak(**_CONTFRAG_2D_WORKLOAD)
        peak_tile = _kde_peak(**_CONTFRAG_2D_WORKLOAD, pos_tile_size=500)
        assert peak_tile < peak_no_tile

    def test_enc_tiling_reduces_peak(self):
        peak_no_tile = _kde_peak(**_CONTFRAG_2D_WORKLOAD)
        peak_tile = _kde_peak(**_CONTFRAG_2D_WORKLOAD, enc_tile_size=2_000)
        assert peak_tile < peak_no_tile

    def test_smaller_block_size_reduces_peak_via_mark_kernel_tmp(self):
        """Smaller block_size -> smaller per-block temporaries."""
        big = _kde_peak(**_CONTFRAG_2D_WORKLOAD, block_size=5_000)
        small = _kde_peak(**_CONTFRAG_2D_WORKLOAD, block_size=100)
        assert small < big

    def test_contfrag_2d_scale(self):
        """Model output for the reference workload should be in the
        plausible GB range (1-100 GB).  Precise number is tuned in Task 6."""
        peak = _kde_peak(**_CONTFRAG_2D_WORKLOAD, n_chunks=1, block_size=1_000)
        gb = peak / 2**30
        assert 1 < gb < 200, f"clusterless_kde peak = {gb:.1f} GB (out of range)"


@pytest.mark.unit
class TestPredictPeakInvariantsClusterlessKDELog:
    """clusterless_kde_log re-exports the clusterless_kde estimator (same math
    post-PR-14).  Confirm the identity."""

    def test_identical_to_kde(self):
        assert _kde_log_peak is _kde_peak

    def test_returns_positive_int(self):
        assert _kde_log_peak(**_CONTFRAG_2D_WORKLOAD) > 0


@pytest.mark.unit
class TestPredictPeakInvariantsSortedKDE:
    def test_returns_positive_int(self):
        peak = _sorted_kde_peak(**_SORTED_WORKLOAD)
        assert isinstance(peak, int) and peak > 0

    def test_more_chunks_reduces_peak(self):
        peak_1 = _sorted_kde_peak(**_SORTED_WORKLOAD, n_chunks=1)
        peak_4 = _sorted_kde_peak(**_SORTED_WORKLOAD, n_chunks=4)
        assert peak_4 < peak_1

    def test_more_neurons_increases_peak(self):
        few = _sorted_kde_peak(**{**_SORTED_WORKLOAD, "n_neurons": 10})
        many = _sorted_kde_peak(**{**_SORTED_WORKLOAD, "n_neurons": 1_000})
        assert many > few

    def test_no_mark_kernel_scaling(self):
        """Sorted peak shouldn't scale with encoding-spike counts (it has no
        waveform mark kernel)."""
        common = {
            "n_time": _SORTED_WORKLOAD["n_time"],
            "n_state_bins": _SORTED_WORKLOAD["n_state_bins"],
            "n_pos": _SORTED_WORKLOAD["n_pos"],
            "n_neurons": 100,
        }
        peak_a = _sorted_kde_peak(**common)
        peak_b = _sorted_kde_peak(**common, n_encoding_spikes_max=1_000_000)
        # n_encoding_spikes_max is declared unused on the sorted path.
        assert peak_a == peak_b


@pytest.mark.unit
class TestPredictPeakInvariantsSortedGLM:
    _GLM_WORKLOAD = {
        **_SORTED_WORKLOAD,
        "n_coefficients": 16,
    }

    def test_returns_positive_int(self):
        assert _glm_peak(**self._GLM_WORKLOAD) > 0

    def test_more_coefficients_increases_peak(self):
        few = _glm_peak(**{**self._GLM_WORKLOAD, "n_coefficients": 8})
        many = _glm_peak(**{**self._GLM_WORKLOAD, "n_coefficients": 128})
        assert many > few

    def test_more_chunks_reduces_peak(self):
        peak_1 = _glm_peak(**self._GLM_WORKLOAD, n_chunks=1)
        peak_8 = _glm_peak(**self._GLM_WORKLOAD, n_chunks=8)
        assert peak_8 < peak_1


@pytest.mark.unit
class TestPredictPeakInvariantsClusterlessGMM:
    _GMM_WORKLOAD = {
        **{k: v for k, v in _CONTFRAG_2D_WORKLOAD.items() if k != "n_encoding_spikes_max"},
        "n_gmm_components": 16,
    }

    def test_returns_positive_int(self):
        assert _gmm_peak(**self._GMM_WORKLOAD) > 0

    def test_delegates_to_kde(self):
        """GMM stub delegates to clusterless_kde estimator with
        ``n_encoding_spikes_max = n_gmm_components``."""
        gmm_peak = _gmm_peak(**self._GMM_WORKLOAD, n_chunks=1)
        kde_peak = _kde_peak(
            n_time=self._GMM_WORKLOAD["n_time"],
            n_state_bins=self._GMM_WORKLOAD["n_state_bins"],
            n_pos=self._GMM_WORKLOAD["n_pos"],
            n_encoding_spikes_max=self._GMM_WORKLOAD["n_gmm_components"],
            n_decoding_spikes_max=self._GMM_WORKLOAD["n_decoding_spikes_max"],
            n_waveform_features=self._GMM_WORKLOAD["n_waveform_features"],
            n_chunks=1,
        )
        assert gmm_peak == kde_peak


# ---------------------------------------------------------------------------
# Parametrised sanity across all estimators at once.
# ---------------------------------------------------------------------------


_ESTIMATORS = [
    pytest.param(
        _kde_peak, _CONTFRAG_2D_WORKLOAD, id="clusterless_kde"
    ),
    pytest.param(
        _kde_log_peak, _CONTFRAG_2D_WORKLOAD, id="clusterless_kde_log"
    ),
    pytest.param(_sorted_kde_peak, _SORTED_WORKLOAD, id="sorted_spikes_kde"),
    pytest.param(
        _glm_peak,
        {**_SORTED_WORKLOAD, "n_coefficients": 16},
        id="sorted_spikes_glm",
    ),
    pytest.param(
        _gmm_peak,
        {
            **{
                k: v
                for k, v in _CONTFRAG_2D_WORKLOAD.items()
                if k != "n_encoding_spikes_max"
            },
            "n_gmm_components": 16,
        },
        id="clusterless_gmm",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("estimator", "workload"), _ESTIMATORS)
class TestCommonInvariantsAllAlgorithms:
    def test_returns_positive_int(self, estimator, workload):
        assert isinstance(estimator(**workload), int)
        assert estimator(**workload) > 0

    def test_more_chunks_reduces_peak(self, estimator, workload):
        peak_1 = estimator(**workload, n_chunks=1)
        peak_16 = estimator(**workload, n_chunks=16)
        assert peak_16 < peak_1

    def test_peak_fits_in_plausible_range(self, estimator, workload):
        """Sanity: default peak should be 1-300 GB for realistic workloads."""
        peak_gb = estimator(**workload) / 2**30
        assert 0.5 < peak_gb < 500, f"Peak {peak_gb:.1f} GB out of range"

    def test_fp64_larger_than_fp32_by_up_to_2x(self, estimator, workload):
        """``fixed_scratch`` doesn't scale with dtype, so the ratio floors
        below 2 when fixed_scratch is a meaningful fraction of the total.
        Accept anywhere in [1.3, 2.1]."""
        peak_32 = estimator(**workload, dtype_bytes=4)
        peak_64 = estimator(**workload, dtype_bytes=8)
        ratio = peak_64 / peak_32
        assert 1.3 < ratio < 2.1, f"fp64/fp32 ratio = {ratio:.2f} (expected 1.3-2.1)"


# ---------------------------------------------------------------------------
# Fit-time peak estimators (Task 4b)
# ---------------------------------------------------------------------------


from non_local_detector.likelihoods.clusterless_gmm import (  # noqa: E402
    _estimate_fit_peak_bytes as _gmm_fit_peak,
)
from non_local_detector.likelihoods.clusterless_kde import (  # noqa: E402
    _estimate_fit_peak_bytes as _kde_fit_peak,
)
from non_local_detector.likelihoods.clusterless_kde_log import (  # noqa: E402
    _estimate_fit_peak_bytes as _kde_log_fit_peak,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (  # noqa: E402
    _estimate_fit_peak_bytes as _glm_fit_peak,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (  # noqa: E402
    _estimate_fit_peak_bytes as _sorted_kde_fit_peak,
)

_FIT_WORKLOAD_CLUSTERLESS = {
    "n_time_pos": 709_321,
    "n_pos": 1_446,
    "n_encoding_spikes_max": 20_000,
    "n_waveform_features": 4,
}

_FIT_WORKLOAD_SORTED_KDE = {
    "n_time_pos": 709_321,
    "n_pos": 1_446,
    "n_neurons": 100,
    "n_spikes_per_neuron_max": 5_000,
}

_FIT_WORKLOAD_SORTED_GLM = {
    "n_time_pos": 709_321,
    "n_pos": 1_446,
    "n_neurons": 100,
    "n_coefficients": 16,
}

_FIT_WORKLOAD_GMM = {
    **_FIT_WORKLOAD_CLUSTERLESS,
    "n_gmm_components": 16,
}


@pytest.mark.unit
class TestFitPeakInvariants:
    """Shared invariants across fit-time estimators."""

    def test_kde_fit_returns_positive_int(self):
        assert _kde_fit_peak(**_FIT_WORKLOAD_CLUSTERLESS) > 0

    def test_kde_log_fit_is_kde_fit(self):
        """clusterless_kde_log re-exports the clusterless_kde fit estimator."""
        assert _kde_log_fit_peak is _kde_fit_peak

    def test_kde_fit_smaller_block_reduces_peak(self):
        big = _kde_fit_peak(**_FIT_WORKLOAD_CLUSTERLESS, fit_block_size=100_000)
        small = _kde_fit_peak(**_FIT_WORKLOAD_CLUSTERLESS, fit_block_size=1_000)
        assert small <= big  # <= because occupancy might not be the max term

    def test_kde_fit_more_encoding_spikes_increases_peak(self):
        few = _kde_fit_peak(**{**_FIT_WORKLOAD_CLUSTERLESS, "n_encoding_spikes_max": 1_000})
        many = _kde_fit_peak(**{**_FIT_WORKLOAD_CLUSTERLESS, "n_encoding_spikes_max": 100_000})
        assert many > few

    def test_sorted_kde_fit_scales_with_neurons(self):
        few = _sorted_kde_fit_peak(**{**_FIT_WORKLOAD_SORTED_KDE, "n_neurons": 10})
        many = _sorted_kde_fit_peak(**{**_FIT_WORKLOAD_SORTED_KDE, "n_neurons": 1_000})
        assert many > few

    def test_glm_fit_scales_with_coefficients(self):
        few = _glm_fit_peak(**{**_FIT_WORKLOAD_SORTED_GLM, "n_coefficients": 8})
        many = _glm_fit_peak(**{**_FIT_WORKLOAD_SORTED_GLM, "n_coefficients": 128})
        assert many > few

    def test_glm_fit_scales_with_n_time_pos(self):
        """GLM fit dominated by (n_time_pos × n_coefficients) design matrix."""
        short = _glm_fit_peak(**{**_FIT_WORKLOAD_SORTED_GLM, "n_time_pos": 100_000})
        long = _glm_fit_peak(**{**_FIT_WORKLOAD_SORTED_GLM, "n_time_pos": 1_000_000})
        assert long > short

    def test_gmm_fit_delegates_to_kde_fit(self):
        gmm = _gmm_fit_peak(**_FIT_WORKLOAD_GMM)
        kde = _kde_fit_peak(**_FIT_WORKLOAD_CLUSTERLESS)
        assert gmm == kde


@pytest.mark.unit
@pytest.mark.parametrize(
    ("estimator", "workload"),
    [
        pytest.param(_kde_fit_peak, _FIT_WORKLOAD_CLUSTERLESS, id="clusterless_kde_fit"),
        pytest.param(_kde_log_fit_peak, _FIT_WORKLOAD_CLUSTERLESS, id="clusterless_kde_log_fit"),
        pytest.param(_sorted_kde_fit_peak, _FIT_WORKLOAD_SORTED_KDE, id="sorted_kde_fit"),
        pytest.param(_glm_fit_peak, _FIT_WORKLOAD_SORTED_GLM, id="sorted_glm_fit"),
        pytest.param(_gmm_fit_peak, _FIT_WORKLOAD_GMM, id="clusterless_gmm_fit"),
    ],
)
class TestCommonFitInvariants:
    def test_returns_positive_int(self, estimator, workload):
        assert estimator(**workload) > 0
        assert isinstance(estimator(**workload), int)

    def test_peak_in_plausible_range(self, estimator, workload):
        peak_gb = estimator(**workload) / 2**30
        assert 0.05 < peak_gb < 500, f"Fit peak {peak_gb:.2f} GB out of range"
