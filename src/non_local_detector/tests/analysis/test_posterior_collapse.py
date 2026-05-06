"""Tests for ``analysis.posterior`` collapse helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from non_local_detector.analysis.posterior import (
    PosteriorReduction,
    _conditional_row,
    _non_local_state_ids,
    _spatial_state_ids,
    collapse_log_likelihood_to_position,
    collapse_posterior_to_position,
    conditional_non_local_posterior,
    select_reduction,
)
from non_local_detector.tests._simulated_detectors import FittedDetector


def _n_pos(detector) -> int:
    return int(detector.environments[0].place_bin_centers_.shape[0])


def _first_finite_row_index(values: np.ndarray) -> int:
    """Return the first time index with at least one finite value in any column."""
    finite_mask = np.isfinite(values).any(axis=-1)
    finite_indices = np.flatnonzero(finite_mask)
    assert finite_indices.size > 0, "No finite rows in supplied array"
    return int(finite_indices[0])


# ---------------------------------------------------------------------------
# select_reduction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectReduction:
    """Auto-detect rules for ``select_reduction``."""

    def test_nl_picks_conditional_non_local(self, nl_fitted: FittedDetector) -> None:
        assert (
            select_reduction(
                nl_fitted.detector.state_names,
                nl_fitted.detector.bin_sizes_,
            )
            is PosteriorReduction.CONDITIONAL_NON_LOCAL
        )

    def test_nsf_picks_conditional_on_spatial(self, nsf_fitted: FittedDetector) -> None:
        assert (
            select_reduction(
                nsf_fitted.detector.state_names,
                nsf_fitted.detector.bin_sizes_,
            )
            is PosteriorReduction.CONDITIONAL_ON_SPATIAL
        )

    def test_cf_picks_marginal(self, cf_fitted: FittedDetector) -> None:
        assert (
            select_reduction(
                cf_fitted.detector.state_names,
                cf_fitted.detector.bin_sizes_,
            )
            is PosteriorReduction.MARGINAL
        )

    def test_dec_picks_marginal(self, dec_fitted: FittedDetector) -> None:
        assert (
            select_reduction(
                dec_fitted.detector.state_names,
                dec_fitted.detector.bin_sizes_,
            )
            is PosteriorReduction.MARGINAL
        )


# ---------------------------------------------------------------------------
# collapse_log_likelihood_to_position
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollapseLogLikelihoodToPosition:
    """Schema-aware log-likelihood collapse."""

    def test_nl_loglik_row_shape_finite_peak_one(
        self, nl_fitted: FittedDetector
    ) -> None:
        log_lik = nl_fitted.results["log_likelihood"].values
        n_pos = _n_pos(nl_fitted.detector)
        t_idx = _first_finite_row_index(log_lik)
        out = collapse_log_likelihood_to_position(log_lik[t_idx], nl_fitted.detector)
        assert out.shape == (n_pos,)
        assert np.all(np.isfinite(out))
        assert np.isclose(out.max(), 1.0)

    def test_cf_loglik_row_shape_peak_one(self, cf_fitted: FittedDetector) -> None:
        log_lik = cf_fitted.results["log_likelihood"].values
        n_pos = _n_pos(cf_fitted.detector)
        t_idx = _first_finite_row_index(log_lik)
        out = collapse_log_likelihood_to_position(log_lik[t_idx], cf_fitted.detector)
        assert out.shape == (n_pos,)
        assert np.isclose(out.max(), 1.0)

    def test_all_nan_row_returns_zeros(self, nl_fitted: FittedDetector) -> None:
        n_state_bins = nl_fitted.detector.n_state_bins_
        n_pos = _n_pos(nl_fitted.detector)
        out = collapse_log_likelihood_to_position(
            np.full(n_state_bins, np.nan), nl_fitted.detector
        )
        np.testing.assert_array_equal(out, np.zeros(n_pos))

    def test_all_neg_inf_row_returns_zeros(self, nl_fitted: FittedDetector) -> None:
        n_state_bins = nl_fitted.detector.n_state_bins_
        n_pos = _n_pos(nl_fitted.detector)
        out = collapse_log_likelihood_to_position(
            np.full(n_state_bins, -np.inf), nl_fitted.detector
        )
        np.testing.assert_array_equal(out, np.zeros(n_pos))

    def test_mixed_finite_and_neg_inf_no_nan(self, nl_fitted: FittedDetector) -> None:
        """One spatial slice all -inf + one finite → no NaN, -inf slice → 0."""
        detector = nl_fitted.detector
        spatial_state_ids = _spatial_state_ids(detector)
        assert spatial_state_ids.size >= 2  # NL has multiple spatial states

        n_state_bins = detector.n_state_bins_
        row = np.zeros(n_state_bins, dtype=np.float64)
        # First spatial state's slice → all -inf.
        first_state_mask = detector.state_ind_ == spatial_state_ids[0]
        row[first_state_mask] = -np.inf
        # Other state bins (incl. singleton states) carry zeros = exp(0) = 1.
        out = collapse_log_likelihood_to_position(row, detector)
        # Output is finite everywhere — no NaN from -inf - finite_max.
        assert not np.any(np.isnan(out))
        # Peak-normalized.
        assert np.isclose(out.max(), 1.0)

    def test_no_spatial_states_raises(self, nl_fitted: FittedDetector) -> None:
        """Hand-build a detector with all singleton bin_sizes_ → ValueError."""
        from copy import copy

        detector = copy(nl_fitted.detector)
        detector.bin_sizes_ = np.array([1, 1, 1, 1])
        n_state_bins = nl_fitted.detector.n_state_bins_
        with pytest.raises(ValueError) as exc_info:
            collapse_log_likelihood_to_position(np.zeros(n_state_bins), detector)
        message = str(exc_info.value).lower()
        assert "spatial state" in message

    @pytest.mark.slow
    def test_singleton_local_drops_local_from_position_curve(
        self, nl_singleton_fitted: FittedDetector
    ) -> None:
        """Phase 1c SlicePanel design check: singleton-Local schema.

        With ``local_position_std=None`` the detector has
        ``bin_sizes_=[1, 1, n_pos, n_pos]`` — both ``Local`` and
        ``No-Spike`` are singleton, leaving the two ``Non-Local``
        states as the only spatial states. The top SlicePanel curve
        is built by ``collapse_log_likelihood_to_position`` summing
        only those spatial-state slices; ``Local``'s scalar
        likelihood gets dropped from the position axis.

        Asserts:
        - Output shape is ``(n_pos,)`` (not ``(2, n_pos)`` /
          ``(n_state_bins,)``).
        - The output equals (within float64 precision) what you get
          by re-running the helper on a synthetic row that zeroes
          out the singleton (``Local`` + ``No-Spike``) bins — i.e.
          singleton entries genuinely don't contribute.
        - Peak-normalized to 1.0 on a finite row.
        """
        detector = nl_singleton_fitted.detector
        bin_sizes = np.asarray(detector.bin_sizes_)
        # Schema sanity: both Local + No-Spike are singleton.
        assert int(bin_sizes[0]) == 1
        assert int(bin_sizes[1]) == 1
        env = detector.environments[0]
        n_pos = int(env.place_bin_centers_.shape[0])

        log_lik = nl_singleton_fitted.results["log_likelihood"].values
        t_idx = _first_finite_row_index(log_lik)
        row = log_lik[t_idx]

        out = collapse_log_likelihood_to_position(row, detector)
        assert out.shape == (n_pos,)
        assert np.isclose(out.max(), 1.0)

        # Zero out the two singleton-state bins and recompute. Result
        # must be identical: singletons don't contribute to the
        # position-axis sum.
        row_no_singletons = row.copy()
        state_ind = np.asarray(detector.state_ind_)
        singleton_ids = np.flatnonzero(bin_sizes == 1)
        singleton_mask = np.isin(state_ind, singleton_ids)
        row_no_singletons[singleton_mask] = -np.inf
        out_no_singletons = collapse_log_likelihood_to_position(
            row_no_singletons, detector
        )
        np.testing.assert_allclose(out, out_no_singletons, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# collapse_posterior_to_position — shared row math + zero-mass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollapsePosteriorRouting:
    """Both public CONDITIONAL_* paths route through ``_conditional_row``."""

    def test_conditional_non_local_matches_private_helper_and_dataset_helper(
        self, nl_fitted: FittedDetector
    ) -> None:
        detector = nl_fitted.detector
        post = nl_fitted.results["acausal_posterior"].values
        t_idx = _first_finite_row_index(post)
        nl_state_ids = _non_local_state_ids(detector)
        expected = _conditional_row(
            post[t_idx],
            detector,
            selected_state_ids=nl_state_ids,
            zero_mass_fill=np.nan,
        )

        from_public = collapse_posterior_to_position(
            post[t_idx],
            detector,
            PosteriorReduction.CONDITIONAL_NON_LOCAL,
            zero_mass_fill=np.nan,
        )
        assert np.array_equal(from_public, expected, equal_nan=True)

        # The dataset-level helper iterates time and calls _conditional_row
        # per row; row t_idx must equal the private-helper output.
        from_dataset = (
            conditional_non_local_posterior(
                nl_fitted.results, detector, zero_mass_fill=np.nan
            )
            .isel(time=t_idx)
            .values
        )
        # The dataset helper also re-pads non-interior bins to NaN —
        # the private helper output for non-interior columns is
        # NaN-from-zero-mass-fill or carries the column-summed NaN, both
        # equal to NaN. ``equal_nan=True`` makes the comparison robust.
        assert np.array_equal(from_dataset, expected, equal_nan=True)

    def test_conditional_on_spatial_matches_private_helper(
        self, nsf_fitted: FittedDetector
    ) -> None:
        detector = nsf_fitted.detector
        post = nsf_fitted.results["acausal_posterior"].values
        t_idx = _first_finite_row_index(post)
        spatial_state_ids = _spatial_state_ids(detector)
        expected = _conditional_row(
            post[t_idx],
            detector,
            selected_state_ids=spatial_state_ids,
            zero_mass_fill=np.nan,
        )

        from_public = collapse_posterior_to_position(
            post[t_idx],
            detector,
            PosteriorReduction.CONDITIONAL_ON_SPATIAL,
            zero_mass_fill=np.nan,
        )
        assert np.array_equal(from_public, expected, equal_nan=True)


@pytest.mark.unit
class TestCollapsePosteriorZeroMassFill:
    """``zero_mass_fill`` plumbing for both ``CONDITIONAL_*`` paths."""

    def test_conditional_non_local_nan_fill(self, nl_fitted: FittedDetector) -> None:
        detector = nl_fitted.detector
        n_state_bins = detector.n_state_bins_
        # Row with zero non-local mass: put all mass on Local + No-Spike
        # (state ids 0 and 1).
        post_row = np.zeros(n_state_bins, dtype=np.float64)
        local_mask = (detector.state_ind_ == 0) & detector.is_track_interior_state_bins_
        post_row[local_mask] = 1.0 / local_mask.sum()
        out = collapse_posterior_to_position(
            post_row,
            detector,
            PosteriorReduction.CONDITIONAL_NON_LOCAL,
            zero_mass_fill=np.nan,
        )
        assert np.all(np.isnan(out))

    def test_conditional_non_local_zero_fill(self, nl_fitted: FittedDetector) -> None:
        detector = nl_fitted.detector
        n_state_bins = detector.n_state_bins_
        post_row = np.zeros(n_state_bins, dtype=np.float64)
        local_mask = (detector.state_ind_ == 0) & detector.is_track_interior_state_bins_
        post_row[local_mask] = 1.0 / local_mask.sum()
        out = collapse_posterior_to_position(
            post_row,
            detector,
            PosteriorReduction.CONDITIONAL_NON_LOCAL,
            zero_mass_fill=0.0,
        )
        np.testing.assert_array_equal(out, np.zeros(_n_pos(detector)))

    def test_conditional_on_spatial_nan_fill(self, nsf_fitted: FittedDetector) -> None:
        detector = nsf_fitted.detector
        n_state_bins = detector.n_state_bins_
        # Put all mass on the singleton "No-Spike" state (id=0).
        post_row = np.zeros(n_state_bins, dtype=np.float64)
        no_spike_mask = detector.state_ind_ == 0
        post_row[no_spike_mask] = 1.0
        out = collapse_posterior_to_position(
            post_row,
            detector,
            PosteriorReduction.CONDITIONAL_ON_SPATIAL,
            zero_mass_fill=np.nan,
        )
        assert np.all(np.isnan(out))

    def test_conditional_on_spatial_zero_fill(self, nsf_fitted: FittedDetector) -> None:
        detector = nsf_fitted.detector
        n_state_bins = detector.n_state_bins_
        post_row = np.zeros(n_state_bins, dtype=np.float64)
        no_spike_mask = detector.state_ind_ == 0
        post_row[no_spike_mask] = 1.0
        out = collapse_posterior_to_position(
            post_row,
            detector,
            PosteriorReduction.CONDITIONAL_ON_SPATIAL,
            zero_mass_fill=0.0,
        )
        np.testing.assert_array_equal(out, np.zeros(_n_pos(detector)))


# ---------------------------------------------------------------------------
# collapse_posterior_to_position — MARGINAL
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollapsePosteriorMarginal:
    """``MARGINAL`` reduction never divides."""

    def test_cf_marginal_row_sums_to_one(self, cf_fitted: FittedDetector) -> None:
        detector = cf_fitted.detector
        post = cf_fitted.results["acausal_posterior"].values
        t_idx = _first_finite_row_index(post)
        out = collapse_posterior_to_position(
            post[t_idx], detector, PosteriorReduction.MARGINAL
        )
        assert np.isclose(np.nansum(out), 1.0, atol=1e-10)

    def test_dec_marginal_row_sums_to_one(self, dec_fitted: FittedDetector) -> None:
        detector = dec_fitted.detector
        post = dec_fitted.results["acausal_posterior"].values
        t_idx = _first_finite_row_index(post)
        out = collapse_posterior_to_position(
            post[t_idx], detector, PosteriorReduction.MARGINAL
        )
        assert np.isclose(np.nansum(out), 1.0, atol=1e-10)

    def test_nl_marginal_override_equals_one_minus_singletons(
        self, nl_fitted: FittedDetector
    ) -> None:
        """For NL with MARGINAL, row sum = 1 - sum(singleton-state mass).

        Only ``No-Spike`` is a singleton when ``local_position_std=1.0``
        (Track 0 default), so this reduces to ``1 - P(No-Spike)``.
        """
        detector = nl_fitted.detector
        post = nl_fitted.results["acausal_posterior"].values
        t_idx = _first_finite_row_index(post)
        out = collapse_posterior_to_position(
            post[t_idx], detector, PosteriorReduction.MARGINAL
        )
        bin_sizes = np.asarray(detector.bin_sizes_)
        singleton_state_ids = np.flatnonzero(bin_sizes == 1)
        singleton_bins_mask = np.isin(detector.state_ind_, singleton_state_ids)
        singleton_mass = np.nansum(post[t_idx][singleton_bins_mask])
        assert np.isclose(np.nansum(out), 1.0 - singleton_mass, atol=1e-10)

    @pytest.mark.slow
    def test_nl_singleton_marginal_override_equals_one_minus_singletons(
        self, nl_singleton_fitted: FittedDetector
    ) -> None:
        """Same MARGINAL invariant for the singleton-Local fixture.

        ``local_position_std=None`` makes both ``Local`` and
        ``No-Spike`` singletons (``bin_sizes_=[1, 1, n_pos, n_pos]``),
        so the row sum should equal ``1 - P(Local) - P(No-Spike)``.
        Tests that a future "fix" that adds renormalization to MARGINAL
        gets caught against this schema as well as the
        ``local_position_std=1.0`` schema.
        """
        detector = nl_singleton_fitted.detector
        post = nl_singleton_fitted.results["acausal_posterior"].values
        t_idx = _first_finite_row_index(post)
        out = collapse_posterior_to_position(
            post[t_idx], detector, PosteriorReduction.MARGINAL
        )
        bin_sizes = np.asarray(detector.bin_sizes_)
        # Schema sanity check: both Local + No-Spike are singleton.
        assert int(bin_sizes[0]) == 1
        assert int(bin_sizes[1]) == 1
        singleton_state_ids = np.flatnonzero(bin_sizes == 1)
        singleton_bins_mask = np.isin(detector.state_ind_, singleton_state_ids)
        singleton_mass = np.nansum(post[t_idx][singleton_bins_mask])
        assert np.isclose(np.nansum(out), 1.0 - singleton_mass, atol=1e-6)


# ---------------------------------------------------------------------------
# collapse_posterior_to_position — CONDITIONAL_ON_SPATIAL on NSF
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollapsePosteriorConditionalOnSpatial:
    """``NoSpikeContFrag`` default reduction integrates to 1.0."""

    def test_nsf_shape_and_unit_sum(self, nsf_fitted: FittedDetector) -> None:
        detector = nsf_fitted.detector
        post = nsf_fitted.results["acausal_posterior"].values
        n_pos = _n_pos(detector)
        # Pick a row where the spatial mass is positive (typical case).
        ax_state_probs = nsf_fitted.results["acausal_state_probabilities"].sel(
            states="No-Spike"
        )
        spatial_mass_per_t = 1.0 - ax_state_probs.values
        # Pick a row whose spatial mass is large enough to avoid zero-mass
        # fill noise.
        candidate = np.flatnonzero(spatial_mass_per_t > 0.01)
        assert candidate.size > 0
        t_idx = int(candidate[0])
        out = collapse_posterior_to_position(
            post[t_idx],
            detector,
            PosteriorReduction.CONDITIONAL_ON_SPATIAL,
            zero_mass_fill=np.nan,
        )
        assert out.shape == (n_pos,)
        assert np.isclose(np.nansum(out), 1.0, atol=1e-10)
        # Implicit divisor == 1 - P(No-Spike).
        spatial_mass = spatial_mass_per_t[t_idx]
        spatial_state_ids = _spatial_state_ids(detector)
        spatial_mask = np.isin(detector.state_ind_, spatial_state_ids)
        spatial_column_sum = np.nansum(
            post[t_idx][spatial_mask].reshape(spatial_state_ids.size, n_pos),
            axis=0,
        )
        # Float32 precision in posterior storage: ~1e-7. Use 1e-6.
        np.testing.assert_allclose(
            out * spatial_mass, spatial_column_sum, atol=1e-6, rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Dataset-level helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConditionalNonLocalPosteriorDataset:
    """``conditional_non_local_posterior`` over the full session."""

    def test_shape_and_dims(self, nl_fitted: FittedDetector) -> None:
        out = conditional_non_local_posterior(nl_fitted.results, nl_fitted.detector)
        assert isinstance(out, xr.DataArray)
        assert out.dims == ("time", "position")
        assert out.shape[1] == _n_pos(nl_fitted.detector)
        assert out.shape[0] == len(nl_fitted.results["time"])

    def test_active_rows_sum_to_one(self, nl_fitted: FittedDetector) -> None:
        out = conditional_non_local_posterior(
            nl_fitted.results, nl_fitted.detector, zero_mass_fill=np.nan
        ).values
        row_sums = np.nansum(out, axis=-1)
        active = ~np.isnan(row_sums) & (row_sums > 0)
        assert active.any()
        # Float32 precision in posterior storage: ~1e-7. Use 1e-6.
        np.testing.assert_allclose(row_sums[active], 1.0, atol=1e-6, rtol=1e-6)
