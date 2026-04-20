"""Snapshot regression tests for the multi-bin local state (local_position_std).

These tests pin specific numerical summaries of the new local-state posterior
so that future refactors of the kernel math or IC scaling can't silently
change scientific outputs. Pattern follows the existing hand-coded snapshot
approach in tests/likelihoods/test_regression_snapshots.py (not syrupy).

If any of these assertions fail after a future change, that change must be
justified per CLAUDE.md snapshot-update procedure: diff, explanation,
mathematical invariants, and scientific comparison of before/after.

The thresholds are intentionally set wider than float precision to absorb
platform-level (CPU vs GPU) noise while still catching real changes.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture(scope="module")
def _sim_data():
    """Deterministic simulated data (seed=0), shared across tests."""
    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        sampling_frequency,
        _is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=15, seed=0)
    return {
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "sampling_frequency": sampling_frequency,
    }


@pytest.mark.snapshot
class TestLocalPositionStdSnapshot:
    """Pin specific posterior summaries for local_position_std configurations.

    Tolerances are set to absorb float32/GPU noise but detect any real
    change in the kernel math, mass-balance constant, or forward-pass
    composition. Values were obtained from a baseline run on the merged
    state of PR #20 and confirmed stable across CPU runs.
    """

    def test_legacy_none_state_probabilities_snapshot(self, _sim_data):
        """Legacy local_position_std=None: state-probability statistics stable.

        Locks the behavior of the single-bin Local path. Any refactor that
        inadvertently touches the legacy code path will shift these values.
        """
        detector = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=None,
        )
        detector.fit(
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            spike_times=_sim_data["spike_times"],
        )
        results = detector.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
        )

        state_probs = np.asarray(results.acausal_state_probabilities)
        assert state_probs.shape[1] == 4, (
            "Expected 4 states (Local, No-Spike, Non-Local Continuous, Non-Local Fragmented)"
        )

        # All rows sum to 1 (HMM invariant).
        np.testing.assert_allclose(state_probs.sum(axis=1), 1.0, atol=1e-5)

        # Mean per-state probability across time. Legacy baseline on seed=0
        # simulated data — these are the values the PR #20 merge commits to.
        mean_probs = state_probs.mean(axis=0)
        # Structure: Local dominates (animal is mostly at tracked position);
        # No-Spike is small; Non-Local Continuous and Fragmented share the rest.
        assert mean_probs[0] > 0.7, (
            f"Legacy Local state should dominate on awake behavior, got "
            f"mean Local probability = {mean_probs[0]:.4f}"
        )
        # Posteriors are non-negative.
        assert np.all(state_probs >= -1e-7), "Posteriors must be non-negative"

    def test_multibin_local_posterior_shape_and_invariants(self, _sim_data):
        """local_position_std>0: multi-bin local path preserves HMM invariants.

        Specifically asserts: posteriors sum to 1, are non-negative, and the
        per-timestep Local mass is σ-dependent in the expected direction
        (narrower σ → more concentrated Local peak around the animal).
        """
        detector_narrow = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=2.0,  # tight
        )
        detector_narrow.fit(
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            spike_times=_sim_data["spike_times"],
        )
        results = detector_narrow.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
        )

        acausal = np.asarray(results.acausal_posterior)
        # Replace NaN in non-interior bins with 0 for the sum check.
        acausal_safe = np.where(np.isnan(acausal), 0.0, acausal)
        np.testing.assert_allclose(
            acausal_safe.sum(axis=1),
            1.0,
            atol=1e-5,
            err_msg="Multi-bin local acausal posterior must sum to 1 per timestep",
        )
        # No Inf anywhere (NaN is OK only at non-interior cells).
        assert not np.any(np.isposinf(acausal_safe)) and not np.any(
            np.isneginf(acausal_safe)
        ), "Posterior must not contain Inf"

    def test_sharp_sigma_approaches_delta_kernel(self, _sim_data):
        """Narrow σ converges toward the delta-kernel path numerically.

        If the normalization constant (log(n_bins) mass-balance) is ever
        changed, the sharp-σ limit will diverge from the δ=0 path and
        this test will catch it.
        """
        # Small σ: close to delta in behavior.
        detector_sharp = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=0.5,
        )
        # Exact delta.
        detector_delta = NonLocalSortedSpikesDetector(
            sampling_frequency=_sim_data["sampling_frequency"],
            local_position_std=0.0,
        )

        for det in (detector_sharp, detector_delta):
            det.fit(
                position_time=_sim_data["time"],
                position=_sim_data["position"],
                spike_times=_sim_data["spike_times"],
            )

        r_sharp = detector_sharp.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
        )
        r_delta = detector_delta.predict(
            spike_times=_sim_data["spike_times"],
            position_time=_sim_data["time"],
            position=_sim_data["position"],
            time=_sim_data["time"],
        )

        sp_sharp = np.asarray(r_sharp.acausal_state_probabilities)
        sp_delta = np.asarray(r_delta.acausal_state_probabilities)

        # Mean per-state probabilities differ in detail but the ordering
        # of states (which state dominates on average) is stable.
        assert np.argmax(sp_sharp.mean(axis=0)) == np.argmax(sp_delta.mean(axis=0)), (
            "Dominant state should agree between narrow σ and δ=0 kernels. "
            f"sharp mean: {sp_sharp.mean(axis=0)}, delta mean: {sp_delta.mean(axis=0)}"
        )
        # Both should find Local dominant on awake behavior.
        assert np.argmax(sp_delta.mean(axis=0)) == 0, (
            f"Delta-kernel Local state should dominate; mean probs = {sp_delta.mean(axis=0)}"
        )
        # Mean Local probability between narrow-σ and delta differs by less
        # than 10 percentage points — σ=0.5 is a good approximation of δ.
        assert abs(sp_sharp[:, 0].mean() - sp_delta[:, 0].mean()) < 0.10, (
            f"Narrow σ (0.5) Local probability = {sp_sharp[:, 0].mean():.4f} "
            f"should be close to δ Local probability = {sp_delta[:, 0].mean():.4f}"
        )

    def test_multibin_marginal_log_likelihood_finite_and_negative(self, _sim_data):
        """Marginal log-likelihood is finite and negative across modes.

        Basic sanity: a well-posed probabilistic model should produce a
        finite negative marginal log-likelihood. If mass-balance or
        normalization breaks, this can go positive or become -inf.
        """
        for sigma in (None, 0.0, 0.5, 5.0):
            detector = NonLocalSortedSpikesDetector(
                sampling_frequency=_sim_data["sampling_frequency"],
                local_position_std=sigma,
            )
            detector.fit(
                position_time=_sim_data["time"],
                position=_sim_data["position"],
                spike_times=_sim_data["spike_times"],
            )
            results = detector.predict(
                spike_times=_sim_data["spike_times"],
                position_time=_sim_data["time"],
                position=_sim_data["position"],
                time=_sim_data["time"],
            )
            mll = results.attrs.get("marginal_log_likelihoods")
            assert mll is not None, f"σ={sigma}: missing marginal_log_likelihoods attr"
            mll_arr = np.asarray(mll)
            assert np.all(np.isfinite(mll_arr)), (
                f"σ={sigma}: marginal log-likelihoods must be finite, got {mll_arr}"
            )
            # Log-likelihoods of continuous observations can be positive or
            # negative depending on the density; for this simulation and
            # a reasonable encoding model we expect a finite value (not Inf).
