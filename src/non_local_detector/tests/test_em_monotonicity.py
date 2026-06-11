"""Test that EM log-likelihood is monotonically non-decreasing.

This is a fundamental guarantee of the EM algorithm: the marginal
log-likelihood must never decrease across iterations. Violation
indicates a bug in the E-step or M-step.

Tests cover:
- Clusterless decoder with discrete-only updates
- Sorted spikes (KDE) encoding update guard integration
- Sorted spikes (GLM) end-to-end smoke test
- Surfacing of monotonicity violations and max-iter exits via
  warnings and model attributes (``converged_``, ``n_iter_``,
  ``em_monotonicity_violations_``).
"""

import warnings

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder, NonLocalSortedSpikesDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture
def sim():
    """Small 2-tetrode clusterless run dataset shared across the EM tests."""
    return make_simulated_run_data(
        n_tetrodes=2,
        place_field_means=np.arange(0, 80, 20),
        n_runs=3,
        seed=42,
    )


def _assert_monotonic(lls, label=""):
    """Assert that log-likelihoods are non-decreasing (with float tolerance)."""
    assert len(lls) >= 2, f"{label}: Expected at least 2 EM iterations, got {len(lls)}"
    for i in range(1, len(lls)):
        assert lls[i] >= lls[i - 1] - 1e-6, (
            f"{label}: EM log-likelihood decreased at iteration {i}: "
            f"{lls[i]:.6f} < {lls[i - 1]:.6f} "
            f"(diff={lls[i] - lls[i - 1]:.2e})"
        )


@pytest.mark.slow
@pytest.mark.integration
class TestEMMonotonicity:
    """EM marginal log-likelihood must be non-decreasing across iterations."""

    def test_clusterless_decoder_em_monotonicity(self, sim):
        """Fit a ClusterlessDecoder with multiple EM iterations and verify
        that marginal_log_likelihoods is non-decreasing."""

        decoder = ClusterlessDecoder()
        results = decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            max_iter=5,
            estimate_encoding_model=False,
        )

        lls = results.attrs["marginal_log_likelihoods"]
        _assert_monotonic(lls, "ClusterlessDecoder (discrete only)")


@pytest.mark.slow
@pytest.mark.integration
class TestEncodingUpdateGuards:
    """Encoding update guards (ESS/mass, rollback, damping) work correctly."""

    def test_sorted_spikes_kde_encoding_with_rollback(self):
        """NonLocalSortedSpikesDetector with KDE encoding updates should
        run to completion with rollback guard active."""
        speed, position, spike_times, time, event_times, sampling_freq, is_event, _ = (
            make_simulated_data(seed=42, n_neurons=5)
        )

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_kde",
        )
        results = detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time=time,
            is_training=~is_event,
            max_iter=5,
            estimate_encoding_model=True,
        )

        lls = results.attrs["marginal_log_likelihoods"]
        assert len(lls) >= 2, f"Expected at least 2 EM iterations, got {len(lls)}"
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results

    def test_well_behaved_em_no_warnings(self, sim):
        """A standard converging fit sets ``converged_`` to True, records
        no monotonicity violations, and emits no ``UserWarning``."""

        decoder = ClusterlessDecoder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time=sim.position_time,
                max_iter=20,
                estimate_encoding_model=False,
            )

        assert decoder.converged_ is True
        assert decoder.em_monotonicity_violations_ == []
        assert decoder.n_iter_ >= 1
        unexpected = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and (
                "did not converge" in str(w.message)
                or "decreased" in str(w.message)
                or "inconsistency between the E-step" in str(w.message)
            )
        ]
        assert unexpected == [], (
            "Unexpected EM convergence warnings emitted: "
            f"{[str(w.message) for w in unexpected]}"
        )

    def test_sorted_spikes_glm_encoding_runs_end_to_end(self):
        """NonLocalSortedSpikesDetector with GLM encoding updates should
        run to completion without errors."""
        speed, position, spike_times, time, event_times, sampling_freq, is_event, _ = (
            make_simulated_data(seed=42, n_neurons=5)
        )

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_glm",
            sorted_spikes_algorithm_params={},
        )
        results = detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time=time,
            is_training=~is_event,
            max_iter=3,
            estimate_encoding_model=True,
        )

        lls = results.attrs["marginal_log_likelihoods"]
        assert len(lls) >= 2, f"Expected at least 2 EM iterations, got {len(lls)}"
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results


@pytest.mark.unit
class TestEMAttributeInitialization:
    """The three EM-result attributes are set all-or-nothing."""

    def test_validation_failure_leaves_no_partial_em_attributes(self):
        """A fit that raises during input validation must not leave a partial
        set of EM-result attributes on the model.

        Calls the base ``_DetectorBase.estimate_parameters`` directly with
        ``time=None`` so validation fails immediately (before any model state
        or data is needed), isolating the attribute-initialization order.
        """
        from non_local_detector.exceptions import ValidationError
        from non_local_detector.models.base import _DetectorBase

        decoder = ClusterlessDecoder()
        with pytest.raises(ValidationError):
            _DetectorBase.estimate_parameters(
                decoder, time=None, estimate_encoding_model=False
            )

        for attr in ("converged_", "n_iter_", "em_monotonicity_violations_"):
            assert not hasattr(decoder, attr), (
                f"{attr} was set despite a validation failure (partial state)"
            )


@pytest.mark.slow
@pytest.mark.integration
class TestEMConvergenceSurfacing:
    """Surfacing of EM monotonicity violations and max-iter exits."""

    def test_em_violation_warned_for_buggy_mstep(self, monkeypatch, sim):
        """Force ``check_converged`` to report a monotonicity violation
        on iteration 2 and verify a ``UserWarning`` is emitted and the
        iteration index is recorded in ``em_monotonicity_violations_``.

        Monotonicity violations are surfaced via ``warnings.warn`` (the same
        channel as the other EM fit-quality diagnostics), not ``logger``.
        """
        from non_local_detector.core import check_converged as _real_check_converged
        from non_local_detector.models import base as base_module

        call_count = {"n": 0}

        def fake_check_converged(curr, prev, tolerance):
            call_count["n"] += 1
            is_converged, _ = _real_check_converged(curr, prev, tolerance)
            # Report a non-monotonic step on the first comparison
            # (i.e., when n_iter==2 in the EM loop). Subsequent comparisons
            # report monotonic so the loop is allowed to converge.
            is_increasing = call_count["n"] != 1
            return is_converged, is_increasing

        monkeypatch.setattr(base_module, "check_converged", fake_check_converged)

        decoder = ClusterlessDecoder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time=sim.position_time,
                max_iter=3,
                estimate_encoding_model=False,
            )

        assert decoder.em_monotonicity_violations_ == [2]
        violation_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "log-likelihood decreased" in str(w.message)
        ]
        assert violation_warnings, (
            "Expected a UserWarning describing the monotonicity violation; "
            f"got: {[str(w.message) for w in caught]}"
        )

    def test_final_e_step_inconsistency_warns(self, monkeypatch, sim):
        """If the post-EM final E-step produces a marginal log-likelihood far
        below the last in-loop value, a ``UserWarning`` flags the E-step/M-step
        inconsistency."""

        decoder = ClusterlessDecoder()
        original_predict = decoder._predict
        state = {"n": 0}

        def perturbed_predict(*args, **kwargs):
            state["n"] += 1
            result = original_predict(*args, **kwargs)
            # With max_iter=1 the final E-step is the SECOND _predict call.
            # Drop its marginal log-likelihood sharply to simulate an
            # E-step/M-step inconsistency.
            if state["n"] == 2:
                result = list(result)
                result[2] = result[2] - 1000.0
                result = tuple(result)
            return result

        monkeypatch.setattr(decoder, "_predict", perturbed_predict)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time=sim.position_time,
                max_iter=1,
                estimate_encoding_model=False,
            )

        inconsistency = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "inconsistency between the E-step" in str(w.message)
        ]
        assert inconsistency, (
            "Expected a final-E-step inconsistency UserWarning; "
            f"got: {[str(w.message) for w in caught]}"
        )

    def test_max_iter_warning_emitted(self, sim):
        """A fit that exits because ``max_iter`` was reached (with an
        unreachable tolerance) sets ``converged_=False``, ``n_iter_==max_iter``,
        and emits a ``UserWarning`` mentioning "did not converge".
        """

        decoder = ClusterlessDecoder()
        with pytest.warns(UserWarning, match="did not converge"):
            decoder.estimate_parameters(
                position_time=sim.position_time,
                position=sim.position,
                spike_times=sim.spike_times,
                spike_waveform_features=sim.spike_waveform_features,
                time=sim.position_time,
                max_iter=1,
                tolerance=1e-20,
                estimate_encoding_model=False,
            )

        assert decoder.converged_ is False
        assert decoder.n_iter_ == 1
