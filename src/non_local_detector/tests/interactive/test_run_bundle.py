"""Tests for ``RunBundle`` construction and validation."""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.exceptions import DataError
from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
)
from non_local_detector.visualization.interactive.view_models.base import RunBundle
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
)


def _bundle_from(fitted: FittedDetector, session: SimulatedSession) -> RunBundle:
    return RunBundle(
        results=fitted.results,
        detector=fitted.detector,
        spike_times=session.spike_times,
        position_time=session.time,
        position=session.position,
        speed=session.speed,
    )


@pytest.mark.unit
class TestRunBundleConstruction:
    """``RunBundle.__post_init__`` enforces internal consistency."""

    def test_construct_succeeds_for_each_detector(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
        cf_fitted: FittedDetector,
        nsf_fitted: FittedDetector,
        dec_fitted: FittedDetector,
    ) -> None:
        for fitted in (nl_fitted, cf_fitted, nsf_fitted, dec_fitted):
            bundle = _bundle_from(fitted, sim_session)
            assert "acausal_posterior" in bundle.results
            assert "acausal_state_probabilities" in bundle.results

    def test_missing_required_results_raises(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        results = nl_fitted.results.drop_vars("acausal_posterior")
        with pytest.raises(ValueError, match="acausal_posterior"):
            RunBundle(
                results=results,
                detector=nl_fitted.detector,
                spike_times=sim_session.spike_times,
                position_time=sim_session.time,
                position=sim_session.position,
            )

    def test_non_monotonic_position_time_raises(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        time = sim_session.time.copy()
        # Swap two adjacent samples to break monotonicity.
        time[10], time[11] = time[11], time[10]
        with pytest.raises(DataError, match="monotonic|increasing"):
            RunBundle(
                results=nl_fitted.results,
                detector=nl_fitted.detector,
                spike_times=sim_session.spike_times,
                position_time=time,
                position=sim_session.position,
            )

    def test_spike_times_length_mismatch_raises(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        # Drop one cell from the spike-time list to provoke the error.
        with pytest.raises(ValueError, match="spike_times"):
            RunBundle(
                results=nl_fitted.results,
                detector=nl_fitted.detector,
                spike_times=sim_session.spike_times[:-1],
                position_time=sim_session.time,
                position=sim_session.position,
            )

    def test_within_bundle_overlay_name_uniqueness(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        overlays = [
            EventOverlay.points(name="dup", times=np.array([1.0, 2.0])),
            EventOverlay.intervals(
                name="dup",
                t_start=np.array([3.0]),
                t_end=np.array([4.0]),
            ),
        ]
        with pytest.raises(ValueError, match="duplicate names"):
            RunBundle(
                results=nl_fitted.results,
                detector=nl_fitted.detector,
                spike_times=sim_session.spike_times,
                position_time=sim_session.time,
                position=sim_session.position,
                event_overlays=overlays,
            )


@pytest.mark.unit
class TestRunBundleAvailableOutputs:
    """``available_outputs`` reports the optional arrays present."""

    def test_default_predict_lacks_optional_outputs(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        outputs = run_bundles["nl_default"].available_outputs
        assert "acausal_posterior" in outputs
        assert "acausal_state_probabilities" in outputs
        assert "log_likelihood" not in outputs
        assert "predictive_posterior" not in outputs

    def test_loglik_predict_adds_log_likelihood(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        outputs = run_bundles["nl_loglik"].available_outputs
        assert "log_likelihood" in outputs
        assert "predictive_posterior" not in outputs

    def test_all_predict_adds_full_panel_set(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        outputs = run_bundles["nl_all"].available_outputs
        assert "log_likelihood" in outputs
        assert "predictive_posterior" in outputs
