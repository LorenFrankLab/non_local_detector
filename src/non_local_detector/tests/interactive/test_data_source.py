"""Tests for ``InMemoryDecoderDataSource``."""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.tests._simulated_detectors import (
    FittedDetector,
    SimulatedSession,
)
from non_local_detector.visualization.interactive.data_source import (
    InMemoryDecoderDataSource,
)
from non_local_detector.visualization.interactive.view_models.base import RunBundle
from non_local_detector.visualization.interactive.view_models.events import (
    EventOverlay,
)


@pytest.mark.unit
class TestSingleRunDataSource:
    """One-bundle source: from_single + hot-path readers."""

    def test_from_single_default_active_run(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        assert ds.active_run_name == "default"
        assert ds.run_names == ["default"]
        assert ds.active_run is run_bundles["nl_all"]

    def test_window_indices_centered_slice(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        time = ds.time
        t_center = float(time[len(time) // 2])
        # 200 ms window — small but covers > 0 samples at 500 Hz.
        sl = ds.window_indices(t_center=t_center, t_width=0.2)
        assert sl.start < sl.stop
        # Slice covers the center.
        assert sl.start <= len(time) // 2 < sl.stop

    def test_load_posterior_returns_window_slice(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        sl = ds.window_indices(
            t_center=float(ds.time[len(ds.time) // 2]),
            t_width=0.2,
        )
        post = ds.load_posterior(sl)
        n_state_bins = ds.active_run.detector.n_state_bins_
        assert post.ndim == 2
        assert post.shape[0] == sl.stop - sl.start
        assert post.shape[1] == n_state_bins
        assert post.dtype == np.float32

    def test_load_likelihood_succeeds_when_present(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_loglik"])
        sl = ds.window_indices(
            t_center=float(ds.time[len(ds.time) // 2]),
            t_width=0.2,
        )
        loglik = ds.load_likelihood(sl)
        n_state_bins = ds.active_run.detector.n_state_bins_
        assert loglik.shape == (sl.stop - sl.start, n_state_bins)

    def test_load_likelihood_raises_keyerror_when_missing(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_default"])
        sl = ds.window_indices(
            t_center=float(ds.time[len(ds.time) // 2]),
            t_width=0.2,
        )
        with pytest.raises(KeyError, match="log_likelihood"):
            ds.load_likelihood(sl)

    def test_load_predictive_returns_none_when_missing(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_default"])
        sl = ds.window_indices(
            t_center=float(ds.time[len(ds.time) // 2]),
            t_width=0.2,
        )
        assert ds.load_predictive(sl) is None

    def test_load_predictive_returns_array_when_present(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        sl = ds.window_indices(
            t_center=float(ds.time[len(ds.time) // 2]),
            t_width=0.2,
        )
        pred = ds.load_predictive(sl)
        assert pred is not None
        assert pred.shape[0] == sl.stop - sl.start

    def test_slice_at_index_likelihood_raises_when_missing(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_default"])
        with pytest.raises(KeyError, match="log_likelihood"):
            ds.slice_at_index(0, which="likelihood")

    def test_available_outputs_reports_optional_arrays(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds_default = InMemoryDecoderDataSource.from_single(
            run_bundles["nl_default"]
        )
        ds_all = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        assert "log_likelihood" not in ds_default.available_outputs
        assert "log_likelihood" in ds_all.available_outputs


@pytest.mark.unit
class TestMultiRunDataSource:
    """Multi-bundle source: alignment validation + active-run swapping."""

    def test_construct_with_aligned_runs(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        assert ds.run_names == ["nl", "cf", "nsf", "dec"]
        # First run is active by default (insertion order).
        assert ds.active_run_name == "nl"

    def test_set_active_run_switches_active(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        ds.set_active_run("cf")
        assert ds.active_run_name == "cf"
        assert ds.active_run is multi_run_bundles["cf"]

    def test_set_active_run_unknown_name_raises(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        with pytest.raises(ValueError, match="No run named"):
            ds.set_active_run("does_not_exist")

    def test_mismatched_time_grids_raise(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
        cf_fitted: FittedDetector,
    ) -> None:
        # Build a bundle with a truncated time grid.
        truncated = nl_fitted.results.isel(time=slice(0, 100))
        nl_bundle = RunBundle(
            results=truncated,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        cf_bundle = RunBundle(
            results=cf_fitted.results,
            detector=cf_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        with pytest.raises(ValueError, match="same time grid"):
            InMemoryDecoderDataSource({"nl": nl_bundle, "cf": cf_bundle})

    def test_overlay_kind_mismatch_raises(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
        cf_fitted: FittedDetector,
    ) -> None:
        nl_bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
            event_overlays=[
                EventOverlay.points(
                    name="events", times=np.array([1.0, 2.0])
                ),
            ],
        )
        cf_bundle = RunBundle(
            results=cf_fitted.results,
            detector=cf_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
            event_overlays=[
                EventOverlay.intervals(
                    name="events",
                    t_start=np.array([3.0]),
                    t_end=np.array([4.0]),
                ),
            ],
        )
        with pytest.raises(ValueError, match="overlay"):
            InMemoryDecoderDataSource({"nl": nl_bundle, "cf": cf_bundle})

    def test_overlay_missing_in_one_bundle_raises(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
        cf_fitted: FittedDetector,
    ) -> None:
        swr = EventOverlay.points(name="swr", times=np.array([1.0, 2.0]))
        theta = EventOverlay.points(name="theta", times=np.array([1.5]))
        nl_bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
            event_overlays=[swr, theta],
        )
        cf_bundle = RunBundle(
            results=cf_fitted.results,
            detector=cf_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
            event_overlays=[swr],
        )
        with pytest.raises(ValueError, match="overlay"):
            InMemoryDecoderDataSource({"nl": nl_bundle, "cf": cf_bundle})

    def test_set_active_run_revalidates_after_overlay_mutation(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        """``RunBundle`` is mutable; a swap that lands on a run whose
        overlays drifted out of schema must raise instead of silently
        rendering inconsistent navigator state.

        Mirrors the plan's "Reset semantics" guarantee: the data
        source re-validates on the next swap and either accepts (still
        aligned) or raises with the same error message.
        """
        # Snapshot to restore after — fixture is session-scoped.
        original = {
            name: list(b.event_overlays)
            for name, b in multi_run_bundles.items()
        }
        try:
            # Construct under aligned (empty) overlays.
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            # Mutate one bundle's overlays after construction.
            multi_run_bundles["cf"].event_overlays.append(
                EventOverlay.points(
                    name="drifted", times=np.array([1.0, 2.0])
                )
            )
            with pytest.raises(ValueError, match="overlay"):
                ds.set_active_run("cf")
        finally:
            for name, bundle in multi_run_bundles.items():
                bundle.event_overlays[:] = original[name]

    def test_set_active_run_aligned_mutation_succeeds(
        self, multi_run_bundles: dict[str, RunBundle]
    ) -> None:
        """If the same overlay is added to *every* bundle the swap
        is accepted — ``set_active_run`` doesn't reject aligned
        mutations.
        """
        original = {
            name: list(b.event_overlays)
            for name, b in multi_run_bundles.items()
        }
        try:
            for run_name, bundle in multi_run_bundles.items():
                bundle.event_overlays.append(
                    EventOverlay.points(
                        name="aligned",
                        times=np.array([1.0 + 0.1 * len(run_name)]),
                    )
                )
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            for name in ds.run_names:
                ds.set_active_run(name)
        finally:
            for name, bundle in multi_run_bundles.items():
                bundle.event_overlays[:] = original[name]
