"""Tests for ``InMemoryDecoderDataSource``."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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

    def test_window_indices_huge_width_covers_full_session(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        time = ds.time
        sl = ds.window_indices(t_center=float(time[0]), t_width=1.0e9)
        assert sl == slice(0, len(time))

    @pytest.mark.parametrize("delta", [0.001, 0.1, 100.0])
    def test_window_indices_clamps_when_center_past_end(
        self, run_bundles: dict[str, RunBundle], delta: float
    ) -> None:
        """Past-end ``t_center`` snaps to a t_width-wide rightmost window
        (NOT a partial slice; NOT collapsed to a single sample). Adjacent
        cases (delta ~ bin width) and far cases (delta >> session) must
        both snap to the same edge window."""
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        time = ds.time
        t_width = 1.0
        sl = ds.window_indices(t_center=float(time[-1]) + delta, t_width=t_width)
        assert sl.stop == len(time)
        assert sl.start < sl.stop
        # Width contract: rightmost window must be t_width-wide, not partial.
        # Allow one bin of slack for searchsorted boundary effects.
        bin_width = float(np.median(np.diff(time)))
        duration = float(time[sl.stop - 1] - time[sl.start])
        assert duration >= t_width - bin_width

    @pytest.mark.parametrize("delta", [0.001, 0.1, 100.0])
    def test_window_indices_clamps_when_center_before_start(
        self, run_bundles: dict[str, RunBundle], delta: float
    ) -> None:
        """Before-start ``t_center`` snaps to a t_width-wide leftmost
        window (NOT a partial slice). Adjacent and far cases match."""
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        time = ds.time
        t_width = 1.0
        sl = ds.window_indices(t_center=float(time[0]) - delta, t_width=t_width)
        assert sl.start == 0
        assert sl.start < sl.stop
        bin_width = float(np.median(np.diff(time)))
        duration = float(time[sl.stop - 1] - time[sl.start])
        assert duration >= t_width - bin_width

    def test_window_indices_no_overshoot_at_session_end(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        time = ds.time
        sl = ds.window_indices(t_center=float(time[-1]), t_width=2.0)
        assert sl.stop == len(time)
        assert sl.start < sl.stop

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

    def test_load_state_probabilities_returns_window_slice(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        """Loader must return state probabilities for every predict variant."""
        for variant in ("nl_default", "nl_loglik", "nl_all"):
            ds = InMemoryDecoderDataSource.from_single(run_bundles[variant])
            sl = ds.window_indices(
                t_center=float(ds.time[len(ds.time) // 2]),
                t_width=0.2,
            )
            probs = ds.load_state_probabilities(sl)
            assert probs.ndim == 2
            assert probs.shape[0] == sl.stop - sl.start
            assert probs.shape[1] == len(ds.active_run.detector.state_names)

    def test_load_state_probabilities_expands_single_state_1d_results(
        self,
        multi_run_bundles: dict[str, RunBundle],
    ) -> None:
        """Single-state NetCDF round-trips may load state probabilities as 1D."""
        bundle = multi_run_bundles["dec"]
        results = bundle.results.copy()
        results["acausal_state_probabilities"] = xr.DataArray(
            np.ones(results.sizes["time"]),
            dims=("time",),
            coords={"time": results["time"].values},
        )
        ds = InMemoryDecoderDataSource.from_single(
            RunBundle(
                results=results,
                detector=bundle.detector,
                spike_times=bundle.spike_times,
                position_time=bundle.position_time,
                position=bundle.position,
                speed=bundle.speed,
            )
        )
        sl = slice(2, 8)
        probs = ds.load_state_probabilities(sl)
        assert probs.shape == (sl.stop - sl.start, 1)
        np.testing.assert_array_equal(probs[:, 0], 1.0)

    def test_slice_at_index_likelihood_raises_when_missing(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_default"])
        with pytest.raises(KeyError, match="log_likelihood"):
            ds.slice_at_index(0, which="likelihood")

    def test_available_outputs_reports_optional_arrays(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds_default = InMemoryDecoderDataSource.from_single(run_bundles["nl_default"])
        ds_all = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        assert "log_likelihood" not in ds_default.available_outputs
        assert "log_likelihood" in ds_all.available_outputs

    def test_event_index_maps_spikes_to_left_edge_bins(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        event = ds.spike_event_at(0)
        assert event.time_index == int(
            np.searchsorted(ds.time, event.time, side="right") - 1
        )
        event_ids = ds.event_ids_at_bin(event.time_index)
        assert event.event_id in set(event_ids.tolist())
        assert event.cell_id in set(ds.event_index.cell_ids[event_ids].tolist())

    def test_event_index_drops_spikes_outside_decoder_range(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        """``from_spike_times`` filters per-cell BEFORE the flat concat.

        Pads each cell's spikes with sentinels far before / after the
        decoder time grid; the resulting index must match the
        unpadded version event-for-event (same total count, same
        ``cell_event_ids`` lengths). This pins the per-cell filter:
        a regression that drops the filter would let the sentinels
        through and bump every count.
        """
        from non_local_detector.visualization.interactive.view_models.base import (
            SpikeEventIndex,
        )

        bundle = run_bundles["nl_all"]
        time = np.asarray(bundle.results["time"].values)
        clean = SpikeEventIndex.from_spike_times(bundle.spike_times, time)

        far_lo = float(time[0]) - 1000.0
        far_hi = float(time[-1]) + 1000.0
        padded_spikes = [
            np.concatenate([[far_lo], st, [far_hi]]) for st in bundle.spike_times
        ]
        padded = SpikeEventIndex.from_spike_times(padded_spikes, time)

        assert padded.n_events == clean.n_events
        for clean_ids, padded_ids in zip(
            clean.cell_event_ids, padded.cell_event_ids, strict=True
        ):
            assert clean_ids.size == padded_ids.size

    def test_event_index_nearest_handles_float32_round_trip(
        self, run_bundles: dict[str, RunBundle]
    ) -> None:
        """Float32-rounded ``t`` (raster click semantic) still resolves.

        Strict-equal lookup fails under the rounding; the nearest
        helper picks the right event within ``atol``.
        """
        ds = InMemoryDecoderDataSource.from_single(run_bundles["nl_all"])
        event = ds.spike_event_at(0)
        # pyqtgraph ScatterPlotItem stores positions as float32.
        rounded_t = float(np.float32(event.time))
        assert ds.event_index.event_id_for_cell_time(event.cell_id, rounded_t) is None
        assert (
            ds.event_index.nearest_event_id_for_cell_time(event.cell_id, rounded_t)
            == event.event_id
        )
        # Beyond tolerance → None.
        assert (
            ds.event_index.nearest_event_id_for_cell_time(
                event.cell_id, event.time + 1.0
            )
            is None
        )


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
                EventOverlay.points(name="events", times=np.array([1.0, 2.0])),
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
            name: list(b.event_overlays) for name, b in multi_run_bundles.items()
        }
        try:
            # Construct under aligned (empty) overlays.
            ds = InMemoryDecoderDataSource(multi_run_bundles)
            # Mutate one bundle's overlays after construction.
            multi_run_bundles["cf"].event_overlays.append(
                EventOverlay.points(name="drifted", times=np.array([1.0, 2.0]))
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
            name: list(b.event_overlays) for name, b in multi_run_bundles.items()
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


@pytest.mark.unit
class TestPositionLoad:
    """``load_position`` interpolation + per-run cache + 2D guard."""

    def test_load_position_aligns_to_decoder_time_grid(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        """1D position is interpolated onto the decoder ``time`` grid."""
        bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        ds = InMemoryDecoderDataSource.from_single(bundle)
        sl = ds.window_indices(t_center=float(ds.time[ds.n_time // 2]), t_width=1.0)
        position_window = ds.load_position(sl)
        assert position_window is not None
        assert position_window.ndim == 1
        assert position_window.size == sl.stop - sl.start
        assert np.all(np.isfinite(position_window))

    def test_load_position_returns_none_when_post_construction_drop(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        """``RunBundle`` is mutable; clearing ``position`` after
        construction makes ``load_position`` return ``None``.

        Mirrors the v1 contract: ``RunBundle`` requires position at
        construction (``__post_init__`` validates 1D vs 2D against the
        detector). The "no-trace" path is reached when the bundle
        loses its position post-hoc, or when the position is multi-
        dim (v3 2D detectors). Either way the data source returns
        ``None`` and the panels skip drawing the trace.
        """
        bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        ds = InMemoryDecoderDataSource.from_single(bundle)
        # Flip the bundle's position to None — bypasses re-validation.
        bundle.position = None
        # Cache wasn't populated yet, so the next call uses the new
        # state and returns None.
        sl = ds.window_indices(t_center=float(ds.time[ds.n_time // 2]), t_width=1.0)
        assert ds.load_position(sl) is None

    def test_load_position_returns_none_for_2d_position(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        """2D position → ``None``; v1 heatmap trace is 1D-only.

        ``RunBundle.__post_init__`` rejects 2D position against a 1D
        detector, so we have to mutate after construction to exercise
        the data source's 2D guard. v3+ ships a 2D detector path that
        will accept this directly.
        """
        bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=sim_session.time,
            position=sim_session.position,
        )
        ds = InMemoryDecoderDataSource.from_single(bundle)
        bundle.position = np.column_stack([sim_session.position, sim_session.position])
        sl = ds.window_indices(t_center=float(ds.time[ds.n_time // 2]), t_width=1.0)
        assert ds.load_position(sl) is None

    def test_load_position_interpolates_when_grids_differ(
        self,
        sim_session: SimulatedSession,
        nl_fitted: FittedDetector,
    ) -> None:
        """Decoder time != ``position_time``: interpolation lands on the
        decoder grid with no NaN at the head/tail (np.interp clips to edges)."""
        # Subsample position to half the decoder rate, so position_time
        # and decoder time genuinely differ in length.
        coarse_pt = sim_session.time[::2]
        coarse_pos = sim_session.position[::2]
        bundle = RunBundle(
            results=nl_fitted.results,
            detector=nl_fitted.detector,
            spike_times=sim_session.spike_times,
            position_time=coarse_pt,
            position=coarse_pos,
        )
        ds = InMemoryDecoderDataSource.from_single(bundle)
        # Pull the full session via a window covering it.
        sl = slice(0, ds.n_time)
        full = ds.load_position(sl)
        assert full is not None
        assert full.size == ds.n_time
        assert np.all(np.isfinite(full))
        # Edges must equal the position-time edges (np.interp clips).
        np.testing.assert_allclose(full[0], coarse_pos[0], rtol=0, atol=1e-3)
        np.testing.assert_allclose(full[-1], coarse_pos[-1], rtol=0, atol=1e-3)

    def test_load_position_caches_per_active_run(
        self,
        multi_run_bundles: dict[str, RunBundle],
    ) -> None:
        """Repeat calls return the same array object; swap re-derives."""
        ds = InMemoryDecoderDataSource(multi_run_bundles)
        first = ds._position_at_decoder_time()
        assert first is not None
        second = ds._position_at_decoder_time()
        # Cached: same ndarray object both calls.
        assert second is first
        # Swap: cache key changes, so the next call may return a fresh
        # array — the contract is that it's still finite + same length.
        next_run = next(name for name in ds.run_names if name != ds.active_run_name)
        ds.set_active_run(next_run)
        after_swap = ds._position_at_decoder_time()
        assert after_swap is not None
        assert after_swap.size == ds.n_time
