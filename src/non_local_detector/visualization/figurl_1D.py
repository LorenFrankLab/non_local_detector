try:
    from typing import Optional

    import numpy as np
    import sortingview.views as vv
    import sortingview.views.franklab as vvf
    import xarray as xr

    from non_local_detector.visualization.static import get_multiunit_firing_rate

    def discretize_and_trim(series: xr.DataArray) -> xr.DataArray:
        """Discretizes and trims a series for visualization.

        Parameters
        ----------
        series : xr.DataArray

        Returns
        -------
        xr.DataArray
        """
        discretized = np.multiply(series, 255).astype(np.uint8)  # type: ignore
        stacked = discretized.stack(unified_index=["time", "position"])
        return stacked.where(stacked > 0, drop=True).astype(np.uint8)

    def get_observations_per_time(
        trimmed_posterior: xr.DataArray, base_data: xr.Dataset
    ) -> np.ndarray:
        """Get the number of observations per time.

        Parameters
        ----------
        trimmed_posterior : xr.DataArray
        base_data : xr.Dataset

        Returns
        -------
        observations : np.ndarray
        """
        times, counts = np.unique(trimmed_posterior.time.values, return_counts=True)
        indexed_counts = xr.DataArray(counts, coords={"time": times})
        _, good_counts = xr.align(base_data.time, indexed_counts, join="left", fill_value=0)  # type: ignore

        return good_counts.values.astype(np.uint8)

    def get_sampling_freq(times: np.ndarray) -> float:
        """Get the sampling frequency from a series of times.

        Parameters
        ----------
        times : np.ndarray

        Returns
        -------
        sampling_rate : float
        """
        round_times = np.floor(1000 * times)
        median_delta_t_ms = np.median(np.diff(round_times)).item()
        return 1000 / median_delta_t_ms  # from time-delta to Hz

    def get_trimmed_bin_center_index(
        place_bin_centers: np.ndarray, trimmed_place_bin_centers: np.ndarray
    ) -> np.ndarray:
        """Get the index of the trimmed bin centers in the full bin centers.

        Parameters
        ----------
        place_bin_centers : np.ndarray, shape (n_position_bins, )
        trimmed_place_bin_centers : np.ndarray, shape (n_trimmed_position_bins, )

        Returns
        -------
        bin_ind : np.ndarray, shape (n_trimmed_position_bins, )
        """
        return np.searchsorted(
            place_bin_centers, trimmed_place_bin_centers, side="left"
        ).astype(np.uint16)

    def create_1D_decode_view(
        posterior: xr.DataArray,
        linear_position: Optional[np.ndarray] = None,
        ref_time_sec: Optional[float] = None,
    ) -> vvf.DecodedLinearPositionData:
        """Creates a view of an interactive heatmap of position vs. time.

        Parameters
        ----------
        posterior : xr.DataArray, shape (n_time, n_position_bins)
        linear_position : np.ndarray, shape (n_time, ), optional
        ref_time_sec : np.float64, optional
            Reference time for the purpose of offsetting the start time

        Returns
        -------
        view : vvf.DecodedLinearPositionData

        """
        if linear_position is not None:
            linear_position = np.asarray(linear_position).squeeze()

        trimmed_posterior = discretize_and_trim(posterior)
        observations_per_time = get_observations_per_time(trimmed_posterior, posterior)
        sampling_freq = get_sampling_freq(posterior.time)
        start_time_sec = posterior.time.values[0]
        if ref_time_sec is not None:
            start_time_sec = start_time_sec - ref_time_sec

        trimmed_bin_center_index = get_trimmed_bin_center_index(
            posterior.position.values, trimmed_posterior.position.values
        )

        return vvf.DecodedLinearPositionData(
            values=trimmed_posterior.values,
            positions=trimmed_bin_center_index,
            frame_bounds=observations_per_time,
            positions_key=posterior.position.values.astype(np.float32),
            observed_positions=linear_position,
            start_time_sec=start_time_sec,
            sampling_frequency=sampling_freq,
        )

    def create_interactive_1D_decoding_figurl(
        position: np.ndarray,
        speed: np.ndarray,
        spike_times: list[np.ndarray],
        results: xr.Dataset,
        view_height: int = 800,
    ) -> str:
        posterior = (
            results.acausal_posterior.unstack("state_bins")
            .drop_sel(state=["Local", "No-Spike"], errors="ignore")
            .sum("state")
        )
        posterior = posterior / posterior.sum("position")
        decode_view = create_1D_decode_view(
            posterior=posterior,
            linear_position=position,
        )

        probability_view = vv.TimeseriesGraph()
        COLOR_CYCLE = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        for state, color in zip(results.states.values, COLOR_CYCLE):
            probability_view.add_line_series(
                name=state,
                t=np.asarray(results.time),
                y=np.asarray(
                    results.sel(states=state).acausal_state_probabilities,
                    dtype=np.float32,
                ),
                color=color,
                width=1,
            )

        speed_view = vv.TimeseriesGraph().add_line_series(
            name="Speed [cm/s]",
            t=np.asarray(results.time),
            y=np.asarray(speed, dtype=np.float32),
            color="black",
            width=1,
        )
        multiunit_firing_rate = get_multiunit_firing_rate(
            spike_times, results.time.values
        )
        multiunit_firing_rate_view = vv.TimeseriesGraph().add_line_series(
            name="Multiunit Rate [spikes/s]",
            t=np.asarray(results.time.values),
            y=np.asarray(multiunit_firing_rate, dtype=np.float32),
            color="black",
            width=1,
        )
        vertical_panel_content = [
            vv.LayoutItem(decode_view, stretch=3, title="Decode"),
            vv.LayoutItem(probability_view, stretch=1, title="Probability of State"),
            vv.LayoutItem(speed_view, stretch=1, title="Speed"),
            vv.LayoutItem(multiunit_firing_rate_view, stretch=1, title="Multiunit"),
        ]

        view = vv.Box(
            direction="horizontal",
            show_titles=True,
            height=view_height,
            items=[
                vv.LayoutItem(
                    vv.Box(
                        direction="vertical",
                        show_titles=True,
                        items=vertical_panel_content,
                    )
                ),
            ],
        )

        return view.url(label="1D Decoding")

except ImportError:
    pass
