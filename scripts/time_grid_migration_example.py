"""Run a small pipeline with irregular tracking and independent decode intervals.

uv run python scripts/time_grid_migration_example.py
"""

import numpy as np
import xarray as xr

from non_local_detector import Environment, SortedSpikesDecoder, calculate_time_edges


def main():
    rng = np.random.default_rng(7)
    intervals = np.array([[0.0, 2.0], [3.0, 5.0]])
    # Explicitly declared continuous acquisition segments, with camera jitter.
    position_time = np.concatenate(
        [np.arange(start + 0.01, stop, 1 / 30) for start, stop in intervals]
    )
    position_time += rng.uniform(-0.003, 0.003, len(position_time))
    position = (50 + 40 * np.sin(position_time * 3))[:, None]
    position[15:18] = np.nan
    spike_times = [np.array([0.11, 0.13, 0.19, 0.57, 1.1, 3.12, 3.19, 4.2])]
    detector = SortedSpikesDecoder(
        environments=Environment(place_bin_size=10, position_range=((0, 100),)),
        infer_track_interior=False,
    ).fit(
        position_time,
        position,
        spike_times,
        encoding_time_range=[0.0, 5.0],
        valid_position_intervals=intervals,
    )
    pieces = []
    decode_intervals = [[0.1, 0.7], [3.1, 3.3]]
    for sequence_id, (start, stop) in enumerate(decode_intervals):
        edges = calculate_time_edges([start, stop], sampling_frequency=500)
        # Explicit upstream ownership; this convention also prevents duplicate
        # boundary spikes when adjacent independent sequences are concatenated.
        shared_stop = (
            sequence_id + 1 < len(decode_intervals)
            and decode_intervals[sequence_id + 1][0] == stop
        )
        selected_spikes = [
            s[(s >= edges[0]) & (s < edges[-1] if shared_stop else s <= edges[-1])]
            for s in spike_times
        ]
        # Give each independent call its own tracking segment. This prevents
        # interpolation between disconnected finite endpoints. Preserve NaNs
        # within a segment so prediction can mask affected observation bins.
        segment_start, segment_stop = next(
            interval
            for interval in intervals
            if interval[0] <= edges[0] and edges[-1] <= interval[1]
        )
        segment = (position_time >= segment_start) & (position_time <= segment_stop)
        segment_time, segment_position = position_time[segment], position[segment]
        # Conservative observed support excludes endpoint extension. Encoding
        # support remains independently defined by valid_position_intervals.
        observed = (edges[:-1] >= segment_time[0]) & (edges[1:] <= segment_time[-1])
        result = detector.predict(
            selected_spikes,
            time_edges=edges,
            position_time=segment_time,
            position=segment_position,
            is_missing=~observed,
        )
        result = result.assign_coords(
            sequence_id=("time", np.full(result.sizes["time"], sequence_id))
        )
        pieces.append(result)
    results = xr.concat(pieces, dim="time")
    assert results.sizes["time"] == 400
    assert results.time_bin_end_inclusive.sum() == 2
    np.testing.assert_allclose(
        results.acausal_posterior.sum("state_bins"), 1.0, rtol=1e-6
    )
    print(
        f"Decoded {results.sizes['time']} bins in 2 independent sequences; "
        f"{int(results.is_missing.sum())} bins have missing observations."
    )


if __name__ == "__main__":
    main()
