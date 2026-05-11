"""Launch the interactive viewer against a synthetic 2D-detector fit.

Run:
    uv run python scripts/inspect_2d_viewer.py

A tiny open-field session (50×50 box, ~8 s, 3 place cells at distinct
``(x, y)`` centers) is fit with ``SortedSpikesDecoder`` on a 5 cm
grid, ``predict`` is called for the full session, and the result is
piped into the Qt viewer's 2D path:

- Left column: raster + state-probability.
- Right column: posterior at cursor, likelihood at cursor, per-cell
  place-field thumbnails for whatever cells fired in the current bin.

Scrub the time slider (or use ``Left``/``Right`` arrow keys) to walk
through the session and watch the 2D images update.
"""

from __future__ import annotations

import numpy as np

from non_local_detector import SortedSpikesDecoder
from non_local_detector.environment import Environment
from non_local_detector.visualization.interactive import launch_qt


def main() -> int:
    rng = np.random.default_rng(0)
    sampling_frequency = 100
    n_time = 800
    time = np.arange(n_time) / sampling_frequency

    # Animal traces a back-and-forth path across a 50 x 50 cm box.
    x = 25.0 + 20.0 * np.sin(2 * np.pi * time / 4.0)
    y = 25.0 + 15.0 * np.cos(2 * np.pi * time / 3.0)
    position = np.column_stack([x, y])

    # Three place cells with Gaussian fields at distinct (x, y) centers.
    centers_xy = np.array([[15.0, 15.0], [35.0, 35.0], [25.0, 25.0]])
    place_std = 6.0
    spike_times: list[np.ndarray] = []
    for cx, cy in centers_xy:
        rate = 12.0 * np.exp(
            -((position[:, 0] - cx) ** 2 + (position[:, 1] - cy) ** 2)
            / (2 * place_std**2)
        )
        spikes = rng.poisson(rate / sampling_frequency)
        spike_times.append(time[spikes > 0])

    env = Environment(
        place_bin_size=5.0,
        position_range=((0.0, 50.0), (0.0, 50.0)),
    )
    detector = SortedSpikesDecoder(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 4.0,
            "block_size": 4096,
        },
        environments=[env],
    )
    detector.fit(position_time=time, position=position, spike_times=spike_times)
    results = detector.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        return_outputs="all",
    )

    return launch_qt(
        detector=detector,
        results=results,
        spike_times=spike_times,
        position=position,
        position_time=time,
        t_width=0.5,
    )


if __name__ == "__main__":
    raise SystemExit(main())
