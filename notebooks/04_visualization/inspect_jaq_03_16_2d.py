# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: non-local-detector
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inspect Jaq_03_16 — 2D viewer smoke
#
# Real-data smoke for the 2D-detector viewer path. Loads the
# `Jaq_03_16` session from `~/Downloads/`, fits a 2D
# `SortedSpikesDecoder` on a short slice, and opens the interactive
# viewer.
#
# Requires:
# - `~/Downloads/Jaq_03_16_position_info.pkl` — 500 Hz position
#   DataFrame with `projected_x_position` / `projected_y_position`.
# - `~/Downloads/Jaq_03_16_spikes.pkl` — 500 Hz spike-count DataFrame
#   with one column per cell.
# - `[viewer]` extra installed.

# %%
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from non_local_detector import SortedSpikesDecoder
from non_local_detector.environment import Environment
from non_local_detector.visualization.interactive import launch_qt

# %% [markdown]
# ## 1. Load Jaq_03_16
#
# Position + spikes are aligned on the same 500 Hz time index
# (timedelta). Convert to relative seconds so the decoder sees a
# plain monotonic `float64` time grid.

# %%
DATA_DIR = Path.home() / "Downloads"
position_pkl = DATA_DIR / "Jaq_03_16_position_info.pkl"
spikes_pkl = DATA_DIR / "Jaq_03_16_spikes.pkl"

with open(position_pkl, "rb") as f:
    position_df = pickle.load(f)
with open(spikes_pkl, "rb") as f:
    spikes_df = pickle.load(f)

# Both DataFrames are indexed by timedelta starting around
# ``6:13:09``. Normalize to seconds from the first sample.
t0 = position_df.index[0]
time_seconds = (position_df.index - t0).total_seconds().to_numpy()
print(f"Session duration: {time_seconds[-1]:.1f} s, n_samples: {time_seconds.size}")
print(f"n_cells: {spikes_df.shape[1]}")
print(
    f"Position bounds (cm): x=[{position_df['projected_x_position'].min():.1f}, "
    f"{position_df['projected_x_position'].max():.1f}], "
    f"y=[{position_df['projected_y_position'].min():.1f}, "
    f"{position_df['projected_y_position'].max():.1f}]"
)

# %% [markdown]
# ## 2. Slice to a manageable window
#
# 15 minutes × 500 Hz × 104 cells is slow to fit interactively. Take
# the first 120 s for the smoke test. Extend `slice_end_s` to fit
# longer windows once you've confirmed the path works.

# %%
slice_end_s = 120.0
slice_mask = time_seconds <= slice_end_s
slice_time = time_seconds[slice_mask]

position_2d = position_df.loc[
    slice_mask, ["projected_x_position", "projected_y_position"]
].to_numpy(dtype=np.float64)
print(f"slice: {slice_time.size} samples, {slice_time[-1]:.1f} s")
print(f"position_2d shape: {position_2d.shape}")

# %% [markdown]
# ## 3. Convert spike-count bins → per-cell spike-time arrays
#
# The decoder expects `list[np.ndarray]` of spike times in seconds.
# Each cell's column holds counts per 2 ms bin; emit one timestamp
# per spike (multiple if `count > 1`).

# %%
slice_spikes_df = spikes_df.loc[slice_mask]
spike_times: list[np.ndarray] = []
for col in slice_spikes_df.columns:
    counts = slice_spikes_df[col].to_numpy()
    nonzero = np.where(counts > 0)[0]
    # Repeat the bin timestamp for cells that fired multiple times in
    # a single 2 ms bin (rare but possible).
    times = np.repeat(slice_time[nonzero], counts[nonzero].astype(int))
    spike_times.append(times.astype(np.float64))

total_spikes = sum(len(st) for st in spike_times)
active_cells = sum(1 for st in spike_times if st.size > 0)
print(f"Total spikes in slice: {total_spikes}")
print(f"Cells with ≥1 spike: {active_cells} / {len(spike_times)}")

# %% [markdown]
# ## 4. Fit a 2D SortedSpikesDecoder
#
# 5 cm bins over the projected track produces a manageable grid.
# `position_std=6.0` gives a moderate KDE smoothing; bump up if
# place fields look noisy or down if they're over-smoothed.

# %%
env = Environment(
    place_bin_size=5.0,
    position_range=(
        (
            float(position_df["projected_x_position"].min()) - 5,
            float(position_df["projected_x_position"].max()) + 5,
        ),
        (
            float(position_df["projected_y_position"].min()) - 5,
            float(position_df["projected_y_position"].max()) + 5,
        ),
    ),
)
detector = SortedSpikesDecoder(
    sorted_spikes_algorithm="sorted_spikes_kde",
    sorted_spikes_algorithm_params={
        "position_std": 6.0,
        "block_size": 4096,
    },
    environments=[env],
)
detector.fit(
    position_time=slice_time,
    position=position_2d,
    spike_times=spike_times,
)

# %% [markdown]
# ## 5. Predict on the same window
#
# `return_outputs="all"` gives the viewer both posterior + likelihood
# (the two at-cursor 2D images).

# %%
results = detector.predict(
    spike_times=spike_times,
    time=slice_time,
    position=position_2d,
    position_time=slice_time,
    return_outputs="all",
)
print("results vars:", list(results.data_vars))
print(
    f"acausal_posterior shape: {results.acausal_posterior.shape}, "
    f"finite fraction: {np.isfinite(results.acausal_posterior.values).mean():.3f}"
)

# %% [markdown]
# ## 6. Launch the viewer
#
# `launch_qt` dispatches on `PositionGrid.ndim` — for 2D detectors
# the right column carries posterior + likelihood images at cursor
# plus per-cell 2D place-field thumbnails. Scrub the time slider or
# use `Left` / `Right` arrows to walk through the session.

# %%
launch_qt(
    detector=detector,
    results=results,
    spike_times=spike_times,
    position=position_2d,
    position_time=slice_time,
    t_width=0.5,
)
