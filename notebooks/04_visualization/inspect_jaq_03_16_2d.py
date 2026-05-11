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
# # Inspect Jaq_03_16 — 2D Non-Local viewer
#
# Real-data smoke for the 2D viewer path. Loads the full Jaq_03_16
# session, fits a `NonLocalSortedSpikesDetector` against the raw
# head position (`nose_x` / `nose_y`) using non-ripple periods as
# the training set, predicts over the whole session, and opens the
# interactive viewer.
#
# Uses raw head position rather than `projected_*_position`: the
# projected columns are the linearization-back-to-graph XY (the
# W-track skeleton), which produces single-bin-wide arms and a
# very sparse interior mask. The head position fills the actual
# 2D maze surface the animal occupied.
#
# Requires (all in `~/Downloads/`):
# - `Jaq_03_16_position_info.pkl` — 500 Hz position DataFrame
#   (`nose_x` / `nose_y`).
# - `Jaq_03_16_spikes.pkl` — 500 Hz spike-count DataFrame, one
#   column per cell.
# - `Jaq_03_16_is_ripple.pkl` — 500 Hz boolean DataFrame marking
#   ripple-event bins; the inverse is the training mask.
# - `[viewer]` extra installed.

# %%
import pickle
from pathlib import Path

import numpy as np

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.visualization.interactive import launch_qt

# %% [markdown]
# ## 1. Load Jaq_03_16
#
# Position, spike counts, and ripple labels all live on the same
# 500 Hz time index (timedelta). Normalize to relative seconds.

# %%
DATA_DIR = Path.home() / "Downloads"
position_pkl = DATA_DIR / "Jaq_03_16_position_info.pkl"
spikes_pkl = DATA_DIR / "Jaq_03_16_spikes.pkl"
is_ripple_pkl = DATA_DIR / "Jaq_03_16_is_ripple.pkl"

with open(position_pkl, "rb") as f:
    position_df = pickle.load(f)
with open(spikes_pkl, "rb") as f:
    spikes_df = pickle.load(f)
with open(is_ripple_pkl, "rb") as f:
    is_ripple_df = pickle.load(f)

t0 = position_df.index[0]
time_seconds = (position_df.index - t0).total_seconds().to_numpy()
position_2d = position_df[["nose_x", "nose_y"]].to_numpy(
    dtype=np.float64
)
is_ripple = is_ripple_df.iloc[:, 0].to_numpy(dtype=bool)

print(f"Session duration: {time_seconds[-1]:.1f} s, n_samples: {time_seconds.size}")
print(f"n_cells: {spikes_df.shape[1]}")
print(
    f"Position bounds (cm): x=[{position_df['nose_x'].min():.1f}, "
    f"{position_df['nose_x'].max():.1f}], "
    f"y=[{position_df['nose_y'].min():.1f}, "
    f"{position_df['nose_y'].max():.1f}]"
)
print(
    f"Ripple bins: {is_ripple.sum()} / {is_ripple.size} "
    f"({100 * is_ripple.mean():.2f}%)"
)

# %% [markdown]
# ## 2. Convert spike-count bins → per-cell spike-time arrays
#
# The detector expects `list[np.ndarray]` of spike times in seconds.
# Each cell's column holds counts per 2 ms bin; emit one timestamp
# per spike (multiple if `count > 1`).

# %%
spike_times: list[np.ndarray] = []
for col in spikes_df.columns:
    counts = spikes_df[col].to_numpy()
    nonzero = np.where(counts > 0)[0]
    # Repeat the bin timestamp for cells that fired multiple times
    # in a single 2 ms bin (rare but possible).
    times = np.repeat(time_seconds[nonzero], counts[nonzero].astype(int))
    spike_times.append(times.astype(np.float64))

total_spikes = sum(len(st) for st in spike_times)
active_cells = sum(1 for st in spike_times if st.size > 0)
print(f"Total spikes: {total_spikes}")
print(f"Cells with ≥1 spike: {active_cells} / {len(spike_times)}")

# %% [markdown]
# ## 3. Fit a 2D Non-Local detector
#
# `NonLocalSortedSpikesDetector` has four states: `Local`,
# `No-Spike`, `Non-Local Continuous`, and `Non-Local Fragmented`.
# The training mask is `~is_ripple` so place fields are estimated
# from awake-locomotion periods rather than from putative replay
# bins.
#
# `place_bin_size=3.5` gives ~30×28 bins over the projected track
# (~2× finer than the 5 cm default — useful for inspecting
# multimodal posterior structure). `local_position_std=1.0`
# gates the local state to a tight neighborhood of the animal;
# tune up if the local prior is too restrictive for your data.

# %%
env = Environment(
    place_bin_size=3.5,
    position_range=(
        (
            float(position_df["nose_x"].min()) - 5,
            float(position_df["nose_x"].max()) + 5,
        ),
        (
            float(position_df["nose_y"].min()) - 5,
            float(position_df["nose_y"].max()) + 5,
        ),
    ),
)
detector = NonLocalSortedSpikesDetector(
    sorted_spikes_algorithm="sorted_spikes_kde",
    sorted_spikes_algorithm_params={
        "position_std": 6.0,
        "block_size": 4096,
    },
    non_local_position_penalty=1.0,
    non_local_penalty_std=5.0,
    local_position_std=1.0,
    environments=[env],
)

# %% [markdown]
# `estimate_parameters` is the EM-fit entry point. Setting
# `return_outputs=None` tells it to skip materializing predictions
# during the fit pass — we run a dedicated `predict` below for the
# viewer-side outputs.

# %%
detector.estimate_parameters(
    position_time=time_seconds,
    position=position_2d,
    spike_times=spike_times,
    is_training=~is_ripple,
    time=time_seconds,
    return_outputs=None,
)

# %% [markdown]
# ## 4. Predict over the full session
#
# `return_outputs="all"` materializes `acausal_posterior`,
# `log_likelihood`, `predictive_posterior`, and the state-probability
# variants — the full set the 2D viewer consumes (posterior +
# likelihood at-cursor images, state-probability lines on the left).

# %%
results = detector.predict(
    spike_times=spike_times,
    time=time_seconds,
    position=position_2d,
    position_time=time_seconds,
    return_outputs="all",
)
print("results vars:", list(results.data_vars))
print(
    f"acausal_posterior shape: {results.acausal_posterior.shape}, "
    f"finite fraction: {np.isfinite(results.acausal_posterior.values).mean():.3f}"
)

# %% [markdown]
# ## 5. Launch the viewer
#
# `launch_qt` dispatches on `PositionGrid.ndim` — for 2D detectors
# the right column carries the posterior + likelihood images at
# cursor and per-cell place-field thumbnails for cells active in
# the cursor bin. State probabilities (Local / No-Spike / Non-Local
# Continuous / Non-Local Fragmented) appear in the left column
# under the raster. Scrub the time slider or use `Left` / `Right`
# arrows to walk the session; `[` / `]` resizes the visible window.

# %%
launch_qt(
    detector=detector,
    results=results,
    spike_times=spike_times,
    position=position_2d,
    position_time=time_seconds,
    t_width=0.5,
)
