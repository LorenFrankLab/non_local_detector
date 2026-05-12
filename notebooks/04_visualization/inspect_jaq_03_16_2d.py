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
import os
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.visualization.interactive import launch_qt


def running_in_notebook() -> bool:
    """Return True when executing inside a Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"

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
position_2d = position_df[["nose_x", "nose_y"]].to_numpy(dtype=np.float64)
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
# ## 1.1. Infer projected-track geometry for the optional 2D→1D view
#
# The detector is fit in raw 2D head-position coordinates, but this
# session also ships `projected_*_position`, `track_segment_id`, and
# `linear_position`. Use those columns to reconstruct the W-track
# graph-linearization geometry needed by the optional projected-1D
# viewer column.

# %%
def infer_projection_graph_from_position_df(position_df):
    """Infer linearization geometry from projected Jaq position columns."""
    required = {
        "linear_position",
        "track_segment_id",
        "projected_x_position",
        "projected_y_position",
    }
    missing = sorted(required - set(position_df.columns))
    if missing:
        raise ValueError(f"position_df is missing projected-track columns: {missing!r}")

    graph = nx.Graph()
    edge_rows = []
    for segment_id in sorted(position_df["track_segment_id"].dropna().unique()):
        segment = position_df.loc[
            position_df["track_segment_id"] == segment_id,
            ["linear_position", "projected_x_position", "projected_y_position"],
        ].dropna()
        if segment.empty:
            continue

        start = segment.loc[segment["linear_position"].idxmin()]
        stop = segment.loc[segment["linear_position"].idxmax()]
        start_node = f"{int(segment_id)}_start"
        stop_node = f"{int(segment_id)}_stop"
        start_xy = (
            float(start["projected_x_position"]),
            float(start["projected_y_position"]),
        )
        stop_xy = (
            float(stop["projected_x_position"]),
            float(stop["projected_y_position"]),
        )
        min_linear = float(start["linear_position"])
        max_linear = float(stop["linear_position"])
        distance = max(
            max_linear - min_linear,
            float(np.linalg.norm(np.subtract(stop_xy, start_xy))),
        )

        graph.add_node(start_node, pos=start_xy)
        graph.add_node(stop_node, pos=stop_xy)
        graph.add_edge(
            start_node,
            stop_node,
            distance=distance,
            edge_id=int(segment_id),
        )
        edge_rows.append((min_linear, max_linear, (start_node, stop_node)))

    if not edge_rows:
        raise ValueError("No finite projected-track segments found.")

    edge_rows.sort(key=lambda item: item[0])
    edge_order = [edge for _, _, edge in edge_rows]
    edge_spacing = [
        max(0.0, next_min - previous_max)
        for (_, previous_max, _), (next_min, _, _) in zip(
            edge_rows[:-1], edge_rows[1:], strict=False
        )
    ]
    return graph, edge_order, edge_spacing


projection_track_graph, projection_edge_order, projection_edge_spacing = (
    infer_projection_graph_from_position_df(position_df)
)
print(
    "Projection graph: "
    f"{projection_track_graph.number_of_edges()} edges, "
    f"edge spacing={np.round(projection_edge_spacing, 3).tolist()}"
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
# Geometry — `place_bin_size=3.5` on the head-position range
# (`nose_x` / `nose_y`, ~120 × ~107 cm) gives ~40×36 bins,
# ~455 of them inside the inferred track interior (~32%).
# `Environment` infers `position_range` from the position passed
# to `estimate_parameters`, so no explicit range needs to be set.
#
# Detector parameters tuned for this session:
#
# - `position_std=sqrt(12.5)` ≈ 3.54 cm — KDE bandwidth for the
#   per-cell place fields, matched to the bin size so the field
#   smoothing is roughly one bin wide.
# - `non_local_position_penalty=0.0` and
#   `non_local_penalty_std=1.0` — no spatial prior on non-local
#   state transitions (the detector's non-local random walk is
#   effectively uniform over the track).
# - `local_position_std=0.0` — point-mass local state pinned
#   exactly at the animal's current bin (no Gaussian gating
#   around it). Tune up if the local prior is too restrictive
#   for your data.

# %%
env = Environment(
    place_bin_size=3.5,
)
detector = NonLocalSortedSpikesDetector(
    sorted_spikes_algorithm="sorted_spikes_kde",
    sorted_spikes_algorithm_params={
        "position_std": np.sqrt(12.5),
        "block_size": 4096,
    },
    non_local_position_penalty=0.0,
    non_local_penalty_std=1.0,
    local_position_std=0.0,
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
#
# The launch mode depends on how this file is executed. In a notebook,
# `block=False` lets the cell return while the Jupyter Qt integration
# keeps processing GUI events. As a plain script (`uv run python ...`),
# `block=True` is required so Qt has an event loop; the process will
# intentionally stay alive until the viewer window is closed.
# Set `NLD_JAQ_2D_SHOW_VIEWER=0` to run the fit/predict smoke path
# without opening Qt.

# %%
launch_blocks = not running_in_notebook()
show_viewer = os.environ.get("NLD_JAQ_2D_SHOW_VIEWER", "1") != "0"
print(
    "Qt viewer launch mode: "
    f"{'blocking script mode' if launch_blocks else 'non-blocking notebook mode'}.",
    flush=True,
)
if show_viewer:
    launch_qt(
        detector=detector,
        results=results,
        spike_times=spike_times,
        position=position_2d,
        position_time=time_seconds,
        t_width=0.5,
        show_projected_1d=True,
        projection_track_graph=projection_track_graph,
        projection_edge_order=projection_edge_order,
        projection_edge_spacing=projection_edge_spacing,
        block=launch_blocks,
    )
else:
    print("Skipping Qt viewer launch because NLD_JAQ_2D_SHOW_VIEWER=0.", flush=True)
