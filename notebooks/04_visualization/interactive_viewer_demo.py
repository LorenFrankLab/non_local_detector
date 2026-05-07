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
# # Interactive decoder viewer — demo
#
# Walks through `non_local_detector.visualization.launch`, the Qt-based
# interactive viewer for decoder results. Uses the in-package simulated
# session so the notebook is self-contained.
#
# 1. Fit a detector on the simulated session.
# 2. Build a `RunBundle` and launch the viewer.
# 3. Get the full panel set with `predict(return_outputs=...)`.
# 4. Compare multiple detectors via a `dict[str, RunBundle]`.
# 5. Three sized ways to add user metrics.
# 6. Attach event overlays.
#
# Requires the `[viewer]` extra: `uv pip install -e '.[viewer]'`.

# %% [markdown]
# ## 1. Simulate a session and fit a detector
#
# `make_simulated_data` produces a 1D Track 0 session with `n_neurons`
# sorted units and ground-truth replay event times. Fit
# `NonLocalSortedSpikesDetector` against the non-event portion.

# %%
import numpy as np

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

(
    speed,
    position,
    spike_times,
    time,
    event_times,
    sampling_frequency,
    is_event,
    place_fields,
) = make_simulated_data(n_neurons=25, seed=0)

detector = NonLocalSortedSpikesDetector(
    sorted_spikes_algorithm="sorted_spikes_kde",
    non_local_position_penalty=1.0,
    non_local_penalty_std=5.0,
    local_position_std=1.0,
)
detector.estimate_parameters(
    position_time=time,
    position=position,
    spike_times=spike_times,
    is_training=~is_event,
    time=time,
    return_outputs=None,
)
detector

# %% [markdown]
# ## 2. Build a `RunBundle` and launch the viewer
#
# `RunBundle` wraps the fitted detector, the predict-time results
# dataset, and the session arrays the viewer needs to render rasters
# and behaviour. Calling `launch(bundle)` opens the Qt window.
#
# `block=False` returns immediately so the notebook stays interactive;
# the window stays alive on its own. Pass `block=True` (default) when
# running outside Jupyter to enter `QApplication.exec()`.

# %%
from non_local_detector.visualization import launch
from non_local_detector.visualization.interactive import RunBundle

results_default = detector.predict(
    spike_times=spike_times,
    time=time,
    position=position,
    position_time=time,
)
bundle = RunBundle(
    results=results_default,
    detector=detector,
    spike_times=spike_times,
    position_time=time,
    position=position,
    speed=speed,
)

# launch(bundle, block=False)  # uncomment to actually open the window
print("RunBundle built. Output variables:", list(bundle.results.data_vars))

# %% [markdown]
# ## 3. Full panel set via `predict(return_outputs=...)`
#
# The default `predict()` returns only `acausal_posterior` and
# `acausal_state_probabilities` — enough for the posterior heatmap and
# state-probability panels, plus a fallback top curve in the slice
# panel.
#
# For the full v1 panel set, also request:
#
# - `log_likelihood` → likelihood heatmap + slice top curve.
# - `predictive_posterior` → predictive overlay in the slice panel
#   (the smoothed alternative is always available because it's
#   derived from `acausal_posterior`).
#
# `return_outputs="all"` gets all of them.

# %%
results_all = detector.predict(
    spike_times=spike_times,
    time=time,
    position=position,
    position_time=time,
    return_outputs="all",
)
bundle_all = RunBundle(
    results=results_all,
    detector=detector,
    spike_times=spike_times,
    position_time=time,
    position=position,
    speed=speed,
)
list(bundle_all.results.data_vars)

# %% [markdown]
# ## 4. Multi-run model comparison
#
# Pass a `dict[str, RunBundle]` to compare detectors side-by-side.
# The viewer shows a "Model" dropdown in the controls bar; `M` cycles
# to the next run. Swapping preserves `t_center` / `t_width` and the
# active overlay; it rebinds each panel to the new schema (state names,
# reduction strategy, place-field sort).

# %%
from non_local_detector import (
    ContFragSortedSpikesClassifier,
    NoSpikeContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)


def fit_and_bundle(detector_cls):
    det = detector_cls(sorted_spikes_algorithm="sorted_spikes_kde")
    det.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=~is_event,
        time=time,
        return_outputs=None,
    )
    res = det.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        return_outputs="all",
    )
    return RunBundle(
        results=res,
        detector=det,
        spike_times=spike_times,
        position_time=time,
        position=position,
        speed=speed,
    )


multi = {
    "nl": bundle_all,
    "cf": fit_and_bundle(ContFragSortedSpikesClassifier),
    "nsf": fit_and_bundle(NoSpikeContFragSortedSpikesClassifier),
    "dec": fit_and_bundle(SortedSpikesDecoder),
}

# launch(multi, block=False)  # uncomment to compare all four detectors
list(multi)

# %% [markdown]
# ## 5. User metrics — three sized affordances
#
# The viewer accepts user-supplied panels in three sizes, smallest to
# largest. Pick the one that matches how much control you need.

# %% [markdown]
# ### 5a. `bundle.extra_metrics` — declarative `MetricSpec`
#
# Smallest. Attach a `MetricSpec` to the bundle and the viewer
# auto-builds a generic series panel for it (line / scatter /
# intervals). No panel code required.

# %%
from non_local_detector.visualization.interactive import MetricSpec

ripple_power = np.abs(np.sin(2 * np.pi * 0.05 * time)) + 0.1 * np.random.default_rng(
    0
).normal(size=time.size)

bundle_all.extra_metrics["ripple_power"] = MetricSpec.line(
    name="Ripple power (sim)",
    t=time,
    y=ripple_power,
    color="#9467bd",
    fill_below=True,
    thresholds=(1.0,),
)

# launch(bundle_all, block=False)
list(bundle_all.extra_metrics)

# %% [markdown]
# ### 5b. Pre-built panel classes — pass via `extra_panels`
#
# Medium. Instantiate one of the generic panels yourself
# (`LineSeriesPanel`, `ScatterSeriesPanel`, `IntervalSeriesPanel`,
# `MultiLineSeriesPanel`) and pass it via `extra_panels`. Use this
# when you want full control over construction (custom colour, panel
# ordering, sharing data with other panels) without writing a new
# panel class.

# %%
# This block requires the [viewer] extra at import time.
# from non_local_detector.visualization.interactive.panels.qt.series import (
#     LineSeriesPanel,
# )
# from non_local_detector.visualization.interactive.view_models.series import (
#     LineSeriesModel,
# )
#
# my_line = LineSeriesPanel(
#     LineSeriesModel(
#         name="My metric",
#         t=time,
#         y=ripple_power,
#         color="#2ca02c",
#         fill_below=False,
#     )
# )
# launch(bundle_all, extra_panels=[my_line], block=False)

# %% [markdown]
# ### 5c. Custom `TimeAxisPanel` subclass — largest
#
# Largest. Implement the `TimeAxisPanel` Protocol (`update_window`,
# `set_event_overlays`, `x_link_target`, optional `rebind_after_swap`).
# Use this when you need rendering the generic panels can't do —
# custom heatmaps, video overlays, multi-trace mosaics.
#
# For per-bin readouts (right-column slice-style plots), implement
# `BinSyncedPanel` instead and pass via `extra_bin_panels`. See the
# in-repo plugin contract docs in
# `src/non_local_detector/visualization/interactive/README.md`.

# %% [markdown]
# ## 6. Event overlays
#
# Attach `EventOverlay` instances to `bundle.event_overlays`. Each
# overlay shows up in the overlay selector dropdown; the selector lets
# the user toggle individual overlays without reopening the viewer.
#
# Two overlay shapes:
#
# - `EventOverlay.points(times=...)` — vertical lines (e.g. SWR peaks).
# - `EventOverlay.intervals(t_start=..., t_end=...)` — shaded bands
#   (e.g. immobility periods, candidate replay windows).

# %%
from non_local_detector.visualization.interactive import EventOverlay

bundle_all.event_overlays.append(
    EventOverlay.points(
        name="Simulated SWR peaks",
        times=event_times,
        color="#d62728",
    )
)
bundle_all.event_overlays.append(
    EventOverlay.intervals(
        name="Immobility",
        t_start=np.asarray([time[200]]),
        t_end=np.asarray([time[400]]),
        color="#7f7f7f",
        alpha=0.2,
    )
)

# launch(bundle_all, block=False)
[(o.name, o.kind) for o in bundle_all.event_overlays]

# %% [markdown]
# ## Notes
#
# - Keyboard: `Space` play/pause, `,` / `.` change speed, `[` / `]`
#   change window width, `Shift+Left`/`Shift+Right` step a window,
#   `M` cycle model, `N`/`Shift+N` cycle event overlay, `R` reset.
# - Slice panel overlay can be set programmatically:
#   `viewer.slice_panel.set_overlay_mode("smoothed")` — or via the
#   dropdown in the slice panel header.
# - The viewer is non-blocking when `block=False` so notebook cells
#   keep running. Close the window with the OS shortcut to release
#   Qt resources.
