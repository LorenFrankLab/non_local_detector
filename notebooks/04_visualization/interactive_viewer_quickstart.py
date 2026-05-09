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
# # Interactive decoder viewer — quickstart
#
# The shortest path from `predict` to the interactive viewer. No
# `RunBundle` import needed — pass the detector and the four arrays
# you already have to `launch_qt(...)`.
#
# Requires the `[viewer]` extra: `uv pip install -e '.[viewer]'`.

# %%
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.visualization.interactive import launch_qt

# %% [markdown]
# ## 1. Fit a detector
#
# `make_simulated_data` produces a 1D Track 0 session. Fit
# `NonLocalSortedSpikesDetector` against the non-event portion.

# %%
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
# ``estimate_parameters`` is the EM-fit entry point; ``is_training``
# is the boolean mask of training timepoints (we exclude the
# simulated replay events from the fit).
detector.estimate_parameters(
    position_time=time,
    position=position,
    spike_times=spike_times,
    is_training=~is_event,
    time=time,
    return_outputs=None,
)

# %% [markdown]
# ## 2. Run `predict` to get the full panel set
#
# `return_outputs="all"` ships every optional output the viewer's
# panels can display (likelihood, predictive overlay, smoothed
# posterior, state probabilities). Smaller `return_outputs` values
# work too — the viewer disables panels for missing outputs and
# explains the rebuild command in the disabled-state UI.

# %%
results = detector.predict(
    spike_times=spike_times,
    time=time,
    position=position,
    position_time=time,
    return_outputs="all",
)

# %% [markdown]
# ## 3. Open the viewer
#
# Pass `detector` + `results` + the four arrays. The viewer builds
# the bundle internally — no `RunBundle` import needed.
#
# `block=True` (default) opens the window and blocks until the user
# closes it; the cell returns the exit code. Set `block=False` to
# launch the viewer in the background while the notebook continues.

# %%
launch_qt(
    detector=detector,
    results=results,
    spike_times=spike_times,
    position=position,
    position_time=time,
    speed=speed,
)

# %% [markdown]
# ## 4. Multi-run comparison (advanced)
#
# When you want to swap between fitted detectors with the M key,
# build `RunBundle` instances directly and pass a `{name: bundle}`
# dict as the first positional argument. See the
# `interactive_viewer_demo` notebook for the full advanced API
# (overlays, extra_metrics, bundle-form construction).
