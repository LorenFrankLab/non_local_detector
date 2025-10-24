import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from non_local_detector.visualization._parallel_video_writer import (
    VideoConfig,
    create_parallel_video,
)
from non_local_detector.visualization.static import get_multiunit_firing_rate


def make_single_environment_movie(
    time_slice,
    classifier,
    results,
    position_info,
    spike_times,
    movie_name="video_name.mp4",
    sampling_frequency=500,
    video_slowdown=8,
    position_name=None,
    direction_name="head_orientation",
    vmax=0.07,
):
    """Generate a movie of the decoding results for a single environment."""

    position_name = (
        ["head_position_x", "head_position_y"]
        if position_name is None
        else position_name
    )

    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_firing_rate(
            spike_times, position_info.index.to_numpy().squeeze()
        ),
        index=position_info.index,
        columns=["firing_rate"],
    )

    # Prepare data
    is_track_interior = classifier.environments[0].is_track_interior_
    posterior = (
        results.acausal_posterior.isel(time=time_slice)
        .unstack("state_bins")
        .sum("state")
        .where(is_track_interior)
    )
    map_position_ind = posterior.argmax(["x_position", "y_position"])
    map_position = np.stack(
        (
            posterior.x_position[map_position_ind["x_position"]].values,
            posterior.y_position[map_position_ind["y_position"]].values,
        ),
        axis=1,
    )

    position = np.asarray(position_info.iloc[time_slice][position_name])
    direction = np.asarray(position_info.iloc[time_slice][direction_name])

    window_size = 501
    window_ind = np.arange(window_size) - window_size // 2
    rate = multiunit_firing_rate.iloc[
        slice(time_slice.start + window_ind[0], time_slice.stop + window_ind[-1])
    ]
    x_min = float(position_info[position_name[0]].min()) - 10
    x_max = float(position_info[position_name[0]].max()) + 10
    y_min = float(position_info[position_name[1]].min()) - 10
    y_max = float(position_info[position_name[1]].max()) + 10

    # Pack everything into frame_data (must be pickle-able: dicts, numpy arrays, etc.)
    frame_data = {
        "posterior": posterior,
        "position": position,
        "direction": direction,
        "map_position": map_position,
        "window_ind": window_ind,
        "rate": np.asarray(rate).squeeze(),  # Convert to numpy for pickle
        "xy_limits": (x_min, x_max, y_min, y_max),
        "sampling_frequency": sampling_frequency,
        "window_size": window_size,
        "position_info": position_info,
        "position_name": position_name,
        "vmax": vmax,
        "r": 4,  # Direction arrow length
    }

    # Configure video
    fps = sampling_frequency / video_slowdown
    config = VideoConfig(
        fps=fps,
        dpi=200,
        max_workers=4,
        bitrate_kbps=8000,  # High quality for scientific visualization
        overwrite=True,
    )

    # Create video
    n_frames = posterior.shape[0]
    return create_parallel_video(
        n_frames=n_frames,
        output_path=movie_name,
        render_frame_func=_render_single_env_frame,
        setup_figure_func=_setup_single_env_figure,
        frame_data=frame_data,
        config=config,
    )


# Define these at MODULE LEVEL (required for pickle-ability)
def _setup_single_env_figure():
    """Setup figure for single environment movie."""
    with plt.style.context("dark_background"):
        fig, axes_array = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            gridspec_kw={"height_ratios": [5, 1]},
            constrained_layout=False,
        )

        # Store all artists and axes in a dict
        axes = {"fig": fig, "ax0": axes_array[0], "ax1": axes_array[1]}

        # Setup top axis
        axes["ax0"].tick_params(colors="white", which="both")
        axes["ax0"].spines["bottom"].set_color("white")
        axes["ax0"].spines["left"].set_color("white")
        axes["ax0"].spines["top"].set_color("black")
        axes["ax0"].spines["right"].set_color("black")
        axes["ax0"].axis("off")

        # Setup bottom axis
        axes["ax1"].set_xlabel("Time [s]")
        axes["ax1"].set_ylabel("Multiunit\n[spikes/s]")
        axes["ax1"].set_facecolor("black")
        axes["ax1"].spines["top"].set_color("black")
        axes["ax1"].spines["right"].set_color("black")

        # Create artists (will be updated each frame)
        axes["position_dot"] = axes["ax0"].scatter(
            [], [], s=80, zorder=102, color="magenta", label="actual position"
        )
        (axes["position_line"],) = axes["ax0"].plot(
            [], [], color="magenta", linewidth=5
        )
        axes["map_dot"] = axes["ax0"].scatter(
            [], [], s=80, zorder=102, color="green", label="decoded position"
        )
        (axes["map_line"],) = axes["ax0"].plot([], [], "green", linewidth=3)
        axes["mesh"] = None  # Will be created on first frame
        axes["title"] = axes["ax0"].set_title("")

        # Add scalebar
        fontprops = fm.FontProperties(size=16)
        scalebar = AnchoredSizeBar(
            axes["ax0"].transData,
            20,
            "20 cm",
            "lower right",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )
        axes["ax0"].add_artist(scalebar)

        (axes["multiunit_line"],) = axes["ax1"].plot(
            [], [], color="white", linewidth=2, clip_on=False
        )

    return fig, axes


def _render_single_env_frame(fig, axes, frame_idx, data):
    """Render a single frame for single environment movie."""

    # Update position
    axes["position_dot"].set_offsets(data["position"][frame_idx])

    r = data["r"]
    axes["position_line"].set_data(
        [
            data["position"][frame_idx, 0],
            data["position"][frame_idx, 0] + r * np.cos(data["direction"][frame_idx]),
        ],
        [
            data["position"][frame_idx, 1],
            data["position"][frame_idx, 1] + r * np.sin(data["direction"][frame_idx]),
        ],
    )

    # Update decoded position
    start_ind = max(0, frame_idx - 5)
    time_slice = slice(start_ind, frame_idx)

    axes["map_dot"].set_offsets(data["map_position"][frame_idx])
    axes["map_line"].set_data(
        data["map_position"][time_slice, 0], data["map_position"][time_slice, 1]
    )

    # Update posterior mesh
    if axes["mesh"] is None:
        # First frame: create mesh and set axis limits
        axes["mesh"] = (
            data["posterior"]
            .isel(time=0)
            .plot(
                x="x_position",
                y="y_position",
                vmin=0.0,
                vmax=data["vmax"],
                ax=axes["ax0"],
                add_colorbar=False,
            )
        )
        axes["ax0"].set_xlabel("")
        axes["ax0"].set_ylabel("")
        x_min, x_max, y_min, y_max = data["xy_limits"]
        axes["ax0"].set_xlim(x_min, x_max)
        axes["ax0"].set_ylim(y_min, y_max)
    else:
        # Update existing mesh
        axes["mesh"].set_array(
            data["posterior"].isel(time=frame_idx).values.ravel(order="F")
        )

    # Update title
    axes["title"].set_text(
        f"time = {data['posterior'].isel(time=frame_idx).time.values:0.2f}"
    )

    # Update multiunit firing rate
    try:
        window_ind = data["window_ind"]
        window_size = data["window_size"]
        sampling_frequency = data["sampling_frequency"]

        axes["multiunit_line"].set_data(
            window_ind / sampling_frequency,
            data["rate"][frame_idx + (window_size // 2) + window_ind],
        )

        # Set y-limit on first frame
        if frame_idx == 0:
            axes["ax1"].set_ylim((0.0, data["rate"].max()))
            axes["ax1"].set_xlim(
                (
                    window_ind[0] / sampling_frequency,
                    window_ind[-1] / sampling_frequency,
                )
            )
    except IndexError:
        pass
