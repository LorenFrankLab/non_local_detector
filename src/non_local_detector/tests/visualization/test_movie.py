"""Smoke tests for ``visualization.movie`` and the parallel video writer.

These render a handful of frames to a temporary mp4 and assert the file
exists and is non-empty. They require ``ffmpeg`` on ``PATH`` and are marked
slow because they spawn worker processes and invoke the encoder.
"""

import shutil

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from non_local_detector.visualization._parallel_video_writer import (  # noqa: E402
    VideoConfig,
    create_parallel_video,
)
from non_local_detector.visualization.movie import (  # noqa: E402
    make_single_environment_movie,
)

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is not available on PATH",
)


def _setup_line_figure():
    """Module-level setup function (must be picklable for multiprocessing)."""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, {"main": ax}


def _render_line_frame(fig, axes, frame_idx, data):
    """Module-level render function (must be picklable for multiprocessing)."""
    ax = axes["main"]
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(data["x"], data["x"] * (frame_idx + 1) / data["n_frames"])


def test_video_config_defaults():
    """``VideoConfig`` is a frozen dataclass with documented defaults."""
    config = VideoConfig()
    assert config.fps == 30.0
    assert config.overwrite is True

    custom = VideoConfig(fps=10.0, dpi=72, max_workers=2)
    assert custom.fps == 10.0
    assert custom.max_workers == 2


@pytest.mark.slow
@ffmpeg_required
def test_create_parallel_video_writes_nonempty_file(tmp_path):
    """``create_parallel_video`` renders frames and stitches a non-empty mp4."""
    out_path = str(tmp_path / "line.mp4")
    frame_data = {"x": np.linspace(0, 1, 20), "n_frames": 5}

    returned = create_parallel_video(
        n_frames=5,
        output_path=out_path,
        render_frame_func=_render_line_frame,
        setup_figure_func=_setup_line_figure,
        frame_data=frame_data,
        config=VideoConfig(fps=5.0, dpi=72, max_workers=2),
    )

    assert returned == out_path
    assert (tmp_path / "line.mp4").exists()
    assert (tmp_path / "line.mp4").stat().st_size > 0


@pytest.mark.slow
@pytest.mark.integration
@ffmpeg_required
def test_make_single_environment_movie_writes_nonempty_file(
    fitted_2d_decoder, tmp_path
):
    """``make_single_environment_movie`` writes a non-empty mp4 to ``tmp_path``."""
    out_path = str(tmp_path / "decode.mp4")

    returned = make_single_environment_movie(
        time_slice=slice(0, 5),
        classifier=fitted_2d_decoder["classifier"],
        results=fitted_2d_decoder["results"],
        position_info=fitted_2d_decoder["position_info"],
        spike_times=fitted_2d_decoder["spike_times"],
        movie_name=out_path,
        sampling_frequency=fitted_2d_decoder["sampling_frequency"],
        video_slowdown=1,
    )

    assert returned == out_path
    assert (tmp_path / "decode.mp4").exists()
    assert (tmp_path / "decode.mp4").stat().st_size > 0
