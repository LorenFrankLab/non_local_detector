"""Generic parallel video writer for matplotlib animations.

This module provides utilities for creating high-quality videos from matplotlib
figures by rendering frames in parallel across multiple processes, then stitching
them together with ffmpeg. This approach is significantly faster than single-process
matplotlib animation for long videos, and more robust to memory leaks.

The core pattern:
1. Partition frames across worker processes
2. Each worker independently renders its assigned frames to PNG files
3. ffmpeg stitches the PNGs into the final video
4. Temporary files are cleaned up

Examples
--------
Create a simple sine wave animation:

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from parallel_video_writer import create_parallel_video, VideoConfig
>>>
>>> # Prepare data (must be pickle-able for multiprocessing)
>>> n_frames = 100
>>> x = np.linspace(0, 4 * np.pi, 200)
>>> y = np.sin(x)
>>> frame_data = {'x': x, 'y': y, 'n_points': len(x)}
>>>
>>> # Define figure setup (called once per worker)
>>> def setup_figure():
...     fig, ax = plt.subplots(figsize=(10, 6))
...     ax.set_xlim(0, 4 * np.pi)
...     ax.set_ylim(-1.5, 1.5)
...     ax.set_xlabel('x')
...     ax.set_ylabel('sin(x)')
...     ax.grid(True, alpha=0.3)
...     return fig, {'main': ax}
>>>
>>> # Define frame rendering (called for each frame)
>>> def render_frame(fig, axes, frame_idx, data):
...     ax = axes['main']
...     ax.clear()
...     n_show = int((frame_idx / 100) * data['n_points'])
...     ax.plot(data['x'][:n_show], data['y'][:n_show], 'b-', linewidth=2)
...     ax.set_xlim(0, 4 * np.pi)
...     ax.set_ylim(-1.5, 1.5)
...     ax.set_xlabel('x')
...     ax.set_ylabel('sin(x)')
...     ax.grid(True, alpha=0.3)
...     ax.set_title(f'Frame {frame_idx + 1}/100')
>>>
>>> # Create video with 4 parallel workers
>>> config = VideoConfig(fps=30.0, dpi=100, max_workers=4)
>>> create_parallel_video(
...     n_frames=100,
...     output_path='sine_wave.mp4',
...     render_frame_func=render_frame,
...     setup_figure_func=setup_figure,
...     frame_data=frame_data,
...     config=config,
... )

Notes
-----
- Requires ffmpeg to be installed and available in PATH
- All data passed to workers must be pickle-able (numpy arrays, dicts, dataclasses, etc.)
- Each worker creates its own matplotlib figure to avoid threading issues
- Temporary PNG files are automatically cleaned up after video creation
"""

from __future__ import annotations

import math
import os
import pickle
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# Headless, deterministic rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
MIN_DPI = 72  # Matplotlib requires minimum 72 DPI for readable output


@dataclass(frozen=True)
class VideoConfig:
    """Configuration for parallel video rendering.

    Parameters
    ----------
    fps : float, default=30.0
        Video frame rate (frames per second).
    dpi : int, default=100
        Resolution for rendered frames. Higher values increase quality and file size.
    max_workers : int | None, default=None
        Number of parallel worker processes. If None, uses half of available CPUs.
    bitrate_kbps : int, default=4000
        Video bitrate in kilobits per second. Higher values increase quality and file size.
    ffmpeg_threads : int | None, default=None
        Number of threads for ffmpeg encoding. If None, uses max_workers value.
    overwrite : bool, default=True
        If True, overwrite existing output file. If False, raise error if file exists.
    verbose_ffmpeg : bool, default=False
        If True, show ffmpeg output for debugging. If False, capture and hide it.

    Examples
    --------
    >>> config = VideoConfig(fps=60.0, dpi=150, max_workers=8, bitrate_kbps=8000)
    >>> config.fps
    60.0
    >>> config_safe = VideoConfig(overwrite=False, verbose_ffmpeg=True)
    >>> config_safe.overwrite
    False
    """

    fps: float = 30.0
    dpi: int = 100
    max_workers: int | None = None
    bitrate_kbps: int = 4000
    ffmpeg_threads: int | None = None
    overwrite: bool = True
    verbose_ffmpeg: bool = False


def _validate_inputs(
    n_frames: int,
    output_path: str,
    frame_data: Any,
    render_frame_func: Callable,
    setup_figure_func: Callable,
    config: VideoConfig,
) -> None:
    """Validate user inputs and provide clear error messages.

    Parameters
    ----------
    n_frames : int
        Number of frames to render.
    output_path : str
        Path to output video file.
    frame_data : Any
        Data to be passed to workers.
    render_frame_func : callable
        Function to update figure for each frame.
    setup_figure_func : callable
        Function to create matplotlib figure.
    config : VideoConfig
        Video configuration.

    Raises
    ------
    ValueError
        If parameters are invalid.
    TypeError
        If frame_data is not pickle-able or setup_figure_func returns wrong type.
    FileNotFoundError
        If output directory doesn't exist.
    """
    # Validate n_frames
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    # Validate config parameters
    if config.fps <= 0:
        raise ValueError(f"fps must be positive, got {config.fps}")

    if config.dpi < MIN_DPI:
        raise ValueError(
            f"dpi must be at least {MIN_DPI} (matplotlib minimum), got {config.dpi}"
        )

    if config.bitrate_kbps <= 0:
        raise ValueError(f"bitrate_kbps must be positive, got {config.bitrate_kbps}")

    if config.max_workers is not None and config.max_workers < 1:
        raise ValueError(
            f"max_workers must be at least 1 or None, got {config.max_workers}"
        )

    # Validate output directory exists
    output_dir = Path(output_path).parent
    if output_dir != Path(".") and not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}\n"
            f"Please create it first: mkdir -p {output_dir}"
        )

    # Check if output file exists when overwrite=False
    if not config.overwrite and Path(output_path).exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Set overwrite=True in VideoConfig to overwrite, or choose a different path."
        )

    # Test pickle-ability of frame_data
    if frame_data is not None:
        try:
            pickle.dumps(frame_data)
        except Exception as e:
            raise TypeError(
                f"frame_data must be pickle-able for multiprocessing, but got error: {e}\n\n"
                f"Common causes:\n"
                f"  - Lambda functions (use def instead)\n"
                f"  - Local/nested functions (define at module level)\n"
                f"  - Open file handles (pass paths, not file objects)\n"
                f"  - Custom classes without __reduce__ method\n\n"
                f"Valid types: dict, list, numpy arrays, dataclasses, primitives"
            ) from e

    # Test that functions are pickleable (required for ProcessPoolExecutor)
    for func, func_name in [
        (render_frame_func, "render_frame_func"),
        (setup_figure_func, "setup_figure_func"),
    ]:
        try:
            pickle.dumps(func)
        except Exception as e:
            raise TypeError(
                f"{func_name} must be pickleable for multiprocessing, but got error: {e}\n\n"
                f"Common causes:\n"
                f"  - Lambda functions: lambda fig, ax, idx, data: ... (not pickleable)\n"
                f"  - Nested/local functions: defined inside another function (not pickleable)\n"
                f"  - Closures: functions that capture variables from outer scope\n\n"
                f"Solution: Define {func_name} at module level (top of file):\n"
                f"  def {func_name}(...):\n"
                f"      ...\n\n"
                f"See examples in parallel_video_writer_examples.py"
            ) from e

    # Test that setup_figure_func returns correct type
    try:
        result = setup_figure_func()
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"setup_figure_func must return (Figure, dict), got {type(result)}\n"
                f"Expected: return fig, {{'name': ax}}"
            )
        fig, axes = result
        if not isinstance(axes, dict):
            raise TypeError(
                f"setup_figure_func must return axes as dict, got {type(axes)}\n"
                f"Expected: return fig, {{'main': ax}}"
            )
        # Clean up test figure
        plt.close(fig)
    except Exception as e:
        if isinstance(e, TypeError) and "setup_figure_func must return" in str(e):
            raise  # Re-raise our own TypeError
        raise RuntimeError(
            f"setup_figure_func() failed during validation: {e}\n"
            f"This function will be called once per worker to create the figure."
        ) from e


def create_parallel_video(
    *,
    n_frames: int,
    output_path: str,
    render_frame_func: Callable[[plt.Figure, dict[str, Any], int, Any], None],
    setup_figure_func: Callable[[], tuple[plt.Figure, dict[str, Any]]],
    frame_data: Any = None,
    config: VideoConfig = VideoConfig(),
) -> str:
    """Create video by rendering matplotlib frames in parallel.

    This function parallelizes video creation by distributing frame rendering
    across multiple worker processes. Each worker independently renders its
    assigned frames to PNG files, which are then stitched together using ffmpeg.

    Parameters
    ----------
    n_frames : int
        Total number of frames to render.
    output_path : str
        Path to output video file (e.g., 'output.mp4').
    render_frame_func : callable
        Function with signature (fig, axes, frame_idx, frame_data) -> None
        that updates the matplotlib figure for a specific frame index.
        Should modify axes in-place. Does not need to call fig.savefig().
    setup_figure_func : callable
        Function with signature () -> (fig, axes_dict) that creates and
        configures the matplotlib figure. Called once per worker process.
        Must return (Figure, dict[str, Any]) where dict can contain Axes and
        other state (e.g., image handles, colorbars) for efficient updates.
    frame_data : Any, optional
        Data structure passed to render_frame_func. Must be pickle-able for
        multiprocessing. Can be dict, numpy arrays, dataclasses, etc.
    config : VideoConfig, optional
        Video rendering configuration.

    Returns
    -------
    str
        Path to the created video file (same as output_path argument).

    Raises
    ------
    ValueError
        If n_frames <= 0 or configuration values are invalid.
    TypeError
        If frame_data is not pickle-able or functions are not pickle-able
        or setup_figure_func returns wrong type.
    FileNotFoundError
        If output directory doesn't exist.
    FileExistsError
        If output file exists and overwrite=False.
    RuntimeError
        If ffmpeg is not found in PATH or encoding fails.

    Examples
    --------
    Create a video of a rotating sine wave:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> def setup_figure():
    ...     fig = plt.figure(figsize=(8, 6))
    ...     ax = fig.add_subplot(111, projection='polar')
    ...     return fig, {'polar': ax}
    >>>
    >>> def render_frame(fig, axes, frame_idx, data):
    ...     ax = axes['polar']
    ...     ax.clear()
    ...     theta = data['theta'] + (frame_idx / 100) * 2 * np.pi
    ...     ax.plot(theta, data['r'], 'b-', linewidth=2)
    ...     ax.set_ylim(0, 1.5)
    ...     ax.set_title(f'Frame {frame_idx}', pad=20)
    >>>
    >>> theta = np.linspace(0, 2 * np.pi, 100)
    >>> r = np.abs(np.sin(3 * theta))
    >>> data = {'theta': theta, 'r': r}
    >>>
    >>> create_parallel_video(
    ...     n_frames=100,
    ...     output_path='rotating_sine.mp4',
    ...     render_frame_func=render_frame,
    ...     setup_figure_func=setup_figure,
    ...     frame_data=data,
    ...     config=VideoConfig(fps=30, dpi=100, max_workers=4),
    ... )

    Notes
    -----
    - Each worker process creates its own figure via setup_figure_func to avoid
      matplotlib threading issues
    - The render_frame_func should be stateless and only depend on frame_idx and
      frame_data
    - Temporary PNG files are written to a temp directory and cleaned up automatically
    - Progress is printed as chunks complete
    """
    # Validate inputs early to provide clear error messages
    _validate_inputs(
        n_frames, output_path, frame_data, render_frame_func, setup_figure_func, config
    )
    _require_ffmpeg()

    max_workers = config.max_workers
    if max_workers is None:
        max_workers = max(2, (os.cpu_count() or 4) // 2)

    print(f"[parallel] Using {max_workers} workers for {n_frames} frames")

    # Partition frames evenly across workers
    chunk_size = math.ceil(n_frames / max_workers)
    chunks = [
        (s, min(n_frames, s + chunk_size)) for s in range(0, n_frames, chunk_size)
    ]

    # Use TemporaryDirectory context manager for automatic cleanup (even on SIGINT)
    with tempfile.TemporaryDirectory(prefix="parallel_video_frames_") as tmpdir:
        pattern = os.path.join(tmpdir, "frame_%06d.png")
        print(f"[parallel] Writing frames to {tmpdir}")

        # Render chunks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _render_chunk,
                    start_frame=start,
                    end_frame=end,
                    pattern=pattern,
                    dpi=config.dpi,
                    render_frame_func=render_frame_func,
                    setup_figure_func=setup_figure_func,
                    frame_data=frame_data,
                )
                for (start, end) in chunks
            ]

            # Wait for all chunks to complete
            for i, fut in enumerate(futures):
                fut.result()
                print(f"[parallel] Chunk {i + 1}/{len(futures)} complete")

        # Stitch frames into video
        _ffmpeg_stitch(
            pattern=pattern,
            output_path=output_path,
            fps=config.fps,
            bitrate=config.bitrate_kbps,
            threads=config.ffmpeg_threads or max_workers,
            config=config,
        )
        # TemporaryDirectory context manager automatically cleans up tmpdir here

    print(f"[parallel] Video saved to: {output_path}")
    return output_path


def _render_chunk(
    start_frame: int,
    end_frame: int,
    pattern: str,
    dpi: int,
    render_frame_func: Callable[[plt.Figure, dict[str, Any], int, Any], None],
    setup_figure_func: Callable[[], tuple[plt.Figure, dict[str, Any]]],
    frame_data: Any,
) -> None:
    """Render a chunk of frames in a worker process.

    This function runs in a separate process and renders frames from start_frame
    to end_frame (exclusive).

    Parameters
    ----------
    start_frame : int
        First frame index to render (inclusive).
    end_frame : int
        Last frame index to render (exclusive).
    pattern : str
        Printf-style pattern for output PNG files (e.g., '/tmp/frame_%06d.png').
    dpi : int
        Resolution for saved figures.
    render_frame_func : callable
        User-provided function to update figure for each frame.
        Signature: (fig, axes, frame_idx, frame_data) -> None
    setup_figure_func : callable
        User-provided function to create matplotlib figure.
        Signature: () -> (Figure, dict[str, Any])
    frame_data : Any
        User data passed to render_frame_func.

    Raises
    ------
    RuntimeError
        If frame rendering fails for any frame in this chunk.

    Notes
    -----
    Frame indices are 0-based in Python but saved with 1-based numbering
    (frame_idx + 1) because ffmpeg's -i pattern expects consecutive files
    starting from 1.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        # Each worker creates its own figure
        fig, axes = setup_figure_func()

        # Render each frame in this chunk
        for frame_idx in range(start_frame, end_frame):
            try:
                # Update figure for this frame
                render_frame_func(fig, axes, frame_idx, frame_data)

                # Save frame (1-based numbering for ffmpeg, MIN_DPI minimum)
                out_path = pattern % (frame_idx + 1)
                fig.savefig(out_path, dpi=max(MIN_DPI, dpi))

            except Exception as e:
                raise RuntimeError(
                    f"Failed rendering frame {frame_idx} (range {start_frame}-{end_frame - 1}):\n"
                    f"Error: {e}\n"
                    f"Check your render_frame_func for errors."
                ) from e

        plt.close(fig)

    except Exception as e:
        if isinstance(e, RuntimeError) and "Failed rendering frame" in str(e):
            raise  # Re-raise frame-specific errors
        raise RuntimeError(
            f"Failed in chunk {start_frame}-{end_frame - 1} during setup:\n"
            f"Error: {e}\n"
            f"Check your setup_figure_func for errors."
        ) from e


def _ffmpeg_stitch(
    pattern: str,
    output_path: str,
    fps: float,
    bitrate: int = 4000,
    threads: int = 4,
    config: VideoConfig | None = None,
) -> None:
    """Stitch PNG frames into video using ffmpeg.

    Parameters
    ----------
    pattern : str
        Printf-style pattern for input PNG files (e.g., '/tmp/frame_%06d.png').
    output_path : str
        Path to output video file.
    fps : float
        Video frame rate.
    bitrate : int, default=4000
        Video bitrate in kilobits per second.
    threads : int, default=4
        Number of threads for ffmpeg encoding.
    config : VideoConfig | None, optional
        Video configuration for overwrite and verbosity settings.

    Raises
    ------
    RuntimeError
        If ffmpeg encoding fails with detailed error message.
    """
    # Get flags from config
    overwrite = config.overwrite if config else True
    verbose = config.verbose_ffmpeg if config else False

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",  # Overwrite or fail if exists
        "-framerate",
        str(fps),  # Pass fps as-is (supports fractional rates like 29.97)
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),  # Pass fps as-is for output
        "-b:v",
        str(bitrate * 1000),  # Convert kbps to bps for ffmpeg
        "-threads",
        str(max(1, threads)),
        output_path,
    ]
    print(f"[ffmpeg] {' '.join(cmd)}")

    # Control ffmpeg output visibility
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.PIPE

    try:
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr, text=True)
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"ffmpeg encoding failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"Stderr: {e.stderr if e.stderr else '(not captured)'}"
        )
        raise RuntimeError(error_msg) from e


def _require_ffmpeg() -> None:
    """Check that ffmpeg is available in PATH.

    Raises
    ------
    RuntimeError
        If ffmpeg is not found.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Please install ffmpeg "
            "(e.g., 'brew install ffmpeg' or 'apt-get install ffmpeg')."
        )
