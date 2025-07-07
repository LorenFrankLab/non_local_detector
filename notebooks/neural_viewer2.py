"""
Neural Decoding Viewer - Fixed Clickable Time Series

Key fixes for clicking functionality:
1. Proper tap event handling with correct coordinate access
2. Invisible scatter renderer sized appropriately for click detection
3. Better event coordinate mapping
4. Debugging output for click events
"""

import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Range1d,
    Select,
    Slider,
    TapTool,
)
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import curdoc, figure

# Optional imports for big data
try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@dataclass
class TimeSeriesConfig:
    """Configuration for a time series plot"""

    name: str
    data: Union[np.ndarray, xr.DataArray, pd.Series]
    time_coords: np.ndarray
    label: str
    color: str
    y_label: str
    line_width: float = 2.5
    alpha: float = 0.9


class DataSimulator:
    """Standalone data simulator - separated from viewer"""

    def __init__(self, n_time: int = 2000, sampling_frequency: float = 250):
        self.n_time = n_time
        self.sampling_frequency = sampling_frequency
        self.times = np.arange(n_time) / sampling_frequency

    def generate_trajectory(self) -> pd.DataFrame:
        """Generate animal trajectory data"""
        # Complex trajectory with multiple behaviors
        x_pos = 50 + 25 * np.sin(self.times * 0.5) + 3 * np.random.randn(self.n_time)
        y_pos = 50 + 25 * np.cos(self.times * 0.5) + 3 * np.random.randn(self.n_time)
        speed = np.maximum(
            0, 15 + 10 * np.sin(self.times * 2) + 5 * np.random.randn(self.n_time)
        )
        head_direction = np.cumsum(np.random.randn(self.n_time) * 0.1)

        return pd.DataFrame(
            {
                "time": self.times,
                "x": x_pos,
                "y": y_pos,
                "speed": speed,
                "head_direction": head_direction,
            }
        )

    def generate_posterior(
        self, position_df: pd.DataFrame, x_coords: np.ndarray, y_coords: np.ndarray
    ) -> xr.DataArray:
        """Generate posterior probability using vectorized operations"""

        # Vectorized posterior generation - MASSIVE speedup!
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        posterior_data = np.zeros((self.n_time, len(x_coords), len(y_coords)))

        # Sample every 5th frame, then fill
        sample_indices = np.arange(0, self.n_time, 5)

        for i, t in enumerate(sample_indices):
            # Variable uncertainty based on speed
            sigma = 5 + (position_df["speed"].iloc[t] / 30) * 5

            # Add decoding noise
            center_x = position_df["x"].iloc[t] + np.random.randn() * 3
            center_y = position_df["y"].iloc[t] + np.random.randn() * 3

            # Vectorized distance calculation
            dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
            prob_slice = np.exp(-dist_sq / (2 * sigma**2))
            prob_slice = prob_slice / (prob_slice.sum() + 1e-10)

            posterior_data[t] = prob_slice

            # Fill intermediate frames
            next_t = min(t + 5, self.n_time)
            for offset in range(1, min(5, next_t - t)):
                posterior_data[t + offset] = prob_slice

        return xr.DataArray(
            posterior_data,
            coords={"time": self.times, "x_position": x_coords, "y_position": y_coords},
            dims=["time", "x_position", "y_position"],
            name="posterior",
        )

    def generate_neural_data(
        self, position_df: pd.DataFrame
    ) -> Dict[str, xr.DataArray]:
        """Generate various neural time series"""

        # MUA firing rate
        base_rate = 15
        mua_rate = (
            base_rate
            + position_df["speed"] * 0.3
            + np.random.exponential(3, self.n_time)
        )

        # Non-local probability
        speed_effect = 1 / (1 + np.exp(-((position_df["speed"] - 20) / 10)))
        nl_prob = 0.1 + 0.4 * speed_effect + 0.1 * np.random.rand(self.n_time)
        nl_prob = np.clip(nl_prob, 0, 1)

        # Theta rhythm
        theta_freq = 8  # Hz
        theta_phase = 2 * np.pi * theta_freq * self.times + np.random.randn() * 0.1
        theta_power = (
            0.5 + 0.3 * np.sin(theta_phase) + 0.1 * np.random.randn(self.n_time)
        )

        # Gamma power
        gamma_power = np.random.gamma(2, 0.5, self.n_time) + position_df["speed"] * 0.02

        return {
            "multiunit_rate": xr.DataArray(
                mua_rate, coords={"time": self.times}, dims=["time"]
            ),
            "non_local_prob": xr.DataArray(
                nl_prob, coords={"time": self.times}, dims=["time"]
            ),
            "theta_power": xr.DataArray(
                theta_power, coords={"time": self.times}, dims=["time"]
            ),
            "gamma_power": xr.DataArray(
                gamma_power, coords={"time": self.times}, dims=["time"]
            ),
        }

    def create_full_dataset(self) -> tuple:
        """Create complete simulated dataset"""

        # Spatial coordinates
        x_coords = np.linspace(0, 100, 40)
        y_coords = np.linspace(0, 100, 40)

        # Generate all data
        position_df = self.generate_trajectory()
        posterior = self.generate_posterior(position_df, x_coords, y_coords)
        neural_data = self.generate_neural_data(position_df)

        return position_df, posterior, neural_data


class ChunkedDataManager:
    """Manages chunked/large data access efficiently for huge datasets"""

    def __init__(self, data: xr.DataArray, chunk_size: int = None):
        self.data = data
        self.is_chunked = self._detect_chunked()

        # Calculate optimal chunk size for huge datasets
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size()
        self.chunk_size = chunk_size

        self.chunk_cache = OrderedDict()
        # For huge datasets, cache fewer chunks to preserve memory
        self.max_cached_chunks = 3 if self.is_chunked else 5

        # Performance stats
        self.cache_hits = 0
        self.cache_misses = 0

        if self.is_chunked:
            print(
                f"ChunkedDataManager: {len(data.time):,} frames, chunk_size={chunk_size}, "
                f"max_cached_chunks={self.max_cached_chunks}"
            )

    def _detect_chunked(self) -> bool:
        """Detect if data should be treated as chunked"""
        # Check if already Dask-backed
        if hasattr(self.data.data, "chunks"):
            return True

        # For your dataset size: 452325 × 61 × 55 × 4 bytes ≈ 6GB
        data_size_gb = self.data.nbytes / (1024**3)

        # Treat as chunked if >500MB (much more conservative)
        should_chunk = data_size_gb > 0.5

        if should_chunk:
            print(
                f"Large dataset detected: {data_size_gb:.1f}GB - enabling chunked mode"
            )

        return should_chunk

    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate chunk size for huge datasets"""

        # For your 452k frames dataset, we need much smaller chunks
        n_frames = len(self.data.time)
        spatial_size = (
            self.data.shape[1] * self.data.shape[2] if len(self.data.shape) > 2 else 1
        )
        bytes_per_frame = spatial_size * 4  # float32

        # Target chunk size: 50MB for huge datasets (very conservative)
        target_chunk_mb = 50
        target_chunk_bytes = target_chunk_mb * 1024 * 1024

        chunk_size = max(1, target_chunk_bytes // bytes_per_frame)

        # For your dataset: 61×55×4 = 13,420 bytes per frame
        # 50MB / 13,420 ≈ 3,725 frames per chunk
        chunk_size = min(chunk_size, 5000)  # Cap at 5k frames
        chunk_size = min(chunk_size, n_frames // 10)  # At least 10 chunks

        return max(100, chunk_size)  # Minimum 100 frames per chunk

    def get_frame(self, time_idx: int) -> np.ndarray:
        """Get frame data efficiently for huge datasets"""
        if not self.is_chunked:
            return self.data.isel(time=time_idx).values

        return self._get_chunked_frame(time_idx)

    def _get_chunked_frame(self, time_idx: int) -> np.ndarray:
        """Load frame from chunked data with aggressive memory management"""
        chunk_id = time_idx // self.chunk_size

        if chunk_id in self.chunk_cache:
            self.cache_hits += 1
            # Move to end for LRU
            chunk_data = self.chunk_cache[chunk_id]
            self.chunk_cache.move_to_end(chunk_id)
        else:
            self.cache_misses += 1
            # Load chunk - this is the expensive operation
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(self.data.time))

            print(
                f"Loading chunk {chunk_id}: frames {start_idx}-{end_idx} "
                f"(cache: {self.cache_hits}/{self.cache_hits + self.cache_misses} hits)"
            )

            try:
                if hasattr(self.data.data, "chunks"):
                    # Already chunked/Dask
                    chunk_data = self.data.isel(
                        time=slice(start_idx, end_idx)
                    ).compute()
                else:
                    # Force chunking for huge in-memory arrays
                    chunk_data = self.data.isel(time=slice(start_idx, end_idx))

            except MemoryError:
                print("MemoryError: Dataset too large for current chunk size")
                # Try smaller chunk
                self.chunk_size = max(10, self.chunk_size // 2)
                return self._get_chunked_frame(time_idx)

            # Aggressive cache management for huge datasets
            while len(self.chunk_cache) >= self.max_cached_chunks:
                removed_chunk = self.chunk_cache.popitem(last=False)
                print(f"Evicted chunk {removed_chunk[0]} from cache")

            self.chunk_cache[chunk_id] = chunk_data

        # Extract frame from chunk
        frame_offset = time_idx - (chunk_id * self.chunk_size)
        if hasattr(chunk_data, "isel"):
            return chunk_data.isel(time=frame_offset).values
        else:
            return chunk_data[frame_offset]


class TimeSeriesManager:
    """Manages multiple time series with decimation and windowing"""

    def __init__(self, decimation_threshold: int = 10000):
        self.series_configs: List[TimeSeriesConfig] = []
        self.decimation_threshold = decimation_threshold
        self._decimation_cache = {}

    def add_series(self, config: TimeSeriesConfig):
        """Add a time series to be managed"""
        self.series_configs.append(config)

    def get_decimated_data(self, series_name: str, factor: int) -> tuple:
        """Get decimated version of time series"""
        cache_key = f"{series_name}_{factor}"

        if cache_key in self._decimation_cache:
            return self._decimation_cache[cache_key]

        # Find series config
        config = next((c for c in self.series_configs if c.name == series_name), None)
        if config is None:
            return None, None

        # Decimate data
        decimated_time = config.time_coords[::factor]
        decimated_values = self._extract_values(config.data)[::factor]

        self._decimation_cache[cache_key] = (decimated_time, decimated_values)
        return decimated_time, decimated_values

    def get_windowed_data(
        self, series_name: str, window_start: float, window_end: float
    ) -> tuple:
        """Get data within time window"""
        config = next((c for c in self.series_configs if c.name == series_name), None)
        if config is None:
            return None, None

        # Create mask for window
        mask = (config.time_coords >= window_start) & (config.time_coords <= window_end)

        windowed_time = config.time_coords[mask]
        windowed_values = self._extract_values(config.data)[mask]

        return windowed_time, windowed_values

    def _extract_values(
        self, data: Union[np.ndarray, xr.DataArray, pd.Series]
    ) -> np.ndarray:
        """Extract numpy array from various data types"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, xr.DataArray):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        else:
            return np.asarray(data)


class NeuralViewer:
    """Clean, focused neural data viewer optimized for huge datasets"""

    def __init__(
        self,
        position_data: pd.DataFrame,
        posterior_data: xr.DataArray,
        time_series_data: Dict[str, Union[xr.DataArray, pd.Series, np.ndarray]],
        sampling_frequency: float = 250,
    ):

        # Core data
        self.position_data = position_data
        self.times = position_data["time"].values
        self.n_time = len(self.times)
        self.sampling_frequency = sampling_frequency

        # Print dataset info
        data_size_gb = posterior_data.nbytes / (1024**3)
        print(
            f"Dataset: {self.n_time:,} frames, {posterior_data.shape[1]}×{posterior_data.shape[2]} spatial, "
            f"{data_size_gb:.1f}GB"
        )

        # Spatial coordinates
        self.x_coords = posterior_data.x_position.values
        self.y_coords = posterior_data.y_position.values

        # Data managers - optimized for huge datasets
        self.posterior_manager = ChunkedDataManager(posterior_data)
        self.time_series_manager = TimeSeriesManager(
            decimation_threshold=50000
        )  # Lower threshold

        # Setup time series
        self._setup_time_series(time_series_data)

        # UI state
        self.current_time_idx = 0
        self.playing = False
        self.window_duration = 0.8

        # Performance - much smaller cache for huge datasets
        self.frame_cache = OrderedDict()
        if self.n_time > 100000:
            self.max_cache_size = 50  # Very small cache for huge datasets
        else:
            self.max_cache_size = min(500, self.n_time // 20)

        print(f"Frame cache size: {self.max_cache_size}")

        # Setup UI
        self._setup_logger()
        self._precompute_trajectory_data()
        self._create_plots()
        self._create_controls()
        self._setup_callbacks()

        self._log_info(
            f"NeuralViewer initialized: {self.n_time:,} frames, cache={self.max_cache_size}"
        )

        # Recommend optimal loading for huge datasets
        if data_size_gb > 2.0:
            self._log_info("HUGE dataset detected! Recommend using:")
            self._log_info("1. Zarr: xr.open_zarr('data.zarr', chunks={'time': 5000})")
            self._log_info(
                "2. Or netCDF: xr.open_dataset('data.nc', chunks={'time': 5000})"
            )

    def _setup_time_series(self, time_series_data: Dict[str, Any]):
        """Setup time series configurations"""

        colors = Category10[10]
        color_idx = 0

        # Add behavioral data from position
        for col in ["speed"]:
            if col in self.position_data.columns:
                config = TimeSeriesConfig(
                    name=col,
                    data=self.position_data[col],
                    time_coords=self.times,
                    label=col.replace("_", " ").title(),
                    color=colors[color_idx % len(colors)],
                    y_label=self._get_y_label(col),
                )
                self.time_series_manager.add_series(config)
                color_idx += 1

        # Add neural data
        for name, data in time_series_data.items():
            config = TimeSeriesConfig(
                name=name,
                data=data,
                time_coords=self.times,
                label=name.replace("_", " ").title(),
                color=colors[color_idx % len(colors)],
                y_label=self._get_y_label(name),
            )
            self.time_series_manager.add_series(config)
            color_idx += 1

    def _get_y_label(self, series_name: str) -> str:
        """Get appropriate y-axis label for series"""
        labels = {
            "speed": "Speed (cm/s)",
            "multiunit_rate": "Firing Rate (spikes/s)",
            "non_local_prob": "Probability",
            "theta_power": "Theta Power",
            "gamma_power": "Gamma Power",
        }
        return labels.get(series_name, series_name)

    def _setup_logger(self):
        """Setup logging display"""
        self.log_div = Div(
            text="",
            width=800,
            height=50,
            styles={
                "font-family": "Monaco, monospace",
                "font-size": "10px",
                "background-color": "#f8f9fa",
                "border": "1px solid #dee2e6",
                "border-radius": "4px",
                "padding": "4px",
            },
        )

    def _log_info(self, message: str):
        """Log info message"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_div.text = f"[{timestamp}] {message}"

    def _precompute_trajectory_data(self):
        """Precompute trajectory visualization data"""
        self.head_radius = 4.0
        self.head_cos = self.head_radius * np.cos(self.position_data["head_direction"])
        self.head_sin = self.head_radius * np.sin(self.position_data["head_direction"])

    def _create_plots(self):
        """Create all plots with clean styling"""

        # Spatial plot
        self.spatial_plot = figure(
            width=500,
            height=500,
            title="Position Decoding",
            x_axis_label="X Position (cm)",
            y_axis_label="Y Position (cm)",
            tools="pan,wheel_zoom,box_zoom,reset",
            match_aspect=True,
        )

        self._style_plot(self.spatial_plot)

        # Add spatial elements
        self._create_spatial_elements()

        # Time series plots with click support
        self.time_series_plots = {}
        self.shared_x_range = Range1d(start=0, end=self.window_duration)

        for config in self.time_series_manager.series_configs:
            plot = self._create_time_series_plot(config)
            self.time_series_plots[config.name] = plot

    def _create_spatial_elements(self):
        """Create spatial plot elements"""

        # Data sources - only for current position, no full trajectory
        self.current_pos_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.head_dir_source = ColumnDataSource(data=dict(x0=[], y0=[], x1=[], y1=[]))
        self.map_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.heatmap_source = ColumnDataSource(
            data=dict(image=[], x=[], y=[], dw=[], dh=[])
        )

        # Heatmap (rendered first, goes behind everything)
        self.heatmap_renderer = self.spatial_plot.image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=self.heatmap_source,
            palette=Viridis256,
            alpha=0.85,
        )

        # Current position marker (on top of heatmap)
        self.spatial_plot.scatter(
            "x",
            "y",
            source=self.current_pos_source,
            size=16,
            color="#FF1493",
            alpha=1.0,
            line_color="white",
            line_width=2,
        )

        # Head direction line (on top of everything, very prominent)
        self.spatial_plot.segment(
            "x0",
            "y0",
            "x1",
            "y1",
            source=self.head_dir_source,
            color="#FF1493",
            line_width=5,
            alpha=1.0,
        )

        # MAP estimate (green dot, less prominent)
        self.spatial_plot.scatter(
            "x",
            "y",
            source=self.map_source,
            size=12,
            color="#32CD32",
            alpha=0.8,
            line_color="white",
            line_width=1,
        )

    def _create_time_series_plot(self, config: TimeSeriesConfig) -> figure:
        """Create a clickable time series plot"""

        plot = figure(
            width=450,
            height=150,
            title=config.label,
            y_axis_label=config.y_label,
            tools="pan,wheel_zoom,box_zoom,reset",  # Removed tap from tools since we handle it manually
            x_range=self.shared_x_range,
        )

        self._style_plot(plot)

        # Data sources for this series
        use_decimation = self.n_time > 10000

        if use_decimation:
            factor = max(1, self.n_time // 10000)
            time_data, values_data = self.time_series_manager.get_decimated_data(
                config.name, factor
            )
        else:
            time_data = config.time_coords
            values_data = self.time_series_manager._extract_values(config.data)

        # Full data source (background)
        full_source = ColumnDataSource(data=dict(time=time_data, values=values_data))

        # Window data source (foreground)
        window_source = ColumnDataSource(data=dict(time=[], values=[]))

        # Time indicator
        time_indicator = ColumnDataSource(data=dict(time=[], y=[]))

        # Store sources
        setattr(self, f"{config.name}_full_source", full_source)
        setattr(self, f"{config.name}_window_source", window_source)
        setattr(self, f"{config.name}_time_ind", time_indicator)

        # Plot lines
        plot.line(
            "time",
            "values",
            source=full_source,
            alpha=0.25,
            line_width=1,
            color="#888888",
        )

        plot.line(
            "time",
            "values",
            source=window_source,
            line_width=config.line_width,
            color=config.color,
            alpha=config.alpha,
        )

        plot.line(
            "time",
            "y",
            source=time_indicator,
            line_width=2,
            color="#FF4444",
            line_dash="dashed",
            alpha=0.8,
        )

        # Make clickable - THIS IS THE FIXED PART
        self._make_plot_clickable(plot, config.name)

        return plot

    def _make_plot_clickable(self, plot: figure, series_name: str):
        """FIXED: Make time series plot clickable to jump to time"""

        # Create invisible clickable scatter points covering the full data range
        full_source = getattr(self, f"{series_name}_full_source")

        # Create an invisible scatter renderer for click detection
        # Use larger size and make it completely transparent
        click_renderer = plot.scatter(
            "time",
            "values",
            source=full_source,
            size=15,  # Larger click target
            alpha=0,  # Completely invisible
            color="red",
        )

        # Store reference to viewer instance for callback
        viewer_ref = self

        def on_plot_tap(event):
            """Handle tap events on the plot"""
            try:
                # Get click coordinates - use sx/sy (screen coordinates) if x/y not available
                click_x = getattr(event, "x", None)

                if click_x is None:
                    # Fallback: try to get from screen coordinates
                    if hasattr(event, "sx") and hasattr(plot, "x_range"):
                        # Convert screen x to data x coordinate
                        click_x = plot.x_range.start + (event.sx / plot.width) * (
                            plot.x_range.end - plot.x_range.start
                        )

                if click_x is not None:
                    # Find closest time index
                    time_diffs = np.abs(viewer_ref.times - click_x)
                    closest_idx = np.argmin(time_diffs)

                    # Jump to that time (this will trigger the slider callback)
                    viewer_ref.time_slider.value = closest_idx

                    # Log the click for debugging
                    viewer_ref._log_info(
                        f"Clicked {series_name} at time {click_x:.3f}s -> frame {closest_idx}"
                    )

                else:
                    viewer_ref._log_info(
                        f"Click on {series_name}: no coordinates available"
                    )

            except Exception as e:
                viewer_ref._log_info(f"Click error on {series_name}: {str(e)}")

        # Bind the tap event to the plot
        plot.on_event(Tap, on_plot_tap)

        # Also make the plot itself selectable for better click detection
        plot.toolbar.active_tap = "auto"

    def _style_plot(self, plot: figure):
        """Apply consistent plot styling"""
        plot.title.text_font_size = "13pt"
        plot.title.text_color = "#2F2F2F"
        plot.axis.axis_label_text_font_size = "11pt"
        plot.axis.major_label_text_font_size = "9pt"
        plot.grid.grid_line_alpha = 0.2
        plot.toolbar_location = None
        plot.outline_line_color = "#E0E0E0"

    def _create_controls(self):
        """Create control widgets"""

        self.time_slider = Slider(
            start=0,
            end=self.n_time - 1,
            value=0,
            step=1,
            title="Timeline",
            width=800,
            height=50,
            bar_color="#4287f5",
        )

        self.play_button = Button(label="▶ Play", width=100, height=40)

        self.speed_select = Select(
            title="Speed",
            value="1x",
            options=["1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x"],
            width=120,
            height=60,
        )

        self.window_select = Select(
            title="Window",
            value="0.8s",
            options=["0.4s", "0.8s", "1.6s", "3.2s", "6.4s"],
            width=120,
            height=60,
        )

        self.info_div = Div(
            text="Ready",
            width=400,
            height=50,
            styles={
                "font-family": "Monaco, monospace",
                "font-size": "12px",
                "background-color": "#f8f9fa",
                "border": "1px solid #dee2e6",
                "border-radius": "4px",
                "padding": "8px",
            },
        )

    def _setup_callbacks(self):
        """Setup all callbacks"""

        self.time_slider.on_change(
            "value", self._on_time_change
        )  # Changed from "value_throttled"
        self.play_button.on_click(self._toggle_play)
        self.speed_select.on_change("value", self._on_speed_change)
        self.window_select.on_change("value", self._on_window_change)

        self.play_callback = None
        self.playback_interval = int(1000 / self.sampling_frequency)

        # Update initial frame
        self._update_frame(0)

    def _on_time_change(self, attr, old, new):
        """Handle time slider changes - FIXED to work with clicks"""
        new_idx = int(new)
        old_idx = int(old) if old is not None else self.current_time_idx

        # Always update if the value actually changed, regardless of source
        if new_idx != old_idx:
            self._log_info(
                f"Slider changed from {old_idx} to {new_idx} - updating frame"
            )
            self._update_frame(new_idx)
        else:
            self._log_info(f"Slider value {new_idx} unchanged - no update needed")

    def _on_speed_change(self, attr, old, new):
        """Handle speed changes"""
        speed_multipliers = {
            "1/8x": 8.0,
            "1/4x": 4.0,
            "1/2x": 2.0,
            "1x": 1.0,
            "2x": 0.5,
            "4x": 0.25,
            "8x": 0.125,
        }

        multiplier = speed_multipliers.get(new, 1.0)
        self.playback_interval = int((1000 / self.sampling_frequency) * multiplier)

        # Restart playback with new speed if currently playing
        if self.playing and self.play_callback:
            curdoc().remove_periodic_callback(self.play_callback)
            self.play_callback = curdoc().add_periodic_callback(
                self._advance_frame, self.playback_interval
            )
            self._log_info(f"Playback speed changed to {new}")

    def _on_window_change(self, attr, old, new):
        """Handle window size changes"""
        window_durations = {
            "0.4s": 0.4,
            "0.8s": 0.8,
            "1.6s": 1.6,
            "3.2s": 3.2,
            "6.4s": 6.4,
        }

        self.window_duration = window_durations.get(new, 0.8)
        self._update_frame(self.current_time_idx)
        self._log_info(f"Time window changed to {new}")

    def _toggle_play(self):
        """Toggle play/pause with improved state management"""

        if not self.playing:
            # Start playing
            self.playing = True
            self.play_button.label = "⏸ Pause"

            # Remove any existing callback first
            if self.play_callback:
                try:
                    curdoc().remove_periodic_callback(self.play_callback)
                except:
                    pass

            # Add new callback
            self.play_callback = curdoc().add_periodic_callback(
                self._advance_frame, self.playback_interval
            )
            self._log_info("Started playback")

        else:
            # Stop playing
            self.playing = False
            self.play_button.label = "▶ Play"

            if self.play_callback:
                try:
                    curdoc().remove_periodic_callback(self.play_callback)
                    self.play_callback = None
                except:
                    pass
            self._log_info("Stopped playback")

    def _advance_frame(self):
        """Advance to next frame with better error handling"""

        if not self.playing:
            return

        try:
            if self.current_time_idx < self.n_time - 1:
                # Advance to next frame
                next_idx = self.current_time_idx + 1
                self._update_frame(next_idx)

                # Update slider value without triggering callback
                self.time_slider.value = next_idx

            else:
                # End of data - stop playback
                self.playing = False
                self.play_button.label = "▶ Play"
                if self.play_callback:
                    curdoc().remove_periodic_callback(self.play_callback)
                    self.play_callback = None
                self._log_info("Reached end of data")

        except Exception as e:
            # Stop on any error
            self.playing = False
            self.play_button.label = "▶ Play"
            if self.play_callback:
                curdoc().remove_periodic_callback(self.play_callback)
                self.play_callback = None
            self._log_info(f"Playback error: {e}")

    def _update_frame(self, time_idx: int):
        """Update all displays for given time index"""
        if time_idx >= self.n_time:
            time_idx = self.n_time - 1

        self.current_time_idx = time_idx
        current_time = self.times[time_idx]

        # Update spatial elements
        self._update_spatial_heatmap(time_idx)
        self._update_spatial_markers(time_idx)

        # Update time series windows
        self._update_time_series_windows(current_time)

        # Update info
        self._update_info_display(time_idx, current_time)

    def _update_spatial_heatmap(self, time_idx: int):
        """Update spatial heatmap efficiently for huge datasets"""

        # Check frame cache first (much smaller cache for huge datasets)
        if time_idx in self.frame_cache:
            cached_data = self.frame_cache[time_idx]
            self.frame_cache.move_to_end(time_idx)  # LRU update
            self.heatmap_source.data = cached_data
            return

        try:
            # Load frame data (this handles chunking automatically)
            posterior_slice = self.posterior_manager.get_frame(time_idx)

            frame_data = dict(
                image=[posterior_slice],
                x=[self.x_coords.min()],
                y=[self.y_coords.min()],
                dw=[self.x_coords.max() - self.x_coords.min()],
                dh=[self.y_coords.max() - self.y_coords.min()],
            )

            self.heatmap_source.data = frame_data

            # Aggressive cache management for huge datasets
            while len(self.frame_cache) >= self.max_cache_size:
                evicted = self.frame_cache.popitem(last=False)

            self.frame_cache[time_idx] = frame_data

        except Exception as e:
            self._log_info(f"Error loading frame {time_idx}: {e}")
            # Continue with previous frame on error

    def _update_spatial_markers(self, time_idx: int):
        """Update position markers efficiently"""

        # Bounds check
        if time_idx >= len(self.position_data):
            time_idx = len(self.position_data) - 1

        # Current position
        curr_x = self.position_data["x"].iloc[time_idx]
        curr_y = self.position_data["y"].iloc[time_idx]
        self.current_pos_source.data = dict(x=[curr_x], y=[curr_y])

        # Head direction (precomputed for efficiency)
        x_end = curr_x + self.head_cos[time_idx]
        y_end = curr_y + self.head_sin[time_idx]
        self.head_dir_source.data = dict(
            x0=[curr_x], y0=[curr_y], x1=[x_end], y1=[y_end]
        )

        # MAP estimate
        try:
            posterior_slice = self.posterior_manager.get_frame(time_idx)
            flat_idx = np.argmax(posterior_slice)
            x_idx, y_idx = np.unravel_index(flat_idx, posterior_slice.shape)
            map_x = self.x_coords[x_idx]
            map_y = self.y_coords[y_idx]
            self.map_source.data = dict(x=[map_x], y=[map_y])
        except Exception as e:
            # Keep previous MAP position on error
            pass

    def _update_time_series_windows(self, current_time: float):
        """Update time series windows efficiently"""

        window_start = current_time - (self.window_duration / 2)
        window_end = current_time + (self.window_duration / 2)

        # Update shared range
        self.shared_x_range.start = window_start
        self.shared_x_range.end = window_end

        # Update each time series
        for config in self.time_series_manager.series_configs:
            self._update_single_time_series(
                config, window_start, window_end, current_time
            )

    def _update_single_time_series(
        self,
        config: TimeSeriesConfig,
        window_start: float,
        window_end: float,
        current_time: float,
    ):
        """Update a single time series window"""

        # Get windowed data
        window_time, window_values = self.time_series_manager.get_windowed_data(
            config.name, window_start, window_end
        )

        if len(window_time) > 0:
            # Update window data
            window_source = getattr(self, f"{config.name}_window_source")
            window_source.data.update(dict(time=window_time, values=window_values))

            # Update time indicator
            full_values = self.time_series_manager._extract_values(config.data)
            y_range = [np.min(full_values), np.max(full_values)]

            time_indicator = getattr(self, f"{config.name}_time_ind")
            time_indicator.data.update(
                dict(time=[current_time, current_time], y=y_range)
            )

    def _update_info_display(self, time_idx: int, current_time: float):
        """Update info display with current values"""

        # Get current values from each time series
        values_text = []
        for config in self.time_series_manager.series_configs:
            if time_idx < len(config.time_coords):
                current_value = self.time_series_manager._extract_values(config.data)[
                    time_idx
                ]
                values_text.append(
                    f"<strong>{config.label}:</strong> {current_value:.2f}"
                )

        self.info_div.text = f"""
        <div style="line-height: 1.4;">
        <strong>Frame:</strong> {time_idx:,} / {self.n_time-1:,} &nbsp;&nbsp;
        <strong>Time:</strong> {current_time:.3f}s<br>
        {' &nbsp;&nbsp; '.join(values_text)}
        </div>
        """

    def create_layout(self):
        """Create the complete layout"""

        # Control bar
        control_bar = row(
            self.play_button,
            self.speed_select,
            self.window_select,
            self.info_div,
            sizing_mode="fixed",
        )

        # Time series column
        time_series_plots = [
            self.time_series_plots[config.name]
            for config in self.time_series_manager.series_configs
        ]
        time_series_column = column(*time_series_plots, sizing_mode="fixed")

        # Main content
        main_row = row(self.spatial_plot, time_series_column, sizing_mode="fixed")

        # Timeline
        timeline_section = column(self.time_slider, sizing_mode="fixed")

        # Complete layout
        return column(
            control_bar,
            main_row,
            timeline_section,
            self.log_div,
            sizing_mode="fixed",
            margin=(10, 10, 10, 10),
        )


# Factory function for clean instantiation
def create_neural_viewer(
    position_data: pd.DataFrame,
    posterior_data: xr.DataArray,
    time_series_data: Dict[str, Any],
    sampling_frequency: float = 250,
) -> NeuralViewer:
    """
    Factory function to create neural viewer

    Parameters:
    -----------
    position_data : pd.DataFrame
        Must contain columns: 'time', 'x', 'y', 'head_direction'
        Optional: 'speed' (will be included in time series)
    posterior_data : xr.DataArray
        Spatial-temporal posterior with dims ['time', 'x_position', 'y_position']
    time_series_data : Dict[str, Union[xr.DataArray, pd.Series, np.ndarray]]
        Dictionary of time series to plot. Keys become plot labels.
        Example: {'multiunit_rate': mua_data, 'theta_power': theta_data}
    sampling_frequency : float
        Data sampling frequency in Hz

    Returns:
    --------
    NeuralViewer
        Configured viewer ready for layout creation
    """
    return NeuralViewer(
        position_data, posterior_data, time_series_data, sampling_frequency
    )


# Demo data generation
def create_demo_data(n_time: int = 2000, sampling_frequency: float = 250) -> tuple:
    """
    Create demo dataset for testing

    Returns:
    --------
    tuple
        (position_data, posterior_data, time_series_data)
    """
    simulator = DataSimulator(n_time, sampling_frequency)
    position_data, posterior_data, neural_data = simulator.create_full_dataset()

    return position_data, posterior_data, neural_data


# Main execution - runs directly when bokeh serve loads the file
try:
    print("Creating demo dataset...")
    position_data, posterior_data, time_series_data = create_demo_data()

    print("Initializing neural viewer...")
    viewer = create_neural_viewer(
        position_data=position_data,
        posterior_data=posterior_data,
        time_series_data=time_series_data,
        sampling_frequency=250,
    )

    print("Setting up layout...")
    layout = viewer.create_layout()
    curdoc().add_root(layout)
    curdoc().title = "Neural Decoding Viewer - Fixed Clicking"

    print("Viewer ready! Features:")
    print("- FIXED: Click on any time series plot to jump to that time")
    print("- Automatic decimation for large datasets")
    print("- Efficient chunked data support")
    print("- Add any time series data to time_series_data dict")

except Exception as e:
    print(f"Error initializing viewer: {e}")
    import traceback

    traceback.print_exc()

    # Minimal fallback
    from bokeh.plotting import figure

    fallback_plot = figure(width=400, height=400, title="Error - Check Console")
    fallback_plot.line([1, 2, 3], [1, 2, 3])
    error_div = Div(text=f"<h3>Error:</h3><p>{str(e)}</p>")
    curdoc().add_root(column(error_div, fallback_plot))
