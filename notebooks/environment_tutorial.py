"""
tutorial_environment_for_neuroscientists.py

A step-by-step tutorial guiding hippocampal neuroscientists through the `environment`
package, focusing on its application to spatial analysis and Bayesian decoding.

The Story:
----------
We, as neuroscientists, have recordings from an animal navigating a maze.
Our goal is often to understand how neural activity relates to the animal's
position, or to decode the animal's position from neural activity. This requires
a precise, discretized model of the experimental environment. This tutorial
demonstrates how to create, manipulate, and analyze such environmental models.

Key Workflow & Features Covered:
---------------------------------
1.  **Initial Data**: Starting with raw animal position data (e.g., from video tracking).
2.  **Coordinate Calibration**: Converting pixel coordinates to physical units (cm).
3.  **Defining Environments from Position Data**: Creating a `RegularGrid` environment based on where the animal went.
4.  **Core Environment Operations**:
    * Mapping between continuous positions and discrete bins (`bin_at`, `bin_center_of`).
    * Understanding spatial properties: `bin_sizes`, `active_mask`, `dimension_ranges`.
    * Analyzing connectivity: `neighbors`, `distance_between`, `shortest_path`.
5.  **Defining Maze-Relevant Regions**: Segmenting the environment into meaningful parts like arms, choice points, and reward wells.
6.  **Alternative Ways to Define Environments**:
    * From an image mask of the maze.
    * Using `GraphLayout` for linear tracks or abstract 1D spaces (like head direction).
    * Using `Polygon` or `TriangularMesh` for complex boundaries.
    * Briefly, other grid types like `HexagonalLayout` and `MaskedGridLayout`.
7.  **Handling Complex Mazes & Multiple Sessions**:
    * `CompositeEnvironment`: Modeling mazes with distinct parts (e.g., a central area плюс arms).
    * `Alignment`: Mapping data (e.g., place fields) between slightly different environments or recording sessions.
8.  **API Discovery**: Finding available layouts and their parameters.

Prerequisites:
-------------
- `non_local_detector` installed.
- `matplotlib`, `scipy`, `shapely`, `track_linearization`.

Let's begin our journey of parameterizing space! 🚀
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from track_linearization import make_track_graph  # For graph-based environments

# Core imports from the environment package
from non_local_detector.environment import get_layout_parameters, list_available_layouts
from non_local_detector.environment.alignment import (
    apply_similarity_transform,
    get_2d_rotation_matrix,
    map_probabilities_to_nearest_target_bin,
)
from non_local_detector.environment.calibration import simple_scale
from non_local_detector.environment.composite import CompositeEnvironment
from non_local_detector.environment.environment import Environment
from non_local_detector.environment.layout.engines.hexagonal import (
    HexagonalLayout,  # Example for isinstance
)
from non_local_detector.environment.regions import plot_regions

# Consistent styling for raw data points in plots
RAW_DATA_STYLE = {"s": 5, "c": "lightcoral", "alpha": 0.5, "label": "Animal Positions"}
MAZE_BOUNDARY_STYLE = {"color": "darkblue", "linewidth": 1.5, "label": "Maze Boundary"}
REWARD_WELL_STYLE = {
    "s": 150,
    "marker": "*",
    "alpha": 0.9,
    "label": "Reward Well",
    "zorder": 10,
}
ARM_REGION_STYLE = {"alpha": 0.2, "linewidth": 1.0, "edgecolor": "black"}

# %% [markdown]
# ## Part 1: From Raw Tracking Data to a Usable Spatial Representation
#
# Imagine we've just run an experiment. We have video tracking data of a rat exploring a U-shaped maze.
# This data is likely in **pixel coordinates** from the camera. Our first step is to convert it to
# **physical units (cm)** and then create a basic spatial model.

# %% [markdown]
# ### 1.1 Simulate Raw Position Data (Pixels)
# For this tutorial, we'll simulate some noisy data. In a real scenario, this would come from your tracking software.
# Let's assume a 640x480 pixel video frame. Our U-shaped maze path in *pixel space*:
# (50,430) -> (50,50) -> (350,50) -> (350,430). (Origin: top-left for raw pixels)

# %%
# Path in PIXEL coordinates (origin top-left)
px_path_y_coords = np.linspace(
    50, 430, 50
)  # Y goes from 50 (top) to 430 (bottom) for first segment
px_path_x_coords = np.linspace(50, 350, 50)  # X for horizontal segment

seg1_px = np.stack(
    (np.ones_like(px_path_y_coords) * 50, px_path_y_coords[::-1]), axis=1
)  # (50, 430) -> (50, 50)
seg2_px = np.stack(
    (px_path_x_coords, np.ones_like(px_path_x_coords) * 50), axis=1
)  # (50, 50) -> (350, 50)
seg3_px = np.stack(
    (np.ones_like(px_path_y_coords) * 350, px_path_y_coords), axis=1
)  # (350, 50) -> (350, 430)

raw_pixel_positions = np.vstack([seg1_px, seg2_px, seg3_px])
raw_pixel_positions += multivariate_normal(mean=[0, 0], cov=2.0).rvs(
    size=raw_pixel_positions.shape[0]
)  # Pixel noise

plt.figure(figsize=(6, 4.8))  # Aspect ratio of 640x480
plt.scatter(
    raw_pixel_positions[:, 0],
    raw_pixel_positions[:, 1],
    s=3,
    label="Raw Pixel Positions",
)
plt.title("Raw Animal Positions (Pixel Coordinates)")
plt.xlabel("X pixel")
plt.ylabel("Y pixel (origin top-left)")
plt.xlim(0, 640)
plt.ylim(480, 0)  # Invert Y axis to match typical image coordinates
plt.axhline(0, color="gray", linestyle=":")
plt.axvline(0, color="gray", linestyle=":")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()

# %% [markdown]
# ### 1.2 Coordinate Calibration: Pixels to Centimeters
# The `Environment` typically works in physical units (like cm). We need to calibrate.
# Let's say the horizontal segment (300 pixels wide: from 50px to 350px) is 60 cm long.
# So, `px_per_cm = 300px / 60cm = 5 px/cm`.
# Let's also define that the point (50px, 50px) in *our pixel data's coordinate system* (top-left origin)
# should become our physical `(0,0) cm` point in a *bottom-left origin* system.

# %%
# Calibration parameters
frame_height_px = 480.0  # For y-flip
px_per_cm_calibration = 5.0
# The physical (0,0) cm origin corresponds to pixel (50, 50) in raw top-left pixel coords.

# We use `non_local_detector.transforms.convert_to_cm` which combines y-flip and scaling.
# It expects cm_per_px, so we need px_per_cm.
# No, `convert_to_cm` takes `cm_per_px`. If 5 px = 1 cm, then 1 px = 1/5 cm = 0.2 cm.
cm_per_px_calibration = 1.0 / px_per_cm_calibration  # 0.2 cm per pixel

# `convert_to_cm` assumes the scaling is applied around a (0,0) of the *y-flipped* frame.
# If (50,50)px_raw is (0,0)cm_physical_bottom_left:
# 1. Y-flip: (50, 50)px_raw -> x=50, y_flipped = 480-50 = 430. So (50, 430) in bottom-left pixel coords.
# 2. Scale this (50,430) in px_bottom_left to get (0,0)cm.
#    x_cm = (x_px_bottom_left - offset_x_px_bottom_left) * cm_per_px
#    y_cm = (y_px_bottom_left - offset_y_px_bottom_left) * cm_per_px
#    So, (offset_x_px_bottom_left, offset_y_px_bottom_left) = (50, 430)
# This is tricky. Let's use the Affine2D composition for clarity:
from non_local_detector.environment.transforms import flip_y, scale_2d, translate

# Transform 1: Flip Y axis (origin from top-left to bottom-left for pixels)
T_flip = flip_y(frame_height_px=frame_height_px)
# After T_flip, our point (50,50)px_raw becomes (50, 480-50=430)px_flipped. This should be (0,0)cm.

# Transform 2: Scale from pixels (bottom-left origin) to cm
T_scale = scale_2d(sx=cm_per_px_calibration, sy=cm_per_px_calibration)

# Transform 3: Translate so that (50,430)px_flipped maps to (0,0)cm.
# We want ( (px_flipped_x * scale_x) + trans_x ) = cm_x
# For (50,430) -> (0,0):
# (50 * cm_per_px) + trans_x = 0  => trans_x = -50 * cm_per_px = -50 * 0.2 = -10
# (430 * cm_per_px) + trans_y = 0 => trans_y = -430 * cm_per_px = -430 * 0.2 = -86
T_translate = translate(tx=-50 * cm_per_px_calibration, ty=-430 * cm_per_px_calibration)

# Compose: apply flip first, then scale, then translate. Order: T_translate @ T_scale @ T_flip
# Correct composition for p' = M3 @ M2 @ M1 @ p is T_translate.compose(T_scale.compose(T_flip))
pixel_to_cm_transform = T_translate @ T_scale @ T_flip

# Apply to our raw pixel positions
position_data_cm = pixel_to_cm_transform(raw_pixel_positions)

print(f"\n--- Coordinate Calibration ---")
print(
    f"Calibration: {px_per_cm_calibration} px/cm, physical origin at raw pixel (50,50)"
)
print(
    f"Example raw pixel point {raw_pixel_positions[0]} converted to cm: {position_data_cm[0]}"
)
print(
    f"Example raw pixel point {raw_pixel_positions[50]} (start of seg2) converted to cm: {position_data_cm[50]}"
)  # (50,50)px -> (0,0)cm
print(
    f"Example raw pixel point {raw_pixel_positions[-1]} converted to cm: {position_data_cm[-1]}"
)

# Plot calibrated (cm) data
plt.figure(figsize=(6, 6))
plt.scatter(position_data_cm[:, 0], position_data_cm[:, 1], **RAW_DATA_STYLE)
plt.title("Calibrated Animal Positions (cm)")
plt.xlabel("X coordinate (cm)")
plt.ylabel("Y coordinate (cm)")
plt.axhline(0, color="gray", linestyle=":")
plt.axvline(0, color="gray", linestyle=":")
plt.axis("equal")
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend()
plt.show()

# %% [markdown]
# ### 1.3 Creating a `RegularGrid` Environment from Calibrated Position Samples
# Now that we have positions in cm, we can create our first `Environment`.
# `Environment.from_samples` is a convenient way to do this when you have a cloud of position samples.
# It will infer the spatial extent and create active bins where the animal visited.

# %%
grid_env_main = Environment.from_samples(
    data_samples=position_data_cm,
    name="MainMaze_Grid",
    layout_kind="RegularGrid",  # Default, but good to be explicit
    bin_size=3.0,  # Each grid cell will be 3cm x 3cm
    infer_active_bins=True,  # Default, keeps only bins with data
    bin_count_threshold=0,  # Bins with at least 0 sample are active
)

print("\n--- Main RegularGrid Environment (from cm data) ---")
print(f"Environment Name: {grid_env_main.name}")
print(f"Layout Type: {grid_env_main.layout_type}")
print(f"Number of active bins: {grid_env_main.n_bins}")
print(f"Number of dimensions: {grid_env_main.n_dims}")
print(f"Spatial extent (dimension_ranges): {grid_env_main.dimension_ranges}")

# Plot this environment
plt.figure(figsize=(6, 6))
ax = grid_env_main.plot(show_connectivity=False)  # Shows pcolormesh of active bins
ax.scatter(position_data_cm[:, 0], position_data_cm[:, 1], **RAW_DATA_STYLE, zorder=5)
ax.set_title(f"{grid_env_main.name}: Active Bins & Calibrated Positions")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.axis("equal")
plt.legend()
plt.show()

# %% [markdown]
# ## Part 2: Interacting with the Discretized Environment
# Once the environment is defined, we can query it. This is crucial for relating continuous data (like animal position or decoded estimates) to the discrete model used by, for example, a Bayesian decoder.

# %% [markdown]
# ### 2.1 Mapping Continuous Positions to Discrete Bins (and back)

# %%
if grid_env_main.n_bins > 0:
    # Pick a test point (e.g., the first recorded cm position)
    test_point_cm = position_data_cm[0].reshape(1, -1)

    # Find which discrete bin this continuous point falls into
    bin_idx = grid_env_main.bin_at(test_point_cm)[0]  # bin_at returns an array
    print(f"\nContinuous point {test_point_cm[0]} cm maps to bin index: {bin_idx}")

    if bin_idx != -1:
        # Get the center coordinates of this bin
        center_of_bin = grid_env_main.bin_center_of(bin_idx)
        print(f"The center of bin {bin_idx} is at: {center_of_bin} cm")

        # Check if other points are inside any active bin
        far_away_point = np.array([[-1000.0, -1000.0]])
        print(
            f"Does {test_point_cm[0]} fall in an active bin? {grid_env_main.contains(test_point_cm)[0]}"
        )
        print(
            f"Does {far_away_point[0]} fall in an active bin? {grid_env_main.contains(far_away_point)[0]}"
        )
else:
    print("\nMain grid environment has no active bins, skipping interaction demo.")

# %% [markdown]
# ### 2.2 Understanding Bin Properties and Connectivity
# The discrete representation isn't just a collection of bins; it also defines how they are connected.
# This connectivity is essential for modeling movement probabilities (transition matrix) in a decoder.

# %%
if grid_env_main.n_bins > 0:
    print(f"\n--- Bin & Connectivity Properties for {grid_env_main.name} ---")
    # Size of bins (area for 2D, length for 1D)
    print(f"Size (area) of the first 5 active bins: {grid_env_main.bin_sizes[:5]} cm^2")

    # Active mask: a boolean array showing active bins on the original *full* grid concept
    print(
        f"Shape of the active_mask (for the full conceptual grid): {grid_env_main.active_mask.shape}"
    )
    print(f"Number of True values in active_mask: {np.sum(grid_env_main.active_mask)}")

    # Boundary bins: active bins at the "edge" of the environment
    print(f"First 10 boundary bin indices: {grid_env_main.boundary_bins[:10]}")

    # Neighbors of a bin
    if bin_idx != -1 and bin_idx < grid_env_main.n_bins:  # Ensure bin_idx is valid
        neighbors_of_bin = grid_env_main.neighbors(bin_idx)
        print(f"Neighbors of bin {bin_idx}: {neighbors_of_bin}")

        # Geodesic distance (shortest path along the graph)
        if len(neighbors_of_bin) > 0:
            # Distance between bin_idx and its first neighbor
            dist_to_neighbor = grid_env_main.distance_between(
                grid_env_main.bin_center_of(bin_idx),
                grid_env_main.bin_center_of(neighbors_of_bin[0]),
            )
            print(
                f"Geodesic distance between bin {bin_idx} and neighbor {neighbors_of_bin[0]}: {dist_to_neighbor:.2f} cm"
            )

            # Shortest path (sequence of bin indices)
            target_bin_idx_for_path = grid_env_main.boundary_bins[0]  # Example target
            if bin_idx != target_bin_idx_for_path:
                path = grid_env_main.shortest_path(bin_idx, target_bin_idx_for_path)
                print(
                    f"Shortest path from bin {bin_idx} to {target_bin_idx_for_path}: {path[:5]}... (first 5 steps)"
                )

    # Plot with connectivity
    plt.figure(figsize=(6, 6))
    ax_conn = grid_env_main.plot(
        show_connectivity=True,
        node_size=5,
        # kwargs for _GridMixin.plot for edges:
        # (No direct edge_kwargs, it draws lines manually)
        # We can make nodes smaller to see edges
        # Or, pass kwargs to underlying networkx draw for more control if layout's plot supports it
    )
    ax_conn.set_title(f"{grid_env_main.name} with Bin Connectivity")
    plt.show()
else:
    print(f"\n{grid_env_main.name} has no active bins, skipping connectivity demo.")


# %% [markdown]
# ## Part 3: Defining and Using Maze-Relevant Regions 🗺️
# For analysis (e.g., "is the animal decoded in the reward well?" or "average firing rate in the center arm"),
# we need to define named spatial regions.

# %%
if grid_env_main.n_bins > 0:
    print(f"\n--- Defining Regions for {grid_env_main.name} ---")
    # Let's define regions based on our U-shaped maze structure (0,60)->(0,0)->(60,0)->(60,60) in cm
    # after our calibration. The data `position_data_cm` has this extent.

    # Reward Well 1 (near start of first arm, e.g., around (0,60) if data started there)
    # Our calibrated data actually starts near (0,60) and goes to (0,0), then (60,0), then (60,60).
    # Let's put wells at the "ends" or "corners".
    # Well A: near (0,0) cm
    well_A_center = ShapelyPoint(2, 2)  # A bit offset from exact corner
    grid_env_main.regions.add(name="Well_A", point=well_A_center)

    # Well B: near (60,0) cm
    well_B_center = ShapelyPoint(58, 2)
    grid_env_main.regions.add(name="Well_B", point=well_B_center)

    # Well C: near (60,60) cm
    well_C_center = ShapelyPoint(58, 58)
    grid_env_main.regions.add(name="Well_C", point=well_C_center)

    # Arm definitions as Polygons
    # Bottom Arm (y ~ 0, x from 0 to 60)
    # Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    bottom_arm_poly = Polygon([(-2, -2), (62, -2), (62, 5), (-2, 5)])
    grid_env_main.regions.add(name="BottomArm", polygon=bottom_arm_poly)

    # Right Vertical Arm (x ~ 60, y from 0 to 60)
    right_arm_poly = Polygon([(55, -2), (62, -2), (62, 62), (55, 62)])
    grid_env_main.regions.add(name="RightArm", polygon=right_arm_poly)

    # Left Vertical Arm (x ~ 0, y from 0 to 60)
    left_arm_poly = Polygon([(-2, -2), (5, -2), (5, 62), (-2, 62)])
    grid_env_main.regions.add(name="LeftArm", polygon=left_arm_poly)

    print(f"Defined regions: {grid_env_main.regions.list_names()}")
    print(f"Area of 'BottomArm': {grid_env_main.regions.area('BottomArm'):.1f} cm^2")
    print(
        f"Area of 'Well_A' (point region): {grid_env_main.regions.area('Well_A')}"
    )  # Should be 0

    # Get bins in 'BottomArm'
    bins_in_bottom_arm = grid_env_main.bins_in_region("BottomArm")
    print(f"Number of bins in 'BottomArm': {len(bins_in_bottom_arm)}")

    # Get a boolean mask for 'Well_B' (useful for indexing data arrays)
    mask_well_B = grid_env_main.mask_for_region("Well_B")
    # Example: if `decoded_probs` is (n_time_bins, n_spatial_bins)
    # `prob_in_well_B = decoded_probs[:, mask_well_B].sum(axis=1)`
    print(f"Number of bins in 'Well_B' from mask: {np.sum(mask_well_B)}")

    # Plot the environment with these defined regions
    fig_regions, ax_regions = plt.subplots(figsize=(7, 7))
    grid_env_main.plot(
        ax=ax_regions, show_connectivity=False, alpha=0.1
    )  # Faint background
    plot_regions(
        ax=ax_regions,
        regions=grid_env_main.regions,
        point_kwargs=REWARD_WELL_STYLE,  # Use pre-defined style for points
        polygon_kwargs={**ARM_REGION_STYLE, "facecolor": "lightblue"},
    )  # Style for polygons

    # Create a custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements_regions = [
        Line2D(
            [0],
            [0],
            marker=REWARD_WELL_STYLE["marker"],
            color="w",
            label="Reward Well",
            markerfacecolor="gold",
            markersize=10,
        ),  # Gold for wells in legend
        Patch(
            facecolor="lightblue",
            edgecolor="black",
            alpha=ARM_REGION_STYLE["alpha"],
            label="Maze Arm/Zone",
        ),
    ]
    # Override point colors for plotting to distinguish them if needed
    # For now, plot_regions handles colors based on its defaults or passed kwargs

    # Manually plot well centers on top if plot_regions doesn't emphasize them enough
    for well_name, style_color in [
        ("Well_A", "red"),
        ("Well_B", "green"),
        ("Well_C", "purple"),
    ]:
        if well_name in grid_env_main.regions:
            well_point = grid_env_main.regions[well_name].data
            ax_regions.scatter(
                well_point[0],
                well_point[1],
                s=REWARD_WELL_STYLE["s"],
                marker=REWARD_WELL_STYLE["marker"],
                color=style_color,  # Distinguish wells
                label=well_name,
                zorder=10,
                edgecolor="black",
            )

    ax_regions.legend(handles=legend_elements_regions)  # Use custom legend
    ax_regions.set_title(f"Maze Regions on {grid_env_main.name}")
    plt.show()
else:
    print(f"\nSkipping Regions demo as {grid_env_main.name} has no active bins.")


# %% [markdown]
# ## Part 4: Diverse Ways to Define Environment Geometries
# While `from_samples` is common, you might define your environment from an image, a graph, or a polygon.

# %% [markdown]
# ### 4.1 From an Image Mask (e.g., from a maze schematic)

# %%
# Create a simple 2D numpy array as a boolean mask
# True means 'part of the maze', False means 'wall' or 'outside'
# Let's make a simple T-maze shape
image_maze_mask = np.zeros((20, 30), dtype=bool)
image_maze_mask[8:12, :] = True  # Horizontal stem
image_maze_mask[:10, 13:17] = True  # Vertical arm

env_from_image = Environment.from_image(
    image_mask=image_maze_mask,
    bin_size=2.0,  # Each "pixel" in the mask corresponds to a 2x2 cm bin
    name="TMaze_FromImage",
)
print(f"\n--- Environment from ImageMask ({env_from_image.name}) ---")
print(f"Number of active bins: {env_from_image.n_bins}")

plt.figure(figsize=(6, 4))
env_from_image.plot(ax=plt.gca())  # gca() gets current axes
plt.title(f"{env_from_image.name}")
plt.show()

# %% [markdown]
# ### 4.2 Graph-Based Environments (for linear tracks, 1D variables)
# Ideal for mazes with clear linear segments (T-maze arms, linear tracks) or even abstract 1D variables like head direction.

# %%
# Define a T-maze graph structure
# Node positions in cm (approximate)
#      (0,20) Node3 (Top of T)
#        |
#      (0,10) Node1 (Junction) -- (10,10) Node2 (Right end)
#        |
#      (0,0)  Node0 (Stem base)

tmaze_nodes = [(0, 0), (0, 10), (10, 10), (0, 20)]
tmaze_edges = [
    (0, 1),
    (1, 2),
    (1, 3),
]  # Stem, Right arm, Left arm (relative to junction)
tmaze_graph_def = make_track_graph(tmaze_nodes, tmaze_edges)

# For a Graph environment, `edge_order` is crucial for linearization
tmaze_edge_order = [
    (0, 1),
    (1, 3),
    (1, 2),
]  # Stem -> Left Arm -> Right Arm (example order)

env_tmaze_graph = Environment.from_graph(
    graph=tmaze_graph_def,
    edge_order=tmaze_edge_order,
    edge_spacing=0.1,  # Small gap between linearized segments for clarity
    bin_size=1.0,  # Each bin along the graph is 1cm long
    name="TMaze_Graph",
)

print(f"\n--- Graph-Based Environment ({env_tmaze_graph.name}) ---")
print(f"Is it 1D? {env_tmaze_graph.is_1d}")  # True
print(f"Number of bins: {env_tmaze_graph.n_bins}")

# Plot its N-D embedding
fig_tgraph, ax_tgraph = plt.subplots(figsize=(5, 5))
env_tmaze_graph.plot(ax=ax_tgraph, bin_node_kwargs={"node_size": 20})
ax_tgraph.set_title(f"{env_tmaze_graph.name} (2D Embedding)")
plt.show()

# Plot its 1D linearized representation
fig_tgraph_1d, ax_tgraph_1d = plt.subplots(figsize=(8, 2))
env_tmaze_graph.plot_1D(ax=ax_tgraph_1d)  # Specific method for 1D graph plots
ax_tgraph_1d.set_title(f"{env_tmaze_graph.name} (1D Linearized)")
plt.show()

# Using to_linear and linear_to_nd
if env_tmaze_graph.n_bins > 0:
    example_nd_point_on_tmaze = np.array([[0, 15.0]])  # On the N arm of T
    linear_coord = env_tmaze_graph.to_linear(example_nd_point_on_tmaze)
    print(
        f"Point {example_nd_point_on_tmaze[0]} on T-maze -> Linear coord: {linear_coord[0]:.2f}"
    )
    reproj_nd_point = env_tmaze_graph.linear_to_nd(linear_coord)
    print(f"Linear coord {linear_coord[0]:.2f} -> N-D coord: {reproj_nd_point[0]}")

# %% [markdown]
# *Aside: Abstract Variables*: A `GraphLayout` could represent a circular variable like head direction (0-360°). The 'nodes' would be angles, 'pos' attributes could be `(cos(angle), sin(angle))` for embedding on a circle, and `bin_size` would be in degrees.

# %% [markdown]
# ### 4.3 Complex Boundaries: `Polygon` and `TriangularMesh`
# For mazes with arbitrary shapes, like a circular arena or a custom-designed open field.

# %%
# Circular Arena defined by a Polygon
# Shapely can approximate a circle with a buffered point
arena_center = ShapelyPoint(0.0, 0.0)
arena_radius = 25.0  # cm
circular_arena_poly = arena_center.buffer(
    arena_radius
)  # Creates a polygonal approximation of a circle

env_circle_poly = Environment.from_polygon(
    polygon=circular_arena_poly,
    bin_size=3.0,  # Bins inside the circle
    name="CircularArena_Polygon",
)
print(f"\n--- Polygon-Defined Environment ({env_circle_poly.name}) ---")
print(f"Number of bins: {env_circle_poly.n_bins}")

fig_poly, ax_poly = plt.subplots(figsize=(6, 6))
env_circle_poly.plot(
    ax=ax_poly, polygon_kwargs={"fc": "lightcyan", "ec": "teal", "alpha": 0.8}
)
ax_poly.set_title(f"{env_circle_poly.name}")
plt.show()

# For even more complex, organic boundaries, or when you want a mesh representation:
# (Reusing the U-shape polygon from your original script for TriangularMesh)
w_mesh = 5.0
u_shape_coords_mesh = [
    (-w_mesh, 30.0 + w_mesh),
    (-w_mesh, -w_mesh),
    (30.0 + w_mesh, -w_mesh),
    (30.0 + w_mesh, 30.0 + w_mesh),
    (30.0 - w_mesh, 30.0 + w_mesh),
    (30.0 - w_mesh, w_mesh),
    (w_mesh, w_mesh),
    (w_mesh, 30.0 + w_mesh),
    (-w_mesh, 30.0 + w_mesh),
]
u_shape_polygon_mesh = Polygon(u_shape_coords_mesh)

if u_shape_polygon_mesh.is_valid:
    env_tri_mesh = Environment.from_layout(
        kind="TriangularMesh",
        layout_params={
            "boundary_polygon": u_shape_polygon_mesh,
            "point_spacing": 3.0,  # Controls density of mesh vertices
        },
        name="UShape_TriMesh",
    )
    print(f"\n--- TriangularMesh Environment ({env_tri_mesh.name}) ---")
    print(f"Number of bins (triangles): {env_tri_mesh.n_bins}")

    fig_mesh, ax_mesh = plt.subplots(figsize=(6, 6))
    env_tri_mesh.plot(ax=ax_mesh, triangle_kwargs={"fc": "lightgreen", "alpha": 0.6})
    ax_mesh.set_title(f"{env_tri_mesh.name}")
    plt.show()
else:
    print(
        f"U-shaped polygon for TriangularMesh is invalid: {u_shape_polygon_mesh.explain_validity()}. Skipping demo."
    )

# %% [markdown]
# ### 4.4 Other Layouts (`HexagonalLayout`, `MaskedGridLayout`)
# These were shown previously and offer alternative ways to tile space or define active areas.
# `HexagonalLayout` can be useful for open fields to reduce orientation bias of square bins.
# `MaskedGridLayout` is great when you have explicit grid lines and a precise boolean mask of active cells.

# %% [markdown]
# ## Part 5: Working with Multiple Environments 🤝
#
# ### 5.1 `CompositeEnvironment`: Modeling Complex Mazes with Distinct Parts
# Some mazes have sections best modeled differently (e.g., an open central area connected to linear arms).
# Let's build a plus-maze: a square center (`RegularGrid`) and four linear arms (`GraphLayout`).

# %%
print("\n--- Advanced Composite Environment (Plus Maze) ---")

# Central Area (e.g., 20x20 cm, centered at (0,0))
# Use a simple RegularGrid for the center, defining its extent
center_pos_data = np.array(
    [  # Some samples to define the center active area
        [-8, -8],
        [8, -8],
        [8, 8],
        [-8, 8],
        [0, 0],
    ]
)
env_center_plusmaze = Environment.from_samples(
    data_samples=center_pos_data,
    bin_size=4.0,  # Coarser bins for center
    name="PlusMaze_CenterArea",
)

# Arms (each 30cm long, 2cm wide conceptually for plotting, binned at 2cm)
arm_len = 30.0
arm_bs = 2.0

# Define nodes relative to the center area's presumed edges
# Center area estimated extent based on samples and binning: roughly -10 to 10.
# So, arms connect around +/-10.
env_arm_N = Environment.from_graph(
    make_track_graph([(0, 10), (0, 10 + arm_len)], [(0, 1)]),
    [(0, 1)],
    0,
    arm_bs,
    "ArmN",
)
env_arm_E = Environment.from_graph(
    make_track_graph([(10, 0), (10 + arm_len, 0)], [(0, 1)]),
    [(0, 1)],
    0,
    arm_bs,
    "ArmE",
)
env_arm_S = Environment.from_graph(
    make_track_graph([(0, -10), (0, -10 - arm_len)], [(0, 1)]),
    [(0, 1)],
    0,
    arm_bs,
    "ArmS",
)
env_arm_W = Environment.from_graph(
    make_track_graph([(-10, 0), (-10 - arm_len, 0)], [(0, 1)]),
    [(0, 1)],
    0,
    arm_bs,
    "ArmW",
)

plus_maze_parts = [env_center_plusmaze, env_arm_N, env_arm_E, env_arm_S, env_arm_W]
plus_maze_parts_valid = [env for env in plus_maze_parts if env.n_bins > 0]

if len(plus_maze_parts_valid) > 1:
    env_plus_maze = CompositeEnvironment(
        plus_maze_parts_valid,
        auto_bridge=True,
        max_mnn_distance=5.0,  # Allow bridges if ends are within 5cm
    )
    print(f"Plus Maze Composite Environment: {env_plus_maze.n_bins} total bins.")

    fig_plus, ax_plus = plt.subplots(figsize=(9, 9))
    # Customize plotting for each sub-environment
    plot_kwargs_list = [
        {"cmap": "Greys", "alpha": 0.4, "node_size": 0},  # Center
        {
            "bin_node_kwargs": {"color": "blue", "node_size": 15},
            "edge_kwargs": {"alpha": 0.6},
        },  # N
        {
            "bin_node_kwargs": {"color": "green", "node_size": 15},
            "edge_kwargs": {"alpha": 0.6},
        },  # E
        {
            "bin_node_kwargs": {"color": "purple", "node_size": 15},
            "edge_kwargs": {"alpha": 0.6},
        },  # S
        {
            "bin_node_kwargs": {"color": "orange", "node_size": 15},
            "edge_kwargs": {"alpha": 0.6},
        },  # W
    ]
    env_plus_maze.plot(
        ax=ax_plus,
        show_sub_env_labels=True,
        sub_env_plot_kwargs=plot_kwargs_list[: len(plus_maze_parts_valid)],
        bridge_edge_kwargs={"color": "red", "linewidth": 1.5, "linestyle": "--"},
    )
    ax_plus.set_title("Composite Plus Maze (Center + 4 Arms)")
    ax_plus.axis("equal")
    plt.show()

    # Test distance across bridges
    # Point in North arm end vs. East arm end
    pt_N_end = np.array(
        [[0, 10 + arm_len - arm_bs / 2]]
    )  # Slightly inside last bin of N arm
    pt_E_end = np.array(
        [[10 + arm_len - arm_bs / 2, 0]]
    )  # Slightly inside last bin of E arm
    dist_N_E_plusmaze = env_plus_maze.distance_between(pt_N_end, pt_E_end)
    print(
        f"Distance from North arm end to East arm end in Plus Maze: {dist_N_E_plusmaze:.2f} cm"
    )
else:
    print(
        "Skipping composite plus maze demo as not enough valid sub-environments were created."
    )


# %% [markdown]
# ### 5.2 `Alignment`: Comparing Data Across Sessions with Environmental Changes
# Often, the maze setup might slightly change between recording sessions (shift, rotation, minor scaling).
# `map_probabilities_to_nearest_target_bin` helps compare data (e.g., place fields) across such sessions.

# %%
print("\n--- Environment Alignment Demo ---")
# Use our main grid environment as the 'source' (Session 1)
if grid_env_main is not None and grid_env_main.n_bins > 0:
    env_session1 = grid_env_main

    # Simulate data for Session 2: shift, rotate, and slightly scale Session 1's positions
    s1_to_s2_shift = np.array([5.0, 3.0])  # Shift by (5cm, 3cm)
    s1_to_s2_angle_deg = -7.0  # Rotate clockwise by 7 degrees
    s1_to_s2_scale = 1.02  # Slightly larger

    s1_to_s2_rot_matrix = get_2d_rotation_matrix(s1_to_s2_angle_deg)

    # Need original position_data_cm that defined grid_env_main (env_session1)
    if "position_data_cm" in locals():
        positions_session2 = apply_similarity_transform(
            position_data_cm, s1_to_s2_rot_matrix, s1_to_s2_scale, s1_to_s2_shift
        )
        env_session2 = Environment.from_samples(
            data_samples=positions_session2,
            bin_size=grid_env_main.layout_parameters.get(
                "bin_size", 3.0
            ),  # Use same bin_size
            name="Maze_Session2",
        )

        if env_session1.n_bins > 0 and env_session2.n_bins > 0:
            # Create a hypothetical "place field" (probability distribution) on Session 1 env
            field_center_s1_idx = env_session1.n_bins // 3  # Arbitrary center
            field_center_s1_coords = env_session1.bin_center_of(field_center_s1_idx)

            sq_dists_s1 = np.sum(
                (env_session1.bin_centers - field_center_s1_coords) ** 2, axis=1
            )
            probs_s1 = np.exp(
                -sq_dists_s1 / (2 * (8.0**2))
            )  # Gaussian field, sigma=8cm
            probs_s1 = probs_s1 / np.sum(probs_s1) if np.sum(probs_s1) > 0 else probs_s1

            # Map this field from Session 1 coordinates to Session 2 environment
            # The transform params are those that take Session 1 points TO Session 2 space
            mapped_probs_s2 = map_probabilities_to_nearest_target_bin(
                source_env=env_session1,
                target_env=env_session2,
                source_probs=probs_s1,
                source_rotation_matrix=s1_to_s2_rot_matrix,
                source_scale_factor=s1_to_s2_scale,
                source_translation_vector=s1_to_s2_shift,
            )
            print(f"Sum of original probabilities (S1): {np.sum(probs_s1):.3f}")
            print(
                f"Sum of mapped probabilities (S2): {np.sum(mapped_probs_s2):.3f}"
            )  # Should be close to S1

            # Plot
            fig_align_sess, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(13, 6))
            common_cbar_args = {
                "cmap": "magma",
                "vmin": 0,
                "vmax": (
                    np.max([probs_s1.max(), mapped_probs_s2.max()])
                    if mapped_probs_s2.size > 0
                    else probs_s1.max()
                ),
            }

            env_session1.plot(ax=ax_s1, show_connectivity=False, alpha=0.05)
            sc1 = ax_s1.scatter(
                env_session1.bin_centers[:, 0],
                env_session1.bin_centers[:, 1],
                c=probs_s1,
                s=40,
                **common_cbar_args,
            )
            plt.colorbar(sc1, ax=ax_s1, label="Prob (Session 1)")
            ax_s1.set_title(f"Place Field on Env Session 1 ({env_session1.name})")
            ax_s1.axis("equal")

            env_session2.plot(ax=ax_s2, show_connectivity=False, alpha=0.05)
            if env_session2.n_bins > 0 and mapped_probs_s2.size == env_session2.n_bins:
                sc2 = ax_s2.scatter(
                    env_session2.bin_centers[:, 0],
                    env_session2.bin_centers[:, 1],
                    c=mapped_probs_s2,
                    s=40,
                    **common_cbar_args,
                )
                plt.colorbar(sc2, ax=ax_s2, label="Mapped Prob (Session 2)")
            ax_s2.set_title(f"Mapped Field on Env Session 2 ({env_session2.name})")
            ax_s2.axis("equal")

            plt.suptitle("Aligning Place Field Across Sessions", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print(
                "Skipping session alignment demo as one of the environments has no bins."
            )
    else:
        print("Skipping session alignment demo: 'position_data_cm' not found.")
else:
    print(
        "Skipping session alignment demo: 'grid_env_main' not available or has no bins."
    )


# %% [markdown]
# ## Part 6: Discovering Available Layouts 🔍
# You can easily list all built-in layout types and their required parameters.

# %%
import inspect  # Needed for pretty printing default values

print("\n--- API Discovery: Available Layouts and Parameters ---")
available_layout_types = list_available_layouts()
print("Available layout kinds (case-insensitive):\n", available_layout_types)

for layout_name in available_layout_types:
    print(f"\nParameters for '{layout_name}':")
    try:
        params = get_layout_parameters(layout_name)
        if not params:
            print(f"  (No specific build parameters listed for {layout_name})")
        for param_name, info in params.items():
            p_type = info.get("annotation", "Any")
            p_default = info.get("default")
            default_str = (
                f"(default: {repr(p_default)})"
                if p_default is not inspect.Parameter.empty
                else "(required)"
            )
            print(f"  - {param_name}: (type: {p_type}, {default_str})")
    except Exception as e:  # Catch any error during introspection
        print(f"  Could not retrieve parameters for {layout_name}: {e}")


# %% [markdown]
# ### Tutorial Complete! 🎉
#
# You've now journeyed through:
# - Calibrating raw pixel data to physical units.
# - Creating `Environment` objects from various data sources (samples, images, graphs, polygons, meshes).
# - Interacting with these environments: mapping points, finding neighbors, calculating distances.
# - Defining meaningful `Regions` for detailed spatial analysis (e.g., reward wells, maze arms).
# - Modeling complex, multi-part mazes with `CompositeEnvironment`.
# - Aligning and mapping probability distributions (like place fields) across different sessions or environmental setups using the `alignment` tools.
# - Discovering the available layout types and their parameters.
#
# This `environment` package provides a powerful and flexible toolkit for representing your experimental spaces,
# which is a cornerstone for robust Bayesian decoding and many other types of spatial data analysis
# in hippocampal research. Happy decoding!
