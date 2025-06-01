"""
tutorial_environment.py

A step-by-step tutorial demonstrating advanced features of the `environment` package.

Features covered:
---------------
1. RegularGrid, Hexagonal, Graph, ImageMask, MaskedGrid, TriangularMesh environments via `from_layout`.
2. Inspecting and plotting bin centers, connectivity, and attributes.
3. Mapping points to bins (`bin_at`) and converting back (`bin_center_of`, `to_linear`, `linear_to_nd`).
4. Querying `neighbors`, `contains`, and `distance_between`.
5. Working with `boundary_bins`, `active_mask`, `bin_sizes`, and `dimension_ranges`.
6. Region-based operations: add/list/query regions, compute areas, get bins in a region, and plot them.
7. Composite Environments: combining multiple environments.
8. Listing available layouts and inspecting their parameters.

Prerequisites:
-------------
- `non_local_detector` installed and importable.
- `matplotlib` for plotting.
- `scipy` for generating synthetic data.
- `shapely` for polygon-based regions and triangular mesh.
- `track_linearization` for graph-based environments.

Run this script in a Jupyter notebook or as a standalone Python file.

Usage:
-----
    python tutorial_environment.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from shapely.geometry import Polygon
from track_linearization import make_track_graph

from non_local_detector.environment import get_layout_parameters, list_available_layouts
from non_local_detector.environment.composite import CompositeEnvironment
from non_local_detector.environment.environment import Environment

# Import specific layout engine if direct attribute access is desired (e.g., layout.hexagon_width)
from non_local_detector.environment.layout.engines.hexagonal import HexagonalLayout
from non_local_detector.environment.regions import plot_regions

# Consistent styling for raw data points
RAW_DATA_STYLE = {"s": 2, "c": "dimgray", "alpha": 0.6, "label": "Track Points"}

# %% [markdown]
# ## 1. Generate Synthetic “Track” Data
# We simulate observations along a U-shaped track (three sides of a square, open at the top),
# with small Gaussian noise. The path goes (0,30) -> (0,0) -> (30,0) -> (30,30).

# %%
# 1.1 Create a “track” by concatenating three line segments:
x_coords_seg = np.linspace(0, 30, 100)

# Vertical down segment: x=0, y from 30 down to 0
seg1 = np.stack((np.zeros_like(x_coords_seg), x_coords_seg[::-1]), axis=1)

# Horizontal right segment: x from 0 to 30, y=0
seg2 = np.stack((x_coords_seg, np.zeros_like(x_coords_seg)), axis=1)

# Vertical up segment: x=30, y from 0 up to 30
seg3 = np.stack((np.ones_like(x_coords_seg) * 30, x_coords_seg), axis=1)

position_data = np.vstack([seg1, seg2, seg3])

# 1.2 Add small Gaussian noise
noise = multivariate_normal(mean=[0, 0], cov=0.1).rvs(size=position_data.shape[0])
position_data += noise

# Plot the noisy track
plt.figure(figsize=(5, 5))
plt.scatter(position_data[:, 0], position_data[:, 1], **RAW_DATA_STYLE)
plt.title("Noisy U-Shaped Track Observations")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.axis("equal")
plt.legend()
plt.show()


# %% [markdown]
# ## 2. Build and Explore a RegularGrid Environment
# We’ll build a 2D grid over the noisy data. `Environment.from_layout("RegularGrid", ...)`
# infers `dimension_ranges` from `data_samples` if not provided.

# %%
# 2.1 Create a RegularGrid environment
grid_env = Environment.from_layout(
    kind="RegularGrid",
    layout_params={
        "data_samples": position_data,
        "bin_size": 2.0,
        "infer_active_bins": True,
    },
    name="TrackGridEnv",
)

# 2.2 Basic properties
print("\n--- RegularGrid Environment ---")
print(f"Name: {grid_env.name}")
print(f"Number of active bins: {grid_env.n_bins}")
print(f"Dimensions (n_dims): {grid_env.n_dims}")
print(f"Dimension ranges: {grid_env.dimension_ranges}")
print(f"Input bin_size parameter: {grid_env.layout_parameters.get('bin_size')}")
if grid_env.n_bins > 0:
    print(f"Calculated size of first 5 active bins (area): {grid_env.bin_sizes[:5]}")

# 2.3 Plot the grid’s active bin areas and centers
plt.figure(figsize=(5, 5))
ax_grid = grid_env.plot(show_connectivity=False)  # plot shows pcolormesh for grid
ax_grid.scatter(position_data[:, 0], position_data[:, 1], **RAW_DATA_STYLE, zorder=2)
ax_grid.legend()
ax_grid.set_title(f"{grid_env.name}: Active Bins & Track Points")
plt.show()

# 2.4 Boundary bins
if grid_env.n_bins > 0:
    print(f"Boundary bin indices (first 10): {grid_env.boundary_bins[:10]}")


# %% [markdown]
# ### 2.5 Inspect Bin and Edge Attributes

# %%
if grid_env.n_bins > 0:
    print("\nFirst 5 rows of bin_attributes:")
    print(grid_env.bin_attributes.head())
    if grid_env.connectivity.number_of_edges() > 0:
        print("\nFirst 5 rows of edge_attributes:")
        print(grid_env.edge_attributes.head())
    else:
        print("\nGrid environment has no edges in its connectivity graph.")
else:
    print("\nGrid environment has no active bins; skipping attribute inspection.")

# %% [markdown]
# ### 2.6 Querying the Grid

# %%
if grid_env.n_bins > 0:
    sample_pt_grid = position_data[0].reshape(1, -1)
    bin_idx_grid = grid_env.bin_at(sample_pt_grid)[0]
    if bin_idx_grid != -1:
        print(
            f"\nPoint {sample_pt_grid[0]} maps to bin index {bin_idx_grid}, "
            f"center {grid_env.bin_center_of(bin_idx_grid)}"
        )
        print(f"Neighbors of bin {bin_idx_grid}: {grid_env.neighbors(bin_idx_grid)}")
    else:
        print(
            f"\nPoint {sample_pt_grid[0]} does not map to any active bin in grid_env."
        )

    far_point = np.array([[-999.0, -999.0]])
    print(f"grid_env.contains(sample_pt_grid): {grid_env.contains(sample_pt_grid)[0]}")
    print(f"grid_env.contains(far_point): {grid_env.contains(far_point)[0]}")
else:
    print("\nGrid environment has no active bins; skipping querying demo.")

# %% [markdown]
# ### 2.7 Shortest-Path Distances on the Grid

# %%
if grid_env.n_bins > 0 and position_data.shape[0] > 200:
    ptA_grid = position_data[10].reshape(1, -1)
    ptB_grid = position_data[200].reshape(1, -1)
    idxA_grid = grid_env.bin_at(ptA_grid)[0]
    idxB_grid = grid_env.bin_at(ptB_grid)[0]

    if idxA_grid != -1 and idxB_grid != -1:
        dist_AB_grid = grid_env.distance_between(ptA_grid, ptB_grid)
        print(f"\nBin index of ptA {ptA_grid[0]} -> {idxA_grid}")
        print(f"Bin index of ptB {ptB_grid[0]} -> {idxB_grid}")
        print(
            f"Graph-shortest-path distance between bins {idxA_grid} and {idxB_grid}: {dist_AB_grid:.3f}"
        )
    else:
        print(
            f"\nCannot compute distance in grid_env; one or both points are outside active bins."
        )
else:
    print("\nSkipping grid shortest-path demo due to insufficient bins or data points.")


# %% [markdown]
# ## 3. Build and Explore a Hexagonal Environment

# %%
hex_env = Environment.from_layout(
    kind="Hexagonal",
    layout_params={
        "data_samples": position_data,
        "hexagon_width": 3.0,
        "infer_active_bins": True,
        "bin_count_threshold": 1,
    },
    name="TrackHexEnv",
)

print("\n--- Hexagonal Environment ---")
print(f"Name: {hex_env.name}")
print(f"Number of active bins: {hex_env.n_bins}")
if isinstance(hex_env.layout, HexagonalLayout):  # Requires import
    print(f"Hexagon width (from layout object): {hex_env.layout.hexagon_width}")
print(
    f"Hexagon width (from layout_params): {hex_env.layout_parameters.get('hexagon_width')}"
)
if hex_env.n_bins > 0:
    print(
        f"Active mask shape (on full conceptual hex grid): {hex_env.active_mask.shape}"
    )
    print(f"Calculated size of first 5 active bins (area): {hex_env.bin_sizes[:5]}")


plt.figure(figsize=(5, 5))
ax_hex = hex_env.plot(show_connectivity=False, show_centroids=True)
ax_hex.scatter(position_data[:, 0], position_data[:, 1], **RAW_DATA_STYLE, zorder=2)
ax_hex.legend()
ax_hex.set_title(f"{hex_env.name}: Active Bins & Track Points")
plt.show()

if hex_env.n_bins > 0:
    print(
        f"grid_shape (approximate layout dims of full hex grid): {hex_env.grid_shape}"
    )


# %% [markdown]
# ### 3.1 Hexagonal Bin Attributes and Neighbors

# %%
if hex_env.n_bins > 0:
    print("\nFirst 5 hex bin attributes:\n", hex_env.bin_attributes.head())
    if hex_env.connectivity.number_of_edges() > 0:
        print("\nFirst 5 hex edge attributes:\n", hex_env.edge_attributes.head())
    else:
        print("\nHexagonal environment has no edges in its connectivity graph.")

    if position_data.shape[0] > 50:
        pt_h_hex = position_data[50].reshape(1, -1)
        hex_idx_hex = hex_env.bin_at(pt_h_hex)[0]
        if hex_idx_hex != -1:
            print(
                f"\nPoint {pt_h_hex[0]} -> hex bin {hex_idx_hex}, "
                f"center {hex_env.bin_center_of(hex_idx_hex)}"
            )
            print(
                f"Neighbors of hex bin {hex_idx_hex}: {hex_env.neighbors(hex_idx_hex)}"
            )
        print(f"hex_env.contains(pt_h_hex): {hex_env.contains(pt_h_hex)[0]}")
else:
    print("\nHexagonal environment has no active bins; skipping attribute/query demo.")

# %% [markdown]
# ## 4. Build and Explore a Graph Environment
# We define a square track graph.

# %%
node_positions_graph = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
track_edges_graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
track_graph_def = make_track_graph(node_positions_graph, track_edges_graph)

graph_env = Environment.from_layout(
    kind="Graph",
    layout_params={
        "graph_definition": track_graph_def,
        "edge_order": track_edges_graph,
        "edge_spacing": 0.0,  # Gap between consecutive edges from edge_order
        "bin_size": 1.0,  # Length of each bin along linearized track segments
    },
    name="SquareGraphEnv",
)

print("\n--- Graph Environment (1D-like) ---")
print(f"Name: {graph_env.name}")
print(f"Number of active bins: {graph_env.n_bins}")
print(f"Is 1D flag: {graph_env.is_1d}")
print(f"Dimension ranges (of N-D embedding): {graph_env.dimension_ranges}")
if graph_env.n_bins > 0:
    print(f"Length of first 5 active bins (1D): {graph_env.bin_sizes[:5]}")


plt.figure(figsize=(6, 6))
# For GraphLayout.plot, pass node_size via bin_node_kwargs for active bins
ax_graph = graph_env.plot(
    show_connectivity=True, bin_node_kwargs={"node_size": 30, "color": "black"}
)
ax_graph.scatter(position_data[:, 0], position_data[:, 1], **RAW_DATA_STYLE, zorder=-1)
ax_graph.legend()
ax_graph.set_title(f"{graph_env.name}: Bins & Original Track Points")
plt.show()

if graph_env.is_1d:
    plt.figure(figsize=(8, 2))
    ax_graph_1d = graph_env.plot_1D()
    ax_graph_1d.set_title(f"{graph_env.name}: 1D Linearized Layout")
    plt.show()

# %% [markdown]
# ### 4.1 Graph bin-level queries

# %%
if graph_env.n_bins > 0:
    pt_near_graph = np.array([[29.5, 0.5]])  # Near (30,0) node
    graph_idx_node = graph_env.bin_at(pt_near_graph)[0]

    if graph_idx_node != -1:
        print(
            f"\npt_near_graph {pt_near_graph[0]} -> graph bin {graph_idx_node}, "
            f"N-D center {graph_env.bin_center_of(graph_idx_node)}"
        )
        if graph_env.is_1d:
            lin_coord = graph_env.to_linear(pt_near_graph)[0]
            print(
                f"N-D point {pt_near_graph[0]} maps to 1D coordinate ~ {lin_coord:.2f}"
            )

            reprojected_nd = graph_env.linear_to_nd(np.array([lin_coord]))[0]
            print(f"1D coordinate {lin_coord:.2f} -> N-D point {reprojected_nd}")
            print(
                f"linear_to_nd(0) maps to N-D point: {graph_env.linear_to_nd(np.array([0.0]))[0]}"
            )
    else:
        print(
            f"\npt_near_graph {pt_near_graph[0]} is outside the graph environment bins."
        )


# %% [markdown]
# ### 4.2 Graph Neighbors, Connectivity, and Shortest-Path

# %%
if graph_env.n_bins > 10:
    u, v = 0, 10
    print(f"\nNeighbors of graph bin {u}: {graph_env.neighbors(u)}")
    print(f"Neighbors of graph bin {v}: {graph_env.neighbors(v)}")
    spath = graph_env.shortest_path(u, v)
    print(f"Shortest path from graph bin {u} to bin {v}: {spath}")

if position_data.shape[0] > 150:
    ptA_graph_dist = position_data[0].reshape(1, -1)
    ptB_graph_dist = position_data[150].reshape(1, -1)
    dist_on_graph = graph_env.distance_between(ptA_graph_dist, ptB_graph_dist)
    print(f"distance_between(ptA, ptB) on graph environment = {dist_on_graph:.3f}")


# %% [markdown]
# ## 5. ImageMask Environment

# %%
image_mask_shape = (20, 20)
image_mask_data = np.zeros(image_mask_shape, dtype=bool)
image_mask_data[5:15, 5:15] = True

image_env = Environment.from_layout(
    kind="ImageMask",
    layout_params={
        "image_mask": image_mask_data,
        "bin_size": 1.5,
    },
    name="SimpleImageEnv",
)

print("\n--- ImageMask Environment ---")
print(f"Name: {image_env.name}")
print(f"Number of active bins: {image_env.n_bins}")
print(f"Dimension ranges: {image_env.dimension_ranges}")

plt.figure(figsize=(5, 5))
ax_img = image_env.plot()
ax_img.set_title(f"{image_env.name}: Active Bins from Image")
plt.show()

# %% [markdown]
# ## 6. MaskedGrid Environment

# %%
grid_edges_for_masked = (
    np.array([0.0, 10.0, 20.0, 30.0]),
    np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
)
active_mask_for_masked = np.array(
    [
        [True, True, False, False],
        [True, True, True, False],
        [False, True, True, True],
    ],
    dtype=bool,
)

masked_grid_env = Environment.from_layout(
    kind="MaskedGrid",
    layout_params={
        "grid_edges": grid_edges_for_masked,
        "active_mask": active_mask_for_masked,
    },
    name="ExplicitMaskEnv",
)

print("\n--- MaskedGrid Environment ---")
print(f"Name: {masked_grid_env.name}")
print(f"Number of active bins: {masked_grid_env.n_bins}")
print(f"Grid shape (from edges): {masked_grid_env.grid_shape}")

plt.figure(figsize=(5, 5))
ax_masked = masked_grid_env.plot()
ax_masked.set_title(f"{masked_grid_env.name}: Explicitly Masked Grid")
plt.show()


# %% [markdown]
# ## 7. Composite Environment

# %%
sub_envs_for_composite = []
if grid_env is not None and grid_env.n_bins > 0:
    sub_envs_for_composite.append(grid_env)
if hex_env is not None and hex_env.n_bins > 0:
    sub_envs_for_composite.append(hex_env)
if graph_env is not None and graph_env.n_bins > 0:
    sub_envs_for_composite.append(graph_env)

if len(sub_envs_for_composite) >= 2:
    comp_env = CompositeEnvironment(
        sub_envs_for_composite, auto_bridge=True, max_mnn_distance=10.0
    )  # Increased distance

    print("\n--- Composite Environment ---")
    print(f"Total bins: {comp_env.n_bins}")

    current_offset = 0
    for i, env_item in enumerate(sub_envs_for_composite):
        print(
            f"Sub-env {i} ({env_item.name}): "
            f"bins [{current_offset}..{current_offset + env_item.n_bins - 1}]"
        )
        current_offset += env_item.n_bins

    if position_data.shape[0] > 50:
        pt_from_grid_data = position_data[5].reshape(1, -1)
        pt_from_hex_data = position_data[50].reshape(1, -1)
    else:  # Fallback if position_data is small
        pt_from_grid_data = np.array([[2.0, 28.0]])
        pt_from_hex_data = np.array([[5.0, 5.0]])

    pt_for_graph_comp = np.array([[1.0, 1.0]])

    idx_g_comp = comp_env.bin_at(pt_from_grid_data)[0]
    idx_h_comp = comp_env.bin_at(pt_from_hex_data)[0]
    idx_gr_comp = comp_env.bin_at(pt_for_graph_comp)[0]

    print("\nComposite bin_at results:")
    print(
        f"Point near grid data ({pt_from_grid_data[0]}) -> composite bin {idx_g_comp}"
    )
    print(f"Point near hex data ({pt_from_hex_data[0]}) -> composite bin {idx_h_comp}")
    print(
        f"Point near graph start ({pt_for_graph_comp[0]}) -> composite bin {idx_gr_comp}"
    )

    if idx_g_comp != -1 and idx_h_comp != -1:
        d_g_h_comp = comp_env.distance_between(pt_from_grid_data, pt_from_hex_data)
        print(f"\ndistance(grid_pt -> hex_pt) in composite = {d_g_h_comp:.3f}")
    if idx_g_comp != -1 and idx_gr_comp != -1:
        d_g_gr_comp = comp_env.distance_between(pt_from_grid_data, pt_for_graph_comp)
        print(f"distance(grid_pt -> graph_pt) in composite = {d_g_gr_comp:.3f}")

    plt.figure(figsize=(7, 7))
    ax_comp = comp_env.plot(
        show_sub_env_labels=True,
        sub_env_plot_kwargs=[
            {"node_size": 5, "cmap": "viridis_r", "alpha": 0.6},
            {"node_size": 5, "hexagon_kwargs": {"alpha": 0.3, "facecolor": "skyblue"}},
            {
                "node_size": 20,
                "bin_node_kwargs": {"node_size": 10, "color": "darkred"},
                "edge_kwargs": {"linewidth": 0.3, "alpha": 0.4},
            },
        ][
            : len(sub_envs_for_composite)
        ],  # Slice kwargs list to match actual sub_envs
        bridge_edge_kwargs={"color": "fuchsia", "linewidth": 1.0, "alpha": 0.6},
    )
    ax_comp.scatter(
        position_data[:, 0], position_data[:, 1], **RAW_DATA_STYLE, zorder=-1
    )
    if ax_comp.legend_ is None:  # Add legend if not already present
        handles, labels = ax_comp.get_legend_handles_labels()
        if handles:  # Only add legend if there are items to show
            ax_comp.legend(handles, labels)
    plt.title("Composite Environment Plot")
    plt.show()
else:
    print(
        "\nSkipping Composite Environment demo: not enough non-empty sub-environments."
    )


# %% [markdown]
# ## 8. Regions: Defining and Querying Spatial Regions

# %%
if hex_env is not None and hex_env.n_bins > 0:
    point_for_region_hex = position_data[0].reshape(1, -1)
    hex_env.regions.add(
        name="point_region_hex", point=point_for_region_hex[0]
    )  # Pass 1D point
    print(f"\nAdded 'point_region_hex' at {point_for_region_hex[0]}")
    print(f"Region 'point_region_hex' area: {hex_env.regions.area('point_region_hex')}")
    print(f"Region object: {hex_env.regions['point_region_hex']}")
    print(f"List of region names: {hex_env.regions.list_names()}")

    fig_hr, ax_region_pt_hex = plt.subplots(figsize=(5, 5))
    hex_env.plot(ax=ax_region_pt_hex)
    plot_regions(ax=ax_region_pt_hex, regions=hex_env.regions)
    ax_region_pt_hex.set_title("Point-Based Region on Hex Environment")
    plt.show()

    rect_poly_hex = Polygon([(-5, -5), (10, -5), (10, 10), (-5, 10)])
    hex_env.regions.add(name="rectangle_region_hex", polygon=rect_poly_hex)
    print(
        f"\nAdded 'rectangle_region_hex'; area: {hex_env.regions.area('rectangle_region_hex')}"
    )

    small_poly_hex = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    hex_env.regions.add(name="small_square_region_hex", polygon=small_poly_hex)
    print(
        f"Added 'small_square_region_hex'; area: {hex_env.regions.area('small_square_region_hex')}"
    )

    bins_rect_hex = hex_env.bins_in_region("rectangle_region_hex")
    bins_small_hex = hex_env.bins_in_region("small_square_region_hex")
    print(f"\nNumber of hex bins in rectangle_region_hex: {len(bins_rect_hex)}")
    print(f"Number of hex bins in small_square_region_hex: {len(bins_small_hex)}")

    fig_pr, ax_region_poly_hex = plt.subplots(figsize=(6, 6))
    hex_env.plot(
        ax=ax_region_poly_hex, hexagon_kwargs={"alpha": 0.1, "facecolor": "lightgray"}
    )
    plot_regions(
        ax=ax_region_poly_hex,
        regions=hex_env.regions,
        polygon_kwargs={
            "linewidth": 1.5,
            "edgecolor": "blue",
            "alpha": 0.6,
            "facecolor": "none",
        },
    )

    if bins_rect_hex.size > 0:
        ax_region_poly_hex.scatter(
            hex_env.bin_centers[bins_rect_hex, 0],
            hex_env.bin_centers[bins_rect_hex, 1],
            s=30,
            facecolors="none",
            edgecolors="dodgerblue",
            linewidth=1.5,
            label="bins in rectangle",
        )
    if bins_small_hex.size > 0:
        ax_region_poly_hex.scatter(
            hex_env.bin_centers[bins_small_hex, 0],
            hex_env.bin_centers[bins_small_hex, 1],
            s=30,
            facecolors="none",
            edgecolors="deeppink",
            linewidth=1.5,
            label="bins in small square",
        )
    if ax_region_poly_hex.get_legend_handles_labels()[0]:  # Check if legend items exist
        ax_region_poly_hex.legend()
    ax_region_poly_hex.set_title("Polygonal Regions and Their Hex Bins")
    plt.show()
else:
    print("\nSkipping Region demo as hex_env is not available or has no active bins.")


# %% [markdown]
# ## 9. TriangularMesh Environment

# %%
# 9.1 Define a U-shaped polygon
w_mesh = 5.0
u_shape_coords_mesh = [
    (-w_mesh, 30.0 + w_mesh),  # P1: Outer Top-Left
    (-w_mesh, -w_mesh),  # P2: Outer Bottom-Left
    (30.0 + w_mesh, -w_mesh),  # P3: Outer Bottom-Right
    (30.0 + w_mesh, 30.0 + w_mesh),  # P4: Outer Top-Right
    (30.0 - w_mesh, 30.0 + w_mesh),  # P5: Inner Top-Right of opening (going inwards)
    (30.0 - w_mesh, w_mesh),  # P6: Inner Bottom-Right of U bend
    (w_mesh, w_mesh),  # P7: Inner Bottom-Left of U bend
    (w_mesh, 30.0 + w_mesh),  # P8: Inner Top-Left of opening
    (-w_mesh, 30.0 + w_mesh),  # back to P1 to close
]
u_shape_polygon_mesh = Polygon(u_shape_coords_mesh)

if not u_shape_polygon_mesh.is_valid:
    print(
        f"Warning: U-shaped polygon for mesh is not valid: {u_shape_polygon_mesh.explain_validity()}. Skipping TriangularMesh demo."
    )
    mesh_env = None  # Ensure mesh_env is defined for later checks
else:
    mesh_env = Environment.from_layout(
        kind="TriangularMesh",
        layout_params={
            "boundary_polygon": u_shape_polygon_mesh,
            "point_spacing": 2.5,
        },
        name="USpaceTriMesh",
    )

    print("\n--- TriangularMesh Environment ---")
    print(f"Name: {mesh_env.name}")
    print(f"Number of bins (triangles): {mesh_env.n_bins}")
    print(f"Dimension ranges: {mesh_env.dimension_ranges}")
    if mesh_env.n_bins > 0:
        print(f"Area of first 5 active triangular bins: {mesh_env.bin_sizes[:5]}")

    plt.figure(figsize=(7, 7))
    ax_mesh_plot = mesh_env.plot(
        show_connectivity=True,
        show_centroids=True,
        triangle_kwargs={"alpha": 0.4, "facecolor": "lightseagreen"},
        centroid_kwargs={"s": 8, "color": "black"},
        connectivity_kwargs={"linewidth": 0.4, "color": "darkslategray"},
    )
    x_ext_mesh, y_ext_mesh = u_shape_polygon_mesh.exterior.xy
    ax_mesh_plot.plot(
        x_ext_mesh,
        y_ext_mesh,
        color="darkblue",
        linewidth=1.5,
        label="U-shaped boundary",
    )
    ax_mesh_plot.legend()
    ax_mesh_plot.set_title(f"{mesh_env.name} inside U-shaped Corridor")
    plt.show()

    if mesh_env.n_bins > 10:
        sample_bin_mesh = min(10, mesh_env.n_bins - 1)  # Ensure valid index
        print(
            f"\nNeighbors of mesh bin {sample_bin_mesh}: {mesh_env.neighbors(sample_bin_mesh)}"
        )

        pt_m1_mesh = np.array([[2.0, 28.0]])
        pt_m2_mesh = np.array([[28.0, 2.0]])

        idx_m1_mesh = mesh_env.bin_at(pt_m1_mesh)[0]
        idx_m2_mesh = mesh_env.bin_at(pt_m2_mesh)[0]

        if idx_m1_mesh != -1 and idx_m2_mesh != -1:
            dist_on_mesh = mesh_env.distance_between(pt_m1_mesh, pt_m2_mesh)
            print(
                f"Mesh shortest-path distance from {pt_m1_mesh[0]} (bin {idx_m1_mesh}) "
                f"to {pt_m2_mesh[0]} (bin {idx_m2_mesh}): {dist_on_mesh:.3f}"
            )
        else:
            print("One or both points for mesh distance are outside active mesh area.")
    elif mesh_env.n_bins > 0:
        print(
            "\nMesh environment has too few bins for neighbor/distance demo with fixed indices."
        )
    else:
        print("\nMesh environment has no active bins.")


# %% [markdown]
# ## 10. Listing Available Layouts and Their Parameters

# %%
print("\n--- API Discovery ---")
available_layouts = list_available_layouts()
print("Available layout kinds:\n", available_layouts)

for layout_kind_name in available_layouts:
    print(f"\nParameters for '{layout_kind_name}':")
    try:
        params_info = get_layout_parameters(layout_kind_name)
        if not params_info:
            print(
                "  (No parameters listed or layout does not require specific build params)"
            )
        for name, params in params_info.items():
            param_type = params.get("annotation", "Any")
            param_default = params.get("default", "(no default / required)")
            print(f"  - {name}: (type: {param_type}, default: {param_default})")
    except ValueError as e:
        print(f"  Error getting parameters: {e}")


# %% [markdown]
# ### Tutorial Complete
# You have now seen how to:
# - Build multiple environment types (RegularGrid, Hexagonal, Graph, ImageMask, MaskedGrid, TriangularMesh).
# - Inspect bin centers, connectivity, and per-bin/per-edge attributes.
# - Map N-D points to bins (`bin_at`), and convert between N-D and 1D (“linear”) for graph environments.
# - Use `distance_between` to compute shortest-path distances.
# - Create and query spatial Regions (point-based and polygonal).
# - Combine sub-environments via `CompositeEnvironment`.
# - List all available layouts and examine their argument structure.
#
# Explore further by adjusting parameters, creating custom Regions, and using these environments in analyses.
