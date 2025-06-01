"""
tutorial_environment.py

A step-by-step tutorial demonstrating advanced features of the `environment` package.

Features covered:
---------------
1. RegularGrid, Hexagonal, Graph, TriangularMesh environments via `from_layout`.
2. Inspecting and plotting bin centers, connectivity, and attributes.
3. Mapping points to bins (`bin_at`) and converting back (`bin_center_of`, `to_linear`, `linear_to_nd`).
4. Querying `neighbors`, `contains`, and `distance_between`.
5. Working with `boundary_bins`, `active_mask`, `bin_sizes`, and `dimension_ranges`.
6. Region-based operations: add/list/query regions, compute areas, get bins in a region, and plot them.
7. Listing available layouts and inspecting their parameters.

Prerequisites:
-------------
- `non_local_detector` installed and importable.
- `matplotlib` for plotting.
- `scipy` for generating synthetic data (optional).
- `shapely` for polygon-based regions and triangular mesh.

Run this script in a Jupyter notebook or as a standalone Python file.

Usage:
-----
    python tutorial_non_local_detector.py
"""

import matplotlib.pyplot as plt

# %%
import numpy as np
from scipy.stats import multivariate_normal
from shapely.geometry import Polygon
from track_linearization import make_track_graph

from non_local_detector.environment import get_layout_parameters, list_available_layouts
from non_local_detector.environment.composite import CompositeEnvironment
from non_local_detector.environment.environment import Environment
from non_local_detector.environment.regions import Region, plot_regions

# %% [markdown]
# ## 1. Generate Synthetic “Track” Data
# We simulate observations along a squared track (counterclockwise), with small Gaussian noise.

# %%
# 1.1 Create a “track” by concatenating three line segments:
x = np.linspace(0, 30, 300)

# Vertical up segment: x=0, y from 30→0 (reversed)
seg1 = np.stack((np.zeros_like(x), x[::-1]), axis=1)

# Horizontal right segment: x from 0→30, y=0
seg2 = np.stack((x, np.zeros_like(x)), axis=1)

# Vertical down segment: x=30, y from 0→30
seg3 = np.stack((np.ones_like(x) * 30, x), axis=1)

# Combine segments into one “track”
position = np.vstack([seg1, seg2, seg3])

# 1.2 Add small Gaussian noise to each observed coordinate
position += multivariate_normal(mean=0, cov=0.1).rvs(position.shape)

# Plot the noisy track
plt.figure(figsize=(5, 5))
plt.scatter(position[:, 0], position[:, 1], s=1, c="gray")
plt.title("Noisy Track Observations")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()


# %% [markdown]
# ## 2. Build and Explore a RegularGrid Environment
# We’ll build a 2D grid over the noisy data. By default, `from_layout("RegularGrid", …)` infers sensible “dimension_ranges” and sets bin_size so that 10 bins span each axis if not specified.

# %%
# 2.1 Create a RegularGrid environment from our position samples
grid_env = Environment.from_layout(
    kind="RegularGrid",
    layout_params={
        "data_samples": position,  # points to define active bins
        "bin_size": 2.0,  # each grid cell is 2×2 units
        "infer_active_bins": True,  # only keep bins containing ≥1 sample by default
        # (you could specify "bin_count_threshold": k to require k points per bin)
    },
    name="TrackGridEnv",
)

# 2.2 Basic properties
print("\n--- RegularGrid Environment ---")
print("Name:", grid_env.name)
print("Number of bins:", grid_env.n_bins)
print("Dimensions (n_dims):", grid_env.n_dims)
print("Dimension ranges:", grid_env.dimension_ranges)  # (xmin, xmax), (ymin, ymax)
print("Bin size (uniform or per-dimension):", grid_env.bin_sizes)

# 2.3 Plot the grid’s active bin centers
plt.figure(figsize=(5, 5))
ax = grid_env.plot(show_connectivity=False)
ax.scatter(position[:, 0], position[:, 1], s=3, c="red", label="Track points")
ax.legend()
ax.set_title("RegularGrid Active Bins + Track Points")
plt.show()

# 2.4 Which bins lie on the boundary?
# `boundary_bins` returns indices of bins that touch the overall bounding box.
print("Boundary bin indices (first 10):", grid_env.boundary_bins[:10])


# %% [markdown]
# ### 2.5 Inspect Bin and Edge Attributes
# - `bin_attributes` is a DataFrame (indexed by bin_id) holding any per-bin metadata.
# - `edge_attributes` is a DataFrame listing all edges (source_bin, target_bin, weights, distances).

# %%
print("\nFirst 5 rows of bin_attributes:")
print(grid_env.bin_attributes.head())

print("\nFirst 5 rows of edge_attributes:")
print(grid_env.edge_attributes.head())


# %% [markdown]
# ### 2.6 Querying the Grid
# - `bin_at(points)` returns the bin index (or -1 if outside).
# - `contains(points)` returns a boolean mask of whether each point lies within any bin.
# - `neighbors(bin_idx)` returns a list of neighbor-bin indices.

# %%
# Pick a sample point
sample_pt = position[0].reshape(1, -1)
bin_idx = grid_env.bin_at(sample_pt)[0]
print(
    f"\nPoint {sample_pt[0]} maps to bin index {bin_idx}, center {grid_env.bin_center_of(bin_idx)}"
)

# Check if a far-away point falls outside
far_point = np.array([[999, 999]])
print("grid_env.contains(sample_pt):", grid_env.contains(sample_pt)[0])
print("grid_env.contains(far_point):", grid_env.contains(far_point)[0])

# Neighbors of that bin
print(f"Neighbors of bin {bin_idx}:", grid_env.neighbors(bin_idx))


# %% [markdown]
# ### 2.7 Shortest-Path Distances on the Grid
# Compute the graph-shortest-path distance (using edge weights = Euclidean distances) between two arbitrary points.

# %%
ptA = position[10].reshape(1, -1)
ptB = position[200].reshape(1, -1)
idxA = grid_env.bin_at(ptA)[0]
idxB = grid_env.bin_at(ptB)[0]
dist_AB = grid_env.distance_between(ptA, ptB)
print(f"\nBin index of ptA {ptA[0]} → {idxA}")
print(f"Bin index of ptB {ptB[0]} → {idxB}")
print(f"Graph-shortest-path distance between bins {idxA} and {idxB}: {dist_AB:.3f}")


# %% [markdown]
# ## 3. Build and Explore a Hexagonal Environment
# Next, we build a hex-layout that covers the same data. A hex grid can better “hug” circular or curved shapes.

# %%
hex_env = Environment.from_layout(
    kind="Hexagonal",
    layout_params={
        "data_samples": position,
        "hexagon_width": 3.0,  # distance across flats = 3.0 units
        "infer_active_bins": True,
        "bin_count_threshold": 5,  # at least 5 points per bin
    },
    name="TrackHexEnv",
)

print("\n--- Hexagonal Environment ---")
print("Name:", hex_env.name)
print("Number of bins:", hex_env.n_bins)
print(
    "Hexagon width:",
    hex_env.hexagon_width if hasattr(hex_env, "hexagon_width") else "n/a",
)
print("Active mask shape:", hex_env.active_mask.shape)

# Plot hex bin centers
plt.figure(figsize=(5, 5))
ax = hex_env.plot(show_connectivity=False, show_centroids=True)
ax.scatter(position[:, 0], position[:, 1], s=3, c="red", label="Track points")
ax.legend()
ax.set_title("Hexagonal Active Bins + Track Points")
plt.show()

# Inspect grid shape (rows × columns if rectangular indexing is used)
print("grid_shape (approximate layout dims):", hex_env.grid_shape)


# %% [markdown]
# ### 3.1 Hexagonal Bin Attributes and Neighbors

# %%
print("\nFirst 5 hex bin attributes:\n", hex_env.bin_attributes.head())
print("\nFirst 5 hex edge attributes:\n", hex_env.edge_attributes.head())

# Pick a point and see which hex bin it maps to
pt_h = position[50].reshape(1, -1)
hex_idx = hex_env.bin_at(pt_h)[0]
print(f"\nPoint {pt_h[0]} → hex bin {hex_idx}, center {hex_env.bin_center_of(hex_idx)}")
print(f"Neighbors of hex bin {hex_idx}:", hex_env.neighbors(hex_idx))
print("hex_env.contains(pt_h):", hex_env.contains(pt_h)[0])


# %% [markdown]
# ## 4. Build and Explore a Graph Environment
# We create a small graph manually to illustrate 1D “linear” functionality. We’ll build a square track graph in 2D.

# %%
# Define node coordinates for a square track
node_positions = [
    (0, 0),  # node 0
    (30, 0),  # node 1
    (30, 30),  # node 2
    (0, 30),  # node 3
]
# Define edges (0→1→2→3→0)
track_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
track_graph = make_track_graph(node_positions, track_edges)

# Build the graph environment
graph_env = Environment.from_layout(
    kind="Graph",
    layout_params={
        "graph_definition": track_graph,
        "edge_order": track_edges,  # lay out the edges in this order in 1D
        "edge_spacing": 1.0,  # density of bins inserted along each edge
        "bin_size": 1.0,  # distance between successive “graph” bins
    },
    name="SquareGraphEnv",
)

print("\n--- Graph Environment (1D-like) ---")
print("Name:", graph_env.name)
print("Number of bins:", graph_env.n_bins)
print("Is 1D flag:", graph_env.is_1d)
print("Dimension ranges:", graph_env.dimension_ranges)

# Plot the graph’s bin centers and connectivity
plt.figure(figsize=(5, 5))
ax = graph_env.plot(show_connectivity=True, show_centroids=True)
ax.scatter(position[:, 0], position[:, 1], s=3, c="red", label="Track points")
ax.legend()
ax.set_title("Graph Environment: Bins + Track Points")
plt.show()


# %% [markdown]
# ### 4.1 Graph bin-level queries
# - `bin_centers` returns all 2D coordinates.
# - `to_linear(point)` projects a 2D location onto the 1D “graph” coordinate (cumulative arc-length).
# - `linear_to_nd(value)` converts a 1D coordinate back to a 2D point on the graph.

# %%
# Choose a point near the graph (e.g., on the top-right corner)
pt_near_graph = np.array([[30.5, 0.2]])
graph_idx = graph_env.bin_at(pt_near_graph)[0]
print(
    f"\npt_near_graph {pt_near_graph[0]} → graph bin {graph_idx}, center {graph_env.bin_center_of(graph_idx)}"
)

# Convert 2D → 1D “linear” coordinate
lin_coord = graph_env.to_linear(pt_near_graph)[0]
print(f"2D point {pt_near_graph[0]} maps to 1D coordinate ≈ {lin_coord:.2f}")

# Convert that 1D coordinate back to an ND point
reprojected_nd = graph_env.linear_to_nd(np.array([lin_coord]))[0]
print(f"1D coordinate {lin_coord:.2f} → ND point {reprojected_nd}")

# If you call linear_to_nd(0), it maps to first bin’s center
print("linear_to_nd(0) →", graph_env.linear_to_nd(np.array([0.0]))[0])


# %% [markdown]
# ### 4.2 Graph Neighbors, Connectivity, and Shortest-Path
# - `neighbors(bin_idx)` returns immediate neighbors along the 1D graph.
# - `shortest_path(u, v)` returns the list of bin indices on the shortest path.
# - `distance_between(pt1, pt2)` uses shortest-path with edge-weights = Euclidean distance.

# %%
u, v = 0, 10  # pick two bin indices (0 = first node, 10 ~ somewhere along edge)
print(f"\nNeighbors of bin {u}:", graph_env.neighbors(u))
print(f"Neighbors of bin {v}:", graph_env.neighbors(v))

# Shortest path between two bins
spath = graph_env.shortest_path(u, v)
print(f"Shortest path from bin {u} to bin {v}:", spath)

# Check distance_between two observations
ptA = position[0].reshape(1, -1)
ptB = position[100].reshape(1, -1)
dist_graph = graph_env.distance_between(ptA, ptB)
print(f"distance_between(ptA, ptB) on graph = {dist_graph:.3f}")


# %% [markdown]
# ## 5. Create a Composite Environment
# We combine `[regular grid, hex grid, graph]` in that order.
# - By default, `auto_bridge=True` links nearest-neighbor pairs between sub-environments.
# - In the composite, `bin_at` queries each sub-env in order until a match.

# %%
comp_env = CompositeEnvironment([grid_env, hex_env, graph_env], auto_bridge=True)

print("\n--- Composite Environment ---")
print("Total bins:", comp_env.n_bins)
print(
    "Sub-env splits:",
    f"Grid [0..{grid_env.n_bins - 1}], ",
    f"Hex [{grid_env.n_bins}..{grid_env.n_bins + hex_env.n_bins - 1}], ",
    f"Graph [{grid_env.n_bins + hex_env.n_bins}..{comp_env.n_bins - 1}]",
)

# Test bin_at on representative points
pt_from_grid = position[5].reshape(1, -1)
pt_from_hex = position[50].reshape(1, -1)
pt_from_graph = np.array([[30.1, 0.1]])

idx_g = comp_env.bin_at(pt_from_grid)[0]
idx_h = comp_env.bin_at(pt_from_hex)[0]
idx_gr = comp_env.bin_at(pt_from_graph)[0]

print("\nComposite bin_at results:")
print(f"Point near grid → bin {idx_g}")
print(f"Point near hex  → bin {idx_h}")
print(f"Point near graph→ bin {idx_gr}")

# Composite distance_between
d_g_h = comp_env.distance_between(pt_from_grid, pt_from_hex)
d_g_gr = comp_env.distance_between(pt_from_grid, pt_from_graph)
print(f"\ndistance(grid_pt→hex_pt) = {d_g_h:.3f}")
print(f"distance(grid_pt→graph_pt) = {d_g_gr:.3f}")

# Plot composite bin centers, color-coded by sub-env index
bc = comp_env.bin_centers
colors = np.zeros(comp_env.n_bins, dtype=int)
colors[: grid_env.n_bins] = 0
colors[grid_env.n_bins : grid_env.n_bins + hex_env.n_bins] = 1
colors[grid_env.n_bins + hex_env.n_bins :] = 2

plt.figure(figsize=(6, 6))
plt.scatter(bc[:, 0], bc[:, 1], c=colors, s=20, cmap="tab10")
plt.scatter(position[:, 0], position[:, 1], s=2, c="gray", alpha=0.3)
plt.title("CompositeEnvironment: Blue=Grid, Orange=Hex, Green=Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# %% [markdown]
# ## 6. Regions: Defining and Querying Spatial Regions
# The `Regions` API lets you create named regions (either as points or polygons), compute areas, and find which bins lie inside a region.

# %%
# Add a point-based region (a single center)
hex_env.regions.add(name="point_region", point=position[0])
print("\nAdded 'point_region' at", position[0])
print("Region 'point_region' area (should be 0):", hex_env.regions.area("point_region"))
print("Region object:", hex_env.regions["point_region"])
print("List of region names:", hex_env.regions.list_names())

# Plot point region on top of hex bins
fig, ax = plt.subplots(figsize=(5, 5))
hex_env.plot(ax=ax)
plot_regions(ax=ax, regions=hex_env.regions)
ax.set_title("Point-Based Region on Hex Environment")
plt.show()


# %% [markdown]
# ### 6.1 Polygonal Regions
# Define a rectangular region and a U-shaped corridor, then query bins inside.

# %%
# 6.1.1 Rectangle polygon
rect_poly = Polygon([(-5, -5), (10, -5), (10, 10), (-5, 10)])
hex_env.regions.add(name="rectangle_region", polygon=rect_poly)
print("\nAdded 'rectangle_region'; area:", hex_env.regions.area("rectangle_region"))

# 6.1.2 Another smaller polygon
small_poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
hex_env.regions.add(name="small_region", polygon=small_poly)
print("Added 'small_region'; area:", hex_env.regions.area("small_region"))

# 6.1.3 Query which bins lie in each region
bins_rect = hex_env.bins_in_region("rectangle_region")
bins_small = hex_env.bins_in_region("small_region")
print("\nNumber of hex bins in rectangle_region:", len(bins_rect))
print("Number of hex bins in small_region:", len(bins_small))

# Plot both polygon regions and hex bins inside them
fig, ax = plt.subplots(figsize=(5, 5))
hex_env.plot(ax=ax)
plot_regions(ax=ax, regions=hex_env.regions)
ax.scatter(
    hex_env.bin_centers[bins_rect, 0],
    hex_env.bin_centers[bins_rect, 1],
    s=50,
    facecolors="none",
    edgecolors="blue",
    label="bins in rectangle_region",
)
ax.scatter(
    hex_env.bin_centers[bins_small, 0],
    hex_env.bin_centers[bins_small, 1],
    s=50,
    facecolors="none",
    edgecolors="magenta",
    label="bins in small_region",
)
ax.legend()
ax.set_title("Polygonal Regions and Their Hex Bins")
plt.show()


# %% [markdown]
# ## 7. TriangularMesh Environment
# Finally, we create a triangular mesh over a U-shaped corridor. The mesh covers only bins within the boundary polygon.

# %%
# 7.1 Define the U-shaped corridor polygon
w = 3.0
u_shape_coords = [
    (-w, 30.0 + w),  # P1: Outer Top-Left
    (-w, -w),  # P2: Outer Bottom-Left
    (30.0 + w, -w),  # P3: Outer Bottom-Right
    (30.0 + w, 30.0 + w),  # P4: Outer Top-Right
    # Now cut out the opening at top-center to form a U shape:
    (30.0 - w, 30.0 + w),  # P5: Inner Top-Right of opening
    (30.0 - w, w),  # P6: Inner Bottom-Right of U bend
    (w, w),  # P7: Inner Bottom-Left of U bend
    (w, 30.0 + w),  # P8: Inner Top-Left of opening
    (-w, 30.0 + w),  # back to P1
]
u_shape_polygon = Polygon(u_shape_coords)

# 7.2 Build a TriangularMesh environment inside the U shape
mesh_env = Environment.from_layout(
    kind="TriangularMesh",
    layout_params={
        "boundary_polygon": u_shape_polygon,
        "point_spacing": 2.0,  # approximate spacing between mesh nodes
    },
    name="USpaceTriMesh",
)

print("\n--- TriangularMesh Environment ---")
print("Name:", mesh_env.name)
print("Number of bins (triangles):", mesh_env.n_bins)
print("Dimension ranges:", mesh_env.dimension_ranges)

# 7.3 Plot mesh connectivity and centroids
plt.figure(figsize=(6, 6))
ax = mesh_env.plot(show_connectivity=True, show_centroids=True)
# Overlay the corridor boundary in blue
x_exterior, y_exterior = u_shape_polygon.exterior.xy
ax.plot(x_exterior, y_exterior, color="blue", linewidth=2, label="U-shaped boundary")
ax.legend()
ax.set_title("Triangular Mesh inside U-shaped Corridor")
plt.show()

# 7.4 Query a bin and its neighbors
sample_bin = 10
print(f"\nNeighbors of mesh bin {sample_bin}:", mesh_env.neighbors(sample_bin))

# 7.5 Compute distance_between two mesh-based points
# Pick two interior points
pt_m1 = np.array([[5.0, 5.0]])
pt_m2 = np.array([[25.0, 5.0]])
dist_mesh = mesh_env.distance_between(pt_m1, pt_m2)
print(f"Mesh shortest-path distance from {pt_m1[0]} to {pt_m2[0]}: {dist_mesh:.3f}")

# %% [markdown]
# ## 8. Listing Available Layouts and Their Parameters
# We can see all supported layout kinds and inspect their required/optional parameters.

# %%
print("\nAvailable layout kinds:\n", list_available_layouts())

print("\nParameters for 'TriangularMesh':\n", get_layout_parameters("TriangularMesh"))

# %% [markdown]
# ### Tutorial Complete
# You have now seen how to:
# - Build multiple environment types (RegularGrid, Hexagonal, Graph, TriangularMesh).
# - Inspect bin centers, connectivity, and per-bin/per-edge attributes.
# - Map 2D points to bins (`bin_at`), and convert between 2D and 1D (“linear”) for graph environments.
# - Use `distance_between` to compute shortest-path distances.
# - Create and query spatial Regions (point-based and polygonal).
# - Combine sub-environments via `CompositeEnvironment`.
# - List all available layouts and examine their argument structure.
#
# Explore further by adjusting parameters (bin_size, hexagon_width, point_spacing, etc.),
# creating custom Regions, and plugging these environments into downstream analyses (e.g., Bayesian decoding).
