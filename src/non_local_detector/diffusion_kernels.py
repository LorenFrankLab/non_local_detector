from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import networkx as nx
import numpy as np

from non_local_detector.environment import add_distance_weight_to_edges


def compute_diffusion_kernels(
    track_graph: nx.Graph,
    interior_mask: np.ndarray,
    bandwidth_sigma: float,
) -> jnp.ndarray:
    """
    Computes diffusion kernels for all valid interior bins using the
    Graph Laplacian method and matrix exponential.

    Parameters
    ----------
    track_graph : nx.Graph
        The graph representing the track, where nodes are bins and edges are connections.
    interior_mask : np.ndarray, shape (n_bins_x, n_bins_y, ...)
        A mask indicating which bins are interior. The size should match the
        number of nodes in the track graph. True values indicate interior bins.
    bandwidth_sigma : float
        The bandwidth of the Gaussian kernel. This controls the spread of the kernel.
    weight : str, optional
        The edge attribute to use as weights for the graph. If None, unweighted edges are used.
        By default None.

    Returns
    -------
    kernel_matrix_interior : jnp.ndarray, shape (n_interior_bins, n_interior_bins)
        The diffusion kernel matrix for the interior bins.
        Therepresents the amount of concentration (or probability mass, if u represents probability)
        that has flowed from node j to node i after time t.
    """
    if track_graph is None:
        raise ValueError("track_graph is required.")

    if interior_mask is None:
        interior_mask = np.ones_like(track_graph.nodes(), dtype=bool)

    if bandwidth_sigma <= 0:
        raise ValueError("bandwidth_sigma must be positive.")

    add_distance_weight_to_edges(track_graph)

    n_bins_total = interior_mask.size
    interior_mask_flat = jnp.asarray(interior_mask.ravel())
    interior_bin_indices_flat = jnp.nonzero(interior_mask_flat)[0]
    n_interior_bins = interior_bin_indices_flat.size

    if n_interior_bins == 0:
        return jnp.zeros((0, 0))

    node_list = sorted(list(track_graph.nodes()))
    if not node_list:
        laplacian_full = jnp.zeros((n_bins_total, n_bins_total), dtype=jnp.float32)
    else:
        try:
            # Attempt to compute the full Laplacian matrix
            # This may fail if the graph is disconnected or has isolated nodes.
            # In that case, we will compute the Laplacian for the subgraph
            # induced by the interior nodes.
            # This is a workaround for the case where the graph is disconnected
            # or has isolated nodes.
            laplacian_full = jnp.array(
                nx.laplacian_matrix(
                    track_graph, nodelist=range(n_bins_total), weight="distance_weight"
                ).toarray(),
                dtype=jnp.float32,
            )
        except nx.NetworkXError:
            # If the graph is disconnected or has isolated nodes,
            # compute the Laplacian for the subgraph induced by the interior nodes.
            L_sparse_sub = nx.laplacian_matrix(
                track_graph, nodelist=node_list, weight="distance_weight"
            )
            L_sub = jnp.array(L_sparse_sub.toarray(), dtype=jnp.float32)
            laplacian_full = jnp.zeros((n_bins_total, n_bins_total), dtype=jnp.float32)
            idx_embed = jnp.ix_(jnp.array(node_list), jnp.array(node_list))
            laplacian_full = laplacian_full.at[idx_embed].set(L_sub)

    exponent_coefficient = bandwidth_sigma**2 / 2.0
    full_kernel_matrix = jax.scipy.linalg.expm(-exponent_coefficient * laplacian_full)

    # Apply the interior mask to the kernel matrix and clip to non-negative values
    idx_filter = jnp.ix_(interior_bin_indices_flat, interior_bin_indices_flat)
    kernel_matrix_interior = jnp.clip(
        full_kernel_matrix[idx_filter], a_min=0.0, a_max=None
    )

    # Normalize the kernel matrix so the sum of each column is 1.
    col_sums = kernel_matrix_interior.sum(axis=0, keepdims=True)
    kernel_matrix_interior = jnp.where(
        col_sums > 1e-15, kernel_matrix_interior / col_sums, 0.0
    )
    return kernel_matrix_interior
