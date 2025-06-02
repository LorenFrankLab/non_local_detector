from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import networkx as nx
import numpy as np


def _assign_gaussian_weights_from_distance(G: nx.Graph, bandwidth_sigma: float) -> None:
    """
    Overwrites each edge's "weight" attribute with
        w_uv = exp( - (distance_uv)^2 / (2 * sigma^2) ).
    Assumes each edge already has "distance" = Euclidean length.
    """
    two_sigma2 = 2.0 * (bandwidth_sigma**2)
    for u, v, data in G.edges(data=True):
        d = data.get("distance", None)
        if d is None:
            raise KeyError(f"Edge ({u},{v}) has no 'distance' attribute.")
        # compute the RBF weight
        data["weight"] = float(np.exp(-(d * d) / two_sigma2))


def compute_diffusion_kernels(
    graph: nx.Graph,
    bandwidth_sigma: float,
) -> jnp.ndarray:
    """
    Computes diffusion kernels for all valid bins using the
    Graph Laplacian method and matrix exponential.

    Parameters
    ----------
    graph : nx.Graph
        The graph representing the track, where nodes are bins and edges are connections.
    bandwidth_sigma : float
        The bandwidth of the Gaussian kernel. This controls the spread of the kernel.

    Returns
    -------
    kernel_matrix : jnp.ndarray, shape (n_bins, n_bins)
        The diffusion kernel matrix for the bins.
        Therepresents the amount of concentration (or probability mass, if u represents probability)
        that has flowed from node j to node i after time t.
    """

    n_bins = graph.number_of_nodes()
    _assign_gaussian_weights_from_distance(graph, bandwidth_sigma)
    laplacian_full = jnp.array(
        nx.laplacian_matrix(graph, nodelist=range(n_bins), weight="weight").toarray(),
        dtype=jnp.float32,
    )

    exponent_coefficient = bandwidth_sigma**2 / 2.0
    kernel_matrix = jax.scipy.linalg.expm(-exponent_coefficient * laplacian_full)

    # Clip to non-negative values
    kernel_matrix = jnp.clip(kernel_matrix, a_min=0.0, a_max=None)

    # Normalize the kernel matrix so the sum of each column is 1.
    col_sums = kernel_matrix.sum(axis=0, keepdims=True)
    kernel_matrix = jnp.where(col_sums == 0.0, 0.0, kernel_matrix / col_sums)

    return kernel_matrix
