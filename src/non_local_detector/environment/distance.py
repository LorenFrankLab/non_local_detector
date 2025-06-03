import networkx as nx
import numpy as np


def euclidean_distance_matrix(centers: np.ndarray) -> np.ndarray:
    """
    Given an (N, n_dims) array of centers, return the (NxN) pairwise
    Euclidean distance matrix.
    """
    from scipy.spatial.distance import pdist, squareform

    if centers.shape[0] == 0:
        return np.empty((0, 0), float)
    if centers.shape[0] == 1:
        return np.zeros((1, 1), float)
    return squareform(pdist(centers, metric="euclidean"))


def geodesic_distance_matrix(
    G: nx.Graph, n_states: int, weight: str = "distance"
) -> np.ndarray:
    """
    Return an (n_states x n_states) matrix of shortest-path lengths
    on graph G, using edge attribute "distance" as weight.
    """
    if G.number_of_nodes() == 0:
        return np.empty((0, 0), float)
    dist_matrix = np.full((n_states, n_states), np.inf, dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)
    for src, lengths in nx.shortest_path_length(G, weight=weight):
        for dst, L in lengths.items():
            dist_matrix[src, dst] = L
    return dist_matrix


def geodesic_distance_between_points(
    G: nx.Graph, bin_from: int, bin_to: int, default: float = np.inf
) -> float:
    """
    Return the shortest‐path length between two states (bin_from, bin_to)
    on G, using weight="distance".  If either index is invalid or no path,
    returns `default`.
    """
    try:
        return nx.shortest_path_length(
            G, source=bin_from, target=bin_to, weight="distance"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return default
