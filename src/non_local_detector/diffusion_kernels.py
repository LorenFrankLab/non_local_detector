from typing import Literal, Optional

import networkx as nx
import numpy as np
import scipy.sparse.linalg
from numpy.typing import NDArray


def _assign_gaussian_weights_from_distance(
    G: nx.Graph,
    bandwidth_sigma: float,
) -> None:
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
        data["weight"] = float(np.exp(-(d * d) / two_sigma2))


def compute_diffusion_kernels(
    graph: nx.Graph,
    bandwidth_sigma: float,
    bin_sizes: Optional[NDArray] = None,
    mode: Literal["transition", "density"] = "transition",
) -> np.ndarray:
    """
    Computes a diffusion-based kernel for all bins (nodes) of `graph` via
    matrix-exponential of a (possibly volume-corrected) graph-Laplacian.

    Args
    ----
    graph : nx.Graph
        Nodes = bins.  Each edge must have a "distance" attribute (Euclidean length).
    bandwidth_sigma : float
        The Gaussian-bandwidth (σ).  We exponentiate with t = σ^2 / 2.
    bin_sizes : Optional[NDArray], shape (n_bins,)
        If provided, bin_sizes[i] is the physical “area/volume” of node i.
        If not provided, we treat all bins as unit-mass.
    mode : "transition" or "density"
        - "transition":  Return a purely discrete transition-matrix P so that ∑_i P[i,j] = 1.
                         (You do *not* need `bin_sizes` in this mode; if you pass it,
                         it will only be used in the exponent step to form L_vol = M^{-1} L,
                         but the final column-normalization is "sum→1".)
        - "density":     Return a continuous-KDE kernel so that ∑_i [K[i,j] * bin_sizes[i]] = 1.
                         Requires `bin_sizes` ≢ None.  (You exponentiate M^{-1} L, then rescale
                         each column so that its weighted-sum by bin_areas is 1.)

    Returns
    -------
    kernel : jnp.ndarray, shape (n_bins, n_bins)
        If mode="transition":   each column j sums to 1 (∑_i K[i,j] = 1).
        If mode="density":      each column j integrates to 1 over area
                                 (∑_i K[i,j] * bin_sizes[i] = 1).
    """
    n_bins = graph.number_of_nodes()
    # 1) Re-compute edge "weight" = exp( - dist^2/(2σ^2) )
    _assign_gaussian_weights_from_distance(graph, bandwidth_sigma)

    # 2) Build unnormalized Laplacian L = D - W
    L = nx.laplacian_matrix(graph, nodelist=range(n_bins), weight="weight")

    # 3) If bin_sizes is given, form M⁻¹ = diag(1/bin_sizes),
    #    then replace L ← M⁻¹ @ L (so we solve du/dt = - M⁻¹ L u).
    if bin_sizes is not None:

        if bin_sizes.shape != (n_bins,):
            raise ValueError(
                f"bin_sizes must have shape ({n_bins},), but got {bin_sizes.shape}."
            )
        M_inv = np.diag(1.0 / bin_sizes)  # shape = (n_bins, n_bins)
        L = M_inv @ L  # now L = M⁻¹ (D - W)

    # 4) Exponentiate: kernel = exp( - (σ^2 / 2) * L )
    t = bandwidth_sigma**2 / 2.0
    kernel = scipy.sparse.linalg.expm(-t * L)

    # 5) Clip tiny negative noise to zero
    kernel = np.clip(kernel, a_min=0.0, a_max=None)

    # 6) Final normalization:
    #   - If mode="transition":  ∑_i K[i,j] = 1  (pure discrete)
    #   - If mode="density":     ∑_i [K[i,j] * areas[i]] = 1  (continuous KDE)
    if mode == "transition":
        # Just normalize each column so it sums to 1
        mass_out = kernel.sum(axis=0, keepdims=True)  # shape = (1, n_bins)
        # scale = 1 / mass_out[j]  (so that ∑_i K[i,j] = 1)
    elif mode == "density":
        if bin_sizes is None:
            raise ValueError("`mode='density'` requires a non-None `bin_sizes` array.")
        # Compute mass_out[j] = ∑_i [kernel[i,j] * areas[i]]
        # shape = (1, n_bins)
        mass_out = (kernel * bin_sizes[:, None]).sum(axis=0, keepdims=True)
        # scale[j] = 1 / mass_out[j]  (so that ∑_i [K[i,j]*areas[i]] = 1)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'transition' or 'density'.")
    scale = np.where(mass_out == 0.0, 0.0, 1.0 / mass_out)  # (1, n_bins)
    kernel = kernel * scale

    return kernel
