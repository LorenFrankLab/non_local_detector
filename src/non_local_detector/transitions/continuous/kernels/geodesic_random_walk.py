from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from scipy.stats import norm

from ....environment import Environment
from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition
from ..utils import _handle_intra_env_kernel_edges, _normalize_row_probability

if TYPE_CHECKING:
    import networkx as nx


def _geodesic_distance_matrix(
    connectivity: nx.Graph, n_states: int, weight: str = "distance"
) -> np.ndarray:
    """
    Return an (n_states x n_states) matrix of shortest-path lengths
    on graph `connectivity`, using edge attribute "distance" as weight.
    """
    if connectivity.number_of_nodes() == 0:
        return np.empty((0, 0), float)
    dist_matrix = np.full((n_states, n_states), np.inf, dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)
    for src, lengths in nx.shortest_path_length(connectivity, weight=weight):
        for dst, L in lengths.items():
            dist_matrix[src, dst] = L
    return dist_matrix


@dataclass
@register_continuous_transition("geodesic_random_walk")
class GeodesicRandomWalkKernel(Kernel):
    """
    Symmetric random-walk inside a **single** environment
    (source and destination environments must be the same).

    Attributes
    ----------
    mean : float
        Mean of the Gaussian movement.
    var : float
        Covariance of the Gaussian movement. If a scalar is provided,
        it is interpreted as the variance for each dimension, resulting
        in a diagonal covariance matrix.

    Raises
    ------
    ValueError
        If `var` is not positive or if `mean` is not a scalar or a 1-element array.
    TypeError
        If `var` is not a float or a 1x1 numpy array, or if `mean` is not a float
        or a 1-element numpy array.
    """

    mean: float  # Mean of the Gaussian movement
    var: float  # Covariance of the Gaussian movement

    def __post_init__(self) -> None:
        if isinstance(self.var, np.ndarray):
            if self.var.shape != (1, 1):
                raise ValueError(
                    "For geodesic kernel, `var` must be scalar or shape (1,1)."
                )
            self.var = float(self.var.item())

        if not isinstance(self.var, (float, int)):
            raise TypeError("`var` must be a float or a 1×1 numpy array.")

        # Normalize mean to a Python float μ
        if isinstance(self.mean, np.ndarray):
            if self.mean.ndim != 1 or self.mean.shape[0] != 1:
                raise ValueError(
                    "For geodesic kernel, `mean` must be a scalar or shape (1,)."
                )
            self.mean = float(self.mean.item())
        elif not isinstance(self.mean, (float, int)):
            raise TypeError("`mean` must be a float or a 1-element numpy array.")

        # Check positivity of var
        if self.var <= 0:
            raise ValueError("`var` must be positive.")

    def block(
        self,
        *,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: Optional[Covariates] = None,  # covariates are ignored
    ) -> Array:
        # Atomic case or cross-environment jump
        transition = _handle_intra_env_kernel_edges(src_env, dst_env)

        if transition is not None:
            return transition

        distance = _geodesic_distance_matrix(src_env.connectivity, src_env.n_bins)
        distance = distance.ravel()
        transition = norm.pdf(distance, loc=self.mean, scale=np.sqrt(self.var))
        transition = transition.reshape((src_env.n_bins, src_env.n_bins))

        return _normalize_row_probability(transition)
