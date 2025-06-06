from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.stats import multivariate_normal

from non_local_detector.environment import Environment

from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition
from ..utils import _handle_intra_env_kernel_edges, _normalize_row_probability


@dataclass
@register_continuous_transition("euclidean_random_walk")
class EuclideanRandomWalkKernel(Kernel):
    """
    Symmetric random-walk inside a **single** environment
    (source and destination environments must be the same).

    Attributes
    ----------
    mean : Array, shape (n_dims,)
        Mean of the Gaussian movement.
    var : Union[Array, float], shape (n_dims, n_dims) or scalar
        Covariance of the Gaussian movement. If a scalar is provided,
        it is interpreted as the variance for each dimension, resulting
        in a diagonal covariance matrix.
    """

    mean: Array  # Mean of the Gaussian movement
    var: Union[Array, float]  # Covariance of the Gaussian movement

    def __post_init__(self) -> None:
        if not isinstance(self.var, (np.ndarray, float)):
            raise TypeError("var must be an ndarray or float")
        if isinstance(self.var, float):
            self.var = np.eye(len(self.mean)) * self.var
        if self.mean.ndim != 1:
            raise ValueError("mean must be a 1D array")
        if self.var.ndim != 2 or self.var.shape[0] != self.var.shape[1]:
            raise ValueError("var must be a square 2D array")
        if self.mean.shape[0] != self.var.shape[0]:
            raise ValueError("mean and var must have the same number of dimensions")
        if not np.all(np.linalg.eigvals(self.var) > 0):
            raise ValueError("var must be a positive-definite matrix")

    def block(
        self,
        *,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: Optional[Covariates] = None,  # covariates are ignored
    ) -> Array:
        """Generate a transition matrix for a random walk in Euclidean space.

        Parameters
        ----------
        src_env : Optional[Environment]
            The source environment from which the transition starts.
        dst_env : Optional[Environment]
            The destination environment to which the transition goes.
        covariates : Optional[Covariates], optional
            Covariates to condition the transition on, by default None
            Ignored in this kernel.

        Returns
        -------
        transition : Array, shape (n_bins, n_bins)
            A transition matrix where each entry represents the probability
            of transitioning from one bin to another in the source environment.
            The matrix is normalized such that each row sums to 1.
        """
        # Atomic case or cross-environment jump
        transition = _handle_intra_env_kernel_edges(src_env, dst_env)
        if transition is not None:
            return transition

        if self.mean.shape[0] != src_env.n_dims:
            raise ValueError(
                "EuclideanRandomWalkKernel: mean vector length "
                f"{self.mean.shape[0]} does not match environment dimension {src_env.n_dims}."
            )
        # distance: shape (src_env.n_bins * src_env.n_bins, n_dims)
        n_bins, n_dims = src_env.n_bins, src_env.n_dims
        centers = src_env.bin_centers
        if n_bins == 0:
            return np.zeros((0, 0))

        distance = (centers[None, :, :] - centers[:, None, :]).reshape(
            (n_bins * n_bins, n_dims)
        )
        gaussian = multivariate_normal(
            mean=self.mean, cov=self.var, allow_singular=True
        )
        log_pdf = gaussian.logpdf(distance)  # shape (n_bins*n_bins,)
        log_pdf = log_pdf.reshape(n_bins, n_bins)
        # subtract row‐max for stability
        row_max = np.max(log_pdf, axis=1, keepdims=True)
        return _normalize_row_probability(np.exp(log_pdf - row_max))
