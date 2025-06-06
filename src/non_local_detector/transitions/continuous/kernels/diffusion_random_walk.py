from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from non_local_detector.diffusion_kernels import compute_diffusion_kernels
from non_local_detector.environment import Environment

from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition
from ..utils import _handle_intra_env_kernel_edges


@dataclass
@register_continuous_transition("diffusion_random_walk")
class DiffusionRandomWalkKernel(Kernel):
    """
    Symmetric random-walk inside a **single** environment
    (source and destination environments must be the same).

    Attributes
    ----------
    var : float
        Covariance of the Gaussian movement. If a scalar is provided,
        it is interpreted as the variance for each dimension, resulting
        in a diagonal covariance matrix.
    """

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

        # Compute the diffusion kernel matrix
        return compute_diffusion_kernels(
            src_env.connectivity, self.var, mode="transition"
        )  # already normalized by compute_diffusion_kernels
