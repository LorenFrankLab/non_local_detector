from dataclasses import dataclass
from typing import Optional

import numpy as np

from ....environment import Environment
from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition
from ..utils import _handle_intra_env_kernel_edges


@dataclass
@register_continuous_transition("uniform")
class UniformKernel(Kernel):
    """
    Jump uniformly into **dst_env** regardless of the source bin.

    Useful for 'entry' transitions (e.g., local → non-local replay).
    """

    def block(
        self,
        *,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: Optional[Covariates] = None,  # covariates are ignored
    ) -> Array:
        transition = _handle_intra_env_kernel_edges(src_env, dst_env)
        if transition is not None:
            return transition

        dst_bins = dst_env.n_bins
        src_bins = src_env.n_bins
        if dst_bins == 0:
            raise ValueError("Destination environment must have at least one bin.")
        if src_bins == 0:
            raise ValueError("Source environment must have at least one bin.")
        return np.full((src_bins, dst_bins), 1.0 / dst_bins)
