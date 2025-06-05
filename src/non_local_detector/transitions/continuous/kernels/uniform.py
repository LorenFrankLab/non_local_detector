from dataclasses import dataclass
from typing import Optional

import numpy as np

from ....environment import Environment
from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition


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
        src_bins = 1 if src_env is None else src_env.n_bins
        dst_bins = 1 if dst_env is None else dst_env.n_bins
        return np.full((src_bins, dst_bins), 1.0 / dst_bins)
