from dataclasses import dataclass
from typing import Optional

import numpy as np

from ....environment import Environment
from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition


@dataclass
@register_continuous_transition("identity")
class IdentityKernel(Kernel):
    """
    Stay in the same bin (used for silent/no-spike states).
    """

    def block(
        self,
        *,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: Optional[Covariates] = None,  # covariates are ignored
    ) -> Array:
        # Only valid when both sides are atomic or identical env
        if src_env.name != dst_env.name:
            raise ValueError("IdentityKernel requires src_env == dst_env.")
        src_bins = 1 if src_env is None else src_env.n_bins
        return np.eye(src_bins)
