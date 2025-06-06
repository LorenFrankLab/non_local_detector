from dataclasses import dataclass
from typing import Optional

import numpy as np

from non_local_detector.environment import Environment

from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition
from ..utils import _handle_intra_env_kernel_edges


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
        transition = _handle_intra_env_kernel_edges(src_env, dst_env)
        if transition is not None:
            return transition

        src_bins = 1 if src_env is None else src_env.n_bins
        return np.eye(src_bins)
