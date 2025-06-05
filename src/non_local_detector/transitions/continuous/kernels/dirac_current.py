"""
DiracToCurrentPosition kernel
-----------------------------

Collapse from any source bin to the *current physical position* bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ....environment import Environment
from ..base import Array, Covariates, Kernel


@dataclass
class DiracToCurrentPosition(Kernel):
    """
    Parameters
    ----------
    position_key
        Name of the covariate that contains the **integer** position bin index
        for each time step (e.g., ``covariates["pos_bin"]``).
    """

    position_key: str = "pos_bin"

    # This kernel ignores src_env and dst_env sizes; they are supplied below.
    def block(
        self,
        *,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: Optional[Covariates] = None,
    ) -> Array:
        if covariates is None or self.position_key not in covariates:
            raise ValueError(
                f"DiracToCurrentPosition requires covariate '{self.position_key}'."
            )

        # n_src and n_dst define matrix shape
        n_src = 1 if src_env is None else src_env.n_bins
        n_dst = 1 if dst_env is None else dst_env.n_bins

        pos_index: Array = covariates[self.position_key]  # shape (n_time,)
        if pos_index.max() >= n_dst or pos_index.min() < 0:
            raise ValueError("position index out of destination-env range.")

        # Build the time-varying matrices and average them (if you need
        # time-specific matrices, move this logic up one level).
        # For static flattened kernel we produce row-identical mapping.
        mapping = np.zeros((n_src, n_dst))
        mapping[:, pos_index[-1]] = 1.0  # use last observed pos for this chunk
        return mapping
