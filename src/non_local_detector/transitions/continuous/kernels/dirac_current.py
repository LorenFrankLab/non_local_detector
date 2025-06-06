"""
DiracToCurrentSample kernel
---------------------------

Collapse every source bin into *one* destination bin - the bin that
corresponds to the **current sample** of an arbitrary covariate.

Examples
--------
*   `sample_key="pos_bin"`     - an integer bin index already in the bundle
*   `sample_key="pos_xy"`      - a 2-D coordinate; the kernel calls
    `dst_env.bin_at()` to obtain the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from non_local_detector.environment import Environment

from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition
from ..utils import _handle_intra_env_kernel_edges


@dataclass
@register_continuous_transition("dirac_current")
class DiracToCurrentSample(Kernel):
    """
    A kernel that collapses every source bin to the single destination bin
    corresponding to the “current” sample. You must pass in exactly one sample
    (an integer bin index or a float coordinate) in `covariates[self.sample_key]`.
    """

    sample_key: str
    mask_key: str = "mask"

    def block(
        self,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: dict[str, np.ndarray],
    ) -> np.ndarray:
        # 1) First let the helper deal with any atomic or cross-env situation:
        pre = _handle_intra_env_kernel_edges(src_env, dst_env)
        if pre is not None:
            return pre  # either a (1×1) identity or a uniform block

        # 2) Now we know src_env == dst_env != None
        n_src = 1 if (src_env is None) else src_env.n_bins
        n_dst = 1 if (dst_env is None) else dst_env.n_bins

        # 3) Fetch the “current” sample. It must be a SINGLE integer or SINGLE 2D coordinate:
        if self.sample_key not in covariates:
            raise ValueError(
                f"DiracToCurrentSample requires covariate '{self.sample_key}', "
                "passed-in array must correspond to exactly one timepoint."
            )

        samples = covariates[self.sample_key]

        # If samples is a 1D integer array of length 1, treat that element as the bin index:
        if samples.ndim == 1 and samples.shape[0] == 1:
            if not np.issubdtype(samples.dtype, np.integer):
                raise ValueError(
                    f"DiracToCurrentSample: expected a single integer bin index for '{self.sample_key}', "
                    f"got {samples.dtype}."
                )
            bin_idx = int(samples[0])

        # If samples is a 2D float array of shape (1, dst_env.n_dims), discretize it:
        elif samples.ndim == 2 and samples.shape[0] == 1:
            if samples.shape[1] != dst_env.n_dims:
                raise ValueError(
                    "DiracToCurrentSample: expected a single coordinate of length "
                    f"{dst_env.n_dims}, got shape {samples.shape}."
                )
            bin_idx = dst_env.bin_at(samples)[0]
            if bin_idx < 0 or bin_idx >= n_dst:
                raise ValueError(
                    f"DiracToCurrentSample: coordinate {samples[0]} falls outside dst_env’s bins."
                )

        else:
            # We refuse to guess which timepoint you want. You passed a longer array.
            raise ValueError(
                "DiracToCurrentSample: you passed an array of shape "
                f"{samples.shape}. This kernel expects exactly one sample "
                "(either a length-1 integer array or a (1xn_dims) float array). "
                "If you have multiple timepoints, call this kernel for each timepoint separately."
            )

        # 4) Build the (n_src × n_dst) mapping, placing 1.0 at column = bin_idx
        mapping = np.zeros((n_src, n_dst), dtype=float)
        mapping[:, bin_idx] = 1.0
        return mapping
