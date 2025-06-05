from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import multivariate_normal

from ....environment import Environment
from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition
from ..utils import _handle_intra_env_kernel_edges, _normalize_row_probability


@dataclass
@register_continuous_transition("empirical")
class EmpiricalKernel(Kernel):
    samples_key: str  # e.g. "pos_xy"
    mask_key: str = "enc_mask"
    speedup: int = 1
    is_time_reversed: bool = False
    _cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _fit_empirical_matrix(
        self, env: Environment, coords: Array, mask: Array
    ) -> Array:
        coords = coords[mask]  # time filter in one line
        if coords.shape[0] < 2:
            raise ValueError("Not enough samples after masking.")

        bin_seq = env.coords_to_bins(coords)
        src, dst = bin_seq[:-1], bin_seq[1:]
        H = np.histogram2d(src, dst, bins=(env.n_bins, env.n_bins))[0]
        P = _normalize_row_probability(H)
        return np.linalg.matrix_power(P, self.speedup) if self.speedup > 1 else P


def block(self, *, src_env, dst_env, covariates=None) -> Array:
    transition = _handle_intra_env_kernel_edges(src_env, dst_env)
    if transition is not None:
        return transition

    env_id = id(src_env)
    if env_id not in self._cache:
        coords = covariates[self.samples_key]
        mask = covariates.get(self.mask_key, np.ones(len(coords), dtype=bool))
        self._cache[env_id] = self._fit_empirical_matrix(src_env, coords, mask)
    return self._cache[env_id]
