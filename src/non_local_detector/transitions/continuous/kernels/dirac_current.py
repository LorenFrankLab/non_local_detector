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

from ....environment import Environment
from ..base import Array, Covariates, Kernel
from ..registry import register_continuous_transition


@dataclass
@register_continuous_transition("dirac_current")
class DiracToCurrentSample(Kernel):
    """
    Parameters
    ----------
    sample_key
        Name of the covariate that encodes the **current location** in the
        destination environment.  It can be either

        * an **integer** array of shape ``(n_time,)`` with valid bin indices, or
        * a **coordinate** array of shape ``(n_time, dst_env.n_dims)`` that
          will be converted via :pyfunc:`Environment.bin_at`.
    """

    sample_key: str = "pos_bin"

    # ------------------------------------------------------------------ #
    #  Kernel API                                                        #
    # ------------------------------------------------------------------ #
    def block(
        self,
        *,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: Optional[Covariates] = None,
    ) -> Array:
        # ------------------ sanity checks -----------------------------
        if dst_env is None:
            return np.ones((1 if src_env is None else src_env.n_bins, 1))

        if covariates is None or self.sample_key not in covariates:
            raise ValueError(
                f"DiracToCurrentSample requires covariate '{self.sample_key}'."
            )

        samples: Array = covariates[self.sample_key]
        n_src = 1 if src_env is None else src_env.n_bins
        n_dst = dst_env.n_bins

        # ------------------ convert sample → bin index ----------------
        if samples.ndim == 1 and np.issubdtype(samples.dtype, np.integer):
            bin_indices = samples
        else:
            # Assume coordinates; let Environment handle discretisation
            bin_indices = dst_env.bin_at(samples)  # shape (n_time,)

        if bin_indices.max() >= n_dst or bin_indices.min() < 0:
            raise ValueError("Sample indices out of destination-env range.")

        # ------------------ build deterministic mapping ---------------
        mapping = np.zeros((n_src, n_dst))
        mapping[:, int(bin_indices[-1])] = 1.0  # use *current* sample (last of chunk)
        return mapping
