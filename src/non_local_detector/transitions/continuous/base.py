"""
non_local_detector.transitions.continuous.base
=============================================

Foundational interfaces and type aliases for ALL continuous-state
transition logic.

A **ContinuousTransition** object produces the *flattened* matrix
``P(bin_{t+1} | bin_t)`` of shape
``(n_bins_total, n_bins_total)`` that the HMM’s forward/backward code
expects.

A lower-level **Kernel** object describes motion for a single
``(source_state, destination_state)`` pair and is used by the
`BlockTransition` orchestrator (see ``block.py``).
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np

from non_local_detector.environment import Environment

# --------------------------------------------------------------------------- #
#  Type aliases                                                               #
# --------------------------------------------------------------------------- #
Array = np.ndarray
Covariates = Dict[str, Array]


# --------------------------------------------------------------------------- #
#  Public protocol objects                                                    #
# --------------------------------------------------------------------------- #
@runtime_checkable
class ContinuousTransition(Protocol):
    """
    High-level interface: returns the full flattened transition matrix.

    Implementations:
    ----------------
    * `BlockTransition` – stitches together per-state Kernels.
    * Any user-supplied class that fulfils this protocol; no inheritance
      required (duck typing).
    """

    n_bins_total: int  # total # of continuous bins

    def matrix(
        self,
        *,
        covariates: Optional[Covariates] = None,
    ) -> Array:
        """
        Return a row-stochastic NumPy array of shape
        ``(n_bins_total, n_bins_total)``.

        If the motion kernel depends on external covariates
        (e.g., running speed → diffusion strength), pass them via
        the `covariates` dictionary.
        """


@runtime_checkable
class Kernel(Protocol):
    """
    Local motion rule for **one** discrete-state transition
    ``(source_state, destination_state)``.  It produces a block of shape
    ``(n_bins_source, n_bins_destination)``.

    Notes
    -----
    * Either `src_env` or `dst_env` can be *None* when the corresponding
      state is *atomic* (i.e. has exactly one bin).
    """

    def block(
        self,
        *,
        src_env: Optional[Environment],
        dst_env: Optional[Environment],
        covariates: Optional[Covariates] = None,
    ) -> Array:
        """
        Produce the sub-matrix for this (src, dst) pair.

        The returned array **need not** be copied; callers may place the
        same view directly into the big flattened matrix.
        """
