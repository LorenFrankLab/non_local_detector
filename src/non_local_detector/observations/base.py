"""
non_local_detector.observations.base
------------------------------------

Read-only scoring interface used during the **E-step**.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np

from ..bundle import DecoderBatch

Array = np.ndarray


@runtime_checkable
class ObservationModel(Protocol):
    """
    *Implementations must not mutate internal parameters.*
    """

    # ---- metadata ----------------------------------------------------
    required_sources: Tuple[str, ...] = ()  # bundle field names

    @property
    def n_bins(self) -> int:  # continuous bins
        ...

    # ---- main API ----------------------------------------------------
    def log_likelihood(self, batch: DecoderBatch) -> Array: ...

    # optional cache step
    def precompute(self, batch: DecoderBatch) -> None: ...  # pragma: no cover
