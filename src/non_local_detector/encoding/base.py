"""
non_local_detector.encoding.base
--------------------------------

Parameter-update interface invoked in the **M-step**.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ..bundle import DecoderBatch

Array = np.ndarray


class UpdatePolicy(Enum):
    NEVER = auto()  # frozen parameters
    INITIAL_FIT = auto()  # only initial_fit() then freeze
    ALWAYS = auto()  # call update each EM iteration


@runtime_checkable
class EncodingModel(Protocol):
    """
    Updates an associated ObservationModel **in-place** using smoothed
    posteriors γₜ(i).
    """

    update_policy: UpdatePolicy  # NEW required property
    update_period: int | None = None  # only for PERIODIC

    def initial_fit(
        self, batch: DecoderBatch, mask: np.ndarray | None = None
    ) -> None: ...

    def update_from_posteriors(
        self,
        batch: DecoderBatch,
        posterior_probability: Array,  # shape (n_time,) or (n_time, n_bins)
        *,
        mask: Optional[Array] = None,
    ) -> None: ...
