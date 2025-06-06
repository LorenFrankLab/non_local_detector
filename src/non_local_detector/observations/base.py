"""
non_local_detector.observations.base
------------------------------------

An *ObservationModel* scores data likelihoods given **fixed** parameters.
It is called in the *E-step* of EM and never mutates itself during that
step.

Concrete examples: Poisson-GLM, KDE, Gaussian-Ca²⁺ model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..bundle import DataBundle  # typed container used across package

Array = np.ndarray


@runtime_checkable
class ObservationModel(Protocol):
    """
    Compute log-likelihoods for ALL time points in one vectorised pass.

    Implementations MUST be side-effect free — no parameter updates here.
    """

    required_sources: tuple[str, ...] = ()

    @property
    def n_bins(self) -> int: ...  # number of continuous bins

    def log_likelihood(self, bundle: DataBundle) -> Array: ...

    # optional hook
    def precompute(self, bundle: DataBundle) -> None: ...
