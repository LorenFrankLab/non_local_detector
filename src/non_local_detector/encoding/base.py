"""
non_local_detector.encoding.base
--------------------------------

An *EncodingModel* owns the *parameter-learning* logic for a state's
observation model.  It sees the posteriors γₜ(i) and — optionally — an
explicit time mask, then updates the parameters **in place**.

Concrete examples: place-field KDE update, Poisson rate re-estimate.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..bundle import DataBundle

Array = np.ndarray


class EncodingModel(Protocol):
    """
    Called during the *M-step* of EM; mutates its internal parameters.
    """

    def update_from_posteriors(
        self,
        bundle: DataBundle,
        discrete_state_probability: Array,
        *,
        mask: Optional[Array] = None,
    ) -> None:
        """
        Update the model in-place using smoothed posteriors.
        """
