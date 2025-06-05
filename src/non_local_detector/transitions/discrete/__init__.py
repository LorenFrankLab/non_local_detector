"""
non_local_detector.transitions.discrete
======================================

Public surface for *discrete-state* transition models.

Typical usage
-------------
>>> from non_local_detector.transitions.discrete import (
...     Stationary,
...     CategoricalGLM,
...     diag_stickiness,
... )

# Stationary “sticky diagonal” example
transition = diag_stickiness([0.95, 0.90, 0.98])

# Covariate-dependent GLM example
glm_transition = (
    CategoricalGLM(
        n_states=3,
        formula="1 + bs(speed, df=4)",
        concentration=2.0,
        stickiness=1e3,
    )
    .initialize_parameters(covariate_data={"speed": speed_array})
)
"""

from __future__ import annotations

# ---- protocol (duck-typing contract) -------------------------------------
from .base import DiscreteTransitionModel
from .kernels.glm import CategoricalGLM

# ---- concrete implementations -------------------------------------------
from .kernels.stationary import Stationary
from .priors import get_dirichlet_prior

# ---- convenience helpers -------------------------------------------------
from .wrappers import diag_stickiness

__all__: list[str] = [
    "DiscreteTransitionModel",
    "Stationary",
    "CategoricalGLM",
    "diag_stickiness",
    "get_dirichlet_prior",
]
