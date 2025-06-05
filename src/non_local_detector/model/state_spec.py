"""
non_local_detector.state_spec
=============================

`StateSpec` is the **single authoritative place** where you declare the
properties of a discrete state:

* its *name* (unique string identifier),
* the associated *Environment* (or ``None`` for atomic states),
* the *ObservationModel* used to score data likelihoods,
* optional *EncodingModel* and *time mask* helpers.

Every high-level component (discrete-transition glue, continuous motion
builder, forward/backward code) consults `StateSpec` to learn how many
continuous bins a state has and which environment object to pass around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from ..encoding.base import EncodingModel
from ..environment import Environment
from ..observations.base import ObservationModel

if TYPE_CHECKING:
    import numpy as np  # pragma: no cover


@dataclass
class StateSpec:
    """
    Description of ONE discrete hidden state.

    Attributes
    ----------
    name
        Unique identifier; used as the key in mapping tables.
    env
        An `Environment` instance describing the state's continuous bins,
        or ``None`` if the state is *atomic* (exactly one bin).
    obs_model
        Object that provides `log_likelihood(bundle) -> np.ndarray`.
    encoding_model
        Optional model updated during the EM *encoding* step.
    encoding_time_mask
        Optional callable ``mask = f(covariates_dict)`` that selects which
        time points count toward the encoding update.
    """

    name: str
    env: Optional[Environment]
    obs_model: ObservationModel
    encoding_model: Optional[EncodingModel] = None
    encoding_time_mask: Optional[Callable[..., "np.ndarray"]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"State name must be a string, got {type(self.name)}.")
        if self.env is not None and not isinstance(self.env, Environment):
            raise TypeError(f"env must be an Environment, got {type(self.env)}.")
        if not isinstance(self.obs_model, ObservationModel):
            raise TypeError(
                f"obs_model must be an ObservationModel, got {type(self.obs_model)}."
            )
        if self.encoding_model is not None and not isinstance(
            self.encoding_model, EncodingModel
        ):
            raise TypeError(
                f"encoding_model must be an EncodingModel, got {type(self.encoding_model)}."
            )
        if self.encoding_time_mask is not None and not callable(
            self.encoding_time_mask
        ):
            raise TypeError(
                f"encoding_time_mask must be callable, got {type(self.encoding_time_mask)}."
            )
        if not hasattr(self.obs_model, "log_likelihood"):
            raise ValueError("obs_model must implement method `log_likelihood(bundle)`")

    # ------------------------------------------------------------------ #
    #  Convenience properties                                            #
    # ------------------------------------------------------------------ #
    @property
    def n_bins(self) -> int:
        """Number of *continuous* bins inside this state."""
        return 1 if self.env is None else self.env.n_bins

    def __repr__(self) -> str:  # pragma: no cover
        env_desc = "atomic" if self.env is None else f"{self.env!r}"
        return (
            f"StateSpec(name={self.name!r}, env={env_desc}, "
            f"obs_model={self.obs_model!r})"
        )
