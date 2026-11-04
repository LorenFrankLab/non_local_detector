"""The keys every likelihood algorithm's fit returns, by data family.

A likelihood algorithm's fit returns its encoding model, a plain dict that a
detector stores and unpacks into the algorithm's predict as keyword arguments.
Each algorithm describes that dict with a ``TypedDict`` derived from
``SortedSpikesEncodingModel`` or ``ClusterlessEncodingModel``, and annotates
its fit with it, so the type checker checks every key and value the fit
returns. The bases hold the keys code outside the algorithm reads.
"""

from typing import Any, TypedDict

import jax
import numpy as np

from non_local_detector.environment import Environment

__all__ = [
    "ClusterlessEncodingModel",
    "EncodingModel",
    "SortedSpikesEncodingModel",
]


class EncodingModel(TypedDict):
    """The keys every fit returns.

    Attributes
    ----------
    algorithm : str
        The registered name of the algorithm whose fit produced the model. A
        detector refuses to decode with another algorithm, and model checking
        selects its intensity from it.
    environment : Environment
        The fitted environment the model's arrays describe.
    environment_grid : dict[str, Any]
        The fit's record of the environment's grid (see
        ``environment_grid_fingerprint``); predict raises if the environment
        has since been refit to another grid.
    rate_units : str
        Units of the fitted rates, ``"Hz"``; predict rejects any other.
    encoding_exposure_seconds : float
        The fit's weighted occupancy time in seconds, for diagnostics.
    disable_progress_bar : bool
        The fit's progress-bar setting, which the detector also applies to
        decoding.
    """

    algorithm: str
    environment: Environment
    environment_grid: dict[str, Any]
    rate_units: str
    encoding_exposure_seconds: float
    disable_progress_bar: bool


class SortedSpikesEncodingModel(EncodingModel):
    """The keys every sorted-spikes fit returns.

    Attributes
    ----------
    place_fields : jax.Array, shape (n_neurons, n_position_bins)
        Each neuron's firing rate in Hz on the full place grid. The detector
        checks the decoded neuron count against it, and the plots order
        neurons by its peaks.
    """

    place_fields: jax.Array


class ClusterlessEncodingModel(EncodingModel):
    """The keys every clusterless fit returns.

    Attributes
    ----------
    mean_rates : jax.Array or list of float, length n_electrodes
        Each electrode's mean spike rate in Hz over the encoding exposure. The
        detector checks the decoded electrode count against it.
    mark_dimensions : list of int, length n_electrodes
        The number of waveform features of each electrode's encoding spikes,
        which the detector checks the decoded waveform features against.
    """

    mean_rates: jax.Array | np.ndarray | list[float]
    mark_dimensions: list[int]
