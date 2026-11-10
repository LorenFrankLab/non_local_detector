"""Observation models for neural decoding environments and conditions.

This module defines the ObservationModel class which represents different
environmental contexts and trial conditions for neural decoding analyses.
It specifies which environment, encoding group, and decoding type should
be used for particular data segments.
"""

from dataclasses import dataclass

from non_local_detector.exceptions import ConfigurationError

__all__ = [
    "ObservationModel",
]


@dataclass(frozen=True, order=True)
class ObservationModel:
    """Determines which environment and data points data correspond to.

    This class encapsulates the metadata needed to associate neural data
    with specific experimental environments, encoding groups, and decoding
    contexts (local vs. non-local, spike vs. no-spike conditions). It is
    immutable, so it can be hashed and sorted; use ``dataclasses.replace``
    for a changed copy.

    Attributes
    ----------
    environment_name : str, default=""
        Name identifier for the experimental environment.
    encoding_group : str or int, default=0
        Identifier for the encoding group or trial type.
    is_local : bool, default=False
        Whether this represents local (current position) decoding.
    is_no_spike : bool, default=False
        Whether this represents no-spike baseline conditions.

    Raises
    ------
    ConfigurationError
        If both ``is_local`` and ``is_no_spike`` are True.

    Examples
    --------
    >>> # Create observation model for non-local decoding in track1
    >>> obs_model = ObservationModel("track1", "A", is_local=False)
    >>> obs_model.environment_name
    'track1'

    >>> # Create observation model for local decoding
    >>> local_model = ObservationModel(is_local=True)
    >>> local_model.is_local
    True

    >>> # Create observation model for no-spike condition
    >>> nospike_model = ObservationModel(is_no_spike=True)
    >>> nospike_model.is_no_spike
    True
    """

    environment_name: str = ""
    encoding_group: str | int = 0
    is_local: bool = False
    is_no_spike: bool = False

    def __post_init__(self) -> None:
        """Reject a model that is both Local and No-Spike."""
        if self.is_local and self.is_no_spike:
            msg = "An observation model cannot set both is_local and is_no_spike"
            raise ConfigurationError(
                msg,
                expected="at most one of is_local=True and is_no_spike=True",
                got="is_local=True, is_no_spike=True",
                hint="A Local state is decoded at the animal's position from the "
                "spikes; a No-Spike state has no position and explains the "
                "absence of spikes. Use one state of each.",
            )
