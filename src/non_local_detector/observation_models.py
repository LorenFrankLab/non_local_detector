"""Observation models for neural decoding environments and conditions.

This module defines the ObservationModel class which represents different
environmental contexts and trial conditions for neural decoding analyses.
It specifies which environment, encoding group, and decoding type should
be used for particular data segments.
"""

from dataclasses import dataclass


@dataclass(order=True)
class ObservationModel:
    """Determines which environment and data points data correspond to.

    This class encapsulates the metadata needed to associate neural data
    with specific experimental environments, encoding groups, and decoding
    contexts (local vs. non-local, spike vs. no-spike conditions).

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

    def __eq__(self, other: object) -> bool:
        """Check equality based on environment name and encoding group.

        Two ObservationModel instances are considered equal if they have
        the same environment_name and encoding_group, regardless of the
        values of is_local and is_no_spike flags.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        bool
            True if both objects have the same environment_name and encoding_group.
        """
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, ObservationModel)
        return (self.environment_name, self.encoding_group) == (
            other.environment_name,
            other.encoding_group,
        )
