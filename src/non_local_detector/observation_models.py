"""Class for representing an environment and a condition (trial type, etc.)"""

from dataclasses import dataclass


@dataclass(order=True)
class ObservationModel:
    """Determines which environment and data points data correspond to.

    Attributes
    ----------
    environment_name : str, optional
    encoding_group : str, optional
    is_local : bool, optional
    is_no_spike : bool, optional

    """

    environment_name: str = ""
    encoding_group: str = 0
    is_local: bool = False
    is_no_spike: bool = False

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.environment_name, self.encoding_group) == (
            other.environment_name,
            other.encoding_group,
        )
